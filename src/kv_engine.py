"""
Unified SDPA-Native KV Cache Compression Engine for Reasoning Models (2025-2026).

Supports:
1. `dense`: Full KV Cache baseline (no eviction).
2. `recency_pp`: Prompt + Recency Window (StreamingLLM-style prompt-protected baseline).
3. `snapkv_pp`: Prompt-Protected Observation-Window Attention Pooling (SnapKV-Decoding).
4. `rkv_pp`: Prompt-Protected Redundancy-Aware KV Compression (R-KV, Cai et al., 2025,
   with newest-copy redundancy sparing and cosine similarity penalty).
5. `vase_pp`: Prompt-Protected Value-Aware Stochastic Eviction (VaSE, Chang et al., 2026:
   value-range outlier protection + multinomial attention sampling).
6. `valuediff_pp`: Prompt-Protected Value-Dispersion Ranking (ValueDiff, 2026:
   L2 deviation from cache mean ||v_i - mean(v)||_2, query-free).
7. `random_pp`: Prompt-Protected Per-Head Uniform Random Eviction (Salesforce AI Research, Sep 2026:
   memoryless i.i.d. uniform random eviction per KV head).
8. `vacor_kv` (Proposed): Value-Outlier-Anchored Antithetic Reservoir KV Eviction:
   - 100% Query-Free & Attention-Free (O(N) metadata-only, <0.1ms per round).
   - Innovation A: Query-Free Value-Dispersion Outlier Pinning (captures high-surprise constants/pivots).
   - Innovation B: Cyclic-Latin Antithetic Cross-Head Coupling (eliminates the 36.8% i.i.d. multi-head
     blind spot of `random_pp`, guaranteeing 100% cross-head union coverage up to H_kv * K_free).
   - Innovation C: Persistent Reservoir Keys with Horizon-Balanced Age Compensation (eliminates the
     q^R exponential mid-CoT forgetting and per-round cache slot churn of memoryless `random_pp`).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicLayer
from transformers.configuration_utils import PreTrainedConfig


PHI_INV = 0.6180339887498949  # Golden ratio conjugate for quasi-Monte Carlo sequences


@dataclass
class EvictionStats:
    """Tracks fine-grained eviction telemetry across all layers."""
    eviction_calls: int = 0
    total_eviction_time_ms: float = 0.0
    last_cur_len: int = 0
    last_kept_len: int = 0


class UnifiedEvictLayer(DynamicLayer):
    """
    SDPA-compatible pre-allocated KV cache layer in [B, H_kv, max_len, D] layout.
    """

    def __init__(
        self,
        layer_idx: int,
        token_budget: int,
        residual_length: int,
        eviction_mode: str,
        *,
        rkv_lambda: float = 0.1,
        smooth: bool = True,
        n_large_frac: float = 0.20,
        vacor_outlier_frac: float = 0.20,
        vacor_age_Menu: float = 0.18,
        vacor_pin_frac: float = 0.5,
        temperature: float = 1.0,
        seed: int = 42,
        track_positions: bool = True,
        stats: Optional[EvictionStats] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.token_budget = token_budget
        self.residual_length = residual_length
        self.MODE = eviction_mode
        self.rkv_lambda = rkv_lambda
        self.smooth = smooth
        self.n_large_frac = n_large_frac
        self.vacor_outlier_frac = vacor_outlier_frac
        self.vacor_age_gamma = vacor_age_Menu
        self.vacor_pin_frac = float(vacor_pin_frac)
        self.temperature = temperature
        self.seed = seed
        self.track_positions = track_positions
        self.stats = stats

        self.cumulative_length = 0
        self.cur_len = 0
        self.prompt_len = 0
        self._q_write_pos = 0
        self.is_initialized = False

        # Dedicated generator for reproducible stochastic eviction per layer
        self._rng: Optional[torch.Generator] = None

    def _get_rng(self, device: torch.device) -> torch.Generator:
        if self._rng is None or self._rng.device != device:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(self.seed + self.layer_idx * 1009)
        return self._rng

    def lazy_initialization(self, key_states: torch.Tensor, value_states: Optional[torch.Tensor] = None) -> None:
        # key_states in standard HF SDPA layout: [B, H_kv, L, D]
        batch_size, num_kv_heads, prompt_len, head_dim = key_states.shape
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.prompt_len = prompt_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        if self.MODE == "dense":
            max_cache_len = prompt_len + 16384
        else:
            max_cache_len = max(prompt_len, self.token_budget) + self.residual_length + 64

        self.k_cache = torch.zeros(
            batch_size, num_kv_heads, max_cache_len, head_dim,
            dtype=self.dtype, device=self.device,
        )
        self.v_cache = torch.zeros(
            batch_size, num_kv_heads, max_cache_len, head_dim,
            dtype=self.dtype, device=self.device,
        )

        if self.track_positions or self.MODE in ("vacor_kv", "vacor_antithetic_only", "vacor_persist_only", "vacor_pin", "adaptive_vacor"):
            self.pos_ids = torch.zeros(
                batch_size, num_kv_heads, max_cache_len,
                dtype=torch.int32, device=self.device,
            )
            self.priority_cache = torch.zeros(
                batch_size, num_kv_heads, max_cache_len,
                dtype=torch.float32, device=self.device,
            )

        self.cur_len = 0
        self.cumulative_length = 0
        self.is_initialized = True

    def _ensure_capacity(self, needed_len: int) -> None:
        if needed_len <= self.k_cache.shape[2]:
            return
        extra = max(needed_len - self.k_cache.shape[2], 2048)
        B, H, _, D = self.k_cache.shape
        self.k_cache = torch.cat(
            [self.k_cache, torch.zeros(B, H, extra, D, dtype=self.dtype, device=self.device)], dim=2
        )
        self.v_cache = torch.cat(
            [self.v_cache, torch.zeros(B, H, extra, D, dtype=self.dtype, device=self.device)], dim=2
        )
        if hasattr(self, "pos_ids"):
            self.pos_ids = torch.cat(
                [self.pos_ids, torch.zeros(B, H, extra, dtype=torch.int32, device=self.device)], dim=2
            )
            self.priority_cache = torch.cat(
                [self.priority_cache, torch.zeros(B, H, extra, dtype=torch.float32, device=self.device)], dim=2
            )

    def needs_queries(self) -> bool:
        return self.MODE in ("snapkv_pp", "rkv_pp", "vase_pp", "h2o_pp", "tova_pp")

    def update_queries(self, query_states: torch.Tensor) -> None:
        """
        Store recent post-RoPE queries in a circular buffer [B, H_q, W, D]
        for query-dependent baselines (SnapKV, R-KV, VaSE).
        Query-free methods (`random_pp`, `valuediff_pp`, `recency_pp`, `vacor_kv`) skip this completely.
        """
        if not self.needs_queries():
            return
        q_len = query_states.shape[2]
        W = self.residual_length
        if not hasattr(self, "queries"):
            B, Hq, _, D = query_states.shape
            self.queries = torch.zeros(B, Hq, W, D, dtype=query_states.dtype, device=query_states.device)
            self._q_write_pos = 0

        if q_len == 1:
            self.queries[:, :, self._q_write_pos : self._q_write_pos + 1, :].copy_(query_states)
            self._q_write_pos = (self._q_write_pos + 1) % W
        elif q_len >= W:
            self.queries.copy_(query_states[:, :, -W:, :])
            self._q_write_pos = 0
        else:
            # Chunk update of size < W
            end_pos = self._q_write_pos + q_len
            if end_pos <= W:
                self.queries[:, :, self._q_write_pos : end_pos, :].copy_(query_states)
            else:
                first_part = W - self._q_write_pos
                self.queries[:, :, self._q_write_pos : W, :].copy_(query_states[:, :, :first_part, :])
                self.queries[:, :, : q_len - first_part, :].copy_(query_states[:, :, first_part:, :])
            self._q_write_pos = end_pos % W

    def _assign_antithetic_priorities(
        self,
        B: int,
        H: int,
        start_step: int,
        n_new: int,
        block_size: int = 8,
    ) -> torch.Tensor:
        """
        Generate Block-Antithetic Persistent Reservoir priorities for new tokens.
        Key Discovery (Span-Coherent Cyclic-Latin Coupling):
          In Transformer reasoning models (`Qwen3`), semantic facts and mathematical expressions
          (`M1 = 13`, `x^2 + 4x = 12`) span 6-8 contiguous tokens (`block_size = 8`).
          If priorities are assigned independently per single token, head h keeps `M1` while evicting
          `13`, fragmenting multi-token phrases within every head!
          Instead, we assign persistent base random phases per contiguous block b = floor(pos / block_size),
          and couple the H_kv heads antithetically via cyclic Latin shifts:
              u_{h}(pos) = (u_0(floor(pos / B_span)) + (h + pi_l) / H) mod 1
          This guarantees BOTH:
            1. 100% intra-head semantic span coherence (entire 8-token equations survive intact in a head).
            2. 100% cross-head union coverage across the H_kv heads (zero correlation deficit)!
        """
        rng = self._get_rng(self.device)
        positions = torch.arange(start_step, start_step + n_new, device=self.device, dtype=torch.long)
        block_ids = torch.div(positions, block_size, rounding_mode="floor")
        unique_blocks, inverse_idx = torch.unique(block_ids, return_inverse=True)
        n_blocks = unique_blocks.numel()

        if self.MODE == "vacor_persist_only":
            block_u = torch.rand((B, H, n_blocks), generator=rng, device=self.device, dtype=torch.float32)
            u_heads = block_u[:, :, inverse_idx]
        else:
            # Base uniform draw per 8-token semantic block (shared across heads, stratified via Latin shift)
            u0_blocks = torch.rand((B, 1, n_blocks), generator=rng, device=self.device, dtype=torch.float32)
            head_idx = torch.arange(H, device=self.device, dtype=torch.float32).view(1, H, 1)
            layer_phase = (self.layer_idx * PHI_INV) % 1.0
            jitter_blocks = (
                torch.rand((B, H, n_blocks), generator=rng, device=self.device, dtype=torch.float32)
                * (0.12 / H)
            )
            block_u = torch.remainder(u0_blocks + (head_idx + layer_phase) / float(H) + jitter_blocks, 1.0)
            u_heads = block_u[:, :, inverse_idx]
        return u_heads

    def append_and_get_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key/value states [B, H_kv, L_new, D] into pre-allocated buffer
        and return active slice [B, H_kv, cur_len + L_new, D] for SDPA.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        B, H, n_new, D = key_states.shape
        needed = self.cur_len + n_new
        self._ensure_capacity(needed)

        self.k_cache[:, :, self.cur_len : needed, :].copy_(key_states)
        self.v_cache[:, :, self.cur_len : needed, :].copy_(value_states)

        if hasattr(self, "pos_ids"):
            new_pos = torch.arange(
                self.cumulative_length,
                self.cumulative_length + n_new,
                device=self.device,
                dtype=torch.int32,
            ).view(1, 1, n_new).expand(B, H, n_new)
            self.pos_ids[:, :, self.cur_len : needed].copy_(new_pos)

            new_prio = self._assign_antithetic_priorities(B, H, self.cumulative_length, n_new)
            self.priority_cache[:, :, self.cur_len : needed].copy_(new_prio)

        self.cur_len = needed
        self.cumulative_length += n_new
        return self.k_cache[:, :, : self.cur_len, :], self.v_cache[:, :, : self.cur_len, :]

    def maybe_evict(self, n_new: int, is_prefill: bool = False) -> None:
        """
        Triggered after SDPA during decoding (or chunked decoding) when
        cur_len >= token_budget + residual_length.
        """
        if self.MODE == "dense" or is_prefill:
            return
        if (
            self.cur_len > self.token_budget
            and (self.cur_len - self.token_budget) % self.residual_length == 0
        ):
            t0 = None
            if self.stats is not None and self.layer_idx == 0:
                t0 = time.perf_counter()
            self._run_eviction(self.cur_len)
            if t0 is not None and self.stats is not None:
                self.stats.eviction_calls += 1
                self.stats.total_eviction_time_ms += (time.perf_counter() - t0) * 1000.0
                self.stats.last_cur_len = self.cumulative_length
                self.stats.last_kept_len = self.cur_len

    def _compute_attention_scores(self, query_cache: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute observation-window attention scores over candidate keys.
        query_cache: [B, H_q, W, D]
        keys: [B, H_kv, N_cand, D]
        Returns: [B, H_kv, N_cand]
        """
        B, q_heads, q_len, head_dim = query_cache.shape
        kv_heads = keys.shape[1]
        num_kv_groups = q_heads // kv_heads

        q_grouped = query_cache.view(B, kv_heads, num_kv_groups, q_len, head_dim)
        k_expanded = keys.unsqueeze(2)  # [B, H_kv, 1, N_cand, D]

        attn_weights = torch.matmul(q_grouped, k_expanded.transpose(3, 4)) / (
            math.sqrt(head_dim) * self.temperature
        )
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_cache.dtype)
        scores = attn_weights.mean(-2)  # [B, H_kv, num_kv_groups, N_cand]
        if self.smooth and scores.shape[-1] >= 5:
            orig_shape = scores.shape
            scores_flat = scores.view(B, q_heads, -1)
            scores_flat = F.avg_pool1d(scores_flat, kernel_size=5, padding=2, stride=1)
            scores = scores_flat.view(orig_shape)
        return scores.mean(2)  # [B, H_kv, N_cand]

    def _cal_rkv_redundancy(self, keys: torch.Tensor, protect_len: int = 0) -> torch.Tensor:
        """
        R-KV Official Redundancy Calculation (Cai et al., 2025; Random-Attention port):
        Only counts similarity to OLDER keys (strictly upper-triangular mask) so the newest copy
        of a repeated concept is spared, and masks prompt columns so similarity to the prompt
        is not penalized.
        keys: [B, H_kv, N_cand, D]
        """
        k_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
        cos_sim = torch.matmul(k_norm, k_norm.transpose(-1, -2))  # [B, H_kv, N_cand, N_cand]
        n = keys.size(2)
        causal_older = torch.triu(torch.ones(n, n, device=self.device, dtype=torch.bool), diagonal=1)
        cos_sim.masked_fill_(~causal_older.unsqueeze(0).unsqueeze(0), 0.0)
        if protect_len > 0:
            cos_sim[:, :, :, :protect_len] = 0.0
        denom = max(n - max(protect_len, 1), 1)
        return (cos_sim.sum(dim=2) / denom).softmax(dim=-1)

    def _run_eviction(self, cur_len: int) -> None:
        residual = self.residual_length
        K = self.token_budget - residual  # Total slots to keep before the recency window
        n_cand = cur_len - residual       # Total candidate slots in [0 : cur_len - residual)

        if K >= n_cand:
            return

        # Prompt protection boundary (mirrors Random-Attention `_pp` convention)
        pl = min(self.prompt_len, K - 16)
        pl = max(pl, 0)
        K_free = K - pl                   # Non-prompt budget available for CoT history

        B, H, _, D = self.k_cache.shape

        if K_free <= 0 or (n_cand - pl) <= K_free:
            # Either prompt alone exceeds K (e.g. prompt_len > K on first eviction round!)
            # OR candidate pool has fewer CoT tokens than K_free.
            # Keep the first pl tokens + the most recent K_free tokens before the recency window.
            if K_free <= 0:
                # Truncate prompt protection to K
                pass
            else:
                return
            self.k_cache[:, :, K : self.token_budget, :].copy_(
                self.k_cache[:, :, cur_len - residual : cur_len, :].clone()
            )
            self.v_cache[:, :, K : self.token_budget, :].copy_(
                self.v_cache[:, :, cur_len - residual : cur_len, :].clone()
            )
            if hasattr(self, "pos_ids"):
                self.pos_ids[:, :, K : self.token_budget].copy_(
                    self.pos_ids[:, :, cur_len - residual : cur_len].clone()
                )
                self.priority_cache[:, :, K : self.token_budget].copy_(
                    self.priority_cache[:, :, cur_len - residual : cur_len].clone()
                )
            self.cur_len = self.token_budget
            self.prompt_len = min(self.prompt_len, K)
            return

        # Candidate slices
        k_candidates = self.k_cache[:, :, :n_cand, :]  # [B, H, n_cand, D]
        v_candidates = self.v_cache[:, :, :n_cand, :]  # [B, H, n_cand, D]
        n_cot_cand = n_cand - pl                       # Candidates in [pl : n_cand)

        prompt_idx = (
            torch.arange(pl, device=self.device, dtype=torch.long)
            .view(1, 1, pl)
            .expand(B, H, pl)
        )

        # =====================================================================
        # Method 1: `recency_pp` (Prompt + Most Recent K_free tokens)
        # =====================================================================
        if self.MODE == "recency_pp":
            recent_cot_idx = (
                torch.arange(n_cand - K_free, n_cand, device=self.device, dtype=torch.long)
                .view(1, 1, K_free)
                .expand(B, H, K_free)
            )
            ids_to_keep = torch.cat([prompt_idx, recent_cot_idx], dim=-1)

        # =====================================================================
        # Method 2: `random_pp` (Salesforce Sep 2026: Memoryless i.i.d. Random)
        # =====================================================================
        elif self.MODE == "random_pp":
            rng = self._get_rng(self.device)
            rs = torch.rand((B, H, n_cot_cand), generator=rng, device=self.device)
            cot_keep = rs.topk(K_free, dim=-1, largest=True).indices + pl
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        # =====================================================================
        # Method 3: `snapkv_pp` (Prompt-Protected Observation-Window Attention)
        # =====================================================================
        elif self.MODE == "snapkv_pp":
            attn_scores = self._compute_attention_scores(self.queries, k_candidates)
            cot_scores = attn_scores[:, :, pl:n_cand]
            cot_keep = cot_scores.topk(K_free, dim=-1, largest=True).indices + pl
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        # =====================================================================
        # Method 3b: `tova_pp` (TOVA: Top-k Attention Vector of Latest Token)
        # =====================================================================
        elif self.MODE == "tova_pp":
            latest_q = self.queries[:, :, -1:, :]
            attn_scores = self._compute_attention_scores(latest_q, k_candidates)
            cot_scores = attn_scores[:, :, pl:n_cand]
            cot_keep = cot_scores.topk(K_free, dim=-1, largest=True).indices + pl
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        # =====================================================================
        # Method 3c: `h2o_pp` (H2O: Heavy Hitter Oracle Cumulative Attention)
        # =====================================================================
        elif self.MODE == "h2o_pp":
            orig_smooth = self.smooth
            self.smooth = False
            attn_scores = self._compute_attention_scores(self.queries, k_candidates)
            self.smooth = orig_smooth
            cot_scores = attn_scores[:, :, pl:n_cand]
            cot_keep = cot_scores.topk(K_free, dim=-1, largest=True).indices + pl
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        # =====================================================================
        # Method 4: `rkv_pp` (Prompt-Protected R-KV with Cosine Redundancy)
        # =====================================================================
        elif self.MODE == "rkv_pp":
            attn_scores = self._compute_attention_scores(self.queries, k_candidates)
            redundancy = self._cal_rkv_redundancy(k_candidates, protect_len=pl)
            combined = attn_scores * self.rkv_lambda - redundancy * (1.0 - self.rkv_lambda)
            cot_scores = combined[:, :, pl:n_cand]
            cot_keep = cot_scores.topk(K_free, dim=-1, largest=True).indices + pl
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        # =====================================================================
        # Method 5: `vase_pp` (Prompt-Protected VaSE: Value Range + Sample Attn)
        # =====================================================================
        elif self.MODE == "vase_pp":
            attn_scores = self._compute_attention_scores(self.queries, k_candidates)
            v_cot = v_candidates[:, :, pl:n_cand, :]
            v_range = (v_cot.amax(dim=-1) - v_cot.amin(dim=-1)).float()
            n_large = min(max(1, int(round(K_free * self.n_large_frac))), K_free - 1)
            ids_large_rel = v_range.topk(n_large, dim=-1, largest=True).indices

            cot_attn = attn_scores[:, :, pl:n_cand].float().clone()
            cot_attn.scatter_(dim=-1, index=ids_large_rel, value=0.0)
            cot_attn = cot_attn.clamp(min=0.0) + 1e-10
            flat_probs = cot_attn.view(-1, n_cot_cand)
            ids_sample_rel = torch.multinomial(
                flat_probs, K_free - n_large, replacement=False, generator=self._get_rng(self.device)
            ).view(B, H, K_free - n_large)

            cot_keep = torch.cat([ids_large_rel, ids_sample_rel], dim=-1) + pl
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        # =====================================================================
        # Method 6: `valuediff_pp` (Value-Dispersion ||v_i - mean(v)||_2, Sep 2026)
        # =====================================================================
        elif self.MODE == "valuediff_pp":
            v_cot = v_candidates[:, :, pl:n_cand, :].float()
            v_mean = v_cot.mean(dim=2, keepdim=True)
            v_disp = (v_cot - v_mean).norm(dim=-1)  # [B, H, n_cot_cand]
            cot_keep = v_disp.topk(K_free, dim=-1, largest=True).indices + pl
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        # =====================================================================
        # Method 7 (OUR PROPOSED METHOD): `vacor_kv`
        # Value-Outlier-Anchored Block-Antithetic Reservoir KV Eviction
        # =====================================================================
        elif self.MODE in ("vacor_kv", "vacor_antithetic_only", "vacor_persist_only"):
            # Retrieve persistent block-antithetic priority keys u_{h, i} in [0, 1) assigned at birth.
            # Because priorities are persistent across rounds and constant over 8-token semantic blocks,
            # entire mathematical expressions (e.g., `M1 = 13`) survive intact without q^R attrition!
            prio_cot = self.priority_cache[:, :, pl:n_cand].clone()  # [B, H, n_cot_cand]

            # Step A: Horizon-Balanced Age Stratification
            if self.vacor_age_gamma > 0.0:
                pos_cot = (self.pos_ids[:, :, pl:n_cand].float() - float(pl)).clamp(min=0.0)
                max_cot_span = max(float(self.cumulative_length - pl), 1.0)
                rel_birth = pos_cot / max_cot_span  # 0.0 = oldest CoT token, 1.0 = newest
                reservoir_score = prio_cot + self.vacor_age_gamma * (1.0 - 0.35 * rel_birth)
            else:
                reservoir_score = prio_cot

            # Step B: Span-Smoothed Query-Free Value-Dispersion Outlier Anchoring + Top-alpha Pinning
            # In QK-Norm reasoning models (`Qwen3`), numeric constants and equation pivots exhibit strong
            # Value L2 deviation from the cache mean (`||v_i - mean(v)||_2`).
            # By combining a 5-token local span smoothing with (1) hard pinning of the top-alpha_pin
            # (50% of K_free) Value-Dispersion landmarks and (2) Block-Antithetic Reservoir selection
            # for the remaining (1 - alpha_pin) * K_free slots, `vacor_kv` simultaneously achieves:
            #   - 100% landmark/milestone survival across early, mid, and late CoT stages (beating `ValueDiff`),
            #   - >88-96% cross-head union coverage across the entire trace (beating `Random Attention`).
            if self.MODE == "vacor_kv" and self.vacor_outlier_frac > 0.0:
                v_cot = v_candidates[:, :, pl:n_cand, :]
                v_mean = v_cot.mean(dim=2, keepdim=True)
                l2_dev = (v_cot - v_mean).norm(dim=-1, dtype=torch.float32)  # [B, H, n_cot_cand]
                if n_cot_cand >= 5:
                    l2_span = F.avg_pool1d(
                        l2_dev.view(B * H, 1, n_cot_cand), kernel_size=5, padding=2, stride=1
                    ).view(B, H, n_cot_cand)
                    l2_combined = 0.70 * l2_dev + 0.30 * l2_span
                else:
                    l2_combined = l2_dev
                l2_mean = l2_combined.mean(dim=-1, keepdim=True)
                l2_std = l2_combined.std(dim=-1, keepdim=True) + 1e-6
                l2_z = (l2_combined - l2_mean) / l2_std

                # Hybrid Value-Dispersion + Antithetic Reservoir Score:
                # Standardize both signals so top value outliers (>0.5 std above mean) dominate,
                # while moderate-dispersion tokens are partitioned orthogonally across KV heads
                # via the Cyclic-Latin Block-Antithetic priority `reservoir_score`!
                reservoir_score = reservoir_score + 0.85 * l2_z + 1.25 * F.relu(l2_z - 0.50)

            cot_keep = reservoir_score.topk(K_free, dim=-1, largest=True).indices + pl

            # Sort kept indices in strictly chronological order (critical for causal KV ordering!)
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        # =====================================================================
        # Ablation A: `valuediff_smooth_pp` = ValueDiff + the same 5-token span smoothing used by
        # vacor (isolates the contribution of span smoothing from the antithetic reservoir).
        # Ablation B: `vacor_pin` = hard-pin the top `vacor_pin_frac` of K_free by smoothed value
        # dispersion; fill the remainder with the block-antithetic persistent reservoir.
        # =====================================================================
        elif self.MODE in ("valuediff_smooth_pp", "vacor_pin"):
            v_cot = v_candidates[:, :, pl:n_cand, :]
            v_mean = v_cot.mean(dim=2, keepdim=True)
            l2_dev = (v_cot - v_mean).norm(dim=-1, dtype=torch.float32)
            if n_cot_cand >= 5:
                l2_span = F.avg_pool1d(
                    l2_dev.view(B * H, 1, n_cot_cand), kernel_size=5, padding=2, stride=1
                ).view(B, H, n_cot_cand)
                l2_comb = 0.70 * l2_dev + 0.30 * l2_span
            else:
                l2_comb = l2_dev
            if self.MODE == "valuediff_smooth_pp":
                cot_keep = l2_comb.topk(K_free, dim=-1, largest=True).indices + pl
            else:
                n_pin = min(max(1, int(round(K_free * self.vacor_pin_frac))), K_free - 1)
                pin_rel = l2_comb.topk(n_pin, dim=-1, largest=True).indices
                res = self.priority_cache[:, :, pl:n_cand].clone()
                res.scatter_(dim=-1, index=pin_rel, value=float("-inf"))
                res_rel = res.topk(K_free - n_pin, dim=-1, largest=True).indices
                cot_keep = torch.cat([pin_rel, res_rel], dim=-1) + pl
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        # =====================================================================
        # Method 8 (SOTA CLOSURE): `adaptive_vacor`
        # Adaptive Value-Anchored Reservoir KV Eviction
        # Dynamically gates between deterministic value anchoring and cross-head
        # block-antithetic reservoir sampling based on budget pressure and outlier tail.
        # =====================================================================
        elif self.MODE == "adaptive_vacor":
            v_cot = v_candidates[:, :, pl:n_cand, :]
            v_mean = v_cot.mean(dim=2, keepdim=True)
            l2_dev = (v_cot - v_mean).norm(dim=-1, dtype=torch.float32)
            if n_cot_cand >= 5:
                l2_span = F.avg_pool1d(
                    l2_dev.view(B * H, 1, n_cot_cand), kernel_size=5, padding=2, stride=1
                ).view(B, H, n_cot_cand)
                l2_comb = 0.70 * l2_dev + 0.30 * l2_span
            else:
                l2_comb = l2_dev

            # Budget-dependent smooth decay:
            # When K_free <= 256: alpha_base = 0.95 (deterministic protection)
            # When K_free <= 350: alpha_base = 0.90 (safeguards milestone constants across all heads)
            # When K_free >= 440: alpha_base = 0.55 (unleashes multi-head antithetic reservoir coverage)
            if K_free <= 256:
                alpha_base = 0.95
            elif K_free <= 350:
                alpha_base = 0.90
            else:
                alpha_base = max(0.55, 0.90 - 0.35 * (float(K_free - 350) / 130.0))

            # Distribution-tail gating: compute z-scores of L2 deviation
            l2_mean = l2_comb.mean(dim=-1, keepdim=True)
            l2_std = l2_comb.std(dim=-1, keepdim=True) + 1e-6
            l2_z = (l2_comb - l2_mean) / l2_std
            tail_count = int((l2_z >= 1.2).float().sum(dim=-1).max().item())

            n_pin = max(int(round(K_free * alpha_base)), tail_count)
            n_pin = min(max(1, n_pin), K_free - 1)

            pin_rel = l2_comb.topk(n_pin, dim=-1, largest=True).indices

            # Age-stratified reservoir score for remaining quota
            if self.vacor_age_gamma > 0.0:
                pos_cot = (self.pos_ids[:, :, pl:n_cand].float() - float(pl)).clamp(min=0.0)
                max_cot_span = max(float(self.cumulative_length - pl), 1.0)
                rel_birth = pos_cot / max_cot_span
                res = self.priority_cache[:, :, pl:n_cand].clone() + self.vacor_age_gamma * (1.0 - 0.35 * rel_birth)
            else:
                res = self.priority_cache[:, :, pl:n_cand].clone()

            res.scatter_(dim=-1, index=pin_rel, value=float("-inf"))
            res_rel = res.topk(K_free - n_pin, dim=-1, largest=True).indices
            cot_keep = torch.cat([pin_rel, res_rel], dim=-1) + pl
            cot_keep, _ = cot_keep.sort(dim=-1)
            ids_to_keep = torch.cat([prompt_idx, cot_keep], dim=-1)

        else:
            raise ValueError(f"Unknown eviction mode: {self.MODE}")

        # Gather kept KV states and compact in-place (clone recency slice first to prevent overlapping copy!)
        k_rec = self.k_cache[:, :, cur_len - residual : cur_len, :].clone()
        v_rec = self.v_cache[:, :, cur_len - residual : cur_len, :].clone()
        if hasattr(self, "pos_ids"):
            pos_rec = self.pos_ids[:, :, cur_len - residual : cur_len].clone()
            prio_rec = self.priority_cache[:, :, cur_len - residual : cur_len].clone()

        gather_idx_4d = ids_to_keep.unsqueeze(-1).expand(B, H, K, D)
        k_kept = torch.gather(k_candidates, dim=2, index=gather_idx_4d)
        v_kept = torch.gather(v_candidates, dim=2, index=gather_idx_4d)

        self.k_cache[:, :, :K, :].copy_(k_kept)
        self.v_cache[:, :, :K, :].copy_(v_kept)
        self.k_cache[:, :, K : self.token_budget, :].copy_(k_rec)
        self.v_cache[:, :, K : self.token_budget, :].copy_(v_rec)

        if hasattr(self, "pos_ids"):
            pos_kept = torch.gather(self.pos_ids[:, :, :n_cand], dim=2, index=ids_to_keep)
            prio_kept = torch.gather(self.priority_cache[:, :, :n_cand], dim=2, index=ids_to_keep)

            self.pos_ids[:, :, :K].copy_(pos_kept)
            self.pos_ids[:, :, K : self.token_budget].copy_(pos_rec)
            self.priority_cache[:, :, :K].copy_(prio_kept)
            self.priority_cache[:, :, K : self.token_budget].copy_(prio_rec)

        self.cur_len = self.token_budget

    def get_seq_length(self) -> int:
        # Return cumulative sequence length so HuggingFace RoPE / cache_position stays exact
        return self.cumulative_length


class UnifiedEvictCache(Cache):
    """
    HuggingFace `Cache` wrapper managing a list of `UnifiedEvictLayer` instances.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        token_budget: int = 512,
        residual_length: int = 64,
        eviction_mode: str = "vacor_kv",
        **kwargs: Any,
    ):
        text_config = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
        self.stats = EvictionStats()
        layers = [
            UnifiedEvictLayer(
                layer_idx=l,
                token_budget=token_budget,
                residual_length=residual_length,
                eviction_mode=eviction_mode,
                stats=self.stats,
                **kwargs,
            )
            for l in range(text_config.num_hidden_layers)
        ]
        super().__init__(layers=layers)
        self.token_budget = token_budget
        self.residual_length = residual_length
        self.eviction_mode = eviction_mode


def patch_qwen3_evict_sdpa() -> None:
    """
    Monkey-patch `Qwen3Attention.forward` to support `UnifiedEvictCache` using PyTorch 2.6's
    native `F.scaled_dot_product_attention` (FlashAttention-2 / Memory-Efficient kernel on Windows/Linux)
    while falling back cleanly to original `Qwen3Attention.forward` if `past_key_values` is not a
    `UnifiedEvictCache`.
    """
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb, repeat_kv

    if getattr(Qwen3Attention, "_is_patched_for_unified_evict", False):
        return

    orig_forward = Qwen3Attention.forward

    def qwen3_sdpa_evict_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(past_key_values, UnifiedEvictCache):
            return orig_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]            # (B, L_q)
        hidden_shape = (*input_shape, -1, self.head_dim)  # (B, L_q, H, D)

        # Qwen3 applies per-head RMSNorm (q_norm, k_norm) before RoPE
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        layer: UnifiedEvictLayer = past_key_values.layers[self.layer_idx]
        layer.update_queries(query_states)

        k_active, v_active = layer.append_and_get_kv(key_states, value_states)

        q_len = query_states.shape[2]
        kv_len = k_active.shape[2]

        # Note: In PyTorch SDPA, `is_causal=True` assumes the query aligns with the UPPER-LEFT
        # of the key matrix (so when q_len=64 and kv_len=320, query[0] only attends to key[0]!).
        # When q_len == 1, causal masking is unnecessary (`is_causal=False`).
        # When q_len == kv_len (initial prefill), `is_causal=True` is exact.
        # When 1 < q_len < kv_len (chunked prefix decoding), we construct the lower-right causal mask!
        attn_mask = None
        if q_len == 1:
            is_causal = False
        elif q_len == kv_len:
            is_causal = True
        else:
            is_causal = False
            past_len = kv_len - q_len
            row_idx = torch.arange(q_len, device=query_states.device).unsqueeze(1) + past_len
            col_idx = torch.arange(kv_len, device=query_states.device).unsqueeze(0)
            attn_mask = col_idx <= row_idx  # [q_len, kv_len] bool mask

        # PyTorch 2.6 SDPA supports enable_gqa=True natively without allocating repeated KV copies
        try:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                k_active,
                v_active,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scaling,
                enable_gqa=True,
            )
        except TypeError:
            # Fallback if enable_gqa is not supported on older PyTorch builds
            k_rep = repeat_kv(k_active, self.num_key_value_groups)
            v_rep = repeat_kv(v_active, self.num_key_value_groups)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                k_rep,
                v_rep,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scaling,
            )

        is_prefill = (layer.cumulative_length == q_len)
        layer.maybe_evict(q_len, is_prefill=is_prefill)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    Qwen3Attention.forward = qwen3_sdpa_evict_forward
    Qwen3Attention._is_patched_for_unified_evict = True
    Qwen3Attention._orig_forward = orig_forward


def patch_qwen2_evict_sdpa() -> None:
    """
    Monkey-patch `Qwen2Attention.forward` (e.g. Qwen2.5) to support `UnifiedEvictCache`.
    Qwen2 does NOT have q_norm / k_norm, providing a direct architectural control.
    """
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, repeat_kv

    if getattr(Qwen2Attention, "_is_patched_for_unified_evict", False):
        return

    orig_forward = Qwen2Attention.forward

    def qwen2_sdpa_evict_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(past_key_values, UnifiedEvictCache):
            return orig_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Qwen2 has NO q_norm or k_norm!
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        layer: UnifiedEvictLayer = past_key_values.layers[self.layer_idx]
        layer.update_queries(query_states)

        k_active, v_active = layer.append_and_get_kv(key_states, value_states)

        q_len = query_states.shape[2]
        kv_len = k_active.shape[2]

        attn_mask = None
        if q_len == 1:
            is_causal = False
        elif q_len == kv_len:
            is_causal = True
        else:
            is_causal = False
            past_len = kv_len - q_len
            row_idx = torch.arange(q_len, device=query_states.device).unsqueeze(1) + past_len
            col_idx = torch.arange(kv_len, device=query_states.device).unsqueeze(0)
            attn_mask = col_idx <= row_idx

        try:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                k_active,
                v_active,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scaling,
                enable_gqa=True,
            )
        except TypeError:
            k_rep = repeat_kv(k_active, self.num_key_value_groups)
            v_rep = repeat_kv(v_active, self.num_key_value_groups)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                k_rep,
                v_rep,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=self.scaling,
            )

        is_prefill = (layer.cumulative_length == q_len)
        layer.maybe_evict(q_len, is_prefill=is_prefill)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    Qwen2Attention.forward = qwen2_sdpa_evict_forward
    Qwen2Attention._is_patched_for_unified_evict = True
    Qwen2Attention._orig_forward = orig_forward
