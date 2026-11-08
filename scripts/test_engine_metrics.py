"""
Unit & Mathematical Property Benchmark for `UnifiedEvictLayer` (2025-2026 SOTA vs. VaCoR-KV).

Tests and quantifies:
1. Cross-Head Union Coverage (`union_frac`) vs. I.I.D. expectation (`indep_frac`) and Correlation Deficit.
2. Long-Horizon Early/Mid-CoT Survival Curve across E = 40 eviction rounds (3,072 steps).
3. High-Surprise Value Outlier Retention Rate (simulating `S4` intermediate constant/pivot tokens).
4. Per-Round Eviction Latency (milliseconds on RTX 3080 Ti GPU).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kv_engine import UnifiedEvictLayer


def run_synthetic_eviction_trace(
    mode: str,
    *,
    prompt_len: int = 128,
    decode_steps: int = 2560,
    token_budget: int = 512,
    residual_length: int = 64,
    num_kv_heads: int = 8,
    num_q_heads: int = 16,
    head_dim: int = 128,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    layer = UnifiedEvictLayer(
        layer_idx=0,
        token_budget=token_budget,
        residual_length=residual_length,
        eviction_mode=mode,
        seed=seed,
        track_positions=True,
    )

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 1. Prefill
    k_prefill = torch.randn(1, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype)
    v_prefill = torch.randn(1, num_kv_heads, prompt_len, head_dim, device=device, dtype=dtype) * 0.5
    q_prefill = torch.randn(1, num_q_heads, prompt_len, head_dim, device=device, dtype=dtype)

    layer.update_queries(q_prefill)
    layer.append_and_get_kv(k_prefill, v_prefill)
    layer.maybe_evict(prompt_len)

    # Plant 16 "High-Surprise Value Landmarks" (simulating intermediate derived constants / S4 gates)
    # spaced every 140 steps across the CoT trace: steps 200, 340, 480, ..., 2300
    landmark_steps = [prompt_len + 100 + i * 140 for i in range(16)]
    landmark_set = set(landmark_steps)

    # Create a shared latent direction so attention-based selectors exhibit realistic cross-head correlation
    shared_topic = torch.randn(1, 1, 1, head_dim, device=device, dtype=dtype)
    shared_topic = shared_topic / (shared_topic.norm(dim=-1, keepdim=True) + 1e-6)

    eviction_times_ms = []

    for step in range(prompt_len, prompt_len + decode_steps):
        # Generate realistic K, V, Q
        noise_k = torch.randn(1, num_kv_heads, 1, head_dim, device=device, dtype=dtype)
        # Periodic attention salience modulation so attention selectors prefer shared peaks across heads
        salience = 1.5 if (step % 37 == 0) else 0.3
        k_step = noise_k + salience * shared_topic

        if step in landmark_set:
            # High-information value landmark (large L2 dispersion / value range, e.g., numeric result)
            v_step = torch.randn(1, num_kv_heads, 1, head_dim, device=device, dtype=dtype) * 3.2
        else:
            v_step = torch.randn(1, num_kv_heads, 1, head_dim, device=device, dtype=dtype) * 0.5

        q_step = torch.randn(1, num_q_heads, 1, head_dim, device=device, dtype=dtype) + shared_topic

        layer.update_queries(q_step)
        layer.append_and_get_kv(k_step, v_step)

        will_evict = (
            layer.cur_len > layer.token_budget
            and (layer.cur_len - layer.token_budget) % layer.residual_length == 0
        )
        if will_evict and device == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            layer.maybe_evict(1)
            torch.cuda.synchronize()
            eviction_times_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            t0 = time.perf_counter()
            layer.maybe_evict(1)
            if will_evict:
                eviction_times_ms.append((time.perf_counter() - t0) * 1000.0)

    # Analyze final surviving token positions [1, H_kv, token_budget]
    surviving_pos = layer.pos_ids[0, :, : layer.cur_len].cpu()  # [H_kv, K_total]

    # Candidate pool spans [prompt_len : prompt_len + decode_steps - residual_length)
    cand_start = prompt_len
    cand_end = prompt_len + decode_steps - residual_length
    total_cot_cands = cand_end - cand_start

    # Build binary survival matrix M[h, t] for t in [cand_start, cand_end)
    M = torch.zeros(num_kv_heads, total_cot_cands, dtype=torch.bool)
    for h in range(num_kv_heads):
        pos_h = surviving_pos[h]
        cot_pos = pos_h[(pos_h >= cand_start) & (pos_h < cand_end)] - cand_start
        M[h, cot_pos.long()] = True

    per_head_rate = M.float().mean().item()  # s = K_free / total_cot_cands
    union_mask = M.any(dim=0)                # Survives in at least 1 head
    union_frac = union_mask.float().mean().item()
    indep_expectation = 1.0 - (1.0 - per_head_rate) ** num_kv_heads
    theoretical_max_union = min(1.0, num_kv_heads * per_head_rate)
    correlation_deficit = indep_expectation - union_frac

    # Survival by age quartiles (Q1 = oldest 25% of CoT, Q4 = newest 25% of CoT before recency window)
    q_size = total_cot_cands // 4
    age_quartiles_union = {}
    age_quartiles_head = {}
    for q_idx, label in enumerate(["Q1_oldest", "Q2_early_mid", "Q3_late_mid", "Q4_recent"]):
        s_idx = q_idx * q_size
        e_idx = (q_idx + 1) * q_size if q_idx < 3 else total_cot_cands
        age_quartiles_union[label] = round(union_mask[s_idx:e_idx].float().mean().item() * 100, 2)
        age_quartiles_head[label] = round(M[:, s_idx:e_idx].float().mean().item() * 100, 2)

    # Landmark retention (both per-head average and cross-head union)
    landmark_rel = torch.tensor([s - cand_start for s in landmark_steps if cand_start <= s < cand_end])
    landmark_head_acc = M[:, landmark_rel].float().mean().item() * 100.0
    landmark_union_acc = union_mask[landmark_rel].float().mean().item() * 100.0

    avg_ms = sum(eviction_times_ms[2:]) / max(len(eviction_times_ms[2:]), 1) if len(eviction_times_ms) > 2 else 0.0

    return {
        "mode": mode,
        "per_head_retention_pct": round(per_head_rate * 100, 2),
        "indep_expected_union_pct": round(indep_expectation * 100, 2),
        "theoretical_disjoint_max_pct": round(theoretical_max_union * 100, 2),
        "actual_union_coverage_pct": round(union_frac * 100, 2),
        "correlation_deficit_pct": round(correlation_deficit * 100, 2),
        "landmark_head_retention_pct": round(landmark_head_acc, 2),
        "landmark_union_retention_pct": round(landmark_union_acc, 2),
        "age_quartiles_union_pct": age_quartiles_union,
        "age_quartiles_head_pct": age_quartiles_head,
        "avg_eviction_latency_ms": round(avg_ms, 4),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running UnifiedEvictLayer Mathematical & Systems Verification on {device}...")

    modes = [
        "recency_pp",
        "snapkv_pp",
        "rkv_pp",
        "vase_pp",
        "valuediff_pp",
        "random_pp",
        "vacor_persist_only",
        "vacor_antithetic_only",
        "vacor_kv",
    ]

    results = []
    for mode in modes:
        res = run_synthetic_eviction_trace(
            mode,
            prompt_len=128,
            decode_steps=2560,
            token_budget=512,
            residual_length=64,
            num_kv_heads=8,
            num_q_heads=16,
            head_dim=128,
            device=device,
            seed=42,
        )
        results.append(res)
        print(
            f"[{mode:22s}] UnionCov: {res['actual_union_coverage_pct']:6.2f}% "
            f"(Indep: {res['indep_expected_union_pct']:5.2f}%, Max: {res['theoretical_disjoint_max_pct']:6.2f}%) | "
            f"Landmark(Head/Union): {res['landmark_head_retention_pct']:5.1f}%/{res['landmark_union_retention_pct']:5.1f}% | "
            f"Oldest-Q1 Union: {res['age_quartiles_union_pct']['Q1_oldest']:5.1f}% | "
            f"Latency: {res['avg_eviction_latency_ms']:.3f} ms"
        )

    out_path = ROOT / "results" / "synthetic_engine_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved detailed telemetry to {out_path}")


if __name__ == "__main__":
    main()
