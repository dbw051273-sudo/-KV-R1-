"""PagedVaCoR: Block-Aligned Reservoir KV Cache Eviction for PagedAttention (vLLM / SGLang)

Solves internal memory fragmentation in physical page allocations (block_size=16 or 32):
1. Aggregates value dispersion at physical block granularity: S(P_j) = max_{t in P_j} ||v_t - \bar{v}||_2
2. Pins top-alpha physical blocks to retain high-dispersion landmark constants.
3. Performs block-level antithetic reservoir sampling across heads for remaining blocks.
4. When a block is evicted, its entire physical frame is freed back to the page allocator.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F

class PagedVaCoRManager:
    """Manages block-aligned KV cache retention for PagedAttention engines."""

    def __init__(
        self,
        num_layers: int = 28,
        num_kv_heads: int = 16,
        head_dim: int = 128,
        block_size: int = 16,
        max_blocks_per_seq: int = 16,  # 16 * 16 = 256 tokens total budget
        pin_ratio: float = 0.5,        # 50% blocks pinned by value dispersion
        sink_blocks: int = 1,          # First block (initial tokens) strictly preserved
        recent_blocks: int = 4,        # Last 4 blocks (recent window) strictly preserved
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks_per_seq
        self.pin_ratio = pin_ratio
        self.sink_blocks = sink_blocks
        self.recent_blocks = recent_blocks

    def compute_block_eviction(
        self,
        value_cache: torch.Tensor,       # [B, H, total_tokens, D]
        allocated_block_ids: List[int],  # list of physical block indices
        head_idx: int = 0,
        seed: int = 42,
    ) -> List[int]:
        """Determines which physical block IDs to keep for a specific KV head.
        
        Returns:
            keep_block_ids: List of retained physical block IDs of length <= max_blocks.
        """
        num_allocated = len(allocated_block_ids)
        if num_allocated <= self.max_blocks:
            return allocated_block_ids

        # Protected block indices (Sink & Recency)
        sink_ids = allocated_block_ids[:self.sink_blocks]
        recent_ids = allocated_block_ids[-self.recent_blocks:]
        candidate_ids = allocated_block_ids[self.sink_blocks : -self.recent_blocks]

        if not candidate_ids:
            return allocated_block_ids[-self.max_blocks:]

        num_to_keep = self.max_blocks - len(sink_ids) - len(recent_ids)
        if num_to_keep <= 0:
            return sink_ids + recent_ids[-self.max_blocks + len(sink_ids):]

        # 1. Compute block-level value dispersion scores
        # candidate tokens in value_cache
        B, H, T, D = value_cache.shape
        mean_v = value_cache[:, head_idx, :, :].mean(dim=1, keepdim=True)  # [B, 1, D]
        v_diff = torch.norm(value_cache[:, head_idx, :, :] - mean_v, p=2, dim=-1)[0]  # [T]

        block_scores = []
        for b_idx in candidate_ids:
            start_t = b_idx * self.block_size
            end_t = min(start_t + self.block_size, T)
            if start_t < T:
                # Max token dispersion in block
                score = v_diff[start_t:end_t].max().item()
            else:
                score = 0.0
            block_scores.append((score, b_idx))

        # 2. Pin top-alpha fraction of candidate blocks
        num_pinned = int(num_to_keep * self.pin_ratio)
        block_scores.sort(key=lambda x: x[0], reverse=True)
        pinned_blocks = [b_idx for _, b_idx in block_scores[:num_pinned]]
        unpinned_blocks = [b_idx for _, b_idx in block_scores[num_pinned:]]

        # 3. Block-level antithetic reservoir sampling for remaining quota
        num_reservoir = num_to_keep - num_pinned
        if num_reservoir > 0 and unpinned_blocks:
            # Deterministic antithetic permutation per head
            # Stride shift: phi_h = (head_idx * 7) % len(unpinned_blocks)
            stride = 7
            shift = (head_idx * stride + seed) % len(unpinned_blocks)
            permuted = unpinned_blocks[shift:] + unpinned_blocks[:shift]
            sampled_blocks = permuted[:num_reservoir]
        else:
            sampled_blocks = []

        # 4. Final union of physical blocks
        retained = sink_ids + pinned_blocks + sampled_blocks + recent_ids
        retained.sort()  # Maintain temporal order of physical pages
        return retained
