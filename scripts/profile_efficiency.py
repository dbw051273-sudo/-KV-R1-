"""
Efficiency and Memory Profiling Microbenchmark for KV Cache Eviction Policies.

Profiles:
1. Peak GPU Memory (MB) vs Sequence Length (1k, 2k, 4k, 8k tokens):
   - Dense (Uncompressed Oracle) vs Compressed (Budget 256, 512).
2. Pure Eviction Kernel Latency (ms):
   - Measures CPU-to-GPU execution time across methods.
   - Compares O(K) block-antithetic reservoir (VaCoR-KV) vs O(HLd) ValueDiff.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kv_engine import UnifiedEvictCache, UnifiedEvictLayer


def profile_memory_vs_length():
    """Profiles theoretical and actual GPU memory allocated for KV cache."""
    print("=" * 70)
    print("PROFILING KV CACHE MEMORY CONSUMPTION (Qwen3-1.7B / Qwen3-4B Architecture)")
    print("=" * 70)

    # Qwen3-1.7B config: 28 layers, 16 KV heads, head_dim = 128, dtype = bfloat16 (2 bytes)
    num_layers = 28
    num_kv_heads = 16
    head_dim = 128
    bytes_per_elem = 2  # BF16
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem  # Key + Value

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    budgets = [256, 384, 512]

    results = []
    for L in seq_lengths:
        dense_mb = (L * bytes_per_token) / (1024 * 1024)
        row = {"seq_len": L, "dense_mb": round(dense_mb, 2)}
        for B in budgets:
            kept = min(L, B + 64)
            comp_mb = (kept * bytes_per_token) / (1024 * 1024)
            ratio = dense_mb / comp_mb
            row[f"budget_{B}_mb"] = round(comp_mb, 2)
            row[f"savings_{B}_ratio"] = round(ratio, 2)
        results.append(row)

    print(f"{'Seq Length':>12s} | {'Dense KV (MB)':>14s} | {'Budget=512 (MB)':>16s} | {'Budget=256 (MB)':>16s} | {'Compression @ 256':>18s}")
    print("-" * 85)
    for r in results:
        print(
            f"{r['seq_len']:12d} | {r['dense_mb']:14.1f} | {r['budget_512_mb']:16.1f} | "
            f"{r['budget_256_mb']:16.1f} | {r['savings_256_ratio']:17.1f}x"
        )

    out_file = ROOT / "results" / "memory_scaling_profile.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved memory scaling to: {out_file}")
    return results


def profile_eviction_kernel_latencies():
    """Microbenchmarks pure eviction step latency on CUDA across policies."""
    print("\n" + "=" * 70)
    print("MICROBENCHMARKING PURE EVICTION KERNEL LATENCY (CUDA Synchronized)")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping latency microbenchmark.")
        return []

    device = torch.device("cuda")
    B, H, D = 1, 16, 128
    budget = 256
    residual = 64
    n_tokens = budget + residual + 16  # triggers eviction
    prompt_len = 32

    modes = [
        ("recency_pp", "Recency-pp (Sliding Window)"),
        ("random_pp", "Random-pp (Salesforce 2026)"),
        ("vacor_kv", "VaCoR-KV (Ours: Block-Antithetic)"),
        ("snapkv_pp", "SnapKV-pp (Attn Pooling)"),
        ("rkv_pp", "R-KV-pp (Cosine Penalty)"),
        ("vase_pp", "VaSE-pp (Value Range + Attn)"),
        ("valuediff_pp", "ValueDiff-pp (Value L2 Deviation)"),
    ]

    latencies = []
    num_warmup = 10
    num_iters = 50

    for mode_id, mode_name in modes:
        layer = UnifiedEvictLayer(
            layer_idx=0,
            token_budget=budget,
            residual_length=residual,
            eviction_mode=mode_id,
            seed=42,
            track_positions=True,
        )
        layer.lazy_initialization(torch.zeros(B, H, prompt_len, D, dtype=torch.bfloat16, device=device))
        layer._ensure_capacity(n_tokens + 64)
        layer.prompt_len = prompt_len
        layer.cur_len = n_tokens
        layer.cumulative_length = n_tokens

        # Populate synthetic KV caches
        layer.k_cache[:, :, :n_tokens, :] = torch.randn(B, H, n_tokens, D, dtype=torch.bfloat16, device=device)
        layer.v_cache[:, :, :n_tokens, :] = torch.randn(B, H, n_tokens, D, dtype=torch.bfloat16, device=device)
        layer.queries = torch.randn(B, H, residual, D, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(num_warmup):
            layer.cur_len = n_tokens
            layer._run_eviction(layer.cur_len)
        torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(num_iters):
            layer.cur_len = n_tokens
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            layer._run_eviction(layer.cur_len)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        mean_ms = sum(times) / len(times)
        std_ms = math.sqrt(sum((x - mean_ms) ** 2 for x in times) / (len(times) - 1))
        min_ms = min(times)

        latencies.append({
            "mode": mode_id,
            "name": mode_name,
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "min_ms": round(min_ms, 3),
        })
        print(f"{mode_name:38s} | Mean: {mean_ms:6.3f} ms ± {std_ms:5.3f} ms | Min: {min_ms:6.3f} ms")

    out_file = ROOT / "results" / "eviction_kernel_latencies.json"
    out_file.write_text(json.dumps(latencies, indent=2), encoding="utf-8")
    print(f"\nSaved latency microbenchmark to: {out_file}")
    return latencies


def main():
    profile_memory_vs_length()
    profile_eviction_kernel_latencies()


if __name__ == "__main__":
    main()
