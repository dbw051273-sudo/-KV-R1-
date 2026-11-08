"""
Diagnostic probe on real `Qwen/Qwen3-1.7B` hidden states & KV cache during Suite A & Suite B:
1. Inspects why multi-token milestone spans (`M1 = 13`) succeed in `valuediff_pp` and how to make
   `vacor_kv` preserve local contiguous semantic spans (Span-Pooling + Value-Dispersion + Antithetic Reservoir).
2. Inspects the actual generation lengths of `Qwen/Qwen3-1.7B` on `GSM8K` and `MATH-500` with
   `enable_thinking=True` vs `enable_thinking=False`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_benchmarks import build_long_cot_milestone_tasks
from src.kv_engine import UnifiedEvictCache, patch_qwen3_evict_sdpa


@torch.no_grad()
def main():
    patch_qwen3_evict_sdpa()
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    tasks = build_long_cot_milestone_tasks(num_tasks=1)
    t0 = tasks[0]
    p_ids = tokenizer(t0["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    c_ids = tokenizer(t0["cot_prefix"], add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    full_ids = torch.cat([p_ids, c_ids], dim=1)
    print(f"Prompt len: {p_ids.shape[1]}, CoT prefix len: {c_ids.shape[1]}, Total: {full_ids.shape[1]}")

    # Locate exact token indices for "M1 = 13", "M2 = 24", "M3 = 15"
    tokens = [tokenizer.decode([tid]) for tid in full_ids[0].tolist()]
    for idx in range(len(tokens) - 4):
        window = "".join(tokens[idx : idx + 6])
        if "M1 =" in window or "M2 =" in window or "M3 =" in window:
            print(f"  Pos {idx:4d}: {tokens[idx:idx+6]}")

    # Run a forward pass with dense cache to inspect value-dispersion scores around M1, M2, M3
    cache = UnifiedEvictCache(model.config, token_budget=4096, residual_length=64, eviction_mode="dense")
    model(input_ids=full_ids, past_key_values=cache, use_cache=True)

    layer = cache.layers[14]  # Middle layer
    v_cot = layer.v_cache[0, :, p_ids.shape[1] : full_ids.shape[1], :].float()  # [H, N_cot, D]
    v_mean = v_cot.mean(dim=1, keepdim=True)
    l2_dev = (v_cot - v_mean).norm(dim=-1)  # [H, N_cot]
    # Also check 1D smoothed (avg_pool1d kernel=5) L2 deviation
    l2_smooth = F.avg_pool1d(l2_dev.unsqueeze(0), kernel_size=5, padding=2, stride=1).squeeze(0)

    print("\nValue L2 deviation percentiles across CoT tokens (Layer 14, Head 0):")
    q_vals = torch.quantile(l2_dev[0], torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.95], device="cuda"))
    print("  Quantiles [10, 25, 50, 75, 90, 95%]:", [round(x, 3) for x in q_vals.tolist()])

    for idx in range(p_ids.shape[1], full_ids.shape[1] - 4):
        window = "".join(tokens[idx : idx + 5])
        if "M1 =" in window or "M2 =" in window or "M3 =" in window:
            rel = idx - p_ids.shape[1]
            raw_scores = l2_dev[:, rel : rel + 5].mean(dim=0).tolist()
            sm_scores = l2_smooth[:, rel : rel + 5].mean(dim=0).tolist()
            print(f"  Milestone @ rel={rel:4d} ({tokens[idx:idx+5]}):")
            print(f"    Raw L2 dev (head avg):    {[round(x, 2) for x in raw_scores]}")
            print(f"    Smooth L2 dev (head avg): {[round(x, 2) for x in sm_scores]}")


if __name__ == "__main__":
    main()
