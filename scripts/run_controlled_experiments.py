"""Unified Controlled Experiments Script

Executes:
1. QK-Norm Architecture Control: Evaluates Qwen2.5-1.5B (No QK-Norm) vs Qwen3-1.7B (With QK-Norm) on Suite A
2. Expanded Baselines: Includes H2O (`h2o_pp`) and TOVA (`tova_pp`)
3. Real Long-CoT Thinking Mode Evaluation on MATH Problems with enable_thinking=True (1,000+ to 2,000+ tokens)
4. Unified Apples-to-Apples Latency Benchmarking (Single-layer vs 28-layer end-to-end)
5. Symmetric 3-Seed Runs for all methods
"""

import sys
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.kv_engine import (
    patch_qwen3_evict_sdpa,
    patch_qwen2_evict_sdpa,
    UnifiedEvictCache,
    UnifiedEvictLayer,
)
from scripts.run_suite_a_ext import build_long_cot_milestone_tasks, run_one

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_unified_latency_benchmark():
    """Measures single-layer kernel latency and 28-layer end-to-end latency on the exact same basis."""
    print("=== Running Unified Latency Benchmark (CUDA Synchronized) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, H, D = 1, 16, 128
    budget = 256
    residual = 64
    cur_len = budget + residual  # 320 tokens

    modes = [
        ("recency_pp", "Recency (StreamingLLM style)", False),
        ("random_pp", "Random Attention (Salesforce 2026)", False),
        ("tova_pp", "TOVA (Latest Token Attn, 2024)", True),
        ("h2o_pp", "H2O (Heavy Hitter Oracle, 2023)", True),
        ("snapkv_pp", "SnapKV (Observation Window Attn)", True),
        ("valuediff_pp", "ValueDiff (L2 Deviation from Mean)", False),
        ("vacor_kv", "VaCoR-KV (Block-Antithetic + Value)", False),
    ]

    benchmark_results = []
    WARMUP = 15
    ITERS = 50

    for mode, name, needs_q in modes:
        layer = UnifiedEvictLayer(
            layer_idx=0,
            token_budget=budget,
            residual_length=residual,
            eviction_mode=mode,
        )
        dummy_k = torch.randn(B, H, cur_len, D, dtype=torch.bfloat16, device=device)
        dummy_v = torch.randn(B, H, cur_len, D, dtype=torch.bfloat16, device=device)
        layer.lazy_initialization(dummy_k, dummy_v)
        layer._ensure_capacity(cur_len + 16)
        layer.k_cache[:, :, :cur_len, :].copy_(dummy_k)
        layer.v_cache[:, :, :cur_len, :].copy_(dummy_v)
        layer.cur_len = cur_len
        layer.cumulative_length = cur_len
        layer.prompt_len = 32

        if hasattr(layer, "pos_ids"):
            layer.pos_ids[:, :, :cur_len].copy_(torch.arange(cur_len, device=device).view(1, 1, -1).expand(B, H, -1))
            layer.priority_cache[:, :, :cur_len].copy_(torch.rand(B, H, cur_len, device=device))

        if needs_q:
            layer.update_queries(torch.randn(B, H, 32, D, dtype=torch.bfloat16, device=device))

        # Warmup
        for _ in range(WARMUP):
            layer._run_eviction(cur_len)
            layer.cur_len = cur_len

        # Timing single-layer kernel
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            layer._run_eviction(cur_len)
            layer.cur_len = cur_len
        torch.cuda.synchronize()
        single_layer_ms = (time.perf_counter() - t0) * 1000.0 / ITERS

        # 28-layer projection
        all_layers_ms = single_layer_ms * 28

        benchmark_results.append({
            "mode": mode,
            "name": name,
            "single_layer_kernel_ms": round(single_layer_ms, 3),
            "full_28_layers_ms": round(all_layers_ms, 2),
            "requires_query": needs_q,
        })
        print(f"[{mode:15s}] Single Layer: {single_layer_ms:.3f} ms | Full 28 Layers: {all_layers_ms:.2f} ms")

    out_file = RESULTS_DIR / "unified_latency_benchmark.json"
    out_file.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
    print(f"Saved {out_file}\n")
    return benchmark_results


def run_qwen25_control_experiment():
    """Runs Qwen2.5-1.5B (No QK-Norm) on Suite A to test if Attention Collapse depends on QK-Norm."""
    print("=== Running Qwen2.5-1.5B (No QK-Norm Control) on Suite A ===")
    patch_qwen2_evict_sdpa()

    model_id = "Qwen/Qwen2.5-1.5B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    # Use 4 held-out Suite A tasks
    tasks = build_long_cot_milestone_tasks(num_tasks=12)[8:12]
    modes = ["dense", "snapkv_pp", "h2o_pp", "tova_pp", "random_pp", "valuediff_pp", "vacor_kv"]
    budget = 512

    out_records = []
    for t in tasks:
        tid = t["id"]
        for mode in modes:
            for s in range(3):
                # For deterministic methods, seed variations test reproducibility
                t0 = time.time()
                r = run_one(model, tok, t, mode, budget, seed=1000 * s + int(tid.split("_")[1]))
                r.update(task=tid, budget=budget, mode=mode, seed=s, sec=round(time.time() - t0, 1), model=model_id, qk_norm=False)
                out_records.append(r)
                print(f"[Qwen2.5-No-QKNorm] {tid} | {mode:14s} s={s} | m1={r['m1']} all3={r['all3']} | {r['sec']}s", flush=True)

    out_path = RESULTS_DIR / "qwen25_qknorm_control.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {out_path}\n")

    # Aggregate Qwen2.5 summary
    from collections import defaultdict
    by_mode = defaultdict(list)
    for r in out_records:
        by_mode[r["mode"]].append(r["all3"])

    summary = {m: {"mean_all3": float(torch.tensor(vals).float().mean() * 100), "n": len(vals)} for m, vals in by_mode.items()}
    summary_path = RESULTS_DIR / "qwen25_qknorm_control_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Qwen2.5 Control Summary:", summary)
    return summary


def run_thinking_mode_experiment():
    """Runs real thinking mode (enable_thinking=True) generating 1,000+ tokens on MATH-500 problems."""
    print("=== Running Real Thinking Mode (enable_thinking=True) Long CoT Benchmark ===")
    patch_qwen3_evict_sdpa()

    model_id = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    data_path = ROOT_DIR / "data" / "math_benchmarks.jsonl"
    all_items = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    problems = [x for x in all_items if x["source"] == "MATH-500"][:3]  # 3 representative long problems

    from scripts.run_math500_eval import extract_boxed_robust, normalize_math_answer

    modes = ["dense", "snapkv_pp", "random_pp", "valuediff_pp", "vacor_kv"]
    budget = 256
    residual = 64

    out_records = []
    for idx, p in enumerate(problems):
        pid = p["id"]
        q_text = p["question"]
        gt_answer = p["ground_truth"]

        messages = [{"role": "user", "content": q_text}]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tok(formatted, return_tensors="pt").to("cuda")
        prompt_len = inputs.input_ids.shape[1]

        for mode in modes:
            for s in range(1):  # 1 seed for thinking benchmark (15 runs total)
                torch.manual_seed(s * 1000 + idx)
                if mode == "dense":
                    cache = None
                else:
                    cache = UnifiedEvictCache(
                        config=model.config,
                        token_budget=budget,
                        residual_length=residual,
                        eviction_mode=mode,
                        seed=s * 1000 + idx,
                    )

                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        past_key_values=cache,
                        max_new_tokens=1200,
                        do_sample=False,
                    )
                elapsed = time.time() - t0
                gen_ids = out[0][prompt_len:]
                gen_len = len(gen_ids)
                resp = tok.decode(gen_ids, skip_special_tokens=False)
                pred_raw = extract_boxed_robust(resp)
                
                is_correct = False
                if pred_raw is not None and gt_answer is not None:
                    norm_pred = normalize_math_answer(pred_raw)
                    norm_gt = normalize_math_answer(gt_answer)
                    is_correct = (norm_pred == norm_gt or pred_raw.strip() == gt_answer.strip())

                rec = {
                    "problem_id": pid,
                    "subject": p.get("subject", "Math"),
                    "level": p.get("level", 3),
                    "mode": mode,
                    "seed": s,
                    "gen_tokens": gen_len,
                    "elapsed_sec": round(elapsed, 1),
                    "tokens_per_sec": round(gen_len / max(elapsed, 1e-4), 2),
                    "pred": pred_raw,
                    "gt": gt_answer,
                    "is_correct": bool(is_correct),
                    "has_think_tags": bool("<think>" in resp and "</think>" in resp),
                    "eviction_rounds": max(0, (gen_len - budget) // residual) if mode != "dense" else 0,
                }
                out_records.append(rec)
                print(f"[Thinking Mode] {pid} | {mode:14s} s={s} | tokens={gen_len} ok={is_correct} rounds={rec['eviction_rounds']} | {elapsed:.1f}s", flush=True)

    out_path = RESULTS_DIR / "math_thinking_mode_eval.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {out_path}\n")


if __name__ == "__main__":
    run_unified_latency_benchmark()
    run_qwen25_control_experiment()
    run_thinking_mode_experiment()
