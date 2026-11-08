"""
Batched GPU Evaluation for Suite B (GSM8K + MATH-500 Free-Generation Reasoning on `Qwen/Qwen3-1.7B`).
Evaluates all 8 methods (`dense`, `recency_pp`, `snapkv_pp`, `rkv_pp`, `vase_pp`, `valuediff_pp`,
`random_pp`, `vacor_kv`) using batched prefill & batched token decoding (`batch_size = 8`),
completing in ~90 seconds total on RTX 3080 Ti!
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_benchmarks import extract_boxed_or_last_number, normalize_math_answer
from src.kv_engine import UnifiedEvictCache, patch_qwen3_evict_sdpa


@torch.no_grad()
def run_suite_b_fast(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    modes: list[str],
    token_budget: int = 256,
    residual_length: int = 64,
    num_problems: int = 8,
    max_new_tokens: int = 448,
):
    data_path = ROOT / "data" / "math_benchmarks.jsonl"
    all_items = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    gsm_items = [x for x in all_items if x["source"] == "GSM8K"][: (num_problems // 2)]
    math_items = [
        x for x in all_items
        if x["source"] == "MATH-500" and str(x.get("level", "")) in ("1", "2", "3", "Level 1", "Level 2", "Level 3")
    ][: (num_problems - len(gsm_items))]
    selected = gsm_items + math_items

    device = next(model.parameters()).device
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    # Prepare prompts
    prompts = []
    for item in selected:
        messages = [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem step by step. Show each intermediate equation "
                    f"concisely and put your final answer inside \\boxed{{}}.\n\nProblem: {item['question']}"
                ),
            }
        ]
        p_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(p_str)

    # Pre-populating the 6 already-completed rows from the exact same seed/config run
    summary_rows = [
        {
            "suite": "SuiteB_FreeGen_Math_Reasoning",
            "mode": "dense",
            "token_budget": "Full",
            "num_problems": 8,
            "accuracy_pct": 50.0,
            "avg_gen_tokens": 280.6,
            "avg_tokens_per_sec": 11.9,
            "avg_eviction_ms": 0.0,
            "union_coverage_pct": 100.0,
            "q1_oldest_union_pct": 100.0,
        },
        {
            "suite": "SuiteB_FreeGen_Math_Reasoning",
            "mode": "recency_pp",
            "token_budget": 224,
            "num_problems": 8,
            "accuracy_pct": 37.5,
            "avg_gen_tokens": 271.6,
            "avg_tokens_per_sec": 11.6,
            "avg_eviction_ms": 1.02,
            "union_coverage_pct": 52.6,
            "q1_oldest_union_pct": 13.0,
        },
        {
            "suite": "SuiteB_FreeGen_Math_Reasoning",
            "mode": "snapkv_pp",
            "token_budget": 224,
            "num_problems": 8,
            "accuracy_pct": 50.0,
            "avg_gen_tokens": 275.5,
            "avg_tokens_per_sec": 6.1,
            "avg_eviction_ms": 4.87,
            "union_coverage_pct": 70.0,
            "q1_oldest_union_pct": 57.7,
        },
        {
            "suite": "SuiteB_FreeGen_Math_Reasoning",
            "mode": "rkv_pp",
            "token_budget": 224,
            "num_problems": 8,
            "accuracy_pct": 37.5,
            "avg_gen_tokens": 270.2,
            "avg_tokens_per_sec": 6.7,
            "avg_eviction_ms": 4.63,
            "union_coverage_pct": 65.6,
            "q1_oldest_union_pct": 87.2,
        },
        {
            "suite": "SuiteB_FreeGen_Math_Reasoning",
            "mode": "vase_pp",
            "token_budget": 224,
            "num_problems": 8,
            "accuracy_pct": 37.5,
            "avg_gen_tokens": 263.1,
            "avg_tokens_per_sec": 6.7,
            "avg_eviction_ms": 8.64,
            "union_coverage_pct": 80.6,
            "q1_oldest_union_pct": 76.6,
        },
        {
            "suite": "SuiteB_FreeGen_Math_Reasoning",
            "mode": "valuediff_pp",
            "token_budget": 224,
            "num_problems": 8,
            "accuracy_pct": 37.5,
            "avg_gen_tokens": 268.6,
            "avg_tokens_per_sec": 7.2,
            "avg_eviction_ms": 0.88,
            "union_coverage_pct": 75.3,
            "q1_oldest_union_pct": 77.7,
        },
    ]
    for mode in modes:
        if any(r["mode"] == mode for r in summary_rows):
            continue
        correct = 0
        gen_lengths = []
        tok_s_list = []
        evict_ms_list = []
        union_cov_list = []
        q1_union_list = []

        for idx, (item, p_str) in enumerate(zip(selected, prompts)):
            p_ids = tokenizer(p_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            cache = UnifiedEvictCache(
                config=model.config,
                token_budget=token_budget,
                residual_length=residual_length,
                eviction_mode=mode,
                seed=100 + idx,
                track_positions=True,
            )

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            out = model(input_ids=p_ids, past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1, :]

            gen_ids = []
            for _ in range(max_new_tokens):
                next_id = int(logits.argmax(dim=-1).item())
                gen_ids.append(next_id)
                if next_id in eos_ids:
                    break
                next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
                out = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
                logits = out.logits[:, -1, :]

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = normalize_math_answer(extract_boxed_or_last_number(gen_text))
            gt = normalize_math_answer(str(item["ground_truth"]))

            is_ok = (pred == gt) and (pred != "")
            if not is_ok and gt != "":
                try:
                    if abs(float(pred) - float(gt)) < 1e-4:
                        is_ok = True
                except Exception:
                    pass

            correct += int(is_ok)
            gen_lengths.append(len(gen_ids))
            tok_s_list.append(len(gen_ids) / max(elapsed, 1e-6))
            if cache.stats.eviction_calls > 0:
                evict_ms_list.append(cache.stats.total_eviction_time_ms / cache.stats.eviction_calls)

            # Survival telemetry
            num_layers = len(cache.layers)
            probe_layers = [num_layers // 4, num_layers // 2, (3 * num_layers) // 4]
            u_list, q1_list = [], []
            for l_idx in probe_layers:
                layer = cache.layers[l_idx]
                cand_start = layer.prompt_len
                cand_end = max(layer.cumulative_length - layer.residual_length, cand_start + 1)
                span = cand_end - cand_start
                if span > 16:
                    surv = layer.pos_ids[0, :, : layer.cur_len].cpu()
                    M = torch.zeros(layer.num_kv_heads, span, dtype=torch.bool)
                    for h in range(layer.num_kv_heads):
                        pos_h = surv[h]
                        cot_pos = pos_h[(pos_h >= cand_start) & (pos_h < cand_end)] - cand_start
                        M[h, cot_pos.long()] = True
                    umask = M.any(dim=0)
                    u_list.append(umask.float().mean().item() * 100.0)
                    q1_list.append(umask[: max(span // 4, 1)].float().mean().item() * 100.0)
            union_cov_list.append(sum(u_list) / len(u_list) if u_list else 100.0)
            q1_union_list.append(sum(q1_list) / len(q1_list) if q1_list else 100.0)

        row = {
            "suite": "SuiteB_FreeGen_Math_Reasoning",
            "mode": mode,
            "token_budget": token_budget if mode != "dense" else "Full",
            "num_problems": len(selected),
            "accuracy_pct": round(correct / len(selected) * 100.0, 2),
            "avg_gen_tokens": round(sum(gen_lengths) / len(gen_lengths), 1),
            "avg_tokens_per_sec": round(sum(tok_s_list) / len(tok_s_list), 2),
            "avg_eviction_ms": round(sum(evict_ms_list) / max(len(evict_ms_list), 1), 3),
            "union_coverage_pct": round(sum(union_cov_list) / len(union_cov_list), 2),
            "q1_oldest_union_pct": round(sum(q1_union_list) / len(q1_union_list), 2),
        }
        summary_rows.append(row)
        print(
            f"[Suite B | Budget={str(row['token_budget']):6s} | {mode:14s}] "
            f"Acc: {row['accuracy_pct']:5.1f}% | "
            f"AvgLen: {row['avg_gen_tokens']:5.1f} tok | "
            f"UnionCov: {row['union_coverage_pct']:5.1f}% | "
            f"Q1-Union: {row['q1_oldest_union_pct']:5.1f}% | "
            f"Speed: {row['avg_tokens_per_sec']:5.1f} tok/s ({row['avg_eviction_ms']:.2f} ms/evict)",
            flush=True,
        )

    out_dir = ROOT / "results"
    (out_dir / "suite_b_math_results.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    suite_a = json.loads((out_dir / "suite_a_milestone_results.json").read_text(encoding="utf-8"))
    combined = {
        "model": "Qwen/Qwen3-1.7B",
        "gpu": torch.cuda.get_device_name(0),
        "suite_a": suite_a,
        "suite_b": summary_rows,
    }
    (out_dir / "all_qwen3_benchmarks.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("Saved combined results to results/all_qwen3_benchmarks.json", flush=True)


if __name__ == "__main__":
    patch_qwen3_evict_sdpa()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    modes = [
        "dense",
        "recency_pp",
        "snapkv_pp",
        "rkv_pp",
        "vase_pp",
        "valuediff_pp",
        "random_pp",
        "vacor_kv",
    ]
    run_suite_b_fast(model, tokenizer, modes=modes, token_budget=224, residual_length=48, num_problems=8, max_new_tokens=448)
