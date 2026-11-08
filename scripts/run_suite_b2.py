"""
Controlled Free-Generation Multi-Step Math Benchmark on `Qwen/Qwen3-1.7B` (`Suite B2`):
Tests 10 multi-step arithmetic & algebraic word problems where the problem statement is concise
(~45 tokens) and the model generates ~280-360 tokens of step-by-step equations (`Step 1` .. `Step 8`)
before combining Step 1 and Step 7 at the end.
Under `token_budget = 160` (`residual_length = 32`, ~2.2x compression of free generation), early
generated steps (`Step 1` and `Step 2`) undergo 5-7 live eviction rounds during free generation!
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_benchmarks import extract_boxed_or_last_number, normalize_math_answer
from src.kv_engine import UnifiedEvictCache, patch_qwen3_evict_sdpa


def build_free_gen_multistep_math_problems() -> list[dict]:
    problems = []
    configs = [
        (14, 3, 25, 4),
        (18, 2, 31, 5),
        (12, 4, 19, 3),
        (22, 3, 15, 6),
        (16, 5, 23, 2),
        (21, 2, 27, 4),
        (15, 4, 17, 5),
        (19, 3, 29, 2),
    ]
    for idx, (a0, d1, b0, d2) in enumerate(configs):
        # The model is asked to compute S1 = a0 * d1, then write out 6 intermediate verification
        # table rows for t=1..6 (generating ~250 tokens of CoT), and finally compute Final = S1 + b0 * d2.
        s1 = a0 * d1
        s2 = b0 * d2
        ans = s1 + s2
        q = (
            f"Follow these exact instructions step by step:\n"
            f"1. First, compute `Alpha = {a0} * {d1}` and state `Alpha = <value>`.\n"
            f"2. Next, for each integer k from 1 to 6, write a full sentence computing "
            f"`Check_k = k^2 + 10*k + {b0}` and explaining each arithmetic step in detail.\n"
            f"3. After finishing k=6, compute `Beta = {b0} * {d2}`.\n"
            f"4. Finally, recall your exact value of `Alpha` from Step 1, add `Beta`, and output "
            f"`Alpha + Beta` inside \\boxed{{}}."
        )
        problems.append({"id": f"freegen_ms_{idx}", "question": q, "alpha": s1, "beta": s2, "ground_truth": str(ans)})
    return problems


@torch.no_grad()
def main():
    patch_qwen3_evict_sdpa()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    problems = build_free_gen_multistep_math_problems()
    device = next(model.parameters()).device
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

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

    token_budget = 144
    residual_length = 32
    max_new_tokens = 384

    rows = []
    for mode in modes:
        correct = 0
        alpha_recalled = 0
        gen_lengths = []
        tok_s_list = []
        evict_ms_list = []
        union_cov_list = []
        q1_union_list = []

        for idx, item in enumerate(problems):
            messages = [{"role": "user", "content": item["question"]}]
            p_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            p_ids = tokenizer(p_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

            cache = UnifiedEvictCache(
                config=model.config,
                token_budget=token_budget,
                residual_length=residual_length,
                eviction_mode=mode,
                seed=200 + idx,
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
            gt = normalize_math_answer(item["ground_truth"])
            is_ok = (pred == gt)

            # Also check if Alpha was recalled in the final paragraph (last 120 chars)
            tail_text = gen_text[-140:]
            alpha_ok = str(item["alpha"]) in tail_text or is_ok

            correct += int(is_ok)
            alpha_recalled += int(alpha_ok)
            gen_lengths.append(len(gen_ids))
            tok_s_list.append(len(gen_ids) / max(elapsed, 1e-6))
            if cache.stats.eviction_calls > 0:
                evict_ms_list.append(cache.stats.total_eviction_time_ms / cache.stats.eviction_calls)

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
            "suite": "SuiteB_FreeGen_MultiStep_Math",
            "mode": mode,
            "token_budget": token_budget if mode != "dense" else "Full",
            "num_problems": len(problems),
            "accuracy_pct": round(correct / len(problems) * 100.0, 2),
            "alpha_recall_pct": round(alpha_recalled / len(problems) * 100.0, 2),
            "avg_gen_tokens": round(sum(gen_lengths) / len(gen_lengths), 1),
            "avg_tokens_per_sec": round(sum(tok_s_list) / len(tok_s_list), 2),
            "avg_eviction_ms": round(sum(evict_ms_list) / max(len(evict_ms_list), 1), 3),
            "union_coverage_pct": round(sum(union_cov_list) / len(union_cov_list), 2),
            "q1_oldest_union_pct": round(sum(q1_union_list) / len(q1_union_list), 2),
        }
        rows.append(row)
        print(
            f"[Suite B | Budget={str(row['token_budget']):6s} | {mode:14s}] "
            f"Acc: {row['accuracy_pct']:5.1f}% | "
            f"Step1-Recall: {row['alpha_recall_pct']:5.1f}% | "
            f"AvgLen: {row['avg_gen_tokens']:5.1f} tok | "
            f"UnionCov: {row['union_coverage_pct']:5.1f}% | "
            f"Q1-Union: {row['q1_oldest_union_pct']:5.1f}% | "
            f"Speed: {row['avg_tokens_per_sec']:5.1f} tok/s ({row['avg_eviction_ms']:.2f} ms/evict)",
            flush=True,
        )

    out_dir = ROOT / "results"
    (out_dir / "suite_b_math_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    suite_a = json.loads((out_dir / "suite_a_milestone_results.json").read_text(encoding="utf-8"))
    combined = {
        "model": "Qwen/Qwen3-1.7B",
        "gpu": torch.cuda.get_device_name(0),
        "suite_a": suite_a,
        "suite_b": rows,
    }
    (out_dir / "all_qwen3_benchmarks.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print("Updated results/all_qwen3_benchmarks.json", flush=True)


if __name__ == "__main__":
    main()
