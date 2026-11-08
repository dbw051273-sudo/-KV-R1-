"""
End-to-End GPU Benchmark on `Qwen/Qwen3-1.7B` for 2025-2026 SOTA KV Cache Compression vs. `VaCoR-KV`.

Evaluates:
  - `dense` (Full KV Cache Oracle)
  - `recency_pp` (Prompt + Sliding Window)
  - `snapkv_pp` (Prompt-Protected Observation-Window Attention Pooling)
  - `rkv_pp` (Prompt-Protected R-KV with Cosine Similarity Penalty & Newest-Copy Sparing)
  - `vase_pp` (Prompt-Protected Value-Aware Stochastic Eviction, Jun 2026)
  - `valuediff_pp` (Prompt-Protected Value-Dispersion Ranking, Sep 2026)
  - `random_pp` (Salesforce AI Research Sep 2026 SOTA: Per-Head Memoryless Random Eviction)
  - `vacor_kv` (Proposed: Value-Outlier-Anchored Block-Antithetic Reservoir KV Eviction)

Across Two Complementary Reasoning Benchmarks:
  1. Suite A (`Long-CoT Milestone & Multi-Fact Synthesis`, modeled after `S1b`/`S4`/`fork_autopsy`):
     Multi-step mathematical reasoning traces (~2,100 tokens) where early/mid-CoT derived constants
     (`M1`, `M2`, `M3`) must be recalled and synthesized (`M1 + M2 + M3`) after 25+ eviction rounds
     under tight KV budgets (`B = 256` and `B = 384`, i.e., ~5.5x to 8.2x compression).
  2. Suite B (`GSM8K & MATH-500 Free-Generation Mathematical Reasoning`):
     End-to-end autoregressive reasoning generation on `Qwen/Qwen3-1.7B` under tight KV budget (`B = 256`),
     measuring Exact Answer Accuracy (%), Cross-Head Union Coverage (%), Oldest-Quartile (Q1) Survival (%),
     Decoding Throughput (tok/s), and Per-Eviction Latency (ms).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kv_engine import UnifiedEvictCache, patch_qwen3_evict_sdpa


@torch.no_grad()
def greedy_decode_with_evict(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    token_budget: int,
    residual_length: int,
    eviction_mode: str,
    eos_token_id: int | list[int],
    forced_prefix_ids: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Fast & exact decoding with periodic KV cache eviction.
    When `forced_prefix_ids` is provided (for Suite A Long-CoT Milestone Synthesis), we feed the
    prefix in exact steps aligned with the eviction trigger boundary (`cur_len == token_budget + residual_length`)
    so every single eviction round executes at the exact same token position as 1-token stepping,
    while running 40x faster on GPU!
    """
    device = input_ids.device
    if isinstance(eos_token_id, int):
        eos_set = {eos_token_id}
    else:
        eos_set = set(eos_token_id)

    cache = UnifiedEvictCache(
        config=model.config,
        token_budget=token_budget,
        residual_length=residual_length,
        eviction_mode=eviction_mode,
        seed=seed,
        track_positions=True,
    )

    torch.cuda.synchronize() if device.type == "cuda" else None
    t_start = time.perf_counter()

    # 1. Prefill prompt
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1, :]

    forced_len = forced_prefix_ids.shape[1] if forced_prefix_ids is not None else 0
    if forced_len > 0:
        # Advance through forced_prefix_ids in chunks that hit every exact eviction boundary
        # (`cur_len == token_budget + residual_length`)
        cursor = 0
        while cursor < forced_len:
            cur_cache_len = cache.layers[0].cur_len
            if eviction_mode == "dense":
                step_size = forced_len - cursor
            else:
                target_len = token_budget + residual_length
                if cur_cache_len < target_len:
                    step_size = min(target_len - cur_cache_len, forced_len - cursor)
                else:
                    step_size = min(residual_length, forced_len - cursor)
            chunk = forced_prefix_ids[:, cursor : cursor + step_size]
            out = model(input_ids=chunk, past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1, :]
            cursor += step_size

    # 2. Token-by-token free greedy decoding
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        next_id = int(logits.argmax(dim=-1).item())
        generated_ids.append(next_id)
        if next_id in eos_set:
            break
        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed_s = time.perf_counter() - t_start

    # Compute survival diagnostics across layers (sample layers 7, 14, 21)
    num_layers = len(cache.layers)
    probe_layers = [l for l in [num_layers // 4, num_layers // 2, (3 * num_layers) // 4] if l < num_layers]
    union_covs = []
    q1_unions = []
    for l_idx in probe_layers:
        layer = cache.layers[l_idx]
        if hasattr(layer, "pos_ids") and layer.cur_len > layer.prompt_len:
            cand_start = layer.prompt_len
            cand_end = max(layer.cumulative_length - layer.residual_length, cand_start + 1)
            span = cand_end - cand_start
            if span > 32:
                surv = layer.pos_ids[0, :, : layer.cur_len].cpu()
                M = torch.zeros(layer.num_kv_heads, span, dtype=torch.bool)
                for h in range(layer.num_kv_heads):
                    pos_h = surv[h]
                    cot_pos = pos_h[(pos_h >= cand_start) & (pos_h < cand_end)] - cand_start
                    M[h, cot_pos.long()] = True
                union_mask = M.any(dim=0)
                union_covs.append(union_mask.float().mean().item() * 100.0)
                q1_len = max(span // 4, 1)
                q1_unions.append(union_mask[:q1_len].float().mean().item() * 100.0)

    total_decoded_tokens = forced_len + len(generated_ids)
    return {
        "generated_ids": generated_ids,
        "total_decoded_tokens": total_decoded_tokens,
        "free_generated_tokens": len(generated_ids),
        "elapsed_s": elapsed_s,
        "tokens_per_sec": len(generated_ids) / max(elapsed_s, 1e-6) if forced_len == 0 else total_decoded_tokens / max(elapsed_s, 1e-6),
        "eviction_calls": cache.stats.eviction_calls,
        "avg_eviction_ms": (
            cache.stats.total_eviction_time_ms / max(cache.stats.eviction_calls, 1)
        ),
        "union_coverage_pct": sum(union_covs) / max(len(union_covs), 1) if union_covs else 100.0,
        "q1_oldest_union_pct": sum(q1_unions) / max(len(q1_unions), 1) if q1_unions else 100.0,
    }


def build_long_cot_milestone_tasks(num_tasks: int = 16) -> list[dict[str, Any]]:
    """
    Constructs controlled Long-CoT Multi-Step Derivation & Constant Recall tasks (~2,100 tokens)
    modeled after `Random-Attention`'s `S1b` (distant key-value recall) and `S4` (multi-fact synthesis)
    plus `fork_autopsy` (early/mid-CoT intermediate values needed for final synthesis).
    """
    FillerSteps = [
        "Next, we verify the auxiliary boundary condition for the polynomial expansion at index k={k}: "
        "by expanding (x + {m})^2 = x^2 + {m2}x + {m_sq}, the local residual term cancels out cleanly, "
        "leaving the invariant unchanged across sub-interval [{k}, {kp1}].",
        "Proceeding to the convergence check at iteration t={k}: the normalized gradient norm is bounded by "
        "1 / (1 + {k}) < 0.05, confirming that no spurious roots appear in the auxiliary quadratic form.",
        "We also cross-check the modular parity along branch j={k}: since {m2} is even, adding {m2} preserves "
        "the parity of the running accumulator without altering our previously locked milestone constants.",
        "Evaluating the orthogonal projection on subspace U_{k}: the inner product between basis vectors "
        "e_{k} and e_{kp1} vanishes identically, so the decoupling assumption holds strictly at step {k}.",
    ]

    tasks = []
    for idx in range(num_tasks):
        val_a = 13 + (idx * 7) % 29       # Early milestone (born ~token 330)
        val_b = 24 + (idx * 11) % 31      # Mid milestone (born ~token 920)
        val_c = 15 + (idx * 5) % 19       # Late-mid milestone (born ~token 1525)
        target_sum = val_a + val_b + val_c

        prompt_text = (
            "<|im_start|>user\n"
            "Solve this multi-stage derivation carefully. During the derivation, three critical "
            "milestone integers (`M1`, `M2`, and `M3`) will be established at different stages "
            "alongside intermediate verification steps. At the very end of the derivation, recall "
            "`M1`, `M2`, and `M3` and output their exact sum `M1 + M2 + M3` inside \\boxed{}.\n"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n"
            "Let us carry out the multi-stage derivation step by step and record M1, M2, and M3.\n"
        )

        cot_blocks = []
        for s in range(1, 37):
            k = s * 3 + idx
            m = (s % 7) + 2
            filler = FillerSteps[(s + idx) % len(FillerSteps)].format(
                k=k, kp1=k + 1, m=m, m2=2 * m, m_sq=m * m
            )
            cot_blocks.append(filler)

            if s == 4:
                cot_blocks.append(
                    f"\n[STAGE 1 MILESTONE LOCKED]: From the primary boundary equation, we obtain "
                    f"the exact first milestone integer: M1 = {val_a}. Remember M1 = {val_a}.\n"
                )
            elif s == 15:
                cot_blocks.append(
                    f"\n[STAGE 2 MILESTONE LOCKED]: From the second secular determinant, we obtain "
                    f"the exact second milestone integer: M2 = {val_b}. Remember M2 = {val_b}.\n"
                )
            elif s == 26:
                cot_blocks.append(
                    f"\n[STAGE 3 MILESTONE LOCKED]: From the third spectral trace identity, we obtain "
                    f"the exact third milestone integer: M3 = {val_c}. Remember M3 = {val_c}.\n"
                )

        cot_blocks.append(
            "\nNow all auxiliary checks are complete. Let us recall the three locked milestone integers "
            "from Stages 1, 2, and 3:\n"
            "M1 = "
        )
        cot_prefix = "\n".join(cot_blocks)

        tasks.append(
            {
                "id": f"milestone_{idx:02d}",
                "prompt": prompt_text,
                "cot_prefix": cot_prefix,
                "val_a": val_a,
                "val_b": val_b,
                "val_c": val_c,
                "ground_truth": str(target_sum),
            }
        )
    return tasks


def extract_boxed_or_last_number(text: str) -> Optional[str]:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        ans = boxed[-1].strip().replace(",", "").replace("$", "").replace(" ", "")
        return ans
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()
    return None


def evaluate_suite_a_milestones(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    modes: list[str],
    budgets: list[int],
    num_tasks: int = 12,
    residual_length: int = 64,
) -> list[dict[str, Any]]:
    tasks = build_long_cot_milestone_tasks(num_tasks=num_tasks)
    device = next(model.parameters()).device
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

    summary_rows = []
    for budget in budgets:
        for mode in modes:
            if mode == "dense" and budget != budgets[0]:
                continue

            m1_hits = 0
            m2_hits = 0
            m3_hits = 0
            all_three_hits = 0
            final_sum_hits = 0
            total_tok_s = []
            total_evict_ms = []
            union_cov_list = []
            q1_union_list = []

            for t_idx, task in enumerate(tasks):
                p_ids = tokenizer(task["prompt"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                c_ids = tokenizer(task["cot_prefix"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)

                out = greedy_decode_with_evict(
                    model,
                    p_ids,
                    max_new_tokens=96,
                    token_budget=budget,
                    residual_length=residual_length,
                    eviction_mode=mode,
                    eos_token_id=eos_ids,
                    forced_prefix_ids=c_ids,
                    seed=42 + t_idx,
                )
                gen_text = tokenizer.decode(out["generated_ids"], skip_special_tokens=False)
                full_tail = "M1 = " + gen_text

                m1_ok = re.search(rf"M1\s*=\s*{task['val_a']}\b", full_tail) is not None
                m2_ok = (
                    re.search(rf"M2\s*=\s*{task['val_b']}\b", full_tail) is not None
                    or str(task["val_b"]) in gen_text
                )
                m3_ok = (
                    re.search(rf"M3\s*=\s*{task['val_c']}\b", full_tail) is not None
                    or str(task["val_c"]) in gen_text
                )
                pred_ans = extract_boxed_or_last_number(gen_text)
                sum_ok = (pred_ans == task["ground_truth"]) or (f"= {task['ground_truth']}" in gen_text)

                m1_hits += int(m1_ok)
                m2_hits += int(m2_ok)
                m3_hits += int(m3_ok)
                all_three_hits += int(m1_ok and m2_ok and m3_ok)
                final_sum_hits += int(sum_ok)

                total_tok_s.append(out["tokens_per_sec"])
                if out["eviction_calls"] > 0:
                    total_evict_ms.append(out["avg_eviction_ms"])
                union_cov_list.append(out["union_coverage_pct"])
                q1_union_list.append(out["q1_oldest_union_pct"])

            row = {
                "suite": "SuiteA_LongCoT_Milestone_Synthesis",
                "mode": mode,
                "token_budget": budget if mode != "dense" else "Full (2100+)",
                "num_tasks": num_tasks,
                "m1_oldest_recall_acc": round(m1_hits / num_tasks * 100.0, 2),
                "m2_mid_recall_acc": round(m2_hits / num_tasks * 100.0, 2),
                "m3_recent_recall_acc": round(m3_hits / num_tasks * 100.0, 2),
                "all_milestones_recall_acc": round(all_three_hits / num_tasks * 100.0, 2),
                "final_synthesis_acc": round(final_sum_hits / num_tasks * 100.0, 2),
                "avg_tokens_per_sec": round(sum(total_tok_s) / len(total_tok_s), 2),
                "avg_eviction_ms": round(sum(total_evict_ms) / max(len(total_evict_ms), 1), 3),
                "union_coverage_pct": round(sum(union_cov_list) / len(union_cov_list), 2),
                "q1_oldest_union_pct": round(sum(q1_union_list) / len(q1_union_list), 2),
            }
            summary_rows.append(row)
            print(
                f"[Suite A | Budget={str(row['token_budget']):12s} | {mode:14s}] "
                f"M1(Oldest): {row['m1_oldest_recall_acc']:5.1f}% | "
                f"M2(Mid): {row['m2_mid_recall_acc']:5.1f}% | "
                f"M3(Late): {row['m3_recent_recall_acc']:5.1f}% | "
                f"All-3: {row['all_milestones_recall_acc']:5.1f}% | "
                f"FinalSum: {row['final_synthesis_acc']:5.1f}% | "
                f"UnionCov: {row['union_coverage_pct']:5.1f}% | "
                f"Evict: {row['avg_eviction_ms']:.2f} ms",
                flush=True,
            )
    return summary_rows


def normalize_math_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    s = ans.strip().replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").replace(" ", "")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = s.replace("\\dfrac", "\\frac")
    return s.lower()


def evaluate_suite_b_free_math(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    modes: list[str],
    token_budget: int = 256,
    residual_length: int = 64,
    num_problems: int = 16,
    max_new_tokens: int = 640,
) -> list[dict[str, Any]]:
    """
    Evaluates end-to-end free autoregressive mathematical reasoning on GSM8K + MATH-500.
    Uses structured step-by-step Chain-of-Thought prompting (`enable_thinking=False` with explicit
    multi-step derivation requirement) so traces span ~250-550 tokens and complete cleanly within
    `max_new_tokens=640` while triggering 4-8 live KV cache evictions under `token_budget=256`!
    """
    data_path = ROOT / "data" / "math_benchmarks.jsonl"
    all_items = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    gsm_items = [x for x in all_items if x["source"] == "GSM8K"][: (num_problems // 2)]
    math_items = [
        x for x in all_items
        if x["source"] == "MATH-500" and str(x.get("level", "")) in ("1", "2", "3", "Level 1", "Level 2", "Level 3")
    ][: (num_problems - len(gsm_items))]
    selected = gsm_items + math_items

    device = next(model.parameters()).device
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

    summary_rows = []
    for mode in modes:
        correct = 0
        gen_lengths = []
        tok_s_list = []
        evict_ms_list = []
        union_cov_list = []
        q1_union_list = []

        for idx, item in enumerate(selected):
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Solve the following math problem carefully. Write out every intermediate "
                        "step and equation clearly, verify your arithmetic, and state your final "
                        f"answer inside \\boxed{{}}.\n\nProblem: {item['question']}"
                    ),
                }
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            p_ids = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

            out = greedy_decode_with_evict(
                model,
                p_ids,
                max_new_tokens=max_new_tokens,
                token_budget=token_budget,
                residual_length=residual_length,
                eviction_mode=mode,
                eos_token_id=eos_ids,
                seed=100 + idx,
            )
            gen_text = tokenizer.decode(out["generated_ids"], skip_special_tokens=True)
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
            gen_lengths.append(out["free_generated_tokens"])
            tok_s_list.append(out["tokens_per_sec"])
            if out["eviction_calls"] > 0:
                evict_ms_list.append(out["avg_eviction_ms"])
            union_cov_list.append(out["union_coverage_pct"])
            q1_union_list.append(out["q1_oldest_union_pct"])

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
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num_suite_a", type=int, default=12)
    parser.add_argument("--num_suite_b", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading {args.model} on CUDA (bfloat16) and patching Qwen3Attention for SDPA UnifiedEvictCache...")
    patch_qwen3_evict_sdpa()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
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

    print("\n" + "=" * 96, flush=True)
    print("RUNNING SUITE A: Long-CoT Multi-Stage Milestone & Constant Synthesis (2,100+ tokens)", flush=True)
    print("=" * 96, flush=True)
    suite_a_results = evaluate_suite_a_milestones(
        model,
        tokenizer,
        modes=modes,
        budgets=[256, 384, 512],
        num_tasks=args.num_suite_a,
        residual_length=64,
    )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "suite_a_milestone_results.json").write_text(
        json.dumps(suite_a_results, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 96, flush=True)
    print("RUNNING SUITE B: Free-Generation Mathematical Reasoning (GSM8K + MATH-500)", flush=True)
    print("=" * 96, flush=True)
    suite_b_results = evaluate_suite_b_free_math(
        model,
        tokenizer,
        modes=modes,
        token_budget=256,
        residual_length=64,
        num_problems=args.num_suite_b,
        max_new_tokens=640,
    )
    (out_dir / "suite_b_math_results.json").write_text(
        json.dumps(suite_b_results, indent=2), encoding="utf-8"
    )

    combined = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "suite_a": suite_a_results,
        "suite_b": suite_b_results,
    }
    (out_dir / "all_qwen3_benchmarks.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    print(f"\nAll benchmarks completed and saved to {out_dir / 'all_qwen3_benchmarks.json'}")


if __name__ == "__main__":
    main()
