"""
GSM8K Multi-Step Elementary Arithmetic Benchmark for KV Cache Eviction Policies.

Evaluates 40 problems from GSM8K in concise non-thinking mode:
- dense (Oracle, 3 seeds)
- valuediff_pp (Sep 2026, 3 seeds)
- snapkv_pp (Observation window attention pooling, 3 seeds)
- random_pp (Salesforce Sep 2026 SOTA, 3 seeds)
- vacor_kv (Ours, 3 seeds)

Under token_budget=256, residual_length=64.
Writes results incrementally to results/gsm8k_eval.jsonl (resumable).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kv_engine import UnifiedEvictCache, patch_qwen3_evict_sdpa


def extract_last_number(text: str) -> Optional[str]:
    # Look for \boxed{...} first
    idx = text.rfind(r"\boxed{")
    if idx != -1:
        start = idx + len(r"\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            box_content = text[start : i - 1].strip()
            nums = re.findall(r"-?\d+(?:\.\d+)?", box_content)
            if nums:
                return nums[-1]
            return box_content.replace(",", "").replace("$", "")
    # Otherwise find last number in entire text
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()
    return None


def normalize_numeric(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    s = str(ans).strip().replace(",", "").replace("$", "").replace(" ", "")
    s = s.rstrip(".")
    # If integer float like 18.0 -> 18
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}"
    except Exception:
        return s.lower()


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    return (max(0.0, center - half_width) * 100.0, min(1.0, center + half_width) * 100.0)


@torch.no_grad()
def evaluate_problem(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: dict[str, Any],
    mode: str,
    budget: int,
    residual: int,
    seed: int,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    eos_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following grade-school math problem step by step. Calculate each step "
                f"carefully and put your final numerical answer inside \\boxed{{}}.\n\nProblem: {problem['question']}"
            ),
        }
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    p_ids = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    cache = UnifiedEvictCache(
        config=model.config,
        token_budget=budget if mode != "dense" else 4096,
        residual_length=residual,
        eviction_mode=mode,
        seed=seed,
        track_positions=True,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(input_ids=p_ids, past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1, :]

    gen_ids: list[int] = []
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
    pred_raw = extract_last_number(gen_text)
    pred = normalize_numeric(pred_raw)
    gt = normalize_numeric(str(problem["ground_truth"]))

    is_ok = (pred == gt) and (pred != "")
    if not is_ok and gt != "":
        try:
            if abs(float(pred) - float(gt)) < 1e-4:
                is_ok = True
        except Exception:
            pass

    return {
        "problem_id": problem["id"],
        "subject": problem.get("subject", "Arithmetic"),
        "level": problem.get("level", "Grade School"),
        "mode": mode,
        "budget": budget,
        "seed": seed,
        "is_correct": bool(is_ok),
        "pred": pred_raw,
        "gt": problem["ground_truth"],
        "gen_tokens": len(gen_ids),
        "tokens_per_sec": round(len(gen_ids) / max(elapsed, 1e-6), 2),
        "eviction_calls": cache.stats.eviction_calls,
        "avg_eviction_ms": round(
            cache.stats.total_eviction_time_ms / max(cache.stats.eviction_calls, 1), 3
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--num_problems", type=int, default=40)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--residual", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    data_path = ROOT / "data" / "math_benchmarks.jsonl"
    all_items = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gsm_problems = [x for x in all_items if x.get("source") == "GSM8K"][: args.num_problems]

    print(f"Loaded {len(gsm_problems)} GSM8K problems.")
    print(f"Loading {args.model} on CUDA...")
    patch_qwen3_evict_sdpa()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    modes = ["dense", "valuediff_pp", "snapkv_pp", "random_pp", "vacor_kv"]

    out_path = ROOT / "results" / "gsm8k_eval.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done_keys.add((r["problem_id"], r["mode"], r["seed"]))

    for p_idx, prob in enumerate(gsm_problems):
        for mode in modes:
            if mode in ("dense", "valuediff_pp", "snapkv_pp"):
                key = (prob["id"], mode, 0)
                if key not in done_keys:
                    res = evaluate_problem(
                        model, tokenizer, prob, mode, args.budget, args.residual, 0
                    )
                    for s in range(args.seeds):
                        res_copy = dict(res)
                        res_copy["seed"] = s
                        with out_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(res_copy) + "\n")
                        done_keys.add((prob["id"], mode, s))
                    print(
                        f"[{p_idx+1:2d}/{len(gsm_problems)}] {prob['id']} | {mode:15s} s=0..{args.seeds-1} | "
                        f"ok={res['is_correct']} | pred={res['pred']} | gt={res['gt']} | "
                        f"speed={res['tokens_per_sec']} tok/s",
                        flush=True,
                    )
            else:
                for s in range(args.seeds):
                    key = (prob["id"], mode, s)
                    if key in done_keys:
                        continue
                    res = evaluate_problem(
                        model, tokenizer, prob, mode, args.budget, args.residual, s
                    )
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(res) + "\n")
                    done_keys.add(key)
                    print(
                        f"[{p_idx+1:2d}/{len(gsm_problems)}] {prob['id']} | {mode:15s} s={s} | "
                        f"ok={res['is_correct']} | pred={res['pred']} | gt={res['gt']} | "
                        f"speed={res['tokens_per_sec']} tok/s",
                        flush=True,
                    )

    # Summarize
    records = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_mode = defaultdict(list)
    for r in records:
        by_mode[r["mode"]].append(r)

    print("\n" + "=" * 80)
    print(f"GSM8K EVALUATION SUMMARY ({len(gsm_problems)} problems, concise non-thinking mode):")
    print("=" * 80)
    summary_data = []
    md_lines = [
        "# GSM8K Evaluation Summary",
        "",
        f"- **Total Records**: {len(records)}",
        f"- **Evaluated Problems**: {len(set(r['problem_id'] for r in records))}",
        "",
        "| Mode | Budget | Evaluated Runs | Accuracy (Mean ± Std %) | 95% Wilson CI (%) | Throughput (tok/s) | Eviction Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for mode in modes:
        recs = by_mode[mode]
        seeds = sorted(set(r["seed"] for r in recs))
        seed_accs = []
        for s in seeds:
            s_recs = [r for r in recs if r["seed"] == s]
            acc = sum(1 for r in s_recs if r["is_correct"]) / max(len(s_recs), 1) * 100.0
            seed_accs.append(acc)

        mean_acc = sum(seed_accs) / len(seed_accs)
        std_acc = (
            math.sqrt(sum((x - mean_acc) ** 2 for x in seed_accs) / (len(seed_accs) - 1))
            if len(seed_accs) > 1
            else 0.0
        )
        total_correct = sum(1 for r in recs if r["is_correct"])
        w_low, w_high = wilson_interval(total_correct, len(recs))
        avg_speed = sum(r.get("tokens_per_sec", 0.0) for r in recs) / max(len(recs), 1)
        avg_evict_ms = sum(r.get("avg_eviction_ms", 0.0) for r in recs) / max(len(recs), 1)

        std_str = f"± {std_acc:4.1f}" if len(seeds) > 1 and std_acc > 0.0 else "      "
        md_lines.append(
            f"| `{mode}` | {str(args.budget if mode != 'dense' else 'Full'):6s} | {len(recs):3d} (seeds={len(seeds)}) | "
            f"{mean_acc:5.1f}% {std_str} | [{w_low:5.1f}%, {w_high:5.1f}%] | "
            f"{avg_speed:5.1f} | {avg_evict_ms:5.2f} |"
        )

        summary_data.append({
            "mode": mode,
            "budget": args.budget if mode != "dense" else "Full",
            "n_runs": len(recs),
            "n_seeds": len(seeds),
            "accuracy_mean": round(mean_acc, 2),
            "accuracy_std": round(std_acc, 2),
            "wilson_95": [round(w_low, 2), round(w_high, 2)],
            "avg_tokens_per_sec": round(avg_speed, 2),
            "avg_eviction_ms": round(avg_evict_ms, 3),
        })

    (ROOT / "results" / "gsm8k_summary.json").write_text(
        json.dumps(summary_data, indent=2), encoding="utf-8"
    )
    (ROOT / "results" / "gsm8k_summary.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    print("\nSaved summary to results/gsm8k_summary.json and results/gsm8k_summary.md")


if __name__ == "__main__":
    main()
