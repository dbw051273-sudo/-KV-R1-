"""
Aggregator and Statistical Summarizer for MATH-500 Evaluation.

Reads `results/math500_eval.jsonl`, performs robust LaTeX math normalization
on `pred` and `gt`, computes exact accuracy per mode across seeds with 95% Wilson
confidence intervals, per-eviction latency, and throughput, and outputs both
`results/math500_summary.json` and `results/math500_summary.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent


def robust_math_normalize(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    s = str(ans).strip()
    s = s.replace("$", "").replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]+)\}", r"\1", s)
    s = s.replace(r"\dfrac", r"\frac")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace(",", "")
    s = s.rstrip(".")
    return s.lower()


def is_math_equal(pred_raw: Optional[str], gt_raw: Optional[str]) -> bool:
    p = robust_math_normalize(pred_raw)
    g = robust_math_normalize(gt_raw)
    if p == "" or g == "":
        return False
    if p == g:
        return True
    try:
        if abs(float(p) - float(g)) < 1e-4:
            return True
    except Exception:
        pass
    return False


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    return (max(0.0, center - half_width) * 100.0, min(1.0, center + half_width) * 100.0)


def aggregate_math500(eval_file: Path) -> tuple[list[dict[str, Any]], str]:
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    lines = [json.loads(l) for l in eval_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"No records found in {eval_file}")

    by_mode = defaultdict(list)
    for r in lines:
        r["is_correct_robust"] = is_math_equal(r.get("pred"), r.get("gt"))
        by_mode[r["mode"]].append(r)

    modes_order = ["dense", "valuediff_pp", "snapkv_pp", "random_pp", "vacor_kv"]
    active_modes = [m for m in modes_order if m in by_mode] + [m for m in by_mode if m not in modes_order]

    summary_list = []
    md_lines = [
        "# MATH-500 Subset Evaluation Summary",
        "",
        f"- **Total Records**: {len(lines)}",
        f"- **Evaluated Problems**: {len(set(r['problem_id'] for r in lines))}",
        "",
        "| Mode | Budget | Evaluated Runs | Accuracy (Mean ± Std %) | 95% Wilson CI (%) | Throughput (tok/s) | Eviction Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for mode in active_modes:
        recs = by_mode[mode]
        seeds = sorted(set(r["seed"] for r in recs))
        budget = recs[0].get("budget", 256) if mode != "dense" else "Full"

        seed_accs = []
        for s in seeds:
            s_recs = [r for r in recs if r["seed"] == s]
            acc = sum(1 for r in s_recs if r["is_correct_robust"]) / max(len(s_recs), 1) * 100.0
            seed_accs.append(acc)

        mean_acc = sum(seed_accs) / len(seed_accs)
        std_acc = (
            math.sqrt(sum((x - mean_acc) ** 2 for x in seed_accs) / (len(seed_accs) - 1))
            if len(seed_accs) > 1
            else 0.0
        )
        total_correct = sum(1 for r in recs if r["is_correct_robust"])
        w_low, w_high = wilson_interval(total_correct, len(recs))
        avg_speed = sum(r.get("tokens_per_sec", 0.0) for r in recs) / max(len(recs), 1)
        avg_evict_ms = sum(r.get("avg_eviction_ms", 0.0) for r in recs) / max(len(recs), 1)

        std_str = f"± {std_acc:4.1f}" if len(seeds) > 1 and std_acc > 0.0 else "      "
        md_lines.append(
            f"| `{mode}` | {str(budget):6s} | {len(recs):3d} (seeds={len(seeds)}) | "
            f"{mean_acc:5.1f}% {std_str} | [{w_low:5.1f}%, {w_high:5.1f}%] | "
            f"{avg_speed:5.1f} | {avg_evict_ms:5.2f} |"
        )

        summary_list.append({
            "mode": mode,
            "budget": budget,
            "n_runs": len(recs),
            "n_seeds": len(seeds),
            "accuracy_mean": round(mean_acc, 2),
            "accuracy_std": round(std_acc, 2),
            "wilson_95": [round(w_low, 2), round(w_high, 2)],
            "avg_tokens_per_sec": round(avg_speed, 2),
            "avg_eviction_ms": round(avg_evict_ms, 3),
        })

    return summary_list, "\n".join(md_lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default="results/math500_eval.jsonl")
    args = parser.parse_args()

    eval_path = ROOT / args.eval_file
    summary_list, md_report = aggregate_math500(eval_path)

    out_json = ROOT / "results" / "math500_summary.json"
    out_md = ROOT / "results" / "math500_summary.md"

    out_json.write_text(json.dumps(summary_list, indent=2), encoding="utf-8")
    out_md.write_text(md_report, encoding="utf-8")

    print(f"Aggregated {len(summary_list)} modes.")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}\n")
    print(md_report)


if __name__ == "__main__":
    main()
