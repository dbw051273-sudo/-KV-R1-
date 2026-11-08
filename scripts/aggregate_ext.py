"""
Aggregate Extended Suite A results from results/suite_a_ext_<tag>.jsonl.
Generates summary tables per budget and paired comparisons of vacor_kv vs valuediff_pp.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

ORDER = [
    "dense",
    "recency_pp",
    "snapkv_pp",
    "rkv_pp",
    "valuediff_pp",
    "valuediff_smooth_pp",
    "vase_pp",
    "random_pp",
    "vacor_kv",
    "vacor_pin",
    "vacor_antithetic_only",
]
METRICS = ["m1", "m2", "m3", "all3", "final"]


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Computes 95% Wilson score interval for binomial proportion k/n.
    Returns (lower, upper) in [0.0, 1.0].
    """
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 95% two-sided normal quantile
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return (lower, upper)


def compute_mode_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Given all records for a specific (mode, budget):
    - Computes mean +/- std across seeds (per-seed average over tasks, then mean/std across seeds)
    - Computes 95% Wilson interval for all3 pooled over all runs
    """
    tasks = sorted(set(r["task"] for r in records))
    seeds = sorted(set(r["seed"] for r in records))
    n_tasks = len(tasks)
    n_seeds = len(seeds)
    n_runs = len(records)

    # Group by seed: seed -> list of records
    seed_records = defaultdict(list)
    for r in records:
        seed_records[r["seed"]].append(r)

    stats: dict[str, Any] = {
        "n_tasks": n_tasks,
        "n_seeds": n_seeds,
        "n_runs": n_runs,
    }

    for m in METRICS:
        seed_rates = []
        for s in seeds:
            s_recs = seed_records[s]
            rate = (sum(1 for r in s_recs if r[m]) / len(s_recs)) * 100.0
            seed_rates.append(rate)

        mean_rate = sum(seed_rates) / len(seed_rates)
        if len(seed_rates) > 1:
            variance = sum((x - mean_rate) ** 2 for x in seed_rates) / (len(seed_rates) - 1)
            std_rate = math.sqrt(variance)
        else:
            std_rate = None

        stats[f"{m}_mean"] = mean_rate
        stats[f"{m}_std"] = std_rate

    # Wilson interval for all3 pooled over all runs
    k_all3 = sum(1 for r in records if r["all3"])
    wilson_low, wilson_high = wilson_score_interval(k_all3, n_runs)
    stats["all3_wilson_low"] = wilson_low * 100.0
    stats["all3_wilson_high"] = wilson_high * 100.0

    return stats


def format_mean_std(mean: float, std: float | None) -> str:
    if std is None:
        return f"{mean:5.1f} ± n/a"
    return f"{mean:5.1f} ± {std:4.1f}"


def format_wilson(low: float, high: float) -> str:
    return f"[{low:5.1f}%, {high:5.1f}%]"


def compute_paired_comparison(
    records: list[dict[str, Any]],
    budget: int | None = None,
    mode_a: str = "vacor_kv",
    mode_b: str = "valuediff_pp",
) -> dict[str, Any]:
    if budget is not None:
        b_records = [r for r in records if r["budget"] == budget]
    else:
        b_records = records

    task_a = defaultdict(list)
    task_b = defaultdict(list)
    for r in b_records:
        if r["mode"] == mode_a:
            task_a[r["task"]].append(1.0 if r["all3"] else 0.0)
        elif r["mode"] == mode_b:
            task_b[r["task"]].append(1.0 if r["all3"] else 0.0)

    common_tasks = sorted(set(task_a.keys()) & set(task_b.keys()))

    a_wins = 0
    b_wins = 0
    ties = 0
    task_details = []

    for t in common_tasks:
        score_a = sum(task_a[t]) / len(task_a[t])
        score_b = sum(task_b[t]) / len(task_b[t])
        diff = score_a - score_b
        if diff > 1e-6:
            winner = mode_a
            a_wins += 1
        elif diff < -1e-6:
            winner = mode_b
            b_wins += 1
        else:
            winner = "tie"
            ties += 1

        task_details.append({
            "task": t,
            f"{mode_a}": score_a,
            f"{mode_b}": score_b,
            "winner": winner,
        })

    return {
        "budget": budget,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "total_tasks": len(common_tasks),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "details": task_details,
    }


def generate_summary(tag: str) -> tuple[str, Path]:
    jsonl_path = ROOT / "results" / f"suite_a_ext_{tag}.jsonl"
    out_md_path = ROOT / "results" / f"suite_a_ext_{tag}_summary.md"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Result file not found: {jsonl_path}")

    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {jsonl_path}")

    # Identify budgets
    all_budgets = sorted(set(r["budget"] for r in records))
    eviction_budgets = [b for b in all_budgets if b != 0]

    dense_records = [r for r in records if r["mode"] == "dense"]
    dense_stats = compute_mode_stats(dense_records) if dense_records else None

    lines = []
    lines.append(f"# Suite A Extended Evaluation Summary (`{tag}`)")
    lines.append("")
    lines.append(f"- **Total Records**: {len(records)}")
    lines.append(f"- **Model**: {records[0].get('model', 'Unknown')}")
    lines.append("")

    # Determine tables to render
    display_budgets = eviction_budgets if eviction_budgets else all_budgets

    for b in display_budgets:
        lines.append(f"## Budget {b}")
        lines.append("")
        lines.append("| Mode | Tasks | Seeds | M1 (%) | M2 (%) | M3 (%) | All-3 (%) | 95% CI All-3 | Final (%) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        # Include dense reference row if budget != 0
        if b != 0 and dense_stats:
            d = dense_stats
            lines.append(
                f"| dense* | {d['n_tasks']} | {d['n_seeds']} | "
                f"{format_mean_std(d['m1_mean'], d['m1_std'])} | "
                f"{format_mean_std(d['m2_mean'], d['m2_std'])} | "
                f"{format_mean_std(d['m3_mean'], d['m3_std'])} | "
                f"{format_mean_std(d['all3_mean'], d['all3_std'])} | "
                f"{format_wilson(d['all3_wilson_low'], d['all3_wilson_high'])} | "
                f"{format_mean_std(d['final_mean'], d['final_std'])} |"
            )

        # Get modes for this budget
        b_records = [r for r in records if r["budget"] == b]
        modes_present = sorted(set(r["mode"] for r in b_records), key=lambda m: ORDER.index(m) if m in ORDER else 999)

        for m in modes_present:
            if m == "dense" and b != 0:
                continue
            m_records = [r for r in b_records if r["mode"] == m]
            stats = compute_mode_stats(m_records)
            lines.append(
                f"| {m} | {stats['n_tasks']} | {stats['n_seeds']} | "
                f"{format_mean_std(stats['m1_mean'], stats['m1_std'])} | "
                f"{format_mean_std(stats['m2_mean'], stats['m2_std'])} | "
                f"{format_mean_std(stats['m3_mean'], stats['m3_std'])} | "
                f"{format_mean_std(stats['all3_mean'], stats['all3_std'])} | "
                f"{format_wilson(stats['all3_wilson_low'], stats['all3_wilson_high'])} | "
                f"{format_mean_std(stats['final_mean'], stats['final_std'])} |"
            )

        if b != 0 and dense_stats:
            lines.append("")
            lines.append(r"\* *dense evaluated with full KV cache (budget 0 / no eviction) as reference.*")

        lines.append("")

        # Paired comparison for this budget
        paired = compute_paired_comparison(records, budget=b, mode_a="vacor_kv", mode_b="valuediff_pp")
        lines.append(f"### Paired Comparison: vacor_kv vs valuediff_pp on All-3 (Budget {b})")
        lines.append("")
        tot = paired["total_tasks"]
        lines.append(f"- **Evaluated Tasks**: {tot}")
        lines.append(f"- **vacor_kv wins**: {paired['a_wins']} ({(paired['a_wins'] / tot * 100.0) if tot else 0.0:.1f}%)")
        lines.append(f"- **valuediff_pp wins**: {paired['b_wins']} ({(paired['b_wins'] / tot * 100.0) if tot else 0.0:.1f}%)")
        lines.append(f"- **Ties**: {paired['ties']} ({(paired['ties'] / tot * 100.0) if tot else 0.0:.1f}%)")
        lines.append("")

        if paired["details"]:
            lines.append("| Task | vacor_kv (avg All-3) | valuediff_pp (avg All-3) | Winner |")
            lines.append("| :--- | :---: | :---: | :---: |")
            for dt in paired["details"]:
                w_str = dt["winner"]
                lines.append(f"| {dt['task']} | {dt['vacor_kv']:.2f} | {dt['valuediff_pp']:.2f} | {w_str} |")
            lines.append("")

    # If multiple eviction budgets exist, also add overall paired summary across all (budget, task) pairs
    if len(display_budgets) > 1:
        lines.append("## Overall Paired Comparison: vacor_kv vs valuediff_pp on All-3 (All Budgets Pooled)")
        lines.append("")
        overall_a_wins = 0
        overall_b_wins = 0
        overall_ties = 0
        overall_total = 0
        for b in display_budgets:
            p = compute_paired_comparison(records, budget=b, mode_a="vacor_kv", mode_b="valuediff_pp")
            overall_a_wins += p["a_wins"]
            overall_b_wins += p["b_wins"]
            overall_ties += p["ties"]
            overall_total += p["total_tasks"]
        lines.append(f"- **Total (Budget, Task) Pairs**: {overall_total}")
        lines.append(f"- **vacor_kv wins**: {overall_a_wins} ({(overall_a_wins / overall_total * 100.0) if overall_total else 0.0:.1f}%)")
        lines.append(f"- **valuediff_pp wins**: {overall_b_wins} ({(overall_b_wins / overall_total * 100.0) if overall_total else 0.0:.1f}%)")
        lines.append(f"- **Ties**: {overall_ties} ({(overall_ties / overall_total * 100.0) if overall_total else 0.0:.1f}%)")
        lines.append("")

    summary_text = "\n".join(lines)
    out_md_path.write_text(summary_text, encoding="utf-8")
    return summary_text, out_md_path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Aggregate Suite A Extended evaluation results.")
    parser.add_argument("--tag", default="q17b", help="Tag of the run (default: q17b)")
    args = parser.parse_args()

    text, out_path = generate_summary(args.tag)
    print(text)
    print(f"\nSaved summary to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
