"""
Generates publication-quality Matplotlib figures from the synthetic engine verification
and the Qwen3-1.7B end-to-end benchmarks, and saves them to both `results/figures/`
and the conversation artifact directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
ARTIFACT_DIR = Path(r"C:\Users\felix\.gemini\antigravity\brain\08550006-e047-4fa7-9bcc-f7ba9a58d24f")


def plot_synthetic_diagnostics():
    synth_path = RESULTS_DIR / "synthetic_engine_verification.json"
    if not synth_path.exists():
        return
    data = json.loads(synth_path.read_text(encoding="utf-8"))
    by_mode = {row["mode"]: row for row in data}

    ordered_modes = [
        ("recency_pp", "Recency-pp\n(Sliding Window)"),
        ("snapkv_pp", "SnapKV-pp\n(Attn Pool)"),
        ("rkv_pp", "R-KV-pp\n(Attn + Redund)"),
        ("vase_pp", "VaSE-pp\n(ValRange + Attn)"),
        ("valuediff_pp", "ValueDiff-pp\n(Val L2 Dev)"),
        ("random_pp", "Random-pp\n(Sept 2026 SOTA)"),
        ("vacor_kv", "VaCoR-KV\n(Ours)"),
    ]

    labels = [m[1] for m in ordered_modes]
    keys = [m[0] for m in ordered_modes]

    union_cov = [by_mode[k]["actual_union_coverage_pct"] for k in keys]
    q1_surv = [by_mode[k]["age_quartiles_union_pct"]["Q1_oldest"] for k in keys]
    landmark_union = [by_mode[k]["landmark_union_retention_pct"] for k in keys]
    latency_ms = [by_mode[k]["avg_eviction_latency_ms"] for k in keys]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), dpi=220)

    # Panel A: Grouped Bar Chart of 3 Core Coverage Metrics
    x = np.arange(len(labels))
    w = 0.26
    ax = axes[0]
    b1 = ax.bar(x - w, union_cov, width=w, label="Cross-Head Union Coverage (%)", color="#2b5c8f")
    b2 = ax.bar(x, q1_surv, width=w, label="Oldest-Quartile (Q1) Survival (%)", color="#d95f02")
    b3 = ax.bar(x + w, landmark_union, width=w, label="High-Value Landmark Recall (%)", color="#2ca02c")

    ax.axhline(66.63, color="#666666", linestyle="--", linewidth=1.1, label="I.I.D. Random Limit (66.6%)")
    ax.set_ylabel("Retention / Coverage (%)")
    ax.set_title("(a) Multi-Head Coverage, Horizon Survival & Landmark Recall\n(40 Eviction Rounds, 8 KV Heads, K=512)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 1.2,
                    f"{h:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    # Panel B: Age Quartile Survival Curves (Q1 Oldest -> Q4 Recent)
    ax2 = axes[1]
    quartiles = ["Q1_oldest", "Q2_early_mid", "Q3_late_mid", "Q4_recent"]
    q_labels = ["Q1 (Oldest 25%\nof CoT)", "Q2 (Early-Mid\n25-50%)", "Q3 (Late-Mid\n50-75%)", "Q4 (Recent\n75-100%)"]
    qx = np.arange(4)

    style_map = {
        "recency_pp": ("#7f7f7f", ":", "v", "Recency-pp"),
        "snapkv_pp": ("#1f77b4", "--", "s", "SnapKV-pp"),
        "rkv_pp": ("#9467bd", "--", "^", "R-KV-pp"),
        "vase_pp": ("#e377c2", "-.", "D", "VaSE-pp"),
        "valuediff_pp": ("#ff7f0e", "-.", "P", "ValueDiff-pp"),
        "random_pp": ("#d62728", "-", "o", "Random-pp (I.I.D. Memoryless)"),
        "vacor_kv": ("#117733", "-", "*", "VaCoR-KV (Ours: Antithetic + Value)"),
    }

    for k in keys:
        color, ls, marker, name = style_map[k]
        y = [by_mode[k]["age_quartiles_union_pct"][q] for q in quartiles]
        lw = 2.8 if k in ("vacor_kv", "random_pp") else 1.7
        ms = 9 if k == "vacor_kv" else 6
        ax2.plot(qx, y, linestyle=ls, marker=marker, color=color, linewidth=lw, markersize=ms, label=name)

    ax2.set_xticks(qx)
    ax2.set_xticklabels(q_labels)
    ax2.set_ylabel("Cross-Head Union Survival (%)")
    ax2.set_ylim(-3, 108)
    ax2.set_title("(b) Token Survival Across CoT History Horizon (Q1 Oldest to Q4 Recent)\nSolving Random-Attention's Exponential q^R Attrition")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIG_DIR / "fig1_coverage_and_horizon_survival.png"
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    if ARTIFACT_DIR.exists():
        shutil.copy2(out_file, ARTIFACT_DIR / out_file.name)
    print(f"Saved {out_file}")


def plot_qwen3_benchmarks():
    bench_path = RESULTS_DIR / "all_qwen3_benchmarks.json"
    if not bench_path.exists():
        return
    data = json.loads(bench_path.read_text(encoding="utf-8"))
    suite_a = data.get("suite_a", [])
    suite_b = data.get("suite_b", [])

    if not suite_a:
        return

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), dpi=220)

    # Panel A: Suite A (Budget = 512 or highest budget available) Milestone Recall & Union Coverage
    budgets = sorted({r["token_budget"] for r in suite_a if isinstance(r["token_budget"], int)})
    target_b = budgets[-1] if budgets else 512
    rows_b = {r["mode"]: r for r in suite_a if r["token_budget"] == target_b}

    ordered = [
        ("recency_pp", "Recency-pp"),
        ("snapkv_pp", "SnapKV-pp"),
        ("rkv_pp", "R-KV-pp"),
        ("vase_pp", "VaSE-pp"),
        ("random_pp", "Random-pp"),
        ("valuediff_pp", "ValueDiff-pp"),
        ("vacor_kv", "VaCoR-KV (Ours)"),
    ]
    ordered = [m for m in ordered if m[0] in rows_b]
    keys = [m[0] for m in ordered]
    labels = [m[1] for m in ordered]

    m1_acc = [rows_b[k]["m1_oldest_recall_acc"] for k in keys]
    avg_milestone_acc = [
        (rows_b[k]["m1_oldest_recall_acc"] + rows_b[k]["m2_mid_recall_acc"] + rows_b[k]["m3_recent_recall_acc"]) / 3.0
        for k in keys
    ]
    union_cov = [rows_b[k]["union_coverage_pct"] for k in keys]

    x = np.arange(len(keys))
    w = 0.26
    ax = axes[0]
    b1 = ax.bar(x - w, m1_acc, width=w, label="Stage-1 Oldest Milestone Recall (%)", color="#c44e52")
    b2 = ax.bar(x, avg_milestone_acc, width=w, label="Mean 3-Stage Milestone Recall (%)", color="#4c72b0")
    b3 = ax.bar(x + w, union_cov, width=w, label="Cross-Head Union Coverage (%)", color="#55a868")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"(a) Qwen3-1.7B Long-CoT Milestone Recall & Coverage\n(Suite A, 2,100+ Tokens, KV Budget = {target_b})")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8.5)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.2,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # Panel B: Speed vs. Coverage / Accuracy Trade-off (Latency ms per Eviction vs Union Coverage & Milestone Recall)
    ax2 = axes[1]
    if suite_b:
        sb_map = {r["mode"]: r for r in suite_b if r["mode"] != "dense"}
        modes_b = [k for k in keys if k in sb_map]
        x_b = np.arange(len(modes_b))
        acc_b = [sb_map[k]["accuracy_pct"] for k in modes_b]
        ucov_b = [sb_map[k]["union_coverage_pct"] for k in modes_b]
        q1_b = [sb_map[k]["q1_oldest_union_pct"] for k in modes_b]

        b_acc = ax2.bar(x_b - w, acc_b, width=w, label="Math Reasoning Accuracy (%)", color="#2b5c8f")
        b_uc = ax2.bar(x_b, ucov_b, width=w, label="Cross-Head Union Coverage (%)", color="#2ca02c")
        b_q1 = ax2.bar(x_b + w, q1_b, width=w, label="Oldest-Quartile (Q1) Survival (%)", color="#ff7f0e")

        ax2.set_xticks(x_b)
        ax2.set_xticklabels([dict(ordered)[k] for k in modes_b], rotation=18, ha="right")
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Percentage (%)")
        ax2.set_title("(b) Qwen3-1.7B Free-Generation Math Reasoning\n(Suite B: GSM8K + MATH-500, KV Budget = 256)")
        ax2.grid(axis="y", linestyle=":", alpha=0.5)
        ax2.legend(loc="upper left", fontsize=8.5)

        for bars in (b_acc, b_uc, b_q1):
            for bar in bars:
                h = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 1.2,
                    f"{h:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIG_DIR / "fig2_qwen3_end_to_end_benchmarks.png"
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    if ARTIFACT_DIR.exists():
        shutil.copy2(out_file, ARTIFACT_DIR / out_file.name)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    plot_synthetic_diagnostics()
    plot_qwen3_benchmarks()
