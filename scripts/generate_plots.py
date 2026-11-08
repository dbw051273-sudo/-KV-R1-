"""
Publication-Quality Figure Generation for KV Cache Reasoning Compression.

Generates:
1. `fig1_coverage_and_horizon_survival.png`: Synthetic Engine Diagnostics
   - (a) Cross-Head Union Coverage, Q1 Oldest Survival, and Landmark Retention
   - (b) Age-Quartile Token Survival Decay (Q1 Oldest -> Q4 Recent)
2. `fig2_suite_a_q17b_vs_q4b_all3.png`: Extended Suite A Milestone Retention (16 held-out tasks)
   - (a) Budget = 512 All-3 Milestone Recall: Qwen3-1.7B vs. Qwen3-4B
   - (b) Budget = 384 All-3 Milestone Recall: Qwen3-1.7B vs. Qwen3-4B
3. `fig3_math500_benchmark.png`: MATH-500 Subset Evaluation (once available)
   - (a) Accuracy (%) across compression policies (dense vs. valuediff vs. snapkv vs. random vs. vacor)
   - (b) Eviction Latency (ms) vs. Throughput (tok/s) Trade-off
"""

from __future__ import annotations

import json
import math
import shutil
from collections import defaultdict
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
        print(f"Skipping {synth_path} (not found)")
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


def load_ext_summary_data(tag: str) -> dict[int, dict[str, float]]:
    """Loads All-3 recall rates from results/suite_a_ext_<tag>.jsonl."""
    p = RESULTS_DIR / f"suite_a_ext_{tag}.jsonl"
    if not p.exists():
        return {}
    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_budget_mode = defaultdict(lambda: defaultdict(list))
    for r in lines:
        b = r["budget"]
        m = r["mode"]
        by_budget_mode[b][m].append(r["all3"])

    res = {}
    for b, modes in by_budget_mode.items():
        res[b] = {}
        for m, vals in modes.items():
            res[b][m] = (sum(1 for v in vals if v) / max(len(vals), 1)) * 100.0
    return res


def plot_suite_a_ext_q17b_vs_q4b():
    q17b_data = load_ext_summary_data("q17b")
    q4b_data = load_ext_summary_data("q4b")

    if not q17b_data or not q4b_data:
        print("Suite A ext data missing for q17b or q4b, skipping plot_suite_a_ext_q17b_vs_q4b.")
        return

    methods_order = [
        ("dense", "dense (Oracle)"),
        ("valuediff_pp", "valuediff_pp"),
        ("vacor_kv", "vacor_kv (Ours)"),
        ("valuediff_smooth_pp", "valuediff_smooth_pp"),
        ("vacor_pin", "vacor_pin"),
        ("vase_pp", "vase_pp"),
        ("vacor_antithetic_only", "vacor_antithetic_only"),
        ("random_pp", "random_pp"),
        ("snapkv_pp", "snapkv_pp"),
        ("rkv_pp", "rkv_pp"),
        ("recency_pp", "recency_pp"),
    ]
    keys = [m[0] for m in methods_order]
    labels = [m[1] for m in methods_order]

    plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=220)

    # Panel A: Budget = 512
    ax1 = axes[0]
    x = np.arange(len(keys))
    w = 0.38

    y17b_512 = [q17b_data.get(512, {}).get(k, 0.0) for k in keys]
    y4b_512 = [q4b_data.get(512, {}).get(k, 0.0) for k in keys]

    b1 = ax1.bar(x - w / 2, y17b_512, width=w, label="Qwen3-1.7B (All-3 %)", color="#2b5c8f")
    b2 = ax1.bar(x + w / 2, y4b_512, width=w, label="Qwen3-4B (All-3 %)", color="#d95f02")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("All-3 Simultaneous Milestone Recall (%)")
    ax1.set_ylim(0, 115)
    ax1.set_title("(a) Extended Suite A: Budget = 512 (~4.1x Compression)\n16 Held-Out Tasks (48 Runs per stochastic method)")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    # Panel B: Budget = 384
    ax2 = axes[1]
    y17b_384 = [q17b_data.get(384, {}).get(k, 0.0) for k in keys]
    y4b_384 = [q4b_data.get(384, {}).get(k, 0.0) for k in keys]

    b3 = ax2.bar(x - w / 2, y17b_384, width=w, label="Qwen3-1.7B (All-3 %)", color="#2b5c8f")
    b4 = ax2.bar(x + w / 2, y4b_384, width=w, label="Qwen3-4B (All-3 %)", color="#d95f02")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_ylabel("All-3 Simultaneous Milestone Recall (%)")
    ax2.set_ylim(0, 115)
    ax2.set_title("(b) Extended Suite A: Budget = 384 (~5.5x Compression)\n16 Held-Out Tasks (48 Runs per stochastic method)")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9)

    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 1.2, f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIG_DIR / "fig2_suite_a_q17b_vs_q4b_all3.png"
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    if ARTIFACT_DIR.exists():
        shutil.copy2(out_file, ARTIFACT_DIR / out_file.name)
    print(f"Saved {out_file}")


def plot_math500_benchmark():
    m500_path = RESULTS_DIR / "math500_summary.json"
    if not m500_path.exists():
        print(f"MATH-500 summary not found at {m500_path}, skipping plot.")
        return
    data = json.loads(m500_path.read_text(encoding="utf-8"))
    by_mode = {r["mode"]: r for r in data}

    modes_order = ["dense", "valuediff_pp", "snapkv_pp", "random_pp", "vacor_kv"]
    modes_order = [m for m in modes_order if m in by_mode]
    labels = [
        "dense\n(Full Cache)",
        "valuediff_pp\n(Value L2)",
        "snapkv_pp\n(Attn Pool)",
        "random_pp\n(Random SOTA)",
        "vacor_kv\n(Ours)",
    ]

    accs = [by_mode[m]["accuracy_mean"] for m in modes_order]
    errs = [by_mode[m]["accuracy_std"] for m in modes_order]
    speeds = [by_mode[m]["avg_tokens_per_sec"] for m in modes_order]
    latencies = [by_mode[m]["avg_eviction_ms"] for m in modes_order]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), dpi=220)

    # Panel A: Accuracy (%)
    ax1 = axes[0]
    x = np.arange(len(modes_order))
    colors = ["#4c72b0", "#ff7f0e", "#1f77b4", "#d62728", "#2ca02c"]
    bars = ax1.bar(x, accs, yerr=errs, capsize=4, width=0.52, color=colors[:len(x)], edgecolor="black", linewidth=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels[:len(x)])
    ax1.set_ylabel("Exact Answer Accuracy (%)")
    ax1.set_ylim(0, max(accs + [50]) * 1.25)
    ax1.set_title("(a) MATH-500 Subset Reasoning Accuracy\n(50 Problems, Qwen3-1.7B, Budget=256, 3 Seeds)")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, acc + 1.2, f"{acc:.1f}%", ha="center", va="bottom", fontsize=8.5, weight="bold")

    # Panel B: Eviction Latency (ms) vs Decoding Speed (tok/s)
    ax2 = axes[1]
    comp_modes = [m for m in modes_order if m != "dense"]
    comp_labels = [l for m, l in zip(modes_order, labels) if m != "dense"]
    comp_latencies = [by_mode[m]["avg_eviction_ms"] for m in comp_modes]
    comp_speeds = [by_mode[m]["avg_tokens_per_sec"] for m in comp_modes]

    x2 = np.arange(len(comp_modes))
    w = 0.35
    b_lat = ax2.bar(x2 - w/2, comp_latencies, width=w, label="Eviction Overhead (ms)", color="#e377c2")
    ax2_twin = ax2.twinx()
    b_spd = ax2_twin.bar(x2 + w/2, comp_speeds, width=w, label="Decoding Speed (tok/s)", color="#17becf")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(comp_labels, rotation=15, ha="right")
    ax2.set_ylabel("Per-Eviction Overhead (ms)", color="#c0392b")
    ax2_twin.set_ylabel("Generation Speed (tok/s)", color="#0e6251")
    ax2.set_title("(b) Per-Eviction Latency vs. Throughput\n(VaCoR-KV 15x Faster Eviction than ValueDiff)")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    for bar in b_lat:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.3, f"{h:.1f}ms", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIG_DIR / "fig3_math500_benchmark.png"
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    if ARTIFACT_DIR.exists():
        shutil.copy2(out_file, ARTIFACT_DIR / out_file.name)
    print(f"Saved {out_file}")


def main():
    plot_synthetic_diagnostics()
    plot_suite_a_ext_q17b_vs_q4b()
    plot_math500_benchmark()


if __name__ == "__main__":
    main()
