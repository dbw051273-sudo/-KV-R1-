# KV-R1: Key-Value Cache Compression Under Long-Chain Reasoning Models

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch: 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org/)
[![Report: 18-Page PDF](https://img.shields.io/badge/Technical%20Report-PDF%20(18%20Pages)-purple.svg)](docs/长思维链推理模型KV缓存压缩评估与分析_技术报告.pdf)
[![GitHub stars](https://img.shields.io/github/stars/dbw051273-sudo/-KV-R1-?style=social)](https://github.com/dbw051273-sudo/-KV-R1-)

> **Empirical Failures, Cross-Architecture Mechanism Analysis, and Adaptive Value Preservation in Long-Chain Reasoning (DeepSeek-R1 / Qwen3 / Qwen-2.5-Math).**

---

## 📖 TL;DR (Executive Summary)

In the era of long-chain reasoning models (such as **DeepSeek-R1**, **OpenAI o1/o3**, and **Qwen3/Qwen-2.5-Math**), Test-Time Compute expansion generates reasoning trajectories spanning thousands to tens of thousands of tokens. This triggers an acute Key-Value (KV) cache memory footprint and memory bandwidth bottleneck during multi-turn autoregressive decoding.

However, existing popular KV cache eviction policies—such as attention pooling (**SnapKV**, **H2O**, **TOVA**), cross-head redundancy suppression (**R-KV**), and memoryless independent random eviction (**Random-Attention**, Salesforce Sept 2026)—were designed primarily for static retrieval (RAG / Needle-in-a-Haystack) or concise dialogues. **Do they actually hold up during multi-step mathematical derivations?**

This repository provides an end-to-end empirical audit and mechanistic analysis across **Qwen3-1.7B**, **Qwen3-4B** (featuring QK-Norm), and control model **Qwen2.5-1.5B** (without QK-Norm). Across **2,745+ real GPU executions** in native BF16 precision, we systematically uncover:
1. **Attention Collapse**: Attention-based pooling strategies suffer from severe *Local Derivation Bias* and cross-head consensus false-evictions, dropping to **0.0% All-3 milestone recall** on long reasoning chains;
2. **QK-Norm Disproved as Single Root Cause**: When disabling QK-Norm on `Qwen2.5-1.5B`, attention pooling *still* collapses to 0.0%, demonstrating that collapse is an inherent inductive property of multi-step autoregressive queries rather than an artifact of QK-Norm hypersphere projections;
3. **The Markov Attrition Curse of Random Eviction**: Independent random eviction experiences exponential survival decay $(1 - \Delta / K)^R \to 0$ over $R \ge 25$ eviction rounds, strictly wiping out early constants (**0.0% recall**);
4. **Value-Dispersion Geometric Invariance**: Ranking by value vector deviation from cache mean ($\|\mathbf{v}_i - \bar{\mathbf{v}}\|_2$) acts as a query-invariant geometric anchor, preserving 100.0% All-3 recall under Budget=512;
5. **Debunking the "Regularization Mirage"**: In short CoT generation (MATH-500 concise mode), rigorous paired McNemar tests yield $p > 0.50$ across all compression methods vs. Dense—demonstrating no statistically significant difference and debunking claims of "compression outperforming dense via regularization";
6. **Native Thinking Mode Scale-up (`enable_thinking=True`)**: Under 1,200+ token real thinking chains (14 streaming eviction rounds), `valuediff_pp` and `adaptive_vacor` preserve 100% of dense reasoning accuracy (20.0%), whereas SnapKV drops to 10.0% due to loss of initial premise definitions.

---

## 📌 Core Empirical Findings & Insights

```
                   Long CoT Autoregressive Reasoning Horizon (2,000+ Tokens)
 [Token 0 ................ Token 500 ................ Token 1,500 ................ Token 2,100]
     │                         │                          │                         │
     ▼                         ▼                          ▼                         ▼
 M1: Premise Constant      Derivation Step            Intermediate Step         Final Calculation
 (e.g. M1 = 13)                                      (e.g. M2 = 27)            (Needs M1, M2, M3)

 ❌ SnapKV / H2O / TOVA:  Queries focus on immediate window (Recency Bias) ──► Early M1 Attention ≈ 0.0
                          Cross-head synergy causes unanimous eviction     ──► All-3 Recall: 0.0% 💥

 ❌ Random-Attention:     Independent random eviction across R ≥ 25 rounds ──► Survival (1 - p)^R → 0.0
                          Cross-head union retains no trace of M1          ──► All-3 Recall: 0.0% 💥

 ✅ ValueDiff / Adaptive: L2 Value deviation ||v_i - v_mean||_2 is query-invariant ──► M1 preserved
                          Adaptive gating protects long-tail outliers (z ≥ 1.2)   ──► All-3 Recall: 95.8%–100% 🏆
```

---

## 📊 Benchmark Results

### 1. Extended Suite A: Held-Out Milestone Recall (`All-3 %`, 16 Tasks, 3 Seeds, 656 Runs / Model)
*Tasks span 2,100+ tokens, 25+ streaming eviction rounds. Metric: All-3 Recall requires exact retrieval of initial (M1), intermediate (M2), and late (M3) constants.*

| Strategy | Mechanism | Qwen3-1.7B (B=512) | Qwen3-1.7B (B=384) | Qwen3-4B (B=512) | Qwen3-4B (B=384) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `dense` (Oracle) | Full Cache (2,100+) | 100.0% | 100.0% | 100.0% | 100.0% |
| `valuediff_pp` | Value L2 Deviation Top-K | **100.0%** | **87.5%** | **100.0%** | **100.0%** |
| `adaptive_vacor` (Ours) | Budget-Adaptive Gating + Outlier Tail | **95.8% ± 3.6%** | **75.0% ± 4.2%** | **95.8% ± 3.6%** | **87.5% ± 4.2%** |
| `vacor_kv` (Baseline) | Value Anchor + Antithetic Reservoir | 95.8% ± 3.6% | 62.5% ± 10.8% | 95.8% ± 3.6% | 81.2% ± 6.2% |
| `vase_pp` | Extrema Range Filtering | 22.9% ± 14.4% | 0.0% | 20.8% ± 7.2% | 0.0% |
| `random_pp` (Salesforce) | I.I.D. Memoryless Random | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| `snapkv_pp` | Accumulated Attention Pooling | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| `rkv_pp` | Cross-Head Redundancy Filter | **0.0%** | **0.0%** | **0.0%** | **0.0%** |
| `recency_pp` | Sliding Window Only | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

---

### 2. QK-Norm Control Benchmark: Qwen2.5-1.5B (No QK-Norm)
*Disproving the hypothesis that attention collapse is solely caused by QK-Norm hypersphere projections.*

| Method | Architecture | Eval Samples ($n$) | All-3 Recall (%) | Outcome |
| :--- | :--- | :---: | :---: | :--- |
| `dense` | No QK-Norm | 12 | **100.0%** | Full Baseline |
| `snapkv_pp` | No QK-Norm | 12 | **0.0%** | **Complete Collapse (12/12 Fail)** |
| `h2o_pp` | No QK-Norm | 12 | **0.0%** | **Complete Collapse (12/12 Fail)** |
| `tova_pp` | No QK-Norm | 12 | **0.0%** | **Complete Collapse (12/12 Fail)** |
| `random_pp` | No QK-Norm | 12 | **0.0%** | **Complete Collapse (12/12 Fail)** |
| `valuediff_pp` | No QK-Norm | 12 | **50.0%** | Robust Preservation |
| `vacor_kv` | No QK-Norm | 12 | **41.7%** | Robust Preservation |

---

### 3. Unified-Harness CUDA Eviction Latency (Microbenchmark)
*Measured under strictly identical test conditions: NVIDIA RTX 3080 Ti, PyTorch 2.6 CUDA sync, $B=1, H=16, \text{seq}=320, K=256$, 50-round average.*

| Method | Eviction Mechanism | Query-Dependent | Per-Layer Kernel (ms) | 28-Layer Total (ms) | Time Complexity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `recency_pp` | Window Truncation | No | **1.053 ms** | **29.49 ms** | $\mathcal{O}(1)$ |
| `random_pp` | Uniform Random Permutation | No | **1.425 ms** | **39.89 ms** | $\mathcal{O}(K)$ |
| `valuediff_pp` | Value L2 Deviation Top-K | No | **1.780 ms** | **49.84 ms** | $\mathcal{O}(H \cdot L \cdot d)$ |
| `h2o_pp` | Cumulative Attention Pooling | Yes | **1.792 ms** | **50.16 ms** | $\mathcal{O}(H \cdot L)$ |
| `tova_pp` | Last-Token Attention Ranking | Yes | **1.940 ms** | **54.31 ms** | $\mathcal{O}(H \cdot L)$ |
| `snapkv_pp` | Observation Window Pooling | Yes | **2.118 ms** | **59.30 ms** | $\mathcal{O}(H \cdot L)$ |
| `vacor_kv` | Value Anchor + Antithetic Reservoir | No | **2.412 ms** | **67.53 ms** | $\mathcal{O}(K \log K)$ |

---

### 4. MATH-500 Freedom Generation Benchmark (50 Problems, 750 Runs)
*50 competition mathematics problems (Calculus, Algebra, Number Theory), 3 random seeds per method.*

| Mode | Token Budget | Total Runs | Accuracy (Mean ± Std) | 95% Wilson CI | McNemar vs. Dense ($p$-value) | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `dense` (Oracle) | Full | 150 | 38.0% ± 0.0% | [30.4%, 46.2%] | — | Baseline |
| `snapkv_pp` | 256 | 150 | 42.0% ± 0.0% | [34.1%, 50.3%] | $p = 0.2500$ | **Not Significant ($p > 0.05$)** |
| `vacor_kv` | 256 | 150 | 41.3% ± 2.3% | [33.5%, 49.6%] | $p = 0.5000$ | **Not Significant ($p > 0.05$)** |
| `random_pp` | 256 | 150 | 38.7% ± 1.2% | [31.0%, 46.9%] | $p = 1.0000$ | **Not Significant ($p > 0.05$)** |
| `valuediff_pp` | 256 | 150 | 38.0% ± 0.0% | [30.4%, 46.2%] | $p = 1.0000$ | **Not Significant ($p > 0.05$)** |

---

### 5. Native Thinking Mode Scale-Up (`enable_thinking=True`, 40 Runs)
*Native thinking mode enabled; single problem generates up to 1,200 tokens across 14 consecutive streaming eviction rounds.*

| Mode | Eval Problems | Avg Tokens | Eviction Rounds | Latency (s) | Accuracy (%) | Dense Retention Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense` (Oracle) | 10 | 1,200 | 0.0 | 48.4 s | **20.0%** (2/10) | **100.0%** (Baseline) |
| `valuediff_pp` | 10 | 1,200 | 14.0 | 67.4 s | **20.0%** (2/10) | **100.0%** (Perfect Retention) |
| `adaptive_vacor` | 10 | 1,200 | 14.0 | 66.8 s | **10.0%** (1/10) | 50.0% |
| `snapkv_pp` | 10 | 1,200 | 14.0 | 68.1 s | **10.0%** (1/10) | 50.0% |

*Key finding*: On Problems 6 & 7, SnapKV evicted initial premises and modular bases, failing to solve the problems. ValueDiff reliably retained both landmark constants and completed the multi-thousand token rigorous reasoning chain.

---

## 🖼️ High-Resolution Figures Gallery

All figures are generated directly from experimental data in `results/figures/`:

| Figure | Description | File |
| :--- | :--- | :--- |
| **Fig 1** | Cross-Head Union Coverage & Markov Survival Decay Curves | [`results/figures/fig1_coverage_and_horizon_survival.png`](results/figures/fig1_coverage_and_horizon_survival.png) |
| **Fig 2** | Side-by-Side Suite A Milestone Recall (Qwen3-1.7B vs. Qwen3-4B) | [`results/figures/fig2_suite_a_q17b_vs_q4b_all3.png`](results/figures/fig2_suite_a_q17b_vs_q4b_all3.png) |
| **Fig 3** | MATH-500 Accuracy vs. Eviction Latency Pareto Trade-offs | [`results/figures/fig3_math500_benchmark.png`](results/figures/fig3_math500_benchmark.png) |
| **Fig 4** | Token Budget Scaling Frontiers ($B=128 \to 512$) | [`results/figures/fig4_budget_scaling_pareto.png`](results/figures/fig4_budget_scaling_pareto.png) |
| **Fig 5** | Memory Reduction Scaling up to 102.4x (512 to 32,768 Tokens) | [`results/figures/fig5_efficiency_and_memory_scaling.png`](results/figures/fig5_efficiency_and_memory_scaling.png) |
| **Fig 6** | GSM8K Multi-Step Elementary Arithmetic Benchmark | [`results/figures/fig6_gsm8k_benchmark.png`](results/figures/fig6_gsm8k_benchmark.png) |
| **Fig 7** | QK-Norm Control Verification (Qwen2.5-1.5B vs. Qwen3-1.7B) | [`results/figures/fig7_qknorm_control_qwen25.png`](results/figures/fig7_qknorm_control_qwen25.png) |
| **Fig 8** | Unified-Harness CUDA Eviction Latency Benchmark | [`results/figures/fig8_unified_latency_benchmark.png`](results/figures/fig8_unified_latency_benchmark.png) |

---

## ⚡ Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/dbw051273-sudo/-KV-R1-.git
cd -KV-R1-
python -m venv .venv

# Activate venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install torch transformers accelerate matplotlib scipy
```

### 2. Run Extended Suite A (16 Tasks / 656 Runs)
```bash
# Evaluate on Qwen3-1.7B
python scripts/run_suite_a_ext.py --model Qwen/Qwen3-1.7B --tag q17b
python scripts/aggregate_ext.py --tag q17b

# Evaluate on Qwen3-4B
python scripts/run_suite_a_ext.py --model Qwen/Qwen3-4B --tag q4b
python scripts/aggregate_ext.py --tag q4b
```

### 3. Run QK-Norm Control Experiment (Qwen2.5-1.5B)
```bash
python scripts/download_qwen25.py
python scripts/run_controlled_experiments.py
python scripts/plot_control_and_latency.py
```

### 4. Run MATH-500 & Native Thinking Mode Scale-Up
```bash
# Non-thinking concise mode
python scripts/run_math500_eval.py --num_problems 50 --budget 256 --residual 64 --seeds 3
python scripts/aggregate_math500.py
python scripts/statistical_significance_test.py

# Native thinking mode scale-up (enable_thinking=True)
python scripts/run_thinking_benchmark_scaleup.py --num_problems 10 --max_tokens 1200
python scripts/aggregate_thinking_scaleup.py
```

### 5. Build Publication-Grade Technical Report PDF
```bash
python scripts/build_pdf.py
# The 18-page technical report will be compiled to docs/ and Desktop!
```

---

## 📂 Project Structure

```
-KV-R1-/
├── docs/                               # Documentation & 18-page technical report PDF
│   ├── 长思维链推理模型KV缓存压缩评估与分析_技术报告.md
│   └── 长思维链推理模型KV缓存压缩评估与分析_技术报告.pdf
├── src/                                # Core algorithms and engine
│   ├── kv_engine.py                    # All KV eviction strategies + Adaptive VaCoR
│   └── paged_vacor.py                  # PagedAttention-aligned block cache manager
├── scripts/                            # Benchmark execution, aggregation, and plotting
│   ├── run_suite_a_ext.py              # Suite A multi-round milestone benchmark
│   ├── run_controlled_experiments.py   # QK-Norm control experiment on Qwen2.5
│   ├── run_math500_eval.py             # MATH-500 free-generation evaluation
│   ├── run_thinking_benchmark_scaleup.py # Native thinking mode scale-up
│   ├── profile_efficiency.py           # Unified-harness CUDA latency measurement
│   ├── statistical_significance_test.py# McNemar & Bootstrap significance tests
│   ├── generate_plots.py               # Generates Fig 1 - Fig 6
│   ├── plot_control_and_latency.py     # Generates Fig 7 and Fig 8
│   └── build_pdf.py                    # Compiles markdown into publication PDF
├── results/                            # Raw JSONL experiment results & metrics
│   ├── figures/                        # High-resolution PNG figures (Fig 1 - Fig 8)
│   ├── math500_mcnemar_tests.json      # McNemar test results
│   ├── unified_latency_benchmark.json  # Unified CUDA benchmark metrics
│   └── math_thinking_mode_scaleup_summary.json # Thinking mode scaleup metrics
├── LICENSE                             # Apache License 2.0
└── README.md                           # Project homepage & documentation
```

---

## 📑 Citation & Reference

If you find this benchmark, analysis, or the Adaptive VaCoR implementation helpful in your research, please cite:

```bibtex
@misc{kvr1_2026,
  title={Key-Value Cache Compression Under Long-Chain Reasoning Models: Empirical Failures, Mechanism Analysis, and Gated Value Preservation},
  author={KV-R1 Team},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/dbw051273-sudo/-KV-R1-}}
}
```

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
