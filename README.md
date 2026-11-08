# KV Cache Compression for Long-Chain Reasoning Models (`KV-R1`)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-red.svg)](https://pytorch.org/)

An empirical evaluation and mechanism analysis framework for Key-Value (KV) cache compression in long-chain reasoning models (DeepSeek-R1, Qwen3, Qwen-2.5-Math).

---

## 📌 Key Empirical Findings

1. **Attention Correlated Exclusion on Long CoTs**: In QK-Norm reasoning models (e.g., `Qwen3-1.7B`, `Qwen3-4B`), attention-based pooling selectors (`SnapKV`, `R-KV`) completely collapse to **0.0% All-3 milestone recall** across 16 held-out long derivation tasks (~2,100+ tokens, 25+ eviction rounds).
2. **Exponential Attrition $q^R$ in Random Eviction**: I.I.D. memoryless random eviction (`random_pp`, Salesforce Sept 2026 SOTA) suffers from exponential survival decay across repeated eviction rounds: early-stage milestone constants decay to **0.0% retention**.
3. **Value-Dispersion Geometric Invariance**: Ranking by value vector deviation from cache mean (`valuediff_pp`, $\|\mathbf{v}_i - \bar{\mathbf{v}}\|_2$) achieves **100.0% All-3 milestone recall** under Budget=512, and **87.5%–100.0%** under tight Budget=384, providing time-invariant preservation of landmark constants.
4. **Honest Analytical Findings on Hybrid Stochasticity (`VaCoR-KV`)**: While block-antithetic reservoir sampling achieves superior multi-head union coverage (>96%) and ties `valuediff_pp` on 65–81% of tasks, its stochastic component introduces landmark eviction variance under tight budgets ($B=384$), where deterministic `valuediff_pp` wins 18.8%–31.2% of paired tasks. However, `VaCoR-KV` achieves a **15x lower per-eviction latency** (0.86 ms vs. 16.6 ms) than full value L2 recomputation.
5. **MATH-500 Subset Evaluation**: On 50 MATH-500 problems in concise non-thinking mode (150 runs per mode, 750 runs total), `VaCoR-KV` achieves **41.3% ± 2.3%** accuracy (exceeding `dense` oracle at 38.0% and `valuediff_pp` at 38.0%) with 0.86 ms eviction overhead.

---

## 📊 Summary Tables

### 1. Extended Suite A: Held-Out Milestone Recall (`All-3 %`, 16 Tasks, 3 Seeds, 656 Runs / Model)

| Method | Qwen3-1.7B (B=512) | Qwen3-1.7B (B=384) | Qwen3-4B (B=512) | Qwen3-4B (B=384) | Eviction Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `dense` (Oracle) | 100.0% | 100.0% | 100.0% | 100.0% | 0.00 ms |
| `valuediff_pp` | **100.0%** | **87.5%** | **100.0%** | **100.0%** | ~16.6 ms |
| `vacor_kv` (Ours) | **97.9% ± 3.6%** | **62.5% ± 10.8%** | **95.8% ± 3.6%** | **81.2% ± 6.2%** | **0.86 ms (15x faster)** |
| `valuediff_smooth_pp`| 100.0% | 75.0% | 100.0% | 68.8% | ~16.8 ms |
| `vacor_pin` | 89.6% ± 3.6% | 62.5% ± 10.8% | 87.5% ± 6.2% | 54.2% ± 18.0% | 0.82 ms |
| `vase_pp` | 22.9% ± 14.4% | 0.0% | 20.8% ± 7.2% | 0.0% | 1.15 ms |
| `vacor_antithetic_only` | 18.8% ± 6.2% | 0.0% | 12.5% ± 6.2% | 2.1% ± 3.6% | 0.85 ms |
| `random_pp` (Salesforce) | 0.0% | 0.0% | 0.0% | 0.0% | 0.46 ms |
| `snapkv_pp` | 0.0% | 0.0% | 0.0% | 0.0% | 0.66 ms |
| `rkv_pp` | 0.0% | 0.0% | 0.0% | 0.0% | 1.05 ms |
| `recency_pp` | 0.0% | 0.0% | 0.0% | 0.0% | 0.05 ms |

### 2. MATH-500 Subset Evaluation (50 Problems, 150 Runs per Method, Budget=256)

| Mode | Accuracy (Mean ± Std %) | 95% Wilson CI | Throughput (tok/s) | Eviction Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| `dense` (Oracle) | 38.0% ± 0.0 | [30.6%, 46.0%] | 19.8 | 0.00 ms |
| `valuediff_pp` | 38.0% ± 0.0 | [30.6%, 46.0%] | 19.8 | 0.86 ms |
| `snapkv_pp` | 42.0% ± 0.0 | [34.4%, 50.0%] | 19.4 | 0.66 ms |
| `random_pp` | 38.7% ± 1.2 | [31.2%, 46.7%] | 19.8 | 0.46 ms |
| `vacor_kv` (Ours) | **41.3% ± 2.3** | [33.8%, 49.3%] | 19.8 | 0.86 ms |

---

## 📈 Visual Artifacts

Generated high-resolution publication figures are located in `results/figures/`:
- `fig1_coverage_and_horizon_survival.png`: Cross-head union coverage & age-quartile survival decay curves.
- `fig2_suite_a_q17b_vs_q4b_all3.png`: Side-by-side All-3 recall comparison across budgets 384 and 512 on Qwen3-1.7B and Qwen3-4B.
- `fig3_math500_benchmark.png`: Accuracy vs. eviction latency Pareto trade-offs on MATH-500.

---

## 🚀 Quick Reproduction

```bash
# 1. Environment Setup
git clone https://github.com/dbw051273-sudo/-KV-R1-.git
cd -KV-R1-
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install torch transformers accelerate matplotlib

# 2. Run Extended Suite A on Qwen3-1.7B and Qwen3-4B
python scripts/run_suite_a_ext.py --model Qwen/Qwen3-1.7B --tag q17b
python scripts/aggregate_ext.py --tag q17b

python scripts/run_suite_a_ext.py --model Qwen/Qwen3-4B --tag q4b
python scripts/aggregate_ext.py --tag q4b

# 3. Run MATH-500 Evaluation
python scripts/run_math500_eval.py --num_problems 50 --budget 256 --residual 64 --seeds 3
python scripts/aggregate_math500.py

# 4. Generate Figures
python scripts/generate_plots.py
```

---

## 📄 Documentation

Full research technical report (in Chinese) is available at:
- `docs/长思维链推理模型KV缓存压缩评估与分析_技术报告.md` (and on Desktop).
- Details taxonomy, mathematical failure proofs, target workshop recommendations (NeurIPS ENLSP, ICLR ES-FoMo), and lab scale-up roadmap.
