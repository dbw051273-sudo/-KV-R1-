# Suite A Extended Evaluation Summary (`q17b`)

- **Total Records**: 656
- **Model**: Qwen/Qwen3-1.7B

## Budget 384

| Mode | Tasks | Seeds | M1 (%) | M2 (%) | M3 (%) | All-3 (%) | 95% CI All-3 | Final (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dense* | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 80.6%, 100.0%] |  93.8 ± n/a |
| recency_pp | 16 | 1 |   0.0 ± n/a |  18.8 ± n/a |   6.2 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| snapkv_pp | 16 | 1 |   0.0 ± n/a |   0.0 ± n/a |  81.2 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| rkv_pp | 16 | 1 |   0.0 ± n/a |   6.2 ± n/a |  25.0 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| valuediff_pp | 16 | 1 |  93.8 ± n/a |  93.8 ± n/a | 100.0 ± n/a |  87.5 ± n/a | [ 64.0%,  96.5%] |  81.2 ± n/a |
| valuediff_smooth_pp | 16 | 1 |  93.8 ± n/a | 100.0 ± n/a |  81.2 ± n/a |  75.0 ± n/a | [ 50.5%,  89.8%] |  68.8 ± n/a |
| vase_pp | 16 | 3 |  25.0 ±  6.2 |  10.4 ±  7.2 |  54.2 ±  7.2 |   0.0 ±  0.0 | [  0.0%,   7.4%] |   0.0 ±  0.0 |
| random_pp | 16 | 3 |   0.0 ±  0.0 |  10.4 ±  7.2 |  10.4 ±  7.2 |   0.0 ±  0.0 | [  0.0%,   7.4%] |   2.1 ±  3.6 |
| vacor_kv | 16 | 3 | 100.0 ±  0.0 |  93.8 ±  6.2 |  62.5 ± 10.8 |  62.5 ± 10.8 | [ 48.4%,  74.8%] |  58.3 ±  9.5 |
| vacor_pin | 16 | 3 |  97.9 ±  3.6 |  85.4 ±  3.6 |  64.6 ± 13.0 |  62.5 ± 10.8 | [ 48.4%,  74.8%] |  50.0 ± 10.8 |
| vacor_antithetic_only | 16 | 3 |  35.4 ± 13.0 |  29.2 ±  3.6 |   8.3 ±  9.5 |   2.1 ±  3.6 | [  0.4%,  10.9%] |   2.1 ±  3.6 |

\* *dense evaluated with full KV cache (budget 0 / no eviction) as reference.*

### Paired Comparison: vacor_kv vs valuediff_pp on All-3 (Budget 384)

- **Evaluated Tasks**: 16
- **vacor_kv wins**: 1 (6.2%)
- **valuediff_pp wins**: 9 (56.2%)
- **Ties**: 6 (37.5%)

| Task | vacor_kv (avg All-3) | valuediff_pp (avg All-3) | Winner |
| :--- | :---: | :---: | :---: |
| milestone_08 | 0.33 | 1.00 | valuediff_pp |
| milestone_09 | 0.67 | 1.00 | valuediff_pp |
| milestone_10 | 0.00 | 0.00 | tie |
| milestone_11 | 1.00 | 1.00 | tie |
| milestone_12 | 0.67 | 1.00 | valuediff_pp |
| milestone_13 | 0.67 | 1.00 | valuediff_pp |
| milestone_14 | 0.67 | 0.00 | vacor_kv |
| milestone_15 | 0.00 | 1.00 | valuediff_pp |
| milestone_16 | 1.00 | 1.00 | tie |
| milestone_17 | 0.67 | 1.00 | valuediff_pp |
| milestone_18 | 1.00 | 1.00 | tie |
| milestone_19 | 0.00 | 1.00 | valuediff_pp |
| milestone_20 | 0.67 | 1.00 | valuediff_pp |
| milestone_21 | 1.00 | 1.00 | tie |
| milestone_22 | 1.00 | 1.00 | tie |
| milestone_23 | 0.67 | 1.00 | valuediff_pp |

## Budget 512

| Mode | Tasks | Seeds | M1 (%) | M2 (%) | M3 (%) | All-3 (%) | 95% CI All-3 | Final (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dense* | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 80.6%, 100.0%] |  93.8 ± n/a |
| recency_pp | 16 | 1 |   0.0 ± n/a |  12.5 ± n/a |   6.2 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| snapkv_pp | 16 | 1 |  31.2 ± n/a |   0.0 ± n/a |  93.8 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| rkv_pp | 16 | 1 |  37.5 ± n/a |  12.5 ± n/a |  31.2 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| valuediff_pp | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 80.6%, 100.0%] |  93.8 ± n/a |
| valuediff_smooth_pp | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 80.6%, 100.0%] |  93.8 ± n/a |
| vase_pp | 16 | 3 |  60.4 ± 13.0 |  47.9 ± 18.0 | 100.0 ±  0.0 |  22.9 ± 14.4 | [ 13.3%,  36.5%] |  22.9 ± 14.4 |
| random_pp | 16 | 3 |   0.0 ±  0.0 |  10.4 ±  3.6 |  54.2 ± 21.9 |   0.0 ±  0.0 | [  0.0%,   7.4%] |   0.0 ±  0.0 |
| vacor_kv | 16 | 3 | 100.0 ±  0.0 | 100.0 ±  0.0 |  97.9 ±  3.6 |  97.9 ±  3.6 | [ 89.1%,  99.6%] |  87.5 ±  0.0 |
| vacor_pin | 16 | 3 | 100.0 ±  0.0 | 100.0 ±  0.0 |  89.6 ±  3.6 |  89.6 ±  3.6 | [ 77.8%,  95.5%] |  79.2 ±  3.6 |
| vacor_antithetic_only | 16 | 3 |  77.1 ±  3.6 |  66.7 ±  9.5 |  37.5 ± 10.8 |  18.8 ±  6.2 | [ 10.2%,  31.9%] |  16.7 ±  3.6 |

\* *dense evaluated with full KV cache (budget 0 / no eviction) as reference.*

### Paired Comparison: vacor_kv vs valuediff_pp on All-3 (Budget 512)

- **Evaluated Tasks**: 16
- **vacor_kv wins**: 0 (0.0%)
- **valuediff_pp wins**: 1 (6.2%)
- **Ties**: 15 (93.8%)

| Task | vacor_kv (avg All-3) | valuediff_pp (avg All-3) | Winner |
| :--- | :---: | :---: | :---: |
| milestone_08 | 1.00 | 1.00 | tie |
| milestone_09 | 1.00 | 1.00 | tie |
| milestone_10 | 1.00 | 1.00 | tie |
| milestone_11 | 1.00 | 1.00 | tie |
| milestone_12 | 1.00 | 1.00 | tie |
| milestone_13 | 1.00 | 1.00 | tie |
| milestone_14 | 1.00 | 1.00 | tie |
| milestone_15 | 1.00 | 1.00 | tie |
| milestone_16 | 1.00 | 1.00 | tie |
| milestone_17 | 1.00 | 1.00 | tie |
| milestone_18 | 1.00 | 1.00 | tie |
| milestone_19 | 0.67 | 1.00 | valuediff_pp |
| milestone_20 | 1.00 | 1.00 | tie |
| milestone_21 | 1.00 | 1.00 | tie |
| milestone_22 | 1.00 | 1.00 | tie |
| milestone_23 | 1.00 | 1.00 | tie |

## Overall Paired Comparison: vacor_kv vs valuediff_pp on All-3 (All Budgets Pooled)

- **Total (Budget, Task) Pairs**: 32
- **vacor_kv wins**: 1 (3.1%)
- **valuediff_pp wins**: 10 (31.2%)
- **Ties**: 21 (65.6%)
