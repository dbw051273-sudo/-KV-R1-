# Suite A Extended Evaluation Summary (`q4b`)

- **Total Records**: 656
- **Model**: Qwen/Qwen3-4B

## Budget 384

| Mode | Tasks | Seeds | M1 (%) | M2 (%) | M3 (%) | All-3 (%) | 95% CI All-3 | Final (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dense* | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 80.6%, 100.0%] | 100.0 ± n/a |
| recency_pp | 16 | 1 |   0.0 ± n/a |   0.0 ± n/a |  12.5 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| snapkv_pp | 16 | 1 |   6.2 ± n/a |   6.2 ± n/a |  50.0 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| rkv_pp | 16 | 1 |   0.0 ± n/a |   6.2 ± n/a |  18.8 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| valuediff_pp | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 80.6%, 100.0%] | 100.0 ± n/a |
| valuediff_smooth_pp | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a |  68.8 ± n/a |  68.8 ± n/a | [ 44.4%,  85.8%] |  68.8 ± n/a |
| vase_pp | 16 | 3 |  16.7 ±  3.6 |  18.8 ±  6.2 |  31.2 ± 22.5 |   0.0 ±  0.0 | [  0.0%,   7.4%] |   0.0 ±  0.0 |
| random_pp | 16 | 3 |   0.0 ±  0.0 |   6.2 ±  0.0 |  12.5 ±  0.0 |   0.0 ±  0.0 | [  0.0%,   7.4%] |   0.0 ±  0.0 |
| vacor_kv | 16 | 3 | 100.0 ±  0.0 | 100.0 ±  0.0 |  81.2 ±  6.2 |  81.2 ±  6.2 | [ 68.1%,  89.8%] |  77.1 ±  7.2 |
| vacor_pin | 16 | 3 | 100.0 ±  0.0 |  85.4 ±  9.5 |  66.7 ± 15.7 |  54.2 ± 18.0 | [ 40.3%,  67.4%] |  52.1 ± 14.4 |
| vacor_antithetic_only | 16 | 3 |  50.0 ± 10.8 |  25.0 ± 12.5 |  16.7 ±  7.2 |   2.1 ±  3.6 | [  0.4%,  10.9%] |   0.0 ±  0.0 |

\* *dense evaluated with full KV cache (budget 0 / no eviction) as reference.*

### Paired Comparison: vacor_kv vs valuediff_pp on All-3 (Budget 384)

- **Evaluated Tasks**: 16
- **vacor_kv wins**: 0 (0.0%)
- **valuediff_pp wins**: 5 (31.2%)
- **Ties**: 11 (68.8%)

| Task | vacor_kv (avg All-3) | valuediff_pp (avg All-3) | Winner |
| :--- | :---: | :---: | :---: |
| milestone_08 | 1.00 | 1.00 | tie |
| milestone_09 | 0.67 | 1.00 | valuediff_pp |
| milestone_10 | 0.33 | 1.00 | valuediff_pp |
| milestone_11 | 1.00 | 1.00 | tie |
| milestone_12 | 1.00 | 1.00 | tie |
| milestone_13 | 1.00 | 1.00 | tie |
| milestone_14 | 0.33 | 1.00 | valuediff_pp |
| milestone_15 | 1.00 | 1.00 | tie |
| milestone_16 | 1.00 | 1.00 | tie |
| milestone_17 | 1.00 | 1.00 | tie |
| milestone_18 | 0.00 | 1.00 | valuediff_pp |
| milestone_19 | 1.00 | 1.00 | tie |
| milestone_20 | 1.00 | 1.00 | tie |
| milestone_21 | 0.67 | 1.00 | valuediff_pp |
| milestone_22 | 1.00 | 1.00 | tie |
| milestone_23 | 1.00 | 1.00 | tie |

## Budget 512

| Mode | Tasks | Seeds | M1 (%) | M2 (%) | M3 (%) | All-3 (%) | 95% CI All-3 | Final (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dense* | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 80.6%, 100.0%] | 100.0 ± n/a |
| recency_pp | 16 | 1 |   0.0 ± n/a |   6.2 ± n/a |  12.5 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| snapkv_pp | 16 | 1 |   6.2 ± n/a |   0.0 ± n/a | 100.0 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| rkv_pp | 16 | 1 |  37.5 ± n/a |  18.8 ± n/a |  12.5 ± n/a |   0.0 ± n/a | [  0.0%,  19.4%] |   0.0 ± n/a |
| valuediff_pp | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 80.6%, 100.0%] | 100.0 ± n/a |
| valuediff_smooth_pp | 16 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 80.6%, 100.0%] | 100.0 ± n/a |
| vase_pp | 16 | 3 |  64.6 ±  9.5 |  33.3 ±  9.5 |  97.9 ±  3.6 |  20.8 ±  7.2 | [ 11.7%,  34.3%] |  16.7 ±  9.5 |
| random_pp | 16 | 3 |   2.1 ±  3.6 |   4.2 ±  3.6 |  35.4 ±  9.5 |   0.0 ±  0.0 | [  0.0%,   7.4%] |   0.0 ±  0.0 |
| vacor_kv | 16 | 3 | 100.0 ±  0.0 | 100.0 ±  0.0 |  95.8 ±  3.6 |  95.8 ±  3.6 | [ 86.0%,  98.8%] |  95.8 ±  3.6 |
| vacor_pin | 16 | 3 | 100.0 ±  0.0 | 100.0 ±  0.0 |  87.5 ±  6.2 |  87.5 ±  6.2 | [ 75.3%,  94.1%] |  85.4 ±  3.6 |
| vacor_antithetic_only | 16 | 3 |  81.2 ±  6.2 |  54.2 ±  3.6 |  29.2 ±  7.2 |  12.5 ±  6.2 | [  5.9%,  24.7%] |  12.5 ±  6.2 |

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
| milestone_13 | 0.33 | 1.00 | valuediff_pp |
| milestone_14 | 1.00 | 1.00 | tie |
| milestone_15 | 1.00 | 1.00 | tie |
| milestone_16 | 1.00 | 1.00 | tie |
| milestone_17 | 1.00 | 1.00 | tie |
| milestone_18 | 1.00 | 1.00 | tie |
| milestone_19 | 1.00 | 1.00 | tie |
| milestone_20 | 1.00 | 1.00 | tie |
| milestone_21 | 1.00 | 1.00 | tie |
| milestone_22 | 1.00 | 1.00 | tie |
| milestone_23 | 1.00 | 1.00 | tie |

## Overall Paired Comparison: vacor_kv vs valuediff_pp on All-3 (All Budgets Pooled)

- **Total (Budget, Task) Pairs**: 32
- **vacor_kv wins**: 0 (0.0%)
- **valuediff_pp wins**: 6 (18.8%)
- **Ties**: 26 (81.2%)
