# Suite A Extended Evaluation Summary (`smoke`)

- **Total Records**: 11
- **Model**: Qwen/Qwen3-1.7B

## Budget 384

| Mode | Tasks | Seeds | M1 (%) | M2 (%) | M3 (%) | All-3 (%) | 95% CI All-3 | Final (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| dense* | 1 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 20.7%, 100.0%] | 100.0 ± n/a |
| recency_pp | 1 | 1 |   0.0 ± n/a |   0.0 ± n/a |   0.0 ± n/a |   0.0 ± n/a | [  0.0%,  79.3%] |   0.0 ± n/a |
| snapkv_pp | 1 | 1 |   0.0 ± n/a |   0.0 ± n/a | 100.0 ± n/a |   0.0 ± n/a | [  0.0%,  79.3%] |   0.0 ± n/a |
| rkv_pp | 1 | 1 |   0.0 ± n/a |   0.0 ± n/a | 100.0 ± n/a |   0.0 ± n/a | [  0.0%,  79.3%] |   0.0 ± n/a |
| valuediff_pp | 1 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 20.7%, 100.0%] | 100.0 ± n/a |
| valuediff_smooth_pp | 1 | 1 | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | 100.0 ± n/a | [ 20.7%, 100.0%] | 100.0 ± n/a |
| vase_pp | 1 | 1 | 100.0 ± n/a | 100.0 ± n/a |   0.0 ± n/a |   0.0 ± n/a | [  0.0%,  79.3%] |   0.0 ± n/a |
| random_pp | 1 | 1 |   0.0 ± n/a |   0.0 ± n/a |   0.0 ± n/a |   0.0 ± n/a | [  0.0%,  79.3%] | 100.0 ± n/a |
| vacor_kv | 1 | 1 | 100.0 ± n/a | 100.0 ± n/a |   0.0 ± n/a |   0.0 ± n/a | [  0.0%,  79.3%] |   0.0 ± n/a |
| vacor_pin | 1 | 1 | 100.0 ± n/a | 100.0 ± n/a |   0.0 ± n/a |   0.0 ± n/a | [  0.0%,  79.3%] |   0.0 ± n/a |
| vacor_antithetic_only | 1 | 1 |   0.0 ± n/a |   0.0 ± n/a |   0.0 ± n/a |   0.0 ± n/a | [  0.0%,  79.3%] |   0.0 ± n/a |

\* *dense evaluated with full KV cache (budget 0 / no eviction) as reference.*

### Paired Comparison: vacor_kv vs valuediff_pp on All-3 (Budget 384)

- **Evaluated Tasks**: 1
- **vacor_kv wins**: 0 (0.0%)
- **valuediff_pp wins**: 1 (100.0%)
- **Ties**: 0 (0.0%)

| Task | vacor_kv (avg All-3) | valuediff_pp (avg All-3) | Winner |
| :--- | :---: | :---: | :---: |
| milestone_08 | 0.00 | 1.00 | valuediff_pp |
