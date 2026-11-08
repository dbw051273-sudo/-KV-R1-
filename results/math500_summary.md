# MATH-500 Subset Evaluation Summary

- **Total Records**: 750
- **Evaluated Problems**: 50

| Mode | Budget | Evaluated Runs | Accuracy (Mean ± Std %) | 95% Wilson CI (%) | Throughput (tok/s) | Eviction Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense` | Full   | 150 (seeds=3) |  38.0%        | [ 30.6%,  46.0%] |  19.8 |  0.00 |
| `valuediff_pp` | 256    | 150 (seeds=3) |  38.0%        | [ 30.6%,  46.0%] |  19.8 |  0.86 |
| `snapkv_pp` | 256    | 150 (seeds=3) |  42.0%        | [ 34.4%,  50.0%] |  19.4 |  0.66 |
| `random_pp` | 256    | 150 (seeds=3) |  38.7% ±  1.2 | [ 31.2%,  46.7%] |  19.8 |  0.46 |
| `vacor_kv` | 256    | 150 (seeds=3) |  41.3% ±  2.3 | [ 33.8%,  49.3%] |  19.8 |  0.86 |
