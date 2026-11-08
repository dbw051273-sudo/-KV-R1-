# MATH-500 Subset Evaluation Summary

- **Total Records**: 42
- **Evaluated Problems**: 3

| Mode | Budget | Evaluated Runs | Accuracy (Mean ± Std %) | 95% Wilson CI (%) | Throughput (tok/s) | Eviction Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense` | Full   |   9 (seeds=3) |  66.7%        | [ 35.4%,  87.9%] |  19.7 |  0.00 |
| `valuediff_pp` | 256    |   9 (seeds=3) |  66.7%        | [ 35.4%,  87.9%] |  19.7 |  5.93 |
| `snapkv_pp` | 256    |   9 (seeds=3) |  66.7%        | [ 35.4%,  87.9%] |  19.4 |  1.12 |
| `random_pp` | 256    |   9 (seeds=3) |  66.7%        | [ 35.4%,  87.9%] |  19.6 |  0.55 |
| `vacor_kv` | 256    |   6 (seeds=3) |  50.0%        | [ 18.8%,  81.2%] |  19.6 |  1.02 |
