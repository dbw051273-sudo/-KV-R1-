# GSM8K Evaluation Summary

- **Total Records**: 300
- **Evaluated Problems**: 20

| Mode | Budget | Evaluated Runs | Accuracy (Mean ± Std %) | 95% Wilson CI (%) | Throughput (tok/s) | Eviction Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense` | Full   |  60 (seeds=3) |  55.0%        | [ 42.5%,  66.9%] |  19.2 |  0.00 |
| `valuediff_pp` | 256    |  60 (seeds=3) |  55.0%        | [ 42.5%,  66.9%] |  19.3 |  1.98 |
| `snapkv_pp` | 256    |  60 (seeds=3) |  55.0%        | [ 42.5%,  66.9%] |  19.1 |  0.49 |
| `random_pp` | 256    |  60 (seeds=3) |  55.0%        | [ 42.5%,  66.9%] |  19.3 |  0.30 |
| `vacor_kv` | 256    |  60 (seeds=3) |  55.0%        | [ 42.5%,  66.9%] |  19.3 |  0.56 |
