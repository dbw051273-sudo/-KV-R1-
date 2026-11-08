# Reference baseline repositories (not vendored; clone into `baselines/`)

- R-KV: https://github.com/Zefan-Cai/R-KV
- SnapKV: https://github.com/FasterDecoding/SnapKV
- SkipKV: https://github.com/TTTTTTris/SkipKV
- TriAttention: https://github.com/WeianMao/triattention
- G-KV: https://github.com/microsoft/G-KV
- R1-Compress: https://github.com/w-yibo/R1-Compress
- VaSE: https://github.com/terarachang/VaSE
- Random-Attention: https://github.com/SalesforceAIResearch/Random-Attention

`src/kv_engine.py` is a from-scratch SDPA re-implementation of simplified variants
(`*_pp` = prompt-protected). They are NOT bit-exact ports of the official code.
