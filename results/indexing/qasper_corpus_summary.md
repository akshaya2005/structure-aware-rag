# Index comparison: recall@5

Questions: 923. Baseline: plain_flat. Seed: 0.

| System | Score | Difference vs baseline (95% CI) | Search ms/query | Index bytes |
|---|---:|---:|---:|---:|
| plain_bm25 | 0.1603 | +0.0381 [+0.0164, +0.0598] | 13.265 | 10099233 |
| plain_flat | 0.1222 | +0.0000 [+0.0000, +0.0000] | 0.448 | 17651757 |
| plain_hnsw | 0.1222 | +0.0000 [+0.0000, +0.0000] | 0.151 | 20776722 |
| contextual_bm25 | 0.1803 | +0.0581 [+0.0379, +0.0795] | 14.182 | 11084008 |
| contextual_flat | 0.1070 | -0.0152 [-0.0323, +0.0026] | 0.456 | 17651757 |
| contextual_hnsw | 0.1032 | -0.0189 [-0.0358, -0.0007] | 0.126 | 20776722 |

Intervals use paired bootstrap resampling by paper (question context for HotpotQA).
Small samples give unstable intervals. These timings exclude query encoding and generation.
BM25 uses the team's existing implementation; see the sparse audit for reference differences.
