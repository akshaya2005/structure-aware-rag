# Index comparison: recall@5

Questions: 923. Baseline: plain_flat. Seed: 0.

| System | Score | Difference vs baseline (95% CI) | Search ms/query | Index bytes |
|---|---:|---:|---:|---:|
| plain_bm25 | 0.4870 | -0.0099 [-0.0442, +0.0219] | 0.064 | 21168603 |
| plain_flat | 0.4969 | +0.0000 [+0.0000, +0.0000] | 0.022 | 17664357 |
| plain_hnsw | 0.4969 | +0.0000 [+0.0000, +0.0000] | 0.032 | 20803650 |
| contextual_bm25 | 0.4966 | -0.0003 [-0.0338, +0.0310] | 0.072 | 22055591 |
| contextual_flat | 0.5313 | +0.0343 [+0.0107, +0.0577] | 0.023 | 17664357 |
| contextual_hnsw | 0.5313 | +0.0343 [+0.0107, +0.0577] | 0.032 | 20803650 |

Intervals use paired bootstrap resampling by paper (question context for HotpotQA).
Small samples give unstable intervals. These timings exclude query encoding and generation.
BM25 uses the team's existing implementation; see the sparse audit for reference differences.
