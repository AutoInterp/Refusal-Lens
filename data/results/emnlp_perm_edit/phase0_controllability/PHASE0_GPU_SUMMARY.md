# Phase 0 GPU Outputs Summary

## 0b-simple — controllability audit

Bare-refuse → COMPLY flip rate per variant (predicts H0-1, H0-2, H0-3):

| Variant | bare flip rate | 95% CI | n |
|---|---:|---:|---:|
| ablate_features_pos | 10.0% | [4.3, 21.4] | 5/50 |
| ablate_features_neg | 8.0% | [3.2, 18.8] | 4/50 |
| ablate_features_all | 8.0% | [3.2, 18.8] | 4/50 |
| ablate_embeddings_all | 6.0% | [2.1, 16.2] | 3/50 |
| ablate_errors_all | 8.0% | [3.2, 18.8] | 4/50 |
| ablate_all_edges | 6.0% | [2.1, 16.2] | 3/50 |
| ablate_all_2x | 6.0% | [2.1, 16.2] | 3/50 |

### Per JB-class flip rate (H0-1 controllability)

| Variant | jb_fictio | jb_rolepl | jb_analyt | jb_comple | jb_cognit |
|---|---:|---:|---:|---:|---:|
| ablate_features_pos | 0.0% (0/19) | 0.0% (0/9) | 0.0% (0/28) | 0.0% (0/0) | 3.0% (1/33) |
| ablate_features_neg | 0.0% (0/19) | 11.1% (1/9) | 0.0% (0/28) | 0.0% (0/0) | 6.1% (2/33) |
| ablate_features_all | 0.0% (0/19) | 0.0% (0/9) | 0.0% (0/28) | 0.0% (0/0) | 3.0% (1/33) |
| ablate_embeddings_all | 5.3% (1/19) | 22.2% (2/9) | 0.0% (0/28) | 0.0% (0/0) | 6.1% (2/33) |
| ablate_errors_all | 0.0% (0/19) | 0.0% (0/9) | 0.0% (0/28) | 0.0% (0/0) | 3.0% (1/33) |
| ablate_all_edges | 5.3% (1/19) | 22.2% (2/9) | 3.6% (1/28) | 0.0% (0/0) | 6.1% (2/33) |
| ablate_all_2x | 5.3% (1/19) | 33.3% (3/9) | 3.6% (1/28) | 0.0% (0/0) | 6.1% (2/33) |

## 0d — top-K feature Pareto sweep (H0-6)

Bare-refuse flip rate per (variant, K):

| Variant\K | 1 | 5 | 10 | 20 | 50 | 100 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| abs | 8.0% | 8.0% | 8.0% | 8.0% | 8.0% | 8.0% | 8.0% |
| neg | 8.0% | 8.0% | 8.0% | 8.0% | 8.0% | 8.0% | 8.0% |
| pos | 8.0% | 10.0% | 10.0% | 10.0% | 10.0% | 10.0% | 8.0% |

## 0e — top-K edge Pareto sweep (H0-7)

Bare-refuse flip rate per (variant, K):

| Variant\K | 1 | 5 | 10 | 50 | 100 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| abs | 8.0% | 8.0% | 6.0% | 6.0% | 6.0% | 6.0% | 6.0% |
| neg | 8.0% | 8.0% | 6.0% | 6.0% | 6.0% | 6.0% | 6.0% |
| pos | 8.0% | 10.0% | 10.0% | 10.0% | 10.0% | 6.0% | 6.0% |