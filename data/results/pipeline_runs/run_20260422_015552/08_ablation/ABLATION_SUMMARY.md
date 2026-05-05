# Stage 08 Subcircuit Ablation — Summary

**Method**: zero-ablation of transcoder features via `ReplacementModel.feature_intervention_generate`.
**Elapsed**: 859.2 min.
**Positions modes**: all, anchors.

## How to read these numbers (load-bearing context)

- The transcoder only decomposes the MLP path. Stage 03 found that **MLP carries ~0.02% of the refusal signal at L15 measurement** (the rest is attention + embeddings). So `universal_refusal_core` is best read as a **ceiling probe on MLP-only ablation**, not a positive control: bare refusal can stay intact even with all 116 universal MLP features ablated, because attention-mediated refusal remains.
- `recovery_rate` = baseline COMPLY → ablated REFUSE. `break_rate` = baseline REFUSE → ablated COMPLY.
- The Stage 07 ctrl-aware rules (`ctrl_shared_refusal`, `jb_*_specific_vs_ctrl`) are **corpus-aggregated top-50 set logic**. A feature can fire on individual prompts during JB inference even when it isn't in the JB's corpus top-50. See the activation audit section for per-prompt hit rates and attribution magnitudes.

## Per-ablation results

### `universal_refusal_core` (116 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 46 | 4 | 40 | 100.0% | 21.7% |
| `jb_roleplay` | 40 | 10 | 34 | 40.0% | 25.0% |
| `ctrl_roleplay` | 48 | 2 | 41 | 100.0% | 18.8% |
| `jb_fiction` | 30 | 20 | 25 | 0.0% | 16.7% |
| `ctrl_fiction` | 48 | 2 | 44 | 100.0% | 12.5% |
| `jb_analytical` | 21 | 29 | 13 | 6.9% | 47.6% |
| `ctrl_analytical` | 45 | 5 | 42 | 80.0% | 15.6% |
| `jb_completion` | 48 | 2 | 48 | 100.0% | 4.2% |
| `ctrl_completion` | 45 | 5 | 41 | 100.0% | 20.0% |
| `jb_cognitive_reframe` | 17 | 33 | 13 | 6.1% | 35.3% |
| `ctrl_cognitive_reframe` | 42 | 8 | 38 | 100.0% | 28.6% |

**Positions: anchors**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 42 | 3 | 35 | 66.7% | 21.4% |
| `jb_roleplay` | 37 | 8 | 39 | 62.5% | 8.1% |
| `ctrl_roleplay` | 43 | 2 | 37 | 100.0% | 18.6% |
| `jb_fiction` | 26 | 19 | 27 | 10.5% | 3.8% |
| `ctrl_fiction` | 43 | 2 | 41 | 100.0% | 9.3% |
| `jb_analytical` | 18 | 27 | 16 | 18.5% | 38.9% |
| `ctrl_analytical` | 40 | 5 | 36 | 80.0% | 20.0% |
| `jb_completion` | 43 | 2 | 44 | 100.0% | 2.3% |
| `ctrl_completion` | 40 | 5 | 38 | 100.0% | 17.5% |
| `jb_cognitive_reframe` | 16 | 29 | 13 | 0.0% | 18.8% |
| `ctrl_cognitive_reframe` | 37 | 8 | 35 | 75.0% | 21.6% |

### `ctrl_shared_refusal` (50 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 46 | 4 | 48 | 75.0% | 2.2% |
| `jb_roleplay` | 40 | 10 | 42 | 40.0% | 5.0% |
| `ctrl_roleplay` | 48 | 2 | 47 | 50.0% | 4.2% |
| `jb_fiction` | 30 | 20 | 29 | 5.0% | 6.7% |
| `ctrl_fiction` | 48 | 2 | 47 | 50.0% | 4.2% |
| `jb_analytical` | 21 | 29 | 22 | 13.8% | 14.3% |
| `ctrl_analytical` | 45 | 5 | 48 | 80.0% | 2.2% |
| `jb_completion` | 48 | 2 | 47 | 100.0% | 6.2% |
| `ctrl_completion` | 45 | 5 | 48 | 100.0% | 4.4% |
| `jb_cognitive_reframe` | 17 | 33 | 18 | 9.1% | 11.8% |
| `ctrl_cognitive_reframe` | 42 | 8 | 46 | 62.5% | 2.4% |

**Positions: anchors**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 42 | 3 | 43 | 66.7% | 2.4% |
| `jb_roleplay` | 37 | 8 | 37 | 37.5% | 8.1% |
| `ctrl_roleplay` | 43 | 2 | 43 | 0.0% | 0.0% |
| `jb_fiction` | 26 | 19 | 26 | 0.0% | 0.0% |
| `ctrl_fiction` | 43 | 2 | 43 | 0.0% | 0.0% |
| `jb_analytical` | 18 | 27 | 21 | 18.5% | 11.1% |
| `ctrl_analytical` | 40 | 5 | 40 | 80.0% | 10.0% |
| `jb_completion` | 43 | 2 | 42 | 50.0% | 4.7% |
| `ctrl_completion` | 40 | 5 | 42 | 60.0% | 2.5% |
| `jb_cognitive_reframe` | 16 | 29 | 16 | 3.4% | 6.2% |
| `ctrl_cognitive_reframe` | 37 | 8 | 39 | 50.0% | 5.4% |

### `jb_fiction_specific_vs_ctrl` (52 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 46 | 4 | 46 | 50.0% | 4.3% |
| `jb_roleplay` | 40 | 10 | 40 | 30.0% | 7.5% |
| `ctrl_roleplay` | 48 | 2 | 49 | 100.0% | 2.1% |
| `jb_fiction` | 30 | 20 | 27 | 0.0% | 10.0% |
| `ctrl_fiction` | 48 | 2 | 48 | 50.0% | 2.1% |
| `jb_analytical` | 21 | 29 | 18 | 6.9% | 23.8% |
| `ctrl_analytical` | 45 | 5 | 46 | 80.0% | 6.7% |
| `jb_completion` | 48 | 2 | 45 | 0.0% | 6.2% |
| `ctrl_completion` | 45 | 5 | 46 | 80.0% | 6.7% |
| `jb_cognitive_reframe` | 17 | 33 | 20 | 9.1% | 0.0% |
| `ctrl_cognitive_reframe` | 42 | 8 | 40 | 37.5% | 11.9% |

**Positions: anchors**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 42 | 3 | 42 | 33.3% | 2.4% |
| `jb_roleplay` | 37 | 8 | 36 | 12.5% | 5.4% |
| `ctrl_roleplay` | 43 | 2 | 44 | 100.0% | 2.3% |
| `jb_fiction` | 26 | 19 | 24 | 0.0% | 7.7% |
| `ctrl_fiction` | 43 | 2 | 42 | 0.0% | 2.3% |
| `jb_analytical` | 18 | 27 | 16 | 3.7% | 16.7% |
| `ctrl_analytical` | 40 | 5 | 40 | 80.0% | 10.0% |
| `jb_completion` | 43 | 2 | 42 | 50.0% | 4.7% |
| `ctrl_completion` | 40 | 5 | 43 | 100.0% | 5.0% |
| `jb_cognitive_reframe` | 16 | 29 | 15 | 3.4% | 12.5% |
| `ctrl_cognitive_reframe` | 37 | 8 | 39 | 37.5% | 2.7% |

### `jb_analytical_specific_vs_ctrl` (69 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 42 | 3 | 37 | 66.7% | 16.7% |
| `jb_roleplay` | 37 | 8 | 35 | 50.0% | 16.2% |
| `ctrl_roleplay` | 43 | 2 | 42 | 100.0% | 7.0% |
| `jb_fiction` | 26 | 19 | 26 | 0.0% | 0.0% |
| `ctrl_fiction` | 43 | 2 | 42 | 0.0% | 2.3% |
| `jb_analytical` | 18 | 27 | 19 | 18.5% | 22.2% |
| `ctrl_analytical` | 40 | 5 | 41 | 100.0% | 10.0% |
| `jb_completion` | 43 | 2 | 41 | 50.0% | 7.0% |
| `ctrl_completion` | 40 | 5 | 36 | 60.0% | 17.5% |
| `jb_cognitive_reframe` | 16 | 29 | 13 | 3.4% | 25.0% |
| `ctrl_cognitive_reframe` | 37 | 8 | 37 | 37.5% | 8.1% |

**Positions: anchors**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 42 | 3 | 40 | 33.3% | 7.1% |
| `jb_roleplay` | 37 | 8 | 35 | 12.5% | 8.1% |
| `ctrl_roleplay` | 43 | 2 | 45 | 100.0% | 0.0% |
| `jb_fiction` | 26 | 19 | 25 | 0.0% | 3.8% |
| `ctrl_fiction` | 43 | 2 | 43 | 0.0% | 0.0% |
| `jb_analytical` | 18 | 27 | 17 | 3.7% | 11.1% |
| `ctrl_analytical` | 40 | 5 | 40 | 40.0% | 5.0% |
| `jb_completion` | 43 | 2 | 44 | 50.0% | 0.0% |
| `ctrl_completion` | 40 | 5 | 39 | 40.0% | 7.5% |
| `jb_cognitive_reframe` | 16 | 29 | 13 | 0.0% | 18.8% |
| `ctrl_cognitive_reframe` | 37 | 8 | 38 | 25.0% | 2.7% |

### `jb_cognitive_reframe_specific_vs_ctrl` (88 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 42 | 3 | 39 | 66.7% | 11.9% |
| `jb_roleplay` | 37 | 8 | 38 | 50.0% | 8.1% |
| `ctrl_roleplay` | 43 | 2 | 44 | 100.0% | 2.3% |
| `jb_fiction` | 26 | 19 | 27 | 5.3% | 0.0% |
| `ctrl_fiction` | 43 | 2 | 42 | 0.0% | 2.3% |
| `jb_analytical` | 18 | 27 | 13 | 3.7% | 33.3% |
| `ctrl_analytical` | 40 | 5 | 42 | 80.0% | 5.0% |
| `jb_completion` | 43 | 2 | 43 | 50.0% | 2.3% |
| `ctrl_completion` | 40 | 5 | 42 | 80.0% | 5.0% |
| `jb_cognitive_reframe` | 16 | 29 | 17 | 6.9% | 6.2% |
| `ctrl_cognitive_reframe` | 37 | 8 | 39 | 75.0% | 10.8% |

**Positions: anchors**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 42 | 3 | 42 | 33.3% | 2.4% |
| `jb_roleplay` | 37 | 8 | 37 | 25.0% | 5.4% |
| `ctrl_roleplay` | 43 | 2 | 40 | 50.0% | 9.3% |
| `jb_fiction` | 26 | 19 | 29 | 15.8% | 0.0% |
| `ctrl_fiction` | 43 | 2 | 42 | 0.0% | 2.3% |
| `jb_analytical` | 18 | 27 | 18 | 7.4% | 11.1% |
| `ctrl_analytical` | 40 | 5 | 37 | 40.0% | 12.5% |
| `jb_completion` | 43 | 2 | 42 | 0.0% | 2.3% |
| `ctrl_completion` | 40 | 5 | 41 | 80.0% | 7.5% |
| `jb_cognitive_reframe` | 16 | 29 | 14 | 0.0% | 12.5% |
| `ctrl_cognitive_reframe` | 37 | 8 | 40 | 37.5% | 0.0% |

## Dissociation (class-specific ablations)

Target class's own JB recovery vs. average across other classes.
Positive `dissociation_delta` = class-selective patching.

| Ablation | Mode | Target class | Target recovery | Others avg | Δ |
|---|---|---|---|---|---|
| `jb_fiction_specific_vs_ctrl` | all | fiction | 0.0% | 11.5% | -11.5pp |
| `jb_fiction_specific_vs_ctrl` | anchors | fiction | 0.0% | 17.4% | -17.4pp |
| `jb_analytical_specific_vs_ctrl` | all | analytical | 18.5% | 25.9% | -7.4pp |
| `jb_analytical_specific_vs_ctrl` | anchors | analytical | 3.7% | 15.6% | -11.9pp |
| `jb_cognitive_reframe_specific_vs_ctrl` | all | cognitive_reframe | 6.9% | 27.3% | -20.4pp |
| `jb_cognitive_reframe_specific_vs_ctrl` | anchors | cognitive_reframe | 0.0% | 12.0% | -12.0pp |

## Activation audit (Stage 02 attribution data)

Per-ablation, per-condition-class top-50 hit rate and mean |attribution|. Diagnoses whether the Stage 07 set logic produces a clean per-prompt separation, and whether class-specific subcircuits are correlationally selective for their target class.

### `universal_refusal_core` (116 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 31.21% | 0.01507 | 0/116 |
| `jb_*` | 33.92% | 0.01558 | 5/116 |
| `ctrl_*` | 37.46% | 0.01995 | 5/116 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 28.60% | 0.01235 |
| `jb_cognitive_reframe` | 32.10% | 0.01447 |
| `jb_completion` | 36.50% | 0.01995 |
| `jb_fiction` | 34.64% | 0.01348 |
| `jb_roleplay` | 37.74% | 0.01763 |

### `ctrl_shared_refusal` (50 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 15.16% | 0.00258 | 0/50 |
| `jb_*` | 9.28% | 0.00196 | 0/50 |
| `ctrl_*` | 13.26% | 0.00272 | 0/50 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 8.84% | 0.00161 |
| `jb_cognitive_reframe` | 10.36% | 0.00229 |
| `jb_completion` | 11.60% | 0.00263 |
| `jb_fiction` | 5.24% | 0.00103 |
| `jb_roleplay` | 10.36% | 0.00222 |

### `jb_fiction_specific_vs_ctrl` (52 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 2.23% | 0.00032 | 35/52 |
| `jb_*` | 3.46% | 0.00063 | 0/52 |
| `ctrl_*` | 0.27% | 0.00004 | 40/52 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 1.42% | 0.00031 |
| `jb_cognitive_reframe` | 1.65% | 0.00029 |
| `jb_completion` | 1.27% | 0.00021 |
| `jb_fiction` | 12.08% | 0.00218 |
| `jb_roleplay` | 0.88% | 0.00017 |

### `jb_analytical_specific_vs_ctrl` (69 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 2.75% | 0.00033 | 41/69 |
| `jb_*` | 3.69% | 0.00057 | 0/69 |
| `ctrl_*` | 0.49% | 0.00007 | 45/69 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 11.22% | 0.00174 |
| `jb_cognitive_reframe` | 3.48% | 0.00052 |
| `jb_completion` | 1.48% | 0.00026 |
| `jb_fiction` | 1.65% | 0.00026 |
| `jb_roleplay` | 0.61% | 0.00010 |

### `jb_cognitive_reframe_specific_vs_ctrl` (88 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 2.55% | 0.00036 | 48/88 |
| `jb_*` | 3.08% | 0.00068 | 0/88 |
| `ctrl_*` | 0.94% | 0.00015 | 60/88 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 2.43% | 0.00044 |
| `jb_cognitive_reframe` | 7.25% | 0.00124 |
| `jb_completion` | 1.59% | 0.00031 |
| `jb_fiction` | 1.64% | 0.00081 |
| `jb_roleplay` | 2.48% | 0.00058 |

