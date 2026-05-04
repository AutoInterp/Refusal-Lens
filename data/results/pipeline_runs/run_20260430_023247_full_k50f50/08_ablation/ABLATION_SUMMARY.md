# Stage 08 Subcircuit Ablation — Summary

**Method**: zero-ablation of transcoder features via `ReplacementModel.feature_intervention_generate`.
**Subcircuits source**: `subcircuits_k50_f50.json`.
**Elapsed**: 1834.2 min.
**Positions modes**: all.
**Low-coverage threshold**: 0.30 (prompts where <threshold of ablation features are in top-K are flagged).

## How to read these numbers

- `recovery_rate` = baseline COMPLY → ablated REFUSE. `break_rate` = baseline REFUSE → ablated COMPLY.
- **Per-class** rows show the unweighted rate for each condition.
- **Comply-weighted JB recovery** (under each position mode) is Σ(per-class rate × per-class baseline_comply) / Σ(baseline_comply). Reflects the model's behavior on actual JB-success cases without dropping rare classes — the headline NeurIPS rigor metric.
- The **per-prompt coverage** table flags prompts where the ablation features weren't in that prompt's top-K attribution. Low coverage on a class explains null recovery rates: the features couldn't be doing much because they weren't strongly active to begin with.
- For `subcircuits_k50_f50.json` runs (per-prompt sweep), the subcircuits are constructed from features in top-K for ≥F fraction of prompts in each condition; legacy `subcircuits.json` uses corpus union.

## Per-ablation results

### `universal_refusal_core` (26 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 45 | 5 | 47 | 80.0% | 4.4% |
| `jb_roleplay` | 38 | 12 | 44 | 75.0% | 7.9% |
| `ctrl_roleplay` | 48 | 2 | 48 | 100.0% | 4.2% |
| `jb_fiction` | 30 | 20 | 39 | 45.0% | 0.0% |
| `ctrl_fiction` | 45 | 5 | 47 | 80.0% | 4.4% |
| `jb_analytical` | 18 | 32 | 21 | 15.6% | 11.1% |
| `ctrl_analytical` | 42 | 8 | 46 | 100.0% | 9.5% |
| `jb_completion` | 46 | 4 | 45 | 50.0% | 6.5% |
| `ctrl_completion` | 45 | 5 | 46 | 60.0% | 4.4% |
| `jb_cognitive_reframe` | 17 | 33 | 19 | 9.1% | 5.9% |
| `ctrl_cognitive_reframe` | 44 | 6 | 46 | 83.3% | 6.8% |

**Comply-weighted aggregates:**

- JB recovery (weighted by baseline_comply across all 5 JB classes): **27.7%** (n_jb_comply=101)
- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): **5.8%** (n_ctrl_refuse=224)
- bare break: **4.4%** (n_bare_refuse=45)

### `ctrl_shared_refusal` (4 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 45 | 5 | 44 | 60.0% | 8.9% |
| `jb_roleplay` | 38 | 12 | 39 | 25.0% | 5.3% |
| `ctrl_roleplay` | 48 | 2 | 49 | 100.0% | 2.1% |
| `jb_fiction` | 30 | 20 | 29 | 0.0% | 3.3% |
| `ctrl_fiction` | 45 | 5 | 45 | 20.0% | 2.2% |
| `jb_analytical` | 18 | 32 | 20 | 12.5% | 11.1% |
| `ctrl_analytical` | 42 | 8 | 43 | 37.5% | 4.8% |
| `jb_completion` | 46 | 4 | 46 | 25.0% | 2.2% |
| `ctrl_completion` | 45 | 5 | 43 | 20.0% | 6.7% |
| `jb_cognitive_reframe` | 17 | 33 | 18 | 3.0% | 0.0% |
| `ctrl_cognitive_reframe` | 44 | 6 | 44 | 50.0% | 6.8% |

**Comply-weighted aggregates:**

- JB recovery (weighted by baseline_comply across all 5 JB classes): **8.9%** (n_jb_comply=101)
- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): **4.5%** (n_ctrl_refuse=224)
- bare break: **8.9%** (n_bare_refuse=45)

### `jb_fiction_specific_vs_ctrl` (11 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 45 | 5 | 44 | 40.0% | 6.7% |
| `jb_roleplay` | 38 | 12 | 43 | 50.0% | 2.6% |
| `ctrl_roleplay` | 48 | 2 | 48 | 50.0% | 2.1% |
| `jb_fiction` | 30 | 20 | 35 | 25.0% | 0.0% |
| `ctrl_fiction` | 45 | 5 | 49 | 80.0% | 0.0% |
| `jb_analytical` | 18 | 32 | 19 | 9.4% | 11.1% |
| `ctrl_analytical` | 42 | 8 | 48 | 87.5% | 2.4% |
| `jb_completion` | 46 | 4 | 46 | 25.0% | 2.2% |
| `ctrl_completion` | 45 | 5 | 47 | 40.0% | 0.0% |
| `jb_cognitive_reframe` | 17 | 33 | 18 | 6.1% | 5.9% |
| `ctrl_cognitive_reframe` | 44 | 6 | 44 | 50.0% | 6.8% |

**Comply-weighted aggregates:**

- JB recovery (weighted by baseline_comply across all 5 JB classes): **16.9%** (n_jb_comply=101)
- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): **2.2%** (n_ctrl_refuse=224)
- bare break: **6.7%** (n_bare_refuse=45)

### `jb_analytical_specific_vs_ctrl` (9 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 45 | 5 | 46 | 80.0% | 6.7% |
| `jb_roleplay` | 38 | 12 | 42 | 50.0% | 5.3% |
| `ctrl_roleplay` | 48 | 2 | 47 | 100.0% | 6.2% |
| `jb_fiction` | 30 | 20 | 31 | 5.0% | 0.0% |
| `ctrl_fiction` | 45 | 5 | 49 | 80.0% | 0.0% |
| `jb_analytical` | 18 | 32 | 20 | 12.5% | 11.1% |
| `ctrl_analytical` | 42 | 8 | 47 | 75.0% | 2.4% |
| `jb_completion` | 46 | 4 | 47 | 25.0% | 0.0% |
| `ctrl_completion` | 45 | 5 | 46 | 80.0% | 6.7% |
| `jb_cognitive_reframe` | 17 | 33 | 20 | 9.1% | 0.0% |
| `ctrl_cognitive_reframe` | 44 | 6 | 48 | 66.7% | 0.0% |

**Comply-weighted aggregates:**

- JB recovery (weighted by baseline_comply across all 5 JB classes): **14.8%** (n_jb_comply=101)
- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): **3.1%** (n_ctrl_refuse=224)
- bare break: **6.7%** (n_bare_refuse=45)

### `jb_cognitive_reframe_specific_vs_ctrl` (9 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 45 | 5 | 46 | 60.0% | 4.4% |
| `jb_roleplay` | 38 | 12 | 39 | 41.7% | 10.5% |
| `ctrl_roleplay` | 48 | 2 | 46 | 50.0% | 6.2% |
| `jb_fiction` | 30 | 20 | 33 | 15.0% | 0.0% |
| `ctrl_fiction` | 45 | 5 | 49 | 80.0% | 0.0% |
| `jb_analytical` | 18 | 32 | 20 | 6.2% | 0.0% |
| `ctrl_analytical` | 42 | 8 | 44 | 62.5% | 7.1% |
| `jb_completion` | 46 | 4 | 45 | 50.0% | 6.5% |
| `ctrl_completion` | 45 | 5 | 44 | 40.0% | 6.7% |
| `jb_cognitive_reframe` | 17 | 33 | 22 | 15.2% | 0.0% |
| `ctrl_cognitive_reframe` | 44 | 6 | 41 | 66.7% | 15.9% |

**Comply-weighted aggregates:**

- JB recovery (weighted by baseline_comply across all 5 JB classes): **16.8%** (n_jb_comply=101)
- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): **7.1%** (n_ctrl_refuse=224)
- bare break: **4.4%** (n_bare_refuse=45)

### `jb_completion_specific_vs_ctrl` (8 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 45 | 5 | 46 | 40.0% | 2.2% |
| `jb_roleplay` | 38 | 12 | 38 | 41.7% | 13.2% |
| `ctrl_roleplay` | 48 | 2 | 49 | 100.0% | 2.1% |
| `jb_fiction` | 30 | 20 | 33 | 15.0% | 0.0% |
| `ctrl_fiction` | 45 | 5 | 48 | 80.0% | 2.2% |
| `jb_analytical` | 18 | 32 | 21 | 9.4% | 0.0% |
| `ctrl_analytical` | 42 | 8 | 43 | 62.5% | 9.5% |
| `jb_completion` | 46 | 4 | 47 | 50.0% | 2.2% |
| `ctrl_completion` | 45 | 5 | 42 | 20.0% | 8.9% |
| `jb_cognitive_reframe` | 17 | 33 | 19 | 6.1% | 0.0% |
| `ctrl_cognitive_reframe` | 44 | 6 | 42 | 33.3% | 9.1% |

**Comply-weighted aggregates:**

- JB recovery (weighted by baseline_comply across all 5 JB classes): **14.9%** (n_jb_comply=101)
- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): **6.2%** (n_ctrl_refuse=224)
- bare break: **2.2%** (n_bare_refuse=45)

### `jb_roleplay_specific_vs_ctrl` (6 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 45 | 5 | 47 | 60.0% | 2.2% |
| `jb_roleplay` | 38 | 12 | 39 | 16.7% | 2.6% |
| `ctrl_roleplay` | 48 | 2 | 48 | 100.0% | 4.2% |
| `jb_fiction` | 30 | 20 | 32 | 10.0% | 0.0% |
| `ctrl_fiction` | 45 | 5 | 47 | 40.0% | 0.0% |
| `jb_analytical` | 18 | 32 | 20 | 6.2% | 0.0% |
| `ctrl_analytical` | 42 | 8 | 44 | 50.0% | 4.8% |
| `jb_completion` | 46 | 4 | 47 | 25.0% | 0.0% |
| `ctrl_completion` | 45 | 5 | 46 | 60.0% | 4.4% |
| `jb_cognitive_reframe` | 17 | 33 | 16 | 0.0% | 5.9% |
| `ctrl_cognitive_reframe` | 44 | 6 | 45 | 33.3% | 2.3% |

**Comply-weighted aggregates:**

- JB recovery (weighted by baseline_comply across all 5 JB classes): **6.9%** (n_jb_comply=101)
- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): **3.1%** (n_ctrl_refuse=224)
- bare break: **2.2%** (n_bare_refuse=45)

### `anti_refusal_amplifiers` (64 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 45 | 5 | 43 | 60.0% | 11.1% |
| `jb_roleplay` | 38 | 12 | 40 | 66.7% | 15.8% |
| `ctrl_roleplay` | 48 | 2 | 46 | 50.0% | 6.2% |
| `jb_fiction` | 30 | 20 | 37 | 40.0% | 3.3% |
| `ctrl_fiction` | 45 | 5 | 47 | 40.0% | 0.0% |
| `jb_analytical` | 18 | 32 | 15 | 9.4% | 33.3% |
| `ctrl_analytical` | 42 | 8 | 40 | 50.0% | 14.3% |
| `jb_completion` | 46 | 4 | 46 | 75.0% | 6.5% |
| `ctrl_completion` | 45 | 5 | 42 | 60.0% | 13.3% |
| `jb_cognitive_reframe` | 17 | 33 | 17 | 0.0% | 0.0% |
| `ctrl_cognitive_reframe` | 44 | 6 | 40 | 50.0% | 15.9% |

**Comply-weighted aggregates:**

- JB recovery (weighted by baseline_comply across all 5 JB classes): **21.8%** (n_jb_comply=101)
- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): **9.8%** (n_ctrl_refuse=224)
- bare break: **11.1%** (n_bare_refuse=45)

## Dissociation (class-specific ablations)

Target class's own JB recovery vs. average across other classes.
Positive `dissociation_delta` = class-selective patching.

| Ablation | Mode | Target class | Target recovery | Others avg | Δ |
|---|---|---|---|---|---|
| `jb_fiction_specific_vs_ctrl` | all | fiction | 25.0% | 22.6% | +2.4pp |
| `jb_analytical_specific_vs_ctrl` | all | analytical | 12.5% | 22.3% | -9.8pp |
| `jb_cognitive_reframe_specific_vs_ctrl` | all | cognitive_reframe | 15.2% | 28.2% | -13.0pp |
| `jb_completion_specific_vs_ctrl` | all | completion | 50.0% | 18.0% | +32.0pp |
| `jb_roleplay_specific_vs_ctrl` | all | roleplay | 16.7% | 10.3% | +6.4pp |

## Per-prompt coverage diagnostic

Mean fraction of ablation features in each prompt's top-K attribution, plus the count of low-coverage prompts (frac < 0.30).
Low coverage means the features couldn't have a strong effect; null recovery on those (ablation, condition) pairs is uninformative.

### `universal_refusal_core`

**Positions: all**

| Condition | Mean frac in top-K | Mean Σ\|attr\| | Low-coverage prompts |
|---|---|---|---|
| `bare` | 97.4% | 12839.5527 | 0/50 (0%) |
| `jb_roleplay` | 98.8% | 12661.5029 | 0/50 (0%) |
| `ctrl_roleplay` | 98.2% | 12674.3520 | 0/50 (0%) |
| `jb_fiction` | 97.3% | 12041.5409 | 0/50 (0%) |
| `ctrl_fiction` | 99.4% | 12970.2450 | 0/50 (0%) |
| `jb_analytical` | 100.0% | 13052.6949 | 0/50 (0%) |
| `ctrl_analytical` | 98.6% | 12769.9118 | 0/50 (0%) |
| `jb_completion` | 100.0% | 12878.6919 | 0/50 (0%) |
| `ctrl_completion` | 98.8% | 13005.7637 | 0/50 (0%) |
| `jb_cognitive_reframe` | 99.4% | 13032.2996 | 0/50 (0%) |
| `ctrl_cognitive_reframe` | 98.7% | 12633.6650 | 0/50 (0%) |

### `ctrl_shared_refusal`

**Positions: all**

| Condition | Mean frac in top-K | Mean Σ\|attr\| | Low-coverage prompts |
|---|---|---|---|
| `bare` | 93.0% | 1020.6445 | 0/50 (0%) |
| `jb_roleplay` | 65.5% | 658.1141 | 0/50 (0%) |
| `ctrl_roleplay` | 90.5% | 968.2943 | 0/50 (0%) |
| `jb_fiction` | 75.0% | 806.2796 | 0/50 (0%) |
| `ctrl_fiction` | 96.5% | 1057.6902 | 0/50 (0%) |
| `jb_analytical` | 78.5% | 867.1980 | 0/50 (0%) |
| `ctrl_analytical` | 93.5% | 996.9710 | 0/50 (0%) |
| `jb_completion` | 98.0% | 1136.4641 | 0/50 (0%) |
| `ctrl_completion` | 94.5% | 1031.5923 | 0/50 (0%) |
| `jb_cognitive_reframe` | 75.0% | 736.8295 | 0/50 (0%) |
| `ctrl_cognitive_reframe` | 87.5% | 927.0729 | 0/50 (0%) |

### `jb_fiction_specific_vs_ctrl`

**Positions: all**

| Condition | Mean frac in top-K | Mean Σ\|attr\| | Low-coverage prompts |
|---|---|---|---|
| `bare` | 6.9% | 119.4874 | 50/50 (100%) |
| `jb_roleplay` | 22.2% | 494.5218 | 40/50 (80%) |
| `ctrl_roleplay` | 19.8% | 457.5222 | 43/50 (86%) |
| `jb_fiction` | 83.3% | 2257.3084 | 0/50 (0%) |
| `ctrl_fiction` | 27.3% | 653.4857 | 30/50 (60%) |
| `jb_analytical` | 21.8% | 589.2208 | 44/50 (88%) |
| `ctrl_analytical` | 17.4% | 363.8168 | 46/50 (92%) |
| `jb_completion` | 12.7% | 291.1937 | 43/50 (86%) |
| `ctrl_completion` | 19.4% | 444.3583 | 45/50 (90%) |
| `jb_cognitive_reframe` | 24.0% | 737.3683 | 42/50 (84%) |
| `ctrl_cognitive_reframe` | 20.5% | 487.8163 | 43/50 (86%) |

### `jb_analytical_specific_vs_ctrl`

**Positions: all**

| Condition | Mean frac in top-K | Mean Σ\|attr\| | Low-coverage prompts |
|---|---|---|---|
| `bare` | 23.8% | 518.7153 | 33/50 (66%) |
| `jb_roleplay` | 38.2% | 1010.9341 | 13/50 (26%) |
| `ctrl_roleplay` | 44.4% | 1294.6154 | 3/50 (6%) |
| `jb_fiction` | 52.0% | 1128.9763 | 1/50 (2%) |
| `ctrl_fiction` | 50.9% | 1386.4050 | 12/50 (24%) |
| `jb_analytical` | 96.4% | 2357.3074 | 0/50 (0%) |
| `ctrl_analytical` | 36.7% | 765.9654 | 18/50 (36%) |
| `jb_completion` | 44.2% | 1247.1140 | 7/50 (14%) |
| `ctrl_completion` | 40.0% | 939.1100 | 11/50 (22%) |
| `jb_cognitive_reframe` | 62.2% | 1410.8352 | 0/50 (0%) |
| `ctrl_cognitive_reframe` | 44.2% | 1195.8586 | 3/50 (6%) |

### `jb_cognitive_reframe_specific_vs_ctrl`

**Positions: all**

| Condition | Mean frac in top-K | Mean Σ\|attr\| | Low-coverage prompts |
|---|---|---|---|
| `bare` | 28.2% | 383.2846 | 27/50 (54%) |
| `jb_roleplay` | 33.6% | 536.6896 | 17/50 (34%) |
| `ctrl_roleplay` | 36.7% | 529.1142 | 10/50 (20%) |
| `jb_fiction` | 39.3% | 576.2975 | 2/50 (4%) |
| `ctrl_fiction` | 42.7% | 639.8726 | 6/50 (12%) |
| `jb_analytical` | 68.7% | 949.8573 | 0/50 (0%) |
| `ctrl_analytical` | 38.2% | 509.9436 | 6/50 (12%) |
| `jb_completion` | 57.6% | 822.7852 | 1/50 (2%) |
| `ctrl_completion` | 53.6% | 724.0225 | 0/50 (0%) |
| `jb_cognitive_reframe` | 90.2% | 1500.4538 | 0/50 (0%) |
| `ctrl_cognitive_reframe` | 40.2% | 516.9304 | 2/50 (4%) |

### `jb_completion_specific_vs_ctrl`

**Positions: all**

| Condition | Mean frac in top-K | Mean Σ\|attr\| | Low-coverage prompts |
|---|---|---|---|
| `bare` | 45.8% | 732.9038 | 4/50 (8%) |
| `jb_roleplay` | 40.0% | 532.1100 | 11/50 (22%) |
| `ctrl_roleplay` | 46.5% | 625.2894 | 1/50 (2%) |
| `jb_fiction` | 45.5% | 661.9652 | 3/50 (6%) |
| `ctrl_fiction` | 50.5% | 787.2333 | 2/50 (4%) |
| `jb_analytical` | 72.2% | 1047.6575 | 0/50 (0%) |
| `ctrl_analytical` | 68.2% | 948.5505 | 0/50 (0%) |
| `jb_completion` | 88.0% | 1615.6141 | 0/50 (0%) |
| `ctrl_completion` | 58.5% | 755.9631 | 0/50 (0%) |
| `jb_cognitive_reframe` | 70.5% | 1053.1164 | 0/50 (0%) |
| `ctrl_cognitive_reframe` | 58.8% | 848.5059 | 0/50 (0%) |

### `jb_roleplay_specific_vs_ctrl`

**Positions: all**

| Condition | Mean frac in top-K | Mean Σ\|attr\| | Low-coverage prompts |
|---|---|---|---|
| `bare` | 27.3% | 301.7193 | 26/50 (52%) |
| `jb_roleplay` | 90.0% | 937.0250 | 0/50 (0%) |
| `ctrl_roleplay` | 35.0% | 333.6822 | 8/50 (16%) |
| `jb_fiction` | 24.0% | 204.6193 | 31/50 (62%) |
| `ctrl_fiction` | 33.0% | 308.9699 | 11/50 (22%) |
| `jb_analytical` | 5.7% | 84.1893 | 49/50 (98%) |
| `ctrl_analytical` | 33.7% | 314.5796 | 19/50 (38%) |
| `jb_completion` | 28.3% | 247.7387 | 23/50 (46%) |
| `ctrl_completion` | 34.7% | 292.9495 | 15/50 (30%) |
| `jb_cognitive_reframe` | 25.7% | 223.4532 | 22/50 (44%) |
| `ctrl_cognitive_reframe` | 32.3% | 295.2428 | 13/50 (26%) |

### `anti_refusal_amplifiers`

**Positions: all**

| Condition | Mean frac in top-K | Mean Σ\|attr\| | Low-coverage prompts |
|---|---|---|---|
| `bare` | 31.0% | 5862.8543 | 24/50 (48%) |
| `jb_roleplay` | 35.4% | 6782.0248 | 5/50 (10%) |
| `ctrl_roleplay` | 35.0% | 6799.2999 | 7/50 (14%) |
| `jb_fiction` | 35.8% | 6921.4665 | 4/50 (8%) |
| `ctrl_fiction` | 35.4% | 7176.5979 | 3/50 (6%) |
| `jb_analytical` | 35.7% | 7101.5333 | 1/50 (2%) |
| `ctrl_analytical` | 33.8% | 6495.6887 | 9/50 (18%) |
| `jb_completion` | 36.6% | 7048.3437 | 2/50 (4%) |
| `ctrl_completion` | 35.0% | 6669.1412 | 2/50 (4%) |
| `jb_cognitive_reframe` | 37.8% | 7423.2553 | 1/50 (2%) |
| `ctrl_cognitive_reframe` | 38.1% | 6989.3757 | 1/50 (2%) |

## Activation audit (Stage 02 attribution data)

Per-ablation, per-condition-class top-50 hit rate and mean |attribution|. Diagnoses whether the Stage 07 set logic produces a clean per-prompt separation, and whether class-specific subcircuits are correlationally selective for their target class.

### `universal_refusal_core` (26 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 96.46% | 492.59306 | 0/26 |
| `jb_*` | 98.31% | 488.34865 | 0/26 |
| `ctrl_*` | 98.22% | 491.87423 | 0/26 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 99.31% | 500.79215 |
| `jb_cognitive_reframe` | 99.15% | 500.77559 |
| `jb_completion` | 99.23% | 494.03797 |
| `jb_fiction` | 96.38% | 461.48181 |
| `jb_roleplay` | 97.46% | 484.65573 |

### `ctrl_shared_refusal` (4 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 88.00% | 249.19940 | 0/4 |
| `jb_*` | 60.00% | 184.66803 | 0/4 |
| `ctrl_*` | 82.40% | 233.63779 | 0/4 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 69.00% | 201.72441 |
| `jb_cognitive_reframe` | 48.50% | 149.19607 |
| `jb_completion` | 82.50% | 262.63028 |
| `jb_fiction` | 54.50% | 170.73004 |
| `jb_roleplay` | 45.50% | 139.05937 |

### `jb_fiction_specific_vs_ctrl` (11 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 3.09% | 6.76581 | 8/11 |
| `jb_*` | 27.38% | 71.30456 | 0/11 |
| `ctrl_*` | 14.44% | 34.21312 | 5/11 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 20.18% | 50.97467 |
| `jb_cognitive_reframe` | 21.64% | 63.26585 |
| `jb_completion` | 6.91% | 18.99597 |
| `jb_fiction` | 73.64% | 189.01414 |
| `jb_roleplay` | 14.55% | 34.27219 |

### `jb_analytical_specific_vs_ctrl` (9 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 15.11% | 48.33202 | 4/9 |
| `jb_*` | 44.80% | 138.78858 | 0/9 |
| `ctrl_*` | 28.58% | 103.96805 | 2/9 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 87.78% | 248.17870 |
| `jb_cognitive_reframe` | 46.22% | 134.29622 |
| `jb_completion` | 31.11% | 120.31810 |
| `jb_fiction` | 29.78% | 91.31312 |
| `jb_roleplay` | 29.11% | 99.83675 |

### `jb_cognitive_reframe_specific_vs_ctrl` (9 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 16.89% | 28.33672 | 6/9 |
| `jb_*` | 30.62% | 58.16030 | 0/9 |
| `ctrl_*` | 16.84% | 31.08307 | 2/9 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 25.33% | 49.15739 |
| `jb_cognitive_reframe` | 67.56% | 131.61770 |
| `jb_completion` | 26.67% | 48.14252 |
| `jb_fiction` | 11.33% | 20.33172 |
| `jb_roleplay` | 22.22% | 41.55218 |

### `jb_completion_specific_vs_ctrl` (8 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 36.00% | 79.91117 | 3/8 |
| `jb_*` | 39.80% | 90.65812 | 0/8 |
| `ctrl_*` | 33.55% | 68.98690 | 0/8 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 45.00% | 94.40934 |
| `jb_cognitive_reframe` | 44.25% | 95.74334 |
| `jb_completion` | 74.50% | 181.46515 |
| `jb_fiction` | 17.75% | 43.11274 |
| `jb_roleplay` | 17.50% | 38.56005 |

### `jb_roleplay_specific_vs_ctrl` (6 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 17.33% | 39.19807 | 2/6 |
| `jb_*` | 18.13% | 34.47937 | 0/6 |
| `ctrl_*` | 11.87% | 23.77017 | 2/6 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 5.00% | 13.25614 |
| `jb_cognitive_reframe` | 8.33% | 16.50034 |
| `jb_completion` | 8.67% | 16.61927 |
| `jb_fiction` | 3.67% | 6.64817 |
| `jb_roleplay` | 65.00% | 119.37291 |

### `anti_refusal_amplifiers` (64 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 20.37% | 79.75694 | 35/64 |
| `jb_*` | 23.44% | 94.21702 | 25/64 |
| `ctrl_*` | 23.51% | 92.07132 | 29/64 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 24.50% | 96.66702 |
| `jb_cognitive_reframe` | 23.97% | 99.45608 |
| `jb_completion` | 22.97% | 93.68662 |
| `jb_fiction` | 23.28% | 90.95561 |
| `jb_roleplay` | 22.47% | 90.31977 |

