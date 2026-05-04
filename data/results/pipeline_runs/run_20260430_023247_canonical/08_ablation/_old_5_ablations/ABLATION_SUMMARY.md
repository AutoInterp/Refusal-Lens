# Stage 08 Subcircuit Ablation — Summary

**Method**: zero-ablation of transcoder features via `ReplacementModel.feature_intervention_generate`.
**Subcircuits source**: `subcircuits.json`.
**Elapsed**: 0.0 min.
**Positions modes**: all.
**Low-coverage threshold**: 0.30 (prompts where <threshold of ablation features are in top-K are flagged).

## How to read these numbers

- `recovery_rate` = baseline COMPLY → ablated REFUSE. `break_rate` = baseline REFUSE → ablated COMPLY.
- **Per-class** rows show the unweighted rate for each condition.
- **Comply-weighted JB recovery** (under each position mode) is Σ(per-class rate × per-class baseline_comply) / Σ(baseline_comply). Reflects the model's behavior on actual JB-success cases without dropping rare classes — the headline NeurIPS rigor metric.
- The **per-prompt coverage** table flags prompts where the ablation features weren't in that prompt's top-K attribution. Low coverage on a class explains null recovery rates: the features couldn't be doing much because they weren't strongly active to begin with.
- For `subcircuits.json` runs (per-prompt sweep), the subcircuits are constructed from features in top-K for ≥F fraction of prompts in each condition; legacy `subcircuits.json` uses corpus union.

## Per-ablation results

### `canonical_pro_refusal` (88 features)

**Positions: all**

| Condition | Baseline REFUSE | Baseline COMPLY | Ablated REFUSE | Recovery rate | Break rate |
|---|---|---|---|---|---|
| `bare` | 0 | 0 | 0 | 0.0% | 0.0% |
| `jb_roleplay` | 0 | 0 | 0 | 0.0% | 0.0% |
| `ctrl_roleplay` | 0 | 0 | 0 | 0.0% | 0.0% |
| `jb_fiction` | 0 | 0 | 0 | 0.0% | 0.0% |
| `ctrl_fiction` | 0 | 0 | 0 | 0.0% | 0.0% |
| `jb_analytical` | 0 | 0 | 0 | 0.0% | 0.0% |
| `ctrl_analytical` | 0 | 0 | 0 | 0.0% | 0.0% |
| `jb_completion` | 0 | 0 | 0 | 0.0% | 0.0% |
| `ctrl_completion` | 0 | 0 | 0 | 0.0% | 0.0% |
| `jb_cognitive_reframe` | 0 | 0 | 0 | 0.0% | 0.0% |
| `ctrl_cognitive_reframe` | 0 | 0 | 0 | 0.0% | 0.0% |

**Comply-weighted aggregates:**

- JB recovery (weighted by baseline_comply across all 5 JB classes): **0.0%** (n_jb_comply=0)
- ctrl break (weighted by baseline_refuse across all 5 ctrl classes): **0.0%** (n_ctrl_refuse=0)
- bare break: **0.0%** (n_bare_refuse=0)

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

## Activation audit (Stage 02 attribution data)

Per-ablation, per-condition-class top-50 hit rate and mean |attribution|. Diagnoses whether the Stage 07 set logic produces a clean per-prompt separation, and whether class-specific subcircuits are correlationally selective for their target class.

### `canonical_pro_refusal` (88 features) — corpus-level activity

| Condition class | Mean top-50 hit rate | Mean \|attr\|/prompt | Features never in top-50 |
|---|---|---|---|
| `bare` | 0.00% | 0.00000 | 88/88 |
| `jb_*` | 5.28% | 15.48791 | 50/88 |
| `ctrl_*` | 4.15% | 12.70505 | 54/88 |

Per-JB-class breakdown (selectivity check):

| JB class | Mean top-50 hit rate | Mean \|attr\|/prompt |
|---|---|---|
| `jb_analytical` | 5.20% | 12.81665 |
| `jb_cognitive_reframe` | 5.59% | 15.55659 |
| `jb_completion` | 4.41% | 11.60663 |
| `jb_fiction` | 7.50% | 25.67834 |
| `jb_roleplay` | 3.68% | 11.78133 |

