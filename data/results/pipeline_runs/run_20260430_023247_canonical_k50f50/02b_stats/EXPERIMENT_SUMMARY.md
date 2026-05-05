# Stage 02b Statistical Analysis

- **Prompts**: 50
- **Model**: google/gemma-3-4b-it
- **Target**: L15 (causal). Two graph modes:
  - **multi** — targets the template anchors [-5, -3, -2] (`<end_of_turn>`, `<start_of_turn>`, `model`)
  - **single** — target pos=-2 only (Tejas-verified causal position)

Comparisons are run for each mode:
- `vs_bare`: bare ↔ jb_<class> — legacy JB-effect delta
- `vs_ctrl`: ctrl_<class> ↔ jb_<class> — token-matched, isolates JB semantics
- `ctrl_vs_bare`: bare ↔ ctrl_<class> — sanity (ctrl should track bare)

## Mode: `multi`

### multi · vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | +51056.6 | +54766.4 | +3709.8 | +7.3% | 1.899e-12*** | +1.58 | [+3066.3, +4358.7] | Pro-refusal recruitment |
| **fiction** | 50 | +51056.6 | +57258.6 | +6202.0 | +12.1% | 5.329e-15*** | +1.84 | [+5293.3, +7129.0] | Pro-refusal recruitment |
| **analytical** | 50 | +51056.6 | +47900.3 | -3156.3 | -6.2% | 7.547e-09*** | -1.05 | [-3974.5, -2356.3] | Balanced |
| **completion** | 50 | +51056.6 | +53825.2 | +2768.6 | +5.4% | 8.23e-09*** | +1.05 | [+2041.2, +3490.4] | Balanced |
| **cognitive_reframe** | 50 | +51056.6 | +46412.7 | -4643.9 | -9.1% | 7.547e-09*** | -1.07 | [-5819.8, -3439.7] | Balanced |

### multi · vs_ctrl

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | +53158.9 | +54766.4 | +1607.5 | +3.0% | 4.214e-05*** | +0.57 | [+839.7, +2385.4] | Amplification-dominant |
| **fiction** | 50 | +55989.7 | +57258.6 | +1268.8 | +2.3% | 0.005942** | +0.45 | [+492.0, +2041.9] | Pro-refusal recruitment |
| **analytical** | 50 | +56774.3 | +47900.3 | -8874.0 | -15.6% | 1.776e-15*** | -3.78 | [-9535.9, -8227.8] | Balanced |
| **completion** | 50 | +55507.5 | +53825.2 | -1682.4 | -3.0% | 3.706e-09*** | -0.97 | [-2166.3, -1215.8] | Anti-suppression |
| **cognitive_reframe** | 50 | +58186.8 | +46412.7 | -11774.1 | -20.2% | 1.776e-15*** | -3.21 | [-12755.0, -10735.3] | Balanced |

### multi · ctrl_vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | +51056.6 | +53158.9 | +2102.3 | +4.1% | 0.0001558*** | +0.60 | [+1107.6, +3056.5] | Balanced |
| **fiction** | 50 | +51056.6 | +55989.7 | +4933.1 | +9.7% | 9.77e-14*** | +1.85 | [+4199.5, +5648.9] | Balanced |
| **analytical** | 50 | +51056.6 | +56774.3 | +5717.7 | +11.2% | 1.776e-15*** | +3.55 | [+5269.0, +6163.7] | Pro-refusal recruitment |
| **completion** | 50 | +51056.6 | +55507.5 | +4450.9 | +8.7% | 1.776e-15*** | +2.53 | [+3978.9, +4926.8] | Balanced |
| **cognitive_reframe** | 50 | +51056.6 | +58186.8 | +7130.2 | +14.0% | 1.776e-15*** | +2.87 | [+6454.8, +7815.0] | Pro-refusal recruitment |
## Mode: `single`

### single · vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | -48886.4 | -52117.5 | -3231.1 | +6.6% | 1.776e-15*** | -5.37 | [-3396.5, -3069.9] | Anti-suppression |
| **fiction** | 50 | -48886.4 | -51693.0 | -2806.6 | +5.7% | 1.776e-15*** | -3.75 | [-3017.4, -2604.0] | Anti-suppression |
| **analytical** | 50 | -48886.4 | -52344.1 | -3457.7 | +7.1% | 1.776e-15*** | -6.33 | [-3606.3, -3305.9] | Anti-suppression |
| **completion** | 50 | -48886.4 | -51580.4 | -2694.0 | +5.5% | 1.776e-15*** | -5.05 | [-2840.5, -2546.9] | Anti-suppression |
| **cognitive_reframe** | 50 | -48886.4 | -53961.3 | -5074.9 | +10.4% | 1.776e-15*** | -5.87 | [-5311.0, -4836.3] | Anti-suppression |

### single · vs_ctrl

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | -52663.9 | -52117.5 | +546.4 | -1.0% | 2.638e-08*** | +0.88 | [+378.2, +718.9] | Balanced |
| **fiction** | 50 | -52086.6 | -51693.0 | +393.5 | -0.8% | 0.0001303*** | +0.61 | [+218.9, +571.1] | Balanced |
| **analytical** | 50 | -51729.2 | -52344.1 | -614.8 | +1.2% | 4.96e-11*** | -1.20 | [-758.8, -472.6] | Dampening-dominant |
| **completion** | 50 | -51898.1 | -51580.4 | +317.7 | -0.6% | 7.156e-05*** | +0.61 | [+180.5, +468.1] | Balanced |
| **cognitive_reframe** | 50 | -52291.8 | -53961.3 | -1669.5 | +3.2% | 3.553e-15*** | -2.03 | [-1895.6, -1446.5] | Anti-suppression |

### single · ctrl_vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | -48886.4 | -52663.9 | -3777.5 | +7.7% | 1.776e-15*** | -7.81 | [-3909.1, -3646.8] | Anti-suppression |
| **fiction** | 50 | -48886.4 | -52086.6 | -3200.2 | +6.5% | 1.776e-15*** | -6.38 | [-3338.9, -3067.0] | Anti-suppression |
| **analytical** | 50 | -48886.4 | -51729.2 | -2842.8 | +5.8% | 1.776e-15*** | -6.60 | [-2962.0, -2723.7] | Anti-suppression |
| **completion** | 50 | -48886.4 | -51898.1 | -3011.7 | +6.2% | 1.776e-15*** | -8.16 | [-3113.4, -2910.6] | Anti-suppression |
| **cognitive_reframe** | 50 | -48886.4 | -52291.8 | -3405.4 | +7.0% | 1.776e-15*** | -7.17 | [-3534.9, -3278.8] | Anti-suppression |

## Direction (Stage 01) Summary

- Best separation layer: **L32** (magnitude 20873.2109)
- Best causal layer: **L15** (used for attribution)
