# Stage 02b Statistical Analysis

- **Prompts**: 3
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
| **roleplay** | 3 | +52371.7 | +56096.9 | +3725.2 | +7.1% | 0.5 | +0.77 | [-1846.2, +7001.8] | Pro-refusal recruitment |
| **fiction** | 3 | +52371.7 | +58188.2 | +5816.5 | +11.1% | 0.25 | +1.50 | [+1630.9, +9288.2] | Pro-refusal recruitment |
| **analytical** | 3 | +52371.7 | +49177.7 | -3194.0 | -6.1% | 0.5 | -1.03 | [-6135.5, +64.2] | Balanced |
| **completion** | 3 | +52371.7 | +55651.8 | +3280.1 | +6.3% | 0.5 | +0.71 | [-2087.9, +6088.2] | Balanced |
| **cognitive_reframe** | 3 | +52371.7 | +46641.3 | -5730.4 | -10.9% | 0.25 | -1.16 | [-9365.5, -82.7] | Anti-suppression |

### multi · vs_ctrl

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 3 | +54213.0 | +56096.9 | +1884.0 | +3.5% | 0.5 | +0.86 | [-420.3, +3924.7] | Amplification-dominant |
| **fiction** | 3 | +56823.7 | +58188.2 | +1364.5 | +2.4% | 0.25 | +0.90 | [+45.8, +3017.9] | Pro-refusal recruitment |
| **analytical** | 3 | +58253.6 | +49177.7 | -9075.9 | -15.6% | 0.25 | -6.54 | [-10479.5, -7706.7] | Balanced |
| **completion** | 3 | +56673.4 | +55651.8 | -1021.6 | -1.8% | 1 | -0.44 | [-3698.1, +322.1] | Balanced |
| **cognitive_reframe** | 3 | +60443.0 | +46641.3 | -13801.7 | -22.8% | 0.25 | -4.01 | [-17416.5, -10573.8] | Balanced |

### multi · ctrl_vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 3 | +52371.7 | +54213.0 | +1841.3 | +3.5% | 0.75 | +0.28 | [-5771.0, +6440.5] | Balanced |
| **fiction** | 3 | +52371.7 | +56823.7 | +4452.0 | +8.5% | 0.5 | +0.87 | [-1387.0, +8258.3] | Balanced |
| **analytical** | 3 | +52371.7 | +58253.6 | +5881.9 | +11.2% | 0.25 | +2.26 | [+2906.0, +7770.9] | Pro-refusal recruitment |
| **completion** | 3 | +52371.7 | +56673.4 | +4301.7 | +8.2% | 0.25 | +1.84 | [+1610.2, +5766.2] | Balanced |
| **cognitive_reframe** | 3 | +52371.7 | +60443.0 | +8071.3 | +15.4% | 0.25 | +3.35 | [+5671.9, +10491.1] | Pro-refusal recruitment |
## Mode: `single`

### single · vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 3 | -49156.6 | -52268.6 | -3112.0 | +6.3% | 0.25 | -5.48 | [-3764.9, -2729.0] | Anti-suppression |
| **fiction** | 3 | -49156.6 | -52202.4 | -3045.7 | +6.2% | 0.25 | -8.61 | [-3261.9, -2637.3] | Anti-suppression |
| **analytical** | 3 | -49156.6 | -52653.4 | -3496.8 | +7.1% | 0.25 | -8.62 | [-3931.4, -3128.7] | Anti-suppression |
| **completion** | 3 | -49156.6 | -51585.3 | -2428.6 | +4.9% | 0.25 | -6.77 | [-2681.5, -2018.3] | Anti-suppression |
| **cognitive_reframe** | 3 | -49156.6 | -54232.0 | -5075.3 | +10.3% | 0.25 | -4.99 | [-5831.7, -3917.7] | Anti-suppression |

### single · vs_ctrl

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 3 | -52493.3 | -52268.6 | +224.7 | -0.4% | 0.75 | +0.41 | [-200.0, +849.1] | Balanced |
| **fiction** | 3 | -52266.0 | -52202.4 | +63.6 | -0.1% | 1 | +0.10 | [-494.6, +723.5] | Balanced |
| **analytical** | 3 | -51564.1 | -52653.4 | -1089.3 | +2.1% | 0.25 | -2.77 | [-1337.7, -636.4] | Balanced |
| **completion** | 3 | -51843.9 | -51585.3 | +258.6 | -0.5% | 0.25 | +1.26 | [+113.2, +493.4] | Balanced |
| **cognitive_reframe** | 3 | -52003.7 | -54232.0 | -2228.2 | +4.3% | 0.25 | -1.94 | [-3257.3, -989.8] | Anti-suppression |

### single · ctrl_vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 3 | -49156.6 | -52493.3 | -3336.7 | +6.8% | 0.25 | -8.20 | [-3578.2, -2867.0] | Anti-suppression |
| **fiction** | 3 | -49156.6 | -52266.0 | -3109.3 | +6.3% | 0.25 | -10.13 | [-3360.8, -2767.3] | Anti-suppression |
| **analytical** | 3 | -49156.6 | -51564.1 | -2407.5 | +4.9% | 0.25 | -8.53 | [-2637.7, -2092.5] | Anti-suppression |
| **completion** | 3 | -49156.6 | -51843.9 | -2687.3 | +5.5% | 0.25 | -15.81 | [-2850.9, -2511.7] | Anti-suppression |
| **cognitive_reframe** | 3 | -49156.6 | -52003.7 | -2847.1 | +5.8% | 0.25 | -11.74 | [-3038.9, -2574.4] | Anti-suppression |

## Direction (Stage 01) Summary

- Best separation layer: **L32** (magnitude 20873.2109)
- Best causal layer: **L15** (used for attribution)
