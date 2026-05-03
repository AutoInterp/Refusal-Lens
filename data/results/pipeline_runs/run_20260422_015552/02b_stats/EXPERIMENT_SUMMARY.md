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
| **roleplay** | 50 | +6.1 | +5.5 | -0.6 | -9.9% | 1.739e-07*** | -0.83 | [-0.8, -0.4] | Anti-suppression |
| **fiction** | 50 | +6.1 | +4.2 | -1.9 | -31.2% | 9.77e-14*** | -1.69 | [-2.2, -1.6] | Dampening-dominant |
| **analytical** | 50 | +6.1 | +3.2 | -2.9 | -47.7% | 1.776e-15*** | -3.16 | [-3.1, -2.6] | Balanced |
| **completion** | 50 | +6.1 | +6.4 | +0.4 | +6.1% | 0.0007549*** | +0.50 | [+0.2, +0.6] | Balanced |
| **cognitive_reframe** | 50 | +6.1 | +3.0 | -3.1 | -51.2% | 3.553e-15*** | -2.09 | [-3.5, -2.7] | Balanced |

### multi · vs_ctrl

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | +5.8 | +5.5 | -0.4 | -6.2% | 0.0005967*** | -0.43 | [-0.6, -0.1] | Dampening-dominant |
| **fiction** | 50 | +6.4 | +4.2 | -2.2 | -34.6% | 1.776e-15*** | -2.21 | [-2.5, -1.9] | Dampening-dominant |
| **analytical** | 50 | +6.3 | +3.2 | -3.2 | -49.8% | 1.776e-15*** | -4.07 | [-3.4, -2.9] | Dampening-dominant |
| **completion** | 50 | +6.2 | +6.4 | +0.3 | +4.2% | 0.002799** | +0.43 | [+0.1, +0.4] | Balanced |
| **cognitive_reframe** | 50 | +6.2 | +3.0 | -3.2 | -51.9% | 1.776e-15*** | -2.05 | [-3.6, -2.8] | Balanced |

### multi · ctrl_vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | +6.1 | +5.8 | -0.2 | -3.9% | 0.0461* | -0.31 | [-0.5, -0.0] | Anti-suppression |
| **fiction** | 50 | +6.1 | +6.4 | +0.3 | +5.2% | 0.0003826*** | +0.58 | [+0.2, +0.5] | Pro-refusal recruitment |
| **analytical** | 50 | +6.1 | +6.3 | +0.3 | +4.2% | 0.006941** | +0.46 | [+0.1, +0.4] | Pro-refusal recruitment |
| **completion** | 50 | +6.1 | +6.2 | +0.1 | +1.8% | 0.5399 | +0.19 | [-0.0, +0.3] | Balanced |
| **cognitive_reframe** | 50 | +6.1 | +6.2 | +0.1 | +1.4% | 0.2955 | +0.13 | [-0.1, +0.3] | Balanced |
## Mode: `single`

### single · vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | +2.1 | +1.4 | -0.6 | -30.6% | 1.776e-15*** | -2.26 | [-0.7, -0.6] | Balanced |
| **fiction** | 50 | +2.1 | +0.9 | -1.2 | -56.7% | 1.776e-15*** | -2.40 | [-1.3, -1.0] | Dampening-dominant |
| **analytical** | 50 | +2.1 | +0.6 | -1.5 | -72.5% | 1.776e-15*** | -4.82 | [-1.6, -1.4] | Dampening-dominant |
| **completion** | 50 | +2.1 | +1.8 | -0.3 | -15.0% | 3.553e-15*** | -2.00 | [-0.4, -0.3] | Anti-suppression |
| **cognitive_reframe** | 50 | +2.1 | +0.6 | -1.5 | -73.5% | 1.776e-15*** | -3.21 | [-1.7, -1.4] | Dampening-dominant |

### single · vs_ctrl

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | +1.6 | +1.4 | -0.2 | -11.9% | 2.231e-06*** | -0.68 | [-0.3, -0.1] | Dampening-dominant |
| **fiction** | 50 | +1.7 | +0.9 | -0.8 | -48.1% | 1.776e-15*** | -2.06 | [-0.9, -0.7] | Dampening-dominant |
| **analytical** | 50 | +1.7 | +0.6 | -1.1 | -66.3% | 1.776e-15*** | -4.20 | [-1.2, -1.1] | Dampening-dominant |
| **completion** | 50 | +1.7 | +1.8 | +0.1 | +5.1% | 0.0005511*** | +0.49 | [+0.0, +0.1] | Balanced |
| **cognitive_reframe** | 50 | +1.5 | +0.6 | -1.0 | -63.3% | 1.776e-15*** | -2.21 | [-1.1, -0.8] | Dampening-dominant |

### single · ctrl_vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | +2.1 | +1.6 | -0.4 | -21.2% | 1.776e-15*** | -1.68 | [-0.5, -0.4] | Balanced |
| **fiction** | 50 | +2.1 | +1.7 | -0.3 | -16.7% | 1.776e-15*** | -1.72 | [-0.4, -0.3] | Balanced |
| **analytical** | 50 | +2.1 | +1.7 | -0.4 | -18.3% | 1.776e-15*** | -3.11 | [-0.4, -0.3] | Balanced |
| **completion** | 50 | +2.1 | +1.7 | -0.4 | -19.1% | 1.776e-15*** | -2.66 | [-0.4, -0.4] | Balanced |
| **cognitive_reframe** | 50 | +2.1 | +1.5 | -0.6 | -27.7% | 1.776e-15*** | -3.04 | [-0.6, -0.5] | Dampening-dominant |

## Direction (Stage 01) Summary

- Best separation layer: **L32** (magnitude 20873.2109)
- Best causal layer: **L15** (used for attribution)
