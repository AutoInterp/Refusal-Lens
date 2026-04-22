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

## Mode: `single`

### single · vs_bare

| Class | N | Baseline | Treatment | ΔNet | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Dominant |
|---|---|---|---|---|---|---|---|---|---|
| **roleplay** | 50 | +70.5 | +31.8 | -38.7 | -54.9% | 1.37e-08*** | -0.91 | [-50.5, -27.0] | Balanced |
| **fiction** | 50 | +70.5 | +5.2 | -65.3 | -92.7% | 3.002e-13*** | -1.57 | [-76.4, -53.9] | Balanced |
| **analytical** | 50 | +70.5 | -3.2 | -73.7 | -104.6% | 5.329e-15*** | -2.37 | [-82.1, -65.2] | Balanced |
| **completion** | 50 | +70.5 | +75.5 | +5.0 | +7.2% | 0.01057* | +0.27 | [-0.0, +10.2] | Balanced |
| **cognitive_reframe** | 50 | +70.5 | +20.3 | -50.2 | -71.3% | 2.487e-14*** | -1.41 | [-60.0, -40.6] | Dampening-dominant |

## Direction (Stage 01) Summary

- Best separation layer: **L32** (magnitude 20873.2109)
- Best causal layer: **L15** (used for attribution)
