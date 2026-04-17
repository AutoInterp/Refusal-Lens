# Attribution Experiment: Statistical Analysis

**Prompts**: 50 | **Model**: google/gemma-3-4b-it
**Direction**: Layer 32 (best separation)

## Main Results

| Class | Net Delta | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Consistency |
|-------|----------|----------|-------------|-----------|--------|-------------|
| **Roleplay** | -38.7 | -54.9% | 1.37e-08*** | -0.91 | [-50.5, -27.0] | 42/50 |
| **Fiction** | -65.3 | -92.7% | 3.002e-13*** | -1.57 | [-76.6, -53.8] | 47/50 |
| **Analytical** | -73.7 | -104.6% | 5.329e-15*** | -2.37 | [-82.1, -64.9] | 49/50 |
| **Completion** | +5.0 | +7.2% | 0.01057* | +0.27 | [-0.1, +10.2] | 15/50 |
| **Cognitive_reframe** | -50.2 | -71.3% | 2.487e-14*** | -1.41 | [-60.2, -40.4] | 49/50 |

## Dual Mechanism Decomposition

| Class | dPos (pro-refusal) | dNeg (anti-refusal) | Net | Dominant |
|-------|--------------------|--------------------|----|----------|
| **Roleplay** | -22.5 (-16.7%) | -16.2 (-25.3%) | -38.7 | Balanced |
| **Fiction** | -43.1 (-32.0%) | -22.2 (-34.6%) | -65.3 | Balanced |
| **Analytical** | -44.7 (-33.2%) | -29.0 (-45.1%) | -73.7 | Balanced |
| **Completion** | +19.7 (+14.6%) | -14.6 (-22.8%) | +5.0 | Pro-refusal recruitment |
| **Cognitive_reframe** | -33.8 (-25.1%) | -16.4 (-25.5%) | -50.2 | Dampening-dominant |

## Feature Comparison

| Class | Bare | JB | Shared % | JB-only % | Sign-flip % |
|-------|------|-----|----------|-----------|------------|
| **Roleplay** | 8342 | 12540 | 65.7% | 56.3% | 17.5% |
| **Fiction** | 8342 | 12996 | 58.5% | 62.5% | 25.6% |
| **Analytical** | 8342 | 12109 | 63.0% | 56.6% | 23.1% |
| **Completion** | 8342 | 11200 | 73.5% | 45.3% | 16.6% |
| **Cognitive_reframe** | 8342 | 11131 | 64.4% | 51.8% | 20.3% |
