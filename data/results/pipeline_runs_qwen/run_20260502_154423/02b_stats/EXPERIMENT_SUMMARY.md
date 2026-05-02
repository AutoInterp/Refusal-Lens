# Attribution Experiment: Statistical Analysis

**Prompts**: 5 | **Model**: Qwen/Qwen3-4B
**Direction**: Layer 34 (best separation)

## Main Results

| Class | Net Delta | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Consistency |
|-------|----------|----------|-------------|-----------|--------|-------------|
| **Roleplay** | -7.0 | -20.7% | 0.0625ns | -1.03 | [-13.0, -3.3] | 5/5 |
| **Fiction** | -21.5 | -63.9% | 0.0625ns | -1.85 | [-30.6, -12.9] | 5/5 |
| **Analytical** | -13.2 | -39.5% | 0.0625ns | -2.54 | [-17.4, -9.3] | 5/5 |
| **Completion** | -5.2 | -15.5% | 0.0625ns | -1.27 | [-8.9, -2.8] | 5/5 |
| **Cognitive_reframe** | -4.9 | -14.7% | 0.0625ns | -1.12 | [-8.9, -2.1] | 5/5 |

## Dual Mechanism Decomposition

| Class | dPos (pro-refusal) | dNeg (anti-refusal) | Net | Dominant |
|-------|--------------------|--------------------|----|----------|
| **Roleplay** | -6.7 (-17.0%) | -0.3 (-4.8%) | -7.0 | Dampening-dominant |
| **Fiction** | -20.6 (-52.4%) | -0.8 (-14.3%) | -21.5 | Dampening-dominant |
| **Analytical** | -12.3 (-31.4%) | -0.9 (-15.6%) | -13.2 | Dampening-dominant |
| **Completion** | -3.8 (-9.6%) | -1.4 (-24.9%) | -5.2 | Dampening-dominant |
| **Cognitive_reframe** | -4.4 (-11.2%) | -0.5 (-8.8%) | -4.9 | Dampening-dominant |

## Feature Comparison

| Class | Bare | JB | Shared % | JB-only % | Sign-flip % |
|-------|------|-----|----------|-----------|------------|
| **Roleplay** | 3981 | 4020 | 59.0% | 41.6% | 8.2% |
| **Fiction** | 3981 | 3994 | 49.0% | 51.2% | 12.8% |
| **Analytical** | 3981 | 4200 | 55.9% | 47.0% | 9.9% |
| **Completion** | 3981 | 4049 | 68.0% | 33.1% | 7.3% |
| **Cognitive_reframe** | 3981 | 4065 | 64.4% | 37.0% | 8.1% |
