# Circuit Analysis of Refusal Mechanisms in Gemma-3-4B-IT

**Author:** Tejas Dahiya  
**Project:** Algoverse AI Safety Research Fellowship  
**Model:** Gemma-3-4B-IT  
**Date:** March-April 2025

## Overview

This directory contains experiments investigating how jailbreaks interact with the refusal direction in Gemma-3-4B-IT using circuit tracing (Anthropic's `circuit-tracer` library with transcoders).

**Key contributions:**
1. Discovery of two mechanistically distinct jailbreak classes at the MLP level
2. Design of novel circuit-informed jailbreaks that bypass refusal on hard topics (11/12) and are immune to refusal-direction steering (1/32 flipped)
3. Important limitation identified: attention heads carry 99.6% of the refusal signal; transcoder-based analysis only captures MLP behavior

## Key Findings

### 1. Corrected Refusal Direction

Fixed critical bugs from initial computation (wrong token position, bfloat16 precision, right-padding). Corrected direction computed at position -2 (`model` token), all 5 end-of-instruction positions tested, float64 accumulation, left-padding.

| Metric | Old (buggy) | Corrected |
|--------|-------------|-----------|
| Harmful mean | +22,963 | **+19,236** |
| Harmless mean | +21,948 | **-1,552** |
| Separation | 1,015 (4.4%) | **20,788 (108%)** |
| Jailbroken mean | +22,895 | **+16,020** |
| JB vs Harmful diff | -68 | **-3,215** |

Harmful and harmless now on **opposite sides of zero** with no overlap.

![Sanity Check](data/tejas_experiments/figures/sanity_check_projections.png)

### 2. Attribution with Corrected Direction (10 pairs, all features)

With corrected direction and ALL active features (no 3k filter):

- Bare mean: 75.5
- JB mean: 56.7
- Mean diff: **-18.7** (was -1.86 with old direction)
- 9/10 JB lower

![Attribution](data/tejas_experiments/figures/attribution_30pairs.png)

### 3. Mechanism Comparison (Corrected Direction)

With corrected direction, both RP and fiction show negative net attribution, but through different feature patterns:

| Topic | Bare (pos/neg/net) | RP (pos/neg/net) | Fiction (pos/neg/net) |
|-------|-------------------|-----------------|----------------------|
| Lock | 93/-89/+4 | 59/-118/**-60** | 61/-91/**-30** |
| Hack | 117/-63/+54 | 44/-96/**-52** | 71/-106/**-34** |
| Phish | 119/-58/+62 | 60/-129/**-69** | 74/-102/**-28** |

**RP jailbreaks:** Pro-refusal features drop MORE aggressively (117→44). Anti-refusal grows.  
**Fiction jailbreaks:** Pro-refusal drops LESS (117→71). Anti-refusal grows MORE (63→106).

Both suppress refusal but through different MLP feature dynamics.

![Mechanism Comparison](data/tejas_experiments/figures/mechanism_comparison.png)

### 4. Critical Limitation: Attention Dominates Refusal

Attribution sum (~75) vs dot product (~18,322). Per-layer probing shows attention + embeddings contribute 99.6% of the refusal dot product. Transcoders only decompose MLPs.

This means our mechanism analysis describes MLP behavior only. The main refusal circuit likely runs through attention heads, which require attention SAEs or direct head attribution to analyze.

### 5. Refusal-Direction Steering Only Patches Some Jailbreaks

Sweep over alpha (15-200) x layers (10, 13, 15, 18) on three successful jailbreaks:

- **Locksmith RP prompt:** 13/16 configs flipped to refusal. Steering works.
- **Mr. Robot fiction prompt:** 0/16 flipped. Even alpha=200 fails.
- **Phishing trainer prompt:** 0/16 flipped. Steering fails completely.

![Steering Sweep](data/tejas_experiments/figures/steering_sweep.png)

### 6. Novel Circuit-Informed Jailbreaks (7/8 bypass)

Based on the insight that fiction jailbreaks work by making the model treat harmful content as analytical OUTPUT rather than harmful INSTRUCTION, we designed 8 novel framings:

| Framing | Result |
|---------|--------|
| Analysis ("is this description correct?") | Bypassed |
| Completion ("complete this manual entry") | Bypassed |
| Translation (informal to formal) | Bypassed |
| Meta/Grading ("grade this student answer") | Bypassed |
| Comparison ("compare three methods") | Bypassed |
| Historical ("document the evolution") | Bypassed |
| Taxonomy ("create a taxonomy with examples") | Bypassed |
| Debug ("debug this script") | Refused |

![Novel Jailbreaks](data/tejas_experiments/figures/novel_jailbreaks.png)

### 7. Novel Jailbreaks Work on Hard Topics AND Are Immune to Steering

Tested on WiFi hacking, phishing, and malware creation (all refused when asked bare):

- WiFi hacking: 3/4 analytical jailbreaks comply
- Phishing: 4/4 comply
- Malware: 4/4 comply

Steering immunity: only **1/32** flipped to refusal (alpha up to 200, layers 10 and 15).

![Hard Topics and Steering](data/tejas_experiments/figures/hard_topics_and_steering.png)

### 8. OVAT Controlled Experiment (160 tests)

20 base harmful prompts x 8 prefix types, one variable at a time:

| Prefix | Compliance Rate |
|--------|----------------|
| technical | 25% |
| hypothetical | 10% |
| none / researcher / educational / fiction | 5% |
| polite / instructor | 0% |

![OVAT Results](data/tejas_experiments/figures/ovat_results.png)

### 9. Cosine Similarity with Ruqiya (Fixed)

With corrected multi-position extraction, agreement with Ruqiya's direction is strong across all layers at matching positions:

| Layer | Cosine Similarity |
|-------|------------------|
| L10 | 0.965 |
| L15 | 0.938 |
| L18 | 0.843 |
| L25 | 0.860 |
| L32 | 0.883 |

The old L18=0.046 result was caused by comparing different token positions (our -1 vs her -2).

![Cosine Similarity](data/tejas_experiments/figures/cosine_similarity.png)

### Summary Figure

![Summary](data/tejas_experiments/figures/summary_figure.png)

## Bug Fixes (Georg's Feedback)

1. **Token position:** Was extracting at position -1 (final `\n`). Fixed to extract at positions [-5,-4,-3,-2,-1]. Best position is -2 (`model` token).
2. **Numerical precision:** Was using bfloat16 accumulation. Fixed to float64.
3. **Padding:** Was using right-padding. Fixed to left-padding.
4. **Feature filter:** Was filtering to 3k features. Tested with all features (~14k active). Filter only loses 0.1-1.3%.
5. **PT vs IT model:** Initial attribution used PT transcoders. Fixed to IT transcoders.
6. **Attribution gap:** Identified that transcoders only decompose MLPs (0.4% of refusal signal). Attention carries 99.6%.

## Methodology

**Refusal Direction:** Difference-in-means (Arditi et al., 2024) on 64 harmful + 64 harmless prompts. Multi-position extraction at [-5,-4,-3,-2,-1], float64 accumulation, left-padding. Best: position -2, layer 32.

**Circuit Attribution:** Anthropic's circuit-tracer with CustomTarget for refusal direction. IT model (gemma-3-4b-it) with IT transcoders (mwhanna/gemma-scope-2-4b-it). All active features (~14k per prompt).

**Steering:** Forward hook adding alpha x refusal_direction to residual stream at target layer during generation on gemma-3-4b-it.

## Limitations

1. Mechanism comparison based on 9 prompts (3 topics x 3 types).
2. Refusal classification uses keyword matching.
3. Transcoders only decompose MLPs — attention heads (99.6% of refusal signal) are not analyzed.
4. Novel jailbreaks not tested on hardest topics (weapons, CSAM).
5. 10-pair attribution (not 30) with corrected direction due to compute constraints.

## File Structure

```
figures/              - Visualization plots (regenerated with corrected data)
results/              - Original experiment data (pre-bugfix)
results_v2/           - Corrected experiment data
    refusal_direction_v2.pt      - Corrected refusal direction
    sanity_check_v2.json         - Corrected sanity check
    separation_table.json        - Position x layer separation table
    v2_attribution_10pairs.json  - Attribution with corrected direction
    v2_mechanism_comparison.json - Mechanism with corrected direction
scripts/              - All experiment scripts (01-14)
```

## References

- Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." arXiv:2406.11717
- Anthropic circuit-tracer: https://github.com/safety-research/circuit-tracer
