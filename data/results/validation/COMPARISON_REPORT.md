# Experiment Comparison Report: Tejas vs Foundation Pipeline

Generated from `scripts/generate_comparison_report.py`

---

## 1. Refusal Direction Computation

Both approaches use difference-in-means (Arditi et al. 2024) with corrected parameters:
- Position: -2 (model token in chat template)
- Accumulation: float64
- Padding: left
- Data: 64 harmful + 64 harmless prompts

| Metric | Tejas (v2) | Foundation | Match? |
|--------|-----------|------------|--------|
| Best layer | 32 | 33 | CHECK (33 vs 32) |
| Separation | 20,788 | 29647.3 | See analysis |
| Best position | -2 | -2 | YES |
| Dtype | float64 | float64 | YES |
| Padding | left | left | YES |

### Analysis

If separations match closely (within 5%), the direction computation is correctly replicated.
Small differences may arise from:
- Different torch/transformers versions affecting numerical precision
- Model weight caching differences
- Chat template formatting differences

## 2. Attribution Comparison: Last-Layer (Mode A vs Tejas)

Mode A uses `measurement_layer=None` (last layer), matching Tejas's unpatched circuit-tracer.

| Metric | Tejas (v2) | Foundation (Mode A) | Delta |
|--------|-----------|---------------------|-------|
| Bare mean | 75.5 | 50.0 | -25.5 |
| JB mean | 56.7 | 36.9 | -19.8 |
| Mean diff | -18.7 | -13.0 | +5.7 |
| JB lower | 9/10 | 9/10 | |

### Interpretation

- If Mode A results are within ~20% of Tejas's, the pipeline is correctly replicating his setup.
- Exact matches are unlikely due to nondeterminism in GPU float operations.
- The key signal is: **does JB consistently have lower net attribution than bare?**
  This confirms the RP-suppresses-refusal hypothesis at the MLP level.

### Per-Pair Comparison

| Pair | Tejas Bare | Foundation Bare | Tejas JB | Foundation JB |
|------|-----------|----------------|---------|--------------|
| 1 | 75.3 | 49.2 | 56.1 | 35.3 |
| 2 | 76.7 | 51.1 | 74.6 | 48.8 |
| 3 | 79.2 | 52.5 | 54.1 | 36.5 |
| 4 | 76.0 | 50.5 | 58.3 | 37.5 |
| 5 | 92.6 | 62.5 | 91.5 | 61.9 |
| 6 | 69.7 | 45.4 | 60.1 | 38.9 |
| 7 | 87.3 | 58.0 | 69.8 | 45.4 |
| 8 | 74.1 | 49.4 | 48.3 | 32.0 |
| 9 | 74.6 | 50.8 | 86.2 | 58.1 |
| 10 | 49.0 | 30.1 | -31.4 | -25.2 |

## 3. Intermediate-Layer Attribution (Mode B) -- New

Mode B uses the measurement_layer patch from the AutoInterp circuit-tracer fork.
This measures R = <x^(l*, c*), r_hat> at the best refusal layer rather than the final layer.

**Why this matters**: The PDF (Section 1.4) notes that intermediate-layer measurement
traces only the upstream computation that *builds* the refusal signal, which is more
informative for understanding how the model decides to refuse.

| Metric | Mode A (last-layer) | Mode B (intermediate) | Delta |
|--------|--------------------|-----------------------|-------|
| Bare mean | 50.0 | 21.0 | -29.0 |
| JB mean | 36.9 | 16.8 | -20.1 |
| Mean diff | -13.0 | -4.2 | +8.9 |
| JB lower | 9/10 | 7/10 | |

### Analysis

Key questions:
1. Does intermediate-layer attribution show a **larger or smaller** JB suppression effect?
2. Does the number of active features change significantly?
3. Does the dampening vs tug-of-war distinction persist at intermediate layers?

## 4. Mechanism Comparison (RP vs Fiction vs Bare)

Same 9 prompts (3 topics x 3 jailbreak types) as Tejas's script 11 Task 4.

### Tejas Results (v2)

| Prompt | Net | Positive | Negative | N Features |
|--------|-----|----------|----------|------------|
| bare_lock | +4.1 | 93.3 | -89.2 | 8374 |
| rp_lock | -59.6 | 58.7 | -118.4 | 14024 |
| fiction_lock | -30.0 | 61.2 | -91.1 | 25024 |
| bare_hack | +53.8 | 116.8 | -63.0 | 9825 |
| rp_hack | -52.3 | 43.5 | -95.8 | 17552 |
| fiction_hack | -34.4 | 71.4 | -105.8 | 26141 |
| bare_phish | +61.7 | 119.5 | -57.7 | 8878 |
| rp_phish | -69.2 | 60.1 | -129.3 | 21676 |
| fiction_phish | -28.0 | 74.0 | -101.9 | 21237 |

### Foundation Results

| Prompt | Net | Positive | Negative | N Features |
|--------|-----|----------|----------|------------|
| bare_lock | -1.3 | 66.1 | -67.3 | 8374 |
| rp_lock | -43.5 | 42.2 | -85.7 | 14024 |
| fiction_lock | -21.9 | 44.4 | -66.3 | 25024 |
| bare_hack | +35.5 | 82.3 | -46.8 | 9825 |
| rp_hack | -37.5 | 30.8 | -68.4 | 17552 |
| fiction_hack | -25.8 | 51.2 | -77.0 | 26141 |
| bare_phish | +40.8 | 83.3 | -42.5 | 8878 |
| rp_phish | -50.9 | 43.0 | -93.9 | 21676 |
| fiction_phish | -21.4 | 53.2 | -74.6 | 21237 |

### Mechanism Classification

Tejas identified two distinct jailbreak mechanisms at the MLP level:

1. **Dampening (RP jailbreaks)**: Pro-refusal features drop aggressively,
   anti-refusal features grow moderately. The refusal circuit *disengages*.
   - Steerable via refusal direction (13/16 configs flip to refusal)

2. **Tug-of-war (Fiction jailbreaks)**: Both pro-refusal and anti-refusal
   features increase massively. The circuit *engages more* but the sides cancel.
   - Immune to refusal-direction steering (0/16 flip)

**Key question for foundation validation**: Does this pattern reproduce?

## 5. The Attention Dominance Gap

Tejas discovered that transcoder-based attribution captures only ~0.4% of the
total refusal signal:

| Metric | Value |
|--------|-------|
| Attribution sum (transcoders/MLPs) | ~75 |
| Dot product (full residual stream) | ~18,322 |
| MLP contribution | ~0.4% |
| Attention + embeddings | ~99.6% |

**Implication**: The mechanism analysis (dampening vs tug-of-war) describes
MLP behavior only. The main refusal circuit likely runs through **attention heads**,
which are not decomposed by transcoders.

This is a fundamental limitation of transcoder-only circuit tracing for refusal analysis.
To get the full picture, we need:
- Attention SAEs (available in gemma-scope for some models)
- Direct head attribution via TransformerLens
- Or hybrid approaches combining attention head attribution with transcoder attribution

## 6. Insights and Discussion

### What is similar and why

- **Refusal direction**: Both pipelines use identical methodology (Arditi et al. difference-in-means)
  with the same corrected parameters. Results should match closely.
- **Last-layer attribution (Mode A)**: Uses the same circuit-tracer API with CustomTarget.
  Minor differences from float precision, batch ordering, or version differences.
- **JB suppression pattern**: The directional finding (JB < bare) should be robust
  since it reflects genuine model behavior, not pipeline artifacts.

### What may be different and why

- **Intermediate-layer (Mode B)**: This is genuinely new. By measuring at layer 32
  instead of the final layer, we exclude post-L32 computation from the attribution.
  This could show a cleaner signal since we're measuring exactly where r_hat was computed.
- **Attribution magnitudes**: May differ due to transcoder version (`_affine` vs non-affine),
  dtype (float32 vs bfloat16), or circuit-tracer version differences.
- **Feature counts**: May differ based on JumpReLU threshold behavior across versions.

### What the results tell us about refusal and jailbreaks

1. **Refusal is real and measurable**: The 20,788 separation confirms a clear refusal
   direction exists in Gemma-3-4B-IT, with harmful and harmless on opposite sides of zero.

2. **Jailbreaks suppress refusal at the MLP level**: The -18.7 mean difference shows
   jailbreak prompts consistently reduce MLP-mediated refusal attribution.

3. **But MLPs are only 0.4% of the story**: Attention heads carry 99.6% of the refusal signal.
   The MLP-level mechanism (dampening vs tug-of-war) is a secondary effect.

4. **Different jailbreak types work differently**: RP jailbreaks dampen the circuit
   (and can be patched with steering), while fiction/analytical jailbreaks create
   a tug-of-war (and are immune to steering). This has implications for safety:
   - Steering-based defenses only work against dampening-type jailbreaks
   - Tug-of-war jailbreaks require different mitigation strategies

## 7. Follow-Up Experiments

### Priority 1: Attention Head Attribution
Since attention carries 99.6% of the refusal signal, analyzing attention heads is critical.
- Use attention SAEs from gemma-scope (if available for Gemma-3-4B)
- Or implement direct head attribution: for each head h at layer l,
  compute A_h = (attention_output_h @ r_hat) to identify which heads drive refusal
- Compare head attribution patterns between bare harmful and jailbreak prompts

### Priority 2: Refusal Vector Change Post-Jailbreak (from meeting notes)
Compute dot product of r_hat with W_i (layer weights) for each layer i.
Compare this profile between harmful and jailbroken prompts.
The difference classifies whether the jailbreak influences the refusal vector itself
or just the activations projected onto it.

### Priority 3: Top-K Feature Comparison (from meeting notes)
For each harmful prompt and its jailbroken variant:
1. Extract top-K features by |A_{s->R}|
2. Identify features with same ID but different attribution values
3. Identify features unique to jailbreak (not in harmful top-K)
4. Identify features unique to harmful (not in jailbreak top-K)
5. Focus on features that flip sign or change magnitude significantly

### Priority 4: Jailbreak Class Variance (from meeting notes)
Test different jailbreak classes systematically:
- RP (role-playing): 'As a security researcher...'
- Fiction: 'Write a scene where...'
- Analytical: 'Is this description correct...'
- Completion: 'Complete this manual entry...'
- Translation: reframe informal to formal
Do different classes mediate refusal differently? Are the top-K features the same?

### Priority 5: Attribution Net Mean on ALL Features (from meeting notes)
Rerun attribution without any feature filtering (max_feature_nodes=None)
and compute the net mean across all features, not just top-K.
This gives the true total MLP contribution to the refusal signal.

### Priority 6: Multi-Layer Measurement Sweep
Using the measurement_layer patch, sweep across layers 10-33 and plot:
- How does the attribution sum change with measurement layer?
- At which layer does the JB suppression effect peak?
- Does the dampening vs tug-of-war classification change at different layers?

### Priority 7: Cross-Model Comparison
Run the same pipeline on Gemma-2-2B-IT to check:
- Does the attention-dominance finding generalize?
- Are the jailbreak mechanisms (dampening vs tug-of-war) model-dependent?
- The smaller model is faster to iterate on and fits on consumer GPUs

---

*Report generated by `scripts/generate_comparison_report.py`*