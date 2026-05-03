# Paper outline — _A 1-D Refusal Axis Across Open-Weight Families: Replicating the Gemma-3 Refusal Mechanism on Qwen3_

**Status:** v1 outline. Author/venue/length TBD.

## Abstract (~200 words)

We test whether the "refusal direction" — a single 1-dimensional residual-stream axis whose addition forces an LLM to refuse and whose subtraction forces it to comply (Arditi et al., 2024) — is a family-invariant phenomenon or a Gemma-specific artefact. We apply an identical pipeline to Gemma-3-4b-it (recreating the L15 result of Tejas et al.) and to Qwen3-4B, controlling everything that can be controlled: same 50-prompt × 11-condition controlled dataset (50 harmful base prompts × 5 jailbreak templates × 5 length-matched neutral controls + bare), same regex refusal classifier, same circuit-tracer attribution stack, same difference-in-means direction extraction. Across both models we (a) compute the per-layer refusal direction `r̂`, (b) attribute MLP-feature contributions to `r̂` at the model's best causal layer using a cross-layer transcoder, (c) ablate the resulting subcircuits to test class-level dissociation, and (d) verify causal sufficiency via Arditi-style direction addition / subtraction at every layer. We report the cross-family comparison along three axes: (1) where the refusal axis lives (best causal vs best separation layer), (2) how strongly it carries the refusal decision (flip rate under direction addition), and (3) whether the same per-class subcircuit dissociation that holds on Gemma also holds on Qwen.

## 1. Introduction

- The refusal-axis claim (Arditi et al.): one direction in residual stream space governs whether an instruction-tuned LLM refuses.
- Why families matter: the claim has been demonstrated on Llama-2, Llama-3, Vicuna, and now Gemma-3, but each within its own paper. **A like-for-like replication on a model from a *different family* (Qwen3 — a Chinese-trained, multilingual, post-RMSNorm, trillion-token-scale family) is a stronger universality test than within-family replication.**
- What this paper *adds* over prior work:
  1. **Same pipeline, two families.** Identical methodology (data, classifier, transcoder framework, hookpoints) — divergences are model-architectural, not methodological.
  2. **A cross-family dissociation matrix.** We extend the standard "is `r` causal" test (flip rate) to "is the *internal decomposition* of refusal consistent" (per-class subcircuit ablation).
  3. **A per-position separation curve as a diagnostic.** We show that the position at which `r` is maximally separable differs between Gemma's `model` token (-2) and Qwen's `\n` (-1), and discuss why that matters for any "use the refusal direction" tooling.

- Contributions list (3–4 bullets).

## 2. Related work

- Arditi et al., 2024 — refusal direction.
- Lee et al., 2024 / Anthropic, 2024 — circuit-level decomposition with cross-layer transcoders.
- Gemma-Scope (Mwhanna et al.) — transcoder weights for Gemma-3-4b-it.
- mwhanna/qwen3-4b-transcoders — transcoder weights for Qwen3-4B (160k width).
- Activation steering, representation engineering literature.

## 3. Method

### 3.1 Models and dataset

- Gemma-3-4b-it (`google/gemma-3-4b-it`, 34 layers, vision-LM wrapper).
- Qwen3-4B (`Qwen/Qwen3-4B`, 36 layers, flat causal-LM, `enable_thinking=False`).
- 50 harmful prompts × (1 bare + 5 JB classes × 2 prefix variants {jb, ctrl}) = 11 conditions × 50 = **550 prompt-condition pairs**.
- Regex classifier (19 phrases, applied identically to both models).

### 3.2 Pipeline (identical for both families)

| Stage | What it computes | Output |
|---|---|---|
| 01 | per-layer refusal direction `r̂_L` (diff-in-means at trailing position) | per-layer `.pt` + per-position `.pt` at causal layer |
| 02 | attribution graphs at best-causal layer L<sub>c</sub>, target = `r̂_L_c` at template anchors | `.pt` graphs + per-prompt feature lists |
| 02b | paired Wilcoxon, Cohen's d, 95 % CI for bare↔ctrl↔jb deltas | effect-size tables |
| 02c | pack `.pt → .json.gz` | shareable graph bundle |
| 03 | reconcile Σ-edges with direct-dot of residual onto `r̂` | verification ratio |
| 04 | label features against transcoder dashboards | feature_labels.json |
| 05 | interactive frontend (overlay subcircuit membership) | static HTML/JS |
| 06 | Arditi causal intervention (`±r` at L<sub>c</sub>, every position) | per-condition flip rate |
| 07 | rule-based subcircuit identification (universal core, ctrl-shared, jb-class-specific) | subcircuits.json |
| 08 | zero-ablation of each subcircuit, measure per-class JB-recovery / bare-break | dissociation matrix |

### 3.3 Architectural divergences (the controlled-replication table)

Same as the table in [COMPARISON_REPORT_gemma_vs_qwen.md](COMPARISON_REPORT_gemma_vs_qwen.md). Highlighted:

- Decoder access path (`model.model.language_model.layers` vs `model.model.layers`).
- Pre-MLP LN (`pre_feedforward_layernorm` vs `post_attention_layernorm`).
- Best causal layer position (template-token differs).
- Transcoder width (16k vs 160k).

We argue these are the **only** divergences; the methodology is otherwise bit-equivalent. Code: `scripts/pipeline/` (Gemma) and `scripts/pipeline_qwen/` (Qwen), each a sibling directory differing only where forced by the architectural divergences above.

## 4. Results

### 4.1 Per-layer separation curves

- Gemma: sharp L32 spike (`~20,873`).
- Qwen: TBD curve. Either flat (distributed refusal) or a different layer's spike. _Figure: 2-panel separation_by_layer, side-by-side._

### 4.2 Best-causal layer flip rates (Stage 06)

- Gemma L15: 100 % pro-add, 98 % anti-sub.
- Qwen L<sub>TBD</sub>: TBD pro-add, TBD anti-sub.

_Table: per-class flip rate (5 JB classes × 2 directions × 2 models = 20 cells, plus benign force-refuse control)._

### 4.3 Coherent flip rate

We report flip rate filtered to coherent responses only (`is_coherent` heuristic from `utils.py`). Same filter on both models. _Figure: stacked bar — coherent / incoherent / no-flip per class per model._

### 4.4 The dissociation matrix (Stage 08)

The headline figure. 5×5 (or 6×6 with universal-core row) matrix per model, plus a delta matrix |Gemma − Qwen|. _Figure: 3-panel heatmap._

### 4.5 Per-position direction at the causal layer

The `r̂` rotates substantially across positions within a layer (Tejas finding on Gemma: cos(L15-pos=-2, L15-pos=-5) = -0.80 — anti-correlated). _Figure: cos heatmap of (layer, position) directions for both models._

### 4.6 Methodology controls

- Bare refuse rate.
- Ctrl refuse rate.
- Manual classifier audit (precision/recall on N=50 sample, both models).
- Σ-edges = direct-dot reconciliation (Stage 03).

## 5. Discussion

### 5.1 Universality of the 1-D claim

If both models hit ≥80 % flip rate (both directions), the refusal axis is family-invariant within "post-RMSNorm decoder LMs trained on roughly comparable instruction-tuning recipes". If one model fails, what specifically blocks 1-D bypass?

### 5.2 Where refusal lives architecturally

- Gemma localises strongly at L32 (separation) but acts at L15 (causal).
- Qwen result will tell us whether this two-layer split is an architectural feature of the family (post-LN spike layer) or a coincidence.

### 5.3 Subcircuit semantics across families

If `jb_fiction_specific_vs_ctrl` produces different feature *sets* on Gemma vs Qwen but the *same behavioral dissociation*, that's evidence that the refusal-decision computation is the same algorithm implemented in different feature bases. Stronger universality claim than just "the axis exists".

### 5.4 Limitations

- Transcoders cover MLP only; ~99.6 % of the L32 refusal signal in Gemma is in attention + embeds. Conclusions about "feature X explains refusal" are bounded to the MLP slice (same caveat for Qwen, though attribution share TBD).
- Single-prompt-condition flip rates are 50-prompt aggregates; no per-prompt confidence intervals (would need bootstrapping).
- We use a regex classifier rather than a strong-LLM judge; we report classifier precision/recall as a control.

## 6. Conclusion

(One paragraph. The headline depends on the GPU runs.)

## Appendix

- A. Hookpoint resolution table (TransformerLens names per family).
- B. Bare-comply prompt list per model (excluded from anti-direction substrate).
- C. Per-class ctrl-leak pair list per model.
- D. Code and replication: GitHub `AutoInterp/Refusal-Lens` (`qwen3-pipeline-scaffold` branch); HF datasets `AutoInterp/refusal-lens-graphs` (Gemma) and `AutoInterp/refusal-lens-graphs-qwen` (Qwen).

## Figure list

1. `separation_by_layer` — Gemma vs Qwen, 2 panels.
2. `flip_rate_by_layer` — Qwen 01b sweep result; Gemma reference Tejas Script 16 if rerunnable.
3. `flip_rate_by_class` — Gemma L15 (existing) + Qwen L_TBD, side-by-side.
4. **`dissociation_matrix`** — headline, 3 panels (Gemma | Qwen | |Gemma − Qwen|).
5. `cosine_heatmap_per_position` — direction rotation across positions within causal layer, both models.
6. `intervention_symmetry` — pro/anti flip-rate side-by-side per model.
7. `feature_class_overlap` — per-class subcircuit feature-set overlap, both models.

## Table list

T1. Setup divergences (Section 3.3).
T2. Per-layer separation peak vs causal layer (Section 4.1).
T3. Per-class flip rate, 2 models × 5 classes × 2 directions (Section 4.2).
T4. Σ-edges / direct-dot ratio + classifier audit precision/recall (Section 4.6).
T5. Subcircuit feature counts and per-class recovery rates (Section 4.4).

## Open questions deferred to v2

1. Cross-model **feature transfer** — does an `r̂` extracted on Gemma applied at the analogous Qwen layer have any non-trivial effect (transferred-direction control)?
2. **Multi-token interventions** — the current Arditi `r ± h` is applied at every position; can we achieve the same flip with intervention at the template-anchor positions only? (Stage 08 `--positions anchors` exists; report is class-level.)
3. **Cross-language**, since Qwen is multilingual: does the refusal axis transfer to non-English JBs?
