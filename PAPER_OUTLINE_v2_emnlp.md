# Refusal-Lens — Paper Outline v2 (EMNLP 2026 main)

**Track**: EMNLP 2026 main conference.
**Status**: Major framing pivot 2026-05-24 after meeting with Georg. Magnitude-gap thesis dropped; replaced with circuit-tracer-as-behavior-editor thesis.
**Companion document**: `EMNLP_RESULTS_LOG.md` (Batches 1-17). Experiment plan: `docs/superpowers/plans/2026-05-24-natural-circuit-edits.md`.

---

## 0. The Georg pivot (2026-05-24)

The earlier "magnitude gap between edge ablation and direction intervention" framing has been retired. Two reasons:

1. **Reviewer fragility**: "Subtracting the entire refusal direction at every position is stronger than ablating individual feature edges" is a *trivial* result. A reviewer would correctly point out that the Arditi-style intervention is an artificial maximum that the model could never produce on its own. Claiming a gap against an artificial ceiling is not a publishable contribution.

2. **Cost-to-completeness**: Even if we did pursue the gap framing, explaining *why* the gap exists (the actual scientific content) is much more work than a 3-week sprint allows. Better to invest that time in a paper that asks a more pointed question.

The new paper is about **how circuit-tracer attribution graphs let us identify natural, interpretable feature edits that steer refusal behavior in a way the model could plausibly produce on its own**. The baseline shifts from "the unnatural full-Arditi intervention" to "the most natural intervention we can construct from the same probe machinery."

---

## 1. Title (drafts in order of preference)

1. **"Editing Refusal via Circuit Tracing: From Probe Geometry to Natural Behavioral Edits"**
2. "Natural Circuit Edits for Refusal Control in Open-Weights LLMs"
3. "Beyond Arditi: Circuit-Tracer-Guided Interventions for Refusal Behavior"

## 2. One-sentence thesis

> A learned refusal direction `r̂` in Gemma-3-4B-IT and Qwen3-4B can be decomposed via circuit-tracer attribution graphs into per-feature contributions, and the same decomposition tells us which features to ablate to achieve refusal flips comparable to a **position-wise direction subtraction** — a natural intervention that the model could in principle produce on its own from prompt context, unlike the unnatural full-Arditi push.

---

## 3. Contributions (revised, 3-in-priority-order)

1. **A natural baseline for refusal interventions**: position-wise direction subtraction (subtract `r̂_pos=-i` at each position `-i`) is much closer to what the model could spontaneously produce given a harmless or jail-broken prompt, and bounds where any "feature-level" edit could realistically achieve. **This becomes the headline benchmark, not Arditi**.
2. **Circuit-tracer-guided feature edits**: identifying which features to ablate *and which to add back* using the harmful↔JB-broken attribution-graph delta, and showing whether this can match the natural baseline's flip rate.
3. **Cross-model validation**: replicate on Qwen3-4B at L18 using a fresh set of attribution graphs we publish (`moon70/refusal-lens-graphs`).

## 4. Headline figure (re-scoped)

`controllability_natural_figure.png` — 4 panels (to be generated):
- **A**: Dose-response curve of position-wise direction subtraction (the natural baseline) on Gemma L15 — how flip rate scales with how many positions we apply at.
- **B**: Edge-ablation flip rates (active-only vs active+newly-active feature sets) compared to the natural baseline.
- **C**: Layer locator on the natural baseline (where in the network does position-wise subtraction work).
- **D**: Qwen3-4B replication of the headline result.

The old 4-panel `controllability_extension_figure.png` (Batch 14) becomes supplementary, illustrating the magnitude-gap finding as context for why we pivoted to natural interventions.

---

## 5. Section structure (8-page main conference)

### § 1. Introduction (≈ 1 page)
- The mechanistic-interpretability promise: identify causal features via probes + attribution → edit those features to control behavior.
- A common but unaddressed gap: behavioral baselines used for "is this intervention effective?" (e.g., Arditi steering) are artificial — they apply the full direction at every position simultaneously, an intervention the model could never produce on its own.
- We propose a **natural baseline**: position-wise direction subtraction (different `r̂` per position, calibrated from harmful↔harmless contrast at that position).
- We show that circuit-tracer-guided feature edits can match this natural baseline — provided we include both *deactivated* and *newly-activated* features from the harmful↔JB attribution-graph delta.

### § 2. Background and Setup (≈ ¾ page)
- 2.1 Arditi-style refusal direction (mean-diff over harmful/harmless prompts)
- 2.2 Per-position refusal directions (Stage 01 output: `positions_LX/pos_-N.pt`)
- 2.3 Circuit-tracer attribution graphs at the chosen layer (CLT-based decomposition)
- 2.4 Models: Gemma-3-4B-IT at L15, Qwen3-4B at L18 (Ruqiya's pipeline)
- 2.5 Controlled dataset (50 prompts × 11 conditions)

### § 3. The Natural-vs-Unnatural Spectrum of Direction Interventions (≈ 1 page)
- 3.1 The "Arditi maximum": subtract canonical `r̂` at every position → ~100% flip on Gemma, ~95% on Qwen (Batch 17). *Note: this is upper-bound; model can't do this on its own.*
- 3.2 The "single-position minimum": subtract `r̂_pos=-2` only at pos=-2 → ~20% flip on Gemma, ~45% on Qwen pos=-1.
- 3.3 The **natural middle**: subtract `r̂_pos=-i` at each position `-i` → expected flip rate somewhere in between, expressing what the model could plausibly do given different prompt contexts. **[pending — Phase 1 E1/E2]**
- 3.4 **Methodological aside — the magnitude framework is rigorous** (1 paragraph or callout box). The Batch 17 supra-antirefuse experiment scales edge-derived deltas by `N` in anti-refuse direction and produces a dose-response curve that **matches EXP 1 at the equivalent effective coefficient to within Wilson noise**. Pooled-JB-refuse counts are identical at coeffs ≥ 0.05. This validates that our intervention math is well-defined and our interpretations of dose-response across this paper rest on rigorous accounting. (Cite as supporting evidence; not the main result.)
- 3.5 Replicating on Qwen3-4B at L18: dose-response inflection at coeff ≈ 0.3-0.4 (vs Gemma's ≈ 0.2), similar shape.

### § 4. Circuit-Tracer Decomposition and Feature Identification (≈ 1¼ pages)
- 4.1 Linearization identity at the target layer: `h · r̂ = Σ edges + baseline_offset`
- 4.2 Active features in the harmful attribution graph: features that contribute non-zero attribution to the target.
- 4.3 **Cross-model decomposition magnitudes are NOT equivalent** (new — from Batch 17). The edge-derived effective coefficient (what the linearization predicts as the "ablate-all-edges" magnitude in the Arditi-canonical basis) is:
  - Gemma at L15: **0.005** — far below the dose-response inflection point (~0.2).
  - Qwen at L18: **1.044** — at the inflection point.

  Direct consequence: `ablate_all_edges` flips 6% of bare prompts on Gemma but **95% on Qwen**, even though both use the same conceptual operation (subtract the predicted edge-bucket contribution from the residual). This is not a methodological inconsistency — it's a structural property of how each model's CLT decomposition projects edge contributions relative to behavioral inflection. **This is itself a paper-worthy observation**: papers claiming "edge ablation tests causality" must explicitly report where their edge-derived coefficient sits relative to their model's dose-response inflection, because that ratio is model-dependent.

- 4.4 The harmful↔JB-broken attribution-graph delta — Georg's circuit-edit hypothesis:
  - Features active in *harmful* but not in *JB-broken*: candidates to *ablate* (the "refusal-promoting" features that the jailbreak suppresses)
  - Features active in *JB-broken* but not in *harmful*: candidates to *add* (the "compliance-supporting" features the jailbreak introduces)
- 4.5 Why active-only ablation underperforms on Gemma but already works on Qwen: on Qwen, the edge-derived magnitude happens to be at the inflection; on Gemma it's sub-threshold by a factor of ~200. Adding back the newly-activated features is hypothesized to close the Gemma gap.

### § 5. Natural Edits vs Circuit Edits: Closing the Loop (≈ 1¾ pages)
- 5.1 Setup: compare three intervention classes:
  - (a) Position-wise direction subtraction (the natural baseline)
  - (b) Active-feature ablation only (what we did in Batches 12-15)
  - (c) Active-feature ablation + newly-active feature addition (the proposed extension)
- 5.2 Per-prompt comparison: does (c) match (a)'s flip rate? On which conditions?
- 5.3 Sanity: are the deactivated and newly-activated feature sets stable across prompts in a class? Or per-prompt idiosyncratic?
- 5.4 Cross-model — and the model-dependence of (b): on Qwen, active-only ablation already flips 95% bare (from Batch 17) because the edge-derived effective coefficient happens to land at the inflection point. On Gemma, active-only sits at 6%. The "active + newly-active" extension is therefore tested most rigorously on Gemma where the gap is large; on Qwen it's a sanity check that the methodology generalizes. **This per-model asymmetry — where the same circuit-edit operation produces vastly different behavioral effects depending on how the CLT decomposition lands relative to inflection — is one of the paper's secondary findings.**

### § 6. Layer-Broad Redundancy (≈ ¾ page)
- 6.1 Where in the network does position-wise subtraction work (layer locator on the natural baseline). **[pending — requires magnitude-normalized layer sweep, see § 6.3]**
- 6.2 Where in the network do circuit edits work (layer locator on (c) above).
- 6.3 **Methodological caveat on layer comparisons** (new — from Batch 17): naive layer locators at fixed `coeff=1.0` are confounded by the per-layer growth of ‖r‖ — Qwen's ‖r‖ ranges from 0.2 at L0 to 260 at L34, so a "constant coefficient" sweep applies vastly different absolute edits to h at each layer. A fair comparison normalizes either by edit magnitude (e.g., ‖edit‖=15 at every layer) or by sweeping coefficients per-layer until each layer's flip-rate threshold is crossed. The Batch 14 Gemma layer locator is largely valid (Gemma's ‖r‖ varies less with depth), but Qwen's needs to be re-run with magnitude normalization before being cited as evidence about cross-layer intervention efficacy. We adopt the magnitude-normalized convention going forward.
- 6.4 Implication: refusal is broadly encoded; feature edits at a single layer recover only part of the natural-baseline flip rate.

### § 7. Discussion (≈ 1 page)
- 7.1 What "natural" means: the model could in principle output the same residual stream geometry given a different prompt; our edit gets it there. Unnatural interventions (full-Arditi) move the residual to a configuration unreachable by any prompt.
- 7.2 Implication for mech-interp: most published feature-ablation results benchmark against unnatural ceilings. Natural baselines change the question from "did you flip behavior?" to "did you achieve the same flip the model could do on its own?".
- 7.3 **The model-dependence of attribution-graph magnitudes** (1 paragraph; cites Batch 17). The edge-derived effective coefficient — the magnitude of `Σ edges / ‖r‖²` expressed in Arditi-canonical basis — is **0.005 on Gemma but 1.044 on Qwen**. The decomposition is *mathematically* the same operation in both cases, but the projection's behavioral coverage differs by ~200×. We suggest this is a structural property of how each model's CLT-based attribution decomposes the projection (how much of `direct_dot` ends up in `Σ edges` vs `baseline_offset`), and recommend that future work explicitly report this ratio when benchmarking edge-ablation interventions. Without it, "edge ablation flipped X%" is uninterpretable across models.
- 7.4 Connections to Arditi 2024, Ball 2024, Wang 2025.

### § 8. Limitations (≈ ¼ page)
- Two model families (Gemma + Qwen, both 4B).
- Keyword classifier; same classifier-noise caveat as Batch 14 stands.
- n=50 prompts.
- Refusal-specific behavior; generalization to other behaviors (sycophancy, etc.) is open.

### § 9. Conclusion (≈ ¼ page)
- "The right benchmark for circuit-level interventions is what the model could do on its own, not what we can do to it by force. Under that benchmark, circuit-tracer-guided edits recover most of the natural intervention's effect — when we include both deactivated and newly-activated features."

---

## 6. Figure plan

| # | Content | Source |
|---|---|---|
| **F1** | Concept diagram: spectrum of interventions (unnatural Arditi → natural per-position → circuit edits) | NEW illustrator |
| **F2** | r̂ probe identification + per-position directions across positions -1..-15 | Per-position direction data already exists; new figure |
| **F3** | Natural-vs-unnatural intervention dose-response (Arditi all positions vs per-position vs pos=-2 only) | NEW experiment + figure |
| **F4** | Active-only ablation vs active+newly-active ablation comparison | NEW experiment + figure |
| **F5 (HEADLINE)** | 4-panel — natural-baseline dose-response (Gemma), edge ablation comparison, layer locator on natural baseline, Qwen replication | NEW |

---

## 7. Experiment status

### ✅ Complete (Gemma-3-4B-IT, valid for new framing)

| Experiment | Result | Where |
|---|---|---|
| Stage 01 (r̂) | clean probe at L15, ‖r̂‖=3101.2; per-position files in `positions_L15/` | run_20260430_023247 |
| Stage 02 (attribution graphs) | 550 single-mode at L15 pos=-2 | moon70/refusal-lens-graphs |
| Stage 04 (feature labels) | universal refusal cluster + per-class features | run_20260430_023247/04_labels/ |
| Stage 06 baseline (Arditi all positions) | 98% bare→COMPLY (now framed as upper bound, not target) | Stage 06 doc |
| Phase 0 0a linearization | `linearization_decomposition.json` | Batch 4 |
| Phase 0 0b edge ablation (7 variants) | pooled JB-refuse 6.7% (now framed as "active-only ablation underperforms") | Batch 12 |
| Phase 0 0d/0e Pareto sweeps | flat across K | Batch 12 |
| Batch 14 EXP 1 — direction sweep all positions | 100% at coeff=1.0; inflection ≈ 0.18 | EXP 1 |
| Batch 14 EXP 2 — direction sweep pos=-2 only | 20% bare flip at coeff=1.0 | EXP 2 |
| Batch 14 EXP 4 — layer locator | L12-L24 band, peak L15-L18 | EXP 4 |

### ✅ Complete (Qwen3-4B replication, valid for new framing)

| Experiment | Result | Where |
|---|---|---|
| Qwen Stage 02 attribution graphs at L18 | 550 single-mode at L18 pos=-1; ‖r‖=15.14 | moon70/refusal-lens-graphs (run_emnlp_qwen_L18_20260522) |
| Qwen 0a linearization | `qwen_linearization_decomposition.json` (all_signed median +15.81) | Batch 15 |
| **Qwen 0b edge ablation** (Batch 17) | **bare 95% / pooled JB 77% / pooled CTRL 66%** — supra-threshold, contradicts Gemma's sub-threshold | Batch 17 |
| **Qwen direction sweep all positions** (Batch 17) | dose-response inflection at coeff ≈ 0.3-0.4; coeff=1.0 → 95% bare | Batch 17 |
| **Qwen direction sweep pos=-1 only** (Batch 17) | coeff=1.0 → 45% bare (similar shape to Gemma's 20% pos=-2-only) | Batch 17 |
| **Qwen layer locator** (Batch 17) | L0-L15 → 0%; L18-L34 → 45-90% — but **confounded by ‖r‖ growth with depth**; needs magnitude-normalized re-run | Batch 17 |

### ✅ Complete (Gemma supra-threshold closure)

| Experiment | Result | Where |
|---|---|---|
| **Gemma supra-antirefuse** (Batch 17) | Sign-corrected supra-threshold matches EXP 1 dose-response curve to within Wilson noise — empirical closure of the magnitude framework | Batch 17 |

### 📋 New experiments (proposed for the new framing)

**Priority 1 — the new headline experiments**:

| Experiment | Purpose | Estimate |
|---|---|---|
| **Position-wise direction subtraction** (Gemma L15) | The new "natural baseline". Subtract `r̂_pos=-i` at each position `-i` simultaneously, sweeping how many positions we apply at. | ~3-4 hr GPU, ~$15 |
| **Position-wise direction subtraction** (Qwen L18) | Same as above on Qwen | ~3-4 hr, ~$15 |
| **Feature-delta clustering** (CPU analysis on existing graphs) | For each (prompt, JB class), compute the active-set difference between harmful and JB-broken graphs. Identify *deactivated* (active in harmful, gone in JB) and *newly-activated* (gone in harmful, active in JB) features. | ~30 min CPU |
| **Active-only vs active+newly-active ablation** (Gemma L15) | The 0b extension. For each prompt, ablate the deactivated set AND add back the newly-activated set. Compare to active-only ablation flip rate and to natural-baseline flip rate. | ~4 hr GPU, ~$15 |
| **Active+newly-active on Qwen L18** | Same on Qwen | ~4 hr GPU, ~$15 |

**Priority 2 — sanity checks**:

| Verification | Status |
|---|---|
| **Per-position alpha is computed correctly** (Georg-flagged) | **AUDITED**: 0a emits a single scalar `all_signed` per (prompt, condition), summed over all source positions. The hook applies this scalar uniformly at every position when `position_mode="all"`. To do *true* per-position alpha, we need 0a to also emit per-source-position deltas. **Tracked as E0 in `docs/superpowers/plans/2026-05-24-natural-circuit-edits.md`** (Priority 1). |
| **Position-wise direction file integrity** | **CONFIRMED**: `positions_L15/pos_-N.pt` (Gemma) and `positions_L18/pos_-N.pt` (Qwen) load with expected norms; verified by smoke test. |
| **Qwen layer locator needs magnitude-normalized re-run** | New (from Batch 17). Cited as a caveat; not blocking submission if we don't redo it, but we should flag in the limitations or do a small follow-up. |

**Priority 3 — deferred**:

| Experiment | Notes |
|---|---|
| LLM-judge classifier validation | Same as before — still useful but not blocking submission |
| Llama-3-8B / Mistral-7B replication | If reviewers push for more models, replicate after submission |

---

## 8. Cuts from earlier outline

The following were the centerpiece of the old (magnitude-gap) framing. They are now relegated to supplementary / context material:

- **The 2×2 magnitude × position table** → demoted to a supplementary "why we initially thought there was a gap" callout in § 3.
- **The 36× / 200× quantification** → still in the discussion as background, but not load-bearing.
- **"L15 is a lever, not just a probe" reframe** → preserved as a true side-finding but not the headline.
- **Supra-threshold edge-ablation scaling** → supplementary; demonstrates that the magnitude story is empirically real but doesn't address the actual question.

The Batch 14 + Batch 15 data are not discarded — they become **background and supporting figures**, not the main claim.

---

## 9. Timeline (revised)

Assuming Path A re-run completes and we kick off the new experiments shortly after:

| Day | Work |
|---|---|
| Day 0 (today, 2026-05-24) | Path A still running; verify alpha computation on the existing codebase |
| Day 1 | Path A lands; CPU feature-delta clustering analysis |
| Day 2-3 | Position-wise direction subtraction (Gemma + Qwen); analyze results |
| Day 4-5 | Active+newly-active ablation (Gemma + Qwen) |
| Day 6-8 | Draft § 3, § 4, § 5 of paper |
| Day 9-11 | Draft § 1, § 2, § 6, § 7, § 8, § 9; figures |
| Day 12-14 | Internal review, revisions |
| Day 15-21 | Slack with Tejas/Georg/Ruqiya; final polish; submit |

Tight but feasible if we don't get blocked on the new experiments.

---

## 10. Open questions raised by Georg

1. **Per-position alpha**: was the edge-ablation alpha computed per-position from each position's attribution, or averaged across positions? — **AUDITED 2026-05-24**: 0a emits one scalar per (prompt, condition); hook applies uniformly. True per-position alpha requires extending 0a (planned as E0 in Phase 1).
2. **Why does edge ablation flatline at ~10% regardless of position-mode on Gemma?** Now partly explained by Batch 17 cross-model data: the edge-derived effective coefficient on Gemma (0.005) is far below the inflection. On Qwen, the same operation flips 95% because Qwen's edge-derived coefficient (1.044) sits at the inflection. So the "flatline" on Gemma is real and explainable; the per-position-mode insensitivity is consistent with both modes being sub-threshold.
3. **Are deactivated + newly-activated features sufficient, or do we need additional combinations?** Open. E4 will answer empirically. May need iterative experimentation.

---

*Last updated 2026-05-24 after Batch 17 + Georg pivot. Paper thesis: natural-circuit-intervention. Old magnitude-gap framing demoted to supplementary; new headline = circuit edits can match natural baseline. Supra-antirefuse closure (Batch 17) and cross-model edge-ablation contrast (Gemma 6% vs Qwen 95%) integrated as supporting evidence in §3 / §4 / §7.*
