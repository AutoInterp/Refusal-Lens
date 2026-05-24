# Refusal-Lens — Paper Outline v2 (EMNLP 2026 main)

**Track**: EMNLP 2026 main conference.
**Status**: Major framing pivot 2026-05-24 after meeting with Georg. Magnitude-gap thesis dropped; replaced with circuit-tracer-as-behavior-editor thesis.
**Companion document**: `EMNLP_RESULTS_LOG.md` (Batches 1-16).

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
- 3.1 The "Arditi maximum": subtract canonical `r̂` at every position → ~100% flip on Gemma. *Note: this is upper-bound; model can't do this on its own.*
- 3.2 The "single-position minimum": subtract `r̂_pos=-2` only at pos=-2 → ~20% flip on Gemma.
- 3.3 The **natural middle**: subtract `r̂_pos=-i` at each position `-i` → expected flip rate somewhere in between, expressing what the model could plausibly do given different prompt contexts.
- 3.4 Replicating on Qwen3-4B at L18.

### § 4. Circuit-Tracer Decomposition and Feature Identification (≈ 1 page)
- 4.1 Linearization identity at the target layer: `h · r̂ = Σ edges + baseline_offset`
- 4.2 Active features in the harmful attribution graph: features that contribute non-zero attribution to the target.
- 4.3 The harmful↔JB-broken attribution-graph delta:
  - Features active in *harmful* but not in *JB-broken*: candidates to *ablate* (these are the "refusal-promoting" features that the jailbreak prompt suppresses)
  - Features active in *JB-broken* but not in *harmful*: candidates to *add* (these are the features the jailbreak prompt activates in lieu)
- 4.4 Why we couldn't see this with active-only ablation: removing the refusal-promoting set is not enough on its own; you also need to add the "compliance-supporting" features that jailbreaks naturally activate.

### § 5. Natural Edits vs Circuit Edits: Closing the Loop (≈ 1¾ pages)
- 5.1 Setup: compare three intervention classes:
  - (a) Position-wise direction subtraction (the natural baseline)
  - (b) Active-feature ablation only (what we did in Batches 12-15)
  - (c) Active-feature ablation + newly-active feature addition (the proposed extension)
- 5.2 Per-prompt comparison: does (c) match (a)'s flip rate? On which conditions?
- 5.3 Sanity: are the deactivated and newly-activated feature sets stable across prompts in a class? Or per-prompt idiosyncratic?
- 5.4 [pending] Cross-model: do the same combinations work on Qwen at L18?

### § 6. Layer-Broad Redundancy (≈ ¾ page)
- 6.1 Where in the network does position-wise subtraction work (layer locator on the natural baseline).
- 6.2 Where in the network do circuit edits work (layer locator on (c) above).
- 6.3 Implication: refusal is broadly encoded; feature edits at a single layer recover only part of the natural-baseline flip rate.

### § 7. Discussion (≈ ¾ page)
- 7.1 What "natural" means: the model could in principle output the same residual stream geometry given a different prompt; our edit gets it there. Unnatural interventions (full-Arditi) move the residual to a configuration unreachable by any prompt.
- 7.2 Implication for mech-interp: most published feature-ablation results benchmark against unnatural ceilings. Natural baselines change the question from "did you flip behavior?" to "did you achieve the same flip the model could do on its own?".
- 7.3 Connections to Arditi 2024, Ball 2024, Wang 2025.

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

### ✅ Complete (Qwen3-4B replication of magnitude data, now reframed)

| Experiment | Result | Where |
|---|---|---|
| Qwen Stage 02 attribution graphs at L18 | 550 single-mode at L18 pos=-1; ||r||=15.14 | moon70/refusal-lens-graphs (run_emnlp_qwen_L18_20260522) |
| Qwen 0a linearization | `qwen_linearization_decomposition.json` | Batch 15 |

### 🔄 In progress (Path A re-run — partially still useful)

| Experiment | Status | Useful under new framing? |
|---|---|---|
| Qwen 0b edge ablation with unnormalized r + enable_thinking=False | Running | Yes — gives us the active-feature ablation baseline on Qwen |
| Qwen direction sweep all positions | Running | Yes — gives us the unnatural upper bound on Qwen for context |
| Qwen direction sweep pos=-1 only | Running | Yes — single-position lower bound for Qwen |
| Qwen layer locator | Running | Yes — band structure on Qwen |
| Gemma supra-threshold (antirefuse variants) | Running | **Lower priority** — confirms magnitude scaling but no longer the headline |

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

| Verification | Why |
|---|---|
| **Per-position alpha is computed correctly** | Georg flagged: Mahmoud said "before I was taking the average across all positions rather than a per-position alpha". Need to verify the code actually computes per-position alpha now, not an average. Affects Batch 14 EXP 2 (pos=-2 only) interpretation. |
| **Position-wise direction file integrity** | Verify per-position direction files `positions_L15/pos_-N.pt` and `positions_L18/pos_-N.pt` are loaded correctly with their unnormalized norms. |

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

1. **Per-position alpha**: was the edge-ablation alpha computed per-position from each position's attribution, or averaged across positions? (Code check required.)
2. **Why does edge ablation flatline at ~10% regardless of position-mode?** Georg said "I'm quite surprised that doing ablation at all doesn't improve anything." Maybe a sign of:
   - Active-set being too narrow (the newly-activated features hypothesis above)
   - Or the alpha computation bug above
   - Or genuinely sub-threshold edits (but then position-wise direction should also be sub-threshold)
3. **Are deactivated + newly-activated features sufficient, or do we need additional combinations?** Open. May need iterative experimentation.

---

*Last updated 2026-05-24 after Georg meeting — paper pivots from magnitude-gap to natural-circuit-intervention thesis. Old framing demoted to supplementary; new headline = circuit edits can match natural baseline when we include both deactivated and newly-activated features.*
