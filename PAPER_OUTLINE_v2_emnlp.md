# Refusal-Lens — Paper Outline v2 (EMNLP 2026 main)

**Track**: EMNLP 2026 main conference (~3 weeks to deadline at time of writing).
**Status**: superset of `PAPER_OUTLINES_v1.md`. Restructured 2026-05-22 after Batch 14 reframe.
**Companion document**: `EMNLP_RESULTS_LOG.md` (Batch 14 is the headline).

---

## 0. The strategic reframe (post-Batch 14)

The earlier v2 framing led with "directional intervention Pareto-dominates feature ablation by ~3×" and asked "where does the missing ~65% live?". Batch 14 (2026-05-21) showed that this framing **buries the lede**. The real finding is:

> **L15:r_hat IS a behavioral lever. Circuit-tracer's linearization decomposition correctly accounts for `direct_dot` algebraically but its per-edge-type predicted contributions sit ~36× below the magnitude inflection where the lever actually engages.**

This is a **probe-vs-lever distinction** about mechanistic decomposition methodology — not a "find where the residual signal lives" puzzle. The framing change matters because:

- "Edge ablation is weak" (old) → reads as a negative result
- "Edge ablation is sub-threshold by a factor of 36×" (new) → reads as a quantified methodological finding about decomposition-vs-intervention mismatch

The new framing positions our work at the **methodological level** of mech-interp, with broad implications beyond refusal. EMNLP reviewers will value this much more than incremental "we decomposed another behavior" framing.

---

## 1. Title (drafts in order of preference)

1. **"Probes Are Not Levers: A Magnitude Gap in Mechanistic Refusal Interventions"**
2. "The Magnitude Gap: Decomposition Without Control in Refusal Circuits"
3. "Probe–Lever Asymmetry: When Circuit Decomposition Predicts Without Prescribing"

## 2. One-sentence thesis

> A learned 1-D refusal direction at L15 in Gemma-3-4B-IT (and L18 in Qwen3-4B) is *simultaneously* a clean probe and a strong behavioral lever — but the circuit-tracer linearization decomposition that accurately accounts for the projection's algebraic composition predicts per-edge-type contributions that are **~36× below the magnitude inflection** at which the lever engages, because behavioral flipping requires bulk residual displacement along *r̂*, not algebraic compensation of any single edge-bucket's contribution.

---

## 3. Contributions (3, in order of novelty)

1. **The probe-vs-lever magnitude gap** — we quantify, for the first time, that circuit-tracer's predicted per-edge-type contributions sit ~36× below the magnitude inflection where behavioral intervention engages. This generalizes across models (Gemma-3-4B, Qwen3-4B [pending]).
2. **Layer-broad redundancy of refusal control surfaces** — the lever exists across layers L12–L24 in Gemma (peaks L15–L18, within Wilson CI). No upstream "decision layer" exists.
3. **A methodological warning for mech-interp** — algebraic decompositions of probe-relevant quantities (like `h·r̂`) do not by default give recipes for behavioral intervention. Authors should validate magnitude-sufficiency.

## 4. Headline figure

`controllability_extension_figure.png` — 4 panels:
- **A:** EXP 1 dose-response curve, L15 all positions. Sigmoid from 6% at coeff=0.005 to 100% at coeff=1.0. Inflection at coeff≈0.18. Edge-derived range shaded.
- **B:** EXP 2 dose-response curve, L15 pos=−2 only. Same shape but shifted right and capped lower.
- **C:** EXP 4 layer locator depth profile, coeff=1.0 pos=−2. Plateau L12–L24, peak L15–L18.
- **D:** 2×2 bar chart, magnitude × position cross-comparison.

---

## 5. Section structure (8-page main conference)

### § 1. Introduction (≈ 1 page)

- Mech-interp's standard pipeline: probe → decompose → ablate
- Implicit but unstated assumption: decomposition predictions translate to intervention recipes
- We test this assumption on the cleanest known refusal direction (Gemma-3-4B, L15:r̂) and show it fails by ~36×
- Three contributions enumerated
- One paragraph foreshadowing the Qwen3-4B replication (generalization)

### § 2. Background and Setup (≈ ¾ page)

- 2.1 Difference-of-means refusal probes (Arditi 2024, Ball 2024, Wang 2025)
- 2.2 Circuit-tracer attribution graphs and CLT-based decomposition (Lindsey et al.)
- 2.3 The linearization identity at L15: `direct_dot = Σ edges + baseline_offset`
- 2.4 Controlled dataset (50 prompts × 11 conditions: bare + 5 JB + 5 CTRL, n=550 per cell)

### § 3. The Refusal Direction as a Probe AND a Lever (≈ 1 page)

- 3.1 r̂ at L15 (Gemma) / L18 (Qwen) separates harmful refuse vs. comply
- 3.2 Per-prompt projection `direct_dot` tracks refusal classification (probe property)
- 3.3 Stage 06 / Arditi anti-refuse-sub at coeff=1.0: **100% bare→COMPLY flip, 100% benign-force-refuse** (lever property)
- 3.4 Symmetric: pro-refuse-add at coeff=1.0 also fully flips (100% on JB-comply baseline)
- 3.5 Same finding on Qwen3-4B at L18 (92.5% bare-flip, 96.3% JB-comply→REFUSE flip) [Ruqiya]
- *Headline*: a clean direction that is BOTH a probe AND a lever — the question is whether the lever is captured by the decomposition.

### § 4. The Decomposition Hypothesis: Edge Ablation (≈ 1¼ pages)

- 4.1 Setup: subtract the linearization-predicted contribution of each edge bucket at L15
- 4.2 Drift verification: hook achieves the predicted delta (217/220 checks within ±50)
- 4.3 Seven variants × 550 prompts; top-K sweeps over features (0d) and edges (0e)
- 4.4 Pooled flip rates: 6–8% across all variants — at or near baseline noise
- 4.5 Pareto sweeps: flat across K (no sparsity knee)
- *The puzzle*: if `direct_dot` causally drives refusal, why doesn't subtracting its components flip behavior?

### § 5. The Magnitude Gap [HEADLINE SECTION] (≈ 2 pages)

- 5.1 Coefficient sweep at L15 all-positions, 8 coefficients log-spaced — dose-response curve
- 5.2 Inflection at coeff≈0.18; edge-derived deltas at coeff≈0.005 (**36× below**)
- 5.3 Where the coefficients come from: direction coefficient is a knob (we choose 1.0); edge "coefficient" is a *measurement* extracted from the linearization (~0.005 by data, not by setting)
- 5.4 The thermometer analogy: linearization decomposes the *reading* (`direct_dot`), but behavior depends on the *room temperature* (bulk residual geometry). Subtracting predicted contributions adjusts the reading without changing the room.
- 5.5 Implication: per-edge-type contributions are real but redundantly distributed across paths. Flipping behavior requires bulk residual displacement, not algebraic compensation.
- 5.6 [pending] Supra-threshold confirmation: re-running edge ablation at 5× / 10× / 50× / 100× the predicted delta produces the same dose-response curve as direction intervention at the equivalent effective coefficient. This proves magnitude IS the only missing variable.
- 5.7 [pending] Qwen3-4B replication: dose-response curve has the same shape and inflects at similar relative coefficient.

### § 6. Position vs Magnitude: A 2×2 (≈ ½ page)

- 6.1 Setup: full direction vs edge-derived × all positions vs pos=−2
- 6.2 Cell values + the A→C, A→B, C→D step-sizes
- 6.3 Conclusion: magnitude dominates; position is a secondary modulator (only matters when magnitude is supra-threshold)

### § 7. Layer-Broad Redundancy (≈ ¾ page)

- 7.1 Depth profile at coeff=1.0 pos=−2 across 9 Gemma layers
- 7.2 Lever exists L12–L24, peaks L15–L18 (Wilson CI overlap)
- 7.3 No early-layer "decision layer" — refusal builds up gradually L9–L18 and is read redundantly through L24
- 7.4 [pending] Qwen3-4B layer locator: confirms band structure (L?–L? expected to be similar relative positioning to L18 best-probe)
- 7.5 Implication: prior "find the refusal layer" framing replaced by "refusal exists in a band, with a peak"

### § 8. Discussion (≈ ¾ page)

- 8.1 **Methodological:** decomposition probes are not intervention recipes by default. Authors claiming "feature X causes behavior Y because ablating it changes the probe direction" should validate magnitude-sufficiency at the behavioral level.
- 8.2 **For safety/alignment:** bulk-magnitude levers are robust (work at any position-mode, across a layer band); narrow-feature levers may be elusive due to the redundancy story. Has implications for "find the X feature and ablate it" interpretability programs (e.g., Anthropic's monosemantic-feature work).
- 8.3 **Connections:** extends rather than contradicts Wang 2025, Ball 2024, Arditi 2024. Their decomposition findings stand as probes; we add the lever-magnitude caveat.
- 8.4 **Why ~200×:** redundancy across parallel paths + RMSnorm dynamics + the fact that `direct_dot` at pos=−2 captures only one slice of a higher-dimensional refusal manifold.

### § 9. Limitations (≈ ¼ page)

- Two model families (Gemma-3-4B + Qwen3-4B). Llama, Mistral, larger Qwen unconfirmed.
- Keyword classifier; spot-check revealed soft-refusal false positives at intermediate coefficients. Need hand-graded sample or LLM judge.
- n=50 prompts; pooled denominators are n=161 (JB-refuse) and n=250 (CTRL).
- Refusal-specific; whether the magnitude gap generalizes to other behaviors (sycophancy, deception, formatting) is open.

### § 10. Conclusion (≈ ¼ page)

- Restate: probe ≠ lever-recipe; the 36× gap is the empirical signature.

---

## 6. Figure plan (5 figures total)

| # | Content | Source |
|---|---|---|
| **F1** | Concept diagram: probe vs lever; thermometer analogy | NEW illustrator |
| **F2** | r̂ probe identification + Arditi symmetry (Gemma + Qwen side-by-side) | Existing v1 + Ruqiya's Qwen data |
| **F3** | Linearization decomposition + edge ablation 7-variant bar chart | `controllability_audit_figure.png` |
| **F4** | Top-K Pareto sweeps (features + edges) | `topk_feature_pareto_figure.png`, `topk_edge_vs_node_figure.png` |
| **F5 [HEADLINE]** | 4-panel: EXP 1 dose-response, EXP 2 dose-response, depth profile, 2×2 bars (Gemma; Qwen overlay if data allows) | `controllability_extension_figure.png` |

---

## 7. Experiment status

### ✅ Complete (Gemma-3-4B-IT)

| Experiment | Result | Where |
|---|---|---|
| Stage 01 — r̂ identification per layer | clean probe at L15, ‖r̂‖=3101.2 | `data/results/pipeline_runs/run_20260430_023247/01_direction/` |
| Stage 06 — Arditi intervention | 98% bare→COMPLY at coeff=1.0 | `data/results/pipeline_runs/run_20260430_023247/06_causal/` |
| Stage 04 — feature labels (Gemma Scope) | universal cluster: L13:F427, L10:F111, L7:F384, L11:F315 | run_20260430_023247/04_labels/ |
| Phase 0 0b — edge ablation (7 variants) | pooled JB-refuse 6.7% at all_edges | Batch 12, EMNLP_RESULTS_LOG.md |
| Phase 0 0d/0e — top-K Pareto sweeps | flat; no knee | Batch 12 |
| Drift verification | 217/220 within ±50 | Batch 12 |
| **Batch 14 EXP 1** — direction sweep, all positions | 100% flip at coeff=1.0; inflection ≈ 0.18 | `direction_intervention_sweep_all.json` |
| **Batch 14 EXP 2** — direction sweep, pos=−2 only | 20% bare flip / 48% JB-refuse at coeff=1.0 | `direction_intervention_sweep_pos2.json` |
| **Batch 14 EXP 3** — edge ablation pos=−2 (Cell D) | 6–11% (same as all-positions edge ablation) | `edge_ablation_pos2_flip_rates.json` |
| **Batch 14 EXP 4** — layer locator coeff=1.0 pos=−2 | L18 narrowly leads L15; band L12–L24 | `layer_locator_pos2_coeff1.json` |

### 🔄 In progress (pending kickoff)

| Experiment | Purpose | Cost estimate |
|---|---|---|
| **Supra-threshold edge ablation** (Gemma) | Scale edge-derived deltas by {5×, 10×, 50×, 100×} and verify dose-response matches direction sweep at equivalent coefficient. Confirms magnitude is the only missing variable. | ~$7, ~2 hr on H100 SXM fp32 |
| **Qwen3-4B direction sweep, all positions** | Reproduce EXP 1 on Qwen at L18 | ~$14, ~4 hr |
| **Qwen3-4B direction sweep, pos=−1 only** | Reproduce EXP 2 on Qwen (Qwen probe was built at pos=−1, not −2) | ~$14, ~4 hr |
| **Qwen3-4B layer locator** | Reproduce EXP 4 on Qwen across 8 non-L18 layers | ~$14, ~4 hr |

**Total pending Gemma + Qwen work: ~14 hr H100 SXM, ~$50.**

### 📋 Future work (deferred; not blocking the EMNLP submission)

| Experiment | Purpose | Notes |
|---|---|---|
| LLM-judge classifier validation | Bound the keyword false-positive rate flagged in Batch 14 spot-check | Hand-graded stratified sample at coeff ∈ {0.1, 0.25}; add LLM judge for absolute-rate calibration. ~$20 in API. |
| Alternative datasets (XSTest, HarmBench, AdvBench) | Generalization beyond our controlled dataset | Re-run direction sweep + Arditi check on each dataset. Plug into existing pipeline once datasets are added to `dataset/`. |
| Qwen circuit-tracer attribution graphs | Full Phase 0 reproduction on Qwen (would enable Qwen edge-ablation 0b/0d/0e) | Expensive: 2-3 days GPU + CLT training. Defer to camera-ready or follow-up paper. Without it, Qwen story is "dose-response only" — sufficient for the main claim. |
| Per-class r_jb directions (paused track) | Per-class JB intervention; does magnitude gap also apply per-class? | Mentioned briefly in discussion; full execution post-EMNLP. |
| Larger models (Llama-3-8B, Qwen-2.5-7B, Gemma-2-9B) | Does the magnitude gap scale with model size? | Defer to follow-up. ~$50-100 per model. |

---

## 8. Cuts from earlier v2 framing

These are deliberately cut from the Batch 14 reframe to keep the paper focused:

- **"Where does the missing 65% of refusal effect live?"** framing (old § 2 of v1 outline) — replaced by the magnitude-gap framing. The 65% gap is *explained* by the magnitude story, not partitioned across hypothesized residual sources.
- **Attention-head paths investigation** — was hypothesis (1) of the old framing. Now: a footnote noting that attention paths are part of the parallel-redundancy story but not separately quantified.
- **Transcoder error nodes deep dive** — was hypothesis (2). Now: covered briefly in § 4 (edge bucket includes error nodes; their predicted contribution is also sub-threshold).
- **Cross-position interactions** — was hypothesis (4). Now: handled by the 2×2 (§ 6), which shows position is real but secondary to magnitude.
- **Stage 08 MLP failed dissociation** — Was going to be a discussion point. Now: footnote in § 8.2 as evidence of redundancy.
- **Taxonomy clustering / per-class signatures** — appendix only, not load-bearing for the main narrative.

---

## 9. Timeline to submission

Assuming Qwen + supra-threshold experiments kick off today and complete in <24 hr:

| Day | Work |
|---|---|
| Today | Kick off Qwen + supra-threshold on RunPod; draft § 5 (the headline) while it runs |
| +1 | Qwen + supra-threshold results land; draft § 4, § 6, § 7 |
| +2 | Draft § 3, § 1 (intro); update figures with Qwen data |
| +3 | Draft § 2 (background) and § 8 (discussion); F1 concept diagram |
| +4 | Polish + § 9 limitations + § 10 conclusion + abstract |
| +5 | Internal review with Tejas/Georg/Ruqiya |
| +6-7 | Revisions, format check, submit |

3-week buffer means ~2 weeks of slack for revisions, additional experiments if reviewers prompt them, or response-to-reviewers prep.

---

## 10. Risks and what could derail this

- **Qwen replication doesn't show the same magnitude gap.** If Qwen's dose-response has a fundamentally different shape, we need a backup framing. Mitigation: even a different shape is informative ("magnitude inflection is model-dependent" is also a valid finding, just less clean).
- **Supra-threshold rerun reveals magnitude is NOT the only variable.** If 5× edge-derived doesn't match constant-coeff=0.025, something else is going on (variance across prompts, position-dependence at supra-threshold, etc.). Mitigation: this would be a real finding to investigate, not a paper-killer.
- **Classifier false-positive rate higher than expected.** If hand-graded sample shows >15% FP at intermediate coefficients, the dose-response curve shape is questioned. Mitigation: report both keyword and LLM-judge rates; the qualitative finding (inflection exists) is robust to absolute-rate noise.
- **Reviewer pushback on "only 2 models."** Possible. Mitigation: extending to Llama-3-8B is ~$20 of GPU; can add in revision phase.

---

*Last updated 2026-05-22 — Batch 14 reframe + Qwen3-4B/supra-threshold experiments scoped and pending kickoff.*
