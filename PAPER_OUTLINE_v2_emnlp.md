# Refusal-Lens — Paper Outline v2 (EMNLP main / NeurIPS main)

**Track**: EMNLP 2026 main conference (deadline ~3 weeks out at the time of writing) or NeurIPS 2026 main (December cycle).
**Status**: superset of `PAPER_OUTLINES_v1.md`; depends on the workshop submission landing first as anchor reference.
**Companion document**: `PAPER_OUTLINES_v1.md` (the 4-page workshop version that this v2 deepens).

---

## 0. Why a v2 paper, separate from the workshop

The 4-page workshop paper (`PAPER_OUTLINES_v1.md`) makes the **observation** that 1-D directional intervention Pareto-dominates sparse feature ablation by ~3× on jailbreak-induced compliance, and stops there. A main-conference paper has to answer the next question: **why?**

The discussion section of v1 lists four candidate hypotheses for the residual ~65 % of refusal effect not recoverable from MLP-feature ablation:

1. Attention-head paths (frozen during attribution).
2. Transcoder reconstruction errors (the "error nodes" in circuit-tracer's Lindsey-et-al. methodology).
3. Per-prompt features outside the top-100 (already largely ruled out by the Pareto plateau evidence in v1, but worth a tighter bound).
4. Cross-position interactions our pos=−2 measurement misses.

A v2 paper answers each of these, partitions the residual recovery quantitatively, and proposes a **mechanistic taxonomy** that distinguishes the gating axis (low-rank, residual-stream-level) from the expression circuitry (high-dimensional, distributed across transcoder features, attention paths, and reconstruction errors).

This document specifies the experiments needed to make each hypothesis falsifiable, plus an updated section structure for the longer paper.

---

## 1. Title (drafts)

- "Decomposing the Direction–Ablation Gap: Where the Residual 65 % of Refusal Lives"
- "Refusal as a Low-Rank Gate over Distributed Expression: An Attribution-Graph Account"
- "Why Directional Interventions Outperform Feature Ablations: A Mechanistic Decomposition of Refusal Circuits"

## 2. One-sentence thesis

> The 1-D refusal direction *r̂* recovers 100 % of jailbreak-induced compliance via additive intervention; sparse transcoder-feature ablation recovers ≤35 %. We decompose the missing ~65 % into three quantified components — (i) frozen attention-head paths, (ii) transcoder reconstruction errors at the L15 measurement point, and (iii) cross-position interactions outside the pos=−2 decision token — and show that **explicitly accounting for all three closes the gap to within X %** on Gemma-3-4B-IT, Qwen3-4B, and Gemma-2-9B-IT, establishing refusal as a structurally distributed mechanism gated by a low-rank residual-stream control.

(`X` to be filled after experiments — target 5–10 %.)

## 3. Section structure (8-page main conference)

#### § 1. Introduction (≈ 1 page)
- Same priors as v1 (Arditi, Ball, Wang).
- v1's headline (the gap) is the *motivation* for v2.
- v2's contribution: the gap decomposition + the closure.

#### § 2. Background and Methods (≈ 1 page)
- Subset of v1's § 2.
- Add: circuit-tracer's linearization identity (Σ edges + baseline + error_nodes = direct_dot), and how feature ablation breaks it.
- Add: edge attribution methodology vs node ablation.

#### § 3. The gap and its components (≈ 2 pages — new)
- Replicate v1's Pareto curve as the anchor.
- Then partition into three components, **each with its own intervention experiment** (see § 4 of this document).

#### § 4. Closing the gap (≈ 2 pages — new)
- Quantify each component's contribution to recovery.
- Composite intervention (direction + edge-restricted ablation + attention-path edit) → target 90 %+ recovery.
- This is the key deliverable: a constructive demonstration that the gap is *decomposable* into named mechanisms, not magic.

#### § 5. Cross-model generalization (≈ 1 page)
- Replicate the gap decomposition on Qwen3-4B and Gemma-2-9B-IT.
- Show the *components* differ in ratio across models but the structure (low-rank gate + distributed expression) is preserved.

#### § 6. Discussion (≈ 1 page)
- Mechanistic taxonomy: low-rank gate vs distributed expression as a re-usable framing for other 1-D-direction mechanism papers (deception detection, sycophancy, persona steering).
- Implications for jailbreak defense: edge-level patching as an alternative to either directional ablation (which collapses helpfulness) or sparse feature ablation (which underperforms).

#### § 7. Limitations (≈ 0.5 page)
- Still MLP+attention only; no token embeddings as targets.
- Bf16 generation drift; some numbers depend on hardware.
- Specific subcircuit construction methodology may not transfer to non-Arditi-style direction extractions.

---

## 4. Experiments needed (the new content vs v1)

Each experiment below has: rationale, hypothesis, method, expected result, GPU/wall cost, dependencies, acceptance criteria. Estimates assume a single RTX 4090 plus occasional H100 access.

---

### 4.1 Edge ablation (Stage 09a)

**Rationale**: Stage 08 ablates *nodes* (transcoder feature output values clamped to 0 across all forward passes). This destroys the feature's contribution everywhere. Edge ablation in circuit-tracer's framework targets specific *connections* in the attribution graph — e.g., "the edge from L11:F127 → L15:F427" — leaving each endpoint feature alive elsewhere in the computation. Hypothesis: edge ablation is strictly more surgical and recovers a measurable additional fraction of the gap by only removing the specific path that recruits a feature into the refusal direction.

**Hypothesis**: edge-restricted ablation of the top-K edges feeding *r̂* at L15, pos=−2, recovers more JB compliance than node-level ablation of the same features at the same magnitude. Specifically: target ≥ 50 % recovery (vs Tier 1 top_50 plateau at 34.8 %).

**Method**:
1. Implement edge-ablation hook in `vendor/circuit-tracer/`. The mechanics: replace the contribution of selected edges to the target node with their baseline (zero-input transcoder activation) value, propagate through, regenerate.
2. Per-prompt selection: rank edges in the attribution graph by |attribution_to_r̂| at pos=−2; ablate top-K edges where K ∈ {10, 50, 100, 500, 1000}.
3. Compare to Tier-1 node ablation at matching feature counts.
4. Renormalize against Stage 06 baselines (same protocol as Tier 1/2).

**Cost**: ~30 hours wall on 4090 (similar to Tier 1; per-prompt edge selection adds ~10 % overhead). One pass.

**Dependencies**: must implement edge-ablation hook in circuit-tracer fork. ~3 person-days for the patch + tests.

**Acceptance**: edge ablation at top_500 edges produces ≥ 5 pp more recovery than node ablation at matched feature count, with non-overlapping Wilson CIs. If null, then "node vs edge" is not a productive partition and we report it as a negative result.

---

### 4.2 Transcoder error-node attribution (Stage 09b)

**Rationale**: circuit-tracer's linearization identity `Σ edges + baseline + Σ error_nodes = direct_dot` decomposes the projection on r̂ into transcoder-explainable contributions (edges + baseline) and transcoder-unexplainable residual (error nodes — places where the transcoder's reconstruction fails to capture the actual residual stream activation). On bare prompts at L15, our existing data shows error-node contribution is small (sub-1 % per § 4 of REPORT). On jb prompts, **we have not measured this**. If error nodes carry significant fraction of the JB-induced shift along r̂, that explains a chunk of the gap directly.

**Hypothesis**: error-node contribution to direct_dot at L15 is significantly larger on jb prompts than on bare prompts; ablating only the error-node contribution closes a measurable fraction of the gap.

**Method**:
1. Modify `03_verify_attribution.py` to also report `Σ error_nodes` per prompt × condition.
2. Run on the existing 50×11 dataset; no new GPU pass needed for the verification.
3. For ablation experiment: introduce an "error-node clamping" hook that zeros out the residual-stream noise outside the transcoder approximation at L15. Generate, classify, renormalize.
4. Per-class table: error-node contribution and its ablation recovery.

**Cost**: ~5 hours wall (re-using existing attribution graphs). Needs `02c_pack_graphs.py` outputs to be locally available.

**Dependencies**: existing .pt graphs (already on HF) need to be pulled to local 4090 (~80 GB). Edge-ablation infrastructure (4.1) is helpful but not required.

**Acceptance**: error-node contribution to direct_dot is ≥ 2 % on jb prompts and ablating it produces ≥ 3 pp recovery. If null, transcoder errors are not the source of the gap and we report it as a negative result.

---

### 4.3 Attention-head attribution (Stage 09c)

**Rationale**: the Lindsey et al. circuit-tracer methodology freezes attention patterns during attribution; only MLP/transcoder paths contribute as "free" adjustable nodes. If a significant chunk of the directional effect arises from attention heads gating information flow into the L15 residual, MLP-feature ablation will miss it. **This is the most likely candidate for the largest single component of the gap.**

**Hypothesis**: a significant fraction (≥ 20 %) of the directional effect lives in 1–5 specific attention heads that route the JB-prefix-induced shifts into the L15 residual at pos=−2.

**Method**:
1. Wire up attention-head attribution in circuit-tracer (its codebase supports this; we just haven't activated it). Specifically: after the existing attribution pass, attribute the contribution of each attention head's output to the *r̂* projection, identical methodology to feature attribution.
2. Identify top-K attention heads by attribution magnitude.
3. Ablation experiment: zero each top-K head's contribution at pos=−2 across forward passes (or, more surgically, zero its contribution to the read-from-r̂ direction).
4. Composite: combine top-K attention head ablation with top-50 feature ablation; measure additive recovery.

**Cost**: ~4 person-days for the attention-attribution wiring (it is the existing-but-disabled feature). ~10 hours wall for the ablation experiment.

**Dependencies**: requires reading circuit-tracer's attention-attribution code (we know it exists). Risk: nnsight backend has had attention-head bugs in our experience; might require TransformerLens.

**Acceptance**: top-5 attention heads ablation produces ≥ 10 pp additional recovery on top of Tier-1 top_50 features. If null, attention heads are not the gap source and we report.

---

### 4.3a Per-class jailbreak-vector intervention (next-priority follow-up to v1 § 5.6)

**Rationale**: v1's § 5.6 ran a *universal* jailbreak-vector intervention (single *r_jb_universal* averaged across the 5 JB classes, where *r_jb_class* = mean(*h_jb_class*) − mean(*h_bare*) under the Ball 2024 / Wang 2025 sign convention pointing TOWARD jailbreak). It obtained Experiment A flip rate **47/89 = 52.8 %** on jb-comply prompts (subtracting *r_jb_universal* to mitigate the JB). The shortfall vs Stage 06's 100 % is dominated by two factors that the universal-vector design conflates:

- **Magnitude shrinkage from averaging**: ‖*r_jb_universal*‖ = 0.65 ‖*r̂*‖, vs per-class magnitudes ranging 0.40 to 1.11 ‖*r̂*‖. Averaging across classes that share most but not all of their direction in the residual stream cancels per-class signal — the universal vector is a structurally lossy summary.
- **Per-class dose mismatch**: the per-class flip rate is *inversely* related to per-class *r_jb_class* magnitude — universal-vector dose under-corrects cognitive_reframe (1.11 ‖*r̂*‖ empirical edit, only 27.3 % flip) and over-corrects fiction (0.49 ‖*r̂*‖ empirical edit, 73.7 % flip).

**Hypothesis**: applying *r_jb_class* of magnitude ≈ ‖*r̂*‖ (i.e., per-class vector, magnitude-matched to *r̂*, subtracted from JB-comply prompts of that class) produces flip rates approaching Stage 06's 100 %, closing most of the universal-version 47 pp gap. This would be the rigorous version of the v1 § 5.6 experiment and would make the "JBs edit *r̂* toward the harmless direction" claim quantitatively decisive.

**Method**:
1. Reuse `scripts/analysis/jb_vector_intervention.py` (post-2026-05-04 sign-convention update) and the saved `02b_stats/residuals_L15_per_cond.pt`. Iterate on the per-class `r_jb_per_class` list instead of the mean. Each entry is computed as `mean(h_jb_class) − mean(h_bare)` (Ball convention; points toward jailbreak).
2. Two intervention conditions per class, both using the **subtraction** hook (mitigate JB):
   - **Empirical magnitude**: subtract *r_jb_class* at its native magnitude (0.40–1.11 ‖*r̂*‖). Discriminates "magnitude" from "direction" as the bottleneck.
   - **Magnitude-matched**: scale *r_jb_class* to ‖*r̂*‖ (i.e., subtract *r_jb_class* / ‖*r_jb_class*‖ × ‖*r̂*‖). Matches Stage 06's 1.0·‖*r̂*‖ dose along the per-class axis.
3. For each (prompt, jb_*) where Stage 06 baseline = COMPLY, apply the relevant per-class hook only to that class's prompts.
4. Compute per-class flip rates and Wilson CIs.
5. Same renormalization protocol as Stage 06.

**Cost**: ~60 min on the 4090 (≈5× the universal v1 run, minus model-load amortization). Full sweep across both magnitude conditions and 5 classes is ~2 hours.

**Dependencies**: nothing — all data is local. Script is ~30 min of refactoring.

**Acceptance**: magnitude-matched per-class intervention produces flip rate ≥ 90 % on at least 3 of 5 classes (where n_baseline_comply > 5); empirical-magnitude intervention's per-class flip rate scales linearly with per-class ‖*r_jb_class*‖. If both pass, the v1 § 5.6 result becomes "the directional component is causally sufficient at full dose" — closing the workshop paper's biggest open question.

**Why this is in v2 not v1**: by user direction (2026-05-04 conversation), the workshop submission keeps the universal-only run for time-budget reasons, and the per-class version moves to v2. The HANDOFF.md § P8c-ii note carries this forward explicitly.

**Sign-convention note**: this section is written in the post-2026-05-04 Ball 2024 / Wang 2025 convention (*r_jb* points toward jailbreak; subtract to mitigate, add to induce). The earlier in-tree v1 work briefly used the opposite convention; HANDOFF.md § P8c documents the rewrite.

---

### 4.4 Steering instead of zeroing (Stage 09d)

**Rationale**: Stage 08 clamps features to zero. This is **not** the only intervention; it could be over-strong (deletes information) or under-strong (the feature was negative-valued, so zeroing pushes the wrong direction). Steering — replacing the feature's value with its mean over harmless prompts, or with a target value drawn from a distribution — is more nuanced and may recover more fraction of the gap.

**Hypothesis**: steering features to harmless-prompt mean recovers ≥ 5 pp more than zeroing at matched feature count.

**Method**:
1. Re-use Stage 08 infrastructure with a new hook mode: `clamp_to_value=mu_harmless[F]` instead of `clamp_to_value=0`.
2. Run on the canonical_pro_refusal subcircuit (§ 9.7) as the smallest-feature, fastest-iteration test. If positive, scale to top-50.
3. Renormalize.

**Cost**: ~10 hours wall on 4090.

**Dependencies**: need harmless-prompt mean activations for each transcoder feature at L15; ~30 minutes to compute from Stage 01's harmless_64 set.

**Acceptance**: steering at top_50 produces ≥ 5 pp more recovery than zeroing at top_50 with non-overlapping CIs. If null, the comparison is "zeroing is sufficient" and we report.

---

### 4.5 Manual circuit inspection (Stage 09e)

**Rationale**: automated subcircuit construction (Stage 07) may miss the *right* features even at top-100 because the construction rules are corpus-aggregated heuristics. Manual inspection of attribution graphs for 5–10 specific prompts could reveal patterns that automated rules can't capture (e.g., a single low-attribution feature that's load-bearing for a specific prompt class but doesn't appear in any global top-K).

**Hypothesis**: manual inspection identifies a class of features that automated subcircuit rules miss, and adding them to the ablation set produces ≥ 5 pp additional recovery on the specific prompts they were identified on.

**Method**:
1. Pick 10 jb prompts where Stage 06 directional intervention flipped behavior 100 % but Stage 08 best-subcircuit ablation did not.
2. For each, manually inspect the attribution graph: look at top-100 features, the Pareto plateau, error-node contribution, and attention-head attribution (if 4.3 is done).
3. Identify any feature with high attribution that is **not** in the existing subcircuits.
4. Construct a "manually-curated" subcircuit per prompt and run Stage 08 at the per-prompt level.

**Cost**: ~3 person-days of frontend time (manual inspection is the bottleneck, not GPU).

**Dependencies**: needs the frontend manual-ablation infrastructure described in `FRONTEND_ABLATION_PLAN.md`. ~70 % built.

**Acceptance**: manual subcircuits recover ≥ 5 pp more than top_50 on the specific 10 prompts, with the gap visible per-prompt. If null, the feature space is genuinely "complete" at top-100 and the gap source is elsewhere.

---

### 4.6 Cross-model gap decomposition

**Rationale**: the workshop paper claims the ~3× gap exists on Gemma-3-4B-IT alone. A main-conference paper needs the gap decomposition (4.1–4.5) replicated on at least one more model from a different family.

**Hypothesis**: the gap decomposition has the same *structure* (attention paths > transcoder errors > out-of-top-K features) on Qwen3-4B as on Gemma-3-4B-IT, even though the absolute fractions differ.

**Method**:
1. Wait for Ruqiya's Qwen3 pipeline to land (currently at scaffold stage; merge needed).
2. Run Stages 06 + 08 on Qwen3 with the same methodology (canonical, full per-prompt sweep, top-N curve).
3. Run 4.1–4.4 on Qwen3 (4.5 manual is optional for Qwen given limited team time).
4. Compare components: which model has more gap in attention paths, which has more in transcoder errors?

**Cost**: ~1 week wall on RunPod (full pipeline) + 30 hours of v2 experiment time on 4090 / RunPod.

**Dependencies**: Qwen3 pipeline must finish first. Mahmoud's recent Slack message to Ruqiya covers the prerequisites (rebase, submodule unification, Stage 06+08 on the controlled dataset).

**Acceptance**: the cross-model gap structure replicates qualitatively (top component is the same kind of effect across models) even if absolute fractions differ. If structure does *not* replicate, we have a stronger negative result: the gap is model-specific, and the paper pivots to that finding.

---

### 4.7 (Optional, stretch) Replication on Gemma-2-9B-IT

**Rationale**: Wang et al. 2025 evaluated Gemma-2-9B-IT extensively. Adding a third model from the same family but different size strengthens the cross-model claim and matches a published comparator. Lower-priority than 4.6.

**Cost**: ~2 weeks wall (model is 2× larger; full pipeline plus v2 experiments).

**Acceptance**: gap replicates within ±10 pp of Gemma-3-4B-IT.

---

## 5. Timeline

Assuming EMNLP main deadline is approximately mid-June (3-4 weeks out from 2026-05-04):

| Week | Goals |
|---|---|
| Week 1 (5/4–5/10) | Submit ICML workshop (v1). Concurrent: Ruqiya finishes Qwen3 pipeline rebase. Implement edge-ablation hook (4.1). |
| Week 2 (5/11–5/17) | Run experiments 4.1, 4.2, 4.4 on Gemma-3-4B-IT. Wire up attention-head attribution (4.3). Local frontend manual inspection (4.5) starts in parallel. |
| Week 3 (5/18–5/24) | Run experiment 4.3 on Gemma-3-4B-IT. Begin cross-model on Qwen3 (4.6) once Qwen pipeline lands. |
| Week 4 (5/25–5/31) | Finish Qwen3 cross-model decomposition. Begin writing v2 paper (longer than v1, ~8 person-days draft). |
| Week 5 (6/1–6/7) | Internal review by Georg, Ruqiya, Tejas. Polish figures. Submit. |

If EMNLP deadline is earlier (e.g., mid-May), pivot to NeurIPS main (mid-July) and use the extra time for 4.7 plus a deeper investigation of any negative result.

---

## 6. Tasking list (per-experiment owner + status)

| # | Experiment | Owner | Status | Hard prereq |
|---|---|---|---|---|
| 4.1 | Edge ablation (Stage 09a) | Mahmoud | Not started | Implement edge-ablation hook in circuit-tracer fork |
| 4.2 | Transcoder error-node attribution | Mahmoud | Not started | Pull .pt graphs locally |
| 4.3 | Attention-head attribution (Stage 09c) | Mahmoud | Not started | Read circuit-tracer attention-attribution path |
| **4.3a** | **Per-class jailbreak-vector intervention (rigorous v1 § 5.6 follow-up)** | **Mahmoud** | **Not started** | **None — script is in place; iterate on `r_jb_per_class` list** |
| 4.4 | Steering vs zeroing (Stage 09d) | Mahmoud | Not started | Compute harmless-prompt mean activations per feature |
| 4.5 | Manual circuit inspection (Stage 09e) | Mahmoud | Not started | Frontend manual ablation cart (≥70 % built — see `FRONTEND_ABLATION_PLAN.md`) |
| 4.6 | Cross-model on Qwen3 | Ruqiya + Mahmoud | Blocked on Qwen3 pipeline rebase | Qwen3 Stage 06 + Stage 08 land |
| 4.7 | Replication on Gemma-2-9B-IT (stretch) | Open | Not started | Compute budget; lower priority |

**Suggested ordering for v2 work**: 4.3a is the cheapest and highest-leverage (closes the v1 § 5.6 gap to a quantitative claim within ~2 hours); run it before any of 4.1–4.5. After 4.3a lands, 4.2 (transcoder error-node attribution, ~5 hours) is the cheapest remaining decomposition lever and answers a directly-falsifiable hypothesis from v1's § 4 Discussion.

---

## 7. Figures (target 6 main figures + 2 supplementary)

| # | Content | New work? |
|---|---|---|
| F1 | Schematic: methodology overview (carries over from v1, refined) | Refine v1 |
| F2 | Pareto curve + decomposition: x = construction method, y = recovery, stacked-bar by component (MLP feature, edge, attention head, error node) | New (depends on 4.1–4.3) |
| F3 | Cross-model gap decomposition: 3 panels for Gemma-3, Qwen3, Gemma-2-9B (if 4.7 done) | New (depends on 4.6 and optionally 4.7) |
| F4 | Direct-dot decomposition: pos × layer heatmap of error-node contribution to r̂, vs feature contribution. Shows **where** in the model the gap lives. | New (depends on 4.2) |
| F5 | Edge-vs-node ablation comparison curves. | New (depends on 4.1) |
| F6 | Steering-vs-zeroing comparison. | New (depends on 4.4) |
| S1 | Attention-head attribution table | Supplementary, depends on 4.3 |
| S2 | Manual circuit case studies | Supplementary, depends on 4.5 |

---

## 8. Risks specific to v2

| Risk | Probability | Mitigation |
|---|---|---|
| Edge ablation doesn't work / no recovery improvement over node | Medium | Reframe as "ablation methodology is not the bottleneck — the gap is fundamentally distributed in non-MLP components." Increases the v2 paper's negative-result component. |
| Attention-head attribution is technically blocked (nnsight bugs, etc.) | Medium-high | Switch to TransformerLens backend earlier, or use a hand-rolled attention-output-to-r̂ projection method that doesn't need circuit-tracer wiring. |
| Cross-model decomposition shows the gap structure is model-specific | Low-medium | This is itself a publishable negative result — "the gap exists everywhere but its decomposition is architecture-specific." Pivot framing. |
| Composite (direction + edge + attention) intervention does not close the gap | Low-medium | Closing to within 10 % is the target; if we close to within 25 %, it is still a strong result vs the baseline 65 % gap. |
| Tooling time (edge ablation, attention head wiring) eats the experiment time | Medium | Schedule explicit ~1-week tooling-only sprint at start of week 2. Have backup of a hand-coded edge-ablation in TransformerLens that doesn't need circuit-tracer changes. |

---

## 9. What v2 needs from the v1 workshop submission

- **v1 must land with the gap claim publicly visible** so v2 can cite it as the anchor finding rather than re-prove it. Even if v1 is rejected from the workshop, the submitted preprint provides the citation point.
- v1 explicitly forward-references this v2 plan in its § 4 Discussion ("each of these is a falsifiable hypothesis for the main-conference follow-up"). This commits the team to v2 publicly and signals to reviewers that v1 is the first paper of a series, not a one-off.

---

*Created 2026-05-04 alongside the v1 workshop pivot, after the weekly check-in with Georg flagged that the workshop framing leaves the "why" question open. v2 is the resolution path. Update as experiments land.*
