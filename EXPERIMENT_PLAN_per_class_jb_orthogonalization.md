# Experiment Plan — Refusal Feature Taxonomy + Per-Class Jailbreak Direction Orthogonalization (EMNLP 2026 Main Track)

> *(filename retained as `EXPERIMENT_PLAN_per_class_jb_orthogonalization.md` for git-history continuity; the title is now broader because Track A expanded to include the refusal feature taxonomy work — see Status section below for the update history.)*

**Status**: drafted 2026-05-17, updated 2026-05-17 with (1) Georg's foundational edge-ablation track, (2) Georg's direction-alignment cosine challenge (H0-5), and (3) refusal feature taxonomy work (H0-6 through H0-9) building on Georg's "simple top-K experiment" framing.
**Branch**: `emnlp-perm-edit` (off `l15-refactor` HEAD). **Do not commit EMNLP work to `l15-refactor`** (frozen as the ICML submission reference).
**Headline contribution (Track A)**: a **mechanistic taxonomy of refusal** in Gemma-3-4B-IT — the first feature-level decomposition showing (a) which transcoder features/edges causally control the refusal direction at L15, (b) how features cluster into discrete causal roles (universal refusal core, JB-recruited, JB-suppressed, anti-refusal), and (c) how each jailbreak class perturbs the taxonomy to bypass refusal. Validates the taxonomy causally via per-prompt top-K feature + edge ablation Pareto sweeps.
**Headline contribution (Track B)**: a permanent per-class model edit that surgically eliminates one jailbreak class's success rate while preserving base refusal, benign helpfulness, and other classes' jailbreak susceptibility — informed by Track A's taxonomy, which predicts which clusters each class perturbs.
**Both tracks feed the same EMNLP paper**: Track A becomes the *mechanistic taxonomy* pillar (novel beyond Arditi/Ball/Wang's monolithic-direction work); Track B becomes the *applied surgical-edit* pillar (constructive demonstration that the taxonomy has predictive power).

---

## 0. One-sentence thesis (both tracks)

> **Track A:** Transcoder features at L15 + earlier layers in Gemma-3-4B-IT decompose into a discrete taxonomy of causal roles for refusal — validated by (i) comprehensive edge ablation recovering ≥X% of direction-level control predicted by the linearization identity, (ii) per-prompt top-K Pareto curves with a clear knee at small K showing sparse concentration of refusal signal in pro-attribution features (and sign-asymmetric Pareto shapes for negative-attribution features), and (iii) per-class perturbation signatures that explain how each jailbreak class bypasses refusal at the cluster level.
>
> **Track B:** Orthogonalizing Gemma-3-4B-IT's `o_proj` and `down_proj` weight matrices against the class-specific orthogonal component `r_jb_C^⊥ = r_jb_C − proj_r̂(r_jb_C)` of each jailbreak class's empirical residual-stream displacement produces a permanent model edit that (i) drops the target class's jailbreak success rate to ≤10% on the controlled 50×11 dataset, (ii) preserves base refusal at ≥48/50, (iii) leaves cross-class jailbreak rates within ±10 pp of unedited baseline, and (iv) degrades helpfulness by ≤5% on a standard benchmark. The taxonomy from Track A predicts which clusters each `r_jb_C^⊥` correlates with, making the edit's mechanism interpretable rather than black-box.

The two tracks are scientifically complementary: Track A asks "what are the building blocks of refusal in this model, and how do JBs manipulate them?" — Track B asks "given the taxonomy, can we construct a surgical per-class permanent edit?". They share infrastructure (the existing run_20260430_023247 attribution graphs, residual tensors, and Stage 06 baselines) and can execute in parallel.

---

## 1. Background and motivation

The ICML 2026 workshop submission (`PAPER_OUTLINES_v1.md`, code on `l15-refactor`) established:

| Claim | Method | Result |
|---|---|---|
| 1-D additive `r̂` intervention at L15 fully recovers jailbreak-induced compliance | Stage 06 forward hook on `hook_resid_post[15]` | **100 %** (89/89) flip JB-comply → REFUSE |
| Per-class `r_jb_C` (Ball/Wang convention, native magnitude) recovers most of it | Runtime subtract `r_jb_C` from prompts of class C | **93.3 %** (83/89), with analytical 100 % and cognitive_reframe 97 % (§ 5.7 REPORT) |
| Strongest sparse MLP-feature ablation plateaus at ~35 % | Stage 08 per-prompt top-50 ablation | **34.8 %** [Wilson 25.7, 45.2] |
| Class-specific feature subcircuits do not dissociate | Stage 08 `jb_{class}_specific_vs_ctrl` | `jb_fiction_specific_vs_ctrl` recovers **0 %** on fiction, 22 % on roleplay |
| Linearization identity holds bit-exact under corrected basis | Stage 03 verification (`measurement_hook="hook_resid_post"`) | `Σ edges + baseline_offset = direct_dot` within <0.4 % per prompt |

Two unresolved questions motivate this plan, one per track:

**Track A — Georg's foundational ask (2026-05-17 mentor exchange):** the 35% Pareto plateau on sparse MLP-feature ablation could be (i) the transcoder framework's structural limit (signal genuinely lives outside features in attention/embeddings/error nodes), (ii) a methodology artifact of *node-level* ablation (which destroys feature signal everywhere it propagates) where *edge-level* ablation would do better, or (iii) a sign-handling or attribution bug in our current pipeline. Georg's framing: "in principle, our expectation should be that we have complete control over the refusal direction — if we don't, we have to figure out why." Phase 0 makes this question quantitatively answerable.

**Track B — Georg's earlier ask ("strong contribution for a main-conference paper"):** whether the per-class directional intervention can be compiled into a **permanent model edit** that dissociates classes, rather than a runtime hook. The original Stage 08b/08c plan envisioned this via CLT decoder vectors but never implemented it; the v2 paper outline does not yet specify it in implementable detail. This plan does, using `r_jb_C^⊥` (not CLT decoders, which Georg flagged as basis-mismatched in his 2026-04-26 mentor feedback) and addressing the Gemma-3 post-LayerNorm complication.

---

## 2. Phase 0 — Transcoder Controllability Audit (Track A, Georg's foundational experiment)

This phase is intentionally orthogonal to the per-class direction approach. It tests how much of the refusal-direction projection (`direct_dot = h[L15, pos=-2] · r̂`) we can causally manipulate by intervening on the transcoder framework's accounting — and where the missing control, if any, lives.

### 2.1 Hypotheses

**H0-1 (controllability completeness):** Comprehensively ablating ALL edges feeding `direct_dot` — including feature edges, embedding edges, error_node edges, and both positive and negative attributions — drives `direct_dot` to the empirically-measured `baseline_offset` (per the Stage 03 linearization identity), and to zero or beyond when ablation is scaled by 2×. Equivalently: the transcoder framework gives us complete control over the refusal-direction projection at L15.

**H0-2 (signed attribution correctness):** Negative-attribution sources (features/embeddings/error nodes that push refusal in the *opposite* direction at L15) should, when ablated, push `direct_dot` in the predicted positive direction. If ablating negative-only sources doesn't shift `direct_dot` in the expected direction, sign-handling has a bug to find before any further claim is published.

**H0-3 (error-node prominence):** Transcoder error nodes carry a measurable but bounded fraction of `direct_dot`. Per the current Stage 03 audit, error_node Σ is implicit in `baseline_offset`; an explicit error-node-only ablation should isolate this fraction and quantify it.

**H0-4 (edge ≠ node):** Comprehensive edge ablation recovers strictly more `direct_dot` drive than node-level ablation at matched feature scope, because edge ablation is surgical and node ablation destroys signal globally. Expected outcome: edge ablation closes a measurable fraction of the v1 35%-plateau gap.

**H0-5 (geometric alignment robustness, Georg's 2026-05-17 cosine challenge):** The per-class cosine similarity between `r_jb_C` and `−r̂` at L15 pos=−2 (reported as +0.72 to +0.94 in REPORT § 5.5.2) is robust under three controls: (a) **per-prompt computation** — when computed on individual prompts' JB displacement vectors `h_jb_C(p) − h_bare(p)` rather than class-mean differences, the per-prompt mean cosine matches the class-mean cosine within 0.10. (b) **random-direction baseline** — the cosine of `r_jb_C` against `r̂` substantially exceeds the 95th percentile of cosines against 1,000 random unit directions in the same d=2560 space. (c) **Pearson-style mean-subtraction** — when both vectors are mean-centered along their 2,560 dimensions before cosine, the Pearson cosine differs from the raw cosine by ≤0.10. Failing any control means the headline "JBs edit toward the harmless direction" cosine is at least partly an artifact of high-dimensional anisotropy or all-ones-direction bias, not pure geometric alignment of two independently-extracted directions — and the magnitude of the claim must be qualified in the paper.

**H0-6 (refusal-signal sparsity / Pareto knee):** Refusal-direction control in the MLP transcoder concentrates in a small number of top-attribution features. Specifically: ablating the per-prompt top-K features (ranked by signed attribution to `direct_dot` at L15 pos=−2) flips bare-refuse → comply at a rate that follows a Pareto curve with a clear knee at small K (likely K ∈ [10, 50]) rather than rising linearly to the comprehensive-ablation rate. Tested separately for positive-attribution top-K (most pro-refusal features) and negative-attribution top-K (most anti-refusal/refusal-suppressing features); these are expected to have asymmetric Pareto shapes — pro-refusal features should drive most bare-refuse flips, while ablating anti-refusal features should leave bare-refuse intact (since they oppose what bare already does).

**H0-7 (edge Pareto > node Pareto):** The Pareto curve for top-K *edge* ablation (using circuit-tracer's edge-level attribution to `direct_dot`) outperforms top-K *node* ablation at every matched K, because edge ablation is strictly more surgical. Generalizes H0-4 from "comprehensive vs comprehensive" to "every K on the Pareto axis." If the edge curve is uniformly above node at every K, it's a clean methodological lever for the paper. If the curves cross or are statistically tied, node ablation is "good enough" and the methodology distinction isn't load-bearing.

**H0-8 (feature role taxonomy):** Transcoder features at L15 + earlier layers cluster into discrete causal roles when characterized by (a) per-condition activation profile across the 11-condition controlled dataset and (b) signed attribution to `direct_dot`. Initial proposed cluster types (validated by clustering, not pre-defined): **universal refusal core** (active across all conditions with positive attribution), **JB-recruited** (active on JB conditions only, signed either way), **JB-suppressed** (active on bare/ctrl but quenched on JB), **anti-refusal** (negative attribution wherever they fire), and **uncategorized / error-attributable**. The taxonomy is causally validated by the top-K Pareto curves from H0-6: features within "high-impact" clusters dominate small-K ablation; features within low-impact clusters only register at large K or not at all.

**H0-9 (JB-class perturbation signature):** Each JB class perturbs specific clusters preferentially. A "perturbation signature" is the per-cluster mean activation difference `Δ(cluster, jb_C) = mean(activation_cluster | jb_C) − mean(activation_cluster | bare)`. The signatures differ across classes — e.g., fiction-class might recruit the JB-recruited cluster X while suppressing universal refusal core; cognitive_reframe might suppress universal core more strongly while recruiting cluster Y. The per-class signature is the mechanistic explanation for how each JB bypasses refusal AND directly informs Track B's per-class `r_jb_C^⊥` interpretation: `r_jb_C^⊥` should correlate with the cluster decoder directions that class C perturbs preferentially.

**Hypothesis-outcome table:**

| Outcome | Interpretation | EMNLP paper implication |
|---|---|---|
| H0-1 holds → comprehensive ablation drives `direct_dot` to baseline | Transcoder framework is mechanistically complete; 35% plateau is node-vs-edge artifact | Framing A strengthened — we explain the missing 65% as methodology, not structure |
| H0-1 fails → `direct_dot` stays well above baseline after comprehensive ablation | Signal lives in something the linearization doesn't capture (e.g., attention paths not in the transcoder graph at all) | Framing A pivot — paper publishes a structural negative result with localization |
| H0-2 fails → negative ablations don't flip sign | Sign-handling bug somewhere in pipeline | Pause Track B; debug attribution math before any further claim |
| H0-3 → error nodes carry >10% of `direct_dot` | Transcoder reconstruction is imperfect; error nodes are a publishable mechanism component | Add error-node-only ablation result as paper figure |
| H0-4 holds → edge > node by ≥10 pp | Methodology lever, immediately publishable | Reframe v1 35% claim with edge-level number |
| H0-5 holds → per-prompt + random-baseline + Pearson cosines all confirm 0.72–0.94 alignment | Geometric alignment of `r_jb_C` with harmless axis is real, not a high-dim artifact | Headline §5.5.2 claim survives Georg's challenge; Track B per-class methodology is on firm geometric footing |
| H0-5 fails → per-prompt cosines diverge from class-mean OR random baseline produces similar values OR Pearson cosine differs >0.10 | Class-mean cosine overstates per-prompt alignment, or the residual stream's anisotropy is doing most of the work | Qualify the headline claim in the paper; investigate whether the orthogonal-component `u_C` design from Track B is meaningfully different from `r_hat` itself |
| H0-6 holds → Pareto curve has clear knee at K ∈ [10, 50] with asymmetric pos/neg curves | Refusal signal is sparse and concentrated in top-attribution features; sign-asymmetry confirms positive/negative roles | Strong taxonomy story for the paper; identifies the small set of features that "do the work" of refusal |
| H0-6 fails → Pareto curve is flat or rises linearly with no knee | Refusal signal is diffuse across many features; top-attribution ranking doesn't identify causal features | Reframe paper: taxonomy isn't built from top-K, must use clustering + perturbation signatures instead |
| H0-7 holds → edge Pareto uniformly above node Pareto at every K | Edge-level surgery is methodologically superior across the entire feature-budget axis | Paper's Framing A becomes "edge ablation reframes the 35% plateau" — strong methodology claim |
| H0-8 holds → clean clusters with discrete causal roles | Refusal direction has a discoverable feature-level decomposition | Headline taxonomy figure for the paper; novel contribution beyond prior monolithic-direction work |
| H0-8 fails → clusters are diffuse or overlapping, no clear role assignment | Taxonomy as a discrete categorization is wrong framing | Paper pivots to a continuous-role description (PCA / NMF axes instead of clusters) |
| H0-9 holds → distinct per-class perturbation signatures with cluster-level localization | Each JB class has a class-specific mechanism that maps to taxonomy clusters | Closes the loop: taxonomy explains both refusal AND how it's bypassed per class; informs Track B's `r_jb_C^⊥` interpretation |

### 2.2 Sub-experiment 0a — Offline linearization decomposition (no GPU)

**Goal:** quantify per-edge-type contribution to `direct_dot` for every (prompt, condition) using only the saved attribution graphs. CPU-only arithmetic on the JSON.gz packed graphs (already on HF; pull locally via `scripts/pipeline/fetch_graph_data.py` if not present).

**Method:**

For each of 550 (prompt, condition) instances:
1. Load the packed attribution graph from `02_attribution/graph_data/<prompt_id>__<condition>.json.gz`.
2. Categorize all source nodes by type: `{feature, embedding, error_node}` (circuit-tracer exposes these in its node-type field).
3. Extract each source node's signed attribution to `direct_dot` at the measurement target (L15, pos=−2).
4. Aggregate per (prompt, condition):
   - `Σ_features_pos`, `Σ_features_neg`, `Σ_features_signed`
   - `Σ_embeddings_pos`, `Σ_embeddings_neg`, `Σ_embeddings_signed`
   - `Σ_errors_pos`, `Σ_errors_neg`, `Σ_errors_signed`
   - `Σ_all_signed = Σ_features_signed + Σ_embeddings_signed + Σ_errors_signed`
   - `baseline_offset = direct_dot − Σ_all_signed` (per linearization identity; should match Stage 03's reported number within 0.4 %)
5. Verify the identity holds within Stage 03's tolerance (<1 % reconstruction error on 550/550 inputs).

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json` — per-(prompt, condition) full breakdown
- `data/results/emnlp_perm_edit/phase0_controllability/decomposition_by_condition.json` — per-condition aggregates (mean, std, min, max) of each component
- `data/results/emnlp_perm_edit/phase0_controllability/decomposition_figure.png` — stacked-bar chart per condition showing feature / embedding / error_node / baseline contributions to `direct_dot`

**Acceptance:**
- Linearization identity `direct_dot = Σ_all_signed + baseline_offset` reconstructs to <1 % error on 550/550 inputs.
- Per-edge-type contributions reported with per-condition means and stds.
- Negative-attribution sums reported separately from positive sums (this is the diagnostic for H0-2 sign-handling).

**Compute:** ~30 min wall on CPU (file IO bound). No GPU needed.

### 2.3 Sub-experiment 0b — Comprehensive edge-ablation runtime intervention (GPU)

**Goal:** causally test the linearization decomposition. By zeroing the contribution of each edge type at runtime (and combinations including over-ablation), measure how much `direct_dot` can actually be controlled and whether the model's refuse/comply classification follows.

Two implementation paths, run sequentially:

**0b-simple (residual-stream r̂-projection modulation, fast):** A runtime hook at L15 `hook_resid_post` that subtracts a per-(prompt, condition) scalar × `r̂_unit` from the residual, with the scalar chosen to zero out the target edge type's contribution to `direct_dot`. Mathematically: if we want to remove `delta` units from `direct_dot`, the hook computes `h_new = h − (delta / ‖r̂‖²) × r̂`. After the hook, `h_new · r̂ = h · r̂ − delta`.

Variants per (prompt, condition), with `delta` chosen from sub-experiment 0a's pre-computed sums:

| Variant | delta subtracted | Target post-intervention `direct_dot` |
|---|---|---|
| `ablate_features_pos` | `Σ_features_pos` | drops by Σ_features_pos |
| `ablate_features_neg` | `Σ_features_neg` (negative number → pushes opposite way) | rises by |Σ_features_neg| — H0-2 sign check |
| `ablate_features_all` | `Σ_features_signed` | `direct_dot − Σ_features_signed` |
| `ablate_embeddings_all` | `Σ_embeddings_signed` | `direct_dot − Σ_embeddings_signed` |
| `ablate_errors_all` | `Σ_errors_signed` | `direct_dot − Σ_errors_signed` |
| `ablate_all_edges` | `Σ_all_signed` | `baseline_offset` (full edge ablation) |
| `ablate_all_2x` | `2 × Σ_all_signed` | `baseline_offset − Σ_all_signed` (over-ablation) |

For each variant × 50 prompts × 11 conditions = 3,850 generations. Greedy `max_new_tokens=80` (Stage 08 convention). Classify each generation as refuse/comply via `utils.classify_response`. Compute flip rates vs Stage 06 baseline.

This approach operates on the residual stream directly, NOT through the transcoder graph. Its purpose is to test whether the linearization decomposition is *behaviorally meaningful*: if we take away the predicted amount of `r̂`-projection per edge-type bucket, does the model flip refusal as the linearization identity predicts?

**0b-rigorous (true edge ablation in vendor/circuit-tracer, if 0b-simple results warrant):** Modify `vendor/circuit-tracer` to support edge-level ablation in the linearization framework — replacing each chosen edge's contribution to the target node with its baseline (zero-input transcoder activation) value, propagating through, and regenerating. Per `PAPER_OUTLINE_v2_emnlp.md` §4.1, this is ~3 person-days of patch + tests. Deferred to Week 2 if 0b-simple results indicate it's needed (e.g., if 0b-simple shows surprising behavior that requires "true" edge ablation to disambiguate).

**Outputs (0b):**
- `data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json` — per-variant flip-rate matrix
- `data/results/emnlp_perm_edit/phase0_controllability/controllability_audit_figure.png` — main figure showing `direct_dot` shift achieved per variant vs the variant's predicted shift from 0a
- `data/results/emnlp_perm_edit/phase0_controllability/sign_audit.md` — focused report on H0-2 (negative-ablation sign correctness)

**Acceptance criteria for Phase 0 as a whole:**
- 0a linearization identity verified on 550/550 inputs (<1% reconstruction error).
- 0b H0-1 test: `ablate_all_edges` drives `direct_dot` measurably toward baseline_offset (within ±10% of predicted, measured by re-extracting `direct_dot` from a forward pass with hook active). `ablate_all_2x` drives `direct_dot` past zero on bare prompts (i.e., flips the sign), and the model's REFUSE→COMPLY flip rate on bare-refuse exceeds 90% under this over-ablation.
- 0b H0-2 test: `ablate_features_neg` shifts `direct_dot` in the OPPOSITE direction from `ablate_features_pos`. Sign correctness verified.
- 0b H0-3 test: per-component flip rates reported. If `ablate_errors_all` alone produces >5% flip rate on bare-refuse or JB-comply, error nodes carry publishable mechanism weight.

If H0-1 or H0-2 fails, Track B can still proceed but Phase 0 results become a paper-grade negative result that reshapes Framing A. If both hold, the v2 paper has a clean "transcoder framework gives complete refusal-direction control" pillar.

### 2.4 Sub-experiment 0c — Direction-alignment robustness audit (Georg's 2026-05-17 cosine challenge)

**Goal:** test whether the +0.72 to +0.94 cosine between `r_jb_C` (Ball/Wang convention, toward jailbreak) and `−r̂` (harmless-pointing axis) reported in REPORT § 5.5.2 is a robust geometric fact about the residual stream or is partly inflated by high-dimensional anisotropy / all-ones-direction bias / class-mean averaging artifacts. Tests H0-5 above.

**Method:** three offline diagnostics on the saved residuals (`02b_stats/residuals_L15_per_cond.pt` — already local; CPU-only, no model loading).

**0c.1 Per-prompt cosine.** For each of 5 JB classes C and 50 prompts p, compute the per-prompt JB displacement at L15 pos=−2:
```
δ(p, C) = h_jb_C(p)[pos=-2] − h_bare(p)[pos=-2]
```
Then `cos_p(C) = cos(δ(p, C), r̂)`. Report per-class:
- Mean and std of the 50 per-prompt cosines
- 95% Wilson CI on the rate of `|cos_p(C)| > 0.5`
- The class-mean cosine (computed on `mu_jb_C − mu_bare`) for comparison

**Pass criterion:** per-prompt mean cosine matches the class-mean cosine (from REPORT § 5.5.2) within ±0.10. Per-prompt std < 0.3 (i.e., individual-prompt alignment is consistent, not driven by averaging across high-variance per-prompt directions).

**0c.2 Random-direction baseline.** For each class C, draw N=1,000 random unit vectors in ℝ^2560 (seed=42 for reproducibility). Compute the empirical distribution of `cos(r_jb_C, random_dir)`. Report:
- 95th percentile of `|cos(r_jb_C, random_dir)|` across the 1,000 samples
- The actual `cos(r_jb_C, r̂)` value
- The rank of `cos(r_jb_C, r̂)` within the random-sample distribution

**Pass criterion:** `|cos(r_jb_C, r̂)|` exceeds the 95th percentile of random cosines. Equivalently: `r̂` is in the top-5% of all possible directions for cosine alignment with `r_jb_C`, ruling out "any high-dim direction would look this aligned."

**0c.3 Pearson-style mean-subtraction (all-ones-direction control).** Compute the "centered" cosine where each direction has its scalar mean (across its 2,560 dimensions) subtracted before normalization:
```
r_jb_C_centered = r_jb_C − μ_scalar(r_jb_C) · 1                # 1 is the all-ones vector
r̂_centered     = r̂ − μ_scalar(r̂) · 1
pearson_cos    = cos(r_jb_C_centered, r̂_centered)
```
Report `pearson_cos`, the raw cosine, and the delta `|raw_cos − pearson_cos|`.

**Pass criterion:** `|raw_cos − pearson_cos| ≤ 0.10`. If the delta is larger, both vectors have a substantial projection on the all-ones direction that's inflating the raw cosine; report both cosines in the paper, not just the raw one.

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/direction_robustness.json` — per-class results of 0c.1, 0c.2, 0c.3
- `data/results/emnlp_perm_edit/phase0_controllability/direction_robustness_figure.png` — two-panel figure: (left) per-prompt cosine scatter overlaid with class-mean line per class; (right) random-baseline cosine distribution histogram with r_hat's cosine marked
- Appendix section in `PHASE0_SUMMARY.md` reporting all three diagnostics per class

**Acceptance:**
- 0c.1, 0c.2, 0c.3 pass criteria all met on ≥4 of 5 classes → H0-5 holds, headline §5.5.2 claim stands as-is in the paper
- Any criterion fails on >1 class → H0-5 partial; the paper's headline cosine claim is qualified with the diagnostic results

**Compute:** ~5–10 minutes wall on CPU. No GPU, no LLM, no network. Pure tensor arithmetic on the already-local residuals.

**Why this lives in Phase 0 rather than Track B:** the per-class direction methodology in Track B builds on the assumption that `r_jb_C` and `r̂` are geometrically distinct directions in residual space. If H0-5 fails — i.e., the cosine alignment is largely an artifact rather than reflecting two independently-meaningful directions — then the per-class `r_jb_C^⊥` (the orthogonal component of `r_jb_C` against `r̂`) may be much smaller in causal weight than we currently believe, weakening Track B's foundation. So 0c is a foundational geometric audit, sibling to 0a (linearization audit) and 0b (controllability audit). Together they validate the methodological foundation for both tracks.

### 2.5 Sub-experiment 0d — Top-K feature ablation Pareto sweep (signed)

**Goal:** test whether the assumption underlying all of Stage 08 — that top-attribution features are the causally important ones — holds when measured against the **refusal-direction target** (`direct_dot` at L15 pos=−2) rather than JB-recovery target. Tests H0-6. Separates positive-attribution top-K from negative-attribution top-K to test sign-asymmetry directly (refines H0-2 with Pareto-axis resolution).

**Method:** per-prompt feature ranking + Pareto sweep, GPU runtime intervention.

For each (prompt, condition) instance, partition the per-prompt features by signed attribution to `direct_dot` (from 0a's per-prompt records):
- **pos-K**: top-K features by *positive* attribution (most pro-refusal at this prompt). Ablating them should reduce `direct_dot` toward zero.
- **neg-K**: top-K features by *negative* attribution (most anti-refusal). Ablating them should INCREASE `direct_dot` (push toward stronger refusal); on bare prompts (already refusing) this should produce no behavioral change because bare is already saturated on the refuse side.
- **abs-K**: top-K features by |attribution| (mixed signs; matches Stage 08 Tier 1 convention). Included for direct comparison to v1 § 9.9 numbers.

Sweep K ∈ {1, 5, 10, 20, 50, 100, 500} for each of the three variants. The ablation method is identical to 0b-simple — for each variant × K, compute the summed attribution `delta_K` over the chosen top-K features, then register a runtime hook at L15 `hook_resid_post` that subtracts `delta_K × r̂_unit / ‖r̂‖` from the residual. After the hook, `direct_dot` is reduced by exactly `delta_K`.

**Evaluation conditions:** for each (variant, K), generate on the full 50×11 controlled dataset with `max_new_tokens=80` greedy. Compute per-condition flip rates with Wilson CIs. Primary read: **bare-refuse → comply flip rate** (direct test of refusal-direction control). Secondary read: per-JB-class flip rates (for the cross-condition Pareto comparison).

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/topk_feature_sweep.json` — per (variant, K) flip-rate matrix + Wilson CIs
- `data/results/emnlp_perm_edit/phase0_controllability/topk_feature_pareto_figure.png` — three Pareto curves (pos-K, neg-K, abs-K) of bare-refuse flip rate vs K, with v1 § 9.9 JB-recovery curve overlaid as reference

**Acceptance:**
- H0-6 strong: pos-K curve has clear knee (e.g., K=50 achieves ≥80% of the K=500 flip rate), confirming sparsity.
- H0-6 sign-asymmetry: pos-K curve at K=50 > neg-K curve at K=50 by ≥20 pp on bare-refuse, confirming signed semantics.
- H0-6 fail: if curves are flat or rising linearly, the signal is diffuse and the "top-K = causal" assumption needs revisiting.

**Compute:** 3 variants × 7 K values × 550 prompts × 80 tokens ≈ ~6 h GPU. Can be reduced to ~3 h by sweeping fewer K values (say K ∈ {5, 50, 500}) initially, then filling in the curve only if H0-6 holds at coarse resolution.

**Per-prompt vs corpus-aggregated:** **per-prompt only.** Stage 07 lesson (REPORT § 9.7.2) showed per-prompt construction dominates corpus-union 2.4× at matched recovery; same lesson applies here.

### 2.6 Sub-experiment 0e — Top-K edge ablation Pareto sweep

**Goal:** generalize H0-4 (edge ≠ node) from a single "comprehensive ablation" comparison to a full Pareto axis. Tests H0-7. Determines whether edge-level surgery is methodologically superior across the entire feature-budget axis or only at specific K values.

**Method:** identical structure to 0d but operating at the edge level. For each (prompt, condition):
1. Load the attribution graph from 0a's input.
2. Identify all edges into the measurement target node (`direct_dot` at L15 pos=−2).
3. Rank edges by signed attribution.
4. For pos-K, neg-K, abs-K × K ∈ {1, 5, 10, 50, 100, 500, 1000}, sum the attributions of the chosen top-K edges to get `delta_K_edge`.
5. Register the same 0b-simple residual-stream hook with `delta_K_edge`, generate, classify.

**Direct comparison to 0d:** plot the edge Pareto curve overlaid with the node Pareto curve from 0d at matching K. Per H0-7, the edge curve should be uniformly above the node curve at every K. The integral of the edge-curve-minus-node-curve area is a single scalar that summarizes "how much methodology lever there is in switching from node to edge ablation."

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/topk_edge_sweep.json` — per (variant, K) flip-rate matrix
- `data/results/emnlp_perm_edit/phase0_controllability/topk_edge_vs_node_figure.png` — overlay figure of edge curve and node curve from 0d, with K on x-axis

**Acceptance:**
- H0-7 strong: edge Pareto uniformly above node Pareto at every K, by ≥5 pp at K=50.
- H0-7 partial: edge wins at some K, ties at others. Note in paper as a more nuanced methodology comparison.
- H0-7 fails: edge curve indistinguishable from node curve at every K — edge ablation isn't a methodology lever for this question. Reframe paper accordingly.

**Compute:** ~6 h GPU (similar to 0d). If running 0d and 0e back-to-back, total ~12 h GPU. Could be partially de-duplicated by running pos/neg/abs at the same K in one pass (different deltas but same K means same prompt × condition loop).

**Caveat (true edge ablation):** 0d and 0e both use the residual-stream r̂-projection proxy, not the rigorous edge-ablation in circuit-tracer. The proxy faithfully models "what would happen if the top-K-edges' contribution to direct_dot were removed" via residual-stream modulation, but it does NOT actually rewire the attribution graph. If we observe surprising results that depend on path-level interactions (e.g., features at layer k affecting features at layer k+1 via propagation), 0b-rigorous would be needed for disambiguation.

### 2.7 Sub-experiment 0f — Feature role clustering + semantic annotation

**Goal:** organize transcoder features into a discrete taxonomy of causal roles based on their attribution + activation patterns across the 11-condition dataset. Tests H0-8. Pairs with 0d (top-K Pareto curves) to validate clusters causally — high-impact clusters should dominate small-K ablation; low-impact clusters should only register at large K.

**Method:** purely offline, CPU-only.

**0f.1 Feature profile extraction.** For each feature in the union of attribution-graph features across 550 inputs, construct a profile vector:
- 11 dimensions: per-condition mean signed attribution to `direct_dot` (one per condition)
- 11 dimensions: per-condition fraction-of-prompts-active (binary indicator of presence in attribution graph)
- 1 dimension: layer (L0 to L33)
- = 23-dimensional feature profile per (layer, feature_id) pair

**0f.2 Clustering.** Standardize each profile dimension (z-score), then run hierarchical agglomerative clustering with Ward linkage, cutting at 5–7 clusters (informed by elbow + silhouette). Initial proposed cluster types (validated, not pre-defined):
- **Universal refusal core** — active across all 11 conditions with positive attribution
- **JB-recruited** — active on JB conditions only, signed either way
- **JB-suppressed** — active on bare/ctrl but quenched on JB
- **Anti-refusal** — negative attribution wherever they fire
- **Generic context features** — fire on many conditions but with small attribution (low-impact background)
- **Error-attributable / out-of-cluster** — features that don't fit the above

**0f.3 Semantic annotation.** For each cluster:
1. List the top-25 features by total |attribution| across the corpus
2. Look up each feature's Stage 04 Neuronpedia label (top-activating contexts, top logits, semantic description)
3. (Optional) Feed cluster's top-25 feature labels to Claude Sonnet 4.6 with a prompt like "These 25 transcoder features cluster together in their attribution patterns to a model's refusal direction. Their top-activating contexts and top logits are listed below. Provide a single-sentence semantic summary of what unifies these features." for an auto-generated cluster description.
4. Manual review to refine cluster names and descriptions.

**0f.4 Causal validation against 0d.** For each cluster, compute the cumulative |attribution| within that cluster as a fraction of total |attribution|. Compare against the cluster's contribution to top-K ablation Pareto curves (from 0d's per-feature contributions): clusters with high cumulative |attribution| should dominate small-K ablation budgets. Quantitatively: for K=50, what fraction of the K features belong to each cluster?

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy.json` — per-feature cluster assignment + per-cluster stats
- `data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy_clusters.md` — human-readable per-cluster description with top-25 features and semantic summaries
- `data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy_figure.png` — clustered heatmap (rows = features sorted by cluster, columns = 11 conditions, color = mean signed attribution)

**Acceptance:**
- H0-8 strong: clusters are well-separated (silhouette score > 0.3), with semantic coherence verified by Stage 04 labels and (optional) LLM annotation.
- H0-8 partial: 2–3 clean clusters emerge but others are diffuse. Report the clean ones, treat the rest as "background."
- H0-8 fails: silhouette score < 0.1, no clear cluster structure. Pivot paper framing to a continuous-role description (PCA / NMF axes) rather than discrete clusters.

**Compute:** ~2 h CPU for clustering + ~30 min LLM API for annotation + ~2 h manual review of cluster names. Total: ~5 h, mostly CPU + human-in-the-loop.

### 2.8 Sub-experiment 0g — JB-class perturbation profile + taxonomy synthesis figure

**Goal:** for each JB class, quantify how it perturbs each cluster's activations, producing a "perturbation signature" that mechanistically explains how that class bypasses refusal. Tests H0-9. The synthesis figure becomes the EMNLP paper's headline.

**Method:** offline, CPU-only.

**0g.1 Per-cluster perturbation signature.** For each cluster C_cl from 0f and each JB class C ∈ {fiction, roleplay, analytical, completion, cognitive_reframe}, compute:
```
activation(C_cl, condition) = mean over features f ∈ C_cl of mean over prompts p of activation(f, p, condition)
Δ(C_cl, jb_C) = activation(C_cl, jb_C) − activation(C_cl, bare)         # perturbation vs bare
Δ_sem(C_cl, jb_C) = activation(C_cl, jb_C) − activation(C_cl, ctrl_C)   # prefix-controlled perturbation
```
Each JB class gets a vector of perturbations (one per cluster). The vector IS the "perturbation signature" for that class.

**0g.2 Comparison to per-class `r_jb_C^⊥`.** Compute the projection of each cluster's decoder direction onto `u_C = r_jb_C^⊥ / ‖r_jb_C^⊥‖` (the Track B per-class direction). Per H0-9 / Track B coherence: clusters with large perturbation signature on class C should also have large decoder-projection onto `u_C`. This is the bridge between Phase 0 (taxonomy) and Phase 1 (per-class orthogonalization) — it predicts which Track B per-class edit will work and explains why.

**0g.3 Headline figure.** A two-panel figure:
- Left: heatmap of perturbation signatures (rows = JB classes, columns = clusters, color = signed Δ_sem). Visualizes "how each JB class moves through the taxonomy."
- Right: scatter of per-class cluster decoder-projection vs per-class perturbation. Each point is one (JB class × cluster). Pearson correlation across all points is the quantitative bridge between Phase 0 and Phase 1. Per-class cluster-of-maximum-perturbation gets a colored label.

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/jb_perturbation_signatures.json` — per (cluster, JB class) Δ and Δ_sem; per-class top-perturbed-cluster
- `data/results/emnlp_perm_edit/phase0_controllability/taxonomy_synthesis_figure.png` — the two-panel headline figure
- `data/results/emnlp_perm_edit/phase0_controllability/TAXONOMY_REPORT.md` — paper-grade write-up of the taxonomy story end-to-end

**Acceptance:**
- H0-9 strong: per-class perturbation signatures are distinct (cosine between any two class signatures ≤ +0.5), and the perturbation-vs-decoder-projection scatter has Pearson |r| ≥ 0.7. Tight class-level localization in taxonomy clusters.
- H0-9 partial: signatures are distinct but the decoder-projection correlation is weak (|r| < 0.4). Taxonomy describes correlation, not full mechanism.
- H0-9 fails: signatures are similar across classes — JBs don't class-specifically perturb the taxonomy. Track B's per-class orthogonalization motivation weakens; paper falls back to "universal bypass mechanism" story.

**Compute:** ~1 h CPU for signatures + ~half-day figure design and write-up.

### 2.9 Phase 0 outputs (summary)

```
data/results/emnlp_perm_edit/phase0_controllability/
    # Foundational audit (0a, 0b, 0c)
    linearization_decomposition.json        # 0a — per-(prompt, condition) decomposition
    decomposition_by_condition.json         # 0a — per-condition aggregates
    decomposition_figure.png                # 0a — stacked-bar component figure
    edge_ablation_flip_rates.json           # 0b — per-variant flip rates
    controllability_audit_figure.png        # 0b — main controllability figure
    sign_audit.md                           # 0b — H0-2 negative-ablation correctness
    direction_robustness.json               # 0c — per-prompt + random-baseline + Pearson cosines
    direction_robustness_figure.png         # 0c — two-panel diagnostic figure

    # Causal Pareto sweeps (0d, 0e)
    topk_feature_sweep.json                 # 0d — per-(variant, K) flip-rate matrix + Wilson CIs
    topk_feature_pareto_figure.png          # 0d — pos-K / neg-K / abs-K Pareto curves on bare-refuse
    topk_edge_sweep.json                    # 0e — per-(variant, K) edge-level flip-rate matrix
    topk_edge_vs_node_figure.png            # 0e — edge curve vs node curve overlay

    # Refusal feature taxonomy (0f, 0g)
    feature_taxonomy.json                   # 0f — per-feature cluster assignment + per-cluster stats
    feature_taxonomy_clusters.md            # 0f — human-readable per-cluster description + Stage 04 annotations
    feature_taxonomy_figure.png             # 0f — clustered heatmap (features × conditions, colored by cluster)
    jb_perturbation_signatures.json         # 0g — per (cluster, JB class) Δ and Δ_sem
    taxonomy_synthesis_figure.png           # 0g — TWO-PANEL HEADLINE FIGURE (perturbation heatmap + decoder-projection scatter)
    TAXONOMY_REPORT.md                      # 0g — paper-grade write-up of the end-to-end taxonomy story

    # Phase summary
    PHASE0_SUMMARY.md                       # human-readable headline (covers 0a + 0b + 0c + 0d + 0e + 0f + 0g)
```

### 2.10 Phase 0 compute estimate

| Run | Wall on RTX 5080 16 GB |
|---|---|
| 0a — offline linearization decomposition (CPU only) | ~30 min |
| 0c — direction-alignment robustness audit (CPU only) | ~5–10 min |
| 0b-simple — 7 variants × 550 prompts × 80 tokens, single L15 hook | ~3.5 h GPU |
| 0d — top-K feature Pareto sweep (3 variants × 7 K × 550 prompts × 80 tokens) | ~6 h GPU |
| 0e — top-K edge Pareto sweep (similar shape to 0d) | ~6 h GPU |
| 0f — feature role clustering + Stage 04 annotation + (optional) LLM summarization | ~5 h CPU + LLM API + manual review |
| 0g — perturbation profile + headline figure design | ~1 h CPU + ~half-day figure work |
| 0b-rigorous (if needed) — vendor/circuit-tracer patch + re-eval | ~3 person-days + ~6 h GPU |
| **Phase 0 minimum (0a + 0c + 0b-simple)** | **~4 h GPU + ~40 min CPU** |
| **Phase 0 with taxonomy (0a + 0b-simple + 0c + 0d + 0e + 0f + 0g)** | **~16 h GPU + ~7 h CPU + LLM API + manual review** |

**Practical sequencing on RTX 5080 16 GB.** The Phase 0 work falls into two GPU phases (each ~3.5–6 h) plus heavy CPU/offline analysis. Recommended order:

1. **CPU first (day 1, ~40 min total)**: 0a (linearization decomposition) + 0c (direction-alignment robustness). Share immediately with Georg.
2. **First GPU phase (day 1–2, ~3.5 h)**: 0b-simple comprehensive edge ablation.
3. **Offline analysis on 0a outputs (parallel, ~5 h CPU)**: start 0f clustering using only the per-prompt feature attributions from 0a — no GPU needed; can run while 0b-simple is using the GPU.
4. **Second + third GPU phases (days 2–3, ~12 h)**: 0d top-K feature sweep + 0e top-K edge sweep, back-to-back. Can be interleaved with Phase 1 Variant 1A (Track B) if GPU contention isn't an issue; otherwise sequence them.
5. **Synthesis (day 4, ~1 h CPU + figure work)**: 0g perturbation profile + headline figure once 0d/0e/0f outputs are available.

**Pareto reduction options if compute budget is tight:** the top-K sweeps (0d, 0e) can be reduced from 7 K values × 3 variants to a coarse 3 K values × 3 variants (K ∈ {5, 50, 500}) for an initial ~2 h GPU each. Fill in additional K values only if the coarse curve shows interesting structure that warrants higher resolution.

---

## 3. Per-class direction construction (Track B foundational diagnostics)

For each class C ∈ {fiction, roleplay, analytical, completion, cognitive_reframe}:

```python
# Inputs: existing run_20260430_023247 outputs
r_hat        = torch.load("01_direction/unnormalized_r.pt")[15].float()      # shape (2560,)
residuals    = torch.load("02b_stats/residuals_L15_per_cond.pt")             # dict: cond → (n_prompts, 3, 2560)
h_bare       = residuals["bare"][:, 2, :].mean(0)                            # pos=-2, mean across 50 prompts
h_jb_C       = residuals[f"jb_{C}"][:, 2, :].mean(0)

# Full per-class JB direction (Ball/Wang convention; points toward jailbreak)
r_jb_C       = h_jb_C - h_bare

# Class-specific orthogonal component (removes the shared harmless-axis component)
proj_r_hat   = (r_jb_C @ r_hat) / (r_hat @ r_hat)                            # scalar
r_jb_C_perp  = r_jb_C - proj_r_hat * r_hat

# Unit direction for projection operations
u_C          = r_jb_C_perp / r_jb_C_perp.norm()
```

**Diagnostics to record before any intervention:**

| Quantity | Expected range | Sanity check |
|---|---|---|
| `‖r_jb_C^⊥‖ / ‖r̂‖` | 0.24–0.38 (per § 5.5.2 of REPORT) | If <0.1: the orthogonal component is too small to carry causal weight; methodology may be vacuous for this class. |
| `cos(r̂, u_C)` | exactly 0.0 (by construction) | Numerical floor: <1e-6 |
| Pairwise `cos(u_C, u_{C'})` across 5 classes | small (≤±0.3 expected) | Large positive cosine means classes share orthogonal machinery → dissociation will be hard. Large negative cosine means classes' orthogonal axes are anti-aligned → unexpected, investigate. |

The pairwise cosines are the load-bearing diagnostic for whether dissociation is achievable at all. Phase 1 begins with this diagnostic before running any intervention.

---

## 4. Phase 1 — Per-class runtime hook validation (Track B)

### 4.1 Implementation

The canonical projection hook function (used by all three variants below; the differences are *where* in the model graph it's attached):

```python
def make_orthogonal_projection_hook(u_C: torch.Tensor):
    u_C = u_C.to(dtype=torch.float32, device="cuda")  # do projection in fp32 for stability
    def hook(module, inputs, output):
        h = output  # (batch, seq, d_model), bfloat16
        h_f32 = h.float()
        proj = (h_f32 * u_C).sum(-1, keepdim=True)    # (batch, seq, 1)
        h_new = h_f32 - proj * u_C
        return h_new.to(h.dtype)
    return hook
```

**Layer choice — three variants tested in Phase 1, each with a specific hook target:**

- **Variant 1A — single-layer residual-stream hook at L15**: hook attached to `model.language_model.layers[15]` block output (equivalently `hook_resid_post[15]` in TransformerLens), projecting `u_C` out of the *full residual stream* at L15 once. Matches the Stage 06 + § 5.7 jb_vector_intervention convention. The lightest intervention; the cleanest paper story if it dissociates, but does NOT mirror the Phase 2 weight edit's per-sublayer write-removal structure.

- **Variant 1B — multi-layer sublayer-output hooks at L=15..L=33**: hook attached to `post_attention_layernorm.output` and `post_feedforward_layernorm.output` at each of the 19 layers from 15 onwards. This is the equivalence baseline for Phase 2 because each hook only removes `u_C` from THIS layer's sublayer write (matching what the γ-corrected weight edit does), while the residual stream pass-through is untouched. Use this for Level 1–5 equivalence verification in § 6.

- **Variant 1C — per-layer sweep**: re-runs Variant 1A's residual-stream hook at each layer L ∈ {0, 11, 15, 19, 25, 33} individually (one at a time). Mechanism-diagnostic — identifies which layers matter most for `u_C` removal, supporting the v2 paper's gap-decomposition framing (Framing A).

**Why hook on post-LN sublayer outputs in 1B, not on `o_proj`/`down_proj` outputs directly:** the Phase 2 weight edit produces `post_attn_LN(o_proj(x))` with zero `u_C` component (after the γ-corrected projection). Hooking on the *post-LN* output and projecting out `u_C` (no γ correction needed) produces the same residual update. Hooking on the *pre-LN* `o_proj` output and projecting out `u_C` directly (without γ correction) is *wrong* — it produces a different result from the weight edit. The post-LN hook is both simpler and correct.

The headline result is whichever variant produces the cleanest dissociation. If 1A succeeds, it's the strongest claim (most surgical — single intervention point). If 1A fails but 1B succeeds, the paper's "permanent edit" story uses 1B → Phase 2 equivalence. If neither succeeds, pivot per § 9 risk register.

**Positions:** apply at all positions (matches Stage 06 + Stage 08 `--positions all`). The weight edit in Phase 2 is structurally position-invariant, so the hook must be too for direct equivalence.

### 4.2 Evaluation conditions per class C

For each of 6 hooks (5 per-class + universal `r̂` control + random-direction control) × existing 50×11 controlled dataset, generate with `max_new_tokens=80` greedy, classify response (refuse/comply/unclear) using the same classifier as Stage 08, compare to Stage 06 baseline. The random-direction control samples a unit vector from the L15 residual subspace with seed=42, matched in magnitude to the mean `‖u_C‖` across the 5 classes — this is the negative control for "any random projection would dissociate."

### 4.3 Phase 1 outputs

```
data/results/emnlp_perm_edit/phase1_runtime_hook/
    direction_diagnostics.json          # u_C norms, pairwise cosines, projection magnitudes
    flip_rates_per_hook.json            # (hook_class × eval_class) flip rate matrix + Wilson CIs
    layer_sweep_results.json            # if L sweep is run
    dissociation_matrix.png             # main figure (5 classes × 5 classes + controls)
    PHASE1_SUMMARY.md
```

### 4.4 Phase 1 acceptance bar (gate to Phase 2)

**Primary bar — full reversal:** target class C's JB-comply → REFUSE flip rate ≥ 90 % (i.e., JB success rate drops to ≤ 10 %), AND every other class's flip rate is within ±10 pp of unedited baseline (≤ 0 % expected since no JB hook is applied), AND bare refusal preservation ≥ 48/50.

**Fallback bar — pp dissociation:** dissociation Δ ≥ +30 pp (target class flip rate minus mean-of-other-classes flip rate) on ≥ 3 of 5 classes, with bare refusal preservation ≥ 48/50.

**Failure mode:** if neither bar holds even for the cleanest class, the methodology is not viable — pivot to attention-head-mediated subcircuit (the other option from our brainstorm) or revisit the direction construction (e.g., switch to `r_jb_sem_C^⊥` controlled for prefix). Do not advance to Phase 2 without one of the two bars cleared.

### 4.5 Phase 1 compute estimate

| Run | Wall on RTX 5080 16 GB |
|---|---|
| Direction diagnostics (no generation) | <1 min |
| Variant 1A — single-layer L15 hook: 7 hooks × 550 prompts × 80 tokens | ~3.5 h |
| Variant 1B — multi-layer L15..L33 hook (same hooks, more attachment points; same gen cost) | ~3.5 h |
| Variant 1C — per-layer sweep (6 layers × 7 hooks × 550 prompts × 80 tokens) | ~21 h |
| **Phase 1 total (1A + 1B + 1C)** | **~28 h** |
| **Phase 1 minimum viable (1A + 1B only)** | **~7 h** |

---

## 5. Phase 2 — Permanent weight edit (Track B)

### 5.1 Corrected math (handles Gemma-3 post-LayerNorm)

Gemma-3 architecture has `post_attention_layernorm` between `o_proj` and the residual add, and `post_feedforward_layernorm` between `down_proj` and the residual add. The naive Arditi recipe of `W_new = (I − u_C u_C^T) W` projects `u_C` out of the **pre-LayerNorm** output of the sublayer — but the residual update is the **post-LayerNorm** output, and γ-scaling per dimension re-introduces a component along `u_C`.

The correct derivation for attention sublayer:

Let `y = o_proj.input @ W^T` where `W = o_proj.weight` of shape `(d_model, d_head·n_heads)`. The residual update is:
```
Δh = post_attention_layernorm(y)
   = γ_post_attn ⊙ (y / RMS(y))
   = c · (γ_post_attn ⊙ y)              where c = 1/RMS(y), a scalar per token position
```

For `u_C^T Δh = 0` to hold for all `y`:
```
u_C^T (γ_post_attn ⊙ y) = 0
(γ_post_attn ⊙ u_C)^T y = 0              (element-wise commutativity)
v_attn^T y = 0                            where v_attn = γ_post_attn ⊙ u_C
```

For this to hold for all `o_proj.input`, we need `v_attn^T W^T = 0`. Apply the left-projection to `W`:
```
v̂_attn = v_attn / ‖v_attn‖
W_new = (I − v̂_attn v̂_attn^T) W
```

Verification: `v_attn^T W_new = v_attn^T (I − v̂_attn v̂_attn^T) W = (v_attn^T − ‖v_attn‖ v̂_attn^T) W = 0` ✓.

**Same logic for `down_proj`** with `v_ff = γ_post_ff ⊙ u_C`:
```
v̂_ff[L] = (γ_post_ff[L] ⊙ u_C) / ‖γ_post_ff[L] ⊙ u_C‖
down_proj.weight[L]_new = (I − v̂_ff[L] v̂_ff[L]^T) · down_proj.weight[L]
```

**Per-layer `γ` correction:** `γ_post_attn[L]` and `γ_post_ff[L]` are *different parameters per layer*. The projectors `v̂_attn[L]` and `v̂_ff[L]` must be recomputed for each of Gemma-3's 34 layers. They are not interchangeable.

**Gemma-3 RMSNorm parameterization caveat:** Gemma's `RMSNorm.forward` computes `output * (1.0 + self.weight)` (not `output * self.weight`). The effective `γ` to use in the derivation is `(1 + post_attention_layernorm.weight)`, not `post_attention_layernorm.weight` directly. Easy to get wrong.

### 5.2 Weight matrices in scope

| Matrix | Phase | Per-layer | Justification |
|---|---|---|---|
| `model.language_model.layers[L].self_attn.o_proj.weight` | 2a | ✓ (34 layers × 5 classes) | Attention output write to residual; passes through `post_attention_layernorm`. |
| `model.language_model.layers[L].mlp.down_proj.weight` | 2a | ✓ (34 layers × 5 classes) | MLP output write to residual; passes through `post_feedforward_layernorm`. |
| `model.embed_tokens.weight` (= `lm_head.weight`, tied) | 2b (extension) | once per class | Token embedding write to residual at input; no γ scaling (no LayerNorm immediately after embed in Gemma-3, just × √d_model). **Deferred to 2b** because embed/lm_head tying causes the edit to also modify unembedding, which may degrade output token distribution. Adding only if Phase 2a alone fails the equivalence bar. |

Excluded: input layer norms, query/key/value projections (these don't write to residual; they read), MLP up_proj/gate_proj (these write to MLP intermediate, not residual).

### 5.3 Phase 2a implementation outline

```python
def edit_model_per_class(model, u_C: torch.Tensor) -> dict:
    """Returns a dict of {param_name: (old_weight, new_weight)} for rollback."""
    rollback = {}
    for L in range(34):
        layer = model.language_model.layers[L]

        # o_proj edit using γ_post_attn[L]
        gamma_attn = 1.0 + layer.post_attention_layernorm.weight.float()
        v_attn = gamma_attn * u_C
        v_hat_attn = v_attn / v_attn.norm()
        W_o = layer.self_attn.o_proj.weight
        rollback[f"layers.{L}.self_attn.o_proj.weight"] = W_o.clone()
        proj_attn = torch.eye(W_o.shape[0], device=W_o.device, dtype=torch.float32) - torch.outer(v_hat_attn, v_hat_attn)
        layer.self_attn.o_proj.weight.copy_((proj_attn.to(W_o.dtype) @ W_o))

        # down_proj edit using γ_post_ff[L]
        gamma_ff = 1.0 + layer.post_feedforward_layernorm.weight.float()
        v_ff = gamma_ff * u_C
        v_hat_ff = v_ff / v_ff.norm()
        W_d = layer.mlp.down_proj.weight
        rollback[f"layers.{L}.mlp.down_proj.weight"] = W_d.clone()
        proj_ff = torch.eye(W_d.shape[0], device=W_d.device, dtype=torch.float32) - torch.outer(v_hat_ff, v_hat_ff)
        layer.mlp.down_proj.weight.copy_((proj_ff.to(W_d.dtype) @ W_d))

    return rollback
```

(Sketch only; production code will live in `scripts/emnlp_perm_edit/08b_direction_edit.py` and have proper dtype handling, dry-run mode, checkpoint saving of the edited model.)

### 5.4 Phase 2 outputs

```
data/results/emnlp_perm_edit/phase2_weight_edit/
    edited_models/                       # 5 per-class edited checkpoints (or LoRA-style delta files)
        gemma3_4b_orthogonalized_fiction/
        gemma3_4b_orthogonalized_roleplay/
        ...
    flip_rates_per_class.json
    equivalence_verification.json        # see § 6 below
    PHASE2_SUMMARY.md
```

### 5.5 Phase 2 acceptance bar

Same primary/fallback bars as Phase 1 — but now applied to the **weight-edited model running without any hook**. Additionally, the equivalence verification protocol in § 6 must pass.

### 5.6 Phase 2 compute estimate

| Run | Wall on RTX 5080 16 GB |
|---|---|
| Apply weight edit (5 classes × 34 layers × 2 matrices) | <5 min total (matrix ops only) |
| Re-evaluate edited models on 50×11 (5 classes × 550 prompts × 80 tokens) | ~2.5 h |
| Equivalence verification (capture residuals on 50×11 for both hook + weight) | ~1 h |
| **Phase 2 total** | **~3.5 h** |

**VRAM note:** Phase 2 keeps two model copies in memory during equivalence verification (unedited + edited). On 16 GB, this is tight. If needed, capture residuals from each model in separate runs and compare offline rather than holding both in VRAM simultaneously.

---

## 6. Equivalence verification protocol (load-bearing for Track B "permanent edit" claim)

The paper's "permanent edit" claim depends on demonstrating that the Phase 2 weight-edited model behaves equivalently to the Phase 1 **Variant 1B** runtime-hooked model (multi-layer sublayer-output hooks at L=15..L=33). Variant 1A (single-layer L15 residual-stream hook) is *not* the equivalence baseline — it's structurally different from the weight edit. We define five levels of equivalence, in decreasing order of mechanistic rigor:

### 6.1 Level 1 — Residual-stream cosine equivalence

For each of 50 prompts × 11 conditions × 34 layers × 3 measurement positions [−5, −3, −2] = ~56,000 measurement points, compute:

```
cos_L_pos = cos( h_runtime_hook[L, pos], h_weight_edit[L, pos] )
```

**Bar:**
- Mean across all 56k points: **≥ 0.99**
- Minimum across all 56k points: **≥ 0.95**
- Reported per-layer mean cosine plotted as a line chart (34 points) — should be flat near 1.0 across layers.

### 6.2 Level 2 — Projection-onto-`u_C` verification

For each (input, layer), compute the residual-stream projection magnitude onto `u_C`:

```
proj_magnitude = |h[L, pos=-2] · u_C| / ‖h[L, pos=-2]‖
```

Both runtime-hook and weight-edit variants should have this near zero at all layers L ≥ initial_hook_layer.

**Bar:**
- Both variants: `proj_magnitude ≤ 0.01` (i.e., `u_C` component is < 1 % of total residual magnitude) at L ≥ 15.
- Variants agree within 5× of each other on this metric (i.e., if hook gives 0.001 and weight gives 0.004, that's fine; if hook gives 0.001 and weight gives 0.05, the γ correction is incomplete).

### 6.3 Level 3 — Output token equivalence

Generate `max_new_tokens=80` greedy under both variants for the same 550 inputs. Compute token-level agreement rate:

```
token_agreement = mean over (input × position) of [tok_hook == tok_weight]
```

**Bar:** ≥ 95 % token-level agreement, averaged across 550 inputs. Mismatches concentrate at semantically equivalent rephrasings (acceptable) rather than refusal/comply boundary flips (would invalidate equivalence).

### 6.4 Level 4 — Classification equivalence

Classify each generation as REFUSE/COMPLY/UNCLEAR using the existing Stage 08 classifier. Compute classification agreement rate:

```
classification_agreement = mean over inputs of [class_hook == class_weight]
```

**Bar:** ≥ 98 % classification agreement, with any disagreements documented per-prompt.

### 6.5 Level 5 — Aggregate dissociation matrix equivalence

The full dissociation matrix (5 hook classes × 5 eval classes + bare + ctrl_avg = 7 × 5 = 35 cells) under both variants. Each cell is a flip rate with Wilson 95 % CI.

**Bar:** Every cell's flip rate matches within **± 2 pp absolute** between variants. Wilson CIs overlap on every cell.

### 6.6 Reporting in the paper

The headline equivalence sentence:

> Across 50 prompts × 11 conditions × 34 layers × 3 positions, the weight-edited model produces residual streams cosine-similar to the runtime-hooked model with mean **≥ 0.99** (minimum **≥ 0.95**); token-level output agreement is **≥ 95 %**; classification agreement is **≥ 98 %**; and the full per-class dissociation matrix matches within **± 2 pp** on every cell with overlapping Wilson 95 % CIs.

If this passes, we have published "a true permanent weight edit." If only Levels 4–5 pass but Levels 1–3 fail, we report "behavioral equivalence (output-level)" honestly and weaken the mechanism claim. If Level 5 fails, the weight edit is not a substitute for the hook; we fall back to framing as "deployable inference-time circuit modification."

---

## 7. Phase 3 — Generalization and helpfulness (Track B)

### 7.1 Out-of-distribution jailbreak set

**Dataset:** HarmBench's 40 standard behaviors × 5 JB templates (the same templates as our controlled dataset, applied to new harmful requests). If HarmBench setup is friction-heavy, fall back to AdvBench's untemplated harmful requests as a sanity check (tests whether the edit generalizes beyond our specific templates).

**Bar per class C:** on the OOD JB set restricted to class C templates, the edit produces flip-rate ≥ 80 % (allowing 10 pp slippage from in-distribution due to domain shift).

**Bar cross-class:** within ±15 pp of unedited baseline (allowing some additional slippage on OOD vs ID).

**Compute:** ~3 h per class on RTX 5080 16 GB.

### 7.2 Helpfulness benchmark

**Dataset:** 100 benign single-turn prompts from MT-Bench's first turn (categories: writing, roleplay, reasoning, math, coding, extraction, STEM, humanities). Generate from unedited and per-class-edited models.

**Metric:** LM judge (Claude Sonnet 4.6, `claude-sonnet-4-6` — or GPT-4o as cross-validator) on a 1–10 quality scale comparing edited vs unedited outputs per prompt, plus output length distribution check.

**Bar:** mean quality score within **≤ 0.5** of unedited (on 1–10 scale). No more than 5 % of prompts get a "significantly worse" verdict from the judge.

**Compute:** ~1 h per class on RTX 5080 16 GB + ~30 min LM judge API calls per class.

### 7.3 Phase 3 outputs

```
data/results/emnlp_perm_edit/phase3_generalization/
    harmbench_ood_results.json
    helpfulness_results.json
    PHASE3_SUMMARY.md
```

---

## 8. Branch strategy and parallel-track timeline

### 8.1 Branch

`emnlp-perm-edit` (already created off `l15-refactor` HEAD, this spec is committed as the branch's first non-trivial work).

**Do not commit EMNLP work to `l15-refactor`.** That branch is the ICML submission's frozen reference. The new branch can freely add `scripts/emnlp_perm_edit/...` and dedicated test files.

### 8.2 Parallel-track timeline (assuming mid-June EMNLP deadline)

Track A (Phase 0: foundational audit 0a/0b/0c + refusal feature taxonomy 0d/0e/0f/0g) and Track B (Phases 1–3) execute **in parallel**, with Track A given slight priority so results come sooner for Georg's foundational question. Phase 0's taxonomy work (0d–0g) reuses 0a's attribution graph outputs and runs partially in parallel with Track B's GPU phases.

| Week | Track A (Phase 0) | Track B (Phases 1–3) | Joint deliverable |
|---|---|---|---|
| **Week 1** (5/18–5/24) | **PRIORITY**: 0a (linearization decomposition, ~30 min CPU) + 0c (direction-alignment robustness, ~10 min CPU). Share with Georg same day. Then implement & launch 0b-simple driver (~3.5 h GPU). Start 0f feature clustering on 0a outputs in parallel (CPU). | Phase 1 direction diagnostics (CPU, <1 min) + Variant 1A single-layer L15 hook on 5 classes + controls (~4 h GPU). | End of Week 1: H0-1/2/5 verdicts, Phase 1A dissociation matrix, preliminary feature clustering. |
| **Week 2** (5/25–5/31) | 0d top-K feature Pareto sweep (~6 h GPU) + 0e top-K edge Pareto sweep (~6 h GPU), back-to-back. Finish 0f clustering + 0g perturbation profile and headline figure synthesis in parallel (CPU). | Variant 1B multi-layer hook (~4 h GPU) + acceptance check on 1A + 1B. If primary or fallback bar clears: begin Phase 2 weight edit implementation. | End of Week 2: H0-6/7/8/9 verdicts, Track A taxonomy figure ready, Track B go/no-go on Phase 2. |
| **Week 3** (6/1–6/7) | (If needed) 0b-rigorous edge ablation in vendor/circuit-tracer (~3 person-days). Variant 1C per-layer sweep also feeds Track A. | Phase 2 weight-edit implementation + equivalence verification (~4 h GPU + iteration time). | End of Week 3: paper-grade Track A taxonomy + Track B headline numbers. |
| **Week 4** (6/8–6/14) | Track A finalization (taxonomy refinements based on writing feedback). | Phase 3 generalization + helpfulness benchmarks (~20 h GPU). Paper drafting, figure polish, supplementary materials. | EMNLP submission draft. |

**Timeline risks with expanded Phase 0 scope:**
- Adding 0d + 0e + 0f + 0g pushes ~12 hours of additional GPU into Week 2 plus ~5 hours of CPU/manual work. Doable but tight. If Week 1's H0-1 sign-audit reveals a bug to debug, the Week 2 GPU plan compresses.
- Mitigation: 0d/0e can be reduced to coarse 3-K-value sweeps (~2 h each instead of ~6 h) if compute is tight. Documented in § 2.10.

If Phase 0's H0-1 fails strongly (comprehensive edge ablation doesn't drive `direct_dot` toward baseline_offset): pause Track B Phase 2 weight-edit work and prioritize understanding the gap. Track A's negative result becomes a paper in its own right and Track B's per-class permanent-edit claim may need to be reframed.

If H0-6 or H0-8 fail (Pareto curves are flat or clusters are diffuse): the taxonomy story weakens; the paper's headline shifts toward the controllability audit (0a/0b) and Track B's permanent-edit demonstration, with the taxonomy demoted to a section rather than a pillar.

If Phase 1's primary/fallback bars BOTH fail: pivot Track B to attention-head-mediated subcircuit (the parallel option from the brainstorm, noted for future Georg discussion). Track A continues regardless.

### 8.3 Compute budget

| Phase | Wall (RTX 5080 16 GB) | Cumulative |
|---|---|---|
| Phase 0 foundational only (0a + 0c + 0b-simple) | ~4 h GPU + ~40 min CPU | 4 h GPU |
| Phase 0 with taxonomy (+ 0d + 0e + 0f + 0g) | +12 h GPU + ~5 h CPU + LLM API | 16 h GPU + ~6 h CPU |
| Phase 1 minimum (1A + 1B only) | ~7 h | 23 h GPU |
| Phase 1 full (1A + 1B + 1C) | ~28 h | 44 h GPU |
| Phase 2 (weight edit + re-eval + equivalence) | ~3.5 h | ~26–47 h GPU |
| Phase 3 (OOD JB + helpfulness, 5 classes) | ~20 h | ~46–67 h GPU |

All on local RTX 5080 16 GB. No RunPod / H100 required. **VRAM caveats** on 16 GB:
- Single-stream sequential generation (batch size 1) is the safe pattern for Gemma-3-4B-IT bf16 (weights ~7.5 GB + KV cache).
- Phase 2 equivalence verification benefits from holding both unedited and edited model copies — likely OOMs on 16 GB. Capture residuals in two passes and compare offline.
- Phase 0 0d and 0e sweeps register/remove hooks per generation; no model-state buildup, so VRAM headroom is fine.

**Scope-trimming options if 4-week timeline gets tight:**
1. Cut 0e (top-K edge sweep) to a coarse 3-K version — saves ~4 h GPU. H0-7 still answerable.
2. Cut 0d to coarse 3-K version — saves another ~4 h GPU. H0-6 still answerable.
3. Skip 0g LLM annotation and rely on Stage 04 Neuronpedia labels alone — saves ~1 h + API cost. Less polished taxonomy descriptions.
4. Defer 0b-rigorous unconditionally (already planned).
5. Phase 1 Variant 1C (per-layer sweep) is the next item to cut — it's mechanism-supporting for Framing A but not gating.

Minimum-viable EMNLP submission with all key claims intact: Phase 0 foundational (0a + 0b-simple + 0c) + 0d coarse sweep + 0f clustering + 0g synthesis + Phase 1A + Phase 1B + Phase 2. Total ~22 h GPU + ~6 h CPU. Doable in 2.5–3 weeks, leaves slack for Phase 3 and paper writing.

---

## 9. Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| Phase 0 H0-1 fails (comprehensive edge ablation does NOT drive `direct_dot` to baseline) | Medium | This is itself a paper-grade publishable negative result: localizes where the gap lives (which edge type is missing). Phase 0b-rigorous (true circuit-tracer edge ablation) becomes the disambiguator if 0b-simple's residual-stream proxy doesn't tell the full story. |
| Phase 0 H0-2 fails (negative-attribution ablations don't flip sign as expected) | Low-medium | Critical — signals a sign-handling or basis bug. PAUSE all downstream claims and debug attribution math. The Stage 03 linearization identity holds to <0.4 % per prompt, so a sign bug would manifest as a per-prompt error correlated with signed-attribution distribution. |
| Phase 0 0a packed graphs not locally available (only on HF) | Low | Pull via `scripts/pipeline/fetch_graph_data.py --source 02c`. Bandwidth-dependent; ~485 MB total for the JSON.gz packed graphs. |
| Phase 1 dissociation fails primary AND fallback bars | Medium-low | Phase 1 is cheap (~7 h minimum). If it fails, the failure mode itself is informative — pivot to attention-head-mediated subcircuit and re-use the existing per-class direction infrastructure. Track A continues regardless. |
| Phase 2 γ-corrected edit has residual `u_C` component | Medium | Level 2 of equivalence verification catches this directly. If proj_magnitude > 0.01 ‖h‖, debug per-layer (likely Gemma's `1 + weight` parameterization handled wrong, or multi-layer interaction not anticipated). |
| Embed/lm_head tying breaks output quality when `W_E` is included | Medium-high | `W_E` deferred to Phase 2b. Default Phase 2a excludes it. Re-test in 2b only if 2a alone fails Level 5 equivalence. |
| Per-class `u_C` vectors are too cosine-similar to dissociate | Medium-low | The direction diagnostic in § 3 measures this BEFORE any intervention. If pairwise `cos(u_C, u_{C'})` > 0.5, dissociation will be hard; flag in Phase 1 plan adjustment. |
| Cross-class promiscuity persists despite the orthogonal-component design | Medium | Our prior data (§ 9.8.3 of REPORT) shows the **activation-selectivity ≠ causal-selectivity** problem at the MLP-feature level. We don't yet have evidence whether `r_jb_C^⊥` resolves it. This is the central empirical question of Phase 1; both outcomes are publishable. Phase 0's H0-4 result also informs this — if edge ablation is strictly more surgical than node ablation in Track A, the same lever applies in Track B's design. |
| RTX 5080 16 GB VRAM constrains Phase 2 equivalence verification (need 2 model copies) | Medium | Capture residuals from each model in separate passes; compare offline. Adds ~30 min wall but avoids OOM. |
| OOD JB set requires unanticipated setup | Low | HarmBench has a standard `harmbench-evaluator` interface; if integration is friction-heavy, AdvBench is plug-and-play. |
| LM judge for helpfulness is biased / unreliable | Low | Run two independent judges (Claude Sonnet 4.6 + GPT-4o); report inter-judge agreement; for headline number use the more conservative judge. |
| Timeline slips past EMNLP deadline | Medium | Phase 0 + Phase 1 + Phase 2 alone constitute a publishable result if Phase 3 slips. EMNLP submission can omit helpfulness benchmark with a "limitations" paragraph and still tell a complete in-distribution story. NeurIPS December cycle is the backup. |

---

## 10. Status of prior open questions (originally for Georg, mostly addressed by 2026-05-17 mentor exchange)

1. **`W_E` inclusion** — Phase 2a defers `W_E` due to embed/lm_head tying. Adds in Phase 2b if Phase 2a alone misses Level 5 equivalence. *Resolution: keep deferred unless empirics force the change.*

2. **Multi-layer `u_C` extraction** — current plan extracts `u_C` once at L15 and applies it (with per-layer γ correction) at every layer. Variant 1C per-layer sweep diagnoses whether per-layer `u_C[L]` is needed; deferred to Week 3. *Resolution: handled by 1C; not blocking.*

3. **Phase 1 fallback to attention-head-mediated subcircuit** — if `r_jb_C^⊥` fails the bar, the brainstorm identified attention-head attribution + targeted W_O edit as the next-best option. *Resolution: documented in § 9 risk register; implement only if Phase 1 fails.*

4. **EMNLP timeline vs NeurIPS December cycle** — if Phase 3 helpfulness benchmarks slip, submit to EMNLP with a limitations note. NeurIPS is the backup. *Resolution: in § 9.*

5. **Cross-model replication (Qwen3, Gemma-2-9B)** — listed in v2 paper outline §4.6/§4.7. *Resolution: out of scope for this EMNLP plan; ride Ruqiya's Qwen3 pipeline rebase as a follow-up.*

6. **Track A (Phase 0) edge-ablation methodology — Georg's 2026-05-17 ask** — *resolution: incorporated as Section 2. Phase 0 is the foundational track running in parallel with Phase 1, with slight priority on early results for Georg.*

---

## 11. Where things live (paths and conventions)

| Artifact | Path |
|---|---|
| Existing per-class direction infrastructure | `scripts/analysis/jb_vector_intervention_per_class.py`, `02b_stats/residuals_L15_per_cond.pt`, `01_direction/unnormalized_r.pt` |
| Existing attribution graph data (Phase 0 input) | `data/results/pipeline_runs/run_20260430_023247/02_attribution/graph_data/<prompt_id>__<condition>.json.gz` (pull locally via `scripts/pipeline/fetch_graph_data.py` if missing) |
| Stage 06 hook helper (reused) | `scripts/pipeline/utils.py::make_intervention_hook` |
| Stage 08 classifier (reused for Phase 0, 1, 2, 3) | `scripts/pipeline/utils.py::classify_response`, `is_coherent` |
| **New Phase 0 entrypoints (foundational audit)** | `scripts/emnlp_perm_edit/00_linearization_decomposition.py` (0a), `scripts/emnlp_perm_edit/00_edge_ablation_runtime.py` (0b-simple), `scripts/emnlp_perm_edit/00_edge_ablation_rigorous.py` (0b-rigorous, if needed), `scripts/emnlp_perm_edit/00_direction_robustness.py` (0c) |
| **New Phase 0 entrypoints (taxonomy)** | `scripts/emnlp_perm_edit/00_topk_feature_sweep.py` (0d), `scripts/emnlp_perm_edit/00_topk_edge_sweep.py` (0e), `scripts/emnlp_perm_edit/00_feature_taxonomy.py` (0f), `scripts/emnlp_perm_edit/00_jb_perturbation_signature.py` (0g) |
| New Phase 1 entrypoints | `scripts/emnlp_perm_edit/01_compute_directions.py`, `scripts/emnlp_perm_edit/01_runtime_hook_v{1A,1B,1C}.py`, `scripts/emnlp_perm_edit/01_runtime_hook_controls.py` |
| New Phase 2 entrypoint | `scripts/emnlp_perm_edit/02_weight_edit.py` |
| New equivalence verification | `scripts/emnlp_perm_edit/03_equivalence_verify.py` |
| New Phase 3 entrypoints | `scripts/emnlp_perm_edit/04_harmbench_ood.py`, `scripts/emnlp_perm_edit/05_helpfulness_judge.py` |
| Phase outputs | `data/results/emnlp_perm_edit/phase{0,1,2,3}_*/` |

---

*Drafted 2026-05-17 from brainstorming session with Mahmoud (Track B); updated 2026-05-17 to incorporate Georg's foundational controllability-audit ask as Phase 0 / Track A and to anchor compute estimates on RTX 5080 16 GB VRAM (laptop). Both tracks begin Week 1; Track A prioritized for early results.*
