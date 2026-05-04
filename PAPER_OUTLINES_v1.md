# Refusal-Lens — Paper Outline v1 (ICML mech-interp workshop)

**Track**: ICML 2026 Mechanistic Interpretability Workshop, 4-page short paper.
**Deadline**: 2026-05-08 EOD (Friday).
**Status**: all experiments complete; results renormalized; figures generated; report (`REPORT_run_20260430_023247.md`) cite-ready. Writing only.
**Companion document**: `PAPER_OUTLINE_v2_emnlp.md` describes the deeper main-conference version (extra experiments + longer timeline).

---

## 0. Positioning relative to prior work

The cited refusal-direction literature establishes three things, all of which are **building blocks** for our paper, not competitors:

| Prior | Establishes | Where it lives in our paper |
|---|---|---|
| Arditi et al. NeurIPS 2024 (`2406.11717`) | Refusal in LLMs is mediated by a single residual-stream direction; ablating it bypasses, adding it forces refusal. | §2 background; we replicate on Gemma-3-4B-IT and Qwen3-4B as a sanity check. |
| Ball, Kreuter, Panickssery 2024 (`2406.09289`) | Per-class JB vectors are mutually parallel (cos 0.4–0.6) and cross-class transferable; effective JBs reduce a separately-extracted "harmfulness direction." Vicuna/Qwen-1.5/MPT, residual-stream only. | §2 background; cite the JB-vector cosine + harmfulness-suppression results. We replicate the bidirectional symmetry on Gemma-3 with the explicit refusal direction (cos +0.94 at pos=−2) and **then ask the question they flag as open**: is the directional effect mediated by sparse features or by something else? |
| Wang et al. May 2025 (`2505.17306`) | Refusal direction is universal across 14 languages; cross-lingual JB vector edits along the same axis. Llama-3 / Qwen-2.5 / Gemma-2, no feature-level analysis. | §2 background; their universality strengthens the "1-D gate" claim our paper makes. We do not address multilinguality. |

**The neighbor closest to risk of overlap is Ball 2024.** Their § 6 Discussion explicitly flags the open question:

> "Given the correlational perspective of our study and less significant harmfulness reduction results for the MPT 7B model, further investigations are necessary to understand whether there is a causal relationship between the harmfulness feature suppression and jailbreak success. An analysis of how different model components contribute to the jailbreak feature and harmfulness directions, and whether any patterns emerge based on jailbreak type, would be valuable."

**Our paper answers that question.** We use circuit-tracer + transcoders (which Ball does not) to attribute the directional effect to specific transcoder features, and find that **even the strongest sparse subcircuit recovers only ~35% of the directional intervention's effect.** That gap, not the directional effect itself, is the paper's contribution.

---

## Outline E — *Refusal as a low-rank gate over a distributed expression*

### 1. Title (drafts)

- "Refusal as a Low-Rank Gate: Why Directional Interventions Outperform Sparse Feature Ablations on Jailbreak-Induced Compliance"
- "The Direction-vs-Ablation Gap in Refusal Circuits"
- "Distributed Expression, Localized Gating: A Circuit-Tracing View of Refusal in Gemma-3"

### 2. One-sentence thesis

> Refusal in instruction-tuned LLMs is gated by a single low-dimensional direction in the residual stream — but its expression is distributed: even the cleanest sparse transcoder-feature subcircuit recovers only **31.5 %** of jailbreak-induced compliance flips, plateauing at **34.8 %** under an unconstrained per-prompt top-N sweep, while a 1-D directional intervention recovers **100 %** at the same magnitude (1·‖*r̂*‖). The ~3× gap is robust across 16 (subcircuit × K/F) configurations and survives baseline renormalization to a Stage-06 reference set.

### 3. Headline numbers (paper-grade, all in hand)

All on Gemma-3-4B-IT, controlled 50-prompt × 11-condition dataset, L15 measurement, baselines renormalized to Stage 06 `causal_results.json` (max_new_tokens=200, H100). 95% CIs are Wilson binomial.

| Quantity | Value | Source |
|---|---|---|
| L15 direction `pro_refusal_add` flip rate (jb-comply → refuse) | **89/89 = 100 %** [95.9, 100] | Stage 06 |
| L15 direction `anti_refusal_sub` flip rate (bare-refuse → comply) | **49/50 = 98 %** [89.5, 99.7] | Stage 06 |
| L15 direction force-refuse on benign | **10/10 = 100 %** | Stage 06 phase-2c |
| Strongest subcircuit (`universal_refusal_core` k100_f20, 47 features) | **31.5 %** [22.8, 41.7] | Stage 08 Tier 2 |
| Per-prompt top-50 unconstrained features | **34.8 %** [25.7, 45.2] | Stage 08 Tier 1 |
| Per-prompt top-100 (Pareto plateau) | 31.5 % [22.8, 41.7] | Stage 08 Tier 1 |
| Per-prompt random-6 control | 9.0 % [4.6, 16.7] | Stage 08 Tier 1 |
| top-5 vs random-6 selectivity ratio | 1.37× (CIs overlap) | Stage 08 Tier 1 |
| cos(*r̂*, mean(bare) − mean(jb)) at pos=−2 | **+0.72 to +0.94** across classes | § 5.5 alignment |
| Magnitude of JB edit, ‖*r_jb*‖ / ‖*r̂*‖ | **0.40 to 1.11** across classes | § 5.5 alignment |
| Linearization identity error | 0.08–0.36 % | Stage 03 |

### 4. Section structure (4-page workshop)

A 4-page workshop submission allots roughly 2 columns × 4 pages = ~3,000 words + figures. The trade is depth-of-experiments-vs-breadth-of-claim. The structure below leads with the gap result and supports it tightly.

#### § 1. Introduction (≈ 0.5 page)

- One paragraph setup: refusal-direction line of work [Arditi 2024, Ball 2024, Wang 2025] established a 1-D directional control of refusal. Cite each on the specific contribution.
- One paragraph open question (cite Ball 2024 § 6 Discussion verbatim or close): given that direction works, *what is it doing in the model's transcoder feature space*?
- One paragraph our contribution: we use circuit-tracer + gemma-scope transcoders to compare 1-D directional intervention against sparse feature ablation on the same dataset. The gap is 100 % vs ≤35 %, robust across 16+ subcircuit constructions, and the saturation suggests the residual ~65 % is genuinely outside the transcoder approximation.
- Three-bullet contributions list.

#### § 2. Methods (≈ 0.5 page)

- Refusal direction extraction (Arditi-style difference-in-means, n=64 harmful + 64 harmless). One sentence.
- Controlled dataset: 50 harmful prompts × 11 conditions (bare + 5 ctrl_* + 5 jb_*); ctrl_* is length-matched to JB to isolate prefix-induced shifts from JB-semantic shifts. Cite our earlier controlled-design infrastructure.
- Stage 06 directional intervention: L15 unnormalized r̂, additive at all token positions every forward pass (Tejas-style). Cite Arditi.
- Stage 08 sparse-feature ablation: clamp transcoder feature value to 0 at all positions across all forward passes during generation; baseline classifications renormalized to Stage 06. Per-prompt subcircuit construction (top-K + ≥F-fraction frequency aggregation).

#### § 3. The direction–ablation gap (≈ 1.5 pages — the meat)

- **3.1** Directional intervention (Stage 06): 89/89 JB→refuse, 49/50 bare→comply, 10/10 benign→refuse. **(Cite as confirmation of Arditi 2024 on Gemma-3 family.)** Single sentence framing.
- **3.2** Cosine alignment of empirical JB direction with r̂: cos(r̂, mean(bare) − mean(jb)) = +0.72 to +0.94 at pos=−2; magnitude 0.40 to 1.11 ‖r̂‖. Confirms JB edits literally happen along r̂. **(Adjacent to Ball 2024 — frame as direct measurement of what they implied.)**
- **3.3** Sparse subcircuit ablation (Tier 2): 8 subcircuits × 2 K/F configs = 16 ablations. Headline table. Best is `universal_refusal_core` at 31.5 %. Class-specific subcircuits underperform (e.g. `jb_fiction_specific_vs_ctrl` recovers 0 % on fiction).
- **3.4** Per-prompt top-N Pareto (Tier 1): 1, 5, 10, 20, 50, 100 features per prompt + random-6 control. Plateau at top-50 (34.8 %), regression-or-plateau at top-100. Random-6 control gives 9.0 %; top_5 vs random_6 is only 1.37× (CIs overlap).
- **3.5** Figure: Pareto curve with direction reference + random control + best subcircuits overlaid. Single load-bearing figure for the paper.

#### § 4. Discussion (≈ 1 page)

- The ~65 % residual: candidates for what it's hiding. (a) Attention-head paths frozen during attribution; (b) transcoder reconstruction errors (the "error nodes"); (c) per-prompt features outside the top-100 (excluded by the Pareto plateau evidence); (d) cross-position interactions our pos=−2 measurement misses. **Each of these is a falsifiable hypothesis for the main-conference follow-up** (forward-reference `PAPER_OUTLINE_v2_emnlp.md`).
- Activation selectivity ≠ causal selectivity: jb_fiction_specific features fire 73.6 % on fiction prompts but ablating them recovers 0 % of fiction-class compliance. This is a reusable methodological warning for the field's "find class-specific features" workflow.
- Per-position sign flip in cosine alignment: cos(r̂, r_jb) is +0.94 at pos=−2 but −0.84 at pos=−5 inside the JB prefix. The model "primes" along r̂ early and "commits" against r̂ at the decision token. Brief, single-paragraph; not load-bearing for the workshop submission but flagged for §5 follow-up.

#### § 5. Limitations (≈ 0.25 page)

- Single model (Gemma-3-4B-IT). Cross-model evidence on Qwen3 forthcoming (Ruqiya); flag in submission, fold into camera-ready if results land.
- 50 prompts. Wilson CIs disclosed; per-class numerator small (n=4 to 33) for some cells.
- `jb_completion` is not a real bypass on this dataset (n_baseline_comply = 0); excluded from the comply-weighted aggregate, listed for transparency.
- MLP-only attribution; attention heads frozen. The residual gap can plausibly hide there.

### 5. Figures (4 main, fits the 4-page constraint)

| # | Content | Source / status |
|---|---|---|
| **F1** | Schematic: bare vs ctrl_* vs jb_* prompt design + intervention vs ablation methodology side-by-side. | New, simple matplotlib + tikz/svg; ~1 hr. |
| **F2** | Per-condition refusal-direction projection at L15 pos=−2, with bare/ctrl/jb bars + Stage-06 ±1·‖r̂‖ reference lines + cosine annotations per class. | `figures/F7_refusal_direction_alignment.png` (already generated). |
| **F3** | The Pareto curve: x = log feature count (1, 5, 10, 20, 50, 100), y = JB recovery, with direction reference (100 %), random-6 control (9 %), and best subcircuit constructions overlaid. Wilson CIs as error bars. | `figures/F5_recovery_vs_features_pareto.png` (already generated). |
| **F4** | Construction-rule robustness: bar chart of all 16 (subcircuit × K/F) ablations + canonical sweep + top-N points, all renormed, all on the same vertical axis. Demonstrates the gap is not a one-off. | `figures/F6_construction_rule_robustness.png` (already generated). |

### 6. What we have vs what we still need

#### Already in hand (no new GPU)
- All Stage 06, Stage 08 Tier 1 + Tier 2, canonical sweep results (renormalized).
- Direction-alignment cosine + magnitude analysis (`scripts/analysis/refusal_direction_alignment.py`).
- All 4 figures generated.
- REPORT_run_20260430_023247.md sections 5.5, 9.7, 9.8, 9.9, 9.10 are cite-ready and have all the numbers.

#### To do for submission (writing only)
1. Draft § 1 Introduction (~½ day). Strict positioning against Ball 2024 + Wang 2025.
2. Draft § 2 Methods (~½ day). Lift from REPORT § 1 and § 8.1, § 9.1.
3. Draft § 3 (~1 day). Lift numbers from REPORT § 5.5, § 9.8, § 9.9, § 9.10. Ensure Wilson CIs everywhere.
4. Draft § 4 + § 5 (~½ day). Discussion is cheap; limitations should be honest.
5. Final figure polish (~½ day). F1 is the only new figure; F2/F3/F4 need only label/CI tweaks.
6. References pass: Arditi 2024 (NeurIPS), Ball 2024, Wang 2025, Lindsey et al. circuit-tracing, Marks/Belrose diff-in-means, Templeton et al. (gemma-scope transcoders if applicable).

**Total writing budget: ~3 person-days. Submission deadline 4 days out (Fri 5/8).**

#### To add for camera-ready (post-acceptance, low risk)
- Cross-model bar (Qwen3 direction-vs-ablation gap; Ruqiya). One row in F4.
- Possibly one of the Option-2 experiments if it lands by then.

### 7. Venue fit

ICML mechanistic interpretability workshop is the right home: 4 pages, technique-focused, allows negative results, accepts replication-with-extension. Our central result is "directional intervention Pareto-dominates sparse feature ablation, even with optimal feature selection per prompt" — that is precisely the kind of falsifiable mechanistic claim the workshop is designed for. The workshop also values explicit positioning against prior work (Ball 2024 will be in the program almost certainly).

NeurIPS or EMNLP **main** is *not* the right home for this 4-page version — main-conference reviewers will (correctly) ask the deeper "why" questions we are deferring to v2.

### 8. Risks and mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Reviewer cites Ball 2024 as "this is done" | High | § 1 leads with the explicit framing of "Ball 2024 has the directional finding; we have the comparison to feature ablation, which they don't." Quote their § 6 verbatim. The cosine alignment in § 3.2 is supporting evidence, not the headline. |
| Reviewer cites Wang 2025 as "this is done" | Medium | Wang is multilingual and uses different "JB vector" definition (bypassed_harmful vs refused_harmful, not bare vs jb_class). Different question. |
| Reviewer asks: why doesn't ablation work? | High (workshop-level acceptable) | We acknowledge in § 4 + flag the candidates (attention paths, transcoder errors, position interactions). Each is a future-work item; the v2 paper plans them. The 4-page workshop submission is not obligated to resolve this. |
| Reviewer asks: single model is a weak base | Medium | Qwen3 cross-model is in pipeline; promise in submission, deliver in camera-ready. Cite Wang 2025's Gemma-2-9B and our 4B-IT result as already-multi-model in the family. |
| Stage 08 baselines drift from Stage 06 | Resolved | All Tier 1 / Tier 2 / canonical-sweep results explicitly renormalized to Stage 06's `causal_results.json`. § 9.7.5 + § 9.8.1 of report. |
| `jb_completion` has 0 comply baseline | Mitigated | Excluded from comply-weighted aggregate; transparently disclosed in § 5 limitations. |
| Pareto plateau is just a saturation artifact | Defensible | Wilson CI on top_50 [25.7, 45.2] does not include 100 %; ctrl_break / bare_break are flat across N (4–12 %) which rules out "more features = more side effects" as a plateau cause. |

### 9. Submission checklist

- [ ] § 1 Introduction draft (Mahmoud, by 5/5 EOD)
- [ ] § 2 Methods draft (Mahmoud, by 5/5 EOD)
- [ ] § 3 Results draft, all numbers + Wilson CIs (Mahmoud, by 5/6 EOD)
- [ ] § 4 + § 5 draft (Mahmoud, by 5/7 noon)
- [ ] F1 schematic figure (Mahmoud, by 5/7 noon)
- [ ] F2/F3/F4 polish — label fonts, CI bars, paper aspect ratios (Mahmoud, by 5/7 EOD)
- [ ] Internal review by Georg + Ruqiya (5/7 EOD → 5/8 noon)
- [ ] Final pass + LaTeX template (5/8 afternoon)
- [ ] Submit (5/8 EOD)
- [ ] Add Qwen3 row to F4 if Ruqiya's Stage 06 + Stage 08 land before submission; otherwise camera-ready.

---

*Last updated 2026-05-04 to reflect the Option-1 (workshop) paper plan after weekly check-in with Georg. Five-outline brainstorm version is preserved in git history.*
