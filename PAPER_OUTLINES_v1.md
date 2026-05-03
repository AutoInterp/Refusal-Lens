# Refusal-Lens — Five Paper Outlines (v1)

**Purpose**: five parallel outlines, one per candidate thesis, for team review. Same skeleton in each so they're directly comparable. Pick one (or a hybrid) before writing.

**Outlines A–D require new experiments** (additional models, replication runs, layer sweeps, attention attribution, top-N sweeps, etc.). **Outline E requires no new compute** and is positioned as a *framework paper*: a unifying account of refusal as a low-rank gate over a distributed expression, evidenced quantitatively by what we already have. The novelty is the framework, not the measurements.

**Source data**: `REPORT_run_20260430_023247.md` (full per-stage report + Section 9.7 canonical sweep). Numbers cited below come from there.

**Target venues**:
- **ICML 2026 Mech Interp Workshop** (~4 pages, mechanism-focused, deadline imminent).
- **NeurIPS 2026 main track** (8 pages + appendix, broader story + generalization, deadline ~3 weeks).

A single paper backbone should serve both: ICML = abridged mechanism-only cut; NeurIPS = full story + replication + generalization. Each outline below specifies the cut for each venue.

---

## Outline A — *Methodology-led*: per-prompt subcircuit construction

### A.1 Title (drafts)

- *NeurIPS*: **"Per-prompt subcircuit construction reveals compact causal mechanisms hidden inside corpus-aggregate transcoder graphs"**
- *ICML workshop*: **"Six features beat eighty-eight: per-prompt aggregation isolates causal subcircuits"**

### A.2 One-sentence thesis

The standard practice of aggregating transcoder attributions at the corpus level dilutes causal mechanism with prompt-specific noise; aggregating instead by **per-prompt frequency** isolates a small feature set whose causal effect is dramatically larger and cleaner — demonstrated on jailbreak refusal in Gemma-3-4B-IT, where 6 per-prompt features outperform 88 corpus-union features on every causal axis simultaneously.

### A.3 Headline numbers (already in hand)

- 88-feature corpus-union canonical: **9.9 %** JB recovery, 11.6 % ctrl break, 17.8 % bare break.
- **6-feature per-prompt canonical (k100_f20)**: **23.7 %** JB recovery, **1.3 %** ctrl break, 8.9 % bare break.
- 1-feature canonical (k50_f50): 6.9 % JB recovery (a single feature still flips ~7 % of JB compliances).
- Coverage diagnostic confirms mediator: per-prompt construction concentrates jb_fiction coverage from 15 % → 87.7 %.
- Methodology generalizes: any (target metric, condition family, transcoded model) tuple.

### A.4 Section structure (NeurIPS 8-page)

1. **Introduction** — corpus-aggregate practice in transcoder/SAE attribution (cite Lindsey 2024, Anthropic circuit-tracer); the dilution hypothesis; why it matters for safety-mechanism work.
2. **Background** — circuit-tracer / CLT framework, attribution graph aggregation conventions, refusal-direction interpretability literature (Arditi, Tejas).
3. **Method** — formal definition of per-prompt top-K + ≥F-frequency aggregation; comparison to corpus-union and to direct linearization. Algorithm box.
4. **Empirical setup** — Refusal-Lens dataset (50 × 11), Stage 02–04 pipeline, Stage 07 sweep configurations.
5. **Headline results** — cross-config table; per-class decomposition; coverage-as-mediator analysis.
6. **Mechanism interpretability** — single-feature spotlight (L13:F427); why per-prompt features cluster at L13–L15 (Tejas's actionable layer band); helpful-template-feature finding (L11:F99 *not* a refusal feature, in the legacy 88).
7. **Generalization study** *(needs new experiment)* — apply per-prompt-vs-corpus-union to one additional model (Gemma-2-9B-IT or Llama-3-8B-Instruct) and one additional task family (toxicity, policy compliance, etc.). Show the 2.4× recovery / 9× specificity gap holds.
8. **Limitations & related work** — bf16 baseline drift, 50-prompt scale, attention-frozen attribution.
9. **Conclusion**.

**ICML workshop cut**: §1 + §3 + §5 + §6 + half of §7 (one extra setting). Drop §2, §4, most of §8. ~4 pages.

### A.5 Figures (5 main)

- F1 — schematic: corpus-union vs per-prompt aggregation diagram.
- F2 — cross-config bar chart: JB recovery, ctrl break, bare break × {legacy, k100_f20, k50_f50}.
- F3 — coverage-as-mediator scatter: x = mean coverage, y = recovery rate, points = (config × class).
- F4 — single-feature spotlight: L13:F427 trigger contexts + ablation effect.
- F5 — generalization: same axes as F2, on second model + second task.

### A.6 What we have / what we still need

| Status | Item |
|---|---|
| ✅ have | Refusal-Lens canonical sweep (3 configs × 1 ablation). |
| ✅ have | Coverage diagnostic, single-feature analysis, layer concentration story. |
| ⚠️ need | **Replication on `run_20260422_015552`** — same pipeline, different prompts. Confirms the methodology effect is not a one-run fluke. ~7 h on the 4090. |
| ⚠️ need | **One additional model** for generalization. Gemma-2-9B-IT has the same Gemma-Scope transcoders available — minimum-friction choice. ~2 days incl. attribution. |
| ⚠️ need | **One additional task** (e.g. toxicity, sycophancy) on the same Gemma-3-4B-IT setup. ~1 day. |
| ⚠️ need | **Ablation study on K, F** across {(K, F): K∈{20, 50, 100, 200}, F∈{0.1, 0.2, 0.3, 0.5}}. Identifies the recovery-vs-specificity Pareto frontier. ~1–2 days. |
| ❌ optional | Theoretical framing — sparse-recovery / dictionary-learning analogy. |

### A.7 Venue fit

- **NeurIPS main**: strong story IF replication + generalization land. The "6 vs 88" is a clean, quotable finding that travels.
- **ICML workshop**: cleanly fits 4 pages with the headline + single-feature spotlight; replication is nice-to-have.

### A.8 Risks

- "We changed two things" critique: per-prompt method differs from corpus-union both in *aggregation* and in *feature filter strictness*. Need an ablation showing it's the per-prompt choice, not the size, that matters (e.g. take 6 random features from the 88 corpus-union set and show they recover < 23.7 %).
- Reviewers may demand a theoretical analysis of WHY per-prompt aggregation works.
- The 88-feature legacy is constructed differently from the 6-feature config (top-K=50 vs top-K=100); cleaner ablation needed.

---

## Outline B — *Mechanism-led*: L15 is the refusal axis, one feature does most of the work

### B.1 Title (drafts)

- *NeurIPS*: **"The refusal axis of Gemma-3-4B-IT lives at layer 15: bidirectional control with one direction and seven features"**
- *ICML workshop*: **"L15 is causal, L32 is decorative: the actionable refusal axis in Gemma-3-4B-IT"**

### B.2 One-sentence thesis

In Gemma-3-4B-IT, refusal behavior is mediated by a single linear axis at layer 15 with full bidirectional control (100 % flip pro→anti, 98 % flip anti→pro), whose attribution graph is dominated by ≤ 7 transcoder features clustered at L13–L15, of which a single feature (L13:F427) accounts for ~7 % of jailbreak-comply→refuse flips on its own.

### B.3 Headline numbers (already in hand)

- L15 *r* additive intervention: **89/89 = 100 %** flip JB→REFUSE.
- L15 *r* subtractive intervention: **49/50 = 98 %** flip bare→COMPLY.
- L15 *r* additive on benign: **10/10 = 100 %** force-refuse.
- L32 separation = 20,873 (stat-best), but L32 *r* intervention = 0/10 ctrl flip, 0/32 JB flip (Tejas, prior work).
- 6-feature per-prompt canonical recovers 23.7 % JB.
- L13:F427 alone recovers 6.9 %; activates only on explicitly harmful prompts (act-freq 0.01 %).
- Layer story: cumulative net contribution to *r* is −20,711 over L0–L11, −32,786 by L19, +666 by L32, +32,125 from L33 alone.

### B.4 Section structure (NeurIPS 8-page)

1. **Introduction** — refusal as a target for interpretability; the gap between "the direction exists" (Arditi) and "what features compose it"; the puzzle of L32 vs L15 (separation ≠ causation).
2. **Background** — direction-finding methods, transcoders/CLT, prior refusal-direction work.
3. **Methods** — direction extraction (Arditi), bidirectional intervention protocol, attribution graph construction, per-prompt subcircuit method (cited from supplementary, since it's not the main claim here).
4. **Bidirectional axis verification** — three-way symmetry table; comparison to L32 (the layer the literature would expect).
5. **Layer-wise circuit decomposition** — cumulative net contribution figure; L0–L11 negative accumulator → L33 override pattern.
6. **Sparse feature decomposition at L13–L15** — per-prompt 6-feature canonical; single-feature L13:F427 deep dive (trigger contexts, activation pattern, response signature).
7. **Distributed redundancy** *(short)* — gap between direction (100 %) and best feature subset (23.7 %) frames Outline C as future work.
8. **Generalization** *(needs new experiment)* — same protocol on Gemma-2-9B-IT or Llama-3-8B-Instruct; show L_n exists, single feature exists, layer-cumulative pattern recurs.
9. **Limitations & related work**.
10. **Conclusion**.

**ICML workshop cut**: §1 + §4 + §5 + §6. ~4 pages, depth-rich.

### B.5 Figures (5 main)

- F1 — bidirectional flip diagram: pro / anti / benign symmetry table + sample generations.
- F2 — cumulative-contribution-to-*r* by layer (the L0→L11→L33 figure already in `per_layer_contribution.png`).
- F3 — L15 intervention vs L32 intervention: bar comparison (separation vs flip rate).
- F4 — single-feature spotlight: L13:F427 triggers + per-class flip rate.
- F5 — generalization: same plots on second model.

### B.6 What we have / what we still need

| Status | Item |
|---|---|
| ✅ have | Bidirectional symmetry, layer-cumulative, single-feature analysis, K/F sweep. |
| ⚠️ need | **Generalization to a second model** (Gemma-2-9B-IT highest-leverage). ~2 days incl. attribution. |
| ⚠️ need | **Layer sweep** of *r*-intervention (L5, L10, L15, L20, L25, L32) to formalize "L15 is causal" beyond Tejas's pilot. ~1 day. |
| ⚠️ need | **Mechanistic interpretation of L13:F427** beyond trigger contexts: connect to attention heads (which heads gate it?), look at where it routes downstream. Could be Stage 09 attention-head attribution. ~3–5 days. |
| ⚠️ need | **Ablation of L13:F427 on the bidirectional axis test** — does ablating this single feature also reduce bare-break under subtractive intervention? Tests whether the feature is on the *r*-axis or auxiliary. |
| ❌ optional | Concept-level interpretation via Neuronpedia / Gemma-Scope dashboard cross-reference. |

### B.7 Venue fit

- **NeurIPS main**: strong if generalization to a second model lands and the L13:F427 mechanistic story holds.
- **ICML workshop**: ideal home — mech interp workshop reviewers will reward the L15-vs-L32 puzzle and the single-feature interpretability story even without generalization.

### B.8 Risks

- L32 statistical-vs-causal puzzle is not new (Tejas pilot, Geiger 2025-style critique); novelty is in the bidirectional symmetry + sparse-feature decomposition combination.
- One feature's interpretation is anecdotal — reviewers will want quantitative interpretability metrics (e.g. activation-fidelity score across a labeled harm corpus).
- "L15 is THE axis" is model-specific until generalized; over-claiming risks rebuttal.

---

## Outline C — *Negative-result-led*: distributed redundancy, ablations don't suffice

### C.1 Title (drafts)

- *NeurIPS*: **"You can't ablate refusal: a 100 % vs 24 % gap between direction interventions and sparse-feature ablations in Gemma-3-4B-IT"**
- *ICML workshop*: **"Distributed redundancy: refusal in Gemma-3-4B-IT is not reducible to sparse subcircuits"**

### C.2 One-sentence thesis

In Gemma-3-4B-IT, a single linear-direction intervention at layer 15 flips 100 % of jailbreak compliances back to refusal, while the best transcoder-feature ablation we can construct flips only 24 %; this 4× gap is robust across construction rules and reveals that refusal is encoded with substantial **distributed redundancy** — implying that mechanistic-interpretability claims of the form "feature set X causes refusal" are systematically incomplete and should be paired with direction-level potency measurements.

### C.3 Headline numbers (already in hand)

- Direction intervention (L15 *r*): 100 % JB→REFUSE flip.
- Best subcircuit ablation (per-prompt 6-feature): 23.7 % JB recovery.
- Best per-class subcircuit (jb_fiction k100_f20): 45 % fiction recovery.
- 88-feature corpus-union canonical: 9.9 % JB recovery — *more features, worse recovery*: redundancy-in-the-graph evidence.
- Linearization identity holds (Σ edges + baseline = direct dot, error 0.08–0.36 %): there is no missing graph signal; the redundancy is an irreducible feature of the model.

### C.4 Section structure (NeurIPS 8-page)

1. **Introduction** — the rise of feature-ablation claims in mech interp; the implicit assumption of *sufficiency* of sparse subcircuits; this paper as a counterexample.
2. **Background** — direction-finding (Arditi), transcoders/CLT, attribution, ablation as a test of causal importance.
3. **Setup** — Refusal-Lens dataset, model, pipeline overview.
4. **The direction recovers everything** — bidirectional symmetry table.
5. **Ablations recover a fraction** — 5-ablation table + canonical sweep table; per-class decomposition; the "more features, worse recovery" inversion (88f vs 6f).
6. **Quantifying the gap** — direction-vs-ablation potency ratio; per-class breakdown. Hypothesis: distributed redundancy ⇒ feature ablation is necessarily lossy.
7. **Mechanism of redundancy** *(speculative section, marked as such)* — partial recovery analysis: which prompts are NOT flipped by ablation that ARE by direction? Are they specific JB classes? Specific harm types? Position-dependent?
8. **Implications for the field** — methodological recommendation: every "feature X causes Y" claim should be paired with a direction-or-baseline potency measurement. Critique of feature-only mechanistic claims.
9. **Limitations**.
10. **Conclusion**.

**ICML workshop cut**: §1 + §4 + §5 + §6 + the "implications" punchline of §8. ~4 pages.

### C.5 Figures (5 main)

- F1 — headline bar comparison: direction (100 %) vs best subcircuit (24 %) vs corpus-union (10 %) — same axes, dramatic visual.
- F2 — per-class direction-vs-ablation gap; shows which classes are hardest to ablate (fiction = 45 %, cog_reframe = 12 %).
- F3 — "more features, worse recovery" inversion — feature count vs recovery, with the 88f point below the 6f point.
- F4 — distributed redundancy schematic: direction = sum over many features, each individually weak.
- F5 — the prompts ablation misses: subset analysis.

### C.6 What we have / what we still need

| Status | Item |
|---|---|
| ✅ have | Direction intervention numbers (Stage 06). |
| ✅ have | All 4 subcircuit ablation runs (orig 5 + canonical 3-config sweep). |
| ✅ have | Linearization identity (Stage 03) — proves no missing graph signal. |
| ⚠️ need | **The "prompts that fail to flip under ablation but succeed under direction" set** — qualitative analysis on these 76 % of prompts. ~0.5 day. |
| ⚠️ need | **Replication on a second model** to claim "distributed redundancy" generalizes. ~2 days. |
| ⚠️ need | **Top-N feature ablation sweep** (N = 1, 2, 5, 10, 20, 50, 100) on the same per-prompt construction — show recovery curve plateaus *below* direction's 100 %. Strongest visual evidence of redundancy. ~1–2 days. |
| ⚠️ need | **Random-feature control**: ablating N random transcoder features should recover ≪ 24 %. Currently implicit via low coverage in legacy; need an explicit control. ~0.5 day. |

### C.7 Venue fit

- **NeurIPS main**: contrarian framing — high reviewer variance. Either championed ("important methodological corrective") or dismissed ("they just didn't ablate enough features"). Needs strong control experiments to win.
- **ICML workshop**: workshop reviewers may welcome a negative-result paper if framed as a methodological warning. The plateau-below-100 figure (C.6) would be especially compelling.

### C.8 Risks

- "You didn't ablate enough features" — requires the top-N sweep showing plateau.
- "Distributed redundancy" is not a unique inference — could also be additivity-of-features, or measurement-noise-in-ablation. Need to distinguish.
- Negative-result papers face higher acceptance bar at NeurIPS main; need a *positive* methodological recommendation (the direction-pairing prescription) to balance.

---

## Outline D — *Combined*: the comprehensive Refusal-Lens paper

### D.1 Title (drafts)

- *NeurIPS*: **"Refusal-Lens: a method, a mechanism, and a measurement for jailbreak-induced refusal failure in instruction-tuned LLMs"**
- *ICML workshop*: **"Refusal-Lens: per-prompt subcircuits and the L15 axis in Gemma-3-4B-IT"**

### D.2 One-sentence thesis

We present **Refusal-Lens**, an end-to-end mechanistic-interpretability pipeline for jailbreak-induced refusal failure, and use it to (i) introduce per-prompt subcircuit construction as a method, (ii) characterize the bidirectional L15 refusal axis with single-feature attribution in Gemma-3-4B-IT, and (iii) quantify the gap between direction-level and feature-level interventions — three contributions woven into one mechanism story.

### D.3 Headline numbers (everything in A + B + C combined)

- **Method**: 6-feature per-prompt canonical vs 88-feature corpus-union canonical (23.7 % vs 9.9 %, A.3).
- **Mechanism**: L15 bidirectional symmetry (100 / 98 / 100, B.3); single-feature L13:F427 = 6.9 % JB recovery alone.
- **Measurement**: direction-vs-ablation potency gap (100 % vs 24 %, C.3).

### D.4 Section structure (NeurIPS 8-page — TIGHT)

1. **Introduction** — refusal as a unified target; three questions (HOW to identify subcircuits, WHAT mechanism is in there, HOW WELL can we ablate it); the Refusal-Lens pipeline as the answer.
2. **Background & related work** — Arditi direction finding, Lindsey 2024 circuit-tracing, transcoders/CLT, prior refusal mech-interp.
3. **The Refusal-Lens pipeline** — Stages 01–08, with emphasis on the bug-fix story (basis-mismatch, measurement_hook) since that's the engineering-correctness throughline.
4. **Method: per-prompt subcircuit construction** — formal definition + Refusal-Lens 6 vs 88 result.
5. **Mechanism: bidirectional L15 axis & single-feature attribution** — symmetry table + L13:F427 deep dive.
6. **Measurement: direction-vs-ablation potency** — the 100 / 24 gap + redundancy.
7. **Generalization to one additional model** — 2-page version of the second-model study.
8. **Limitations & conclusion**.

**ICML workshop cut**: §3 (compressed) + §4 + §5. Drop §6/§7. ~4 pages, mechanism-only.

### D.5 Figures (5 main, each one carrying a contribution)

- F1 — pipeline diagram + bug-fix throughline.
- F2 — per-prompt vs corpus-union comparison (from Outline A).
- F3 — bidirectional + layer-cumulative + L13:F427 spotlight composite (from B).
- F4 — direction vs ablation potency gap (from C).
- F5 — generalization snapshot.

### D.6 What we have / what we still need

| Status | Item |
|---|---|
| ✅ have | Everything in A.6 + B.6 + C.6. |
| ⚠️ need | **Generalization to a second model** (still the long pole). ~2 days. |
| ⚠️ need | **Replication on `run_20260422_015552`** for the per-prompt method. ~7 h. |
| ⚠️ need | **Top-N feature ablation sweep + random-feature control** for the redundancy claim. ~2 days. |
| ⚠️ need | **Tight writing pass** — three contributions in 8 pages is the hardest writing job. Likely sacrifice some bug-fix throughline + some statistical rigor (Stage 02b). |
| ⚠️ need | **Single coherent figure of the pipeline + three contributions** — F1 alone is high-stakes. |

### D.7 Venue fit

- **NeurIPS main**: most ambitious, highest upside if it lands; risk is reviewer "this should be three papers". The "Refusal-Lens" framing as a *system contribution* (not just three separate findings) is the rhetorical glue.
- **ICML workshop**: works as a focused 4-page abridgment (mechanism + method, drop measurement). The ICML version becomes a "preview" of the NeurIPS submission.

### D.8 Risks

- Reviewer dispersion: each of the three contributions can be the target of an attack, and the surface area is large. Each contribution must individually defensible at the section-level depth allowed.
- 8 pages is tight for three contributions — each gets ~2 pages, which is below typical NeurIPS section depth. Either sacrifice depth or expand to a 9-page submission with the extra page from "Refusal-Lens system" framing.
- The "system contribution" framing risks being read as soft (no algorithmic novelty in the pipeline-as-a-whole). Anchoring with §4 (per-prompt method) as the technical novelty mitigates.

---

## Outline E — *No-new-experiments*: refusal as a low-rank gate over a distributed expression

### E.1 Title (drafts)

- *NeurIPS*: **"Refusal is gated, not localized: a low-rank gate and a distributed expression in instruction-tuned LLMs"**
- *ICML workshop*: **"The refusal gate: why a single direction recovers what 88 features cannot in Gemma-3-4B-IT"**

### E.2 One-sentence thesis

Refusal behavior in instruction-tuned LLMs decomposes algebraically into **(i) a low-rank gating signal**, carried on a single linear direction at a specific mid-layer and admitting full bidirectional control with one vector, and **(ii) a high-rank, distributed downstream expression**, which is constructed at inference time from the model's general response-generation machinery and resists localization to a sparse feature subset; jailbreaks succeed by **suppressing the gate**, not by routing around the refusal mechanism — a unifying account that explains, in a single framework, why the refusal direction exists as a clean linear axis, why a small per-prompt feature set partially recovers the behavior, and why no sparse subcircuit fully recovers it.

### E.3 What's novel about this thesis (vs. existing literature)

- **Arditi et al. (2024)** showed the refusal direction exists and that a single vector controls refusal, but not *why* it admits a 1-D linear control while the underlying behavior is distributed. We close that loop: the direction *is* the gate, distinguished from the expression.
- **Lindsey et al. (2024) / circuit-tracer / Anthropic transcoder work** decompose model behaviors into sparse subcircuits but assume the decomposition is the mechanism. We show that for refusal, the sparse decomposition is *partial by construction* — it captures gate activity but not the distributed expression — and we quantify the gap (4× potency).
- **Tejas's pilot finding** that L15 is causal where L32 is statistically dominant becomes a *prediction* of the framework: gating signals are localized; statistical separation in the residual stream is the downstream expression's signature, not the gate's.
- **The "two mechanisms" account** (dampening + tug-of-war, Tejas 2026-04) becomes a special case: dampening = suppressing the gate; tug-of-war = recruiting opposing expression features. The framework subsumes both.
- **Predicted, generalizable, falsifiable**: any instruction-tuned LLM with a refusal direction should exhibit the same triple — bidirectional 1-D gate, sparse-but-prompt-conditional expression hits, and a direction-vs-ablation potency gap. We don't show generalization in this paper, but the framework makes it a one-experiment falsification test.

### E.4 Headline numbers (all already in hand — zero new experiments)

The three pillars of the gate-vs-expression decomposition, each independently quantified:

**Pillar 1 — the gate is 1-D and bidirectional**:
- L15 unnormalized-*r* additive intervention: **89/89 = 100 %** flip JB-comply → REFUSE.
- L15 *r* subtractive intervention: **49/50 = 98 %** flip bare-refuse → COMPLY.
- L15 *r* additive on benign: **10/10 = 100 %** force-refuse.
- Cross-layer comparison: L32 (separation = 20,873, the statistically-dominant layer) yields **0/10** ctrl flip and **0/32** JB flip — separation is the expression's footprint, not the gate.

**Pillar 2 — the expression is sparse but prompt-conditional**:
- 6-feature per-prompt canonical (k100_f20): **23.7 %** JB recovery, 1.3 % ctrl break.
- Same construction at the corpus level (88 features, top-K=50 union): **9.9 %** recovery, 11.6 % ctrl break — *more features, worse recovery*: the inversion is the algebraic signature of an expression that *cannot* be aggregated across prompts.
- Per-prompt coverage of the 6 canonical features ranges 1 % (bare) to 87.7 % (jb_fiction): the expression's feature identity is prompt-dependent, but its *count* (~6) and *layer band* (L13–L15) are stable.
- Single-feature L13:F427 alone recovers 6.9 %; activates only on explicitly harmful prompts (act-freq 0.01 %): a candidate for a *gate-detector* feature whose role is binary harm-classification rather than refusal expression.

**Pillar 3 — the expression resists localization**:
- Direction-level intervention recovers 100 % (Pillar 1).
- Best sparse-feature ablation we can construct recovers 23.7 % (Pillar 2).
- Recovery-vs-feature-count Pareto curve from already-computed Stage 08 ablations (orig 5 + canonical 3) is **non-monotone**: 1f → 6f → 26f rises (gate features being included), 26f → 88f *falls* (corpus-union dilution). Plateau lies far below the direction's 100 %.
- Linearization identity holds (Stage 03, error 0.08–0.36 %): the gap is not a measurement artifact; it is a fundamental algebraic property of the model's representation.

**Pillar 4 (mechanistic implication, supports thesis)** — refusal under JB co-opts the helpful-response machinery:
- Three of the 6 per-prompt canonical features are *helpful-template features* (e.g. `L15:F442` fires on `"Okay, let's plan/brainstorm"`): they participate in the refusal expression but are *not* refusal-specific. The expression is constructed by re-routing the model's general response-generation machinery, gated by the harm signal.
- This explains why ablation underperforms direction: ablating an expression feature also ablates its helpful-response role, but the gate signal continues to fire on the (now-suppressed) downstream pathway, producing partially-coherent compliance instead of refusal.

### E.5 Section structure (NeurIPS 8-page)

1. **Introduction** — three observations from the literature in apparent tension: (i) a single direction controls refusal (Arditi); (ii) sparse subcircuits partially explain refusal (transcoder work); (iii) ablations of those subcircuits underperform direction interventions (this paper). We propose a unifying framework: refusal is **gated**, not localized.
2. **Background & related work** — direction-finding (Arditi 2024), transcoder/CLT attribution (Lindsey 2024, circuit-tracer), prior refusal mech-interp, the dampening-vs-tug-of-war characterization (Tejas).
3. **Methods** — Refusal-Lens: a corrected-basis attribution pipeline (§ A.1, methodological notes on three measurement-basis bugs we encountered and fixed; full diffs in supplementary), per-prompt subcircuit construction, controlled 50 × 11 jailbreak benchmark.
4. **Pillar 1: the gate is low-rank and bidirectional** — § 8 of the report. Symmetry table, sample generations, L15-vs-L32 separation-vs-causation analysis.
5. **Pillar 2: the expression is sparse but prompt-conditional** — § 7 + § 9.7 of the report. Per-prompt vs corpus-union, coverage-as-mediator, layer band concentration.
6. **Pillar 3: the expression resists localization** — § 9.5 + § 9.7.2 of the report. Recovery-vs-features Pareto, the non-monotonicity at 26→88, the irreducibility argument backed by the linearization identity.
7. **Mechanistic implication: the expression is borrowed from the helpful-response machinery** — § 9.7.6 + Appendix C. L13:F427 single-feature spotlight, helpful-template features in the canonical pro-refusal set, qualitative case studies of recruitment.
8. **Discussion** — why the framework predicts each of (Arditi, Tejas, our work)'s findings; how to falsify it on a second model with a single 4-experiment protocol; implications for safety claims (feature-causes-refusal claims must be paired with direction-level potency tests; ablation-only safety verification undercounts the model's refusal capacity).
9. **Limitations** — single model, single dataset, attention-frozen attribution, bf16-baseline drift caveat, 50-prompt scale; the bug-fix appendix as a reproducibility statement.
10. **Conclusion + release** — Refusal-Lens pipeline, controlled benchmark, and the 1,100 corrected attribution graphs released publicly so the framework's predictions can be tested on any model with available transcoders.

**ICML workshop cut**: §1 (preview) + §3 (1-paragraph methods) + §4 + §5 + §6 + §8 (1 paragraph). Drop §2, §7, §9. ~4 pages, focused on the three pillars.

### E.6 Figures (5 main, all data-already-exists)

- **F1 — the framework**: a schematic split into (a) gate (low-rank, mid-layer, ±1D) and (b) expression (high-rank, distributed across L13–L33, recruited from helpful-response machinery). The conceptual figure that anchors the paper.
- **F2 — Pillar 1**: bidirectional symmetry table + sample generations + the L15-vs-L32 separation-vs-causation bar comparison (the "separation is the expression's footprint" finding).
- **F3 — Pillar 2**: per-prompt vs corpus-union recovery (88 / 6 / 1 features); coverage-as-mediator scatter (x = per-prompt coverage, y = recovery, colored by config); the *non-monotonicity* line clearly visible.
- **F4 — Pillar 3**: recovery-vs-features Pareto curve (x = log n_features, y = JB recovery), with the direction-intervention 100 % horizontal line and the plateau-far-below visible. **The figure that shows why feature ablation can't replace direction work.**
- **F5 — Pillar 4 (mechanism)**: L13:F427 trigger-context table; per-class flip rate from L13:F427 alone vs the 6-feature canonical (the gate vs expression decomposition seen at the feature level); the cumulative-contribution-to-*r* by layer figure (L0→L11→L33) showing where gate features concentrate.

### E.7 What we have / what we still need

| Status | Item |
|---|---|
| ✅ have | All four pillars' numerical evidence from `run_20260430_023247` + canonical sweep + `run_20260422_015552`. |
| ✅ have | All five figures' data; only plotting code needs writing. |
| ✅ have | Bug-fix history, located in commit chain (`7c6cfa4` and `refusal-lens-multi-position-fix` submodule branches) — appendix material, not centerpiece. |
| ✅ have | Full corrected pipeline runs end-to-end on a single H100 in ~24 h. |
| ⚠️ need (writing only) | The five figures rendered to publication quality. ~1 day. |
| ⚠️ need (writing only) | A 1-page methods appendix on the three measurement-basis fixes (the bug story) — supports the rigor claim in §3 but is **not the paper's headline**. |
| ⚠️ need (writing only) | Public release: pipeline, benchmark, 1,100 corrected attribution graphs. The release is positioned as enabling the framework's falsification, not as the contribution itself. |
| ⚠️ need (writing only) | The framework section (§1 + §8) — the conceptual heavy lift; this is where the paper rises or falls. |
| ❌ optional | Two-run consistency analysis on bug-immune Stage 06 (Stage 06 numbers replicate ±2 % across runs; Stage 02-derived numbers shift) — useful as a sanity check in the limitations section. |

**Compute required for the paper itself: zero new GPU runs.** The novelty is the *framework*, not new measurements.

### E.8 Venue fit

- **NeurIPS main**: a unifying framework that subsumes prior refusal-mech-interp findings (Arditi, Lindsey, Tejas) and predicts the direction-vs-ablation gap is a strong NeurIPS narrative. The single-model evidence base is a known weakness, but it is mitigated by (a) the framework being mechanistically falsifiable in a one-day follow-up experiment, (b) the public release lowering the cost of replication for the community, (c) the framework explaining empirical findings that prior accounts could not. Reviewer split: framework-friendly reviewers will champion; one-model-skeptics will push for replication. The release shifts the burden.
- **ICML mech interp workshop**: ideal — the workshop has historically rewarded conceptual frameworks over empirical breadth, and the gate-vs-expression decomposition is the kind of mechanistic theorizing the workshop is aimed at.

### E.9 Risks

1. **"One model" critique**. Mitigation: explicitly frame the paper as proposing a *predicted* framework with quantitative falsification protocol, evidenced on Gemma-3-4B-IT. The release lowers the cost of replication; we provide a 4-experiment falsification recipe in §8 so a reader can decisively confirm or deny the framework on any model.
2. **"The gate-vs-expression decomposition is just renaming what's already known"**. Mitigation: § E.3 explicitly distinguishes us from Arditi (only showed the gate, not its decomposition from expression), Lindsey (assumed sparse decomposition is the mechanism), and Tejas (showed L15 is causal but not why). Our novel contribution is the algebraic separation and its empirical signatures (the 4× potency gap, the non-monotonicity, the helpful-template-feature recruitment).
3. **The framework requires the reader to take "compositional gating" as more than a metaphor**. Mitigation: ground each pillar in a measurable quantity. Pillar 1 = bidirectional flip rate. Pillar 2 = per-prompt vs corpus recovery delta. Pillar 3 = direction/ablation potency ratio. Pillar 4 = single-feature ablation rate vs canonical-set rate. None of these are metaphorical.
4. **"Why is the recovery curve flat below 100 %? Maybe you just need more features."** Mitigation: the 26 → 88 *non-monotonicity* answers this directly — adding more features makes recovery *worse*, which is incompatible with the "more features ⇒ closer to direction" hypothesis but consistent with "expression is constructed compositionally from helpful-response machinery, which adding more random helpful features only contaminates further". The non-monotonicity is the framework's strongest single piece of evidence.
5. **Bug-fix appendix risks looking defensive**. Mitigation: keep it short (1 page), frame as "to enable replication on a corrected basis we share these fixes", and put it post-conclusion in the supplementary. The headline contribution is the framework, not the bug fixes.

### E.10 Why E is distinct from A–D

- **A** is purely methodological ("per-prompt construction works"); E uses A's finding as Pillar 2 evidence for a larger framework claim about model behavior.
- **B** is mechanistically narrow ("L15 is the axis"); E generalizes it as Pillar 1 of a framework that distinguishes the axis from the rest.
- **C** is a negative result ("ablations don't suffice"); E uses C's gap as Pillar 3 evidence and provides a *positive* explanation (the gap is the algebraic signature of compositional gating, not an artifact of insufficient features).
- **D** is a system-and-three-findings paper; E is a *framework* paper with three findings as evidence and one mechanistic spotlight, which is a more standard NeurIPS rhetorical posture.

The narrative arc:

> "We propose that refusal in instruction-tuned LLMs is implemented as a low-rank gating signal multiplexing into a high-rank, distributed expression that is constructed at inference time from the model's general response-generation machinery. This single framework predicts, in a unified way, the bidirectional 1-D refusal direction (Arditi), the partial-recovery sparse subcircuits (Lindsey), and the direction-vs-ablation potency gap (this paper). We characterize all three quantitatively in Gemma-3-4B-IT, propose a falsification protocol for any other model, and release the pipeline, benchmark, and 1,100 corrected attribution graphs to enable that replication."

---

## Cross-outline summary table

| Outline | Anchor finding | Lift required | NeurIPS upside | ICML fit | Total new compute |
|---|---|---|---|---|---|
| **A — Methodology** | 6 vs 88 feature recovery gap | replication + 1 new model + 1 new task | high (clean, quotable) | strong | ~4 days |
| **B — Mechanism** | L15 bidirectional + L13:F427 | 1 new model + layer sweep + attention attribution | medium-high | ideal | ~6 days |
| **C — Negative result** | direction vs ablation 100/24 gap | top-N sweep + random-feature ctrl + 1 new model | medium (high variance) | strong-if-framed | ~3 days |
| **D — Combined** | all three | all of the above (less depth on each) | high IF framing holds | preview-cut | ~7 days |
| **E — Gate vs expression framework** | refusal = low-rank gate × distributed expression; 4 pillars of evidence | **none — writing + framing only** | high (unifying framework, falsifiable) | ideal | **0 days** |

## Recommendation seed (not committed)

Default lean: **E**.
- E is now a *framework* paper with a generalized thesis about model behavior (refusal is gated, not localized) — the kind of unifying account that NeurIPS reviewers reward. It uses our existing data as evidence for the framework rather than as the contribution itself, which is the standard NeurIPS rhetorical posture.
- A is the strongest single-claim methodological NeurIPS bet *if* the second-model replication lands. Otherwise A's empirical surface is narrower than E's framework surface.
- B is the safest ICML workshop bet but risks "this is one model" critique at NeurIPS main.
- C is the highest-variance — championed or dismissed.
- D requires the most writing discipline; thin on each contribution.

Hybrid worth considering: **E for both venues**, with the ICML version a 4-page focused cut on Pillars 1+2+3 (the framework + evidence) and the NeurIPS version expanding §7 (mechanism implication / case studies) and §8 (falsification protocol + safety implications). Same paper, two depths. The 3 weeks between deadlines go into writing depth, not new experiments.

If the second-model replication becomes feasible during those 3 weeks, the NeurIPS version expands by adding a § 7.5 "evidence on a second model" — but the framework holds even without it.

---

## Open questions for the team

1. Is a Gemma-2-9B-IT replication realistic in the NeurIPS window (~3 weeks)? Transcoders are available; attribution wall on a 9B is ~2× our 4B times, plus dataset re-prep.
2. Are we OK shipping the ICML workshop version as the "Method + Mechanism" cut (Outlines A + B subsets) and saving the negative-result framing for NeurIPS? Or do we want both venues to have the same backbone?
3. Do we have access to a second JB-style controlled dataset (toxicity, sycophancy, deception)? If not, we should start that data collection in parallel.
4. Naming: is "Refusal-Lens" the brand for the pipeline (Outline D), or do we keep the title focused on the finding (A/B/C) and treat the pipeline as supplementary?
