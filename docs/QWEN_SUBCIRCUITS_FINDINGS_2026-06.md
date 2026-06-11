# Qwen Subcircuit Replication — Findings & Next Steps (2026-06-11)

Narrative record of the full Qwen3-4B (L18) subcircuit run: Stage 07 rule-based
identification + Stage 08 feature-zeroing ablation + the new Top-K sparsity
sweep. Companion to the auto-generated `QWEN_SUBCIRCUIT_REPORT.md` (which holds
the raw Pareto tables) and the design spec
`docs/superpowers/specs/2026-06-01-qwen-subcircuits-topk-design.md`.

**Run:** `run_emnlp_qwen_L18_20260522`, H100 80GB, ~33 h wall, all 8 steps OK,
0/14,400 sweep generations incoherent. Baselines/ablations greedy, mnt=80,
`enable_thinking=False`.

---

## TL;DR (the five things worth telling Georg)

1. **The subcircuit methodology *works* on Qwen where it failed on Gemma.**
   Zeroing the 122-feature `universal_refusal_core` at all positions breaks
   bare refusal **90%** of the time on Qwen — vs **4.3%** on Gemma with the
   identical pipeline. This is the behavioral confirmation of the
   residual-stream-norm hypothesis: feature edits that were sub-threshold on
   Gemma are potent on Qwen.
2. **Breaking refusal is extremely sparse; removing a jailbreak is not.** Zeroing
   the top **~5–10** attribution-ranked features (per prompt) breaks bare refusal
   in the majority of cases (K=5→59%, K=10→85%, K=25→98%). Restoring refusal on a
   jailbroken prompt needs **~100+** features and never exceeds 63% even at K=250.
   A clean, quantified **asymmetry**.
3. **Edges beat features, and attribution beats activation** — both cleanly. The
   "edge > node" Pareto (more behavioral change per unit ablated) holds on Qwen,
   and attribution-ranked selection is far more efficient than
   activation-ranked. These answer two of the questions the sweep was designed for.
4. **Fine-grained class-specific dissociation mostly does NOT replicate** (same
   as Gemma). Of 5 `jb_X_specific_vs_ctrl` sets, only fiction shows a clean
   positive dissociation; the rest are null/negative, and the one large
   positive (completion +76pp) is a tiny-denominator artifact.
5. **A validity threat we must resolve before quoting any of these numbers as
   final:** the keyword `classify_response` is known to over-count compliance
   (see [[project_dataset_v2_behavioral_findings]]), and several Qwen jailbreak
   classes barely jailbreak at all. Treat the rates as directionally strong but
   provisional pending re-scoring.

---

## 1. Stage 08 — subcircuit ablation

### 1.1 Positive control succeeds on Qwen (the headline)

| metric (universal_refusal_core, all-positions) | Qwen | Gemma (run_20260430) |
|---|---|---|
| bare refusal break | **90.2%** (37/41) | 4.3% |
| n features | 122 | 26 |

This is the single most important result: the **identical** Stage 07/08
methodology that produced a near-null positive control on Gemma produces a
strong one on Qwen. It validates that the cross-model difference we traced in
the Batch-17/18 audits (Gemma's ~200× larger residual-stream norm burying
feature-level edits) is real and behaviorally decisive.

### 1.2 The "universal core" is genuinely refusal-causal, but broad

Ablating the universal core breaks refusal not just on `bare` (90%) but across
every `ctrl_*` condition too (44–67%):

| condition | break rate | condition | break rate |
|---|---|---|---|
| bare | 90% | ctrl_roleplay | 67% |
| ctrl_cognitive_reframe | 65% | ctrl_fiction | 61% |
| ctrl_analytical | 44% | ctrl_completion | 44% |

So it's a true refusal mechanism, not a `bare`-specific quirk — but it is **not
selective**: it knocks out refusal wherever refusal is happening. Good for "we
found *the* refusal circuit," weaker for "we found a *targeted* lever."
(jb_recovery is ~0 by construction — you cannot restore refusal by removing
pro-refusal features from an already-complying prompt.)

### 1.3 Position is the dominant lever (all ≫ anchors)

Same subcircuit, same features, only the intervention positions differ:

| universal_refusal_core | bare break |
|---|---|
| all positions | **90%** |
| template anchors `[-5,-3,-1]` only | **2.4%** |

The effect comes from displacing the refusal features across the *whole*
sequence, not at the measured template tokens. This mirrors the Task-3 direction
sweep (all-positions coeff=1.0 → 100% flip vs pos-only → 10%) and is itself a
finding worth stating: refusal is maintained by distributed, position-spread
feature activity, not a single anchor token.

### 1.4 Class-specific dissociation: mostly fails (consistent with Gemma)

`dissociation_delta` = target-class recovery − mean other-class recovery
(positive = the class-specific subcircuit selectively restores refusal for its
own class):

| subcircuit (all-pos) | Δ | verdict |
|---|---|---|
| jb_completion_specific | +76.2pp | **artifact** (see below) |
| jb_fiction_specific | +14.0pp | weak positive |
| jb_cognitive_reframe_specific | +3.5pp | null |
| jb_roleplay_specific | −6.1pp | negative |
| jb_analytical_specific | −7.7pp | negative |

Only fiction shows a believable class-selective effect. This reproduces the
Gemma "failed dissociation" outcome and says the corpus-level
`jb_X_specific_vs_ctrl` set logic does not cleanly isolate per-class causal
circuits — directly motivating Georg's pivot to *active + newly-active* feature
sets ([[batch16-georg-paper-pivot-may24-2026]]).

**Why completion's +76pp is not real:** the completion *jailbreak barely works*
on Qwen — only 9/50 completion prompts comply at baseline (82% still refuse,
identical to bare). 77.8% "recovery" is ~7 of 9 prompts; the denominator is too
small and the baseline too refusal-dominated to trust.

### 1.5 Coverage caveat (why the dissociation nulls are partly mechanical)

Fraction of prompts where the subcircuit's features were **not** in that
prompt's top-K active features (`low_coverage`):

| subcircuit | low-coverage prompts |
|---|---|
| universal_refusal_core | 8.7% (features genuinely active) |
| jb_cognitive_reframe_specific | 83% |
| jb_analytical_specific | 89% |
| jb_fiction/roleplay/completion_specific | 91% |
| ctrl_shared_refusal | **100%** |

The class-specific and ctrl sets are largely composed of features that **don't
fire strongly** on the prompts we ablate them on — so their near-null effects
are partly "we ablated features that weren't doing anything here," not "these
features are causally irrelevant." Any conclusion about class-specific circuits
is confounded by this until the set-construction is fixed.

---

## 2. Top-K sparsity sweep (the novel, quantitative contribution)

Per-prompt top-K ablation, K ∈ {1,3,5,10,25,50,100,250}. "break" = bare
REFUSE→COMPLY; "recovery" = jb COMPLY→REFUSE.

### 2.1 Refusal is sparse; jailbreak-suppression is distributed

True feature zeroing, attribution-ranked:

| K | break refusal (bare, `pos`) | remove jailbreak (jb, `neg`) |
|---|---|---|
| 1 | 24% | 13% |
| 5 | 59% | 18% |
| 10 | 85% | 24% |
| 25 | 98% | 39% |
| 50 | **100%** | 48% |
| 100 | 100% | 52% |
| 250 | 98% | 63% |

**~5–10 features carry refusal**; it saturates by K=50. **Removing a jailbreak
is fundamentally more distributed** — it climbs steadily and never saturates in
our range. Interpretation: a jailbroken prompt's compliance is held up by many
small contributions, while refusal rides on a compact, fragile feature set. This
"refusal = sparse default basin / jailbreak = distributed suppression" framing
is the most paper-worthy idea from the run.

### 2.2 Edge > node (H0-7 holds on Qwen)

Break-refusal knees (smallest K to reach threshold), proxy mechanism:

| source | break@50% | break@80% | recovery@K=250 |
|---|---|---|---|
| proxy **edges** (`pos`) | **K=10** | **K=25** | 64% |
| proxy **features** (`pos`) | K=25 | K=250 | 45% |

Ranking *edges* (feature + embedding + error contributions to the L18 target)
is markedly more efficient per-K than ranking features alone — consistent with
the edge-vs-node Pareto we hypothesized. Edges also recover more jailbreaks at
high K (64% vs 45%).

### 2.3 Attribution ranking ≫ activation ranking

True zeroing, break refusal: `pos` (attribution) reaches 100% by K=50; the
`activation` ranking only reaches 51% at K=100 and needs K=250 for 100%.
Strongest-*activated* features are *not* the strongest *causal* features —
attribution to the refusal target is the right selector. (Answers the "both,
compare" design question decisively.)

### 2.4 True zeroing > residual proxy at low K

Break refusal at K=10: zero **85%** vs proxy-features **36%**. Zeroing removes
all downstream paths of a feature; the proxy only subtracts its direct
r̂-projection mass. The gap quantifies how much of a feature's behavioral effect
is indirect/non-linear — large, as the Task-3 audit predicted.

---

## 3. The baseline-heterogeneity problem (read before trusting recovery numbers)

Qwen's jailbreak classes differ wildly in how often they actually jailbreak
(baseline comply rate on the 50 prompts):

| class | jb comply | class | jb comply |
|---|---|---|---|
| jb_analytical | 96% | jb_fiction | 30% |
| jb_roleplay | 94% | jb_completion | **18%** |
| jb_cognitive_reframe | 90% | | |

`completion` and `fiction` barely jailbreak Qwen, so their "recovery" denominators
are tiny and noisy (this is what produced the completion artifact in §1.4). The
pooled `n_jb_comply=164` is dominated by roleplay/analytical/cognitive_reframe.

**Compounding threat:** the keyword classifier `classify_response` is known to
over-count compliance by ~2–3× and the v2 dataset's jailbreaks are weak
([[project_dataset_v2_behavioral_findings]]). So both the high break rates and
the comply baselines may be inflated by classifier false-positives. The
*relative* patterns (sparse-vs-distributed, edge>node, attribution>activation,
Qwen≫Gemma) are robust to a uniform classifier bias; the *absolute* rates are
not. **Do not quote absolute percentages as final until re-scored.**

---

## 4. What replicated vs. what's new

| | Gemma | Qwen |
|---|---|---|
| universal-core positive control | ❌ 4.3% | ✅ 90% |
| class-specific dissociation | ❌ failed | ❌ failed (same) |
| position: all ≫ anchors | ✅ | ✅ |
| refusal sparser than jailbreak-suppression | (untested at this K grid) | ✅ new |
| edge > node | suggested (Task-3) | ✅ confirmed |
| attribution > activation ranking | (untested) | ✅ new |

The methodology *executes* on Qwen (the Gemma blocker is gone), the *coarse*
refusal circuit is real and sparse, but the *fine-grained per-class* story
still doesn't hold — pointing the same direction Georg already pivoted toward.

---

## 5. Recommended next steps

**Validity (do first — gates everything else):**
1. **Re-score with a trustworthy classifier.** Re-run classification over the
   already-saved generations (they're in the result JSONs — no GPU needed) with
   an LLM-judge or the improved scorer from the v2 work, and regenerate the
   Pareto curves + Stage 08 summary. Confirms whether the 90% break and the
   asymmetry survive. **Cheap, high-leverage.**
2. **Restrict recovery analysis to the strong jailbreak classes** (roleplay,
   analytical, cognitive_reframe; drop completion/fiction or report them
   separately), so recovery denominators are meaningful.

**Science (the paper-worthy threads):**
3. **Make the sparse-vs-distributed asymmetry rigorous** — per-class Pareto
   curves with CIs, and an explicit "minimum features to break refusal"
   statistic. This is the strongest novel finding.
4. **Fix class-specific set construction** via Georg's active + newly-active
   feature definition, then re-run Stage 08 dissociation — the current
   83–100% low-coverage shows the present sets are the wrong features.
5. **Head-to-head Gemma Top-K** on the same K grid + rankings, so the
   sparse-vs-distributed and edge>node claims are cross-model, not Qwen-only.

**Plumbing:**
6. **Verify the frontend** picked up `subcircuits_frontend.json` (rule-based +
   `topk_refusal/jailbreak_K*` sets) on HF and renders in the panel.
7. Optionally label the Qwen features (backburnered) so the subcircuits in the
   viewer are human-readable when we present them.

---

## Artifacts

| file | what |
|---|---|
| `…/08_ablation/ablation_summary.json` | full dissociation matrix + coverage |
| `qwen_subcircuits/pareto_curves.json` | flip-rate vs K, Wilson CIs, knees |
| `qwen_subcircuits/pareto_curves.png` | break/recovery curves (all mechanisms) |
| `qwen_subcircuits/topk_sweep_{zero,proxy}_*.json` | per-generation sweep records (re-scorable) |
| `qwen_subcircuits/subcircuits_frontend.json` | rule-based + Top-K sets for the viewer |
| `07_subcircuits/subcircuits.json` | 18 rule-based subcircuits |
