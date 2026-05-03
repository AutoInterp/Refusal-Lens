# Refusal-Lens Comparison: Gemma-3-4b-it vs Qwen3-4B

**Status:** living document. Gemma side is fixed (l15-refactor `run_20260430_023247`). Qwen side fills in as each stage completes — the `TBD` cells track which GPU phase produces them.

---

## TL;DR (to be filled in after all phases)

> **Headline 1.** _(Stage 06)_ At its best causal layer, Qwen3-4B's refusal direction `r` flips JB-comply → REFUSE at **TBD%** (Phase E2) — compare with Gemma-3-4b-it's **100% (89/89)** at L15. Anti-direction (REFUSE → COMPLY): Qwen **TBD%**, Gemma **98% (49/50)**.
>
> **Headline 2.** _(Stage 08)_ Class-specific subcircuit ablation shows **TBD** dissociation on Qwen vs Gemma's existing per-class breakdown (positive control breaks bare refuse; class-specific subcircuit drops its own JB flip rate disproportionately).
>
> **Headline 3.** _(generalisation claim)_ The same `r ± h` mechanism transfers across families with comparable potency / weaker / stronger (TBD), supporting the claim that the refusal axis is a **family-invariant 1-D residual-stream feature** (or refining where it is not).

---

## Setup

| Surface | Gemma-3-4b-it | Qwen3-4B |
|---|---|---|
| HF model | `google/gemma-3-4b-it` | `Qwen/Qwen3-4B` |
| Decoder layers | 34 | 36 |
| d_model | 2560 | 2560 |
| Chat-template tail | `<start_of_turn>model\n` | `<\|im_start\|>assistant\n` (`enable_thinking=False`) |
| Direction position | `-2` (`model` token) | `-1` (trailing `\n`; **placeholder**, verify via Stage 01 sweep) |
| Best separation layer | **L32** (`~20,873`) | **L34** (`~260` — placeholder; orders of magnitude lower than Gemma; needs investigation) |
| Best causal layer | **L15** (Tejas Script 16 sweep) | **L18** (placeholder; verify via `01b_layer_sweep.py`) |
| Cos(L_sep, L_caus) | -0.115 | TBD |
| Block pre-MLP LN | `pre_feedforward_layernorm` (Gemma quad-LN) | `post_attention_layernorm` (Qwen pre-LN) |
| Decoder access path | `model.model.language_model.layers[L]` | `model.model.layers[L]` |
| Config access path | `model.config.text_config.hidden_size` | `model.config.hidden_size` |
| Transcoders | `mwhanna/gemma-scope-2-4b-it` (16k width) | `mwhanna/qwen3-4b-transcoders` (160k width, ~10×) |
| MLP-first-token mask | `[0:4]` | `[0:4]` (verify via tokenized template) |
| Measurement hook | `hook_resid_post` | `hook_resid_post` |
| Backend | `transformerlens` | `transformerlens` |
| Dataset | 50 prompts × 11 conditions (controlled) | same dataset, re-validated by `verify_dataset_qwen.py` |
| Classifier | regex (19 phrases, model-agnostic) | **same** regex (paper-grade symmetry) |

---

## Stage 01 — refusal direction (per-layer separation)

| | Gemma (L32) | Gemma (L15) | Qwen (L34) | Qwen (L18) |
|---|---|---|---|---|
| Separation `\|r\|` | 20,873.4 | 3,128.7 | 260.0 (placeholder) | TBD |
| cos(L_sep, L_caus) | — | cos(L32, L15) = +0.94 | — | TBD |

**Note.** Qwen3-4B's per-layer separation curve at pos=-1 peaks at ~260 — about 80× lower than Gemma's L32. Three plausible causes (to disambiguate before Stage 02):

1. The chosen pos=-1 (trailing newline after `assistant`) is the wrong analog of Gemma's pos=-2 (`model` token). Run Stage 01 with the new per-position sweep to find Qwen's max-separation position.
2. Qwen3-4B's residual stream is genuinely smaller-norm at safety-relevant tokens (training-distribution effect). Less interesting; means the magnitude is just calibrated differently and the *direction* is still load-bearing.
3. Qwen distributes the refusal computation more evenly across layers/heads (no L32-style "spike layer"). This would be a **substantive finding** for the paper.

> **Action item.** Compare the per-layer curves visually (`02b_stats/separation_by_layer.png` from each run). If Qwen has a flat curve and Gemma has a sharp L32 spike, that's the headline architectural difference.

---

## Stage 02 — attribution graphs

| | Gemma run_20260430_023247 | Qwen run_TBD |
|---|---|---|
| Target layer | L15 | L18 (causal) |
| Target positions (multi) | `[-5, -3, -2]` | `[-5, -3, -1]` (Qwen template anchors) |
| Target positions (single) | `[-2]` | `[-1]` |
| #features per graph (median) | TBD | TBD |
| MLP attribution share | 0.4% (rest = attn + embeds) | TBD |
| Sign-flipped features (vs bare, per JB class) | per `02b_stats/EXPERIMENT_SUMMARY.md` | TBD |

**Reading guide.** The l15 Stage 02b emits effect sizes (Cohen's d, Wilcoxon p, 95 % CIs) for three pairwise comparisons per JB class:

- `vs_bare` — legacy delta (JB vs bare prompt)
- `vs_ctrl` — token-matched delta (JB vs length-matched neutral control)
- `ctrl_vs_bare` — sanity (ctrl should track bare; large effect here = prefix confound)

For the paper, **`vs_ctrl` is the comparison that matters** — it isolates JB *semantics* from token-position artifacts. Comparison rows below:

### Gemma `vs_ctrl` headline (multi, L15) — from `run_20260430_023247`

| Class | ΔNet | Cohen's d | Dominant |
|---|---|---|---|
| roleplay | +1607.5 | +0.57 | Amplification-dominant |
| fiction | +1268.8 | +0.45 | Pro-refusal recruitment |
| analytical | -8874.0 | -3.78 | Balanced |
| completion | -1682.4 | -0.97 | Anti-suppression |
| cognitive_reframe | -11774.1 | -3.21 | Balanced |

### Qwen `vs_ctrl` (multi, L18) — TBD after Phase C3

| Class | ΔNet | Cohen's d | Dominant |
|---|---|---|---|
| roleplay | TBD | TBD | TBD |
| fiction | TBD | TBD | TBD |
| analytical | TBD | TBD | TBD |
| completion | TBD | TBD | TBD |
| cognitive_reframe | TBD | TBD | TBD |

---

## Stage 03 — verification (Σ edges ≈ direct dot product)

| | Gemma | Qwen |
|---|---|---|
| Σ-edges / direct-dot ratio | ~1.00 (verified bug-free, see [MENTEE_NOTE_three_bugs.md](MENTEE_NOTE_three_bugs.md)) | TBD |
| Per-layer contribution plot | `03_verification/per_layer_contribution.png` (Gemma) | TBD |

**What we're checking.** Is `attribute()`'s reported feature contribution consistent with the residual stream's actual dot product against `r̂`? Both pipelines hit `~1.00` validates the residual-stream hook (rather than the post-RMSNorm pre-MLP default that breaks magnitudes by ~1700×).

---

## Stage 04 — feature labelling

| | Gemma | Qwen |
|---|---|---|
| Transcoders covered | 32 layers × 16k features | 36 layers × 160k features |
| Features in top-100 union | 4,997 | TBD |
| Top-features report | `04_labels/top_features_report.md` | TBD |

**Caveat.** Gemma3-4b-it isn't on Neuronpedia (uses `mwhanna/gemma-scope-2-4b-it` byte-range fetches). Qwen3-4B uses `mwhanna/qwen3-4b-transcoders` — labels via the same byte-range pattern, but **160k features per layer is 10× wider so coverage of the active set may differ.**

---

## Stage 06 — Arditi causal intervention (the headline)

| | Gemma L15 | Qwen L_TBD |
|---|---|---|
| `pro_refusal_add` flip rate (overall) | **100% (89/89)** | TBD |
| ↳ analytical | 100% (28/28) | TBD |
| ↳ cognitive_reframe | 100% (33/33) | TBD |
| ↳ completion | n/a (0 baseline-comply) | TBD |
| ↳ fiction | 100% (19/19) | TBD |
| ↳ roleplay | 100% (9/9) | TBD |
| `anti_refusal_sub` flip rate (bare → comply) | **98% (49/50)** | TBD |
| Benign force-refuse control | **100% (10/10)** | TBD |
| Coherent flip rate | 100% on both directions | TBD |

**What this measures.** A non-trivial claim: across 11 conditions × 50 prompts, the same 1-D direction in the residual stream at a single layer can:
- push every JB-comply baseline back to refuse (`+r`), and
- pull every bare-refuse baseline into comply (`−r`),

while remaining coherent (≥80 % of flipped responses parseable, no gibberish). That's the operational definition of "_r is the refusal axis_".

> **Discussion seed.** If Qwen reproduces ≥80% on both directions, the paper's universality claim is strong. If it doesn't, the interesting question becomes: is the refusal mechanism (a) distributed across multiple directions (no single "axis"), or (b) gated by a non-linearity that 1-D add/sub can't bypass?

---

## Stage 08 — subcircuit ablation (the dissociation matrix)

| Subcircuit (positive/negative control) | Gemma comply-weighted JB recovery | Qwen comply-weighted JB recovery |
|---|---|---|
| `universal_refusal_core` (positive) | 26.6% (n_jb_comply=94) | TBD |
| `ctrl_shared_refusal` (negative) | _(ablating ctrl-shared should NOT recover JB)_ | TBD |
| `jb_fiction_specific_vs_ctrl` | TBD (fiction recovery > other classes) | TBD |
| `jb_analytical_specific_vs_ctrl` | TBD (analytical recovery > other classes) | TBD |
| `jb_cognitive_reframe_specific_vs_ctrl` | TBD (cogn. reframe recovery > other classes) | TBD |

**The dissociation figure.** The headline NeurIPS-style result is the heatmap `dissociation_matrix.png`:

- rows = subcircuits (universal core, per-class specific)
- columns = JB classes
- cell = per-class JB recovery rate when that subcircuit is zero-ablated

A diagonally-dominant matrix means **each class-specific subcircuit drives its own class disproportionately** — proof of dissociation. Off-diagonal leakage = shared mechanism. Same matrix on Gemma and Qwen → universality. Different patterns → architectural divergence in how refusal is decomposed.

---

## Methodology controls (per-paper)

| Control | Gemma | Qwen |
|---|---|---|
| Bare REFUSE rate | 50/50 (100%) | TBD via `verify_dataset_qwen.py` |
| Ctrl REFUSE rate | 216/225 (96%) | TBD |
| Classifier audit (false-positive / false-negative rate on N=50) | n/a (assumed clean for paper) | run `audit_classifier.py` after Stage 06 |
| Σ-edges = Σ(per-feature contributions) | yes (Stage 03) | TBD |
| Coherent-only flip rate (Stage 06) | identical to raw flip rate (no gibberish) | TBD |

---

## Reproducibility checklist

- [x] vendored `circuit-tracer` pinned to a measurement-hook-aware branch (Gemma: `multi-position fix`; Qwen: `refusal-lens-residual-stream-hook`).
- [x] same `classify_response()` regex on both runs.
- [x] same controlled dataset (`dataset/refusal_lens_controlled_dataset.json`); Qwen-side ctrl-leak pairs recorded in `dataset/qwen_dataset_verification.json`.
- [ ] Qwen Stage 01 re-run with per-position sweep (`PER_POSITION_LAYER`, `PER_POSITION_POSITIONS`) so multi-target Stage 02 has all the position files.
- [ ] Qwen Stage 02 re-run with `--save-graphs` so Stage 02c → Stage 05 → manual frontend ablation cart workflow works.
- [ ] Qwen Stage 06 + Stage 08 results committed.
- [ ] Per-stage commit hash linked from each table cell.

---

## Pending Qwen-side runs (in execution order)

1. **A2** — `python3 scripts/pipeline_qwen/01b_layer_sweep.py --run-dir <run>` (~1 h) → BEST_CAUSAL_LAYER. Update `pipeline_qwen/config.py`, then commit.
2. **B2** — `python3 scripts/pipeline_qwen/verify_dataset_qwen.py` (~30 min) → bare/ctrl/JB classifications + ctrl-leak pair list. Commit `dataset/qwen_dataset_verification.{json,md}`.
3. **C3** — `python3 scripts/pipeline_qwen/01_compute_direction.py --recompute` (~10 min, adds per-position files) → then `python3 scripts/pipeline_qwen/02_run_attribution.py --run-dir <run> --save-graphs` (~3-5 h on 80 GB GPU) → then `python3 scripts/pipeline_qwen/02c_pack_graphs.py --run-dir <run>` (~30 min).
4. **F0b** — `python3 scripts/pipeline_qwen/07_identify_subcircuits.py --run-dir <run>` (CPU, ~5 min) — produces the `ctrl_shared_refusal` + `jb_<class>_specific_vs_ctrl` subcircuit keys Stage 08 needs.
5. **D** — `python3 scripts/pipeline_qwen/05_visualize_circuits.py --run-dir <run>` (CPU, ~10 min) — frontend bundle.
6. **E2** — `python3 scripts/pipeline_qwen/06_causal_intervention.py --run-dir <run> --max-prompts 2` (smoke) → full (~1-2 h on 80 GB GPU).
7. **F2** — `python3 scripts/pipeline_qwen/08_ablate_subcircuits.py --run-dir <run> --subcircuits universal_refusal_core,ctrl_shared_refusal,jb_fiction_specific_vs_ctrl,jb_analytical_specific_vs_ctrl,jb_cognitive_reframe_specific_vs_ctrl --positions both` (~4-6 h on 80 GB GPU).

After each run completes, fill in the relevant TBD cells above, attach the commit hash, and tick the matching box in the reproducibility checklist.

---

## Open questions for the paper

1. **Why is L_sep separation 80× weaker on Qwen than Gemma at the trailing template token?** Test by re-running Stage 01 at multiple positions; compare per-position curves.
2. **Does the L_sep ↔ L_caus near-orthogonality hold on Qwen?** Gemma: cos(L32, L15) = -0.115 (essentially orthogonal). Qwen: TBD.
3. **Does the MLP-only-explains-0.4 % result hold across families?** Gemma: 99.6% of refusal at L32 is attn + embeds. If Qwen also concentrates refusal in attention, we have a strong universality claim about *what doesn't carry refusal*.
4. **Are the per-class subcircuits cross-family stable?** A `jb_fiction_specific` set on Gemma and a `jb_fiction_specific` set on Qwen — do they have semantically similar feature labels (Stage 04), or are they entirely different sets that happen to mediate the same behavior?

---

## Citation footnotes

- Gemma reference run: `data/results/pipeline_runs/run_20260430_023247/` (l15-refactor branch, 50 prompts × 11 conds, full Stage 02–08).
- Gemma report: [REPORT_run_20260430_023247.md](REPORT_run_20260430_023247.md).
- Three-bugs incident report (motivates `measurement_hook="hook_resid_post"`): [MENTEE_NOTE_three_bugs.md](MENTEE_NOTE_three_bugs.md).
- Paper-side framings already drafted: [PAPER_OUTLINES_v1.md](PAPER_OUTLINES_v1.md), [FRONTEND_ABLATION_PLAN.md](FRONTEND_ABLATION_PLAN.md), [EXPERIMENT_PLAN_canonical_pro_refusal_and_frontend_ablation.md](EXPERIMENT_PLAN_canonical_pro_refusal_and_frontend_ablation.md).
