# Qwen Subcircuit Identification + Ablation + Top-K Sparsity Sweep — Design Spec

**Date:** 2026-06-01 · **Owner:** Mahmoud · **Status:** approved-in-conversation, implementing

## 1. Goal

Replicate the Gemma subcircuit methodology (Stage 07 identification + Stage 08
ablation) on **Qwen3-4B at L18**, and add a new **Top-K sparsity sweep** that
answers: *how many features (or edges) must be removed to break (a) the refusal
mechanism and (b) the jailbreak mechanism?* Then produce a results report and
surface both the rule-based subcircuits and the Top-K sets in the frontend
graph viewer.

Why Qwen first: the Batch-17/18 audits showed Qwen's edge-attribution mass sits
at effective coefficient ≈ 1.0 (vs Gemma's 0.005, ~20–50× below behavioral
threshold), so circuit-level edits that were sub-threshold on Gemma should be
behaviorally potent on Qwen.

## 2. Approved decisions (from design Q&A)

| Decision | Choice |
|---|---|
| Ablation mechanism | **Both**: true feature-zeroing (Stage-08 style, features only) AND residual-attribution proxy (features + edges) |
| Ranking | **Both**: by attribution-to-refusal-target AND by raw activation; compare Pareto curves |
| K schedule | **[1, 3, 5, 10, 25, 50, 100, 250]**, per-prompt rankings (user: "features activated for each prompt") |
| Upstream data | User offered full re-run; design defaults to **reuse** (see §4) with `REGEN_UPSTREAM=1` flag for the full 01→02→02c re-run |
| Subcircuit identification | Keep the existing Stage 07 **rule-based set-logic** methodology (NOT 02b correlation — corrected misconception), identical to Gemma |
| Execution | RunPod H100, tmux session + separate watcher window that pushes results to GitHub/HF on completion |

## 3. Methodology

### 3.1 Stage 07/08 replication (existing scripts, new inputs)

`scripts/pipeline_qwen/07_identify_subcircuits.py` and
`08_ablate_subcircuits.py` already exist (ports of the Gemma versions). They
need their upstream inputs materialized for the L18 run
(`run_emnlp_qwen_L18_20260522`):

- `02_attribution/attribution_results.json` — **reconstructed** from the 550
  packed graphs already on HF (see §4).
- `04_labels/feature_labels.json` + `feature_class_sets.json` — Stage 04 with
  `--skip-download` (bookkeeping only; label text stays empty — labeling is
  backburnered per 2026-05-31 decision). Verified: `--skip-download` with an
  empty cache still writes complete `(layer, attribution, conditions_seen,
  top50_conditions)` records, which is all Stage 07 consumes.
- Stage 07 runs with `--graph-mode single` (our graphs are single-mode).
- Stage 08 runs the 5 default subcircuits × {all, anchors} positions ×
  11 conditions × 50 prompts, `--max-new-tokens 80` (parity with all sweep
  cells; config default 200 would ~2.5× the cost). **Patch**: add
  `--graph-mode` flag (default `multi` for back-compat) — currently hardcoded
  `mode="multi"` at the coverage-index call site, which would silently disable
  the low-coverage diagnostic on single-mode data.

Stage 08 measures the Gemma-identical dissociation matrix: positive control
(`universal_refusal_core` → bare break), negative control
(`ctrl_shared_refusal` → no JB effect), per-class dissociation
(`jb_{cls}_specific_vs_ctrl`).

### 3.2 Top-K sparsity sweep (new driver)

`scripts/emnlp_perm_edit/00_topk_circuit_sweep_qwen.py`. For each
(prompt, condition), load its packed graph, extract per-feature records
aggregated by `(layer, feat_idx)` across context positions
(`signed_attribution` = sum of edge weights into the L18 refusal target node;
`activation` = max node activation), then ablate the top-K under the cell's
ranking and mechanism. All generations greedy, `max_new_tokens=80`,
`enable_thinking=False`.

**Mechanisms**

- `proxy` — `make_scalar_rhat_subtraction_hook(r_unnorm, delta)` on L18 with
  `delta = (Σ top-K signed_attribution) × ‖r_unnorm‖` (the established
  normalized→unnormalized conversion; identical math to the v2 Qwen edge
  runtime). fp32, TF32 off. Supports `--source features` and
  `--source edges` (all edge types).
- `zero` — circuit-tracer `ReplacementModel.feature_intervention_generate`
  zeroing the top-K `(layer, feat_idx)` at all positions (Stage-08 `all`
  positions-mode convention). bf16. `--source features` only (true edge
  zeroing is unsupported — deferred like Gemma's "rigorous" variant).

**Rankings × target conditions** (per mechanism, per K):

| ranking | selects | run on | measures |
|---|---|---|---|
| `attr_pos` | top-K most **pro-refusal** attribution | `bare` (50) | refusal break: REFUSE→COMPLY |
| `attr_neg` | top-K most **anti-refusal** attribution | `jb_*` (250) | jailbreak removal: COMPLY→REFUSE |
| `activation` (features) / `abs` (edges) | top-K by activation / \|attr\| | bare + jb_* (300) | both directions |

Rationale for condition restriction: each target is only measurable where the
baseline behavior exists (refusal break on refusing prompts; JB removal on
complying JB prompts). Ctrl conditions stay in Stage 08 (controls live there);
excluding them from the sweep cuts ~45% of sweep compute without losing the
Pareto knee. A no-intervention **baseline cell** (300 gens) is run once per
mechanism family (plain model for proxy, ReplacementModel for zero) for flip
computation.

Per-prompt K is clamped to available records (`n_used` recorded), matching the
existing Gemma top-K sweep convention.

**Cell count**: 3 mechanism-sources × 8 K × 600 gens = 14,400 + 600 baselines
= **15,000 generations**, plus Stage 08's 6,050.

### 3.3 Aggregation, report, frontend

`scripts/emnlp_perm_edit/qwen_subcircuits_aggregate.py` (CPU):

- Pareto curves: flip-rate (with Wilson 95% CI) vs K per (mechanism, ranking),
  knee identification; proxy-vs-zero agreement on features; feature-vs-edge
  comparison; Gemma side-by-side (existing `topk_*_sweep.json`).
- `QWEN_SUBCIRCUIT_REPORT.md` — dissociation matrix digest + Pareto knees.
- **Frontend export**: corpus-level Top-K sets (features ranked by mean |attr|
  across bare graphs for the refusal mechanism, across jb graphs for the JB
  mechanism; K ∈ {10, 25, 50}) emitted as `topk_refusal_K{N}` /
  `topk_jailbreak_K{N}` entries and **merged with Stage 07's
  `subcircuits.json`**. Verified: `annotate_subcircuits()` stamps membership
  onto graph nodes at `stage_frontend()` time and unknown subcircuit names
  pass the filter, so the existing panel renders them with zero JS changes.
  Watcher uploads the merged file to HF `runs/<run>/subcircuits.json`;
  collaborators get it via the existing `fetch_graph_data.py` flow.
  (Note: the sweep itself uses per-prompt rankings; the corpus-level sets are
  for visual inspection only and are labeled as such in the report.)

## 4. Upstream data strategy

The 550 packed L18 graphs (`node_threshold=0.8`, `edge_threshold=0.98`) are on
HF and locally. The only missing Stage-07/08 inputs are derivable from them:

- `qwen_rebuild_attribution_index.py` (CPU) reconstructs
  `attribution_results.json` in the exact Stage-02 schema rows
  (`{prompt_idx, prompt_id, conditions: {cond: {graphs: {single:
  {top_features, top50_features, net, …}}}}, feature_comparison: {cls:
  {vs_bare, vs_ctrl, ctrl_vs_bare}}}`) with `metadata.reconstructed: true`.
  `compare_features` logic is replicated verbatim (can't import Stage 02
  locally — it needs circuit-tracer).
- **Known approximations** (documented in output metadata): (a) packed graphs
  are node/edge-thresholded, so feature sets are the pruned top mass — top-50
  membership is effectively unaffected; (b) Stage 02's per-feature attribution
  takes the last position-instance on key collision, the rebuild **sums across
  position instances** — more principled and stable for ranking; set
  membership is robust to this.
- Direction files (`positions_L18/pos_-1_unnormalized.pt`,
  `direction_metadata.json`, `directions/layer_18.pt`) come from the proven
  Path-A git-show fallback (committed on `temp/gemma-vs-qwen-pipeline`); these
  are the exact directions the L18 graphs were generated from.

`REGEN_UPSTREAM=1` runs the full 01→02→02c chain instead (+~9–13 h GPU,
+$30–40) for a fully self-contained run; default is reuse.

## 5. Orchestration

- `runpod_qwen_subcircuits.sh` — tmux self-relaunch (session
  `qwen_subcircuits`), venv + `-e . -e ./vendor/circuit-tracer` install, CUDA
  check, prereq staging (HF graph pull if absent; direction git-show fallback;
  dataset check), then `run_step`-wrapped steps in de-risked order:
  rebuild-index → 04 → 07 (CPU, results land early) → 08 (RM) → zero sweep
  (RM) → proxy features → proxy edges → aggregate. DONE/FAIL marker files;
  incremental JSON saves throughout; `DRY_RUN=1` prints every command without
  GPU execution (used in local testing).
- `watch_and_commit_qwen_subcircuits.sh` — second tmux window; polls DONE
  marker; on completion: git add/commit/push the result JSONs + report to the
  branch, and `huggingface_hub` upload of merged `subcircuits.json` (+ Stage 04
  set files) to `moon70/refusal-lens-graphs` `runs/<run>/`.
- `smoke_test_qwen_subcircuits.sh` — every step with `--max-prompts 2`,
  `K=1,5`, 1 subcircuit, then schema assertions on outputs (~20–30 min, ~$1.5).

## 6. Local testing (pre-pod gate)

1. Unit tests (`scripts/emnlp_perm_edit/tests/test_qwen_subcircuit_orchestration.py`,
   base-venv runnable): ranking selection (pos/neg/abs/activation), K clamping,
   delta conversion math, `compare_features` replica vs synthetic fixtures,
   feature-key round-trip, intervention-tuple building.
2. `bash -n` on all three shell scripts.
3. Launcher `DRY_RUN=1` end-to-end against the real local repo (prereq checks
   exercised against the locally fetched graphs).
4. **Real CPU pipeline run locally**: rebuild-index → Stage 04
   `--skip-download` → Stage 07 `--graph-mode single` on the actual 550 local
   graphs; assert `subcircuits.json` is produced with non-empty
   `universal_refusal_core` and the 5 `jb_*_specific_vs_ctrl` sets. This both
   tests the orchestration and delivers the Qwen subcircuit definitions before
   any GPU spend.

## 7. H100 budget (single 80 GB H100 SXM)

Grounded in measured Path-A throughput (plain Qwen fp32 mnt=80: 1.6–2.8 s/gen
→ plan 2.0; ReplacementModel bf16 estimated 5 s/gen — transcoders are
36 × ~1.7 GB ≈ 60 GB VRAM, fits alongside the 8 GB model; same stack as the
graph-generation run):

| step | gens | s/gen | wall |
|---|---|---|---|
| setup + model/transcoder pulls | — | — | ~0.7 h |
| CPU steps (rebuild, 04, 07, aggregate) | — | — | ~0.4 h |
| Stage 08 (5 subcircuits × 2 modes × 11 conds × 50 + baselines) | 6,050 | 5 | ~8.4 h |
| zero sweep (8 K × 600) | 4,800 | 5 | ~6.7 h |
| proxy features (8 K × 600) | 4,800 | 2 | ~2.7 h |
| proxy edges (8 K × 600) | 4,800 | 2 | ~2.7 h |
| baselines (plain + RM) | 600 | 2–5 | ~0.6 h |
| **Total (reuse path)** | ~21,050 | | **~22 h ± 20%** |

≈ **$65–80** at $2.99/h (secure) / ~$60–70 community. `REGEN_UPSTREAM=1` adds
~9–13 h → ~31–35 h ≈ $95–110. If wall time must shrink: drop `proxy edges`
(−2.7 h) or halve the zero-sweep K list — both flagged as env-var knobs.

## 8. Out of scope

Feature labeling (backburnered), Tejas dataset v2, EMNLP natural-circuit-edits
plan (shelved until Georg's 3 tasks complete), true transcoder edge-patching.
