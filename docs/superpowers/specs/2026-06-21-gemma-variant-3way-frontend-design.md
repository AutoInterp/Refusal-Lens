# Gemma Variant Attribution Graphs → 4-Way Frontend Comparison — Design

*Branch `emnlp-perm-edit`. 2026-06-21. Companion to `docs/REFUSAL_DIRECTION_INVESTIGATION_2026-06-16.md` (Georg's finding) and the Qwen subcircuits work.*

## 1. Motivation

Georg's investigation overturned the standing "Gemma's refusal isn't edge-carried"
conclusion. Gemma's diff-in-means refusal direction is **~81% (by squared norm) a single
causally-inert massive-activation dimension #443**; that contamination — not a mechanistic
difference — is what broke edge ablation on Gemma. Splitting the direction into `outlier`
(#443 only) and `complement` (#443 zeroed) and attributing toward each shows the
**complement is the real, feature-carried refusal circuit in both models** (Gemma 84% /
Qwen 48% bare→comply under the LLM judge), while the **outlier is inert** (0%).

The dashboard should let us *see* this: put the Gemma-complement, Gemma-full,
Gemma-outlier, and Qwen attribution graphs side by side for the same prompt, so the
"complement looks like Qwen's distributed circuit; full is artifact-dominated; outlier is
inert" story is visual, not just tabular.

## 2. Goals / Non-goals

**Goals**
- Reproducibly regenerate Gemma Stage-02 attribution graphs targeting three refusal-direction
  variants (**complement, full, outlier**) at L15 pos −2, with the residual-stream
  `measurement_hook=hook_resid_post` (vendored circuit-tracer fork), float32, `--save-graphs`.
- Pack + annotate + push each variant as its own frontend run to `moon70/refusal-lens-graphs`.
- A **manifest-driven N-column compare frontend** (4 columns for v1) with a shared
  prompt+condition picker; each column is a complete viewer for one target: attribution
  graph → input/output (node-connections) attribution list → top-activations, stacked.
- A correctness gate that verifies the regeneration reproduces Georg's attributed
  magnitudes before we trust the graphs.

**Non-goals (v1)**
- No feature-label baking into `clerp` (the existing Qwen graphs are unlabeled — only
  1/1718 nodes carry a label — so unlabeled Gemma panes are at parity; labeling is a later
  enhancement for all four columns).
- No subcircuit panel for the Gemma variants (they have no Stage 07 corpus sets; the panel
  stays empty for those columns, populated only for Qwen).
- No new backend; the compare site stays static, served via `python -m http.server` as today.

## 3. Data layout & naming (single source of truth)

- **HF dataset repo (push + fetch):** `moon70/refusal-lens-graphs` (dataset). The user has
  WRITE here; the Qwen run already lives at `runs/run_emnlp_qwen_L18_20260522/` (551
  `graph_data/*.json.gz`, `graph-metadata.json`, `subcircuits.json`, `run_info.json`).
- **New run-ids (one per Gemma variant):**
  `run_gemma_complement_L15`, `run_gemma_full_L15`, `run_gemma_outlier_L15`.
- **Slug scheme:** `{prompt_idx:03d}_{condition}_single` (single-mode, `--skip-multi-graph`).
  This matches Qwen's existing slugs exactly, so a shared `(idx, condition)` resolves across
  all four runs.
- **Conditions (11):** `bare`, `jb_*` (5), `ctrl_*` (5) from
  `dataset/refusal_lens_controlled_dataset.json` — identical 50-prompt set across both
  models, so `000_bare` is the *same* harmful request in every column.

## 4. Unit A — deterministic variant directions

`scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py` (CPU, idempotent).

- Load the canonical Gemma **unnormalized** direction at L15 (source:
  `data/results/pipeline_runs/run_20260430_023247/01_direction/unnormalized_r.pt`,
  `dict[layer]→tensor`; the per-position `pos_-2` unnormalized is the cross-check).
- `outlier_dim = int(r.abs().argmax())` — **assert == 443**.
- `full = r`; `outlier = zeros_like(r); outlier[443] = r[443]`;
  `complement = r.clone(); complement[443] = 0`.
- Unit-normalize each (`v / v.norm()`) and write to
  `data/results/pipeline_runs/gemma_var_{complement,full,outlier}/01_direction/`:
  - `directions/layer_15.pt` and `positions_L15/pos_-2.pt` (both the same unit vector;
    Stage 02 loads per-position).
- **Self-check (fails loudly):** `complement[443]==0`, `outlier` nonzero only at 443,
  `‖outlier‖/‖full‖ ≈ 0.90` and squared-norm share ≈ 0.81, reproducing
  `data/results/emnlp_perm_edit/phase0_controllability/gemma_outlier_split_stats.json`. If
  variant run-dirs already carry committed direction files, assert the reconstruction matches
  them (cosine ≈ 1) rather than silently overwriting divergent values.

Rationale: Georg placed these manually; codifying the recipe makes a fresh RunPod
self-contained and creates the `outlier` run-dir if missing.

## 5. Unit B — regeneration + correctness gate (GPU, RunPod 48GB float32)

`scripts/emnlp_perm_edit/runpod_gemma_variants.sh` (tmux launcher, mirrors the Qwen
orchestration we already built). **Prerequisite:** vendored circuit-tracer fork installed
editable (`uv pip install -e vendor/circuit-tracer`, branch
`refusal-lens-multi-position-fix`) — required for `hook_resid_post`; the stock pip wheel
measures at the wrong basis.

Per variant `v in {complement, full, outlier}`:
1. **Attribution:** `python scripts/pipeline/02_run_attribution.py --run-dir
   data/results/pipeline_runs/gemma_var_<v> --n-prompts 50 --skip-multi-graph
   --target-layer 15 --single-position-target -2 --measurement-hook hook_resid_post
   --backend transformerlens --dtype float32 --batch-size 128 --save-graphs --resume`
   → 550 `.pt` graphs (50×11) in `02_attribution/graphs/`.
   (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid allocator thrash.)
2. **Correctness gate** *(the key safety net)*: read the new
   `02_attribution/attribution_results.json` `net` (= `all_signed`) per (prompt, condition)
   and compare against the committed
   `data/results/emnlp_perm_edit/phase0_controllability/gemma_var_nets.json`
   (complement ≈ +909, full ≈ −48k, outlier ≈ −55k on `bare`). Tolerance check; if it
   doesn't reproduce, **abort that variant** (signals fork-not-installed / wrong hook /
   wrong direction) and record the failure.
3. **Pack + overlap-annotate + stage:** convert `.pt` → gzipped JSON and write
   overlap buckets (`annotate_overlap_3way`: bare / shared_with_bare / jb_unique / …) into
   the nodes, staged into `gemma_var_<v>/05_frontend/`. Flow:
   `02c_pack_graphs.py --run-dir gemma_var_<v>` then
   `05_visualize_circuits.py --run-dir gemma_var_<v> --mode single --skip-convert
   --skip-subcircuits --gzip` (the implementation plan confirms whether 05 re-converts or
   annotates the 02c output in place; either way the **annotated** `graph_data` lands in
   `05_frontend/`). `--skip-subcircuits` because variants have no Stage 07.
4. **Push:** `python scripts/pipeline/push_graph_data.py --run-dir gemma_var_<v>
   --source 05_frontend --dataset-repo moon70/refusal-lens-graphs` to HF run-id
   `run_gemma_<v>_L15`. Push from `05_frontend` (the **annotated** copy), not `02c`. The HF
   run-id must be `run_gemma_<v>_L15`, not the run-dir basename `gemma_var_<v>` — pass an
   explicit `--run-name run_gemma_<v>_L15` (add the flag to `push_graph_data.py` if it
   currently derives the run-id from the run-dir basename).
5. **Purge `.pt`** for that variant to bound disk, then continue.

A **watcher** window (`watch_and_commit_gemma_variants.sh`) commits the small committed
artifacts (attribution summaries, nets, packed `graph-metadata.json`) to the branch and
confirms each HF upload on its DONE marker.

**Compute/cost:** 3 variants × 550 = 1,650 graphs; Gemma's gemma-scope transcoders are far
lighter than Qwen's 160k-wide ones → ~3–5 h total, ~$2–6 on a 48 GB card (A40/A6000).
**Disk:** pack-then-purge keeps `.pt` peak ~110 GB → **150 GB network volume** at
`/workspace` (`HF_HOME=/workspace/hf`).

## 6. Unit C — compare frontend assembler (CPU)

`scripts/pipeline/assemble_compare_frontend.py`. Driven by a small config
(`scripts/pipeline/05_frontend_patches/compare_config.json`) listing the columns:

```json
{ "title": "Refusal circuit: Gemma variants vs Qwen",
  "dataset_repo": "moon70/refusal-lens-graphs",
  "columns": [
    {"label": "Gemma · complement (no #443)", "run": "run_gemma_complement_L15", "model": "gemma", "target": "complement"},
    {"label": "Gemma · full (+#443)",         "run": "run_gemma_full_L15",       "model": "gemma", "target": "full"},
    {"label": "Gemma · outlier #443 only",     "run": "run_gemma_outlier_L15",    "model": "gemma", "target": "outlier"},
    {"label": "Qwen (L18)",                    "run": "run_emnlp_qwen_L18_20260522","model": "qwen","target": "full"}
  ] }
```

Steps:
1. For each column, fetch its run via `fetch_graph_data.py` (`snapshot_download` +
   `stage_frontend`), producing a self-contained viewer at `compare/<run>/05_frontend/`
   (`index.html`, `graph_data/`, `data/graph-metadata.json`, patches).
2. Build **`compare/compare_manifest.json`**:
   - `columns`: `[{label, dir, model, target}]` in order, where `dir` is the served viewer
     root **`<run>/05_frontend`** (so the iframe src is `./<dir>/index.html?slug=…`).
   - `prompts`: prompt indices present in **all** columns (intersection), with prompt text.
   - `conditions`: conditions present (per prompt) in all columns.
   - `slugmaps`: per-column `{ "<idx>_<cond>": "<actual slug>" }` resolved from that
     column's `graph-metadata.json` (handles any `_single`/`_multi` suffix differences
     without client-side guessing; a missing cell maps to `null`).
3. Stage `compare/compare.html` (Unit D) and the `compact-mode` patch (see Unit D) at the
   `compare/` root and into each column.

Output: `data/results/compare_3way/compare/` — a static site servable as one root.

## 7. Unit D — N-column compare harness

`scripts/pipeline/05_frontend_patches/compare_multi.html` (staged as `compare/compare.html`).

- Loads `./compare_manifest.json`.
- Top bar: **prompt** dropdown (from `prompts`) + **condition** dropdown (from `conditions`).
- Renders **one column per `columns[]` entry**, each an
  `<iframe src="./<dir>/index.html?slug=<slugmaps[col][idx_cond]>&compact=1">`.
  A `null` slug renders a "no graph for this cell" placeholder instead of a broken iframe.
- On picker change, recompute `idx_cond` and update every iframe `src`.
- Column header = `columns[].label`. Responsive: min column width with horizontal scroll
  when the viewport is narrow (4 full viewers is wide by design).

**Compact mode** — to get the stacked "graph → I/O attribution → top activations" look
inside a narrow column without editing the vendored submodule, stage a patch
(`compact-mode.css` + tiny `compact-mode.js`, injected into each column's `index.html` the
same way `overlap-colors.css`/`gzip-fetch.js` are injected by `stage_frontend`). When
`?compact=1` is present it: hides the per-iframe graph selector/header chrome (the parent
bar drives selection) and applies a single-column flex layout so the viewer's panels stack
vertically to fill the column.

**Risk (the one real unknown):** whether the vendored viewer's panels reflow cleanly into a
narrow stacked column via CSS alone. Plan A is the `compact-mode` patch (low coupling,
reuses the whole viewer). Fallback (deferred unless A renders poorly): a thin custom column
that mounts the three circuit-tracer subviews (`init-cg`, node-connections, feature-detail)
directly, stacked. We validate Plan A in the visual smoke (§8) before committing to it.

## 8. Testing

- **CPU unit tests** (`scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py`):
  - Unit A: `outlier_dim==443`, `complement[443]==0`, outlier-only-nonzero-at-443, norm
    ratios match `gemma_outlier_split_stats.json`.
  - Manifest builder: shared `(idx,cond)` intersection across mocked metadata, slug
    resolution, `null` for missing cells.
- **GPU smoke** (`scripts/emnlp_perm_edit/smoke_test_gemma_variants.sh`): 2 prompts × the
  `complement` variant end-to-end (attribution → nets gate on a 2-prompt tolerance →
  02c pack → push `--dry-run`) into a throwaway `/tmp` run-dir. Must pass before the full run.
- **Visual smoke:** assemble a tiny compare site (the 2-prompt Gemma smoke run + Qwen),
  serve, confirm all columns load matched slugs and the `compact=1` stack renders legibly.

## 9. Rollout

1. (local CPU) Unit A → variant directions present + self-check green.
2. (local CPU) unit tests green.
3. (RunPod) install fork, preflight (HF read/write probe on `moon70/refusal-lens-graphs`,
   git push dry-run), GPU smoke green.
4. (RunPod) full run B → 3 Gemma runs on HF + nets gate green for each.
5. (local CPU) Unit C assemble + Unit D serve → 4-column comparison live.
6. Setup doc `docs/RUNPOD_GEMMA_VARIANTS_SETUP.md` (SSH→env→preflight→smoke→launch,
   adapted from the Qwen setup doc).

## 10. File inventory

| file | unit | what |
|---|---|---|
| `scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py` | A | deterministic variant directions into run-dirs |
| `scripts/emnlp_perm_edit/runpod_gemma_variants.sh` | B | per-variant attribution → gate → pack → annotate → push → purge |
| `scripts/emnlp_perm_edit/watch_and_commit_gemma_variants.sh` | B | watcher: commit artifacts + confirm HF upload |
| `scripts/emnlp_perm_edit/smoke_test_gemma_variants.sh` | B | 2-prompt GPU smoke |
| `scripts/pipeline/assemble_compare_frontend.py` | C | fetch 4 runs + build `compare_manifest.json` + stage harness |
| `scripts/pipeline/05_frontend_patches/compare_config.json` | C | the 4-column config |
| `scripts/pipeline/05_frontend_patches/compare_multi.html` | D | manifest-driven N-column harness |
| `scripts/pipeline/05_frontend_patches/compact-mode.{css,js}` | D | `?compact=1` column-stack patch |
| `scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py` | — | CPU unit tests |
| `docs/RUNPOD_GEMMA_VARIANTS_SETUP.md` | — | RunPod setup runbook |
