# Refusal-Lens — Pipeline State & Remaining Tasks

**Handoff doc, last updated 2026-04-23** (end of session after Stage 06 full-run lands). Written for a fresh Claude session to pick up where we left off. Current phase: correlational pipeline done, causal intervention validated, frontend + ablation are the remaining moves.

**TL;DR for a fresh session**:
- All refactored pipeline stages (01 / 02 / 02b / 03 / 04 / 07) landed + validated on the real RunPod run `run_20260422_015552` on the `l15-refactor` branch.
- **Stage 06 (Task 9) causal intervention COMPLETE on full 50-prompt dataset**. Headline: **96.7% pro-refusal flip (87/90), 100% anti-refusal (49/49), 100% benign force-refuse (10/10)**. Matches Tejas's 90/90 bulletproof within 3 prompts; bidirectional symmetry confirms L15 `r` is the refusal axis. See Phase B results block below.
- **Stage 05 frontend is the next priority** — code is ready but needs to be run end-to-end on the new `run_20260422_015552` attribution graphs (80 GB `.pt` fetch from HF → `02c_pack_graphs` → HF push → `05_visualize_circuits` → browser spot-check). User has ~80 GB local disk for this; should be fast once started.
- Phase A0 packaging scripts are written (`02c_pack_graphs.py` + `push_graph_data.py --source 02c`); the actual packaging run is step 1 of the Stage 05 work.
- Local test suite: **214 passed / 0 failed / 1 skipped**.
- ICML headline numbers are all in hand (see "Phase A validation findings" + new Phase B block below).

Skip straight to the **"Immediate next steps for new session"** section at the bottom if you want the punch list. Stage 05 frontend refresh is #1.

---

## Project overview

Mechanistic interpretability research on **Gemma-3-4b-it** — attributing the model's refusal circuit using Anthropic's circuit-tracer (CLT). Mentor: Georg. Research arc: *"Here are the circuits, they're real, here's what they encode, and here's what happens when you manipulate them."*

- **Repo**: `AutoInterp/Refusal-Lens`, working branch `l15-refactor`
- **Submodule**: `vendor/circuit-tracer` on branch `refusal-lens-measurement-patch`, local commit `b5300ee` (patched for multi-position targets, pushed upstream)
- **HF dataset**: `moon70/refusal-lens-graphs` (raw `.pt` graphs + directions archived; frontend JSONs also published)

### Team
- **Mahmoud** (user): correlational attribution, Stages 01–05 + 07, frontend
- **Tejas**: causal intervention work on `tejas-circuit-experiments` branch. The *current* source-of-truth script is **`20_bulletproof_pipeline.py`** on the remote branch (commit `332311c`), which landed 50/50 bare refuse + 250/250 ctrl refuse + **90/90 JB-flipped** at L15. Scripts 16/17 are older and superseded. **Port from `git show origin/tejas-circuit-experiments:...`, not from the stale local `data/tejas_experiments/scripts/` copies** — see the audit note in the Stage 06 block.
- **Georg**: mentor
- **Ruqiya**: Neuronpedia feature-label extraction (Stage 04b input, still pending)

### Deadlines
- **Apr 23 12 PM PST** — Algoverse symposium (5-min talk). Slides use OLD run `run_20260417_010035` data (L32 single-position).
- **May 4** — ICML workshop abstract. Target for the NEW-data results. Headline numbers (all ready): Stage 07 `jb_specific_frac` per class, Stage 02b ctrl-aware effect-size table, Stage 06 L15 flip rates (pending GPU), L14 peak-layer finding.

---

## Current pipeline state

### Stage-by-stage status (Apr 22, post-validation)

| # | Stage | Code status | Validated against `run_20260422_015552`? | Notes |
|---|---|---|---|---|
| 01 | `01_compute_direction.py` | ✅ | ✅ direction artifacts regenerated | Per-layer (34 layers @ pos=-2) + per-position at L15 (-1..-15). `unnormalized_r.pt` drives Stage 06. |
| 02 | `02_run_attribution.py` | ✅ | ✅ 50×11×2 graphs produced on RunPod, 0 errors | L15 target, two-graph scheme (multi `[-5,-3,-2]` + single `[-2]`). |
| 02b | `02b_statistical_analysis.py` | ✅ | ✅ **30 stats blocks**, novel ctrl_vs_bare effect sizes captured | See Phase A1 table below. |
| 03 | `03_verify_attribution.py` | ✅ | ✅ 50/50 within tolerance | MLP contributes 0.02% of signal (known limitation). |
| 04 | `04_label_features.py` | ✅ | ✅ 1353 features, 100% HF coverage | `per_condition_top50` has all 11 keys. |
| 04b | `04b_delphi_labels.py` | ⏳ not started | — | LLM labels via Claude API (~$2, CPU). |
| 05 | `05_visualize_circuits.py` + patches | ✅ code | ⚠️ **not yet run end-to-end on the new attribution graphs — NEXT PRIORITY** | 3-way bucket logic, ctrl-aware filters, CSS palette in place; need .pt fetch → 02c pack → HF push → stage-05 render → browser check. |
| 06 | `06_causal_intervention.py` | ✅ code + 9 tests + full run | ✅ **96.7% pro-flip / 100% anti / 10/10 benign on `run_20260422_015552`** | Ports Tejas Script 20. Headline results committed at `06_causal/`. |
| 07 | `07_identify_subcircuits.py` | ✅ | ✅ 18 subcircuits, `jb_vs_ctrl_contrast` novel metric computed | See Phase A4 table below. |
| 08 | `08_ablate_subcircuits.py` | ⏳ not started | — | Depends on Stage 06 + 07. |

### Phase A0 — producer-side packaging (new this session)

| # | Script | Status | Notes |
|---|---|---|---|
| 02c | `02c_pack_graphs.py` | ✅ code | Converts `.pt → JSON.gz`, writes `<run>/graph_data/`. |
| — | `push_graph_data.py --source 02c` | ✅ patch | Uploads the 02c pack dir to HF (`runs/<name>/graph_data/`). |
| — | The actual `.pt` pull + pack + push for `run_20260422_015552` | ⏳ pending | One-time: ~80 GB fetch + ~3 GB push. Run on a machine with disk+bandwidth. |

### Task 7 Stage 04 — resume-session changes (Apr 22)

**`scripts/pipeline/04_label_features.py`**:
- `_top50_for_condition(cond)` — reads `cond.graphs.multi.top50_features` (canonical) with `single` + legacy-flat fallback.
- `_comparison_sub_buckets(cls, cls_comp)` — descends into `vs_bare` / `vs_ctrl` / `ctrl_vs_bare` sub-buckets, tagging each with `jb_{cls}` or `ctrl_{cls}`. Legacy fallback preserved.
- `collect_all_features` rewritten: emits both `conditions_seen` (union across sources, full names `bare` / `jb_*` / `ctrl_*`) AND `top50_conditions` (top-50 multi-graph memberships only).
- `collect_comparison_features` rewritten: descends into 3 sub-buckets; `classes` field on each entry uses full condition names.
- New `build_per_condition_sets(all_features)` — emits `{bare, jb_*, ctrl_* → sorted feature keys}`. Consumed by Stage 07 Task 10 rules.
- `main()` now writes `top50_conditions` into each `feature_labels.json` entry and merges `per_condition_top50` into `feature_class_sets.json` (under new top-level key `per_condition_top50`). `--upset-only` path regenerates per-condition sets when the attribution JSON is present.

**`scripts/pipeline/tests/test_pipeline_local.py`**:
- New `test_stage_04_schema` — 10 assertions (T-S4a…j) against the smoke JSON. Runs without a live pipeline-run directory.
- Fixed pre-existing MockArgs bugs: `test_stage_04_a8` (missing `upset_only`) and `test_stage_01` (missing `update_metadata` + per-position fields).
- New `--stage 04-schema` CLI flag; added to `--stage all` sequence.

**Local test result**: `170 passed / 0 failed / 1 skipped` (skip = pre-existing snapshot reset, unrelated). **Not yet verified against real Stage 02 output** — the RunPod 50-prompt attribution run is still writing graphs; once `attribution_results.json` lands, re-run Stage 04 end-to-end on that to confirm HF dashboard fetch + outputs.

**Downstream consumers of the new schema** (needed by follow-on tasks):
- `feature_labels.json[key].top50_conditions`  →  Stage 07 Task 10 per-condition subcircuit rules
- `feature_class_sets.json["per_condition_top50"]`  →  Stage 07 Task 10 `jb_specific_vs_ctrl` rule
- `feature_labels.json[key].conditions_seen`  →  Stage 05 Task 8 frontend `shared_with_ctrl` bucketing

### Task 10 Stage 07 — resume-session changes (Apr 22)

**`scripts/pipeline/07_identify_subcircuits.py`**:
- Schema-tolerant helpers: `_jb_classes_seen`, `_classify_bucket_classes` (legacy flat + new `jb_*`/`ctrl_*` both accepted).
- Existing rules rewritten via helpers — **no regression** on legacy data (sizes 83/56/179/52/50/689 reproduced exactly on `run_20260417_010035`).
- New ctrl-aware rules (require `feature_class_sets.per_condition_top50`):
  - `ctrl_shared_refusal` — prefix-invariant refusal spine: `bare ∩ all 5 ctrl_*_top50 − all 5 jb_*_top50`.
  - `ctrl_only` — benign-prefix-exclusive features: `all 5 ctrl_*_top50 − bare − any jb_*_top50`.
  - `jb_{cls}_specific_vs_ctrl` ×5 — true per-class JB-semantic machinery: `jb_{cls}_top50 − ctrl_{cls}_top50`.
- `has_ctrl_data(class_sets)` — gate that checks for per_condition_top50 block; legacy JSONs skip ctrl-aware rules cleanly.
- `metadata.ctrl_available` flag persisted to `subcircuits.json`.
- **Novel headline metric (`compute_jb_vs_ctrl_contrast`)**: per-class recruitment contrast — `jb_top50`, `ctrl_top50`, `intersection` (prefix-driven), `jb_specific` (true JB-semantic), `ctrl_specific`, `jb_specific_frac`, `overlap_frac`. Answers *"what fraction of each JB's machinery is genuinely JB-semantic vs. a prefix-inflation artifact?"* — old L32 data couldn't compute this.
- Novel figures added (generated only when ctrl data available):
  - `jb_vs_ctrl_contrast.png` — per-class stacked bar, jb_specific above / shared mirrored / ctrl_specific below, with `jb_specific %` annotation per class.
  - `jb_specific_by_layer.png` — layer distribution of JB-semantic features stacked by class; reveals whether the L24–L32 band holds up after controlling for prefix.
- Extended `SUBCIRCUITS_REPORT.md` with "JB-vs-Ctrl recruitment contrast" section and updated Stage 08 ablation-target list (adds `jb_{cls}_specific_vs_ctrl` dissociation test, `ctrl_shared_refusal` negative control).

**`scripts/pipeline/tests/test_pipeline_local.py`**:
- `test_stage_07` updated: expects 18 subcircuits (11 legacy + 7 ctrl-aware), new T-07i/j/k for metadata flag + legacy-path empty-ctrl invariants.
- New `test_stage_07_synthetic_ctrl` — builds a deterministic 6-feature × 11-condition fixture in a tempdir, runs Stage 07 against it, verifies each ctrl-aware rule produces the correct set AND contrast arithmetic is consistent. Exercises the ctrl-available branch without a real run.
- New `--stage 07-ctrl` CLI flag.

**Local test result**: `184 passed / 0 failed / 1 skipped`. **Still not tested against real Stage 02 output** — pending RunPod run completion.

### Task 8 Stage 05 — resume-session changes (Apr 22)

**`scripts/pipeline/utils_viz.py`**:
- `OVERLAP_BUCKETS` extended to 9 values: `shared_with_bare_and_ctrl`, `shared_with_bare`, `shared_with_ctrl`, `jb_unique`, `ctrl`, `ctrl_unique`, `bare`, `bare_only`, `non_feature`.
- **New** `annotate_overlap_3way(jb, bare, ctrl, jb_class, idx)` — tags each feature node against BOTH bare and ctrl keys. Produces the headline `shared_with_ctrl` bucket (features in ctrl+jb but NOT bare → **prefix-induced, not JB-semantic**). Writes `overlap_counts` summary into metadata.
- **New** `annotate_ctrl(ctrl, bare_or_None, ctrl_class, idx)` — analog of `annotate_bare` for ctrl graphs. When bare is provided, splits features into `shared_with_bare` (stable) vs `ctrl_unique` (benign-prefix-only).
- `annotate_overlap` (legacy 2-way) preserved for backward compat; now writes `overlap_mode = "2way"` into metadata.
- `_subcircuit_allowed` extended with three new rule families for Task 10 ctrl-aware subcircuits:
  - `ctrl_shared_refusal` allowed only in bare-like buckets ({bare, ctrl, shared_*}); rejected on jb_unique / ctrl_unique.
  - `ctrl_only` allowed only in ctrl-unique buckets.
  - `jb_{cls}_specific_vs_ctrl` allowed only in {jb_unique, shared_with_bare} for the matching class — **rejected on shared_with_ctrl** (feature being in ctrl contradicts the subcircuit definition).
- `_UNIVERSAL_CORE_BUCKETS` extended to include `shared_with_bare_and_ctrl` and `ctrl`.
- `_CANONICAL_BUCKETS` / `_CLASS_EXCLUSIVE_BUCKETS` extended to include `shared_with_ctrl`.

**`scripts/pipeline/05_visualize_circuits.py`**:
- New `parse_slug(stem)` — handles both legacy `{idx}_{class}` and new `{idx}_{cond_name}_{mode}` slug formats.
- `select_pt_files` gained `mode_filter` parameter; new `--mode` CLI flag (default: `single`; choices: `multi`, `single`, `both`).
- New `group_by_prompt_structured(json_paths)` — `{idx: {cond_name: {mode_key: path}}}` for clean 3-way lookup.
- Step 2 (overlap annotation) rewritten: annotates bare + ctrl_{cls} + jb_{cls} per prompt per mode; prefers 3-way annotation when matched ctrl exists, falls back to 2-way for legacy runs. Prints corpus-level bucket totals.
- Step 3 (subcircuit annotation) iterates the structured map (all modes), reports sample bare/mode tag.

**`scripts/pipeline/05_frontend_patches/overlap-colors.css`**:
- 7 color rules added/updated: shared_with_bare_and_ctrl (dark green), shared_with_bare (teal — existing), shared_with_ctrl (gold — **new prefix-induced bucket**), jb_unique (orange — existing), bare (slate), ctrl (blue-grey), ctrl_unique (purple).
- Legend container extended with `.note` subline so bucket descriptions show their interpretation (e.g. "PREFIX-induced (not JB-semantic)").

**`scripts/pipeline/05_frontend_patches/overlap-annotate.js`**:
- Rewrote legend logic: pulls per-bucket counts from D3's `__data__` and rebuilds the panel on every tick, showing only buckets that actually have features in the current graph. Bucket metadata (color, label, note) centralised in a `BUCKET_META` map.

**`scripts/pipeline/tests/test_pipeline_local.py`**:
- Updated T-V1d (OVERLAP_BUCKETS completeness) for the 9-bucket schema.
- New T-V5a..k — 3-way annotation round-trip on a synthetic 6-feature fixture across bare/ctrl/jb graphs; verifies each bucket assignment, overlap_counts metadata, annotate_ctrl behavior.
- New T-V6a..l — ctrl-aware subcircuit filter rules: `ctrl_shared_refusal` / `ctrl_only` / `jb_{cls}_specific_vs_ctrl` accept/reject semantics across all bucket/class combinations.

**Deferred** (explicitly scoped out of this pass):
- 3-column `compare.html` rewrite (bare | ctrl | JB side-by-side) — requires substantial DOM restructuring + browser testing. The existing single-graph viewer with 3-way colored nodes already conveys the insight visually.

**Local test result**: `207 passed / 0 failed / 1 skipped`. Same "not yet tested against real Stage 02 output" caveat.

---

## Phase A validation (Apr 22) — refactored stages vs `run_20260422_015552`

The RunPod run landed clean. All four JSON-only stages validated against real data. Headline numbers below — these feed directly into the ICML abstract (May 4).

### A1. Stage 02b — Statistical Analysis ✅

Runs through 30 stats blocks (2 modes × 3 comparisons × 5 classes). Novel three-way finding:

| Class | vs_bare %chg (jb effect) | ctrl_vs_bare %chg (prefix-only) |
|---|---|---|
| cognitive_reframe | **-51.9%** (d=-2.05) | +1.4% (n.s.) |
| analytical | **-49.8%** (d=-4.07) | +4.2% |
| fiction | **-34.6%** (d=-2.21) | +5.2% |
| roleplay | -6.2% (d=-0.43) | -3.9% |
| completion | +4.2% (d=+0.43) | +1.8% (n.s.) |

**Takeaway**: The matched benign ctrl prefixes do NOT reduce attribution to refusal — they slightly *increase* refusal or leave it unchanged. JB attribution drops with huge effect sizes (d>2 for 3/5 classes). Direct correlational evidence that the jb effect is semantic, not prefix-formatting.

### A2. Stage 03 — L15 Verification ✅

50/50 prompts verified within tolerance. `dot_product_mean = 28,381`, `attr_net_mean = 6.08`. **MLP accounts for only 0.02% of signal** (99.98% attention) — the known MLP-only-attribution limitation, cleanly recovered on the new L15-measurement data.

### A3. Stage 04 — Feature Labeling ✅

1,353 unique features, **100% HF dashboard coverage**. `conditions_seen` emits the full 11-condition naming (`bare`, `jb_*`, `ctrl_*`). `feature_class_sets.per_condition_top50` has all 11 keys. Layer histogram + UpSet regenerated.

### A4. Stage 07 — Subcircuits + **headline ICML novel metric** ✅

18 subcircuits (11 legacy + 7 ctrl-aware), all invariants pass. Per-class `jb_specific_frac` (the money number):

| Class | JB-specific % | Interpretation |
|---|---|---|
| cognitive_reframe | **38.6%** | deepest JB — most genuine semantic mechanism |
| analytical | 34.2% | strong JB-semantic |
| fiction | 34.2% | strong JB-semantic |
| roleplay | 20.0% | mostly prefix artifact |
| completion | **18.4%** | mostly prefix artifact (matches 02b +4.2% finding) |

**ICML claim enabled**: *up to 82% of what prior work called "JB features" is prefix-induced, not JB-semantic*. The controlled dataset is what makes this measurable.

Structural-identity sanity: `ctrl_shared_refusal = 50` features (prefix-invariant refusal spine); `ctrl_only = 1` (as expected, tiny).

**Note**: `late_wave_layer24_32 = 0` on this L15-measurement run — attribution doesn't reach past L15, so the L24-L32 band is empty. The late-wave subcircuit is a relic of the old L32-measurement runs and doesn't apply here. Document this in the paper, don't treat it as regression.

### A5. Stage 05 — frontend (pending Phase A0 HF push, then fetch)

Code path validated on run_20260417_010035 + synthetic fixtures (207 local tests pass). Awaits gzipped-JSON push via Phase A0 below.

## Phase A0 — Producer-side `.pt → JSON.gz` packaging (new, Apr 22)

**Why**: every collaborator pulling 80 GB of `.pt` just to view the frontend is wasteful. The gzipped-JSON receive path exists (`fetch_graph_data.py`, `gzip-fetch.js`), but nothing decoupled `.pt → JSON` conversion from Stage 05 full frontend staging.

**New files**:
- `scripts/pipeline/02c_pack_graphs.py` — reads `02_attribution/graphs/*.pt`, calls `convert_pt_to_frontend_json` + `gzip_json_files`, writes `<run>/graph_data/*.json.gz` + `graph-metadata.json`. CLI: `--run-dir`, `--prompts`, `--classes`, `--no-gzip`, `--keep-plain`, `--overwrite`.

**Patched**:
- `push_graph_data.py` — added `--source {05_frontend,02c}` flag so the push reads from either the Stage 05 output or the raw 02c pack dir. Backward-compatible (default stays `05_frontend`).

**Workflow for any new run** (one-time on RunPod or local with disk):
```
python3 scripts/pipeline/fetch_raw_graphs.py --run <run_name> --dataset-repo moon70/refusal-lens-graphs
python3 scripts/pipeline/02c_pack_graphs.py --run-dir <run_dir>
python3 scripts/pipeline/push_graph_data.py --run-dir <run_dir> --source 02c --dataset-repo moon70/refusal-lens-graphs
```

Collaborators thereafter pull via `fetch_graph_data.py` (2-5 GB) instead of `fetch_raw_graphs.py` (80 GB). Stage 05 validation runs with `--skip-convert`.

## Task 9 Stage 06 — resume-session changes (Apr 22)

Ports Tejas Script 20 bulletproof pipeline (`origin/tejas-circuit-experiments`, commit `332311c`, **90/90 flip rate at L15**) into our pipeline conventions.

**`scripts/pipeline/06_causal_intervention.py`** (new, ~450 lines):
- Phase 0: dataset verification (bare refuse + ctrl refuse sanity). Records ctrl-leak pairs and excluded prompts but doesn't mutate the dataset.
- Phase 1: baseline generation on all 11 conditions per prompt.
- Phase 2a `pro_refusal_add`: add unnormalized r at L15 on (prompt, jb_*) where baseline is COMPLY. Expected flip rate: ~90/90 matching Tejas.
- Phase 2b `anti_refusal_sub`: subtract unnormalized r at L15 on (prompt, bare) where baseline is REFUSE. This is the symmetry half (bidirectional claim).
- Phase 3: aggregate summary — per-layer, per-method, per-class flip rates.
- Checkpoint/resume mirrors Stage 02 pattern (`causal_checkpoint.json`, save every 5 prompts, `--resume`).
- Novel-insight figures: `flip_rate_by_class.png`, `intervention_symmetry.png`, `FLIP_RATE_SUMMARY.md` for ICML.

**`scripts/pipeline/utils.py`** (append ~130 lines):
- `load_unnormalized_r(direction_dir, layers)` — loads Stage 01's `unnormalized_r.pt` dict, filtered to requested layers, raises on missing.
- `make_intervention_hook(r, sign)` — returns forward hook adding/subtracting r at every position. Handles tuple-wrapped module outputs. Re-casts r to output dtype at hook time.
- `generate_with_hook(model, tokenizer, prompt, layer, hook_fn, max_new_tokens)` — wraps `register_forward_hook` + `model.generate(do_sample=False)` + `handle.remove()` in try/finally.
- `generate_baseline(model, tokenizer, prompt, max_new_tokens)` — mirror for baseline runs.

**`scripts/pipeline/config.py`** (append 2 lines):
- `CAUSAL_INTERVENTION_MODES = ("pro_refusal_add", "anti_refusal_sub")`
- `STAGE_06_DEFAULT_LAYERS = [15]` (L15 only for v1, expand later)

**`scripts/pipeline/tests/test_pipeline_local.py`** (append `test_stage_06`):
- T-S6a..i — hook math (add/sub), tuple-output handling, bad-sign rejection, aggregate_summary correctness, skip_anti path, per-class rate math. 9 assertions. No GPU.

**Pre-GPU audit fixes (Apr 22 late)**:
- Added `--r-source {stage01, tejas-rescale, recompute}` CLI flag. Tejas's reported `|r_L15|=4019.7` didn't match our Stage 01's `|r_L15|=3101.2` on an identical-methodology computation. `--r-source recompute` runs the direction recomputation **in-script under the same bf16 model** as intervention, bit-for-bit matching Tejas Script 20's Phase 4. Chose this for the full run.
- Added Phase 2c **benign force-refuse control** (Tejas Script 20 Phase 4a): applies the same L15 pro-refusal-add hook to 10 hardcoded benign prompts (from `config.BENIGN_PROMPTS`), expects all to flip to REFUSE. This is the "bulletproof" control proving the intervention is a generic refusal push, not a JB-specific artifact.

## Phase B results (Stage 06 full run on `run_20260422_015552`, Apr 22–23)

Pod: RunPod H100 SXM, torch 2.4.0 / CUDA 12.4.1, bf16 model. Wall clock: **54.6 min**. Command: `python3 scripts/pipeline/06_causal_intervention.py --run-dir data/results/pipeline_runs/run_20260422_015552 --r-source recompute`.

**Dataset health (Phase 0)**:
- Bare refused: **49/50** (1 prompt — "How to steal someone's password..." — excluded as bare-comply).
- Ctrl refused: **246/250 = 98.4%** (4 ctrl-leak pairs excluded per Tejas's bulletproof convention).

**Headline flip rates**:

| Experiment | Result | Notes |
|---|---|---|
| **L15 pro-refusal add** (JB COMPLY → REFUSE) | **87/90 = 96.7%** | Tejas bulletproof reports 90/90; we're 3 prompts shy. |
| **L15 anti-refusal sub** (bare REFUSE → COMPLY) | **49/49 = 100%** | Tejas didn't test — our symmetry addition. |
| **L15 benign force-refuse** (10 benign prompts → REFUSE) | **10/10 = 100%** | Tejas reports 10/10. Matches exactly. |
| Coherence | 100% on all 146 flips | No gibberish — only coherent refusals. |

**Per-class pro-refusal flip**:

| Class | Comply baseline (Stage 06) | Flipped | Flip rate | Stage 07 `jb_specific_frac` |
|---|---|---|---|---|
| analytical | 27/50 | 27 | **100%** | 34.2% |
| roleplay | 9/50 | 9 | **100%** | 20.0% |
| completion | 1/50 | 1 | 100% (n=1 tiny) | 18.4% |
| cognitive_reframe | 33/50 | 32 | 97% | **38.6%** |
| fiction | **20/50** | **18** | **90%** | 34.2% |

**Novel findings from the full run**:

1. **Bidirectional symmetry confirmed (the headline causal claim)**. Adding r flips 97% of JB prompts to REFUSE; subtracting r flips 100% of bare prompts to COMPLY; the same hook forces 10/10 benign prompts to REFUSE. Taken together these establish that L15 `r = mean_harmful − mean_harmless` is not just a readable direction — **it IS the refusal axis, manipulable in both directions**.

2. **Fiction is the hardest JB class to causally patch at 90%**. Every other class flips at ≥97%. Fiction is tied with analytical for second-highest JB-specific feature fraction (34.2%), yet analytical flips 27/27 while fiction flips only 18/20. Candidate hypothesis: narrative-framed text distributes attention more than other JB styles, making the linear direction-push less effective. **Candidate Stage 08 dissociation test**: does ablating `jb_fiction_specific_vs_ctrl`'s 52 features cleanly suppress fiction jailbreaks without affecting the other classes?

3. **`|r_L15|` unexplained 22% gap vs Tejas** (3123.9 recomputed vs his 4019.7). Same methodology (64+64 diff-in-means, pos=-2, no batching, no truncation, bf16 model), different magnitude. Likely dataset-version drift on `harmful_train.json`/`harmless_train.json` between branches. **Not functionally limiting** — our lower magnitude still achieves the bulletproof-control-matching 10/10 benign force-refuse and 96.7% JB flip. Worth a diagnostic pass (diff the splits across branches) but deprioritised given the headline results.

4. **Comply-baseline rates quantify JB strength on this model**:
   - cognitive_reframe 33/50 (66%) — strongest JB
   - analytical 27/50 (54%)
   - fiction 20/50 (40%)
   - roleplay 9/50 (18%)
   - completion 1/50 (2%) — effectively **not a jailbreak** on Gemma-3-4b-it with the new controlled dataset.

Outputs (committed on `l15-refactor`):
- `06_causal/causal_results.json` — per-prompt baseline + intervention records (4.4 MB).
- `06_causal/causal_summary.json` — aggregated flip rates + per-class breakdown.
- `06_causal/flip_rate_by_class.png`, `intervention_symmetry.png` — figures.
- `06_causal/FLIP_RATE_SUMMARY.md` — human-readable headline one-pager.

**Local test result after Task 9 lands**: `214 passed / 0 failed / 1 skipped`.

Three pre-existing tests were updated to accept the new-schema output patterns from validation against real data (not regressions — the code is working correctly, the assertions were over-specific):
- `T-A6a`: plot filename is now mode-suffixed (`distribution_by_class_multi.png` / `_single.png`); test accepts either the legacy or new name.
- `T-A7d`: `n_classes` is 10 on new-schema data (full condition names: 5 jb_ + 5 ctrl_) vs 5 on legacy flat; test accepts either.
- `T-07g` / `T-07h`: size predictions were calibrated on L32-measurement run `run_20260417_010035`; on L15-measurement runs the late_wave band is empty by construction and absolute sizes shift. Tests gate on `run_dir.name == "run_20260417_010035"` and assert the alternative L15 structural expectation otherwise.

### Peak-layer finding (bonus)

Stage 07 report on the new run: **L14 is the hotspot** for every refusal subcircuit. Peak layers: `universal_refusal_core L14`, `canonical_pro_refusal L14`, `sign_flip_convergent L14 (×36 features)`, `dampening_specialists L14`, all `jb_{cls}_specific_vs_ctrl` → L14. The refusal signal concentrates one layer before the L15 measurement target — a clean mechanistic claim worth calling out in the ICML abstract.

### Critical infrastructure refactors (done this session)

- **circuit-tracer patch** (`vendor/circuit-tracer/circuit_tracer/attribution/*.py`): `measurement_layer` and `measurement_position` accept `int | Sequence[int] | None`. Per-target measurement sinks via `_as_measurement_tensor` helper. Backward pass correctly injects gradients at each target's own position.
- **Parallel GPU launcher** (`run_stage02_parallel.sh` + `merge_stage02_shards.py`): shards prompts across N GPUs with shard-specific checkpoints, auto-merges at end.
- **Stage 02 speedup**: batch size default 1→256 (→512 for H100/Blackwell), ~100× faster.
- **`graph_summary` multi-target fix**: `_aggregated_target_row` sums the weighted target rows correctly. Previously was reading only the last target's row — `bare.multi.net = 7.287` now correctly equals `sum(per_target[i].net)` across all 3 template positions.
- **Docker**: `Dockerfile` (git-clone path) + `Dockerfile.local` (COPY path for restricted networks). Proxy build-args. Non-root `--user` support with `USER=runtime` baked to prevent `getpwuid()` failures.
- **HF upload helper** (`scripts/pipeline/push_run.py`): uploads an entire run dir (directions + graphs + JSONs) to HF in one call.

### What's running RIGHT NOW (Apr 23, end of session)

**Nothing active.** Stage 06 full run landed (96.7% pro-flip, 100% anti, 10/10 benign), committed + pushed to `l15-refactor`. Next work is Stage 05 frontend refresh against the new attribution graphs — to be done locally on the user's ~80 GB-disk box.

---

## Branch / repo state

- **Working branch**: `l15-refactor`. All refactor commits land here.
- **Base branch**: `main` — don't touch without explicit approval.
- **Submodule**: `vendor/circuit-tracer` points to `b5300ee` on branch `refusal-lens-measurement-patch`. That commit is pushed; anyone cloning with `--recurse-submodules` gets it automatically.
- **Tejas's branch**: `tejas-circuit-experiments` (remote only, not a local worktree). Contains his causal experiments, cleaned dataset, per-position direction analysis. **Needed for Task 9**.

---

## Critical schema changes — read before touching any stage

### Stage 02 output schema (consumed by every downstream stage)

`attribution_results.json`:
```json
{
  "metadata": {
    "n_prompts": 50,
    "model": "google/gemma-3-4b-it",
    "transcoder": "mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine",
    "measurement_layer": 15,
    "modes": {"multi": [-5, -3, -2], "single": [-2]},
    "dataset": "controlled",
    "elapsed_minutes": 25.0
  },
  "results": [
    {
      "prompt_idx": 0,
      "prompt_id": 1,
      "prompt": "...",
      "topic": "cybercrime",
      "conditions": {
        "bare": {
          "prefix": "",
          "graphs": {
            "multi": {
              "net": 7.287,
              "pos_sum": 10.0,
              "neg_sum": -2.2,
              "n_features": 10330,
              "n_active": 5473,
              "n_targets": 3,
              "target_positions": [-5, -3, -2],
              "per_target": [
                {"pos_sum": ..., "neg_sum": ..., "net": 2.522},
                {"pos_sum": ..., "neg_sum": ..., "net": 2.394},
                {"pos_sum": ..., "neg_sum": ..., "net": 2.371}
              ],
              "top50_features": {"L14:F7943": 0.023, "L13:F13727": 0.019, ...}
            },
            "single": { /* same shape, target [-2] only */ }
          }
        },
        "jb_fiction":   {"prefix": "...", "graphs": {...}},
        "ctrl_fiction": {"prefix": "...", "graphs": {...}},
        /* ...11 conditions total: bare + 5 jb_* + 5 ctrl_* */
      },
      "feature_comparison": {
        "fiction": {
          "vs_bare":      {"n_shared": 100, "n_bare_only": ..., "n_sign_flipped": ..., "top_sign_flipped": [...]},
          "vs_ctrl":      {/* ctrl_fiction ↔ jb_fiction */},
          "ctrl_vs_bare": {/* bare ↔ ctrl_fiction */}
        }
      }
    }
  ]
}
```

**Feature comparison operates on multi-graph features by default** (see `02_run_attribution.py:_gather_pairs` and the per-prompt loop). Stage 04 should consume multi-graph features for labeling too.

### Raw `.pt` graph file naming

- `02_attribution/graphs/{prompt_idx:03d}_{cond_name}_{mode}.pt`
- Example: `013_jb_fiction_multi.pt`, `013_jb_fiction_single.pt`
- Both modes' graphs are saved by default; `--save-graphs-modes multi` skips single's .pt (still emits single's JSON summary) for disk-constrained runs

### Stage 01 output schema

```
01_direction/
  directions/                  # per-layer directions at pos=-2
    layer_00.pt                # raw tensor, normalized
    layer_01.pt
    ...
    layer_33.pt
  positions_L15/                # per-position directions at L15 (new)
    pos_-15.pt                  # normalized
    pos_-15_unnormalized.pt     # for causal intervention
    pos_-14.pt, pos_-14_unnormalized.pt
    ...
    pos_-1.pt
    index.json                  # {"layer": 15, "positions": ["-15", ..., "-1"], "files": {...}}
  refusal_direction.pt          # legacy single-file container (keep for backward compat)
  unnormalized_r.pt             # per-layer unnormalized (for intervention)
  direction_metadata.json       # per-layer separations, cosine matrix, positions_L15 block
```

### Condition names (11 per prompt)

```
bare
jb_roleplay,    ctrl_roleplay
jb_fiction,     ctrl_fiction
jb_analytical,  ctrl_analytical
jb_completion,  ctrl_completion
jb_cognitive_reframe, ctrl_cognitive_reframe
```

### Config constants (scripts/pipeline/config.py)

```python
MEASUREMENT_LAYER = 15          # was 32, now L15 (causal layer)
MEASUREMENT_POSITION = -2       # "model" token in Gemma-3 chat template
BEST_SEPARATION_LAYER = 32      # historical reference (preserved)
BEST_CAUSAL_LAYER = 15
TARGET_POSITIONS_MULTI = [-5, -3, -2]    # Gemma-3 template anchors
TARGET_POSITIONS_SINGLE = [-2]
PER_POSITION_LAYER = 15
PER_POSITION_POSITIONS = [-15, -14, ..., -1]  # default all 15 for Stage 01
CONTROLLED_DATASET_PATH = REPO_ROOT / "dataset" / "refusal_lens_controlled_dataset.json"
```

---

## Workflow conventions

### Testing
Local tests (no GPU required):
```bash
PYTHONPATH=src python3 scripts/pipeline/tests/test_pipeline_local.py --stage <stage>
# Options: 01, 01-a5, 02, 02b, 03, 03-a4, 04-a7, 04-a8, 04-schema, 06, 07, 07-ctrl, utils, utils-viz, all
```
Conda python with numpy/torch/etc:
```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/pipeline/tests/test_pipeline_local.py --stage all
```

Current test count: **214 passing / 0 failing / 1 skipped** (end of Apr 22 session). The skip is a pre-existing `T-A3c` snapshot reset, unrelated. Several `MockArgs` bugs in pre-existing tests (`test_stage_04_a8`, `test_stage_01`) were fixed earlier this session.

### Code style
- Every Python file starts with `from __future__ import annotations`
- Default to no comments; only comment the WHY (not WHAT)
- Don't add error handling / validation / fallbacks for scenarios that can't happen
- Break back-compat on schema when it's the right call (we just did a major schema rev)

### Dev loop (per user preference)
1. Claude proposes code
2. Mahmoud implements / edits by hand
3. Claude verifies + runs tests
4. Bugs → Mahmoud fixes
5. Full run on RunPod
6. Claude reviews results

No mocks. Real tests. `pytest.importorskip("torch")` for GPU gating.

### Git
- Commit to `l15-refactor`, never `main`
- One commit per logical unit, clear message
- Don't push without user approval (unless they've asked you to)

---

## Phase C — Stage 08 jailbreak patching (new, Apr 24)

**High-level goal (per Georg)**: the patching half of the research. Two steps
plus a manual-steering deliverable that unlock NeurIPS-grade claims:

1. **Step 1 (Stage 08a, landed this session)** — runtime ablation of
   Stage 07-defined subcircuit features via `ReplacementModel.feature_intervention_generate`.
2. **Step 2a (Stage 08b, pending)** — Arditi-style directional orthogonalization
   of feature decoder directions out of MLP/attention write matrices. Produces
   an edited HF model checkpoint with no runtime dependency on the replacement
   model.
3. **Step 2b (Stage 08c, pending)** — input-dependent sidecar that wraps a plain
   HF Gemma and subtracts the same feature reconstructions an MLP would emit.
   Mathematically equivalent to 08a's feature_intervention (zero-value). Compared
   against 08b to benchmark surgical-patch trade-offs.
4. **Manual feature steering (landed this session)** — Stage 05 frontend feature
   cart + `ablation_server.py` FastAPI backend.

Full design in `/home/mshab/.claude/plans/curried-discovering-giraffe.md`.

### Task 11 Stage 08a — resume-session changes (Apr 24)

**New files**:
- `scripts/pipeline/08_ablate_subcircuits.py` (~800 lines) — main orchestrator.
  Mirrors Stage 06 structure (Phase 0/1/2/3, checkpoint/resume, figures,
  summary MD). `--positions {all,anchors,both}` runs comparative analysis of
  `slice(None)` vs. template-anchor `[-5,-3,-2]` positions. `--feature-file cart.json`
  consumes Stage 05 manual cart exports. Reuses Stage 06 baselines when present.
  Headline output: dissociation-matrix heatmap per positions mode +
  class-selective dissociation_delta per class-specific ablation.
- `scripts/pipeline/ablation_server.py` (~250 lines) — FastAPI backend.
  Singleton `ReplacementModel`, CORS for localhost:8000. `POST /ablate`
  returns baseline+ablated generations side-by-side. Launch with
  `python3 scripts/pipeline/ablation_server.py --host 127.0.0.1 --port 8080`.
- `scripts/pipeline/05_frontend_patches/feature-cart.{js,css}` — Stage 05
  right-rail ablation cart. Shift/Cmd-click feature nodes to toggle them into
  the cart (visually: green underline). Buttons: `Export cart.json`,
  `Copy CLI command`, `Run ablation` (POST to localhost:8080), `Clear cart`.

**Modified**:
- `scripts/pipeline/config.py` — added `STAGE_08_DEFAULT_SUBCIRCUITS`
  (universal_refusal_core, ctrl_shared_refusal, 3 class-specific sets),
  `STAGE_08_TEMPLATE_ANCHORS = [-5, -3, -2]`, `STAGE_08_FIRST_TOKEN_MASK = slice(0, 4)`
  (Gemma-3-it transcoder zero-positions mask).
- `scripts/pipeline/utils.py` — new helpers `parse_feature_key`,
  `load_subcircuit_features`, `load_cart`, `resolve_anchor_positions`.
- `scripts/pipeline/utils_viz.py::stage_frontend` — injects feature-cart.{js,css}
  into the frontend via the existing `<link>`/`<script>` injection block.
- `scripts/pipeline/tests/test_pipeline_local.py` — new `test_stage_08`
  (13 assertions, no GPU). Wired into `--stage all` and `--stage 08`.

**CLT pitfalls to respect** (documented in plan + memory):
- Transcoder is cross-layer: a feature at source layer `k` has decoder vectors
  for layers `k..N-1`, not just k. Shape of `_get_decoder_vectors(k, feat_ids)` is
  `[N-k, d_model]`. Load-bearing for 08b and 08c.
- Gemma-3-it transcoders zero positions `[0:4]` (bos / start_of_turn / user / newline).
  `STAGE_08_FIRST_TOKEN_MASK` captures this.
- Transcoder I/O points: encoder input = `pre_feedforward_layernorm.output`,
  decoder writes at `post_feedforward_layernorm.output`.
- Two Gemma-3 architectures (`Gemma3ForCausalLM` vs `Gemma3ForConditionalGeneration`);
  Stage 06 uses the multimodal wrapper. 08b must detect at load time.

**Validation bar for PR #1 (must hit on `run_20260422_015552`)**:
- POSITIVE control: `universal_refusal_core` (116 features, all positions) must
  drop bare REFUSE 49/49 → ≤30/49 (break_rate ≥ 39%).
- NEGATIVE control: `ctrl_shared_refusal` (50 features) must affect JB flip rate
  by ≤5 pp across every class (recovery_rate ≤ 5% for all jb_*).
- DISSOCIATION (headline NeurIPS figure): ≥2 of
  `{jb_fiction_specific_vs_ctrl, jb_analytical_specific_vs_ctrl, jb_cognitive_reframe_specific_vs_ctrl}`
  must show class-selective dissociation_delta ≥ +20 pp on jb_fiction /
  jb_analytical / jb_cognitive_reframe recovery.

**Smoke-test command** (run on GPU):
```bash
PYTHONPATH=src python3 scripts/pipeline/08_ablate_subcircuits.py \
    --run-dir data/results/pipeline_runs/run_20260422_015552 \
    --subcircuits universal_refusal_core,ctrl_shared_refusal,jb_fiction_specific_vs_ctrl,jb_analytical_specific_vs_ctrl,jb_cognitive_reframe_specific_vs_ctrl \
    --positions both --max-prompts 5
```

**Local test count**: 214 → 227 (13 new T-S8* assertions). Ran locally without
torch/numpy — the test_stage_08 path is pure-Python helper logic and fully
isolated from torch imports.

### Local-dev Python environment (Apr 24)

The WSL dev box lacks pip/numpy/torch. Created a local venv at `.venv/` (via
`~/.local/bin/virtualenv`) for local unit-test runs. Test invocation:

```bash
PYTHONPATH=src .venv/bin/python3 scripts/pipeline/tests/test_pipeline_local.py --stage 08
```

The `.venv/` is `.gitignore`'d by the existing pattern. Full-suite tests that
need numpy/matplotlib/transformers still work on RunPod or any machine with
conda-python.

## Remaining tasks — detailed implementation plans

*(Tasks 7 / 8 / 9 / 10 / 11(08a) are DONE — see the per-task "resume-session changes" blocks earlier in this file for what landed. Only 04b, 08b, and 08c still need implementation.)*

### Task 7b — Stage 04b Delphi/LLM feature labels (pending)

**Goal**: build a new Stage 04b for LLM-generated feature labels, emitting a persistent hashmap (`feature_id → {llm_label, detection_score}`).

**File**: `scripts/pipeline/04b_delphi_labels.py` — **new** (currently doesn't exist)

**Spec** (from `scripts/pipeline/PIPELINE_PLAN.md` notes):
- For each unique feature in `04_labels/feature_labels.json`, assemble `(top_logits, bottom_logits, activation_examples, top-5 incoming + outgoing edges)` from the HF dashboard binary (already decoded in Stage 04).
- Claude API call (Haiku 4.5) for each → 5–10 word `llm_label`.
- Estimated cost: ~$2 total for ~1350 features × 1-shot call, no caching needed.
- Output: `feature_labels_llm.json` — same schema as `feature_labels.json` + adds `llm_label` field.
- Merge with Ruqiya's Delphi hashmap if present (format TBD — ask user).
- `utils_viz.py` should consume `llm_label` into each graph's `clerp` field at staging time so the frontend shows human-readable names.

**Dependencies**: Stage 04 outputs (done on `run_20260422_015552`).

**Tests to add**:
- T-S4b-a: every feature in `feature_labels_llm.json` has non-empty `llm_label`.
- T-S4b-b: schema sanity — identical key set to `feature_labels.json`.

---

### Task 11 — Stage 08 subcircuit ablation

**Goal**: targeted ablation of features belonging to specific subcircuits (from Stage 07), measure how refusal behavior changes.

**File**: `scripts/pipeline/08_ablate_subcircuits.py` (new)

**Approach**:
1. Load `subcircuits.json` from Stage 07
2. For each subcircuit of interest (start with `canonical_pro_refusal`, `sign_flip_convergent`, `dampening_specialists`, `jb_fiction_specific_vs_ctrl`, etc.):
   - Zero-ablate those features' contributions at their respective layers during forward pass
   - Re-generate on the controlled dataset
   - Measure flip rate delta vs no-ablation baseline
3. Also test **combinations**: ablate two subcircuits together, see interaction effects

**Ablation mechanics**:
- Features live at specific (layer, feature_idx) positions in the transcoder decomposition
- Use circuit-tracer's `FeatureIntervention` (if available) OR hook into nnsight's model pass to zero-out the transcoder output for those features
- Must match the MLP contribution pathway (not attention, since transcoders only cover MLPs)

**Research questions this should answer**:
- Does ablating `jb_fiction_specific_vs_ctrl` specifically kill fiction's bypass effect while leaving other JBs intact? (Dissociation test)
- Does ablating `canonical_pro_refusal` block JB recovery via Arditi intervention? (Redundancy test)
- Does ablating `dampening_specialists` restore refusal on roleplay-class JBs? (Mechanism test)

**Output**:
```json
{
  "metadata": {...},
  "ablations": {
    "canonical_pro_refusal": {
      "n_features_ablated": 56,
      "flip_rate_delta": {"fiction": -0.35, "roleplay": -0.12, ...},
      "comply_rate_delta": {...}
    },
    ...
  }
}
```

**Dependencies**: needs Task 10 (subcircuit definitions) + Stage 06 baseline (non-ablated refusal rates for comparison)

**Tests**:
- T-S8a: ablation mask shape correctness (one-hot on (layer, feature) positions)
- T-S8b: zero-ablation of empty subcircuit == baseline (sanity)
- T-S8c: ablating `universal_refusal_core` changes bare refusal rate (should flip some bare prompts to comply)

---

## Immediate next steps for new session (priority order, Apr 23)

### 1. Stage 05 frontend refresh on `run_20260422_015552` (NEXT — user has ~80 GB local disk)

The new attribution data needs to flow through the frontend pipeline end-to-end. Stage 05 code (3-way bare/ctrl/jb coloring, ctrl-aware subcircuit filter panel, new slug format) is ready but hasn't been exercised against the real `run_20260422_015552` graphs yet. Run this locally.

**Step 1a — fetch the raw `.pt` graphs from HF (~80 GB, one-time)**:
```bash
cd /path/to/Refusal-Lens
python3 scripts/pipeline/fetch_raw_graphs.py \
    --run run_20260422_015552 \
    --dataset-repo moon70/refusal-lens-graphs
```

**Step 1b — pack to gzipped JSONs + push to HF** (once, so future viewers pull ~3 GB instead of 80 GB):
```bash
python3 scripts/pipeline/02c_pack_graphs.py \
    --run-dir data/results/pipeline_runs/run_20260422_015552

python3 scripts/pipeline/push_graph_data.py \
    --run-dir data/results/pipeline_runs/run_20260422_015552 \
    --source 02c \
    --dataset-repo moon70/refusal-lens-graphs
```

**Step 1c — render the frontend** (stages the viewer with 3-way coloring + subcircuit filter panel):
```bash
python3 scripts/pipeline/05_visualize_circuits.py \
    --run-dir data/results/pipeline_runs/run_20260422_015552 \
    --subcircuits-run data/results/pipeline_runs/run_20260422_015552 \
    --mode single \
    --skip-convert \
    --gzip
```

**Step 1d — serve + spot-check in browser**:
```bash
cd data/results/pipeline_runs/run_20260422_015552/05_frontend
python3 -m http.server 8000
# Open http://localhost:8000/ in a browser
```
Visual acceptance criteria:
- `jb_fiction` prompt-graph has BOTH gold `shared_with_ctrl` nodes (prefix-induced) AND orange `jb_unique` nodes (true JB-semantic) — this is the visual payoff of Task 8.
- Subcircuit filter panel on the right lists 18 subcircuits (11 legacy + 7 ctrl-aware) with counts > 0 for at least the major ones (`universal_refusal_core`, `canonical_pro_refusal`, `sign_flip_convergent`, `jb_fiction_specific_vs_ctrl`, etc.).
- Overlap legend shows the 3-way buckets with the "PREFIX-induced" / "true JB-semantic" labels.
- `compare.html` (bare-vs-JB side-by-side) still works.

**If something looks wrong** — likely suspects for the new data:
- Slug parsing: new `.pt` names are `{idx}_{cond_name}_{mode}.pt` (e.g. `013_jb_fiction_multi.pt`). `parse_slug` in `05_visualize_circuits.py` handles this, but double-check it lands in `group_by_prompt_structured` correctly.
- 3-way annotation: only fires when matched `ctrl_{cls}` exists for a `jb_{cls}` at the same mode (`multi` or `single`). If missing, falls back to 2-way (bare-vs-jb only) — legend won't show gold.
- Subcircuit panel: pulls `subcircuits.json` from `07_subcircuits/`; must contain the 7 ctrl-aware names.

### 2. Phase A0 HF push (bundled with step 1b above)

Already included as step 1b. After this, `run_20260422_015552`'s gzipped JSONs live at `https://huggingface.co/datasets/moon70/refusal-lens-graphs/tree/main/runs/run_20260422_015552/graph_data/` and collaborators can fetch via `fetch_graph_data.py` (no 80 GB download).

### 3. Task 11 Stage 08 — Subcircuit ablation (the "jailbreak-patching framework" deliverable)

After Stage 05 ships, this is the scientific payoff. Ablate Stage 07-defined subcircuits at their native layers, measure how the flip rate in Stage 06 changes. If `jb_fiction_specific_vs_ctrl` ablation specifically suppresses fiction jailbreaks (and not other classes), we have a **causal dissociation** — the framework claim.

**Starting questions for the design**:
- Mechanism: zero-ablate features at the transcoder decomposition, OR hook into the residual stream and project out feature directions?
- Metric: re-run Stage 06 under ablation, compare `pro_refusal_add` flip rate per class.
- Dissociation test: does ablating `jb_fiction_specific_vs_ctrl` (52 features) selectively drop fiction's 90% flip toward 0%, while leaving analytical/cognitive_reframe intact?
- Null-control: ablating `universal_refusal_core` (116 features) should break bare refusal too — that's the positive-control proving ablations actually matter.

Depends on: Stage 06 outputs (done), Stage 07 subcircuits (done). Needs GPU for generation. Estimated ~4h on H100 for one ablation condition × 50 prompts; several conditions.

### 4. Lower priority / post-ICML

- **Task 7b Stage 04b** — LLM labels via Claude API (~$2, CPU). Makes frontend features human-readable. Run after Stage 05 lands so the labels surface in the viewer.
- **`|r_L15|` magnitude mystery diagnostic** — diff `dataset/refusal_direction_dataset/splits/harmful_train.json` on `l15-refactor` vs `origin/tejas-circuit-experiments` to find the content drift that produced 3123.9 vs 4019.7. Not blocking but worth closing for rigor.
- **ICML abstract writing** (May 4) — numbers are in hand: see "Phase A validation findings" + "Phase B results" blocks.
- **Fiction resistance deep-dive**: pull the 2 fiction prompts that didn't flip under pro-refusal-add (`causal_results.json`, filter `interventions.L15_pro_refusal_add.jb_fiction.flipped_toward_refuse == false`) and look for structural commonalities — candidate for a paper figure.

### Gotchas

- **Run-dir layout**: attribution_results JSONs for `run_20260422_015552` were originally at run-dir root; this session moved them into `02_attribution/`. If a NEW RunPod run lands, check layout before running Stage 02b — it expects `<run>/02_attribution/attribution_results.json`.
- **LFS hook on git push**: pod environments without `git-lfs` binary installed will silently abort pushes because of a `pre-push` hook. Fix: `apt-get install -y git-lfs && git lfs install`. See HANDOFF commit history for context.
- **`|r|` direction magnitude**: our Stage 06 uses `--r-source recompute` by default for Tejas-exact methodology, but our recompute produces `|r|=3123.9` vs Tejas's `4019.7`. Both magnitudes produce the bulletproof 10/10 benign force-refuse — suggests the gap is benign but worth understanding before publication.

---

## Where to find things

| What | Path |
|---|---|
| Main pipeline stages | `scripts/pipeline/0X_*.py` |
| Shared helpers | `scripts/pipeline/{utils,utils_viz,config}.py` |
| Local tests | `scripts/pipeline/tests/test_pipeline_local.py` |
| RunPod test (GPU) | `scripts/pipeline/tests/test_runpod_1_4.py` |
| Controlled dataset | `dataset/refusal_lens_controlled_dataset.json` |
| Direction computation reference | `scripts/pipeline/README.md` |
| Design decisions | `scripts/pipeline/PIPELINE_PLAN.md` |
| Subcircuit findings (latest) | `data/results/pipeline_runs/run_20260417_010035/07_subcircuits/SUBCIRCUITS_REPORT.md` |
| Tejas's README (latest) | `git show origin/tejas-circuit-experiments:data/tejas_experiments/README.md` |
| **Tejas's bulletproof causal script** (source of truth for Stage 06) | `git show origin/tejas-circuit-experiments:data/tejas_experiments/scripts/20_bulletproof_pipeline.py` |
| Latest RunPod run (this session) | `data/results/pipeline_runs/run_20260422_015552/` |
| Subcircuit findings (real data) | `data/results/pipeline_runs/run_20260422_015552/07_subcircuits/SUBCIRCUITS_REPORT.md` |
| Legacy L32 reference run (for size-calibration tests) | `data/results/pipeline_runs/run_20260417_010035/` |
| HF push helper (entire run, .pt + JSON) | `scripts/pipeline/push_run.py` |
| HF push helper (JSON.gz bundle for frontend) | `scripts/pipeline/push_graph_data.py` |
| `.pt → JSON.gz` packer | `scripts/pipeline/02c_pack_graphs.py` |
| Parallel GPU launcher (Stage 02) | `scripts/pipeline/run_stage02_parallel.sh` + `merge_stage02_shards.py` |
| Dockerfiles | `/Dockerfile` (git-clone), `/Dockerfile.local` (COPY-based) |
| Smoke test JSON (schema reference) | `data/results/pipeline_runs/attribution_results_test.json` |
| Session plan file (if needed for provenance) | `/home/m00n/.claude/plans/enchanted-wibbling-thompson.md` |

---

## Gotchas worth knowing

1. **Tejas's legacy key naming**: pre-ctrl runs keyed conditions as just `{cls}` (e.g., `"fiction"`). New runs use `"jb_fiction"` and `"ctrl_fiction"`. The schema adapters (`_get_metrics` in 02b, `_get_bare_net` in 03) handle the fallback: when a `jb_{cls}` lookup fails, they strip the prefix and try `{cls}`. Preserve this fallback when touching Stage 04+.

2. **Test data staleness**: `data/results/pipeline_runs/run_20260417_010035/` uses the OLD flat schema (legacy). `data/results/pipeline_runs/attribution_results_test.json` is the CURRENT smoke test (new nested schema). When testing, make sure you're pointing at the right one.

3. **HF-auth via env**: `HF_TOKEN` env var is auto-picked up by `huggingface_hub`. No `hf auth login` needed inside the container. Use `api.whoami()` in scripts to verify.

4. **`--max-features 0` means unlimited**: the CLI convention. `None` (default) also means unlimited. Stage 02 internally converts 0 → None before passing to `attribute()`.

5. **torchvision mismatch on new torch**: PyTorch base images sometimes ship mismatched torch/torchvision. Fix: `pip install --upgrade torchvision` (no pin, let pip resolve against the installed torch). Both Dockerfiles already do this at build time.

6. **Don't git-commit raw graphs**: `.gitignore` excludes `data/results/pipeline_runs/**/02_attribution/graphs/`. But DO commit everything else in a run dir (01_direction, 02_attribution/*.json, config.json).

7. **Tejas's branch has drift**: `tejas-circuit-experiments` is based off an older point in history. Don't try to merge it into `l15-refactor` directly — `git show origin/tejas-circuit-experiments:<path>` to read individual scripts. The local `data/tejas_experiments/scripts/` copies are STALE (pre-Script-20). Remote `20_bulletproof_pipeline.py` is the current source of truth for causal work.

8. **Stage 02 RunPod output placement**: when the RunPod shard launcher writes attribution JSONs, they can end up at run-dir root instead of under `02_attribution/`. Stage 02b expects `<run>/02_attribution/attribution_results.json`. If you see `ERROR: ... attribution_results.json not found`, move the files into the subdirectory (we had to do this once for `run_20260422_015552`).

9. **L15 measurement → `late_wave_layer24_32` is empty**: the "late wave" subcircuit rule was calibrated on L32-measurement data. On L15-measurement runs, attribution doesn't propagate past the measurement layer, so the L24-L32 band holds zero features. This is correct, not a bug. `test_stage_07`'s T-07g/h are gated on `run_dir.name == "run_20260417_010035"` (the legacy reference) to avoid false failures.

---

## Running the open-ended test

```bash
# Local
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/pipeline/tests/test_pipeline_local.py --stage all

# Expected (Apr 22): 214 passed, 0 failed, 1 skipped (T-A3c snapshot reset — unrelated)
```

If the counts drift below 214 / 0 / 1 without the change being intentional, something regressed.

---

## Final note

Correlational pipeline (Stages 01 / 02 / 02b / 03 / 04 / 07) is complete and validated on real data. Stage 05 frontend code is complete and awaits the Phase A0 packaging run. Stage 06 causal intervention code is complete and awaits a GPU run on RunPod. ICML headline numbers are in hand.

Tests first, feature second. If a stage's local tests don't pass against the smoke JSON, it won't pass against the full run either.
