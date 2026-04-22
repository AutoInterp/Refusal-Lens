# Refusal-Lens — Pipeline State & Remaining Tasks

**Handoff doc, 2026-04-21.** Written for a fresh Claude session to resume pipeline work while a full attribution run executes on RunPod.

---

## Project overview

Mechanistic interpretability research on **Gemma-3-4b-it** — attributing the model's refusal circuit using Anthropic's circuit-tracer (CLT). Mentor: Georg. Research arc: *"Here are the circuits, they're real, here's what they encode, and here's what happens when you manipulate them."*

- **Repo**: `AutoInterp/Refusal-Lens`, working branch `l15-refactor`
- **Submodule**: `vendor/circuit-tracer` on branch `refusal-lens-measurement-patch`, local commit `b5300ee` (patched for multi-position targets, pushed upstream)
- **HF dataset**: `moon70/refusal-lens-graphs` (raw `.pt` graphs + directions archived; frontend JSONs also published)

### Team
- **Mahmoud** (user): correlational attribution, Stages 01–05 + 07, frontend
- **Tejas**: causal intervention work on `tejas-circuit-experiments` branch. Scripts at `data/tejas_experiments/scripts/` — especially `16_causal_arditi.py`, `17_causal_georg_arditi.py`, and cleaned-dataset experiments (95/95 causal flip proven at L15).
- **Georg**: mentor
- **Ruqiya**: Neuronpedia feature-label extraction (Stage 04b input)

### Deadlines
- **Apr 23 12 PM PST** — Algoverse symposium (5-min talk). Slides use OLD run `run_20260417_010035` data (L32 single-position); the new L15 multi-position run won't be ready in time.
- **May 4** — ICML workshop abstract. Target for the new-data results.

---

## Current pipeline state

### Stages that are REFACTORED and tested locally

| # | Stage | Status | Notes |
|---|---|---|---|
| 01 | `01_compute_direction.py` | ✅ | Per-layer (34 layers at pos=-2) + per-position at L15 (positions -1..-15). Matches Tejas's direction at cos=0.9991. |
| 02 | `02_run_attribution.py` | ✅ | L15 target, 11 conditions, two-graph scheme (**multi** = positions [-5,-3,-2] template anchors, **single** = [-2] causally-verified). Per-position directions, multi-target attribute() call. |
| 02b | `02b_statistical_analysis.py` | ✅ | 2 modes × 3 comparisons (`vs_bare`, `vs_ctrl`, `ctrl_vs_bare`) × 5 classes. Schema-aware + legacy-fallback. 7 tests pass. |
| 03 | `03_verify_attribution.py` | ✅ | L15 per-position direction loading, multi-target dot product verification. A4 tests still pass. |

### Stages that NEED REFACTORING (pending work)

| # | Stage | Task | Dependency |
|---|---|---|---|
| 04 | `04_label_features.py` | Task 7 | Needs 11-cond + 2-graph adaptation |
| 04b | `04b_delphi_labels.py` | Task 7 | **NEW stage** — Delphi-style LLM labels |
| 05 | `05_visualize_circuits.py` + frontend | Task 8 | Add `shared_with_ctrl` / `ctrl_unique` overlap buckets |
| 06 | `06_causal_intervention.py` | Task 9 | **NEW stage** — wrap Tejas's scripts |
| 07 | `07_identify_subcircuits.py` | Task 10 | Add `ctrl_shared` / `ctrl_unique` subcircuit rules |
| 08 | `08_ablate_subcircuits.py` | Task 11 | **NEW stage** — depends on 07 |

### Critical infrastructure refactors (done this session)

- **circuit-tracer patch** (`vendor/circuit-tracer/circuit_tracer/attribution/*.py`): `measurement_layer` and `measurement_position` accept `int | Sequence[int] | None`. Per-target measurement sinks via `_as_measurement_tensor` helper. Backward pass correctly injects gradients at each target's own position.
- **Parallel GPU launcher** (`run_stage02_parallel.sh` + `merge_stage02_shards.py`): shards prompts across N GPUs with shard-specific checkpoints, auto-merges at end.
- **Stage 02 speedup**: batch size default 1→256 (→512 for H100/Blackwell), ~100× faster.
- **`graph_summary` multi-target fix**: `_aggregated_target_row` sums the weighted target rows correctly. Previously was reading only the last target's row — `bare.multi.net = 7.287` now correctly equals `sum(per_target[i].net)` across all 3 template positions.
- **Docker**: `Dockerfile` (git-clone path) + `Dockerfile.local` (COPY path for restricted networks). Proxy build-args. Non-root `--user` support with `USER=runtime` baked to prevent `getpwuid()` failures.
- **HF upload helper** (`scripts/pipeline/push_run.py`): uploads an entire run dir (directions + graphs + JSONs) to HF in one call.

### What's running RIGHT NOW

Mahmoud is on RunPod (2× RTX 5090 Blackwell, 1 TB volume, torch 2.11 + CUDA 13) running the full 50-prompt × 11-condition × 2-graph attribution with `--max-features 0` (unlimited features) per Georg's explicit ask. Expected wall-clock ~20-40 min on Blackwell. After completion: HF push (~30-90 min for ~530 GB) + git commit of JSON summaries.

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
# Options: 01, 01-a5, 02, 02b, 03, 03-a4, 04-a7, 04-a8, 07, utils, utils-viz, all
```
Conda python with numpy/torch/etc:
```bash
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/pipeline/tests/test_pipeline_local.py --stage all
```

Current test count: **86 passing**, 1 pre-existing failure in `test_stage_04_a8` (MockArgs missing `upset_only` attribute — Task 7 should fix in passing).

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

## Remaining tasks — detailed implementation plans

### Task 7 — Stage 04 + 04b feature labeling **(in progress, top priority)**

**Goal**: adapt HuggingFace Gemma Scope labeling (Stage 04) to the new 11-cond × 2-graph schema. Build a new Stage 04b for Delphi-style LLM labels that emits a persistent hashmap (`feature_id → {explanation, detection_score}`).

**Files**:
- `scripts/pipeline/04_label_features.py` — refactor
- `scripts/pipeline/04b_delphi_labels.py` — **new** (currently doesn't exist)

**Stage 04 changes needed**:
1. Read from new schema `conditions[cond].graphs.{multi,single}.top50_features` + `feature_comparison[cls].{vs_bare,vs_ctrl,ctrl_vs_bare}`.
2. Build the union of unique features from:
   - All 11 conditions' `top50_features` (multi graph — the canonical source)
   - All `vs_bare` / `vs_ctrl` / `ctrl_vs_bare` comparison buckets (`sign_flipped`, `dampened`, `amplified_anti`) per class
3. Output `feature_labels.json` + `feature_class_sets.json` with a **`conditions_seen`** field that now includes the `bare / jb_{cls} / ctrl_{cls}` names (not just the legacy `bare / fiction / ...`).
4. Fix the pre-existing `test_stage_04_a8` MockArgs bug (add `upset_only = False`).

**Stage 04b (new) spec** (from `scripts/pipeline/PIPELINE_PLAN.md` notes):
- For each unique feature, assemble `(top_logits, bottom_logits, activation_examples, top-5 incoming + outgoing edges)` from the HF dashboard binary (already decoded in Stage 04)
- Claude API call (Haiku 4.5) for each → 5–10 word `llm_label`
- Estimated cost: ~$2 total for ~876 features × 1 shot call, no caching needed
- Output: `feature_labels_llm.json` — same schema as Stage 04's feature_labels + adds `llm_label` field
- Merge with Ruqiya's Delphi hashmap if present (format TBD — ask user)
- `utils_viz.py` should consume `llm_label` into each graph's `clerp` field at staging time so the frontend shows human-readable names

**Dependencies**: waits on full Stage 02 run (to have the feature universe)

**Tests to add**:
- T-S4a: `feature_class_sets.json` has `conditions_seen` keys covering `bare`, `jb_fiction`, `ctrl_fiction`, etc. (not legacy `fiction`)
- T-S4b: counts match (n_unique_features from features collected ≈ len(feature_labels))
- T-S4c: Delphi schema sanity — every feature has `llm_label` populated

---

### Task 10 — Stage 07 subcircuits on new data + ctrl buckets

**Goal**: run rule-based subcircuit identification on the new 11-cond data, adding ctrl-aware rules.

**File**: `scripts/pipeline/07_identify_subcircuits.py`

**Existing subcircuits** (from `07_identify_subcircuits.py` header):
- `universal_refusal_core` — features in bare + all 5 JB
- `canonical_pro_refusal` — features in all 5 JB, not in bare
- `sign_flip_convergent` — sign_flipped in ≥3 JB classes
- `dampening_specialists` — dampened in ≥3 JB classes
- `anti_refusal_amplifiers` — amplified_anti in ≥3 JB classes
- `late_wave_layer24_32` — all features in L24–L32
- `{class}_exclusive` (×5) — features in exactly one JB class

**New subcircuits to add** (ctrl-aware):
- `ctrl_shared_refusal` — features present in bare AND all 5 ctrl (but NOT all 5 JB). "Features the refusal circuit uses regardless of prefix."
- `ctrl_only` — features in all 5 ctrl but not in bare or any JB. (Probably tiny — if non-empty, indicates that token-matched control prefixes recruit distinct features.)
- `jb_specific_vs_ctrl` — for each class, features in `jb_{cls}`'s top-50 but NOT in `ctrl_{cls}`'s top-50. This is the **cleanest** JB-semantic subcircuit and should be surfaced prominently.

**Schema update**: `feature_class_sets.json` output from Stage 04 needs to provide per-class sets for `jb_{cls}` AND `ctrl_{cls}` top-50s separately (not just JB vs bare). Task 7 must emit this structure for Task 10 to consume.

**Expected outputs** (preserve backward compat):
- `subcircuits.json` — same shape, with new subcircuit names added
- `subcircuits_summary.json` — pairwise overlap matrix
- `subcircuits_treemap.png`, `subcircuits_by_layer.png`, `subcircuits_overlap.png`
- `SUBCIRCUITS_REPORT.md`

**Key finding to reproduce**: the two 85% structural identities
- `canonical_pro_refusal ∩ sign_flip_convergent = 48/56 (86%)` — "JB-recruited refusal = sign-flipped refusal"
- `universal_refusal_core ∩ dampening_specialists = 44/52 (85%)` — "dampening attacks the canonical core"
- `canonical ∩ dampening = 2/52 (4%)` — disjoint mechanisms

**Dependencies**: waits on Task 7 (Stage 04 output)

**Tests**: T-S7a through T-S7e already exist, focused on rule correctness. Add new T-S7f..T-S7j for the new ctrl-aware rules.

---

### Task 8 — Stage 05 3-way frontend overlap

**Goal**: extend the circuit viewer to show bare / ctrl / JB distinctions, not just bare / JB.

**Files**:
- `scripts/pipeline/utils_viz.py` — extend `annotate_overlap()` to 3-way
- `scripts/pipeline/05_visualize_circuits.py` — pass ctrl condition references
- `scripts/pipeline/05_frontend_patches/overlap-colors.css` — new palette entries
- `scripts/pipeline/05_frontend_patches/overlap-annotate.js` — new bucket handling
- `scripts/pipeline/05_frontend_patches/compare.html` — 3-column layout (bare | ctrl | JB) instead of bare | JB

**Current overlap buckets** (from `utils_viz.py:OVERLAP_BUCKETS`):
```python
("shared_with_bare", "jb_unique", "bare_only", "bare", "non_feature")
```

**New buckets** to add:
```python
("shared_with_ctrl", "ctrl_unique", "ctrl", "shared_with_bare_and_ctrl")
```

**Rule logic** (for a JB graph being annotated against both bare and ctrl graphs):
- Feature in bare AND ctrl AND jb → `shared_with_bare_and_ctrl` (green, strongest stability)
- Feature in bare AND jb, not in ctrl → `shared_with_bare` (existing)
- Feature in ctrl AND jb, not in bare → `shared_with_ctrl` (new — interesting signal!)
- Feature in jb only → `jb_unique`

For a ctrl graph being annotated:
- Feature in bare AND ctrl → `shared_with_bare`
- Feature in ctrl only → `ctrl_unique`

**Subcircuit filter bug fix (already done, from Georg TODO 1)**: the `annotate_subcircuits()` function already filters corpus memberships against per-graph `overlap_bucket`. Need to extend the filter rules for new subcircuits from Task 10:
- `ctrl_shared_refusal` → only if `overlap_bucket in {bare, ctrl, shared_with_bare, shared_with_ctrl, shared_with_bare_and_ctrl}`
- `jb_specific_vs_ctrl` → only if `overlap_bucket == jb_unique`
- `ctrl_only` → only if `overlap_bucket == ctrl_unique`

**Dependencies**: needs Task 10's subcircuit names finalized; Task 8 itself doesn't depend on the full GPU run (only frontend JS/CSS + annotation function)

**Tests**: extend `T-V3*` and `T-V4*` to cover the new bucket rules.

---

### Task 9 — Stage 06 causal intervention (Tejas integration)

**Goal**: wrap Tejas's causal intervention scripts into a proper pipeline stage that reproduces his 95/95 JB flip + 10/10 benign force-refusal on the cleaned dataset.

**Source material** (on branch `tejas-circuit-experiments`):
- `data/tejas_experiments/scripts/16_causal_arditi.py` — Arditi method (add unnormalized r at all positions every forward step) → 32/32 flipped
- `data/tejas_experiments/scripts/17_causal_georg_arditi.py` — Georg's exact-magnitude variant → 8/32
- `data/tejas_experiments/scripts/18_cleaned_dataset.py` (or similar) — the 95/95 result on the controlled dataset
- `data/tejas_experiments/scripts/19_disentangle_2x2.py` — 2×2 disentangle showing "every-step" is the critical factor

**New file**: `scripts/pipeline/06_causal_intervention.py`

**Must follow pipeline conventions**:
- `config.py` constants (MODEL_NAME, MEASUREMENT_LAYER, etc.)
- `get_stage_dir(run_dir, "06_causal")` for output
- `utils.load_controlled_dataset` for dataset
- Resume-able checkpoint (same pattern as Stage 02)

**Core logic** (Arditi method, from Tejas's Script 16):
1. For each prompt × condition (bare, jb_{cls}, ctrl_{cls}):
   - Generate baseline (no intervention). Classify refused/complied (use `utils.classify_response`)
2. For each (prompt, jb_{cls}) where baseline is complied AND ctrl_{cls} refused:
   - Run intervention: add `r_L15 @ pos=-2` unnormalized direction at every forward step
   - Regenerate. Classify.
   - Record as `jb_flipped` if originally complied now refuses
3. For each (prompt, bare) where baseline refused:
   - Run anti-intervention: subtract r_L15 to force non-refusal (force-comply)
   - Record as `bare_comply` if originally refused now complies
4. Also run the control: generate under ctrl_{cls} baseline (should refuse in 96% of cases, matches Tejas).

**Output schema** (`06_causal/causal_results.json`):
```json
{
  "metadata": {...},
  "results": [
    {
      "prompt_idx": 0,
      "baseline": {
        "bare":                {"refused": true, "coherent": true, "response": "I can't..."},
        "jb_fiction":          {"refused": false, "coherent": true, "response": "..."},
        "ctrl_fiction":        {"refused": true, ...}
      },
      "intervention": {
        "jb_fiction":          {"flipped": true, "response_after": "..."},
        ...
      }
    }
  ],
  "summary": {
    "total_jb_complied": 95,
    "total_jb_flipped": 95,
    "flip_rate": 1.0,
    "per_class": {"fiction": {...}, "roleplay": {...}, ...}
  }
}
```

**Dependencies**: needs Stage 01 outputs (per-layer unnormalized directions for L15); independent of Stage 02.

**Tests**:
- T-S6a: output schema sanity
- T-S6b: small smoke test: 2 prompts × 2 classes × bare + ctrl + jb → baseline generation works, intervention flips a known example
- Can't test on CPU; mark GPU-required

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

## Suggested order for a parallel session

While the RunPod run executes (~25 min compute + 30-90 min HF upload), work through these in order. Tasks with no GPU dependency can land before the run finishes.

1. **Task 7 Stage 04 refactor** (no GPU) — can ship against the smoke JSON at `data/results/pipeline_runs/attribution_results_test.json`. The full 50-prompt run will just scale it up.
2. **Task 7 Stage 04b new** (GPU optional — Haiku API calls) — parallel to Stage 04 refactor; the Claude-API labeling step is CPU-bound.
3. **Task 10 Stage 07 refactor** (no GPU) — needs Task 7's output schema to be finalized. Can be started against smoke data once Task 7 lands.
4. **Task 8 Stage 05 frontend** (no GPU) — pure JS/CSS/Python annotation changes. Can ship independently.
5. **Task 9 Stage 06 causal** (needs GPU for smoke; code design can start now) — review Tejas's scripts first, produce a design doc, then implement + test smoke on RunPod.
6. **Task 11 Stage 08 ablation** (needs GPU) — last because it depends on 10's definitions + 9's baseline.

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
| Tejas's README | `git show origin/tejas-circuit-experiments:data/tejas_experiments/README.md` |
| Tejas's causal scripts | `git show origin/tejas-circuit-experiments:data/tejas_experiments/scripts/16_causal_arditi.py` |
| HF push helper | `scripts/pipeline/push_run.py` |
| Parallel launcher | `scripts/pipeline/run_stage02_parallel.sh` + `merge_stage02_shards.py` |
| Dockerfiles | `/Dockerfile` (git-clone), `/Dockerfile.local` (COPY-based) |
| Smoke test JSON (schema reference) | `data/results/pipeline_runs/attribution_results_test.json` |

---

## Gotchas worth knowing

1. **Tejas's legacy key naming**: pre-ctrl runs keyed conditions as just `{cls}` (e.g., `"fiction"`). New runs use `"jb_fiction"` and `"ctrl_fiction"`. The schema adapters (`_get_metrics` in 02b, `_get_bare_net` in 03) handle the fallback: when a `jb_{cls}` lookup fails, they strip the prefix and try `{cls}`. Preserve this fallback when touching Stage 04+.

2. **Test data staleness**: `data/results/pipeline_runs/run_20260417_010035/` uses the OLD flat schema (legacy). `data/results/pipeline_runs/attribution_results_test.json` is the CURRENT smoke test (new nested schema). When testing, make sure you're pointing at the right one.

3. **HF-auth via env**: `HF_TOKEN` env var is auto-picked up by `huggingface_hub`. No `hf auth login` needed inside the container. Use `api.whoami()` in scripts to verify.

4. **`--max-features 0` means unlimited**: the CLI convention. `None` (default) also means unlimited. Stage 02 internally converts 0 → None before passing to `attribute()`.

5. **torchvision mismatch on new torch**: PyTorch base images sometimes ship mismatched torch/torchvision. Fix: `pip install --upgrade torchvision` (no pin, let pip resolve against the installed torch). Both Dockerfiles already do this at build time.

6. **Don't git-commit raw graphs**: `.gitignore` excludes `data/results/pipeline_runs/**/02_attribution/graphs/`. But DO commit everything else in a run dir (01_direction, 02_attribution/*.json, config.json).

7. **Pre-existing test bug** (`test_stage_04_a8`): MockArgs doesn't set `upset_only`. Unrelated to any current refactor. Task 7 is a natural place to fix it since it's in Stage 04.

8. **Tejas's branch has drift**: `tejas-circuit-experiments` is based off an older point in history. Don't try to merge it into `l15-refactor` directly — cherry-pick the scripts you need for Task 9 instead.

---

## Running the open-ended test

```bash
# Local
PYTHONPATH=src /opt/anaconda3/bin/python3 scripts/pipeline/tests/test_pipeline_local.py --stage all

# Expected: 86 pass, 1 skip (upsetplot not installed), 1 fail (test_stage_04_a8 pre-existing)
```

If any of those change without the task being intentional, something regressed.

---

## Final note

The full RunPod attribution run will finish in ~30 min (batch=512 on Blackwell). HF push takes another ~30-90 min depending on upload speed. By the time all Tasks 7-11 code lands, the new-dataset `.pt` graphs will be on HF ready to be consumed end-to-end.

Tests first, feature second. If a stage's local tests don't pass against the smoke JSON, it won't pass against the full run either.
