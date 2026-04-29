# Refusal-Lens — Pipeline State & Remaining Tasks

**Last updated 2026-04-29** — Mentor's three-bug fixes integrated (P1-P3). Stage 07/08 refactored for per-prompt subcircuit construction + comply-weighted ablation summary (P4-P6). Local test suite green at **244 passed / 0 failed / 1 skipped**. **P7 (full pipeline rerun on RunPod) is the only remaining task.**

## TL;DR for new session

1. **Mentor (Georg) found 3 stacked bugs** in attribution measurement (cherry-picked as `f307c65` on `l15-refactor`). The fix changes `circuit-tracer`'s measurement basis from `pre_feedforward_layernorm.output[L]` → `hook_resid_post[L]`, which matches where Stage 01 extracts the refusal direction. **All previous Stage 02/02b/04/05/07/08 outputs are stale until the rerun.** Stage 06 causal results are unaffected and will be reused.
2. **All P1-P6 refactors are landed and tested** (244 passed / 0 failed / 1 skipped). Smoke test on RunPod confirmed identity holds: `Σ edges + baseline ≈ direct_dot`, `attr/dot ratio = 1.6841` (mentor saw 1.73). Backend switched to TransformerLens (the nnsight backend has runtime issues with `hook_resid_post`).
3. **P7 is the only outstanding task**: full pipeline rerun on RunPod. **Smoke first** (3 prompts, ~30 min), then **full** (50 prompts, ~10-12 h).
4. **Priority for new session**: walk through the smoke commands in [P7 — RunPod execution plan](#p7--runpod-execution-plan-only-remaining-task) below, capture the verdict, then launch the full run.
5. All P1-P6 changes are pushed to `origin/l15-refactor` — just `git pull` on RunPod and proceed.

---

## Project overview

Mechanistic interpretability research on **Gemma-3-4b-it** — attributing the model's refusal circuit using Anthropic's circuit-tracer (CLT). Mentor: Georg. Research arc: *"Here are the circuits, they're real, here's what they encode, and here's what happens when you manipulate them."*

- **Repo**: `AutoInterp/Refusal-Lens`, working branch `l15-refactor`
- **Submodule**: `vendor/circuit-tracer` on branch `refusal-lens-residual-stream-hook`, commit `7c6cfa4` (mentor's residual-stream measurement_hook patch)
- **HF dataset**: `moon70/refusal-lens-graphs`

### Team
- **Mahmoud** (user): correlational attribution, Stages 01–05 + 07, frontend
- **Tejas**: causal intervention work on `tejas-circuit-experiments` branch (Script `20_bulletproof_pipeline.py`, ported into our Stage 06)
- **Georg**: mentor — found and fixed the 3 stacked attribution bugs (see P1 below)
- **Ruqiya**: Neuronpedia feature-label extraction (Stage 04b input, still pending)

### Deadlines
- **May 4** — ICML workshop abstract. Need P7 results.
- **NeurIPS** — full paper target. Stage 08 results (per-prompt subcircuits + comply-weighted dissociation) are the headline.

---

## Current session work — P1–P6 (2026-04-28 → 2026-04-29)

### P1: Cherry-pick mentor's bug-fix commit ✅ (`f307c65` on `l15-refactor`)

Brought in `db9b008` from `origin/foundation` with three-bug fix from Georg. Mentor's note: `MENTEE_NOTE_three_bugs.md` at repo root.

| Bug | Description | Fix |
|---|---|---|
| 1 (Stage 03) | Verification was reading `hidden_states[L+1]` (post-block residual, ~30k norm) but circuit-tracer measured at `pre_feedforward_layernorm.output[L]` (post-RMSNorm pre-MLP, ~18 norm). The "MLP=0.4%" finding was a magnitude mismatch, not real coverage. | Forward-hook on `pre_feedforward_layernorm` to compare at the same point. |
| 2 (Stage 02) | `attribute()` was called without `measurement_layer`/`measurement_position`. Metadata claimed L32 pos=-2 but circuit-tracer defaulted to post-stack pos=-1. **Already fixed independently on `l15-refactor`** — no-op merge. | Pass them explicitly. |
| 3 (basis) | r̂ extracted at residual stream (Arditi-style, `hidden_states[L+1]`) but applied as cotangent at `pre_feedforward_layernorm.output[L]` — different basis, ~1700× off in magnitude. | Patch circuit-tracer with new `measurement_hook` parameter. `measurement_hook="hook_resid_post"` injects cotangent at residual stream where r̂ lives. |

Submodule `vendor/circuit-tracer` bumped from `b5300ee` → `7c6cfa44` on the new `refusal-lens-residual-stream-hook` branch. Verification identity from mentor: `Σ edges (-51,205) + baseline (+21,625) = direct_dot (-29,580)` at L=15 hook_resid_post — bit-exact reconstruction of the Anthropic circuit-tracing methodology.

### P2: Thread `measurement_hook` through Stage 02 ✅

- `scripts/pipeline/config.py`: added `MEASUREMENT_HOOK = "hook_resid_post"`, `BACKEND = "transformerlens"`, `SAVE_TOP_FEATURES = 100`.
- `scripts/pipeline/02_run_attribution.py`:
  - New CLI flags `--measurement-hook` (default from config) and `--backend {nnsight, transformerlens}` (default `transformerlens`).
  - `attribute()` call conditionally passes `measurement_hook` kwarg.
  - Backend + measurement_hook recorded in metadata.
  - Saves a new `top_features` field (top-100 by |attribution|); keeps `top50_features` as legacy alias for back-compat with Stage 04.

**Why TransformerLens, not nnsight**: the nnsight backend has a `.grad-on-non-module-output` runtime limitation that triggers `ValueError: Execution complete but '...' grad was not provided` when using `measurement_hook="hook_resid_post"`. Mentor flagged this; TL works bit-exact with Gemma-3-4b-it.

### P3: Hook-aware Stage 03 verification ✅

- `scripts/pipeline/03_verify_attribution.py`:
  - New `_capture_residual` dispatcher reads `metadata.measurement_hook` from Stage 02 output and uses the matching residual point (`hidden_states[L+1]` for `hook_resid_post`, forward-hook on `pre_feedforward_layernorm` for legacy/default).
  - New summary fields: `measurement_hook`, `baseline_offset_mean`, `baseline_offset_std`, `attr_to_dot_ratio_mean`, `attr_to_dot_ratio_std` (renamed from `mlp_ratio_*`).
  - Removed the meaningless "MLP=0.4%" headline.

### P4: Stage 07 per-prompt subcircuit construction ✅

Root-cause fix for the negative-dissociation findings on `run_20260422_015552`: corpus-aggregated top-50 sets were too noisy because a feature in fiction's *aggregate* top-50 fired in only 12% of individual fiction prompts.

- `scripts/pipeline/07_identify_subcircuits.py`:
  - New CLI flag `--sweep-configs "K1:F1,K2:F2,..."` with default `"50:0.5,20:0.5,100:0.2"` (the three configs we agreed for NeurIPS rigor).
  - For each `(K, F)` config: build per-prompt top-K feature sets from Stage 02's `top_features` field, then aggregate by frequency (feature qualifies for the per-condition set iff ≥F fraction of prompts in that condition include it in top-K).
  - Synthesize a derived `feature_labels` whose `conditions_seen` is rebuilt per-prompt-frequency, then run the existing `build_*` rules over it.
  - Each sweep config emits its own files in `07_subcircuits/`: `subcircuits_k50_f50.json` + `_summary.json`, etc. Legacy corpus-aggregated path still emits the canonical `subcircuits.json` for back-compat.
  - New `--graph-mode {multi, single}`, `--skip-legacy`, `--skip-sweep` flags.

**Validated against existing `run_20260422_015552`**: per-prompt subcircuits are 5-10× smaller than legacy corpus sets — exactly the surgical targeting we wanted. `jb_fiction_specific_vs_ctrl` at `k50_f50` = 15 features (vs 52 in legacy); k100_f20 = 14 (warning fires that Stage 02 only saved 50 — needs P7 rerun for full K=100).

### P5: Stage 08 ablation refactor ✅

- `scripts/pipeline/08_ablate_subcircuits.py`:
  - New `--subcircuits-file` flag (default `subcircuits.json`; can target `subcircuits_k50_f50.json` etc.).
  - Per-prompt coverage diagnostic: each (ablation, prompt, condition) gets a `coverage` block recording how many ablation features were in *that prompt's* top-K. Auto-flagged `low_coverage: True` if frac < `--low-coverage-threshold` (default 0.30).
  - New `aggregate_coverage_summary` rolls up per-prompt coverage to per-(ablation, position-mode, condition) means + low-coverage prompt counts.
  - **Comply-weighted summary** (NeurIPS rigor): `aggregate_weighted_summary` adds a `weighted` block per (ablation, position-mode) with:
    - `jb_weighted_recovery_rate` = Σ(per-class rate × baseline_comply) / Σ(baseline_comply across all 5 JBs). Keeps weak JBs (completion 2%, roleplay 18%) in the average without letting them drop the headline number.
    - `ctrl_weighted_break_rate` analogous over ctrl_*.
    - `bare_break_rate` and per-class breakdown unchanged.
  - `--ablate-with-mean` flag stub raises `NotImplementedError` (mean-ablation requires a separate pre-pass; deferred).
  - `ABLATION_SUMMARY.md` updated: removed the stale "MLP=0.02%" claim, added comply-weighted aggregates section + per-prompt coverage diagnostic section.
  - Headline output prints comply-weighted recovery + low-coverage prompt counts.

**Coverage diagnostic on existing buggy-baseline data**: `jb_fiction_specific_vs_ctrl` (52 features) had **12.1% mean coverage** on jb_fiction prompts; **all 50/50 prompts flagged low_coverage** — precisely explaining why recovery_rate was 0% on fiction. After P7 rerun with per-prompt subcircuits the coverage should be much higher (those features ARE in many individual prompts' top-K by construction).

### P6: Local test suite update ✅

Added `T-S7sweep` (9 tests) covering per-prompt subcircuit construction with sweep configs, and `T-S8cov` (8 tests) covering compute_coverage / aggregate_coverage_summary / aggregate_weighted_summary. Fixed `MockArgs` in two existing Stage 07 tests to add the new flags.

| Test stage | Count |
|---|---|
| Pre-P6 baseline | 227 |
| **Post-P6** | **244** |
| Failed | 0 |
| Skipped | 1 (T-A3c snapshot — pre-existing, unrelated) |

```bash
PYTHONPATH=src python3 scripts/pipeline/tests/test_pipeline_local.py --stage all
# 244 passed, 0 failed, 1 skipped — 108s
```

New stage flags: `--stage 07-sweep` and `--stage 08-cov`.

### Smoke verification on RunPod (P3 sanity)

Ran a 1-prompt × 11-condition smoke through Stage 02 + Stage 03 with the new `measurement_hook="hook_resid_post"` + `backend="transformerlens"`:

| Metric | Smoke result | Mentor's reference | Status |
|---|---|---|---|
| `direct_dot` | -28,706 | -29,580 | ✓ same regime (different prompt) |
| `Σ edges` | -48,345 | -51,205 | ✓ same regime |
| `baseline` (= direct_dot − Σ edges) | +19,639 | +21,625 | ✓ residual baseline as expected |
| `attr/dot ratio` | **1.6841** | 1.73 | ✓ in expected [1.5, 2.0] range |
| Per-layer recon error | 0.000807 | — | ✓ near-zero |

n_feat counts now 10,326-18,111 per condition (vs ~5,000 in buggy regime) — confirms the basis change is consequential. **All previous Stage 02/02b/04/05/07/08 outputs are stale.**

---

## P7 — RunPod execution plan (only remaining task)

### Step 0: Pull latest code + verify

```bash
cd /workspace/Refusal-Lens
git pull
git submodule update --init vendor/circuit-tracer
python3 -c "
from circuit_tracer.attribution.attribute import attribute
import inspect
print('measurement_hook:', 'measurement_hook' in inspect.signature(attribute).parameters)
"
# Expected: measurement_hook: True
```

If circuit-tracer was installed non-editably and the new param isn't picked up, reinstall editable:
```bash
pip install -e ./vendor/circuit-tracer
```

### Step 1: SMOKE — 3 prompts end-to-end (~30 min on H100)

Catches wiring errors before the long run. Reuses existing Stage 01 directions (skips ~1h recomputation).

```bash
cd /workspace/Refusal-Lens
SMOKE=data/results/pipeline_runs/full_smoke_$(date +%Y%m%d_%H%M%S)
mkdir -p $SMOKE
ln -s $(pwd)/data/results/pipeline_runs/run_20260422_015552/01_direction $SMOKE/01_direction
echo "SMOKE = $SMOKE"

# Stage 02 — 3 prompts × 11 conditions × 2 modes (~10 min)
PYTHONPATH=src python3 scripts/pipeline/02_run_attribution.py \
    --run-dir $SMOKE --n-prompts 3 --batch-size 256 \
    2>&1 | tee /tmp/smoke_02.log

# Stage 03 — verify identity (Σ edges + baseline ≈ direct_dot)
PYTHONPATH=src python3 scripts/pipeline/03_verify_attribution.py \
    --run-dir $SMOKE --graph-mode multi   2>&1 | tee /tmp/smoke_03_multi.log
PYTHONPATH=src python3 scripts/pipeline/03_verify_attribution.py \
    --run-dir $SMOKE --graph-mode single  2>&1 | tee /tmp/smoke_03_single.log

# Stage 02b — statistical analysis
PYTHONPATH=src python3 scripts/pipeline/02b_statistical_analysis.py \
    --run-dir $SMOKE 2>&1 | tee /tmp/smoke_02b.log

# Stage 04 — HF feature labels (needs HF_TOKEN env if private; otherwise no-auth)
PYTHONPATH=src python3 scripts/pipeline/04_label_features.py \
    --run-dir $SMOKE 2>&1 | tee /tmp/smoke_04.log

# Stage 02c — pack to gzipped JSON
PYTHONPATH=src python3 scripts/pipeline/02c_pack_graphs.py \
    --run-dir $SMOKE 2>&1 | tee /tmp/smoke_02c.log

# Stage 07 — legacy + 3 sweep configs
PYTHONPATH=src python3 scripts/pipeline/07_identify_subcircuits.py \
    --run-dir $SMOKE 2>&1 | tee /tmp/smoke_07.log

# Stage 08 smoke: 3 prompts, 2 ablations, 'all' positions (~5 min)
PYTHONPATH=src python3 scripts/pipeline/08_ablate_subcircuits.py \
    --run-dir $SMOKE --max-prompts 3 --positions all \
    --subcircuits universal_refusal_core,jb_fiction_specific_vs_ctrl \
    2>&1 | tee /tmp/smoke_08.log

# Smoke verdict
echo "==== SMOKE VERDICT ===="
grep -E "DONE!|ERROR|FAIL|Traceback" /tmp/smoke_*.log | head -30
```

**Pass criteria**:
- Stage 02 stdout shows `bare [single]: net=…` lines (no errors)
- Stage 03 prints `attr/dot ratio` ≈ 1.5-2.0 for both modes; `baseline_offset_std` low relative to its mean
- Stage 07 emits `subcircuits.json` + `subcircuits_k50_f50.json` + `subcircuits_k20_f50.json` + `subcircuits_k100_f20.json`
- Stage 08 prints headline `bare_break`, `jb_recovery_avg`, `jb_recovery_weighted` and a coverage diagnostic line

If smoke fails: paste the relevant log block in chat — likely suspects are circuit-tracer install (re-install editable), HF auth (set `HF_TOKEN`), or backend (try `--backend nnsight` only as legacy fallback).

### Step 2: FULL run — 50 prompts (~10-12h total)

Run unattended in tmux:

```bash
tmux new -s rerun
cd /workspace/Refusal-Lens
RUN=data/results/pipeline_runs/run_$(date +%Y%m%d_%H%M%S)
mkdir -p $RUN
echo "RUN = $RUN"

# Stage 01 — recompute directions (~1h). Optional: ln -s existing 01_direction if you trust it.
PYTHONPATH=src python3 scripts/pipeline/01_compute_direction.py \
    --run-dir $RUN 2>&1 | tee $RUN/_01.log

# Stage 02 — full attribution on 50 prompts × 11 conditions × 2 modes (~3-4h on H100)
PYTHONPATH=src python3 scripts/pipeline/02_run_attribution.py \
    --run-dir $RUN --n-prompts 50 --batch-size 256 \
    2>&1 | tee $RUN/_02.log

# Stages 02b / 03 / 04 — CPU-bound, ~10 min total
PYTHONPATH=src python3 scripts/pipeline/02b_statistical_analysis.py --run-dir $RUN 2>&1 | tee $RUN/_02b.log
PYTHONPATH=src python3 scripts/pipeline/03_verify_attribution.py --run-dir $RUN --graph-mode multi  2>&1 | tee $RUN/_03_multi.log
PYTHONPATH=src python3 scripts/pipeline/03_verify_attribution.py --run-dir $RUN --graph-mode single 2>&1 | tee $RUN/_03_single.log
PYTHONPATH=src python3 scripts/pipeline/04_label_features.py --run-dir $RUN 2>&1 | tee $RUN/_04.log

# Stage 02c — pack graphs to JSON.gz (~5-10 min)
PYTHONPATH=src python3 scripts/pipeline/02c_pack_graphs.py --run-dir $RUN 2>&1 | tee $RUN/_02c.log

# Stage 07 — produces legacy + 3 sweep configs (<1 min)
PYTHONPATH=src python3 scripts/pipeline/07_identify_subcircuits.py --run-dir $RUN 2>&1 | tee $RUN/_07.log

# Stage 06 — REUSE existing causal results (independent of attribution basis)
ln -s $(pwd)/data/results/pipeline_runs/run_20260422_015552/06_causal $RUN/06_causal

# Stage 08 — full ablation on the strongest sweep config (~14h with all 5 default subcircuits)
PYTHONPATH=src python3 scripts/pipeline/08_ablate_subcircuits.py \
    --run-dir $RUN --positions both \
    --subcircuits-file subcircuits_k50_f50.json \
    2>&1 | tee $RUN/_08_k50_f50.log

# Detach: Ctrl-b d. Reattach: tmux attach -t rerun
```

### Step 3: After completion — push to HF + record headline numbers

```bash
PYTHONPATH=src python3 scripts/pipeline/push_graph_data.py \
    --run-dir $RUN --source 02c \
    --dataset-repo moon70/refusal-lens-graphs
```

**Headline numbers to record for NeurIPS write-up**:
1. **Stage 03**: `attr_to_dot_ratio_mean` (multi + single), `baseline_offset_mean/std` from `03_verification/verification_results.json["summary"]`.
2. **Stage 07** `jb_vs_ctrl_contrast` per class, for each sweep config (compare to legacy). Files: `subcircuits.json`, `subcircuits_k50_f50.json`, `subcircuits_k20_f50.json`, `subcircuits_k100_f20.json`.
3. **Stage 08** for `subcircuits_k50_f50.json` from `08_ablation/ablation_summary.json`:
   - Per-ablation `summary["per_ablation"][abl]["positions"][mode]["weighted"]["jb_weighted_recovery_rate"]` + `bare_break_rate` + `ctrl_weighted_break_rate`
   - Per-class dissociation Δ from `summary["dissociation"]`
   - Mean coverage / low-coverage prompt counts from `summary["coverage"]`

---

## Stage-by-stage status (2026-04-29)

| # | Stage | Code status | Validated post-fix? | Notes |
|---|---|---|---|---|
| 01 | `01_compute_direction.py` | ✅ | ✅ (existing run reusable — independent of fix) | Per-layer @ pos=-2 + per-position at L15. |
| 02 | `02_run_attribution.py` | ✅ + `--measurement-hook` + `--backend` flags | ✅ (1-prompt smoke) | TL backend; saves top-100 features. |
| 02b | `02b_statistical_analysis.py` | ✅ no changes | ⏳ pending P7 | Will produce new effect sizes. |
| 02c | `02c_pack_graphs.py` | ✅ no changes | ⏳ pending P7 | — |
| 03 | `03_verify_attribution.py` | ✅ hook-aware | ✅ (smoke ratio = 1.6841) | Identity check `Σ edges + baseline ≈ direct_dot`. |
| 04 | `04_label_features.py` | ✅ no changes | ⏳ pending P7 | HF feature labels. |
| 04b | `04b_delphi_labels.py` | ⏳ not started | — | Post-NeurIPS / lower priority. |
| 05 | `05_visualize_circuits.py` | ✅ no changes | ⏳ pending P7 | Browser sanity check after rerun. |
| 06 | `06_causal_intervention.py` | ✅ done | ✅ on `run_20260422_015552` (independent of fix) | **Symlink existing results into new run dir.** |
| 07 | `07_identify_subcircuits.py` | ✅ + sweep configs (P4) | ✅ (legacy + 3 sweep configs on existing data) | Emits `subcircuits.json` + `subcircuits_k{K}_f{F:02.0f}.json` ×3. |
| 08 | `08_ablate_subcircuits.py` | ✅ + coverage + weighted (P5) | ⏳ pending P7 | `--subcircuits-file`, per-prompt coverage diagnostic, comply-weighted summary. |

---

## Commits on `l15-refactor` from this session (all pushed)

```
3c463f0 local tests passed                                    [P5/P6]
37e56a0 refactored stage 07 to handle per prompt feature segmentation  [P4]
3ec0492 add --backend flag for stage 02                       [P2/P3]
4c5188d f307c65 on l15-refactor; vendor/circuit-tracer → 7c6cfa4 with measurement_hook API, config.py with new measurement hook from georg  [P1 + config bump]
f307c65 Fix three stacked bugs in attribution pipeline        [P1 cherry-pick]
```

Files modified across the session:
- `scripts/pipeline/config.py` — `MEASUREMENT_HOOK`, `BACKEND`, `SAVE_TOP_FEATURES`
- `scripts/pipeline/02_run_attribution.py` — `--measurement-hook`, `--backend`, `top_features` field
- `scripts/pipeline/03_verify_attribution.py` — hook-aware verification, baseline disclosure
- `scripts/pipeline/07_identify_subcircuits.py` — sweep configs + per-prompt aggregation
- `scripts/pipeline/08_ablate_subcircuits.py` — subcircuits-file, coverage, weighted
- `scripts/pipeline/tests/test_pipeline_local.py` — T-S7sweep + T-S8cov tests, MockArgs fixes
- `scripts/pipeline/tests/test_runpod_1_4.py` — hook-aware Stage 03 check
- `vendor/circuit-tracer` (submodule) → `7c6cfa4`
- `MENTEE_NOTE_three_bugs.md` (new) — Georg's three-bug writeup

---

## New gotchas (this session)

1. **TransformerLens, not nnsight, for Stage 02**. The `hook_resid_post` measurement hook needs the TL backend; nnsight's `.grad-on-non-module-output` limitation breaks it at runtime with `ValueError: Execution complete but '...' grad was not provided`. Mentor verified TL works bit-exact with Gemma-3-4b-it on the patched circuit-tracer. Memory note "Gemma-3 requires nnsight" is out of date.

2. **`circuit-tracer` editable install required**. After `git submodule update`, the new `measurement_hook` API only works if circuit-tracer is installed editable (`pip install -e ./vendor/circuit-tracer`). On a fresh pod the first run can fail with `unexpected keyword argument 'measurement_hook'` if the install is non-editable. Detect with `python3 -c "from circuit_tracer.attribution.attribute import attribute; import inspect; print('measurement_hook' in inspect.signature(attribute).parameters)"`.

3. **All previous attribution-derived outputs are stale**. Stage 02/02b/04/05/07/08 outputs from `run_20260422_015552` were produced before the bug-3 fix. Don't compare numbers to them — they're at the wrong basis. Stage 06 (causal intervention) and Stage 01 (direction extraction) are unaffected and reusable.

4. **Stage 07 sweep configs default to `"50:0.5,20:0.5,100:0.2"`**. K=100 needs Stage 02 to have saved at least 100 features per (prompt, condition, mode); current `SAVE_TOP_FEATURES = 100` covers this. Existing run_20260422_015552 only saved 50 → k100_f20 sweep on that run logs a `WARN: capping to k=50` message.

5. **`baseline_offset` is real, not a bug**. After hook_resid_post fix, `direct_dot = Σ edges + baseline` where `baseline ≈ +20k` at L15. This is the genuine linearization offset (accumulated transcoder b_dec terms propagating through frozen-attention layers without final-norm damping). Disclose in any absolute claim; relative comparisons unaffected.

6. **Stage 06 reuse via symlink**. The new run's `06_causal/` should symlink to `run_20260422_015552/06_causal/` rather than rerunning Stage 06 — the causal intervention is independent of attribution basis and the existing 96.7% pro-flip / 100% anti-flip / 10/10 benign result still holds. Save ~1h.

---

## Historical context (pre-P1 work, preserved for reference)

### Phase B results — Stage 06 full causal intervention (2026-04-22, on `run_20260422_015552`)

**Headline (still valid, will be reused via symlink in P7)**:

| Experiment | Result |
|---|---|
| L15 pro-refusal add (JB COMPLY → REFUSE) | **87/90 = 96.7%** |
| L15 anti-refusal sub (bare REFUSE → COMPLY) | **49/49 = 100%** |
| L15 benign force-refuse | **10/10 = 100%** |
| Coherence | 100% on all 146 flips |

Per-class pro-refusal flip:

| Class | Comply baseline | Flipped | Flip rate |
|---|---|---|---|
| analytical | 27/50 | 27 | 100% |
| roleplay | 9/50 | 9 | 100% |
| completion | 1/50 | 1 | 100% (n=1 tiny) |
| cognitive_reframe | 33/50 | 32 | 97% |
| fiction | 20/50 | 18 | **90%** (hardest to patch) |

**Bidirectional symmetry confirmed**: L15 `r = mean_harmful − mean_harmless` is THE refusal axis, manipulable in both directions. Fiction's lower flip rate is a candidate for a Stage 08 dissociation test post-P7.

Outputs (committed on `l15-refactor`, will be symlinked into new run):
- `06_causal/causal_results.json` — per-prompt baseline + intervention records (4.4 MB)
- `06_causal/causal_summary.json` — aggregated flip rates + per-class breakdown
- `06_causal/flip_rate_by_class.png`, `intervention_symmetry.png`
- `06_causal/FLIP_RATE_SUMMARY.md`

### Phase A validation findings — STALE post-P1

These numbers are from `run_20260422_015552` produced before mentor's bug-3 fix. They will be replaced by P7 numbers and should NOT be used in the NeurIPS abstract.

- ❌ Stage 02b effect sizes (e.g. cognitive_reframe d=-2.05, analytical d=-4.07) — basis-dependent, will shift.
- ❌ Stage 03 "MLP=0.02%" — magnitude mismatch artifact, removed in P3.
- ❌ Stage 04 1,353 features — feature set will shift with new basis.
- ❌ Stage 07 `jb_specific_frac` per class (38.6% / 34.2% / 34.2% / 20.0% / 18.4%) — corpus-aggregated AND wrong basis. P4 sweep replaces this.
- ❌ Peak-layer "L14 hotspot" finding — likely persists qualitatively but absolute numbers shift.

After P7, regenerate all of these. The P5 coverage diagnostic on the existing run (`jb_fiction_specific_vs_ctrl` 12% mean coverage, 50/50 prompts low-coverage) is what proves the existing numbers were misleading.

### Stage 08a code (P5 supersedes the original Apr 24 spec)

The original Stage 08a was implemented on Apr 24 with corpus-aggregated subcircuits and unweighted recovery rates. P5 (this session) added the per-prompt coverage diagnostic + comply-weighted summary on top of that code. The validation bar from Apr 24 ("dissociation_delta ≥ +20pp on at least 2 of 3 class-specific ablations") still applies — but the right `--subcircuits-file` to test it is now `subcircuits_k50_f50.json`, not `subcircuits.json`.

---

## Where to find things

| What | Path |
|---|---|
| Main pipeline stages | `scripts/pipeline/0X_*.py` |
| Shared helpers | `scripts/pipeline/{utils,utils_viz,config}.py` |
| Local tests | `scripts/pipeline/tests/test_pipeline_local.py` |
| RunPod test (GPU) | `scripts/pipeline/tests/test_runpod_1_4.py` |
| Controlled dataset | `dataset/refusal_lens_controlled_dataset.json` |
| **Mentor's three-bug writeup** | `MENTEE_NOTE_three_bugs.md` |
| Latest run (post-fix, pending P7) | `data/results/pipeline_runs/run_<TIMESTAMP>/` |
| Pre-fix reference run (Stage 06 reusable) | `data/results/pipeline_runs/run_20260422_015552/` |
| Tejas's bulletproof causal script | `git show origin/tejas-circuit-experiments:data/tejas_experiments/scripts/20_bulletproof_pipeline.py` |
| HF push helper (JSON.gz bundle) | `scripts/pipeline/push_graph_data.py` |
| `.pt → JSON.gz` packer | `scripts/pipeline/02c_pack_graphs.py` |
| Patched circuit-tracer | `vendor/circuit-tracer` @ `7c6cfa4` (`refusal-lens-residual-stream-hook` branch) |

---

## Schema reminders

### Stage 02 output (consumed by every downstream stage)

```jsonc
{
  "metadata": {
    "n_prompts": 50,
    "model": "google/gemma-3-4b-it",
    "transcoder": "...",
    "measurement_layer": 15,
    "measurement_hook": "hook_resid_post",  // NEW (P2)
    "backend": "transformerlens",            // NEW (P2)
    "modes": {"multi": [-5, -3, -2], "single": [-2]},
    "dataset": "controlled"
  },
  "results": [
    {
      "prompt_idx": 0, "prompt_id": 1, "prompt": "...", "topic": "...",
      "conditions": {
        "bare": {
          "prefix": "",
          "graphs": {
            "multi":  {"net": ..., "top_features": {...}, "top50_features": {...}},
            "single": {...}
          }
        },
        "jb_fiction": {...}, "ctrl_fiction": {...},
        // ... 11 conditions: bare + 5 jb_* + 5 ctrl_*
      },
      "feature_comparison": {...}
    }
  ]
}
```

`top_features` (NEW, P2): top-100 features by |attribution|. `top50_features` (legacy alias): first 50 of `top_features`.

### Condition names (11 per prompt)

```
bare
jb_roleplay,    ctrl_roleplay
jb_fiction,     ctrl_fiction
jb_analytical,  ctrl_analytical
jb_completion,  ctrl_completion
jb_cognitive_reframe, ctrl_cognitive_reframe
```

### Config constants (`scripts/pipeline/config.py`)

```python
MEASUREMENT_LAYER = 15
MEASUREMENT_POSITION = -2
MEASUREMENT_HOOK = "hook_resid_post"      # NEW (P2)
BACKEND = "transformerlens"                # NEW (P2)
SAVE_TOP_FEATURES = 100                    # NEW (P2/P4) — top-K per condition saved by Stage 02
TARGET_POSITIONS_MULTI = [-5, -3, -2]
TARGET_POSITIONS_SINGLE = [-2]
PER_POSITION_LAYER = 15
CONTROLLED_DATASET_PATH = REPO_ROOT / "dataset" / "refusal_lens_controlled_dataset.json"
STAGE_08_DEFAULT_SUBCIRCUITS = (
    "universal_refusal_core", "ctrl_shared_refusal",
    "jb_fiction_specific_vs_ctrl",
    "jb_analytical_specific_vs_ctrl",
    "jb_cognitive_reframe_specific_vs_ctrl",
)
```

---

## Workflow conventions

### Testing
```bash
# Local (no GPU)
PYTHONPATH=src python3 scripts/pipeline/tests/test_pipeline_local.py --stage all
# 244 passed / 0 failed / 1 skipped (Apr 29)

# Stage flags: 01, 01-a5, 02, 02b, 03, 03-a4, 04-a7, 04-a8, 04-schema,
#              06, 07, 07-ctrl, 07-sweep (NEW), 08, 08-cov (NEW),
#              utils, utils-viz, all
```

### Code style
- Every Python file starts with `from __future__ import annotations`
- Default to no comments; only comment the WHY
- Don't add error handling / validation / fallbacks for scenarios that can't happen
- Break back-compat on schema when it's the right call (we just did a major schema rev)

### Git
- Commit to `l15-refactor`, never `main`
- Don't push without user approval (unless they've asked you to)
- Keep `vendor/circuit-tracer` submodule pointed at `7c6cfa4` (`refusal-lens-residual-stream-hook` branch)

---

## Final note for new session

You're picking up at the cleanest possible handoff: all refactor + bug-fix work is done, all tests pass, and the only outstanding work is a single RunPod run sequence (smoke + full). Walk through the [P7 — RunPod execution plan](#p7--runpod-execution-plan-only-remaining-task) above end-to-end. Smoke first, share the verdict, then launch the full run unattended in tmux. Capture the headline numbers from `03_verification`, `07_subcircuits/subcircuits_k50_f50.json`, and `08_ablation/ablation_summary.json` for the NeurIPS write-up.
