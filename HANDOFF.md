# Refusal-Lens — Pipeline State & Remaining Tasks

**Last updated 2026-05-01** — **P7 COMPLETE.** Full pipeline finished on `data/results/pipeline_runs/run_20260430_023247`; HF push complete (raw `.pt`, packed `.json.gz`, run metadata at `moon70/refusal-lens-graphs/runs/run_20260430_023247/`). Comprehensive analysis report at `REPORT_run_20260430_023247.md`. Local 500 GB volume can be deleted; pod torn down. Three new follow-up tracks queued: (a) canonical_pro_refusal ablation, (b) frontend interactive ablation server (~70% pre-built), (c) repo cleanup before public release.

## TL;DR for new session

1. **P7 complete, results pushed.** Headline numbers locked in. Read `REPORT_run_20260430_023247.md` for the full analysis. Five-bullet summary:
   - Stage 03 linearization identity holds: `Σ edges + baseline = direct_dot`, attr/dot ratio **1.659 ± 0.025** (single), baseline offset **+19,419 ± 574** — matches mentor's 1.73, +21,625 reference.
   - Stage 06 causal: **100% / 98% / 100%** (pro-refusal flip / anti-refusal sub / benign force-refuse). Up from 96.7% / 100% / 100% on prior run. Fiction now 100% (was 90%).
   - Stage 08 unblocked: `universal_refusal_core` (26 feat) recovers **26.6%** of weighted JB complies; `jb_fiction_specific_vs_ctrl` passes dissociation (Δ = +16.0 pp); analytical/cog_reframe show **negative dissociation** (causally promiscuous despite correlationally selective).
   - Layer story: L0–L11 cumulative −20,711 (anti-refusal accumulator), L33 alone +32,125 (pro-refusal flip). The previous "L14 hotspot" was a basis-mismatch artifact.
   - Direction ≫ subcircuit potency: Stage 06 r-vector flips fiction at 100%, Stage 08 fiction-specific features ablate at 30%. The residual stream integrates a more distributed signal than any sparse feature subset captures.

2. **Highest-leverage follow-up: ablate `canonical_pro_refusal`** (88-feature subcircuit, "all 5 JBs but not bare"). Not in Stage 08 default; was the missing experiment from this run. Plan in `EXPERIMENT_PLAN_canonical_pro_refusal_and_frontend_ablation.md`. Single Stage 08 invocation, ~3–6 h on H100.

3. **Frontend interactive ablation is ~70% pre-built.** `scripts/pipeline/ablation_server.py` (224 lines, FastAPI) + `scripts/pipeline/05_frontend_patches/feature-cart.js` (click-to-cart UI). Needs: end-to-end test, extension to `compare.html`, supernode-grouping UI. Also in the experiment plan above.

4. **Two upstream patches still pending merge into mentor's branch** (vendor/circuit-tracer):
   - The list-typed `measurement_position` fix lives only on our submodule branch `refusal-lens-multi-position-fix`. Mention to Georg post-deadline.

5. **Repo cleanup before public release** — see [Repo cleanup audit](#repo-cleanup-audit-recommendations) below. Several non-pipeline scripts (`scripts/run_*_experiments.py`, ancillary pipeline shells, `data/results/pipeline_runs/run_20260422_015552` 75 GB) are candidates for removal.

---

## Project overview

Mechanistic interpretability research on **Gemma-3-4b-it** — attributing the model's refusal circuit using Anthropic's circuit-tracer (CLT). Mentor: Georg. Research arc: *"Here are the circuits, they're real, here's what they encode, and here's what happens when you manipulate them."*

- **Repo**: `AutoInterp/Refusal-Lens`, working branch `l15-refactor`
- **Submodule**: `vendor/circuit-tracer` on branch `refusal-lens-multi-position-fix` (one commit on top of mentor's `7c6cfa4` measurement_hook patch — fixes list-typed `measurement_position` crash, see [P7 — multi-position fix](#p7-session-2026-04-29-evening--multi-position-fix--launch))
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

### P7 (session 2026-04-29 evening) — Multi-position fix + launch

This session brought P7 from "ready-to-run" to "launched and validated end-to-end". Key landings:

#### Orchestrator: `scripts/pipeline/run_p7.sh`

Single bash entrypoint. Self-relaunches into a detached tmux session named `p7` so it survives SSH disconnects. Modes: `smoke` (3 prompts, ~1.5 h), `full` (50 prompts, ~17-28 h), `both` (chain). Every stage runs **fresh in each new run dir** — no symlinks from prior runs, so the bug fixes are exercised end-to-end. Stage order in both phases:

```
01 → 02 (--save-graphs) → 02b → 03×2 (multi+single) → 04 → 02c → 07 → 06 → 08 (--skip-baseline)
```

Auto-pushes raw `.pt` (`push_raw_graphs.py`), packed `.json.gz` (`push_graph_data.py --source 02c`), and run metadata (`push_run.py --skip-graphs`) to HF after the full completes. `--git-push-results` flag optionally also commits small result JSONs (03/04/06/07/08 + 02b_stats) back to `l15-refactor`.

Logs: per-stage `<run-dir>/_<stage>.log`, combined `/tmp/p7_pipeline_<ts>.log` (symlinked to `/tmp/p7_pipeline.log`). Markers: `<run-dir>/.P7_DONE` on success; `<run-dir>/.P7_FAIL` (with phase/step name) on failure plus tail of dying stage's log.

Stage 08 invocations carry `--max-new-tokens 80` (Tejas's overnight optimization — refusal/comply classification is decided in first ~30 tokens), `--skip-baseline` (reuses Stage 06's), and for the full also `--resume --checkpoint-every 5` (free safety net for the 9-21 h ablation pass).

#### Multi-position fix in vendor/circuit-tracer

Smoke first attempt (`full_smoke_20260429_150126`) hit `ERROR — full(): argument 'fill_value' (position 2) must be Number, not list` on every `[multi]` graph for every condition. Single mode worked. Root cause was in mentor's `7c6cfa4` patch:

```python
# vendor/circuit-tracer/circuit_tracer/attribution/attribute_transformerlens.py L225 (and same in attribute_nnsight.py L243)
positions=torch.full((batch.shape[0],), _mp),  # _mp is a list when measurement_position=[-5,-3,-2]
```

Mentor's API accepts both scalar and list `measurement_position`, but the Phase-3 inner loop hardcoded `torch.full(shape, scalar)`. Fixed by branching on `isinstance(_mp, (list, tuple))`:

```python
positions = (
    torch.as_tensor(_mp[i : i + batch.shape[0]], dtype=torch.long)
    if isinstance(_mp, (list, tuple))
    else torch.full((batch.shape[0],), _mp)
)
```

Identical patch in both backends. Committed on a new submodule branch `refusal-lens-multi-position-fix` (one commit on top of `7c6cfa4`). `.gitmodules` retargeted. **Coordinate with Georg post-ICML to merge upstream into `refusal-lens-residual-stream-hook`.**

#### Other Stage-02 / Stage-08 fixes via the orchestrator

Two additional flag-coverage gaps caught during smoke and added to the script:

| Issue | Fix in `run_p7.sh` |
|---|---|
| Stage 02 didn't save `.pt` files → Stage 02c failed with "graphs/ does not exist" → push_raw_graphs would also have failed | Added `--save-graphs` to both smoke and full Stage 02 calls |
| Stage 08 smoke read legacy `subcircuits.json` (the orchestrator default) instead of the per-prompt `subcircuits_k50_f50.json` that the full uses → smoke didn't validate the production headline path | Smoke now passes `--subcircuits-file $SUBCIRCUITS_FILE` (default `subcircuits_k50_f50.json`), same as full |
| Stage 08 max_new_tokens defaulted to 200 → wall clock 2.5× longer than necessary | Both smoke and full Stage 08 now pass `--max-new-tokens 80` |

#### Smoke validation (`full_smoke_20260429_170345`, N=3)

Wall: ~1.8 h end-to-end. Identity / correctness / wiring all green:

| Check | Smoke result | Reference / verdict |
|---|---|---|
| Stage 03 `attr/dot ratio` (single) | **1.6988** | mentor: 1.73 — ✓ in expected `[1.5, 2.0]` |
| Stage 03 `attr/dot ratio` (multi) | **1.8170** | ✓ multi-position fix working; both modes report numbers, no `ERROR — full(): ...` lines anywhere |
| Stage 03 `baseline_offset` (single) | **+20,221 ± 532** | mentor: +21,625 — ✓ within 7%; per-prompt diffs [+19,639, +20,100, +20,925] — stable |
| Stage 03 reconstruction error | 0.0008 – 0.0036 | ✓ linearization is essentially exact |
| Stage 06 pro / anti / benign | **100% / 100% / 100%** (5/5, 3/3, 10/10) | matches/improves on prior `run_20260422_015552` 96.7/100/100; perfect bidirectional intervention |
| Stage 07 sweep configs produced | `subcircuits.json` + 3 sweep configs | ✓ all invariants pass; `subcircuits_k50_f50.json` is 5-10× smaller than legacy (universal=25, jb_*_specific=8-17 features) |
| Stage 08 activation audit | jb_fiction features: 57% top-50 hit rate on `jb_fiction`, **0% on all `ctrl_*`**, 2-12% on other JBs | ✓ class-specificity confirmed |
| Stage 08 dissociation Δ (target=fiction) | **-37.5pp (uninterpretable)** | ⚠ vacuous: 0 baseline complies on jb_fiction at N=3, denominator=0. NOT a failure of the subcircuit — small-N artifact. At N=50 fiction will have ~20 complies (per Phase B). |

#### New per-prompt drilldown: `scripts/pipeline/08_recovery_drilldown.py`

Post-hoc script that reads `08_ablation/ablation_results.json` and emits a comply-baseline-only per-prompt view. **Doesn't change Stage 08 itself** (zero risk to the long-running ablation pass; can be run any time after Stage 08 completes). For each baseline-COMPLY JB case, emits `(prompt_id, jb_class, ablation, baseline_response, ablated_response, flipped_to_refuse, ablated_coherent, coverage)`.

Outputs (in `<run-dir>/08_ablation/`):
- `recovery_drilldown.json` — `by_class[cls].per_ablation[abl][pos_mode].prompts[]` with both `recovery_rate` and `coherent_recovery_rate`
- `recovery_drilldown.csv` — flat one-row-per-(prompt × ablation × pos_mode) for spreadsheet review

Validated against the smoke output: produces 10 records across 5 baseline complies, recovery rates match the aggregate `ablation_summary.json` exactly.

Run after the full completes:
```bash
PYTHONPATH=src python3 scripts/pipeline/08_recovery_drilldown.py \
    --run-dir data/results/pipeline_runs/run_<TS>
```

#### Paper-grade insights surfaced during smoke validation

These are real findings from the smoke that should land in the writeup (caveat: N=3, will be confirmed at N=50):

1. **L33 dominates the late-layer pro-refusal flip.** Stage 03 per-layer decomposition: L0–L19 cumulative ≈ -28,200 (anti-refusal, with L11/L10/L9 contributing -4,855 / -3,969 / -3,437 respectively), L20–L32 cumulative ≈ +5,000 (mostly pro-refusal), and **L33 alone = +33,160** — overrides everything else combined. The two-stage circuit story is cleaner than the previous "L14 hotspot" framing (which was a basis-mismatch artifact).
2. **Effect sizes 2-3× larger in the corrected basis.** `vs_bare` Cohen's d went from -2.05 → -4.99 (cognitive_reframe), -4.07 → -8.62 (analytical). At N=50 these will easily clear significance.
3. **Class-specific subcircuits ARE specific** (per activation audit, smoke). `jb_fiction_specific_vs_ctrl` features fire on 57% of jb_fiction prompts vs 0% on every `ctrl_*` and 2-12% on other JBs.
4. **Empirical Stage 08 wall-clock model**: 65 s/gen at max_new_tokens=200 with 26-feature subcircuits on H100 SXM. HANDOFF's 14 h estimate proved ~3× optimistic; realistic budget for full at max_new_tokens=80 + 12-feature avg subcircuits + `--positions all` is **17-28 h** for the whole P7 pipeline.

### Smoke verification on RunPod (P3 sanity, pre-this-session)

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

## P7 — Run results (`run_20260430_023247`, completed 2026-05-01)

**Wall: ~24 h end-to-end on H100 SXM** (Stage 02 = 205 min, Stage 06 = 54 min, Stage 08 = 985 min, push = ~3 h via `upload_large_folder` for the 355 GB raw `.pt` archive). All HF pushes complete; report at `REPORT_run_20260430_023247.md`.

**Headline numbers (full N=50 confirmation)**:

| Stage | Metric | Value | Prior run (`run_20260422_015552`, buggy basis) |
|---|---|---|---|
| 03 | attr/dot ratio (single, mean) | **1.6594 ± 0.025** | n/a (different schema) |
| 03 | baseline_offset (single, mean) | **+19,419 ± 574** | n/a |
| 06 | pro-refusal flip rate | **100 % (89/89)** | 96.7 % (87/90) |
| 06 | anti-refusal flip rate | **98 % (49/50)** | 100 % (49/49) |
| 06 | benign force-refuse rate | **100 % (10/10)** | 100 % |
| 06 | fiction flip rate (was holdout) | **100 % (19/19)** | 90 % (18/20) |
| 07 | jb_specific_frac (k50_f50, mean across 5 classes) | **25 % (range 13–33)** | similar but at corpus-level union, much noisier |
| 08 | universal_refusal_core JB weighted recovery | **26.6 %** | vacuous (12 % coverage) |
| 08 | jb_fiction_specific_vs_ctrl dissociation Δ | **+16.0 pp** ✓ | vacuous |
| 08 | jb_analytical_specific_vs_ctrl Δ | **−13.8 pp** ✗ | vacuous |
| 08 | jb_cognitive_reframe_specific_vs_ctrl Δ | **−11.2 pp** ✗ | vacuous |
| 08 | mean coverage on target class (per-prompt sweep) | **83–96 %** | 12.1 % |
| Stage 03 layer story | L0–L11 cumulative | **−20,711** (anti-refusal accumulator) | obscured by basis bug |
| Stage 03 layer story | L33 alone | **+32,125** (pro-refusal flip) | "L14 hotspot" was a basis artifact |

**Two pivotal new findings (post-fix):**

1. **Negative dissociation on analytical and cognitive_reframe.** Their "class-specific" features are correlationally selective (top-50 hit rate 88 % / 68 % on target) but causally promiscuous (ablation recovers OTHER classes more than target). Only fiction passes the per-class dissociation test. This is publishable as a methodological caution against treating top-K class-specific features as faithful per-class circuits at this K/F regime.
2. **Direction ≫ subcircuit potency.** Stage 06 r-vector intervention recovers 100 % of fiction; Stage 08 fiction-specific feature ablation recovers 30 %. The residual stream integrates a more distributed signal than any sparse feature subset captures. This argues against framing the JB circuit as "N specific features" and motivates the still-unrun ablation of `canonical_pro_refusal` (88 features, "all 5 JBs but not bare").

See the [report](REPORT_run_20260430_023247.md) for the full breakdown and figures.

> *Historical execution-plan reference (kept for reproduction).* `bash scripts/pipeline/run_p7.sh` chains a fresh smoke → verdict → fresh full → HF push inside a detached tmux session `p7`. Every stage runs fresh in each new run dir; nothing is symlinked from prior runs. Add `--git-push-results` to commit small result JSONs back to `l15-refactor`. Detailed step-by-step is preserved below for debugging a failed sub-step.

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

## Stage-by-stage status (2026-05-01, post-P7)

| # | Stage | Code status | Validated at N=50? | Notes |
|---|---|---|---|---|
| 01 | `01_compute_direction.py` | ✅ | ✅ run_20260430 | Per-layer @ pos=-2 + per-position at L15. ‖r‖ at L15 = 3,101.2; at L32 = 20,873.2. Runs fresh each time, no longer reused via symlink. |
| 02 | `02_run_attribution.py` | ✅ + `--measurement-hook` + `--backend` + multi-position-fix in vendor | ✅ run_20260430 (1,100 graphs, both modes, 205 min) | TL backend; orchestrator passes `--save-graphs`. n_features per condition ~12,500 — 2× the buggy regime. |
| 02b | `02b_statistical_analysis.py` | ✅ no changes | ✅ run_20260430 | vs_bare single-mode Cohen's d: analytical −6.33, cog_reframe −5.87, fiction −3.75. **2–3× the buggy basis.** vs_ctrl (the rigorous comparison) shows only cog_reframe (d=−2.03) and analytical (d=−1.20) clear |d|>1. |
| 02c | `02c_pack_graphs.py` | ✅ no changes | ✅ run_20260430 (1,100 .pt → 1,100 .json.gz, 6.94× compression, 484 MB total) | Requires Stage 02 `--save-graphs`. |
| 03 | `03_verify_attribution.py` | ✅ hook-aware | ✅ run_20260430 (single ratio 1.6594 ± 0.025; baseline +19,419 ± 574; multi ratio 1.8170) | Linearization identity `Σ edges + baseline ≈ direct_dot` verified to <0.4% per prompt. Mentor reference 1.73 / +21,625. |
| 04 | `04_label_features.py` | ✅ no changes | ✅ run_20260430 (100 % label coverage on union top-50) | HF feature labels. |
| 04b | `04b_delphi_labels.py` | ⏳ not started | — | Post-NeurIPS / lower priority. Awaiting Ruqiya. |
| 05 | `05_visualize_circuits.py` | ✅ no changes | ⏳ pending — not in orchestrator | Frontend regeneration: `PYTHONPATH=src python3 scripts/pipeline/05_visualize_circuits.py --run-dir <run-dir>`. **Companion `feature-cart.js` + `ablation_server.py` are pre-built — see frontend ablation plan.** |
| 06 | `06_causal_intervention.py` | ✅ done | ✅ run_20260430 (100 % / 98 % / 100 %; fiction now 100 %, was 90 %) | Bidirectional symmetry confirmed at N=50. Runs fresh in each P7 run dir. |
| 07 | `07_identify_subcircuits.py` | ✅ + sweep configs (P4) | ✅ run_20260430 (all 4 configs produced; k50_f50 sizes: universal=26, ctrl_shared=4, jb_*_specific=8–17) | Emits `subcircuits.json` (legacy) + `subcircuits_k{K}_f{F:02.0f}.json` ×3 (sweep). |
| 08 | `08_ablate_subcircuits.py` | ✅ + coverage + weighted (P5) | ✅ run_20260430 (5 ablations × 11 conditions × 50 prompts × 1 position-mode in 985 min) | Coverage 83–96 % on target class (vs 12 % on prior buggy run). **`canonical_pro_refusal` not in default — open follow-up.** |
| 08b | `08_recovery_drilldown.py` | ✅ done | ⏳ pending — run on local copy of run_20260430 | Post-hoc; emits `recovery_drilldown.json` + `.csv`. Run after pulling result JSONs from HF. |
| 09 | (proposed) attention-head attribution | ⏳ not started | — | circuit-tracer supports it; wiring is the only barrier. ICML stretch / NeurIPS likely. |

---

## Commits on `l15-refactor` from this session (all pushed)

```
# === P7 launch session (2026-04-29 evening) ===
<NEW>  Stage 08 smoke + full: --max-new-tokens 80, smoke now uses subcircuits_k50_f50
<NEW>  Stage 02 smoke + full now pass --save-graphs (Stage 02c needs the .pt files)
<NEW>  Bump vendor/circuit-tracer to multi-position fix branch
<NEW>  Add P7 unified pipeline runner (smoke + full + HF push, tmux-persistent)

# Plus on the SUBMODULE (vendor/circuit-tracer, refusal-lens-multi-position-fix branch):
<NEW>  Fix list-typed measurement_position in measurement_hook patch

# === Prior P1-P6 session (still on l15-refactor) ===
3c463f0 local tests passed                                    [P5/P6]
37e56a0 refactored stage 07 to handle per prompt feature segmentation  [P4]
3ec0492 add --backend flag for stage 02                       [P2/P3]
4c5188d f307c65 on l15-refactor; vendor/circuit-tracer → 7c6cfa4 with measurement_hook API, config.py with new measurement hook from georg  [P1 + config bump]
f307c65 Fix three stacked bugs in attribution pipeline        [P1 cherry-pick]
```

(The four `<NEW>` commits above are from this session — replace with actual SHAs once `git log` shows them on `origin/l15-refactor`.)

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

## New gotchas (P7 launch session, 2026-04-29 evening)

1. **`measurement_position` as a list crashes mentor's patch** — fixed locally on `refusal-lens-multi-position-fix` submodule branch but **NOT yet upstreamed**. If you ever rebase `vendor/circuit-tracer` onto a fresh upstream commit (e.g., a future Anthropic release), make sure the `torch.full(shape, _mp)` → branch-on-`isinstance(_mp, (list, tuple))` patch in both `attribute_transformerlens.py` and `attribute_nnsight.py` survives. See P7-session details above.

2. **Stage 02 needs `--save-graphs` for downstream consumers**. The orchestrator passes it. If you invoke Stage 02 manually (without the orchestrator), remember to add it — otherwise Stage 02c, push_raw_graphs, and push_graph_data all fail with "graphs/ does not exist".

3. **Stage 08's `recovery_rate` is `n_recovered_refusal / n_baseline_comply`** — when a target class has 0 baseline complies (likely at small N for fiction/completion/roleplay), the rate defaults to 0.0 but is **mathematically vacuous**. The `dissociation_delta` can read negative for purely small-sample reasons. Cross-check with the `activation_audit` block (in `08_ablation/ABLATION_SUMMARY.md` and `ablation_results.json`) — per-class top-50 hit rate is the correctness check that's robust to N. Per Phase B, at N=50 expect baseline complies: analytical 27, cognitive_reframe 33, fiction 20, roleplay 9, completion 1 — so completion's recovery_rate will still be N-limited even at full size.

4. **HANDOFF's Stage 08 14-h estimate is ~3× optimistic** for the new measurement_hook regime. Empirical: 65 s/gen at max_new_tokens=200 with 26-feature subcircuits on H100 SXM. Realistic for full at max_new_tokens=80 + 12-feature avg subcircuits + `--positions all`: ~17-28 h end-to-end (Stage 08 ~9-21 h alone). Use `--positions both` only if you have a 30-50 h budget. The orchestrator's `--resume --checkpoint-every 5` for Stage 08 protects against pod restarts mid-run.

5. **`run_p7.sh` self-relaunches into tmux session `p7`**. If a prior session is still alive (e.g., a failed run that didn't clean up), the script aborts. Either `tmux kill-session -t p7 2>/dev/null || true` first, or pass `--session OTHER_NAME`. The `--no-tmux` flag bypasses self-relaunch (useful for debugging in the foreground).

6. **GHP_TOKEN was exposed locally on the user's Mac** in `.git/modules/vendor/circuit-tracer/config` (URL-embedded PAT). Not pushed to GitHub (verified: `git log --all -p -S "ghp_..."` finds nothing). User to rotate token + `git -C vendor/circuit-tracer remote set-url origin https://github.com/AutoInterp/circuit-tracer.git` after the run wraps. Do NOT proactively echo the token in tool output if you re-investigate.

## Older gotchas (P1-P6 session)

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
| **P7 unified orchestrator (NEW this session)** | `scripts/pipeline/run_p7.sh` |
| **Stage 08 recovery drilldown (NEW this session)** | `scripts/pipeline/08_recovery_drilldown.py` |
| Main pipeline stages | `scripts/pipeline/0X_*.py` |
| Shared helpers | `scripts/pipeline/{utils,utils_viz,config}.py` |
| Local tests | `scripts/pipeline/tests/test_pipeline_local.py` |
| RunPod test (GPU) | `scripts/pipeline/tests/test_runpod_1_4.py` |
| Controlled dataset | `dataset/refusal_lens_controlled_dataset.json` |
| **Mentor's three-bug writeup** | `MENTEE_NOTE_three_bugs.md` |
| **Validated smoke run (this session)** | `data/results/pipeline_runs/full_smoke_20260429_170345/` |
| **Full run (in flight or completed)** | `data/results/pipeline_runs/run_<TIMESTAMP>/` |
| Pre-fix reference run (no longer reused — kept for archaeology) | `data/results/pipeline_runs/run_20260422_015552/` |
| Tejas's bulletproof causal script | `git show origin/tejas-circuit-experiments:data/tejas_experiments/scripts/20_bulletproof_pipeline.py` |
| HF push helper (JSON.gz bundle) | `scripts/pipeline/push_graph_data.py` |
| HF push helper (raw .pt) | `scripts/pipeline/push_raw_graphs.py` |
| HF push helper (full run meta, --skip-graphs flag) | `scripts/pipeline/push_run.py` |
| `.pt → JSON.gz` packer | `scripts/pipeline/02c_pack_graphs.py` |
| Patched circuit-tracer (multi-position fix on top of mentor's hook patch) | `vendor/circuit-tracer` @ `refusal-lens-multi-position-fix` branch |

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

## Remaining tasks (queued for next session, post-P7)

In rough priority order:

1. **Run `08_recovery_drilldown.py` locally** on the result JSONs you've already pulled:
   ```bash
   PYTHONPATH=src python3 scripts/pipeline/08_recovery_drilldown.py \
       --run-dir data/results/pipeline_runs/run_20260430_023247
   ```
   No GPU. Produces per-prompt baseline-COMPLY → ablated-{REFUSE | COMPLY} table for every (prompt × ablation × pos_mode). Useful for the per-class qualitative writeup on which prompts the fiction-specific ablation works on vs which it misses.

2. **HIGHEST PRIORITY follow-up — ablate `canonical_pro_refusal`** (88-feature subcircuit, "in all 5 JB top-50 but not in bare top-50"). The single most theoretically interesting subcircuit, missing from the Stage 08 default set. Expected wall ~3–6 h on H100. Plan in `EXPERIMENT_PLAN_canonical_pro_refusal_and_frontend_ablation.md` (companion doc).

3. **Stand up the live ablation server + frontend cart**. Existing infrastructure is ~70 % built:
   - `scripts/pipeline/ablation_server.py` (224 lines, FastAPI)
   - `scripts/pipeline/05_frontend_patches/feature-cart.js` (click-to-cart)
   Needs: end-to-end test on a real run, extension to `compare.html`, supernode-grouping UI. Same companion plan as above.

4. **Repo cleanup before public release** — see [Repo cleanup audit](#repo-cleanup-audit-recommendations) below. ~500 GB of stale run artifacts and several non-pipeline scripts can be safely removed.

5. **Stage 08 (K, F) sweep on alternate sweep configs** to resolve negative dissociation on analytical/cog_reframe. Run Stage 08 on `subcircuits_k20_f50.json` (more selective) and `subcircuits_k100_f20.json` (more permissive). ~9–18 h per config on H100. If Δ stays negative across configs, the negative dissociation is real (general-bypass weight); if it goes positive at K=100, it's a top-K aliasing artifact.

6. **Optionally re-run with `--positions anchors`** for the position-mode comparison. Anchors-only on existing run dir is `08_ablate_subcircuits.py --positions anchors` ~half the time of `--positions all` (~5–10 h). Adds the anchors column to the dissociation matrix figure for the methods section.

7. **Coordinate with Georg (mentor)** to merge the `refusal-lens-multi-position-fix` submodule branch back upstream into `refusal-lens-residual-stream-hook`.

8. **Stage 04b (Delphi labels via Ruqiya)** — still unstarted. Lower priority; post-NeurIPS unless Ruqiya delivers in the meantime.

9. **Stage 05 frontend regeneration** — not run by the orchestrator. Invoke manually on the new run dir:
   ```bash
   PYTHONPATH=src python3 scripts/pipeline/05_visualize_circuits.py --run-dir data/results/pipeline_runs/run_20260430_023247
   ```

10. **Replace `jb_completion`** in the dataset before camera-ready. With 0/50 baseline complies it's a refusal-strengthening style, not a JB. Reframe as a "JBs that don't bypass" anchor or substitute a real bypass-style JB.

11. **Rotate the leaked GHP_TOKEN** on the user's Mac (URL-embedded in `.git/modules/vendor/circuit-tracer/config`). Not pushed publicly but exposed in the conversation transcript. Then `git -C vendor/circuit-tracer remote set-url origin https://github.com/AutoInterp/circuit-tracer.git` to remove the embedded credential.

---

## Repo cleanup audit (recommendations)

Audit performed 2026-05-01 against current `l15-refactor` HEAD. **All recommendations are advisory — confirm with the user before deleting anything.**

### Safe to delete

**Top-level**
- `run.log` (28 KB, last touched Apr 12) — orphan log file, no references.

**Old pipeline run artifacts** (in `data/results/pipeline_runs/`)
- `attribution_results_test.json` (112 K) — orphan top-level file, presumably from a failed early Stage 02 run.
- `run_20260417_010035/` (9.8 MB) — pre-foundation-merge run, no references in HANDOFF or active code.
- `full_smoke_20260429_170345/` (37 MB) — smoke validation for P7 launch; superseded by `run_20260430_023247`.
- `run_20260422_015552/` (**75 GB**) — the largest disk hog. Per HANDOFF this was kept for archaeology, but post-P7 the corrected-basis numbers in `run_20260430_023247` strictly supersede everything attribution-derived from this run. Stage 06 (96.7 % flip) is the only result that's even worth comparing against, and that comparison is already inlined in the report. **Recommend deletion** after confirming the small JSONs (06_causal/, 07_subcircuits/, 08_ablation/) have been pulled to a separate archive if needed for historical comparison.

**Ancillary pipeline scripts** (in `scripts/pipeline/`)
- `auto_fetch_pack_push.sh` (132 lines) — superseded by `run_p7.sh`'s push phase. Not referenced.
- `run_stage02_parallel.sh` (83 lines) — superseded by orchestrator's serial stage-02 invocation. Not referenced.
- `run_stage08_overnight.sh` (135 lines) — superseded by orchestrator's `--resume --checkpoint-every 5`. Not referenced.
- `merge_stage02_shards.py` (149 lines) — for the sharded-Stage-02 mode that's no longer used. Not referenced by `run_p7.sh`.

**Non-pipeline scripts** (in `scripts/`) — these all import from `src/refusal_lens` (the legacy library). **Verify before deleting** if you want to preserve them as historical reproducibility:
- `run_meeting_experiments.py` — early-meeting experiments, superseded by Stage 06.
- `run_scaled_experiments.py` — pre-pipeline scaling exploration, superseded by Stage 02.
- `validate_tejas_replication.py` — pre-pipeline Tejas-causal validation, superseded by Stage 06.

### Verify with user before deleting

- `REFACTORING_GUIDE.md` (repo root) — historical 15-step refactoring plan. Mostly superseded by HANDOFF.md, but contains references to project memory and pre-pipeline state. **Probably safe to archive but worth a re-read.**
- `data/tejas_experiments/scripts/16_causal_arditi.py` — explicitly documented in HANDOFF as superseded by Tejas's Script 20 (which is the source for our Stage 06). Other files in `data/tejas_experiments/` (KEY_FINDINGS.txt, MECHANISM_FINDING.txt, README.md, figures/) are research-archive material — keep.
- `scripts/pipeline/diagnose_stage_08.py` (192 lines) — useful for reading old runs without rerun, but redundant with `08_recovery_drilldown.py`. **Decide based on whether you want to keep multiple post-hoc tools.**
- `scripts/pipeline/rebuild_graph_metadata.py` (68 lines) — utility for regenerating `graph-metadata.json` if it gets corrupted. Probably keep as a small utility.
- `scripts/demo_attribution.py` — early-meeting demo. Superseded by Stage 02 + `run_p7.sh --mode smoke`. Decide based on whether you want a "hello world" entry point for new contributors.
- `Dockerfile.local` — useful for reproducibility. **Keep**, but verify it builds successfully against the current `pyproject.toml`.

### Keep (load-bearing)

- `src/refusal_lens/` — imported by 6 active pipeline scripts (01, 02, 06, 08, config, utils). **Do not remove.**
- `test/` (legacy test suite) — imports from `src/refusal_lens`; verifies the library code that the pipeline depends on. **Keep**, run `PYTHONPATH=src python3 -m pytest test/ -v` to confirm it still passes.
- All `scripts/pipeline/0X_*.py` (the 9 main stages + 02b, 02c, 04b, 08_recovery_drilldown).
- `scripts/pipeline/run_p7.sh` (orchestrator).
- `scripts/pipeline/{utils,utils_viz,config}.py` (shared helpers).
- `scripts/pipeline/tests/test_pipeline_local.py` (244-test suite).
- `scripts/pipeline/{push_*,fetch_*,02c_pack_graphs}.py` (HF helpers).
- `scripts/pipeline/ablation_server.py` (frontend backend, in active use).
- `scripts/pipeline/05_frontend_patches/` (frontend patches injected by Stage 05 and `feature-cart.js`).
- `vendor/circuit-tracer/` (patched submodule).
- `dataset/refusal_lens_controlled_dataset.json`.
- `data/results/pipeline_runs/run_20260430_023247/` (canonical run, just completed; HF backup exists).
- `MENTEE_NOTE_three_bugs.md`, `HANDOFF.md`, `REPORT_run_20260430_023247.md`, `Attribution_Circuits_to_Refusal_Direction.pdf`, `images/`, `LICENSE`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `README.md`.

### Estimated disk freed

- ~75 GB from deleting `run_20260422_015552/` (the big one).
- ~50 MB from smaller cleanups (script files, smoke run, run.log, attribution_results_test.json).
- Net repo footprint goes from ~75 GB to ~36 MB pre-vendored-package — much more reasonable for a public release.

---

## New gotchas (P7 results session)

1. **Pre-existing frontend ablation infrastructure**. `scripts/pipeline/ablation_server.py` (224-line FastAPI service) and `scripts/pipeline/05_frontend_patches/feature-cart.js` (click-to-cart with `SERVER_URL = http://localhost:8080/ablate`) are already in the repo. Don't re-implement; extend them. Server uses `nnsight` backend for the ablation forward pass (NOT TransformerLens — `feature_intervention_generate` lives on `ReplacementModel.nnsight`). This is a deliberate choice: ablation needs `feature_intervention_generate`, which only exists on the nnsight side; attribution needs `hook_resid_post`, which requires TransformerLens. The two parts of the pipeline use different backends for legitimate reasons.

2. **Stage 08 metadata block is empty in `ablation_summary.json`** — only `per_ablation`, `dissociation`, `coverage` keys exist at the top level. The metadata (subcircuits-source-file, low-coverage threshold, elapsed minutes) lives only in `ABLATION_SUMMARY.md` (the human-readable summary). If the next iteration needs metadata in machine-readable form, add it back to `ablation_summary.json`'s top level.

3. **355 GB of raw `.pt` archive on HF**. Pushed via `huggingface_hub.HfApi.upload_large_folder` (NOT the regular `upload_folder` — that times out on >100 GB single commits). The CLI form `hf upload-large-folder` does NOT support `--path-in-repo`; we used the staging-symlink workaround. If you need to re-upload, use the same pattern. See `EXPERIMENT_PLAN_canonical_pro_refusal_and_frontend_ablation.md` § 4 for the exact incantation.

## Final note for new session

P7 is complete. Headline science is in: bidirectional refusal-axis symmetry at L15 (100/98/100 %), clean linearization identity (1.66 ratio, +19.4k baseline), and a unblocked Stage 08 with one clean dissociation (fiction +16 pp), one negative-dissociation puzzle (analytical/cog_reframe), and a major direction-vs-subcircuit potency gap. The full report is at `REPORT_run_20260430_023247.md`.

Three things to do next, in rough priority order:
1. **Ablate `canonical_pro_refusal`** — the missing experiment that closes the JB-only refusal recruitment story (companion plan).
2. **Ship the live ablation server** — pre-built infrastructure ready for last-mile testing + frontend cart UX (companion plan).
3. **Repo cleanup before public release** — recommendations in this file (audit section above).

Local tests: 244 passed / 0 failed / 1 skipped (Apr 29). Add new tests when you wire up canonical_pro_refusal or extend the ablation server.
