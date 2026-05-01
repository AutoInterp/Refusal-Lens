# Qwen3-4B Port of `scripts/pipeline/`

Plan for replicating the Gemma-3-4b-it refusal-circuit pipeline on `Qwen/Qwen3-4B` using `mwhanna/qwen3-4b-transcoders`.

## What's available for Qwen3-4B

Verified before drafting:

- **[mwhanna/qwen3-4b-transcoders](https://huggingface.co/mwhanna/qwen3-4b-transcoders)** is the direct analogue of `mwhanna/gemma-scope-2-4b-it`. From its `config.yaml` and `wandb-config.yaml`:
  - `model_name: "Qwen/Qwen3-4B"`, `model_kind: "transcoder_set"`
  - **Same hookpoints as Gemma**: `feature_input_hook: "mlp.hook_in"`, `feature_output_hook: "mlp.hook_out"` — circuit-tracer attribution works identically in shape.
  - **36 layers**, one safetensors per layer, **`d_feature: 163840` ≈ 160k (10× Gemma's 16k)**, `d_model: 2560`, `act_fn: relu`. Each layer file is 1.68 GB → ~60 GB total. No subpath, the repo *is* the transcoder set.
  - Ships `features/index.json.gz` + `features/layer_*.bin` — the **same dashboard format** the Gemma Scope pipeline reads in Stage 04. Feature labeling is essentially a repo-name swap.
- **[Qwen-Scope collection](https://huggingface.co/collections/Qwen/qwen-scope)** is a separate Qwen-org release: residual-stream SAEs (not cross-layer transcoders) on Qwen3/Qwen3.5 **base** models at sizes 1.7B / 8B / 30B / etc. **There is no Qwen3-4B SAE in this collection**, the SAEs target Base (not Instruct), and they're not directly compatible with circuit-tracer's CLT attribution path. Useful as a future steering/probing comparison, not a substitute.
- **[Qwen-Scope blog](https://qwen.ai/blog?id=qwen-scope)** is consistent with the collection above (residual-stream SAEs, exposed via the [QwenScope HF Space](https://huggingface.co/spaces/Qwen/QwenScope)).

**Conclusion**: use `mwhanna/qwen3-4b-transcoders`. Qwen-Scope is the wrong artifact for this pipeline.

## Architectural decision

Two ways to add Qwen support:

- **Option A — Sibling pipeline `scripts/pipeline_qwen/`.** Copy the directory, change ~5 files, leave Gemma untouched. Pro: zero risk of regressing the Gemma pipeline you just stabilized after the three-bug fix. Con: ~80% file overlap, fixes need to land twice. **Recommended for v1.**
- **Option B — Parametrized single pipeline.** Replace `config.py` with `config_gemma.py` + `config_qwen.py` + a profile selector; add architecture-branching helpers (`get_measurement_module`, `get_decoder_layers`). Pro: single source of truth, comparison-ready. Con: every existing stage's `import config` becomes a profile-aware import — significant edit surface, easy to break Gemma along the way.

Land **A first** (a week to working Qwen results), refactor to B once Qwen has stabilized. The rest of this plan assumes A.

## Stage-by-stage port effort

| Stage | Reuse | Qwen-specific changes |
|---|---|---|
| `config.py` | new | `MODEL_NAME = "Qwen/Qwen3-4B"`, `N_LAYERS = 36`, `TRANSCODER_PATH = "mwhanna/qwen3-4b-transcoders"`, **drop hardcoded `BEST_SEPARATION_LAYER` / `BEST_CAUSAL_LAYER` until Stage 01 discovers them**, `MEASUREMENT_POSITION = None` (rediscover) |
| `utils.py` | ~95% reuse | `format_prompt()` must pass `enable_thinking=False` — Qwen3's chat template appends `<think>\n` after `assistant\n` by default, which would shift every trailing-token position. Already established in `qwen_experiments/CONFIG.py`. |
| `01_compute_direction.py` | high | `model.config.text_config.hidden_size` → `model.config.hidden_size` (Qwen has a flat config); sweep all 36 layers AND positions `[-5..-1]` (don't assume `-2`); same float64 + left-pad recipe |
| `02_run_attribution.py` | high | Same `attribute()` call; new `measurement_layer` / `measurement_position` from Stage 01; **must keep `measurement_hook="hook_resid_post"`** — and that requires the Qwen-3 entry in circuit-tracer's `tl_nnsight_mapping.py`. Likely needs `--max-features` cap to keep graph size tractable at 160k feature width. |
| `02b_statistical_analysis.py` | 100% | Pure post-processing |
| `03_verify_attribution.py` | medium | Layer-module resolver currently hooks `pre_feedforward_layernorm` (Gemma-3's quad-LN block). Qwen3 uses the standard pre-LN block; the post-RMSNorm-pre-MLP module is `model.model.layers[L].post_attention_layernorm`. **This must be replaced or it'll silently capture the wrong tensor — the exact failure mode of Bug 1 from MENTEE_NOTE.** |
| `04_label_features.py` | high | Qwen transcoder repo ships the same dashboard format. Replace the repo URL; verify payload schema is identical. |
| `05_visualize_circuits.py` + frontend | high | Static viewer; just point at new run dir |
| `07_identify_subcircuits.py` | 100% | Pure post-processing on `feature_labels.json` + class-set rules |

## circuit-tracer submodule changes

`vendor/circuit-tracer` is on the `refusal-lens-measurement-patch` branch. The Gemma patch added two relevant entries:

- `utils/tl_nnsight_mapping.py` — Gemma-3-specific `hook_resid_pre` resolver
- `replacement_model/replacement_model_*.py::get_measurement_loc()` — Gemma-3 case

**Both need a parallel Qwen-3 entry.** Plan: extend the same `refusal-lens-measurement-patch` branch (don't fork a new one), push, bump the submodule SHA. The submodule isn't currently checked out locally (`git submodule status` shows `-7c6cfa44…`); first action is `git submodule update --init --recursive`.

## Resource & runtime expectations

| Resource | Gemma | Qwen3-4B (predicted) |
|---|---|---|
| Transcoder disk/RAM | ~14 GB | **~60 GB** (10× wider) |
| Attribution graph nodes/layer | ~16k | ~160k |
| Stage 02 runtime, 50 prompts | 3–4 h on 48 GB | likely 8–12 h on 80 GB |
| Recommended GPU | A6000 48 GB | **A100 / H100 80 GB** — A6000 will OOM |

The 10× transcoder width is the dominant cost. Start with `--n-prompts 10` and `--max-features 5000` for a first end-to-end pass, then ramp.

## Phase plan

1. **Scaffold** (`scripts/pipeline_qwen/` copy + `config.py` edits + `utils.py::format_prompt` patch + Qwen `_get_layer_module` in Stage 03). Half day. Lands without GPU.
2. **Stage 01 sweep** — all 36 layers × positions `[-5..-1]` × 64 harmful + 64 harmless. Pick `(best_pos, best_separation_layer)`. ~30 min on A100. **Output: Qwen's "L32" and "pos=-2" replacements.** Update `config.py`. Cannibalize `qwen_experiments/scripts/01_compute_direction_and_sanity.py` — it's already ported.
3. **circuit-tracer Qwen-3 patch** — minimal hook-name additions on `refusal-lens-measurement-patch`. Validate by loading `ReplacementModel.from_pretrained("Qwen/Qwen3-4B", "mwhanna/qwen3-4b-transcoders")` and running one attribution. Half day.
4. **Stage 02 + 03 verification on 5 prompts** — this is where the methodology check happens. Σ-edges + baseline must reconstruct `r̂·h` at the Qwen post-RMSNorm-pre-MLP point. **Don't proceed to full runs until Stage 03's MLP-ratio number looks sane** — that's the canary that caught the three Gemma bugs. Half day setup, ~1 h compute.
5. **Stage 02 full 50-prompt run** + 02b stats + 04 labeling + 07 subcircuits. ~1 day on A100 elapsed.
6. **Comparison artifact** — sibling `qwen_vs_gemma_comparison.md` (placeholder already exists at `qwen_experiments/qwen_vs_gemma_comparison.md`). Numerical fields: best-separation layer, best-causal layer, MLP attribution ratio, per-class Cohen's d, universal-core size, sign-flip rate per class. Half day.
7. **Causal intervention (post-pipeline)** — Stage 06 isn't in the Gemma pipeline yet either. Ports of `qwen_experiments/scripts/16_causal_arditi.py` are the natural addition; not in scope for v1.

## Open decisions

1. **Sibling pipeline (Option A) vs parametrized (Option B)?** Assuming A.
2. **Where should `scripts/pipeline_qwen/` live — inside `Refusal-Lens` or in the existing `qwen_experiments/` parent folder?** The existing `qwen_experiments/` is outside the Refusal-Lens repo, and its `MIGRATION_NOTES.md` assumes a `data/qwen_experiments/` layout that doesn't match either location. Recommend `Refusal-Lens/scripts/pipeline_qwen/` for git/CI parity with the Gemma one; treat the standalone `qwen_experiments/` folder as donor scaffolding (cannibalize Stage 01, ignore the rest).
3. **Is an A100/H100 lined up?** The 48 GB pod that ran Gemma will OOM here.
4. **`max_feature_nodes` budget for the first run** — `None` (all 160k×36 active features) or capped (say 5000) for a fast smoke test? The Gemma pipeline used `None`; for Qwen, cap on the first pass and remove once verified.

## Sources

- [mwhanna/qwen3-4b-transcoders model card](https://huggingface.co/mwhanna/qwen3-4b-transcoders)
- [Qwen-Scope collection](https://huggingface.co/collections/Qwen/qwen-scope)
- [QwenScope HF Space](https://huggingface.co/spaces/Qwen/QwenScope)
- [Example Qwen-Scope SAE — SAE-Res-Qwen3.5-27B-W80K-L0_50](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50)

---

## Phase 1 — Scaffold complete (2026-05-01)

Option A executed. Sibling pipeline lives at [scripts/pipeline_qwen/](scripts/pipeline_qwen/). Copied from `scripts/pipeline/` then surgically edited; everything else lifted verbatim.

### Files edited

| File | Change |
|---|---|
| [scripts/pipeline_qwen/config.py](scripts/pipeline_qwen/config.py) | Full rewrite. `MODEL_NAME = "Qwen/Qwen3-4B"`, `N_LAYERS = 36`, `TRANSCODER_PATH = "mwhanna/qwen3-4b-transcoders"`, `RESULTS_BASE = data/results/pipeline_runs_qwen/`, `MAX_FEATURES = 5000` (cap for first run), `SWEEP_POSITIONS = [-5..-1]`. Placeholders `MEASUREMENT_LAYER=34`, `MEASUREMENT_POSITION=-1`, `CAUSAL_LAYER=18`, `BEST_SEPARATION_LAYER=34`, `BEST_CAUSAL_LAYER=18` flagged with TODO comments — **must be verified via Stage 01 before Stage 02 produces meaningful numbers**. `JB_CLASSES`, `TOPIC_KEYWORDS`, `BENIGN_PROMPTS` reused verbatim (prompt-side only). |
| [scripts/pipeline_qwen/utils.py](scripts/pipeline_qwen/utils.py) | `format_prompt()` now passes `enable_thinking=False` to `apply_chat_template`. Load-bearing: Qwen3's default template appends `<think>\n` after `<\|im_start\|>assistant\n`, which would shift every trailing-token position the direction sweep analyzes. |
| [scripts/pipeline_qwen/01_compute_direction.py](scripts/pipeline_qwen/01_compute_direction.py) | Added `get_hidden_size()` helper that handles both Qwen3 flat config (`model.config.hidden_size`) and Gemma3 nested (`model.config.text_config.hidden_size`). Replaced hardcoded `pos = input_ids.shape[1] - 2` with negative-indexed `pos = config.DIRECTION_POSITION` so the position is configurable without editing the function. Key-layer cosine indices retargeted from `[15,18,25,32]` to `[15,18,27,34]` for Qwen's 36-layer stack. Header rewritten. |
| [scripts/pipeline_qwen/03_verify_attribution.py](scripts/pipeline_qwen/03_verify_attribution.py) | **Bug 1 surface from MENTEE_NOTE.** Measurement-point hook resolver now picks `post_attention_layernorm` (Qwen3's standard pre-LN block, the LN sitting between attention and MLP). Falls back to `pre_feedforward_layernorm` if the layer module exposes it, so cross-model use still works. Header + print statements updated. |
| [scripts/pipeline_qwen/04_label_features.py](scripts/pipeline_qwen/04_label_features.py) | `HF_REPO = "mwhanna/qwen3-4b-transcoders"`, `HF_FEATURES_PATH = "features"` (Qwen ships `features/` at the repo root, not nested under `transcoder_all/width_16k_l0_small_affine/` like Gemma Scope). Histogram default `n_layers=36`. |
| [scripts/pipeline_qwen/07_identify_subcircuits.py](scripts/pipeline_qwen/07_identify_subcircuits.py) | `n_layers` defaults flipped from `34` to `36` in `summarize_subcircuit()` and `plot_by_layer()`. |
| [scripts/pipeline_qwen/utils_viz.py](scripts/pipeline_qwen/utils_viz.py) | `convert_pt_to_frontend_json(scan=…)` default → `"mwhanna/qwen3-4b-transcoders"` (label-only, no semantic dependency). |
| [scripts/pipeline_qwen/README.md](scripts/pipeline_qwen/README.md) | Replaced the long Gemma-results README with a Qwen-specific stub: differences table, placeholder list, outstanding-work list. Points back to the parent docs for shared methodology. |

### Files copied verbatim (no Gemma-specific code paths)

`02_run_attribution.py`, `02b_statistical_analysis.py`, `05_visualize_circuits.py`, `05_frontend_patches/`, `fetch_graph_data.py`, `fetch_raw_graphs.py`, `push_graph_data.py`, `push_raw_graphs.py`, `rebuild_graph_metadata.py`, `PIPELINE_PLAN.md`. The fetch/push scripts still default to `--dataset-repo moon70/refusal-lens-graphs` (the Gemma dataset); pass a different repo for Qwen runs.

### Verification

- `python3 -m ast` parses all 10 edited Python files cleanly.
- `import config` from `pipeline_qwen/` loads with `MODEL_NAME=Qwen/Qwen3-4B`, `N_LAYERS=36`, `TRANSCODER=mwhanna/qwen3-4b-transcoders`, `MEAS=(34, -1)`, `DIR_LAYERS len=36`, `SWEEP_POSITIONS=[-5,-4,-3,-2,-1]`, `RESULTS_BASE` resolves under repo root.
- Grep audit confirms remaining `gemma`/`Gemma` references in `pipeline_qwen/*.py` are intentional fallback comments and migration-context docstrings only.

### Decisions resolved during Phase 1

- **Open decision #1 (sibling vs parametrized)** → Option A, sibling pipeline.
- **Open decision #2 (where it lives)** → `Refusal-Lens/scripts/pipeline_qwen/` inside this repo. The standalone `qwen_experiments/` folder is now donor scaffolding only.
- **Open decision #4 (`max_feature_nodes` budget)** → capped at 5000 in `config.py::MAX_FEATURES`. Lift to `None` after Stage 03's methodology check passes.
- **Open decision #3 (A100/H100 availability)** → still open.

### Known caveats — these block an actual Qwen run

1. ~~**circuit-tracer Qwen-3 hookpoint patch (not done).**~~ **Resolved in Phase 2 (below).**
2. **`tests/` not rewritten.** The local pipeline tests in `pipeline_qwen/tests/` were copied unchanged and assert Gemma-specific values (e.g. L32 separation ~20k, MLP ratio ~0.4%). They will fail on Qwen. Either rewrite or skip until Qwen reference numbers exist.
3. **Position discovery still owed.** `MEASUREMENT_POSITION = -1` is a guess. Run `qwen_experiments/scripts/01_compute_direction_and_sanity.py` (already ported, sweeps `[-5..-1]`) before Stage 01 here to discover the right position, then update `config.py`.

---

## Phase 2 — circuit-tracer Qwen-3 patch complete (2026-05-01)

The Stage 02 `measurement_hook="hook_resid_post"` path is now wired through for Qwen3-4B. The patch is much smaller than originally scoped — the submodule was already most of the way there.

### What needed patching (and what didn't)

Inspecting the actual `vendor/circuit-tracer` submodule revealed that:

- **`get_measurement_loc()` is already architecture-agnostic.** It dispatches purely through the `feature_hook_mapping` registered for the model's architecture. No per-architecture branches inside the function — it just looks up `hook_resid_pre` for the next layer when asked to resolve `hook_resid_post`.
- **`qwen_3_mapping` was already registered** in `circuit_tracer/utils/tl_nnsight_mapping.py` with `mlp.hook_in` and `mlp.hook_out` correctly pointing at `post_attention_layernorm.output` / `mlp.output`. But it was missing the residual-stream hookpoints that `get_measurement_loc` needs.
- **The TransformerLens backend doesn't need a patch at all.** `attribute_transformerlens` attaches hooks at TL-native names (`blocks.{L}.hook_resid_post` etc.) which `HookedTransformer` provides natively for any supported architecture, including Qwen3. The patch is only needed for the nnsight backend, which is what `pipeline_qwen/02_run_attribution.py` uses.

So the work reduced to: add two missing hookpoints to the existing Qwen-3 mapping. **+5 lines, zero code-path changes.**

### The patch

In [vendor/circuit-tracer/circuit_tracer/utils/tl_nnsight_mapping.py](vendor/circuit-tracer/circuit_tracer/utils/tl_nnsight_mapping.py), `qwen_3_mapping.feature_hook_mapping` gained two entries:

```python
"hook_resid_mid": ("model.layers[{layer}].post_attention_layernorm", "input"),
"hook_resid_pre": ("model.layers[{layer}].input_layernorm", "input"),
```

These mirror the Gemma-3 flat-config mapping with the only architectural difference being `pre_feedforward_layernorm` (Gemma-3 quad-LN block) → `post_attention_layernorm` (Qwen3 standard pre-LN block).

### Why `hook_resid_pre` is the load-bearing one

`get_measurement_loc(layer=L, measurement_hook="hook_resid_post")` resolves the post-layer residual point via the identity `hook_resid_post[L] == hook_resid_pre[L+1]` — i.e. the input of the *next* layer's `input_layernorm`. Without `hook_resid_pre` in the mapping, the resolver raises `ValueError: hook_resid_post requires hook_resid_pre to be defined in the mapping for Qwen3ForCausalLM`. Adding that one entry unblocks the entire residual-stream measurement path that Stage 02 relies on.

`hook_resid_mid` was added for symmetry with the Gemma-3 mappings but isn't load-bearing for our pipeline — it's the residual stream point between attention and MLP, useful as an alternative measurement target if needed later.

### Validation

- Import-level check: `qwen_3_mapping.feature_hook_mapping` exposes `hook_resid_pre`, `hook_resid_mid`, `mlp.hook_in`, `mlp.hook_out`. `get_mapping("Qwen3ForCausalLM")` returns the patched entry with the right module paths.
- Full validation (load `ReplacementModel.from_pretrained("Qwen/Qwen3-4B", "mwhanna/qwen3-4b-transcoders", backend="nnsight")` and run a tiny attribution) requires GPU and is deferred to Phase 3.

### What's still owed before Stage 02 will produce numbers

Caveat #1 from Phase 1 is now resolved. The two remaining blockers:

- **Caveat #2 — `pipeline_qwen/tests/` still asserts Gemma-specific reference values.** Will fail on Qwen until rewritten or skipped.
- **Caveat #3 — `MEASUREMENT_POSITION` and `MEASUREMENT_LAYER` in `pipeline_qwen/config.py` are unverified placeholders.** Discovery via `qwen_experiments/scripts/01_compute_direction_and_sanity.py` (sweeps positions `[-5..-1]`) is still required before any meaningful run.

Both need GPU access — the natural Phase 3.

---

## Phase 3 — Running on RunPod

### 1. Pick a pod

| Resource | Recommended | Why |
|---|---|---|
| GPU | **1× A100 80 GB** or **1× H100 80 GB** | Qwen3-4B transcoders are ~60 GB total on disk (10× wider than Gemma Scope). 48 GB cards (A6000, A40, RTX 4090) **will OOM** on Stage 02. |
| Container | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | PyTorch + CUDA preinstalled. |
| Container disk | 50 GB | Code + venv + temp. |
| Volume | **100 GB on `/workspace`** | Persistent across pod restarts; holds the HF cache (~30 GB for Qwen3-4B + ~60 GB transcoders = ~90 GB total) and run outputs. **Do not put these on the container disk** — they vanish on pod stop. |

Cost estimate: A100 80 GB on community cloud is ~$1.20/hr. A typical end-to-end Phase 3 run (position sweep + Stage 01 + Stage 02 on 50 prompts + Stages 02b/03/04/07) takes 12–18 hours.

### 2. One-time setup on the pod

SSH in via the RunPod web terminal or `ssh root@<pod-ip> -p <port>`, then:

```bash
# Persistent caches on the volume — saves ~30 GB of redownloads on pod restart
cd /workspace
export HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp
mkdir -p "$HF_HOME" "$TMPDIR"
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp
EOF

# Clone the Qwen scaffold branch WITH the circuit-tracer submodule.
# --recurse-submodules picks up vendor/circuit-tracer at the patched SHA
# (which now includes the Qwen3 hook_resid_pre entry).
git clone --recurse-submodules -b qwen3-pipeline-scaffold \
    https://github.com/AutoInterp/Refusal-Lens.git
cd Refusal-Lens

# Python env
python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install -e ".[runpod]"
pip install -e vendor/circuit-tracer            # editable install of patched fork

# HuggingFace login. Qwen/Qwen3-4B is sometimes gated for new accounts;
# log in with a token from https://huggingface.co/settings/tokens.
huggingface-cli login
```

### 3. Discover Qwen3's best position and layer (BEFORE Stage 01)

`pipeline_qwen/config.py` ships with placeholder values (`MEASUREMENT_POSITION = -1`, `BEST_SEPARATION_LAYER = 34`). The `qwen_experiments/scripts/01_compute_direction_and_sanity.py` already exists and sweeps positions `[-5..-1]` × all 36 layers.

```bash
# This script lives outside Refusal-Lens — clone separately if it's not
# already on the pod, or sync the qwen_experiments folder over rsync.
# It reads no state from pipeline_qwen and writes to its own results dir.
python qwen_experiments/scripts/01_compute_direction_and_sanity.py
```

Expected runtime: ~30 min on A100. The last printed line tells you exactly what to put in config:

```
Update CONFIG.py: QWEN_BEST_POSITION=-?, QWEN_BEST_LAYER=??
```

Then edit `scripts/pipeline_qwen/config.py` and update **all five** placeholders so they're consistent:

```python
MEASUREMENT_LAYER = <best_layer>
MEASUREMENT_POSITION = <best_pos>
DIRECTION_POSITION = <best_pos>          # match MEASUREMENT_POSITION
BEST_SEPARATION_LAYER = <best_layer>     # same as MEASUREMENT_LAYER
BEST_CAUSAL_LAYER = <causal_layer>       # tune separately later; 18 is the seed
```

Commit the config update on the `qwen3-pipeline-scaffold` branch so the chosen values are part of the run record.

### 4. Run the pipeline (in order)

All commands run from the repo root with the venv active. Use `nohup` + a log file so SSH can disconnect during long stages.

```bash
mkdir -p data/results/pipeline_runs_qwen

# Stage 01 — per-layer refusal directions, ~15 min on A100
nohup python scripts/pipeline_qwen/01_compute_direction.py \
    > data/results/pipeline_runs_qwen/01.log 2>&1 &
tail -f data/results/pipeline_runs_qwen/01.log
# Note the printed run dir, e.g. data/results/pipeline_runs_qwen/run_20260501_120000

# Stage 02 — SMOKE TEST FIRST: 5 prompts, 5000 features. ~30 min.
# This is the methodology canary. Do NOT skip.
RUN=data/results/pipeline_runs_qwen/run_<timestamp>
nohup python scripts/pipeline_qwen/02_run_attribution.py \
    --run-dir $RUN --n-prompts 5 --max-features 5000 \
    > $RUN/02_smoke.log 2>&1 &
tail -f $RUN/02_smoke.log

# Stage 03 — verify Σ-edges ≈ r̂·h on the 5-prompt smoke run.
# Failure here means the circuit-tracer Qwen3 patch isn't wired through
# correctly — fix before going further.
python scripts/pipeline_qwen/03_verify_attribution.py --run-dir $RUN

# Only AFTER Stage 03 looks sane: full 50-prompt Stage 02 run.
# Expect 8–12 hours on A100 80 GB at d_feature=160k.
nohup python scripts/pipeline_qwen/02_run_attribution.py \
    --run-dir $RUN --n-prompts 50 --max-features 5000 --resume \
    > $RUN/02_full.log 2>&1 &
tail -f $RUN/02_full.log

# Downstream stages — minutes each
python scripts/pipeline_qwen/02b_statistical_analysis.py --run-dir $RUN
python scripts/pipeline_qwen/04_label_features.py        --run-dir $RUN
python scripts/pipeline_qwen/07_identify_subcircuits.py  --run-dir $RUN
```

### 5. Pull results back

Either rsync the run directory to your laptop:

```bash
# From your laptop, NOT the pod
rsync -avz --progress \
    -e "ssh -p <pod-port>" \
    root@<pod-ip>:/workspace/Refusal-Lens/data/results/pipeline_runs_qwen/run_<ts>/ \
    ./data/results/pipeline_runs_qwen/run_<ts>/
```

…or commit and push from the pod (raw `02_attribution/graphs/*.pt` files are gitignored by the parent pipeline's `.gitignore`, so only the JSON + plots get committed):

```bash
git add data/results/pipeline_runs_qwen/run_<ts>
git commit -m "Qwen3 pipeline run <timestamp>: stages 01-04 + 07"
git push origin qwen3-pipeline-scaffold
```

### 6. Stop the pod

RunPod charges per second the pod is on. After pulling results:

- **Stop** preserves `/workspace` (cheap storage cost) — pick this if you'll re-run within a week. The HF cache and venv survive.
- **Terminate** deletes everything including the volume — only after results are pulled and you're sure you won't iterate.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` on Stage 02 | 48 GB card | Move to 80 GB. `--max-features 3000` is a stopgap (loses ~0.5% precision). |
| `ValueError: hook_resid_post requires hook_resid_pre to be defined` | Submodule didn't pick up the Qwen3 patch | `git submodule update --init --recursive` and check `git -C vendor/circuit-tracer log --oneline` shows the `Add Qwen3 hook_resid_pre…` commit. |
| `Repository ... is gated` on `Qwen/Qwen3-4B` | Not logged in to HF | `huggingface-cli login` with a token that's accepted the Qwen license. |
| Stage 03 ratio nowhere near 1.0 | `MEASUREMENT_LAYER` / `MEASUREMENT_POSITION` in config doesn't match what Stage 01 found | Re-read `01_direction/direction_metadata.json::best_separation_layer` and update config; rerun Stage 02. |
| HF cache redownloaded after pod restart | `HF_HOME` not exported in the new shell | `echo $HF_HOME` should print `/workspace/.cache/huggingface`. Add to `~/.bashrc` if missing. |
| Loss of results after pod stop | Wrote to container disk, not volume | Always `cd /workspace/...`; never write under `/root` or `~/`. |

For the watcher / tmux pattern that auto-commits Gemma runs on completion, see [scripts/pipeline/README.md § Deployment](scripts/pipeline/README.md). The same pattern works here — only the branch name and commit message need adjusting.
