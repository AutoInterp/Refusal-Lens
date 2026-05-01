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

1. **circuit-tracer Qwen-3 hookpoint patch (not done).** `vendor/circuit-tracer` on `refusal-lens-measurement-patch` only knows about Gemma-3's `hook_resid_post` resolution. Until a Qwen-3 entry is added to `tl_nnsight_mapping.py` and `replacement_model::get_measurement_loc()`, attribution at the residual stream won't work. Submodule isn't checked out yet — first action: `git submodule update --init --recursive`.
2. **`tests/` not rewritten.** The local pipeline tests in `pipeline_qwen/tests/` were copied unchanged and assert Gemma-specific values (e.g. L32 separation ~20k, MLP ratio ~0.4%). They will fail on Qwen. Either rewrite or skip until Qwen reference numbers exist.
3. **Position discovery still owed.** `MEASUREMENT_POSITION = -1` is a guess. Run `qwen_experiments/scripts/01_compute_direction_and_sanity.py` (already ported, sweeps `[-5..-1]`) before Stage 01 here to discover the right position, then update `config.py`.
