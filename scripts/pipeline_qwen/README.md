# Refusal-Lens Pipeline — Qwen3-4B

Sibling of [`scripts/pipeline/`](../pipeline/) (Gemma-3-4b-it). Same 7 stages, same data formats, same circuit-tracer dependency — different model, different transcoders, different chat template. See the parent [PIPELINE_PLAN.md](../pipeline/PIPELINE_PLAN.md) and [README.md](../pipeline/README.md) for the methodology, mechanistic story, and stage-by-stage results from the Gemma run; this directory diverges only in what's listed below.

The port plan and rationale are in [QWEN_PORT_PLAN.md](../../QWEN_PORT_PLAN.md) at the repo root.

## What's different from `scripts/pipeline/`

| Surface | Gemma | Qwen3-4B |
|---|---|---|
| Model | `google/gemma-3-4b-it` | `Qwen/Qwen3-4B` |
| Transcoders | `mwhanna/gemma-scope-2-4b-it` (16k width) | `mwhanna/qwen3-4b-transcoders` (**160k width, 36 layers, ~60 GB total**) |
| Layer count | 34 | 36 |
| Chat template | `<start_of_turn>model` | `<\|im_start\|>assistant\n`, **`enable_thinking=False`** required |
| Best position | `-2` (the "model" token) | TBD — sweep first; placeholder is `-1` |
| Best separation layer | 32 | TBD — placeholder `34` |
| Best causal layer | 15 | TBD — placeholder `18` |
| Block LN before MLP | `pre_feedforward_layernorm` (Gemma quad-LN) | `post_attention_layernorm` (Qwen3 standard pre-LN) |
| Config access | `model.config.text_config.hidden_size` | `model.config.hidden_size` (flat) |
| Decoder layer access | `model.model.language_model.layers[L]` | `model.model.layers[L]` |
| Run dir | `data/results/pipeline_runs/` | `data/results/pipeline_runs_qwen/` |

## Placeholders that must be verified before running Stage 02

[config.py](config.py) seeds five values from the position sweep in `qwen_experiments/scripts/CONFIG.py`. **They are not validated for this pipeline** and may change after Stage 01 runs:

- `MEASUREMENT_LAYER` (currently `34`)
- `MEASUREMENT_POSITION` (currently `-1`)
- `CAUSAL_LAYER` (currently `18`)
- `BEST_SEPARATION_LAYER` (currently `34`)
- `BEST_CAUSAL_LAYER` (currently `18`)

Workflow:

1. Run Stage 01. Read `01_direction/direction_metadata.json::best_separation_layer`.
2. Update the five constants in `config.py`.
3. Run Stage 02 onward.

Position-sweep tooling already exists at `qwen_experiments/scripts/01_compute_direction_and_sanity.py` — use it to discover the right `MEASUREMENT_POSITION` *before* this Stage 01, since this script only computes at one position per run.

## Outstanding work — DO NOT skip before running Stage 02

1. **circuit-tracer Qwen-3 entry.** [vendor/circuit-tracer](../../vendor/circuit-tracer) on `refusal-lens-measurement-patch` knows about Gemma-3 hookpoint resolution but not Qwen-3. The `tl_nnsight_mapping.py` and `replacement_model::get_measurement_loc()` need a parallel Qwen-3 case before `attribute()` can run with the right `measurement_hook`. First action: `git submodule update --init --recursive`.
2. **Tests.** [tests/](tests/) and `tests/test_pipeline_local.py` assert Gemma-specific values (e.g. L32 separation ~20k). **They will fail on Qwen.** Either rewrite or skip until Qwen reference numbers exist.
3. **`max_feature_nodes` cap.** [config.py](config.py) sets `MAX_FEATURES = 5000` for the first run because the Qwen transcoders are 10× wider. Lift to `None` only after Stage 03 verifies the methodology.
4. **GPU.** A6000 48 GB will OOM. Use A100/H100 80 GB.

## Frontend / dataset push

[fetch_graph_data.py](fetch_graph_data.py), [push_graph_data.py](push_graph_data.py), and the raw-graph counterparts are copied verbatim and still default to `--dataset-repo moon70/refusal-lens-graphs` (the Gemma dataset). For Qwen runs pass a different `--dataset-repo` so the bundles don't collide.

[utils_viz.py](utils_viz.py) `convert_pt_to_frontend_json(scan=…)` defaults to `"mwhanna/qwen3-4b-transcoders"` — circuit-tracer's frontend uses this as a label, no semantic dependency.
