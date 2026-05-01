# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Mechanistic-interpretability research pipeline for analyzing the refusal circuit in `google/gemma-3-4b-it`. It computes a per-layer "refusal direction" via diff-in-means, runs CLT (cross-layer transcoder) attribution against that direction using a **vendored fork of `circuit-tracer`**, decomposes the result into subcircuits, and renders an interactive attribution-graph viewer.

The repo has two distinct surfaces that are easy to confuse:

1. **Library** — `src/refusal_lens/` is a pip-installable package (`refusal-lens`). It exports building blocks (`load_model`, `compute_refusal_directions`, `attribute_to_refusal`, `SupernodeAnalyzer`, …) and is the API used by ad-hoc notebooks/scripts in `scripts/`.
2. **Pipeline** — `scripts/pipeline/` is the actual end-to-end research workflow (Stages 01 → 07) that produces the numbered run directories under `data/results/pipeline_runs/run_YYYYMMDD_HHMMSS/`. It does **not** import from `src/refusal_lens` for its core path; it has its own [config.py](scripts/pipeline/config.py) and [utils.py](scripts/pipeline/utils.py).

When the user asks about "the pipeline" or numerical results in the project's README, they almost always mean #2. When they ask about API helpers (model loading, prompt templates, classifiers), they mean #1. Don't unify them without being asked — they're independent on purpose.

## Common commands

```bash
# Dev environment (CPU-only — sufficient for unit tests + frontend work)
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"

# Tests (unit, no GPU)
pytest                              # full suite
pytest test/test_attribution.py     # one file
pytest -k "test_compute" -v         # one test by name pattern

# Pipeline smoke test (no GPU; falls back to pre-recorded run dirs for heavy stages)
PYTHONPATH=src python3 -m pytest scripts/pipeline/tests/test_pipeline_local.py -v -W ignore::DeprecationWarning

# Lint + format (ruff + mypy via pre-commit)
pre-commit run -a                   # all files
pre-commit run                      # changed files only
ruff check src test                 # lint only
ruff format src test                # format only
mypy                                # type-check (strict on src/refusal_lens/*)
```

`scripts/pipeline/` stages are run individually as `python3 scripts/pipeline/0X_*.py …`; the runpod orchestrator at [scripts/pipeline/tests/test_runpod_1_4.py](scripts/pipeline/tests/test_runpod_1_4.py) chains 01 → 04 with `--resume` and per-stage `--skip-stage` flags. Stage 02 is the only expensive one (~3–4 h on a 48 GB GPU for 50 prompts) and checkpoints per-prompt.

## Vendored circuit-tracer submodule (critical)

[vendor/circuit-tracer](vendor/circuit-tracer) is a **git submodule** pinned to the `refusal-lens-measurement-patch` branch of `AutoInterp/circuit-tracer`. The patch adds a `measurement_hook` parameter to `attribute()` so the cotangent (refusal direction) can be injected at `hook_resid_post` (residual stream) instead of the default `pre_feedforward_layernorm.output` (post-RMSNorm pre-MLP).

This matters because **the refusal direction is extracted at the residual stream and must be applied at the same point** — the two locations differ by ~1700× in magnitude in Gemma-3. See [MENTEE_NOTE_three_bugs.md](MENTEE_NOTE_three_bugs.md) for the full incident report. When debugging attribution numbers that "look wrong," verify:

1. `attribute(...)` in [02_run_attribution.py](scripts/pipeline/02_run_attribution.py) is explicitly passing `measurement_layer`, `measurement_position`, and `measurement_hook="hook_resid_post"` (the metadata field is descriptive only — it does not drive the call).
2. [03_verify_attribution.py](scripts/pipeline/03_verify_attribution.py) captures the residual via a `register_forward_hook` on `pre_feedforward_layernorm` of the measurement layer, **not** `out.hidden_states[L+1]`.
3. The direction in [01_compute_direction.py](scripts/pipeline/01_compute_direction.py) and the cotangent passed to `attribute()` are extracted at the same hook point.

Clone with `--recurse-submodules`; if already cloned, run `git submodule update --init --recursive`.

## Two layer constants — do not conflate

Defined in both [src/refusal_lens/config.py](src/refusal_lens/config.py) and [scripts/pipeline/config.py](scripts/pipeline/config.py):

- `MEASUREMENT_LAYER = 32` / `BEST_SEPARATION_LAYER = 32` — best harmful-vs-harmless **separation** (`~20,873`). Used as the **attribution target**.
- `CAUSAL_LAYER = 15` / `BEST_CAUSAL_LAYER = 15` — best **causal effectiveness** (Tejas's Script 16). Used for **interventions** (activation addition, directional ablation).

`cos(r̂_L15, r̂_L32) = −0.115` — these are near-orthogonal directions in different "regimes" of the network. **Causal intervention at layer L must use `r̂` computed at layer L** — that's why `DIRECTION_LAYERS = list(range(N_LAYERS))` and Stage 01 saves a per-layer direction file. Don't substitute one layer's direction for another's.

`MEASUREMENT_POSITION = -2` is the "model" token in Gemma-3's chat template — the position with peak refusal signal.

## Output layout (pipeline runs)

```
data/results/pipeline_runs/run_YYYYMMDD_HHMMSS/
├── 01_direction/     # per-layer r̂, separation metadata
├── 02_attribution/   # attribution_results.json + .pt graphs (raw, gitignored)
├── 02b_stats/        # paired Wilcoxon, Cohen's d, plots, EXPERIMENT_SUMMARY.md
├── 03_verification/  # Σ-edges-vs-direct-dot reconciliation
├── 04_labels/        # feature_labels.json (HF Gemma Scope dashboards)
├── 05_frontend/      # interactive viewer bundle (gitignored, regen from HF)
└── 07_subcircuits/   # 11 rule-based subcircuits, treemap + overlap heatmap
```

Two paths are gitignored and live on the HF dataset `moon70/refusal-lens-graphs`:
- `02_attribution/graphs/*.pt` (~80 GB for 50 prompts) — push with `push_raw_graphs.py`, pull with `fetch_raw_graphs.py`.
- `05_frontend/` (~180 MB gzipped) — push with `push_graph_data.py`, pull with `fetch_graph_data.py`. Collaborators only need this + `python -m http.server` to view graphs locally — no GPU, no model weights.

The pipeline README in [scripts/pipeline/README.md](scripts/pipeline/README.md) is the source of truth for the latest experiment's numbers, the mechanistic interpretation, and the deployment / RunPod instructions. The top-level [README.md](README.md) is mostly the package boilerplate.

## Optional dependency groups

`pyproject.toml` defines four groups: `dev` (pytest + pre-commit), `steering` (torch + transformers + nnsight + matplotlib), `circuit-tracer` (just the vendored package), and `runpod` (pinned versions for the GPU pod, including PyTorch nightly's Blackwell support — RTX PRO 6000 / sm_120 needs `torch --pre` from the cu128 nightly index). The base install is intentionally minimal (`numpy` only) so CI can run cheaply on macOS/Linux without GPU deps.

## Caveats baked into the methodology

- **Transcoders cover MLP only.** ~99.6% of the refusal signal at L32 is carried by attention + embeddings (`MLP ratio ≈ 0.4%`). Any conclusion about "feature X explains refusal" is bounded to the MLP slice.
- **Layer 33 collapse is RMSNorm magnitude, not direction.** `cos(L32, L33) = +0.83` but separation drops `20,873 → 287`. Attribution is measured at L32 to avoid this.
- **Gemma-3-4b-it is not on Neuronpedia.** Stage 04 fetches dashboard payloads directly from HF (`mwhanna/gemma-scope-2-4b-it`, byte-range against `index.json.gz`). Many top-token patches are byte-level / polyglot noise — always inspect `examples` alongside `top_logits` before trusting a label.
