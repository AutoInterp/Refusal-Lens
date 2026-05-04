# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Mechanistic-interpretability research pipeline for analyzing the refusal circuit in `google/gemma-3-4b-it`. It computes a per-layer "refusal direction" via diff-in-means, runs CLT (cross-layer transcoder) attribution against that direction using a **vendored fork of `circuit-tracer`**, decomposes the result into subcircuits, and renders an interactive attribution-graph viewer.

The repo has three distinct surfaces that are easy to confuse:

1. **Library** — `src/refusal_lens/` is a pip-installable package (`refusal-lens`). It exports building blocks (`load_model`, `compute_refusal_directions`, `attribute_to_refusal`, `SupernodeAnalyzer`, …) and is the API used by ad-hoc notebooks/scripts in `scripts/`.
2. **Gemma pipeline** — `scripts/pipeline/` is the end-to-end research workflow (Stages 01 → 08) for `google/gemma-3-4b-it`, producing run directories under `data/results/pipeline_runs/run_YYYYMMDD_HHMMSS/`. It does **not** import from `src/refusal_lens` for its core path; it has its own [config.py](scripts/pipeline/config.py) and [utils.py](scripts/pipeline/utils.py).
3. **Qwen pipeline** — `scripts/pipeline_qwen/` is a sibling port of #2 for `Qwen/Qwen3-4B` (36 layers vs Gemma's 34). Same stage shape, different model constants and chat template. The current branch (`temp/gemma-vs-qwen-pipeline`) is doing the cross-model comparison; see [QWEN_PORT_PLAN.md](QWEN_PORT_PLAN.md), [COMPARISON_REPORT_gemma_vs_qwen.md](COMPARISON_REPORT_gemma_vs_qwen.md), [PAPER_GEMMA_VS_QWEN_v1.md](PAPER_GEMMA_VS_QWEN_v1.md). Keep the two pipelines separate — fixes to one are not automatically valid for the other (different layer counts, hook names, position offsets).

When the user asks about "the pipeline" or numerical results, they almost always mean #2 or #3. When they ask about API helpers (model loading, prompt templates, classifiers), they mean #1. Don't unify them without being asked — they're independent on purpose.

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
# The -W flag is required because pyproject sets `filterwarnings = ["error"]`,
# which turns torch/transformers DeprecationWarnings into hard test failures.
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

Two related pins:
- `BACKEND = "transformerlens"` in pipeline config — **do not** switch to `nnsight`. The nnsight backend has a `.grad-on-non-module-output` runtime limitation that breaks `measurement_hook="hook_resid_post"`. TransformerLens is the only verified-bulletproof path and gives bit-exact cache match for Gemma-3-4b-it.
- The Qwen pipeline depends on a separate hookpoint patch in the same submodule branch (see commit `8dbb7a9`); both pipelines share the submodule, so don't pin different commits per pipeline.

Clone with `--recurse-submodules`; if already cloned, run `git submodule update --init --recursive`.

## Two layer constants — do not conflate

Defined in both [src/refusal_lens/config.py](src/refusal_lens/config.py) and [scripts/pipeline/config.py](scripts/pipeline/config.py):

- `MEASUREMENT_LAYER = 15` / `CAUSAL_LAYER = 15` / `BEST_CAUSAL_LAYER = 15` — best **causal effectiveness** (Tejas Script 16: intervention here flips 95/95 jailbroken prompts; L32 flips 0/10). This is the **current attribution target** — we attribute against the layer we can actually intervene on.
- `BEST_SEPARATION_LAYER = 32` — best harmful-vs-harmless **separation** (~7× stronger than L15), but too late in the network to drive the refusal decision. Retained as a metadata flag, **not** the attribution target.

This is a deliberate methodology change from earlier runs that used L32 as `MEASUREMENT_LAYER`; existing run directories on disk (e.g. `run_20260417_*`) may still encode the L32 choice. If a result looks inconsistent with current code, check which layer that run targeted before assuming a bug.

`cos(r̂_L15, r̂_L32) = −0.115` — these are near-orthogonal directions in different "regimes" of the network. **Causal intervention at layer L must use `r̂` computed at layer L** — that's why `DIRECTION_LAYERS = list(range(N_LAYERS))` and Stage 01 saves a per-layer direction file. Don't substitute one layer's direction for another's.

`MEASUREMENT_POSITION = -2` is the "model" token in Gemma-3's chat template — the position with peak refusal signal at L15. Stage 02 also supports **multi-position attribution** at L15 (`PER_POSITION_LAYER = 15`, target positions in `STAGE2_TARGET_POSITIONS`); positions `-5` (`<end_of_turn>`), `-3` (`<start_of_turn>`), and `-2` (`model`) are template-anchored and prompt-length-invariant. Content positions `-15..-6` have higher nominal separation but are not anchored.

Qwen3-4B uses different anchors — see [scripts/pipeline_qwen/config.py](scripts/pipeline_qwen/config.py) for its `MEASUREMENT_POSITION` and chat-template offsets. Don't copy Gemma constants across.

## Output layout (pipeline runs)

```
data/results/pipeline_runs/run_YYYYMMDD_HHMMSS/
├── 01_direction/     # per-layer r̂, separation metadata
├── 02_attribution/   # attribution_results.json + .pt graphs (raw, gitignored)
├── 02b_stats/        # paired Wilcoxon, Cohen's d, plots, EXPERIMENT_SUMMARY.md
├── 03_verification/  # Σ-edges-vs-direct-dot reconciliation
├── 04_labels/        # feature_labels.json (HF Gemma Scope dashboards)
├── 05_frontend/      # interactive viewer bundle (gitignored, regen from HF)
├── 06_causal/        # activation-addition / directional-ablation flip rates
├── 07_subcircuits/   # 11 rule-based subcircuits, treemap + overlap heatmap
└── 08_ablation/      # per-subcircuit ablation + recovery drilldown
```

The latest 50-prompt Stage 06 hit 96.3% pro-refusal and 92.5% anti-refusal flip rates — quoted in the README and recent commit messages; if you're regenerating these numbers, check that the run targeted L15.

Two paths are gitignored and live on the HF dataset `moon70/refusal-lens-graphs`:
- `02_attribution/graphs/*.pt` (~80 GB for 50 prompts) — push with `push_raw_graphs.py`, pull with `fetch_raw_graphs.py`.
- `05_frontend/` (~180 MB gzipped) — push with `push_graph_data.py`, pull with `fetch_graph_data.py`. Collaborators only need this + `python -m http.server` to view graphs locally — no GPU, no model weights.

The pipeline README in [scripts/pipeline/README.md](scripts/pipeline/README.md) is the source of truth for the latest experiment's numbers, the mechanistic interpretation, and the deployment / RunPod instructions. The top-level [README.md](README.md) is mostly the package boilerplate.

## Optional dependency groups

`pyproject.toml` defines four groups: `dev` (pytest + pre-commit), `steering` (torch + transformers + nnsight + matplotlib), `circuit-tracer` (just the vendored package), and `runpod` (pinned versions for the GPU pod, including PyTorch nightly's Blackwell support — RTX PRO 6000 / sm_120 needs `torch --pre` from the cu128 nightly index). The base install is intentionally minimal (`numpy` only) so CI can run cheaply on macOS/Linux without GPU deps.

## Caveats baked into the methodology

- **Transcoders cover MLP only.** At L32, ~99.6% of the refusal signal is carried by attention + embeddings (`MLP ratio ≈ 0.4%`). The MLP slice is correspondingly small at L15 too. Any conclusion about "feature X explains refusal" is bounded to the MLP slice.
- **Layer 33 collapse is RMSNorm magnitude, not direction.** `cos(L32, L33) = +0.83` but separation drops `20,873 → 287` — a reason to never read separation in raw magnitude across late layers, even though current attribution targets L15 (not L32).
- **Gemma-3-4b-it is not on Neuronpedia.** Stage 04 fetches dashboard payloads directly from HF (`mwhanna/gemma-scope-2-4b-it`, byte-range against `index.json.gz`). Many top-token patches are byte-level / polyglot noise — always inspect `examples` alongside `top_logits` before trusting a label.
