# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

### Common Commands

```bash
# Development setup
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Testing
pytest                                    # Run all tests
pytest test/test_supernode_analyzer.py   # Run specific test file
pytest -k "test_name"                     # Run specific test
pytest --cov=refusal_lens                 # Generate coverage report

# Code quality
pre-commit run -a                         # Check all files (ruff, mypy, formatting)
ruff check src/                           # Run linter
ruff format src/ test/                    # Format code
mypy src/                                 # Type checking

# Running real research experiments
# (NOT the top-level scripts/ — those are scaffolding. See "Where the real code lives" below.)
python data/tejas_experiments/scripts/11_fix_all_georg.py    # Gemma corrected pipeline
python data/qwen_experiments/scripts/01_compute_direction_and_sanity.py  # Qwen direction
python scripts/visualize_figures.py                          # Publication figures
```

### Where the real code lives

This repo has three layers; only one of them is the active research codebase:

| Path | Purpose | Status |
|---|---|---|
| `src/refusal_lens/` | Original 4-step pipeline scaffolding (`RefusalDirectionComputer`, `SupernodeAnalyzer`, etc.) | **Largely unused** — research moved to standalone scripts. Don't add to it without checking. |
| `scripts/` (top level) | Thin wrappers around `src/refusal_lens/` | Same — scaffolding only. |
| `data/tejas_experiments/scripts/` | **Actual Gemma research** (22 numbered scripts, ~4400 lines) | Active. This is where findings come from. |
| `data/qwen_experiments/scripts/` | **Active Qwen port** of the Gemma research | Active. CONFIG.py-driven. |

When asked to "add to the pipeline" or "run an experiment", default to `data/<model>_experiments/scripts/` unless explicitly told otherwise.

## Research Status & Current Work

The project has completed core refusal direction computation and is currently focused on **circuit analysis using Anthropic's circuit-tracer library with transcoders**. Key phase:

- ✅ **Completed**: Refusal direction computation (corrected for token position, precision, padding)
- ✅ **Completed**: Circuit attribution and jailbreak mechanism analysis
- 🔄 **Current**: Circuit-informed jailbreak design and steering experiments (tejas-circuit-experiments branch)

See [README.md](README.md) for detailed findings on refusal mechanisms, jailbreak variants, and steering experiments.

## Architecture Overview

The active research codebase is organized as a **per-model experiment folder** with numbered standalone scripts. Each script is self-contained: it loads the model, runs an analysis, writes results to JSON/PT, and exits. Scripts later in the sequence load earlier scripts' output files.

### Pipeline shape (per model)

```
01_compute_direction_and_sanity.py  →  refusal_direction_v2.pt + separation_table.json
                                          ↓ (best position, best layer)
11_*.py                Attribution + RP-vs-fiction MLP mechanism comparison
13/14.py               Verify attribution sum ≈ dot product → reveals attention dominance
15-17.py               Causal intervention (Arditi method): does adding r flip JB → refuse?
                                          ↓ (causal layer)
19.py                  Disentangle: every-step vs prefill, all-pos vs single-pos
20.py                  Bulletproof end-to-end on 50-prompt × 5-class controlled dataset
21-22.py               Q/K attention head attribution + ablation
```

### Four analytical primitives that recur everywhere

These ideas appear in nearly every script and are the conceptual core of the codebase:

1. **Refusal direction** `r = mean(act_harmful) − mean(act_harmless)` at chosen `(position, layer)`. Script 01 sweeps positions × layers and picks the strongest separation.
2. **Projection** `(activation @ direction)` at the causal layer. Negative = harmless side, positive = harmful side. Successful jailbreaks land in the harmless range despite being harmful prompts — that's the mechanism Arditi intervention exploits.
3. **Circuit-tracer attribution**: `attribute(prompt, model, [CustomTarget(vec=direction)])` from the `circuit_tracer` library. Returns per-feature contributions decomposed via transcoders. **MLPs only**, not attention.
4. **Arditi intervention**: a forward hook on the causal layer that adds the unnormalized `r` to the residual stream at every generation step. This is the test for whether `r` is causally responsible for refusal (vs merely correlated).

### Architectural quirks per model

These bit me when porting Gemma → Qwen and will bite future Claude instances:

| Concern | Gemma-3-4B-IT (multimodal wrapper) | Qwen3-4B (flat causal LM) |
|---|---|---|
| Hidden size | `model.config.text_config.hidden_size` | `model.config.hidden_size` |
| Decoder layers | `model.model.language_model.layers[L]` | `model.model.layers[L]` |
| Num blocks | 34 | 36 |
| Chat template tail | `<start_of_turn>model\n` | `<|im_start|>assistant\n` |
| Position -2 means | the literal `model` token | `\n` after `assistant` |

The Qwen scripts handle this via `get_hidden_size(model)` / `get_decoder_layers(model)` helpers in `data/qwen_experiments/scripts/CONFIG.py`. **Use those helpers when writing model-agnostic code.**

### Tokenizer setup pattern (always the same)

Every script that loads a tokenizer does this — replicate it for new scripts:

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"               # right-padding misaligns the trailing positions
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### `src/refusal_lens/` (legacy scaffolding — usually skip)

The `src/refusal_lens/` package (`RefusalDirectionComputer`, `AttributionGraph`, `SupernodeAnalyzer`, `RefusalDetector`, etc.) was the planned 4-step pipeline from `PLAN.md`, but actual research bypassed it for the standalone scripts above. Pre-commit (ruff + mypy) still runs against `src/`, so changes there must pass type checks, but **don't add new research logic there** — put it in `data/<model>_experiments/scripts/`.

## Critical Implementation Details

### Refusal Direction Computation (Bug Fixes Applied)

Critical bugs have been fixed in refusal direction computation. Use these settings:

1. **Token position**: Extract at position **-2** (the `model` token), not -1 (final newline). Multi-position extraction at [-5,-4,-3,-2,-1] useful for comparison.
2. **Numerical precision**: Use **float64** accumulation (not bfloat16) to prevent precision loss
3. **Padding**: Use **left-padding** (not right-padding) for consistent alignment
4. **Feature filtering**: Include all active features (~14k) rather than filtering to 3k; filter loses only 0.1-1.3%
5. **Model variant**: Use IT (instruction-tuned) transcoders for IT model (`mwhanna/gemma-scope-2-4b-it`)

See [README.md](README.md) sections 1-3 for detailed before/after metrics showing impact of these fixes.

### Configuration Management

All experimental settings are in `config.py`:
- Model name (default: google/gemma-3-4b-it)
- Device and model loading settings
- Layer ranges to analyze (default: layers 8-23)
- Dataset paths (harmful_train.json, harmless_train.json)
- Output directories for results

Update `config.py` to change experiments—do not hardcode paths in scripts.

### Data Organization
- **Inputs**: `dataset/refusal_direction_dataset/splits/` contains harmful/harmless prompt pairs
- **Outputs**: `data/results/` organized by step:
  - `computed_directions/`: Layer vectors and separation scores
  - `circuits/`: Attribution analysis results
  - `neuronpedia_cache/`: Supernode feature data
  - `jailbreak_tests/`: Jailbreak attempt results

### Inference Patterns
- Models are loaded via `transformers.AutoModel*` with `accelerate` for efficient multi-GPU inference
- Activations are captured at specified positions (default: last_prompt_token) during forward passes
- Results are stored as PyTorch tensors or JSON for reproducibility

## Testing Strategy

- **Unit tests** in `test/`: Cover individual components (supernode_analyzer, jailbreak_tracer)
- **Integration tests**: Script execution tests validate full pipelines
- **Type checking**: Strict mypy configuration in pyproject.toml for src/ files
- Pre-commit runs ruff (linting + formatting) and mypy on all changes

## Important Notes

### Key Research Findings

**Refusal Mechanism Insights**:
- Attention heads + embeddings carry ~99.6% of refusal signal; MLPs only ~0.4%
- Refusal and harmless now have zero overlap when using corrected direction at position -2
- Separation improves from 4.4% (buggy) to 108% with corrections

**Jailbreak Mechanisms**:
- Identified two mechanistically distinct jailbreak classes:
  - **Role-play jailbreaks**: Suppress pro-refusal MLP features aggressively (drop 60-120 points)
  - **Fiction jailbreaks**: Increase anti-refusal features more (add 40-50 points)
- Novel circuit-informed jailbreaks (analytical framing, completion prompts, meta-grading) bypass refusal on hard topics with 7/8 success
- Most jailbreaks are immune to refusal-direction steering (only 1/32 flipped when adding alpha=200 x refusal_direction)

**Steering Limitations**:
- Steering works for some jailbreaks (locksmith RP: 13/16 flipped) but fails for others (fiction: 0/16 flipped)
- Immunity likely due to jailbreak bypassing refusal at attention level, which steering cannot directly target

See [README.md](README.md) sections 4-9 for full experimental results, visualizations, and quantitative metrics.

### Working with Transcoders

- Transcoders decompose only MLPs, not attention
- Neuronpedia transcoders stored in `data/results/neuronpedia_cache/`
- For attention analysis, direct head attribution or attention SAEs would be needed (not currently implemented)
- Feature attribution scores show positive (pro-refusal) and negative (anti-refusal) contributions

### Model Support

Currently tested with: google/gemma-3-4b-it. The code uses Hugging Face transformers, so other causal LMs should work but may require:
- Adjusting config.REFUSAL_LAYERS based on model depth
- Finding appropriate SAE/transcoder checkpoints (Neuronpedia provides community transcoders)

### Memory Considerations

- Computing directions for all 16 layers with large datasets is memory-intensive
- Circuit tracing with full feature activation logs requires significant VRAM; run on multi-GPU if available
- Use smaller batches or fewer prompts for testing (`config.py` settings)
- Some scripts support `--layer` argument to focus on specific layers
- Neuronpedia cache is stored locally to avoid repeated downloads

### Dependencies

Key packages: torch, transformers, numpy, scikit-learn, plotly (visualization). See pyproject.toml for full list. The project uses:
- **circuit-tracer**: Anthropic's library for activation attribution
- **nnsight**: For capturing activations during forward passes
- **Neuronpedia transcoders**: For MLP feature decomposition (cached locally)

---

## Tejas Experiments Scripts (`data/tejas_experiments/scripts/`)

This folder contains the complete experimental progression (22 numbered scripts) documenting the circuit analysis research. Scripts are organized by research phase and should generally be reviewed/run in order to understand the evolution of findings.

### Script Organization

**Phase 1: Foundation (Scripts 01–09)**
- `01_compute_direction_and_sanity.py` — Compute refusal direction and verify separation
- `02_attribution_30pairs.py` — Initial attribution analysis on 30 harmful/jailbreak pairs
- `03_control_experiment.py` — OVAT control experiment (varying prefixes)
- `04_jailbreak_effectiveness.py` — Test effectiveness of different jailbreak types
- `05_steering_sweep.py` — Sweep alpha values x layers for refusal direction steering
- `06_novel_jailbreaks_and_hard_topics.py` — Design and test novel circuit-informed jailbreaks
- `07_it_model_attribution.py` — Attribution analysis on IT model with IT transcoders
- `08_it_control_experiment.py` — Control experiment on IT model
- `09_layer13_attribution.py` — Focused analysis on layer 13

**Phase 2: Bug Fixes (Scripts 11–14)**
- `11_fix_all_georg.py` — Implement all corrections from Georg: position -2, float64, left-padding, all features
- `12_cosine_with_ruqiya.py` — Verify cosine similarity with Ruqiya's directions across layers
- `13_dot_product_check.py` — Verify attribution sum equals dot product
- `14_probe_attribution_gap.py` — Investigate why attribution sum is ~75 vs dot product ~18,322 (attention dominance)

**Phase 3: Causal Intervention (Scripts 15–17)**
- `15_causal_intervention.py` — Attempt direct intervention on refusal direction (failed)
- `16_causal_arditi.py` — Implement Arditi method: E[x_harmful - x_clean_direction]
- `17_causal_georg_arditi.py` — Compare Arditi and direct approaches with Georg's methodology

**Phase 4: Controlled Dataset & Deep Analysis (Scripts 19–22)**
- `19_disentangle.py` — 2x2 analysis: every step vs all positions, all layers vs single layers
- `20_bulletproof_pipeline.py` — Clean, verified 50-prompt dataset with token-matched controls
- `21_qk_full_scale.py` — Q/K projection analysis at scale
- `22_qk_deep_rigorous.py` — Deep rigorous Q/K analysis with ablation studies

**Note:** Script 10 and 18 are intentionally omitted (not part of final pipeline)

### Key Experimental Findings Documented in Scripts

**Corrected Refusal Direction (Script 11)**:
- Separation improves from 4.4% → 108%
- Position -2 (model token) is optimal, not -1 (final newline)
- Float64 precision essential; bfloat16 loses critical information

**Causal Mechanism (Scripts 15-17)**:
- Arditi method proves refusal direction is causal: flips 95/95 jailbroken prompts at L15
- L15 is causally effective layer (L32 has strongest direction but too late)
- Jailbroken prompts sit in harmless range at L15 (mean -32,260 vs harmless -32,847)

**Disentanglement (Script 19)**:
- Per-position directions are anti-correlated (cosine -0.76 to -0.80)
- "Every step" is critical factor, not "all positions"
- Position -2 + every-step performs as well as all-positions variant

**Controlled Dataset (Script 20)**:
- 50 harmful prompts × 5 jailbreak classes with token-matched neutral controls
- Verified: 50/50 bare refuse, 96% controls refuse
- Enables robust cross-validation of findings

**Q/K Attribution (Scripts 21-22)**:
- Complete attention head analysis
- Ablation studies on query/key contribution
- Deep mechanistic understanding of attention-based refusal

### Running Experiment Scripts

Each script is standalone and can be run with:
```bash
cd data/tejas_experiments/scripts/
python XX_script_name.py  # Outputs to results/ or results_v2/
```

Results are organized in:
- `data/tejas_experiments/results/` — Initial experiments
- `data/tejas_experiments/results_v2/` — Corrected experiments (post-Georg feedback)
- `data/tejas_experiments/figures/` — Generated visualizations

### Reproducing Experimental Results

To reproduce the main findings:
1. Run scripts 01-09 for initial foundation
2. Run script 11 to apply all corrections
3. Run scripts 12-14 to verify corrections
4. Run scripts 15-17 for causal analysis
5. Run scripts 19-22 for controlled analysis and deep dives

See `data/tejas_experiments/README.md` for detailed results summary and `data/tejas_experiments/scripts/INDEX.md` for script groupings.

---

## Qwen3-4B Experiments (`data/qwen_experiments/`)

Parallel of `data/tejas_experiments/` for `Qwen/Qwen3-4B` with
`mwhanna/qwen3-4b-transcoders`. Same methodology, different base model.

- **Centralized config:** `data/qwen_experiments/scripts/CONFIG.py` is the
  single source of truth for model name, transcoder path, layer/position
  hyperparameters, and output paths. Every script imports from it.
- **Fully ported scripts:** `01_compute_direction_and_sanity.py`,
  `11_fix_all_qwen.py`, `20_bulletproof_pipeline.py` (core phases).
- **Stubs awaiting port:** 12, 13, 14, 15, 16, 17, 19, 21, 22 — each one
  imports CONFIG and points to its Gemma source.
- **Migration rationale:** `data/qwen_experiments/MIGRATION_NOTES.md`
  documents every Gemma→Qwen change (config access, decoder-layer access,
  chat template, hyperparameters that need re-tuning vs. carry over).
- **Hyperparameters that must be re-tuned** (Gemma's values are NOT valid
  for Qwen): best position, best layer, causal layer, transcoder subpath.

## Model Comparison (`data/MODEL_COMPARISON.md`)

Side-by-side comparison of refusal-circuit findings across Gemma-3-4B-IT and
Qwen3-4B. Gemma columns are populated from JSON results; Qwen columns are
TBD until the Qwen scripts are run. Used to determine which findings are
architectural universals vs Gemma-specific artifacts.
