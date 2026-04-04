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

# Running research steps
python scripts/compute_directions.py      # Step 1: Compute refusal directions
python scripts/compute_circuits.py        # Step 2: Analyze attribution circuits
python scripts/explore_supernodes.py      # Step 3: Explore supernodes
python scripts/test_jailbreaks.py         # Step 4: Test jailbreak variants
python scripts/visualize_figures.py       # Generate publication-ready figures
```

## Architecture Overview

The project implements a 4-week research plan for analyzing refusal behavior in language models. The architecture is organized around four sequential computational steps:

### Step 1: Refusal Direction Computation
**File**: `src/refusal_lens/circuits/refusal_direction.py`

Computes refusal directions for each layer using the formula: `r_ℓ = E[x_ℓ | harmful] - E[x_ℓ | harmless]`

- `RefusalDirectionComputer`: Main class that processes harmful vs harmless prompts through each layer
- Outputs a numerical direction vector that best separates harmful from harmless responses
- Results saved to `data/results/computed_directions/` with per-layer vectors and a summary.json with separation scores
- Key metric: **separation score** indicates how well the direction discriminates refusal behavior

### Step 2: Attribution Circuit Analysis
**File**: `src/refusal_lens/circuits/attribution.py`

Traces how refusal computation flows through the network by attributing to the refusal direction.

- `AttributionGraph`: Maps token-level contributions to the refusal direction
- `attribute_to_direction()`: Analyzes a single prompt to show which tokens/features contribute most to refusal
- Produces token-level attribution scores identifying top features
- Results help understand the causal pathways of refusal

### Step 3: Supernode Analysis
**File**: `src/refusal_lens/supernode_analyzer.py`

Uses Neuronpedia supernode data to understand feature semantics at refusal-relevant layers.

- `SupernodeAnalyzer`: Integrates with Neuronpedia cache to study high-level features
- Maps low-level activations to interpretable feature concepts
- Helps explain *what* the model is computing, not just how

### Step 4: Refusal Detection & Jailbreak Testing
**Files**: `src/refusal_lens/refusal_detector.py`, `src/refusal_lens/jailbreak_tracer.py`

Tests model robustness by generating responses and detecting whether refusal occurred.

- `RefusalDetector`: Classifies responses into refusal categories (STRONG_REFUSAL, WEAK_REFUSAL, NO_REFUSAL)
- `RefusalClassifier`: Language-model-based classification of refusal behavior
- Used to test jailbreak variants and evaluate steering effectiveness

### Supporting Modules

- **config.py**: Centralized configuration for model, layers, dataset paths, and experimental settings. All paths and hyperparameters should be modified here.
- **prompt_template.py**: Templates for generating jailbreak variants (role-play, indirect requests, etc.)
- **experiment_runner.py**: Orchestration of full experimental pipelines
- **utils/**: Utility functions for data loading, path management, etc.

## Key Design Patterns

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

### Model Support
Currently tested with: google/gemma-3-4b-it. The code uses Hugging Face transformers, so other causal LMs should work but may require adjusting config.REFUSAL_LAYERS based on model depth.

### Memory Considerations
- Computing directions for all 16 layers with large datasets is memory-intensive
- Use smaller batches or fewer prompts for testing (`config.py` settings)
- Some scripts support `--layer` argument to focus on specific layers

### Dependencies
Key packages: torch, transformers, numpy, scikit-learn, plotly (visualization). See pyproject.toml for full list. The venv includes circuit_tracer and nnsight for activation capture.
