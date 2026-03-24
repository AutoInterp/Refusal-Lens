# Refusal-Lens Research Roadmap

A 4-week research plan for analyzing refusal behavior in neural network models.

---

## 📋 Overview

This roadmap implements the 4-week research plan for understanding how language models refuse harmful requests:

| Week | Step | Description | Status |
|------|------|-------------|--------|
| Week 1 | 1 | Compute refusal directions | ✅ Complete |
| Week 2 | 2 | Analyze attribution circuits | ✅ Complete |
| Week 3 | 3 | Explore supernodes | ✅ Complete |
| Week 4 | 4 | Test jailbreak variants | ✅ Complete |

---

## 🎯 Current Status

### ✅ Completed (This Update)

- [x] **config.py** - Centralized configuration
- [x] **pyproject.toml** - Complete dependencies
- [x] **circuits/** - Step 1 & 2 implementation
- [x] **RefusalDirectionComputer** - Compute r_ℓ = E[x|harmful] - E[x|harmless]
- [x] **AttributionGraph** - Trace refusal computation
- [x] **SupernodeAnalyzer** - Step 3 supernode exploration
- [x] **RefusalDetector** - Step 4 jailbreak testing
- [x] **Gemma-3 support** - All scripts work with Gemma-3-4B model
- [x] **explore_supernodes.py** - New script for Step 3

### 🔄 Remaining

- [ ] Run full experiments with larger datasets
- [ ] Add more model support (other transformers)

---

## 📅 Week 1: Compute Refusal Directions

### Goal
Compute refusal directions for each layer: `r_ℓ = E[x_ℓ | harmful] - E[x_ℓ | harmless]`

### Files
- [`src/refusal_lens/circuits/refusal_direction.py`](src/refusal_lens/circuits/refusal_direction.py)
- [`src/refusal_lens/config.py`](src/refusal_lens/config.py)

### Usage

```python
from refusal_lens import (
    RefusalDirectionComputer,
    config,
    save_directions,
    find_best_layer,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Load model
model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

# Load dataset
with open(config.DATASET_HARMFUL_TRAIN) as f:
    harmful = json.load(f)
with open(config.DATASET_HARMLESS_TRAIN) as f:
    harmless = json.load(f)

# Compute directions
computer = RefusalDirectionComputer(model, tokenizer)
results = computer.compute_all_layers(
    harmful_prompts=harmful,
    harmless_prompts=harmless,
    layers=config.REFUSAL_LAYERS,
)

# Save results
save_directions(results, config.COMPUTED_DIRECTIONS_DIR)

# Find best layer
best_layer, best_result = find_best_layer(results)
print(f"Best layer: {best_layer} (separation: {best_result.separation:.4f})")
```

### Expected Output
```
data/results/computed_directions/
├── layer_08.pt
├── layer_09.pt
...
├── layer_23.pt
└── summary.json
```

### Key Metrics
- **Separation score**: How well the direction separates harmful vs harmless
- **Best layer**: Layer with highest separation (typically mid-to-late layers)

---

## 📅 Week 2: Analyze Attribution Circuits

### Goal
Trace how refusal is computed by attributing to the refusal direction

### Files
- [`src/refusal_lens/circuits/attribution.py`](src/refusal_lens/circuits/attribution.py)

### Usage

```python
from refusal_lens import (
    AttributionGraph,
    attribute_to_direction,
    load_directions,
    config,
)
import json

# Load pre-computed direction
directions = load_directions(config.COMPUTED_DIRECTIONS_DIR)
best_direction = directions[best_layer]  # Use best layer from Week 1

# Analyze a prompt
result = attribute_to_direction(
    model=model,
    prompt="How do I hack a website?",
    refusal_direction=best_direction,
    layer=best_layer,
)

# Get top features contributing to refusal
print(f"Top features: {result.top_features[:10]}")
```

### Expected Output
- Token-level attribution scores
- Top features contributing to refusal
- Attribution graph nodes

---

## 📅 Week 3: Supernode Exploration

### Goal
Use SAE supernodes to understand feature semantics

### Files
- [`src/refusal_lens/supernode_analyzer.py`](src/refusal_lens/supernode_analyzer.py)

### Usage

```python
from refusal_lens import SupernodeAnalyzer

analyzer = SupernodeAnalyzer()
# Study feature semantics at refusal-relevant layers
```

---

## 📅 Week 4: Jailbreak Testing

### Goal
Test jailbreak variants using the refusal detector

### Files
- [`src/refusal_lens/refusal_detector.py`](src/refusal_lens/refusal_detector.py)
- [`src/refusal_lens/jailbreak_tracer.py`](src/refusal_lens/jailbreak_tracer.py)

### Usage

```python
from refusal_lens import RefusalDetector, DetectionStatus

detector = RefusalDetector(model, tokenizer)

# Test a jailbreak attempt
response = model.generate(prompt)
result = detector.detect(response)

if result.status == DetectionStatus.STRONG_REFUSAL:
    print("Model refused successfully")
else:
    print(f"Model may be vulnerable: {result.status}")
```

---

## 📁 Project Structure

```
refusal-lens/
├── src/refusal_lens/
│   ├── __init__.py              # Exports
│   ├── config.py                 # ✅ Centralized config
│   ├── circuits/
│   │   ├── __init__.py
│   │   ├── refusal_direction.py # ✅ Step 1
│   │   └── attribution.py       # ✅ Step 2
│   ├── supernode_analyzer.py    # ✅ Step 3
│   ├── refusal_detector.py      # ✅ Step 4
│   ├── refusal_classifier.py
│   ├── experiment_runner.py
│   ├── jailbreak_tracer.py
│   └── prompt_template.py
├── scripts/
│   ├── compute_directions.py    # ✅ Step 1 script
│   ├── compute_circuits.py      # ✅ Step 2 script
│   ├── explore_supernodes.py    # ✅ Step 3 script
│   └── test_jailbreaks.py       # ✅ Step 4 script
├── dataset/
│   └── refusal_direction_dataset/
│       └── splits/
│           ├── harmful_train.json
│           ├── harmless_train.json
│           └── ...
├── data/results/
│   ├── computed_directions/     # Week 1 output
│   ├── circuits/                # Week 2 output
│   └── jailbreak_tests/         # Week 4 output
├── PLAN.md                      # This file
└── pyproject.toml               # ✅ Complete dependencies
```

---

## 🚀 Quick Start

### Installation

```bash
pip install -e .
# Or with all dependencies:
pip install -e ".[dev]"
```

### Run Step 1 (Compute Directions)

```bash
python scripts/compute_directions.py
```

### Run Step 2 (Analyze Circuits)

```bash
python scripts/compute_circuits.py --layer 9
```

### Run Step 3 (Explore Supernodes)

```bash
python scripts/explore_supernodes.py
```

### Run Step 4 (Test Jailbreaks)

```bash
python scripts/test_jailbreaks.py
```

---

## 📊 Configuration Options

See [`src/refusal_lens/config.py`](src/refusal_lens/config.py) for all settings:

| Category | Key | Default | Description |
|----------|-----|---------|-------------|
| Model | `MODEL_NAME` | google/gemma-3-4b-it | Model to analyze |
| Model | `DEVICE` | cuda | Computation device |
| Step 1 | `REFUSAL_LAYERS` | list(range(8, 24)) | Layers to compute |
| Step 1 | `REFUSAL_POSITION` | last_prompt_token | Activation position |
| Step 2 | `MEASUREMENT_LAYER` | 20 | Attribution layer |
| Step 3 | `NEURONPEDIA_CACHE_DIR` | data/results/neuronpedia_cache | Supernode data |
| Testing | `N_BASE_PROMPTS` | 15 | Base prompts |
| Testing | `MAX_NEW_TOKENS` | 512 | Max generation |

---

## 📝 Next Steps

1. **Run Week 1**: `python scripts/compute_directions.py`
2. **Identify best layer**: Check `data/results/computed_directions/summary.json` for separation scores
3. **Run Week 2**: `python scripts/compute_circuits.py --layer <best_layer>`
4. **Run Week 3**: `python scripts/explore_supernodes.py`
5. **Run Week 4**: `python scripts/test_jailbreaks.py`

## 🧪 Tested Models

- google/gemma-3-4b-it ✅

---

## 📚 References

- Arditi et al. 2024 - Refusal direction computation
- Conmy et al. 2023 - Attribution circuit analysis
- Anthropic - Supercircuits research
