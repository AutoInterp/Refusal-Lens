# Feature Descriptions

Generate feature descriptions from `feature_labels_cache.json` files.

## Usage

```bash
# Local generation (from logits) - no API key needed
# Default output: same folder as input (run_20260430_023247/feature_labels/)
python3 descriptions/generate_descriptions.py data/results/pipeline_runs/run_20260430_023247/04_labels/feature_labels_cache.json --no-api

# Specify custom output location
python3 descriptions/generate_descriptions.py data/results/pipeline_runs/run_20260430_023247/04_labels/feature_labels_cache.json --no-api -o feature_labels

# With Neuronpedia API (requires NEURONPEDIA_API_KEY in .env)
# Default output: same folder as input (run_20260430_023247/feature_labels/)
python3 descriptions/generate_descriptions.py data/results/pipeline_runs/run_20260430_023247/04_labels/feature_labels_cache.json

# Specify custom output location with API
python3 descriptions/generate_descriptions.py data/results/pipeline_runs/run_20260430_023247/04_labels/feature_labels_cache.json -o feature_labels

# Regenerate labels from existing output files
python3 descriptions/generate_descriptions.py --regenerate-labels feature_labels
```

## Setup

1. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp descriptions/.env.example descriptions/.env
   # Edit descriptions/.env to add your keys
   ```

2. Install dependencies:
   ```bash
   python3 -m pip install requests python-dotenv
   ```

## Output

Per-layer JSON files in `feature_labels/` (or run folder if no `-o` specified):

```
feature_labels/
├── feature_labels_layer_0.json  (16 files, layers 0-15)
```

Each file format:
```json
{
  "0": {
    "explanation": {
      "label": "≤7 word label",
      "description": "full explanation text"
    }
  },
  "1": { ... }
}
```

### Label Format
- Labels are automatically truncated to ≤7 words
- "activates on:"/"suppresses:" prefixes are removed from labels
- Example: `"label": "amic, Descent, Company, Preface, incompatible"`