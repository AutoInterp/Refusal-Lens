# Feature Descriptions

Generate feature descriptions from `feature_labels_cache.json` files using the Neuronpedia API (with local logits as a fallback).

## Quick Start (Test First!)

Before running on the full cache (which can take hours), test on 3 features:

```bash
python3 descriptions/generate_descriptions.py \
  data/results/pipeline_runs/run_20260430_023247/04_labels/feature_labels_cache.json \
  -n 3
```

This verifies your API key, network, and output format in under a minute.

## Usage

```bash
# Quick test — first N features only (great for debugging)
python3 descriptions/generate_descriptions.py path/to/feature_labels_cache.json -n 3
python3 descriptions/generate_descriptions.py path/to/feature_labels_cache.json --limit 10

# Local generation (from logits) — no API key needed
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

## CLI Flags

| Flag | Description |
|------|-------------|
| `cache_file` | Path to `feature_labels_cache.json` (positional, required for normal runs) |
| `-o, --output DIR` | Output directory (default: `<run_dir>/feature_labels/`) |
| `-n, --limit N` | Process only the first N features — useful for testing |
| `--no-api` | Skip Neuronpedia/OpenRouter, generate from logits only |
| `--api-key KEY` | Neuronpedia API key (overrides `NEURONPEDIA_API_KEY` env var) |
| `--regenerate-labels DIR` | Recompute short labels from existing description files |

## Setup

1. Copy `.env.example` to `.env` and add your API keys:
   ```bash
   cp descriptions/.env.example descriptions/.env
   # Edit descriptions/.env to add your keys
   ```

   Required (for API mode):
   - `NEURONPEDIA_API_KEY` — get from https://neuronpedia.org/account
   
   Optional fallback:
   - `OPENROUTER_API_KEY` — used if Neuronpedia returns no result

2. Install dependencies:
   ```bash
   python3 -m pip install requests python-dotenv
   ```

## Configuration

Edit the constants at the top of `generate_descriptions.py` to change the target SAE/model:

```python
MODEL_ID                  = "gemma-3-4b-it"
SAE_TEMPLATE              = "{layer}-gemmascope-2-transcoder-16k"
DEFAULT_EXPLANATION_TYPE  = "oai_token-act-pair"
DEFAULT_EXPLANATION_MODEL = "gpt-5-nano"
```

## Progress Output

The script prints a live progress line for every feature:

```
[   42/ 1500]   2.8% | L 6:F1234  | NP  | elapsed   1m 23s | ETA  48m 17s | activates on tokens related to refusal…
```

| Column | Meaning |
|--------|---------|
| `[42/1500]` | Current / total |
| `2.8%` | Percent complete |
| `L 6:F1234` | Layer and feature index |
| `NP` / `OR` / `LOG` / `--` | Source: Neuronpedia / OpenRouter / local logits / empty |
| `elapsed` | Time since start |
| `ETA` | Estimated time remaining |
| Last column | First 55 chars of the description |

### Checkpointing

The script saves partial output every **100 features**, so `Ctrl+C` is safe — anything already processed is on disk.

## Time Estimates

| # of features | Mostly cached | Mixed | All need generation |
|---------------|--------------:|------:|--------------------:|
| 100           | ~3 min        | ~15 min | ~25 min |
| 1,000         | ~33 min       | ~2.5 h | ~4 h |
| 5,000         | ~3 h          | ~12 h  | ~21 h |

`/api/feature` (read existing) is fast (~1–2s). `/api/explanation/generate` triggers a server-side LLM call and may take 5–30s per feature.

## Output

Per-layer JSON files in `feature_labels/` (or the run folder if no `-o` specified):

```
feature_labels/
├── feature_labels_layer_0.json
├── feature_labels_layer_1.json
└── ...                          (one file per layer)
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
- `"activates on:"` / `"suppresses:"` prefixes are removed from labels
- Example: `"label": "amic, Descent, Company, Preface, incompatible"`

## Source Priority

For each feature, the script tries sources in this order, stopping at the first success:

1. **Existing Neuronpedia explanation** — `GET /api/feature/...`
2. **Newly generated Neuronpedia explanation** — `POST /api/explanation/generate` (requires API key)
3. **OpenRouter LLM fallback** — only if `OPENROUTER_API_KEY` is set
4. **Local logits fallback** — built from the cache's `top_logits` / `bottom_logits`

The final summary tells you how many features came from each source:

```
Sources: neuronpedia=1200, openrouter=0, local_logits=250, empty=50
(of which 50 features had no Neuronpedia activations)
```

## Troubleshooting

**`zsh: command not found: python`** → use `python3`.

**`No such file or directory`** → check the path. If your folder name has spaces, wrap it in quotes:
```bash
python3 descriptions/generate_descriptions.py "data/.../feature_labels_cache.json"
```

**`HTTP 429`** → you're being rate-limited. The script retries with backoff. If it persists, increase `SLEEP_BETWEEN_REQUESTS` at the top of the script.

**`HTTP 400 "No activations found"`** → expected for sparsely-populated SAE features. The script tallies these and falls back to local logits.

**`"An auto-interp ... already exists"`** → the explanation already exists; the script will use it via the `GET /api/feature` step instead.