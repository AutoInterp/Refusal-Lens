# Circuit Analysis of Refusal Mechanisms in Qwen3-4B

**Model:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
**Transcoders:** [mwhanna/qwen3-4b-transcoders](https://huggingface.co/mwhanna/qwen3-4b-transcoders)
**Neuronpedia data:** [v1/qwen3-4b/](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/qwen3-4b/)

This is a parallel of [`data/tejas_experiments/`](../tejas_experiments/) — same
methodology, different base model. The goal is to compare refusal mechanisms
across model families: are the findings (corrected direction, causal layer,
two-jailbreak-class taxonomy) general, or Gemma-specific?

## Status

All scripts are templates. Run them in order to populate `results_v2/` and
`figures/`, then update [`../MODEL_COMPARISON.md`](../MODEL_COMPARISON.md) with
the actual numbers.

| Script | Status |
|---|---|
| `01_compute_direction_and_sanity.py` | Ported (full) — run this first |
| `11_fix_all_qwen.py` | Ported (full) |
| `20_bulletproof_pipeline.py` | Ported (core phases) |
| 12, 13, 14, 15, 16, 17, 19, 21, 22 | Stubs — port from Gemma versions |

See [`scripts/INDEX.md`](scripts/INDEX.md) for the porting checklist and run order.

## What needs to be re-discovered for Qwen

These were hardcoded for Gemma; **all must be re-tuned empirically**:

- **Best position** (Gemma: -2 = `model` token). Qwen's chat template is
  `<|im_start|>assistant\n` — different end-of-prompt structure entirely.
- **Best layer** (Gemma: 32 of 34 for separation strength).
- **Causal layer** (Gemma: 15 — where Arditi intervention works). Qwen has
  36 layers; the causally effective one is unknown a priori.
- **Transcoder subpath** (Gemma: `transcoder_all/width_16k_l0_small_affine`).
  Inspect the HF repo and pick the analogous Qwen variant.

## What stays the same

- **Numerical recipe** (Georg's corrections): float64 accumulation, left padding,
  multi-position sweep at `[-5,-4,-3,-2,-1]`.
- **Dataset**: `dataset/refusal_direction_dataset/splits/` (harmful + harmless),
  and `dataset/refusal_lens_controlled_dataset.json` (50 prompts × 5 classes).
- **Methodology**: difference-in-means refusal direction, circuit-tracer
  attribution, Arditi-style causal intervention, Q/K head decomposition.

## Running on RunPod

The Gemma scripts in `data/tejas_experiments/scripts/` were originally run
on RunPod (their hardcoded `/workspace/...` paths come from RunPod's
persistent-volume mount point). The Qwen scripts use repo-relative paths via
`CONFIG.py`, so they're portable, but RunPod is still the recommended
environment because each script needs ~16-24 GB VRAM (model in bfloat16 +
transcoders + gradient buffers for circuit-tracer).

### 1. Pick a pod

| Resource | Recommended | Why |
|---|---|---|
| GPU | **1× A100 80GB** or **1× H100 80GB** | Circuit-tracer attribution with `max_feature_nodes=None` peaks above 40 GB; 24 GB cards (A5000, 3090) will OOM on script 11. |
| Container | RunPod's **`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`** template | PyTorch + CUDA preinstalled. |
| Container disk | 50 GB | Code + venv + temp. |
| Volume | **100 GB on `/workspace`** | Persistent across pod restarts; holds HF cache (~30 GB for Qwen3-4B + transcoders) and results. Don't put these on the container disk — you'll lose them when the pod stops. |

If you only have 24 GB available you can still run scripts 01 and 20 (forward passes only); script 11 (circuit-tracer attribution) needs the bigger card.

### 2. One-time setup on the pod

SSH in via the RunPod web terminal or `ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519`, then:

```bash
# Move into the persistent volume so nothing you build is lost on pod restart
cd /workspace

# Clone the repo (if not already there)
git clone https://github.com/AutoInterp/Refusal-Lens.git
cd Refusal-Lens
git checkout qwen_experiments    # or your working branch

# Persistent caches on the volume — saves re-downloading 30 GB of weights
export HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

# Make these survive new shells
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/.cache/huggingface
export TMPDIR=/workspace/tmp
EOF

# Python env
python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install -e ".[dev]"
pip install nnsight
pip install git+https://github.com/safety-research/circuit-tracer.git

# HuggingFace login (Qwen3-4B is gated only for some accounts; do this if you hit a 403)
huggingface-cli login
```

### 3. Fill in `CONFIG.py`

Open `data/qwen_experiments/scripts/CONFIG.py` and set:

```python
TRANSCODER_SUBPATH = "..."   # inspect https://huggingface.co/mwhanna/qwen3-4b-transcoders
                             # Gemma analogue was "transcoder_all/width_16k_l0_small_affine"
```

Easiest way to inspect from the pod:

```bash
huggingface-cli download mwhanna/qwen3-4b-transcoders --include "*.json" --local-dir /tmp/qwen-tc
ls /tmp/qwen-tc        # see what subdirectories exist
```

### 4. Run the experiments in order

Use `nohup` + a log file so you can disconnect SSH while long jobs run.

```bash
cd /workspace/Refusal-Lens
source /workspace/venv/bin/activate

# Step 01: discover (best_pos, best_layer). ~30 min on A100.
nohup python data/qwen_experiments/scripts/01_compute_direction_and_sanity.py \
      > data/qwen_experiments/results_v2/01.log 2>&1 &
tail -f data/qwen_experiments/results_v2/01.log
# When done, the last line tells you what to put in CONFIG.py:
#   "Update CONFIG.py: QWEN_BEST_POSITION=-?, QWEN_BEST_LAYER=??"

# Edit CONFIG.py with those values.
nano data/qwen_experiments/scripts/CONFIG.py

# Step 11: 10-pair attribution + RP-vs-fiction mechanism. ~2-3 hr on A100.
nohup python data/qwen_experiments/scripts/11_fix_all_qwen.py \
      > data/qwen_experiments/results_v2/11.log 2>&1 &
tail -f data/qwen_experiments/results_v2/11.log

# Port + run scripts 15/16/17 to find the causal layer (sweep candidates).
# Update QWEN_CAUSAL_LAYER in CONFIG.py with the chosen value.

# Step 20: bulletproof end-to-end. ~3-4 hr on A100.
nohup python data/qwen_experiments/scripts/20_bulletproof_pipeline.py \
      > data/qwen_experiments/results_v2/20.log 2>&1 &
tail -f data/qwen_experiments/results_v2/20.log
```

### 5. Pull results back to your laptop

Results live in `data/qwen_experiments/results_v2/` and `data/qwen_experiments/figures/`. From your local machine:

```bash
# rsync just the output directories (skip caches and the venv)
rsync -avz --progress \
    -e "ssh -p <pod-ssh-port> -i ~/.ssh/id_ed25519" \
    root@<pod-ip>:/workspace/Refusal-Lens/data/qwen_experiments/results_v2/ \
    ./data/qwen_experiments/results_v2/

rsync -avz --progress \
    -e "ssh -p <pod-ssh-port> -i ~/.ssh/id_ed25519" \
    root@<pod-ip>:/workspace/Refusal-Lens/data/qwen_experiments/figures/ \
    ./data/qwen_experiments/figures/
```

Or commit and push from the pod:

```bash
cd /workspace/Refusal-Lens
git add data/qwen_experiments/results_v2 data/qwen_experiments/figures
git commit -m "Qwen3-4B: results from script XX"
git push
```

### 6. Stop the pod

**Important.** RunPod charges by the second the pod is running. After pulling results:

- "Stop" preserves the volume (cheap storage cost) — use this if you'll re-run soon.
- "Terminate" deletes everything including the volume — only after you've pulled results.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` on script 11 | 24 GB card | Move to 80 GB pod, or pass `max_feature_nodes=3000` (loses 0.1-1.3% precision). |
| `OSError: ... not found` on transcoder | Wrong `TRANSCODER_SUBPATH` | Inspect repo with `huggingface-cli download --include "*.json"`. |
| `Repository ... is gated` on Qwen | Need HF login | `huggingface-cli login` with a token that has accepted the Qwen license. |
| Loss of results after pod stop | Wrote to container disk, not volume | Always work under `/workspace/`; check `pwd`. |
| HF cache redownloaded after pod restart | `HF_HOME` not set | `echo $HF_HOME` should print `/workspace/.cache/huggingface`. |

### After all scripts finish

Update [`../MODEL_COMPARISON.md`](../MODEL_COMPARISON.md) — replace the
`_TBD_` placeholders in the Qwen columns with the numbers from
`results_v2/` JSON files. The same JSON file paths exist for both Gemma
and Qwen, so the mapping is straightforward.

See [`MIGRATION_NOTES.md`](MIGRATION_NOTES.md) for the full Gemma→Qwen
adaptation rationale.
