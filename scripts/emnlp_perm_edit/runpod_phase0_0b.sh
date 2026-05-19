#!/usr/bin/env bash
# RunPod launch script for Phase 0 sub-experiment 0b-simple.
#
# Assumes a fresh RunPod instance (preferably H100 SXM or A100 80GB; RTX 4090
# 24GB also works) with /workspace as the working volume. Tested against:
#   - RunPod template: pytorch:2.x-py3.12-cuda12.x
#   - Volume: 50 GB minimum (model ~8 GB, HF graph data ~485 MB, code ~50 MB,
#             HF cache ~10 GB headroom)
#
# Usage (from RunPod terminal at /workspace):
#   git clone https://github.com/<your-fork>/Refusal-Lens.git
#   cd Refusal-Lens
#   git checkout emnlp-perm-edit
#   git submodule update --init --recursive
#   bash scripts/emnlp_perm_edit/runpod_phase0_0b.sh
#
# Wall: ~2 h on H100 SXM; ~3.5 h on RTX 4090. tmux-persistent so you can detach.
# Result: data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json
# Pull result back via: scp / rsync / hf upload (manual; not automated).

set -euo pipefail

SESSION="phase0_0b"
LOG="/tmp/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
ROOT="${ROOT:-$(pwd)}"

echo "============================================================"
echo "Phase 0 0b-simple RunPod launcher"
echo "Repo root: $ROOT"
echo "Log: $LOG"
echo "============================================================"

# --- Re-launch into tmux if not already detached ---
if [[ -z "${TMUX:-}" ]] && [[ "${NO_TMUX:-0}" != "1" ]]; then
  echo "Relaunching into tmux session '$SESSION' (set NO_TMUX=1 to disable)..."
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists. Kill it first:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" "bash $0 2>&1 | tee $LOG"
  echo "Detached tmux session started. Reattach with:"
  echo "  tmux attach -t $SESSION"
  echo "Watch live log:"
  echo "  tail -f $LOG"
  exit 0
fi

# --- Verify we're at the right place ---
cd "$ROOT"
if [[ ! -f "scripts/emnlp_perm_edit/00_edge_ablation_runtime.py" ]]; then
  echo "ERROR: expected to run from Refusal-Lens repo root. cwd=$(pwd)"
  exit 1
fi

# --- Step 1: Python env ---
echo ""
echo "=== Step 1: Python environment ==="
if [[ ! -d ".venv" ]]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip -q

echo "Installing pyproject deps + vendor/circuit-tracer (editable)..."
pip install -e . -q
pip install -e ./vendor/circuit-tracer -q
pip install pytest -q

# Verify CUDA torch
python3 -c "import torch; assert torch.cuda.is_available(), f'CUDA not available; torch.__version__={torch.__version__}'; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}')"

# --- Step 2: Pull HF graph data (needed for 0a outputs that 0b consumes) ---
echo ""
echo "=== Step 2: HF graph data ==="
if [[ ! -d "data/results/pipeline_runs/run_20260430_023247/05_frontend/graph_data" ]] || \
   [[ $(ls data/results/pipeline_runs/run_20260430_023247/05_frontend/graph_data/*_single.json.gz 2>/dev/null | wc -l) -lt 550 ]]; then
  echo "Pulling packed graphs from moon70/refusal-lens-graphs (~485 MB)..."
  PYTHONPATH=scripts/pipeline python3 scripts/pipeline/fetch_graph_data.py \
    --run run_20260430_023247 --dataset-repo moon70/refusal-lens-graphs
else
  echo "Graph data already present locally; skipping pull."
fi

# --- Step 3: Regenerate 0a output if needed (cheap CPU step) ---
echo ""
echo "=== Step 3: 0a linearization decomposition (CPU prerequisite for 0b deltas) ==="
DECOMP_PATH="data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json"
if [[ ! -f "$DECOMP_PATH" ]]; then
  echo "Running 0a (linearization decomposition)..."
  PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_linearization_decomposition.py
else
  echo "0a output already exists; skipping. To force re-run, delete: $DECOMP_PATH"
fi

# --- Step 4: 0b-simple full run ---
echo ""
echo "=== Step 4: 0b-simple full run (7 variants x 550 prompts) ==="
echo "Expected wall: ~2 h on H100 SXM, ~3.5 h on RTX 4090."
echo "Saves incrementally every 100 generations + after each variant."
echo ""

PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py

# --- Step 5: Summary ---
echo ""
echo "============================================================"
echo "Phase 0 0b-simple DONE"
echo "============================================================"
ls -la data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json
echo ""
echo "Pull result back to laptop with (example):"
echo "  scp -P <port> root@<pod-ip>:$ROOT/data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json ."
echo ""
echo "Or push to HF (via push helpers if you set HF_TOKEN):"
echo "  PYTHONPATH=scripts/pipeline python3 scripts/pipeline/push_run.py \\"
echo "      --run-dir data/results/pipeline_runs/run_20260430_023247 --skip-graphs"
