#!/usr/bin/env bash
# RunPod launch script for ALL Phase 0 GPU sub-experiments: 0b + 0d + 0e.
#
# Runs sequentially in a single tmux session to minimize cold-start overhead.
# Expected total wall:
#   - 0b: ~2h (H100) / ~3.5h (4090) — 7 variants × 550 prompts × 80 tokens
#   - 0d: ~6h (H100) / ~6h+ (4090) — 3 variants × 7 K × 550 prompts × 80 tokens
#   - 0e: ~6h (H100) / ~6h+ (4090) — same shape as 0d but all edge types
#   Total: ~14 h on H100, ~16+ h on 4090.
#
# Scope-trim option: set K_MODE=coarse to reduce K_values for 0d/0e
# (`--k-values 5,50,500` instead of full 7-point sweep). Saves ~8 h GPU total.
#
# Assumes a fresh RunPod instance (H100 SXM 80GB recommended; A100 80GB ok;
# RTX 4090 24GB works but slower) with /workspace as the working volume.
# Tested template: pytorch:2.x-py3.12-cuda12.x
# Volume: 50 GB minimum.
#
# Usage (from RunPod terminal at /workspace):
#   git clone https://github.com/<your-fork>/Refusal-Lens.git
#   cd Refusal-Lens
#   git checkout emnlp-perm-edit
#   git submodule update --init --recursive
#   bash scripts/emnlp_perm_edit/runpod_phase0_all.sh
#   # Optional: K_MODE=coarse bash scripts/.../runpod_phase0_all.sh
#
# After completion, pull all 3 result JSONs back via scp/rsync. Then run
# `00_aggregate_phase0_gpu.py` locally to produce figures + summary MD.

set -euo pipefail

SESSION="phase0_all"
LOG="/tmp/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
ROOT="${ROOT:-$(pwd)}"
K_MODE="${K_MODE:-full}"   # "full" | "coarse"

echo "============================================================"
echo "Phase 0 0b + 0d + 0e RunPod launcher"
echo "Repo root: $ROOT"
echo "K_MODE: $K_MODE"
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
  tmux new-session -d -s "$SESSION" "K_MODE='$K_MODE' bash $0 2>&1 | tee $LOG"
  echo "Detached tmux session started. Reattach with:"
  echo "  tmux attach -t $SESSION"
  echo "Watch live log:"
  echo "  tail -f $LOG"
  exit 0
fi

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

pip install --upgrade pip -q
pip install -e . -q
pip install -e ./vendor/circuit-tracer -q
pip install pytest -q

python3 -c "import torch; assert torch.cuda.is_available(), f'CUDA not available; torch={torch.__version__}'; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}')"

# --- Step 2: HF graph data ---
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

# --- Step 3: 0a linearization decomposition (CPU prerequisite for 0b) ---
echo ""
echo "=== Step 3: 0a linearization decomposition (CPU prerequisite for 0b) ==="
DECOMP_PATH="data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json"
if [[ ! -f "$DECOMP_PATH" ]]; then
  echo "Running 0a..."
  PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_linearization_decomposition.py
else
  echo "0a output present; skipping."
fi

# --- Step 4: 0b-simple ---
echo ""
echo "=== Step 4: 0b-simple (7 variants × 550 prompts × 80 tokens) ==="
echo "Expected: ~2h H100 / ~3.5h 4090. Saves incrementally."
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py

# --- Step 5: 0d top-K feature sweep ---
echo ""
echo "=== Step 5: 0d top-K feature sweep ==="
if [[ "$K_MODE" == "coarse" ]]; then
  K_FLAG="--k-values 5,50,500"
  echo "Coarse mode: $K_FLAG (~2h on H100)"
else
  K_FLAG=""
  echo "Full mode: 7 K values (~6h on H100)"
fi
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_sweep.py --mode features $K_FLAG

# --- Step 6: 0e top-K edge sweep ---
echo ""
echo "=== Step 6: 0e top-K edge sweep ==="
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_sweep.py --mode edges $K_FLAG

# --- Step 7: Summary ---
echo ""
echo "============================================================"
echo "Phase 0 0b + 0d + 0e DONE"
echo "============================================================"
echo ""
echo "Result files:"
ls -la data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json
ls -la data/results/emnlp_perm_edit/phase0_controllability/topk_feature_sweep.json
ls -la data/results/emnlp_perm_edit/phase0_controllability/topk_edge_sweep.json
echo ""
echo "Pull all 3 result JSONs back to laptop, then run locally:"
echo "  PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_aggregate_phase0_gpu.py"
echo ""
echo "This produces controllability_audit_figure.png, topk_feature_pareto_figure.png,"
echo "topk_edge_vs_node_figure.png, flip_rate_summary.json, and PHASE0_GPU_SUMMARY.md."
