#!/usr/bin/env bash
# RunPod launch script for ALL Phase 0 GPU sub-experiments: drift check + 0b + 0d + 0e.
#
# Continue-on-failure semantics: if any sub-experiment crashes (OOM, transient
# CUDA error, model-load issue), the remaining sub-experiments still run.
# Per-step status is recorded in .PHASE0_STEP_FAILED.txt and the final summary.
#
# Expected total wall:
#   - drift check: ~3 min
#   - 0b: ~2h (H100) / ~3.5h (4090) — 7 variants × 550 prompts × 80 tokens
#   - 0d: ~6h (H100) — 3 variants × 7 K × 550 prompts × 80 tokens (full mode)
#   - 0e: ~6h (H100) — same shape as 0d (full mode)
#   Total: ~14 h H100 full / ~6 h H100 coarse / ~16+ h 4090 full
#
# Scope-trim option: K_MODE=coarse reduces 0d/0e K values to {5, 50, 500} (~4h instead of ~12h).
#
# RunPod template: pytorch:2.x-py3.12-cuda12.x; volume 50 GB minimum.
#
# Usage (from RunPod terminal at /workspace):
#   git clone https://github.com/<your-fork>/Refusal-Lens.git
#   cd Refusal-Lens
#   git checkout emnlp-perm-edit
#   git submodule update --init --recursive
#   bash scripts/emnlp_perm_edit/runpod_phase0_all.sh
#   # Optional: K_MODE=coarse bash scripts/.../runpod_phase0_all.sh
#
# Completion signals:
#   data/results/emnlp_perm_edit/phase0_controllability/.PHASE0_DONE     (always touched at end)
#   data/results/emnlp_perm_edit/phase0_controllability/.PHASE0_STEP_FAILED.txt (one line per failed step)
#
# After completion, scripts/emnlp_perm_edit/watch_and_commit_phase0.sh
# (run in a separate terminal) picks up the marker and pushes results to GitHub.

set -uo pipefail   # NB: no -e so per-step failures don't kill the script

SESSION="phase0_all"
LOG="/tmp/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
ROOT="${ROOT:-$(pwd)}"
K_MODE="${K_MODE:-full}"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability"
DONE_FILE="$OUT_DIR/.PHASE0_DONE"
FAIL_FILE="$OUT_DIR/.PHASE0_STEP_FAILED.txt"

echo "============================================================"
echo "Phase 0 GPU launcher (continue-on-failure)"
echo "Repo root: $ROOT"
echo "K_MODE: $K_MODE"
echo "Log: $LOG"
echo "Completion marker: $DONE_FILE"
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
  echo ""
  echo "To auto-commit results when complete, in a SEPARATE terminal run:"
  echo "  bash scripts/emnlp_perm_edit/watch_and_commit_phase0.sh"
  exit 0
fi

cd "$ROOT"
if [[ ! -f "scripts/emnlp_perm_edit/00_edge_ablation_runtime.py" ]]; then
  echo "ERROR: expected to run from Refusal-Lens repo root. cwd=$(pwd)"
  exit 1
fi

mkdir -p "$OUT_DIR"
# Clear stale markers from previous runs
rm -f "$DONE_FILE" "$FAIL_FILE"

# Helper: run a step and record success/failure without aborting the script.
declare -a STEP_NAMES
declare -a STEP_RESULTS
run_step() {
  local name="$1"; shift
  echo ""
  echo "########## STEP: $name ##########"
  local t0=$(date +%s)
  if "$@"; then
    local t1=$(date +%s)
    echo "########## STEP $name OK (${t0}s wall) ##########"
    STEP_NAMES+=("$name")
    STEP_RESULTS+=("OK")
  else
    local rc=$?
    local t1=$(date +%s)
    echo "########## STEP $name FAILED (exit $rc) — continuing to next step ##########"
    echo "$name (exit $rc)" >> "$FAIL_FILE"
    STEP_NAMES+=("$name")
    STEP_RESULTS+=("FAILED")
  fi
}

# --- Step A: env setup ---
echo ""
echo "=== Step A: Python environment ==="
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip -q
pip install -e . -q
pip install -e ./vendor/circuit-tracer -q
pip install pytest scikit-learn matplotlib -q

# Verify CUDA torch is actually available — abort EARLY if not, since no point continuing
python3 -c "import torch; assert torch.cuda.is_available(), f'CUDA NOT available; torch={torch.__version__}. Re-install torch with CUDA support before retrying.'; print(f'CUDA OK: torch {torch.__version__} device={torch.cuda.get_device_name(0)}')"
if [[ $? -ne 0 ]]; then
  echo "FATAL: CUDA torch not available. Aborting before GPU steps." | tee -a "$FAIL_FILE"
  touch "$DONE_FILE"
  exit 1
fi

# --- Step B: HF graph data ---
echo ""
echo "=== Step B: HF graph data ==="
if [[ ! -d "data/results/pipeline_runs/run_20260430_023247/05_frontend/graph_data" ]] || \
   [[ $(ls data/results/pipeline_runs/run_20260430_023247/05_frontend/graph_data/*_single.json.gz 2>/dev/null | wc -l) -lt 550 ]]; then
  echo "Pulling packed graphs from moon70/refusal-lens-graphs (~485 MB)..."
  PYTHONPATH=scripts/pipeline python3 scripts/pipeline/fetch_graph_data.py \
    --run run_20260430_023247 --dataset-repo moon70/refusal-lens-graphs
else
  echo "Graph data already present (>=550 single-mode files); skipping pull."
fi

# --- Step C: 0a linearization decomposition (CPU prerequisite for 0b) ---
echo ""
echo "=== Step C: 0a linearization decomposition (CPU prerequisite for 0b) ==="
DECOMP_PATH="$OUT_DIR/linearization_decomposition.json"
if [[ ! -f "$DECOMP_PATH" ]]; then
  PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_linearization_decomposition.py || \
    echo "0a CLI failed — continuing but 0b will likely abort too." | tee -a "$FAIL_FILE"
else
  echo "0a output present; skipping."
fi

# --- Step 1: drift sanity check ---
run_step "drift_check" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_directdot_drift_verify.py"

# --- Step 2: 0b-simple (7 variants × 550 prompts × 80 tokens) ---
run_step "0b_simple" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py"

# --- Step 3 + 4: top-K sweeps (features then edges) ---
if [[ "$K_MODE" == "coarse" ]]; then
  K_FLAG="--k-values 5,50,500"
else
  K_FLAG=""
fi

run_step "0d_topk_features" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_sweep.py --mode features $K_FLAG"

run_step "0e_topk_edges" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_sweep.py --mode edges $K_FLAG"

# --- Step 5: aggregation (run even if some sweeps failed; aggregator skips missing inputs) ---
run_step "aggregate_gpu_outputs" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_aggregate_phase0_gpu.py"

# --- Final summary ---
echo ""
echo "============================================================"
echo "PHASE 0 GPU RUN COMPLETE"
echo "============================================================"
echo ""
echo "Per-step status:"
for i in "${!STEP_NAMES[@]}"; do
  echo "  ${STEP_NAMES[$i]} : ${STEP_RESULTS[$i]}"
done
echo ""
if [[ -f "$FAIL_FILE" ]]; then
  echo "Some steps failed; see $FAIL_FILE:"
  cat "$FAIL_FILE"
else
  echo "All steps OK."
fi
echo ""
echo "Result files in $OUT_DIR:"
ls -la "$OUT_DIR"/ 2>/dev/null | grep -v "^total" | grep -v "^d"

# Always touch the DONE marker so the watcher knows we're finished (even on partial failure)
touch "$DONE_FILE"
echo "Touched $DONE_FILE"
echo ""
echo "To auto-commit + push, run (in another terminal):"
echo "  bash scripts/emnlp_perm_edit/watch_and_commit_phase0.sh"
