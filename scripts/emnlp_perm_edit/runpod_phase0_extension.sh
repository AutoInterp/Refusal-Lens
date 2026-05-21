#!/usr/bin/env bash
# RunPod launch script for the Phase 0 extension:
#   2x2 (direction vs edge x position-mode) + full coefficient sweep.
#
# Tests Georg's hypothesis: does our edge-ablation result reflect L15:r_hat's
# true causal role, or is it confounded by magnitude / position differences vs
# the canonical Arditi-style direction intervention?
#
# Sub-experiments (all fp32 for consistency with the rest of the Phase 0 suite):
#
#   STEP 1 — direction sweep @ all positions, 8 coefficients
#     {0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0}
#     Subtract coeff * r_hat at every position every forward step (Arditi).
#     At coeff=1.0 this should reproduce the 98% bare->comply from Stage 06;
#     at coeff~0.005 it should ~match the 6% from existing edge-ablation runs.
#
#   STEP 2 — direction sweep @ pos=-2 only, same 8 coefficients
#     Same as STEP 1 but the hook only edits seq pos=-2 of the prompt encoding
#     pass. Tells us how much of the all-positions flip depends on position vs
#     magnitude. coeff=1.0 here is Cell B of the 2x2.
#
#   STEP 3 — edge ablation @ pos=-2 only, 2 variants
#     {ablate_all_edges, ablate_all_2x}
#     Re-runs the existing edge ablation with position-mode=last_prompt_only,
#     giving us Cell D of the 2x2. The other two cells (all_edges + 2x at all
#     positions) are already in edge_ablation_flip_rates.json.
#
#   STEP 4 — layer locator @ pos=-2, coeff=1.0, 8 non-L15 layers
#     Layers: {0, 3, 6, 9, 12, 18, 21, 24}  (every 3rd, skipping L15 since
#     STEP 2 covers it)
#     Pre-emptive depth profile in case STEP 2's L15-pos=-2 result is weak.
#     If one of these layers lights up at coeff=1.0 pos=-2, that's a clear
#     signal for where the actual causal locus is.
#
# Expected wall on H100 SXM fp32:
#   STEP 1: ~30 min * 8 = 4 h
#   STEP 2: ~30 min * 8 = 4 h
#   STEP 3: ~30 min * 2 = 1 h
#   STEP 4: ~30 min * 8 = 4 h
#   Total:  ~13 h
#
# Continue-on-failure: per-step failures recorded but the launcher proceeds.
# .PHASE0_EXT_DONE marker is always touched at the end so the watcher pushes.
#
# Usage (from RunPod terminal at /workspace):
#   git clone https://github.com/<your-fork>/Refusal-Lens.git
#   cd Refusal-Lens
#   git checkout emnlp-perm-edit
#   git submodule update --init --recursive
#   bash scripts/emnlp_perm_edit/runpod_phase0_extension.sh
#
# Optional env vars:
#   COEFFS               override coefficient list (CSV)
#   NO_TMUX=1            skip the self-relaunch into tmux
#
# Completion signals:
#   data/results/emnlp_perm_edit/phase0_controllability/.PHASE0_EXT_DONE          (always touched at end)
#   data/results/emnlp_perm_edit/phase0_controllability/.PHASE0_EXT_STEP_FAILED.txt (one line per failed step)

set -uo pipefail   # NB: no -e so per-step failures don't kill the script

SESSION="phase0_ext"
LOG="/tmp/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
ROOT="${ROOT:-$(pwd)}"
COEFFS="${COEFFS:-0.001,0.005,0.01,0.05,0.1,0.25,0.5,1.0}"
LOCATOR_LAYERS="${LOCATOR_LAYERS:-0,3,6,9,12,18,21,24}"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability"
DONE_FILE="$OUT_DIR/.PHASE0_EXT_DONE"
FAIL_FILE="$OUT_DIR/.PHASE0_EXT_STEP_FAILED.txt"

echo "============================================================"
echo "Phase 0 extension launcher (continue-on-failure)"
echo "Repo root: $ROOT"
echo "Coefficients: $COEFFS"
echo "Locator layers: $LOCATOR_LAYERS"
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
  tmux new-session -d -s "$SESSION" "COEFFS='$COEFFS' LOCATOR_LAYERS='$LOCATOR_LAYERS' bash $0 2>&1 | tee $LOG"
  echo "Detached tmux session started. Reattach with:"
  echo "  tmux attach -t $SESSION"
  echo "Watch live log:"
  echo "  tail -f $LOG"
  echo ""
  echo "To auto-commit results when complete, in a SEPARATE terminal run:"
  echo "  bash scripts/emnlp_perm_edit/watch_and_commit_phase0_extension.sh"
  exit 0
fi

cd "$ROOT"
if [[ ! -f "scripts/emnlp_perm_edit/00_direction_intervention_sweep.py" ]]; then
  echo "ERROR: expected to run from Refusal-Lens repo root. cwd=$(pwd)"
  exit 1
fi

mkdir -p "$OUT_DIR"
rm -f "$DONE_FILE" "$FAIL_FILE"

declare -a STEP_NAMES
declare -a STEP_RESULTS
run_step() {
  local name="$1"; shift
  echo ""
  echo "########## STEP: $name ##########"
  local t0=$(date +%s)
  if "$@"; then
    local t1=$(date +%s)
    echo "########## STEP $name OK ($((t1-t0))s wall) ##########"
    STEP_NAMES+=("$name")
    STEP_RESULTS+=("OK")
  else
    local rc=$?
    local t1=$(date +%s)
    echo "########## STEP $name FAILED (exit $rc, $((t1-t0))s wall) — continuing ##########"
    echo "$name (exit $rc)" >> "$FAIL_FILE"
    STEP_NAMES+=("$name")
    STEP_RESULTS+=("FAILED")
  fi
}

# --- Env setup (mirrors runpod_phase0_all.sh) ---
echo ""
echo "=== Step A: Python environment ==="
if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip -q
pip install -e . -q
pip install -e ./vendor/circuit-tracer -q

python3 -c "import torch; assert torch.cuda.is_available(), f'CUDA NOT available; torch={torch.__version__}'; print(f'CUDA OK: torch {torch.__version__} device={torch.cuda.get_device_name(0)}')"
if [[ $? -ne 0 ]]; then
  echo "FATAL: CUDA torch not available. Aborting." | tee -a "$FAIL_FILE"
  touch "$DONE_FILE"
  exit 1
fi

# --- STEP 1: direction sweep @ L15, all positions ---
run_step "direction_sweep_all_positions" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep.py \
      --layers 15 \
      --position-mode all \
      --coefficients '$COEFFS' \
      --out '$OUT_DIR/direction_intervention_sweep_all.json'"

# --- STEP 2: direction sweep @ L15, pos=-2 only ---
run_step "direction_sweep_pos_neg2" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep.py \
      --layers 15 \
      --position-mode last_prompt_only \
      --coefficients '$COEFFS' \
      --out '$OUT_DIR/direction_intervention_sweep_pos2.json'"

# --- STEP 3: edge ablation @ pos=-2 only (Cell D) ---
run_step "edge_ablation_pos_neg2" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py \
      --position-mode last_prompt_only \
      --variants 'ablate_all_edges,ablate_all_2x' \
      --out '$OUT_DIR/edge_ablation_pos2_flip_rates.json'"

# --- STEP 4: layer locator @ pos=-2, coeff=1.0, non-L15 layers ---
# Pre-emptive depth profile in case STEP 2 (L15 pos=-2) is weak.
run_step "layer_locator_pos_neg2" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep.py \
      --layers '$LOCATOR_LAYERS' \
      --position-mode last_prompt_only \
      --coefficients '1.0' \
      --out '$OUT_DIR/layer_locator_pos2_coeff1.json'"

# --- Final summary ---
echo ""
echo "============================================================"
echo "PHASE 0 EXTENSION COMPLETE"
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
ls -la "$OUT_DIR"/direction_intervention_sweep_*.json 2>/dev/null
ls -la "$OUT_DIR"/edge_ablation_pos2_flip_rates.json 2>/dev/null
ls -la "$OUT_DIR"/layer_locator_pos2_coeff1.json 2>/dev/null

# Always touch the DONE marker so the watcher knows we're finished
touch "$DONE_FILE"
echo "Touched $DONE_FILE"
echo ""
echo "To auto-commit + push, run (in another terminal):"
echo "  bash scripts/emnlp_perm_edit/watch_and_commit_phase0_extension.sh"
