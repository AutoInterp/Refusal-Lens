#!/usr/bin/env bash
# RunPod launcher — Path A re-run (Qwen thinking-mode fix + Gemma supra sign-flip).
#
# Re-runs ONLY the steps invalidated by the two Batch 15 bugs:
#
# Qwen (enable_thinking=False applied via the patched format_prompt):
#   STEP 1 — Qwen 0b edge ablation @ L18 (7 variants)
#   STEP 2 — Qwen direction sweep @ L18 all positions (8 coeffs)
#   STEP 3 — Qwen direction sweep @ L18 pos=-1 (8 coeffs)
#   STEP 4 — Qwen layer locator @ pos=-1 coeff=1.0 (8 layers)
#
# Gemma (sign-flipped supra variants):
#   STEP 5 — Gemma supra-threshold antirefuse {5x, 10x, 50x, 100x, 200x}
#
# Skipped (still valid from Batch 15):
#   - Stage 02 attribution graph generation (550 graphs at L18, on HF + local cache)
#   - Stage 02c packing (graphs already .json.gz on disk)
#   - Qwen 0a linearization (computed from graphs, no model generation involved)
#
# Estimated wall on H100 SXM fp32: ~12 hr, ~$40
#   STEP 1: ~3 hr
#   STEP 2: ~2 hr
#   STEP 3: ~2 hr
#   STEP 4: ~2 hr
#   STEP 5: ~2.5 hr
#
# Usage:
#   bash scripts/emnlp_perm_edit/runpod_path_a_rerun.sh

set -uo pipefail

SESSION="path_a_rerun"
LOG="/tmp/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
ROOT="${ROOT:-$(pwd)}"
COEFFS="${COEFFS:-0.001,0.005,0.01,0.05,0.1,0.25,0.5,1.0}"
LOCATOR_LAYERS="${LOCATOR_LAYERS:-0,5,10,15,22,28,32,34}"
QWEN_RUN_DIR="$ROOT/data/results/pipeline_runs_qwen/run_emnlp_qwen_L18_20260522"
QWEN_DIRECTIONS_RUN="$QWEN_RUN_DIR/01_direction/directions"
QWEN_UNNORM_PATH="$QWEN_RUN_DIR/01_direction/positions_L18/pos_-1_unnormalized.pt"
QWEN_METADATA="$QWEN_RUN_DIR/01_direction/direction_metadata.json"
QWEN_DECOMP="$ROOT/data/results/emnlp_perm_edit/phase0_controllability/qwen_linearization_decomposition.json"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability"
DONE_FILE="$OUT_DIR/.PATH_A_DONE"
FAIL_FILE="$OUT_DIR/.PATH_A_STEP_FAILED.txt"

echo "============================================================"
echo "Path A re-run launcher"
echo "Qwen run dir: $QWEN_RUN_DIR"
echo "Coefficients: $COEFFS"
echo "Locator layers: $LOCATOR_LAYERS"
echo "============================================================"

# Self-relaunch into tmux
if [[ -z "${TMUX:-}" ]] && [[ "${NO_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' exists. Kill: tmux kill-session -t $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" "COEFFS='$COEFFS' LOCATOR_LAYERS='$LOCATOR_LAYERS' HF_TOKEN='${HF_TOKEN:-}' HF_HOME='${HF_HOME:-}' bash $0 2>&1 | tee $LOG"
  echo "Detached tmux. Reattach: tmux attach -t $SESSION"
  echo "Watch log: tail -f $LOG"
  echo "Watcher (another terminal): bash scripts/emnlp_perm_edit/watch_and_commit_path_a_rerun.sh"
  exit 0
fi

cd "$ROOT"
mkdir -p "$OUT_DIR"
rm -f "$DONE_FILE" "$FAIL_FILE"

declare -a STEP_NAMES STEP_RESULTS
run_step() {
  local name="$1"; shift
  echo ""
  echo "########## STEP: $name ##########"
  local t0=$(date +%s)
  if "$@"; then
    local t1=$(date +%s)
    echo "########## STEP $name OK ($((t1-t0))s) ##########"
    STEP_NAMES+=("$name"); STEP_RESULTS+=("OK")
  else
    local rc=$?
    local t1=$(date +%s)
    echo "########## STEP $name FAILED (exit $rc, $((t1-t0))s) — continuing ##########"
    echo "$name (exit $rc)" >> "$FAIL_FILE"
    STEP_NAMES+=("$name"); STEP_RESULTS+=("FAILED")
  fi
}

# --- Env setup ---
echo ""
echo "=== Step A: Env ==="
if [[ ! -d ".venv" ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
pip install -e ./vendor/circuit-tracer -q
python3 -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}')" || {
  echo "FATAL: CUDA not available." | tee -a "$FAIL_FILE"; touch "$DONE_FILE"; exit 1
}

# --- Verify prerequisites are still on disk ---
echo ""
echo "=== Step B: Prerequisites ==="
for f in "$QWEN_DIRECTIONS_RUN/layer_18.pt" "$QWEN_RUN_DIR/graph_data/000_bare_single.json.gz" "$QWEN_DECOMP"; do
  if [[ ! -f "$f" ]]; then
    echo "FATAL: prerequisite missing: $f" | tee -a "$FAIL_FILE"
    touch "$DONE_FILE"; exit 1
  fi
done

# Ensure direction_metadata.json + unnormalized per-position direction are present
QWEN_METADATA="$QWEN_RUN_DIR/01_direction/direction_metadata.json"
QWEN_UNNORM_PATH="$QWEN_RUN_DIR/01_direction/positions_L18/pos_-1_unnormalized.pt"
if [[ ! -f "$QWEN_METADATA" ]] || [[ ! -f "$QWEN_UNNORM_PATH" ]]; then
  echo "Pulling metadata + unnormalized per-position direction from temp/gemma-vs-qwen-pipeline..."
  git fetch origin temp/gemma-vs-qwen-pipeline 2>/dev/null || true
  mkdir -p "$QWEN_RUN_DIR/01_direction/positions_L18"
  QWEN_SRC_PREFIX="data/results/pipeline_runs_qwen/run_20260502_154423/01_direction"
  git show "origin/temp/gemma-vs-qwen-pipeline:$QWEN_SRC_PREFIX/direction_metadata.json" \
    > "$QWEN_METADATA" 2>/dev/null || true
  for P in $(seq -1 -1 -15); do
    git show "origin/temp/gemma-vs-qwen-pipeline:$QWEN_SRC_PREFIX/positions_L18/pos_${P}.pt" \
      > "$QWEN_RUN_DIR/01_direction/positions_L18/pos_${P}.pt" 2>/dev/null || true
    git show "origin/temp/gemma-vs-qwen-pipeline:$QWEN_SRC_PREFIX/positions_L18/pos_${P}_unnormalized.pt" \
      > "$QWEN_RUN_DIR/01_direction/positions_L18/pos_${P}_unnormalized.pt" 2>/dev/null || true
  done
fi
for f in "$QWEN_METADATA" "$QWEN_UNNORM_PATH"; do
  if [[ ! -f "$f" ]]; then
    echo "FATAL: could not fetch $f from temp/gemma-vs-qwen-pipeline" | tee -a "$FAIL_FILE"
    touch "$DONE_FILE"; exit 1
  fi
done
echo "  All prerequisites present (including metadata + unnormalized r)."

# --- STEP 1: Qwen 0b edge ablation (FIXED — uses unnormalized r) ---
run_step "qwen_0b_edge_ablation_fixed" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime_qwen.py \
      --decomposition '$QWEN_DECOMP' \
      --rhat-path '$QWEN_UNNORM_PATH' \
      --out '$OUT_DIR/qwen_edge_ablation_flip_rates_v2.json'"

# --- STEP 2: Qwen sweep all positions (FIXED) ---
run_step "qwen_direction_sweep_all_positions_fixed" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep_qwen.py \
      --directions-dir '$QWEN_DIRECTIONS_RUN' \
      --metadata-path '$QWEN_METADATA' \
      --layers 18 \
      --position-mode all \
      --coefficients '$COEFFS' \
      --out '$OUT_DIR/qwen_direction_intervention_sweep_all_v2.json'"

# --- STEP 3: Qwen sweep pos=-1 (FIXED) ---
run_step "qwen_direction_sweep_pos_neg1_fixed" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep_qwen.py \
      --directions-dir '$QWEN_DIRECTIONS_RUN' \
      --metadata-path '$QWEN_METADATA' \
      --layers 18 \
      --position-mode last_prompt_only \
      --target-position -1 \
      --coefficients '$COEFFS' \
      --out '$OUT_DIR/qwen_direction_intervention_sweep_pos1_v2.json'"

# --- STEP 4: Qwen layer locator (FIXED) ---
run_step "qwen_layer_locator_pos_neg1_fixed" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep_qwen.py \
      --directions-dir '$QWEN_DIRECTIONS_RUN' \
      --metadata-path '$QWEN_METADATA' \
      --layers '$LOCATOR_LAYERS' \
      --position-mode last_prompt_only \
      --target-position -1 \
      --coefficients '1.0' \
      --out '$OUT_DIR/qwen_layer_locator_pos1_coeff1_v2.json'"

# --- STEP 5: Gemma sign-flipped supra-threshold (FIXED) ---
run_step "gemma_suprathreshold_antirefuse" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py \
      --variants 'ablate_all_edges_antirefuse_5x,ablate_all_edges_antirefuse_10x,ablate_all_edges_antirefuse_50x,ablate_all_edges_antirefuse_100x,ablate_all_edges_antirefuse_200x' \
      --out '$OUT_DIR/gemma_suprathreshold_antirefuse.json'"

# --- Final summary ---
echo ""
echo "============================================================"
echo "PATH A RE-RUN COMPLETE"
echo "============================================================"
for i in "${!STEP_NAMES[@]}"; do
  echo "  ${STEP_NAMES[$i]} : ${STEP_RESULTS[$i]}"
done
if [[ -f "$FAIL_FILE" ]]; then
  echo ""
  echo "Failures: $(cat "$FAIL_FILE")"
fi
echo ""
echo "Result files:"
ls -la "$OUT_DIR"/qwen_*_v2.json 2>/dev/null
ls -la "$OUT_DIR"/gemma_suprathreshold_antirefuse.json 2>/dev/null

touch "$DONE_FILE"
echo "Touched $DONE_FILE"
echo ""
echo "Watcher: bash scripts/emnlp_perm_edit/watch_and_commit_path_a_rerun.sh"
