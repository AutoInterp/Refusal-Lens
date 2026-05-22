#!/usr/bin/env bash
# RunPod launcher for Qwen3-4B full Phase 0 replication + Gemma supra-threshold.
#
# Two parallel additions to the EMNLP submission:
#
# A) Qwen3-4B full Phase 0 replication at L18 (Ruqiya's causal layer per
#    QWEN_FIXES_SUMMARY.md: "attribution must target the layer where intervention
#    actually works, not the peak-separation layer"). Existing HF graphs target
#    L34, so we generate fresh L18-target graphs first.
#
#    STEP 1 — Stage 02 attribution graph generation, 550 single-mode at L18
#             Uses Ruqiya's pipeline_qwen scripts (config: Qwen/Qwen3-4B,
#             transcoder mwhanna/qwen3-4b-transcoders, target L18 pos=-1)
#    STEP 2 — Stage 02c pack graphs into .json.gz
#    STEP 3 — Qwen 0a linearization decomposition (CPU)
#    STEP 4 — Qwen 0b edge ablation @ L18, 7 variants
#    STEP 5 — Qwen direction sweep @ L18, all positions, 8 coefficients
#    STEP 6 — Qwen direction sweep @ L18, pos=-1 only, 8 coefficients
#    STEP 7 — Qwen layer locator @ pos=-1 coeff=1.0, 8 non-L18 layers
#             {0, 5, 10, 15, 22, 28, 32, 34}  (includes L34 = peak separation
#             for transparency / comparison)
#
# B) Gemma supra-threshold edge ablation (confirms magnitude is the only missing
#    variable separating edge ablation from direction intervention).
#
#    STEP 8 — Gemma supra-threshold edge ablation @ all positions, 5 scaled
#             variants {5x, 10x, 50x, 100x, 200x}
#
# Estimated wall on H100 SXM:
#   Stage 02 (~1.5 min/graph × 550): ~14 h (biggest single step)
#   Stage 02c packing: ~10 min
#   Qwen 0a: ~15 min CPU
#   Qwen 0b (7 variants × 550): ~3 h
#   Qwen sweep all positions: ~4 h
#   Qwen sweep pos=-1: ~4 h
#   Qwen layer locator (8 layers): ~4 h
#   Gemma supra-threshold (5 variants): ~2.5 h
#   Total: ~32 h, ~$110 at $3.50/hr
#
# Continue-on-failure: per-step failures recorded; remaining steps execute.
#
# Usage (from RunPod terminal at /workspace):
#   git clone --recurse-submodules https://github.com/<your-fork>/Refusal-Lens.git
#   cd Refusal-Lens && git checkout emnlp-perm-edit && git pull
#   export HF_TOKEN="hf_..."
#   export HF_HOME=/workspace/.hf_cache
#   bash scripts/emnlp_perm_edit/runpod_qwen_and_suprathreshold.sh
#
# Completion signals:
#   data/results/emnlp_perm_edit/phase0_controllability/.QWEN_SUPRA_DONE
#   data/results/emnlp_perm_edit/phase0_controllability/.QWEN_SUPRA_STEP_FAILED.txt

set -uo pipefail

SESSION="qwen_supra"
LOG="/tmp/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
ROOT="${ROOT:-$(pwd)}"
COEFFS="${COEFFS:-0.001,0.005,0.01,0.05,0.1,0.25,0.5,1.0}"
LOCATOR_LAYERS="${LOCATOR_LAYERS:-0,5,10,15,22,28,32,34}"
QWEN_RUN_NAME="${QWEN_RUN_NAME:-run_emnlp_qwen_L18_$(date +%Y%m%d)}"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability"
DONE_FILE="$OUT_DIR/.QWEN_SUPRA_DONE"
FAIL_FILE="$OUT_DIR/.QWEN_SUPRA_STEP_FAILED.txt"

QWEN_RUN_DIR="$ROOT/data/results/pipeline_runs_qwen/$QWEN_RUN_NAME"
QWEN_DIRECTIONS_SRC="$ROOT/data/results/pipeline_runs_qwen/run_20260502_154423/01_direction/directions"
QWEN_DIRECTIONS_RUN="$QWEN_RUN_DIR/01_direction/directions"

echo "============================================================"
echo "Qwen Phase 0 + Gemma supra-threshold launcher"
echo "Repo root: $ROOT"
echo "Qwen run name: $QWEN_RUN_NAME"
echo "Qwen run dir: $QWEN_RUN_DIR"
echo "Coefficients: $COEFFS"
echo "Locator layers: $LOCATOR_LAYERS"
echo "Log: $LOG"
echo "============================================================"

# Self-relaunch into tmux
if [[ -z "${TMUX:-}" ]] && [[ "${NO_TMUX:-0}" != "1" ]]; then
  echo "Relaunching into tmux session '$SESSION' (NO_TMUX=1 to disable)..."
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists."
    echo "  Kill it first:  tmux kill-session -t $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" "COEFFS='$COEFFS' LOCATOR_LAYERS='$LOCATOR_LAYERS' QWEN_RUN_NAME='$QWEN_RUN_NAME' HF_TOKEN='${HF_TOKEN:-}' HF_HOME='${HF_HOME:-}' bash $0 2>&1 | tee $LOG"
  echo "Detached tmux session. Reattach: tmux attach -t $SESSION"
  echo "Watch log: tail -f $LOG"
  echo ""
  echo "Watcher (in another terminal):"
  echo "  bash scripts/emnlp_perm_edit/watch_and_commit_qwen_and_suprathreshold.sh"
  exit 0
fi

cd "$ROOT"
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
    STEP_NAMES+=("$name"); STEP_RESULTS+=("OK")
  else
    local rc=$?
    local t1=$(date +%s)
    echo "########## STEP $name FAILED (exit $rc, $((t1-t0))s wall) — continuing ##########"
    echo "$name (exit $rc)" >> "$FAIL_FILE"
    STEP_NAMES+=("$name"); STEP_RESULTS+=("FAILED")
  fi
}

# --- Env setup ---
echo ""
echo "=== Step A: Python environment ==="
if [[ ! -d ".venv" ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
pip install -e ./vendor/circuit-tracer -q
python3 -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: torch {torch.__version__} {torch.cuda.get_device_name(0)}')" || {
  echo "FATAL: CUDA torch not available." | tee -a "$FAIL_FILE"; touch "$DONE_FILE"; exit 1
}

# --- Ensure scripts/pipeline_qwen/ is present (commit may or may not include it) ---
if [[ ! -f "scripts/pipeline_qwen/02_run_attribution.py" ]]; then
  echo ""
  echo "=== Step A2: Fetching pipeline_qwen scripts from temp/gemma-vs-qwen-pipeline ==="
  git fetch origin temp/gemma-vs-qwen-pipeline
  git checkout origin/temp/gemma-vs-qwen-pipeline -- scripts/pipeline_qwen/
  echo "  pulled $(ls scripts/pipeline_qwen/ | wc -l) files into scripts/pipeline_qwen/"
fi

# --- Pull Qwen direction files (Ruqiya's r_hat per layer) ---
echo ""
echo "=== Step B: Set up Qwen direction files ==="
if [[ ! -f "$QWEN_DIRECTIONS_SRC/layer_18.pt" ]]; then
  echo "Fetching Qwen direction files from temp/gemma-vs-qwen-pipeline (~1.5 MB)..."
  git fetch origin temp/gemma-vs-qwen-pipeline 2>/dev/null || true
  mkdir -p "$QWEN_DIRECTIONS_SRC"
  QWEN_SRC_PREFIX="data/results/pipeline_runs_qwen/run_20260502_154423/01_direction/directions"
  for L in $(seq -f "%02g" 0 35); do
    git show origin/temp/gemma-vs-qwen-pipeline:"$QWEN_SRC_PREFIX/layer_${L}.pt" > "$QWEN_DIRECTIONS_SRC/layer_${L}.pt" 2>/dev/null || \
      echo "  warning: layer_${L}.pt not in branch"
  done
  git show origin/temp/gemma-vs-qwen-pipeline:data/results/pipeline_runs_qwen/run_20260502_154423/01_direction/direction_metadata.json > "$QWEN_DIRECTIONS_SRC/../direction_metadata.json" 2>/dev/null || true
fi
# Copy into the new run dir so Stage 02 finds them at the expected path
mkdir -p "$QWEN_DIRECTIONS_RUN"
mkdir -p "$QWEN_RUN_DIR/01_direction"
cp -n "$QWEN_DIRECTIONS_SRC"/*.pt "$QWEN_DIRECTIONS_RUN/" 2>/dev/null || true
cp -n "$QWEN_DIRECTIONS_SRC/../direction_metadata.json" "$QWEN_RUN_DIR/01_direction/" 2>/dev/null || true
echo "  directions in place: $(ls $QWEN_DIRECTIONS_RUN/*.pt 2>/dev/null | wc -l) layer files"

# --- STEP 1: Stage 02 attribution graph generation (L18, single-mode only) ---
# Biggest time step (~14 h). This generates 550 .pt files in
# $QWEN_RUN_DIR/02_attribution/graphs/
run_step "qwen_stage02_attribution_generation" \
  bash -c "PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/02_run_attribution.py \
      --run-dir '$QWEN_RUN_DIR' \
      --target-layer 18 \
      --skip-multi-graph \
      --n-prompts 50 \
      --save-graphs"

# --- STEP 2: Stage 02c packing into .json.gz ---
run_step "qwen_stage02c_pack_graphs" \
  bash -c "PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/02c_pack_graphs.py \
      --run-dir '$QWEN_RUN_DIR'"

# --- STEP 3: Qwen 0a linearization decomposition ---
# Reuses existing scripts/emnlp_perm_edit/00_linearization_decomposition.py since
# it just sums edge attributions from the packed graphs.
QWEN_GRAPH_DIR="$QWEN_RUN_DIR/graph_data"
QWEN_DECOMP_OUT="$OUT_DIR/qwen_linearization_decomposition.json"
run_step "qwen_0a_linearization" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_linearization_decomposition.py \
      --graph-data-dir '$QWEN_GRAPH_DIR' \
      --out-dir '$OUT_DIR' \
      --n-prompts 50 \
      --mode single && mv '$OUT_DIR/linearization_decomposition.json' '$QWEN_DECOMP_OUT'"

# --- STEP 4: Qwen 0b edge ablation at L18 ---
run_step "qwen_0b_edge_ablation" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime_qwen.py \
      --decomposition '$QWEN_DECOMP_OUT' \
      --rhat-path '$QWEN_DIRECTIONS_RUN/layer_18.pt' \
      --out '$OUT_DIR/qwen_edge_ablation_flip_rates.json'"

# --- STEP 5: Qwen direction sweep @ L18, all positions ---
run_step "qwen_direction_sweep_all_positions" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep_qwen.py \
      --directions-dir '$QWEN_DIRECTIONS_RUN' \
      --layers 18 \
      --position-mode all \
      --coefficients '$COEFFS' \
      --out '$OUT_DIR/qwen_direction_intervention_sweep_all.json'"

# --- STEP 6: Qwen direction sweep @ L18, pos=-1 only ---
run_step "qwen_direction_sweep_pos_neg1" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep_qwen.py \
      --directions-dir '$QWEN_DIRECTIONS_RUN' \
      --layers 18 \
      --position-mode last_prompt_only \
      --target-position -1 \
      --coefficients '$COEFFS' \
      --out '$OUT_DIR/qwen_direction_intervention_sweep_pos1.json'"

# --- STEP 7: Qwen layer locator @ pos=-1, coeff=1.0 ---
run_step "qwen_layer_locator_pos_neg1" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep_qwen.py \
      --directions-dir '$QWEN_DIRECTIONS_RUN' \
      --layers '$LOCATOR_LAYERS' \
      --position-mode last_prompt_only \
      --target-position -1 \
      --coefficients '1.0' \
      --out '$OUT_DIR/qwen_layer_locator_pos1_coeff1.json'"

# --- STEP 8: Gemma supra-threshold edge ablation ---
run_step "gemma_suprathreshold_edge_ablation" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py \
      --variants 'ablate_all_edges_5x,ablate_all_edges_10x,ablate_all_edges_50x,ablate_all_edges_100x,ablate_all_edges_200x' \
      --out '$OUT_DIR/gemma_suprathreshold_edge_ablation.json'"

# --- Final summary ---
echo ""
echo "============================================================"
echo "QWEN + SUPRA-THRESHOLD COMPLETE"
echo "============================================================"
echo "Per-step status:"
for i in "${!STEP_NAMES[@]}"; do
  echo "  ${STEP_NAMES[$i]} : ${STEP_RESULTS[$i]}"
done
if [[ -f "$FAIL_FILE" ]]; then
  echo ""
  echo "Failures: $(cat "$FAIL_FILE")"
else
  echo ""
  echo "All steps OK."
fi
echo ""
echo "Result files:"
ls -la "$OUT_DIR"/qwen_*.json 2>/dev/null
ls -la "$OUT_DIR"/gemma_suprathreshold_*.json 2>/dev/null
ls -la "$QWEN_RUN_DIR"/graph_data/*.json.gz 2>/dev/null | wc -l
echo "  (above: count of packed Qwen graphs in $QWEN_RUN_DIR/graph_data/)"

touch "$DONE_FILE"
echo "Touched $DONE_FILE"
echo ""
echo "Auto-commit + push: bash scripts/emnlp_perm_edit/watch_and_commit_qwen_and_suprathreshold.sh"
