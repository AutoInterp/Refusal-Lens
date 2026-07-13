#!/usr/bin/env bash
# RunPod launcher: behavioral verification of the v2 controlled jailbreak dataset
# against BOTH Gemma-3-4B-IT and Qwen3-4B, run in parallel, with a cross-model
# comparison report of compliance/refusal rates per class.
#
# Design intent verified:
#   bare (base alone)        -> REFUSE
#   ctrl (neutral prefix)    -> REFUSE
#   jb   (jailbreak prefix)  -> COMPLY   <- the key thing we're checking
#
# Parallelism:
#   - >=2 GPUs: Gemma on GPU0, Qwen on GPU1 (true parallel, ~half wall).
#   - 1 GPU:    both share GPU0 as concurrent processes (both 4B models fit in
#               an 80GB H100; wall ~= sum, but both progress + one log to watch).
#
# Estimated wall (H100): ~1050 gens/model. ~3-5 s/gen => ~1-1.5h/model.
#   2 GPUs: ~1.5h total. 1 GPU shared: ~3h total. Smoke (--max-prompts 3): ~2 min.
#
# Usage (from RunPod /workspace, on a branch that contains scripts/dataset_checks/):
#   git clone --recurse-submodules <fork> && cd Refusal-Lens
#   git checkout <branch-with-these-scripts> && git pull
#   export HF_TOKEN="hf_..."; export HF_HOME=/workspace/.hf_cache
#   bash scripts/dataset_checks/runpod_verify_dataset.sh           # full
#   SMOKE=1 bash scripts/dataset_checks/runpod_verify_dataset.sh   # 3-prompt smoke
#
# Completion signals:
#   data/results/dataset_checks/.VERIFY_DONE
#   data/results/dataset_checks/.VERIFY_STEP_FAILED.txt

set -uo pipefail

SESSION="verify_ds"
LOG="/tmp/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
ROOT="${ROOT:-$(pwd)}"
DATASET_BRANCH="${DATASET_BRANCH:-tejas/dataset-10-classes}"
DATASET_PATH="${DATASET_PATH:-dataset/refusal_lens_controlled_dataset_v2.json}"
GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-3-4b-it}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-4B}"
MAX_PROMPTS_ARG=""
[[ "${SMOKE:-0}" == "1" ]] && MAX_PROMPTS_ARG="--max-prompts 3"

OUT_DIR="$ROOT/data/results/dataset_checks"
GEMMA_OUT="$OUT_DIR/v2_behavioral_gemma.json"
QWEN_OUT="$OUT_DIR/v2_behavioral_qwen.json"
CMP_JSON="$OUT_DIR/v2_behavioral_comparison.json"
CMP_MD="$OUT_DIR/V2_BEHAVIORAL_COMPARISON.md"
DONE_FILE="$OUT_DIR/.VERIFY_DONE"
FAIL_FILE="$OUT_DIR/.VERIFY_STEP_FAILED.txt"

echo "============================================================"
echo "v2 dataset behavioral verification (Gemma || Qwen)"
echo "Repo root: $ROOT"
echo "Dataset:   $DATASET_PATH (from $DATASET_BRANCH)"
echo "Models:    $GEMMA_MODEL  ||  $QWEN_MODEL"
echo "Smoke:     ${SMOKE:-0}"
echo "Log:       $LOG"
echo "============================================================"

# --- Self-relaunch into tmux ---
if [[ -z "${TMUX:-}" ]] && [[ "${NO_TMUX:-0}" != "1" ]]; then
  echo "Relaunching into tmux session '$SESSION' (NO_TMUX=1 to disable)..."
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' exists. Kill it: tmux kill-session -t $SESSION"; exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "SMOKE='${SMOKE:-0}' DATASET_BRANCH='$DATASET_BRANCH' HF_TOKEN='${HF_TOKEN:-}' HF_HOME='${HF_HOME:-}' bash $0 2>&1 | tee $LOG"
  echo "Detached. Reattach: tmux attach -t $SESSION ; Watch: tail -f $LOG"
  echo "Watcher (2nd terminal): bash scripts/dataset_checks/watch_and_commit_verify_dataset.sh"
  exit 0
fi

cd "$ROOT"
mkdir -p "$OUT_DIR"
rm -f "$DONE_FILE" "$FAIL_FILE"

# --- Env setup ---
echo ""; echo "=== Step A: Python environment ==="
if [[ ! -d ".venv" ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install --upgrade pip -q
# 'steering' extra = torch + transformers + accelerate + safetensors + hf_hub
# (everything generation needs; no circuit-tracer — this job doesn't use graphs).
pip install -e ".[steering]" -q
python3 -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: torch {torch.__version__} {torch.cuda.get_device_name(0)}')" || {
  echo "FATAL: CUDA torch not available." | tee -a "$FAIL_FILE"; touch "$DONE_FILE"; exit 1
}

# --- Fetch the v2 dataset file from Tejas's branch (not on this branch) ---
echo ""; echo "=== Step B: Fetch v2 dataset from $DATASET_BRANCH ==="
if [[ ! -f "$DATASET_PATH" ]]; then
  git fetch origin "$DATASET_BRANCH" 2>/dev/null || true
  git checkout "origin/$DATASET_BRANCH" -- "$DATASET_PATH" 2>/dev/null || {
    echo "FATAL: could not fetch $DATASET_PATH from origin/$DATASET_BRANCH" | tee -a "$FAIL_FILE"
    touch "$DONE_FILE"; exit 1; }
fi
echo "  dataset present: $(wc -c < "$DATASET_PATH") bytes"

# --- GPU pinning ---
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo ""; echo "=== Step C: launch both models (GPUs visible: $N_GPU) ==="
if [[ "$N_GPU" -ge 2 ]]; then
  GEMMA_GPU=0; QWEN_GPU=1; echo "  >=2 GPUs: Gemma->GPU0, Qwen->GPU1 (true parallel)"
else
  GEMMA_GPU=0; QWEN_GPU=0; echo "  1 GPU: both share GPU0 (concurrent)"
fi

GEMMA_LOG="/tmp/${SESSION}_gemma.log"; QWEN_LOG="/tmp/${SESSION}_qwen.log"

CUDA_VISIBLE_DEVICES=$GEMMA_GPU python3 scripts/dataset_checks/verify_dataset_behavioral.py \
    --dataset "$DATASET_PATH" --model "$GEMMA_MODEL" --thinking-mode default \
    --out "$GEMMA_OUT" $MAX_PROMPTS_ARG > "$GEMMA_LOG" 2>&1 &
PID_G=$!
echo "  Gemma PID $PID_G (log $GEMMA_LOG)"

CUDA_VISIBLE_DEVICES=$QWEN_GPU python3 scripts/dataset_checks/verify_dataset_behavioral.py \
    --dataset "$DATASET_PATH" --model "$QWEN_MODEL" --thinking-mode off \
    --out "$QWEN_OUT" $MAX_PROMPTS_ARG > "$QWEN_LOG" 2>&1 &
PID_Q=$!
echo "  Qwen  PID $PID_Q (log $QWEN_LOG)"

wait $PID_G; RC_G=$?
wait $PID_Q; RC_Q=$?
echo "  Gemma exit $RC_G ; Qwen exit $RC_Q"
echo "  --- tail Gemma ---"; tail -n 25 "$GEMMA_LOG"
echo "  --- tail Qwen  ---"; tail -n 25 "$QWEN_LOG"
[[ "$RC_G" -ne 0 ]] && echo "gemma_verify (exit $RC_G)" >> "$FAIL_FILE"
[[ "$RC_Q" -ne 0 ]] && echo "qwen_verify (exit $RC_Q)"  >> "$FAIL_FILE"

# --- Cross-model comparison report ---
echo ""; echo "=== Step D: cross-model comparison ==="
CMP_ARGS=""
[[ -f "$GEMMA_OUT" ]] && CMP_ARGS="$CMP_ARGS --gemma $GEMMA_OUT"
[[ -f "$QWEN_OUT"  ]] && CMP_ARGS="$CMP_ARGS --qwen $QWEN_OUT"
python3 scripts/dataset_checks/compare_behavioral.py $CMP_ARGS \
    --out-json "$CMP_JSON" --out-md "$CMP_MD" || echo "compare (failed)" >> "$FAIL_FILE"

echo ""; echo "============================================================"
echo "VERIFICATION COMPLETE"
[[ -f "$FAIL_FILE" ]] && { echo "Failures:"; cat "$FAIL_FILE"; } || echo "All steps OK."
echo "Report: $CMP_MD"
echo "============================================================"
touch "$DONE_FILE"
echo "Touched $DONE_FILE. Auto-commit: bash scripts/dataset_checks/watch_and_commit_verify_dataset.sh"
