#!/usr/bin/env bash
# Watch the Qwen + supra-threshold run and auto-commit results when complete.
#
# Run this in a SECOND pod terminal while the main launcher is in tmux session 'qwen_supra'.
# Polls every 2 minutes for the .QWEN_SUPRA_DONE marker, then commits + pushes
# the new result JSONs.
#
# Usage (from RunPod):
#   bash scripts/emnlp_perm_edit/watch_and_commit_qwen_and_suprathreshold.sh

set -uo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability"
DONE_FILE="$OUT_DIR/.QWEN_SUPRA_DONE"
FAIL_FILE="$OUT_DIR/.QWEN_SUPRA_STEP_FAILED.txt"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-24}"
GIT_REMOTE_BRANCH="${GIT_REMOTE_BRANCH:-emnlp-perm-edit}"

cd "$ROOT"
echo "============================================================"
echo "Qwen + supra-threshold watcher"
echo "Watching: $DONE_FILE"
echo "Poll: ${POLL_INTERVAL_SEC}s; timeout ${TIMEOUT_HOURS}h"
echo "Push to: origin/$GIT_REMOTE_BRANCH"
echo "============================================================"

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "$GIT_REMOTE_BRANCH" ]]; then
  echo "WARNING: current branch is '$current_branch', not '$GIT_REMOTE_BRANCH'."
fi

max_polls=$(( TIMEOUT_HOURS * 3600 / POLL_INTERVAL_SEC ))
poll=0
while [[ ! -f "$DONE_FILE" ]]; do
  if [[ $poll -ge $max_polls ]]; then
    echo "TIMEOUT after ${TIMEOUT_HOURS}h. Aborting watcher."
    exit 2
  fi
  poll=$(( poll + 1 ))
  ts=$(date +"%H:%M:%S")
  echo "[$ts] poll $poll/$max_polls — no marker yet, sleeping ${POLL_INTERVAL_SEC}s..."
  sleep "$POLL_INTERVAL_SEC"
done

echo ""
echo "[$(date +"%H:%M:%S")] Marker detected. Preparing commit."

if [[ -f "$FAIL_FILE" ]]; then
  echo "WARNING: some steps failed:"
  cat "$FAIL_FILE"
fi

echo "Result artifacts:"
ls -la "$OUT_DIR"/qwen_*.json 2>/dev/null
ls -la "$OUT_DIR"/gemma_suprathreshold_*.json 2>/dev/null

RESULT_FILES=(
  "$OUT_DIR/qwen_linearization_decomposition.json"
  "$OUT_DIR/qwen_edge_ablation_flip_rates.json"
  "$OUT_DIR/qwen_direction_intervention_sweep_all.json"
  "$OUT_DIR/qwen_direction_intervention_sweep_pos1.json"
  "$OUT_DIR/qwen_layer_locator_pos1_coeff1.json"
  "$OUT_DIR/gemma_suprathreshold_edge_ablation.json"
)
OPTIONAL_FILES=("$FAIL_FILE")

to_add=()
for f in "${RESULT_FILES[@]}"; do
  if [[ -f "$f" ]]; then to_add+=("$f"); else echo "  (missing: $(basename "$f"))"; fi
done
for f in "${OPTIONAL_FILES[@]}"; do
  if [[ -f "$f" ]]; then to_add+=("$f"); fi
done

if [[ ${#to_add[@]} -eq 0 ]]; then
  echo "Nothing to commit. Aborting."; exit 3
fi

COMMIT_MSG_FILE=$(mktemp)
{
  echo "emnlp phase 0: Qwen3-4B replication + Gemma supra-threshold edge ablation"
  echo ""
  echo "Automated commit by watch_and_commit_qwen_and_suprathreshold.sh."
  echo ""
  echo "Two parallel additions for the EMNLP submission:"
  echo "  A) Qwen3-4B replication of the magnitude-gap finding"
  echo "     (direction sweep + layer locator on Ruqiya's Qwen direction files)"
  echo "  B) Gemma supra-threshold edge ablation"
  echo "     (scale edge-derived deltas by {5x, 10x, 50x, 100x, 200x})"
  echo "     Confirms magnitude is the only missing variable separating edge"
  echo "     ablation from direction intervention."
  echo ""
  if [[ -f "$FAIL_FILE" ]]; then
    echo "Step failures:"
    sed 's/^/  - /' "$FAIL_FILE"; echo ""
  else
    echo "All steps OK."
    echo ""
  fi
  echo "Artifacts:"
  for f in "${to_add[@]}"; do
    size=$(du -h "$f" 2>/dev/null | cut -f1)
    echo "  - $(realpath --relative-to="$ROOT" "$f") ($size)"
  done
  echo ""
  echo "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
} > "$COMMIT_MSG_FILE"

git add "${to_add[@]}"
if git diff --staged --quiet; then
  echo "Nothing to commit (no changes). Already committed?"
  rm "$COMMIT_MSG_FILE"; exit 0
fi
git commit -F "$COMMIT_MSG_FILE"
COMMIT_SHA=$(git rev-parse --short HEAD)
rm "$COMMIT_MSG_FILE"
echo "Local commit: $COMMIT_SHA"

echo "Pushing to origin/$GIT_REMOTE_BRANCH..."
if git push origin "HEAD:$GIT_REMOTE_BRANCH"; then
  echo "[$(date +"%H:%M:%S")] PUSH OK. Commit $COMMIT_SHA on origin/$GIT_REMOTE_BRANCH."
else
  echo "[$(date +"%H:%M:%S")] PUSH FAILED. Commit $COMMIT_SHA is local-only."
  echo "To push manually: git push origin HEAD:$GIT_REMOTE_BRANCH"
  exit 4
fi

echo ""
echo "============================================================"
echo "watch_and_commit DONE."
echo "============================================================"
