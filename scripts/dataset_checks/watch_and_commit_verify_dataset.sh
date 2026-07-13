#!/usr/bin/env bash
# Watch the v2 dataset behavioral verification and auto-commit results on completion.
# Run in a SECOND pod terminal while runpod_verify_dataset.sh runs in tmux 'verify_ds'.
#
# Usage:
#   bash scripts/dataset_checks/watch_and_commit_verify_dataset.sh

set -uo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT_DIR="$ROOT/data/results/dataset_checks"
DONE_FILE="$OUT_DIR/.VERIFY_DONE"
FAIL_FILE="$OUT_DIR/.VERIFY_STEP_FAILED.txt"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-12}"
GIT_REMOTE_BRANCH="${GIT_REMOTE_BRANCH:-$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)}"

cd "$ROOT"
echo "Watching $DONE_FILE ; poll ${POLL_INTERVAL_SEC}s ; push to origin/$GIT_REMOTE_BRANCH"

max_polls=$(( TIMEOUT_HOURS * 3600 / POLL_INTERVAL_SEC )); poll=0
while [[ ! -f "$DONE_FILE" ]]; do
  if [[ $poll -ge $max_polls ]]; then echo "TIMEOUT after ${TIMEOUT_HOURS}h."; exit 2; fi
  poll=$(( poll + 1 ))
  echo "[$(date +%H:%M:%S)] poll $poll/$max_polls — waiting ${POLL_INTERVAL_SEC}s..."
  sleep "$POLL_INTERVAL_SEC"
done

echo "[$(date +%H:%M:%S)] Done marker found."
[[ -f "$FAIL_FILE" ]] && { echo "Step failures:"; cat "$FAIL_FILE"; }

RESULT_FILES=(
  "$OUT_DIR/v2_behavioral_gemma.json"
  "$OUT_DIR/v2_behavioral_qwen.json"
  "$OUT_DIR/v2_behavioral_comparison.json"
  "$OUT_DIR/V2_BEHAVIORAL_COMPARISON.md"
)
to_add=()
for f in "${RESULT_FILES[@]}" "$FAIL_FILE"; do [[ -f "$f" ]] && to_add+=("$f"); done
[[ ${#to_add[@]} -eq 0 ]] && { echo "Nothing to commit."; exit 3; }

MSG=$(mktemp)
{
  echo "dataset v2: behavioral verification (Gemma || Qwen) compliance/refusal rates"
  echo ""
  echo "Automated commit by watch_and_commit_verify_dataset.sh."
  echo "Per-class jb-COMPLY / ctrl-REFUSE / bare-REFUSE for both models + comparison."
  [[ -f "$FAIL_FILE" ]] && { echo ""; echo "Failures:"; sed 's/^/  - /' "$FAIL_FILE"; }
  echo ""
  echo "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
} > "$MSG"

git add "${to_add[@]}"
git diff --staged --quiet && { echo "No changes."; rm "$MSG"; exit 0; }
git commit -F "$MSG"; rm "$MSG"
SHA=$(git rev-parse --short HEAD)
echo "Committed $SHA. Pushing to origin/$GIT_REMOTE_BRANCH..."
git push origin "HEAD:$GIT_REMOTE_BRANCH" && echo "PUSH OK ($SHA)" || {
  echo "PUSH FAILED — local commit $SHA. Manual: git push origin HEAD:$GIT_REMOTE_BRANCH"; exit 4; }
