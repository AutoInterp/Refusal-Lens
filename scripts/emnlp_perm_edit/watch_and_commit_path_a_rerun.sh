#!/usr/bin/env bash
# Watch the Path A re-run and auto-commit + push results.
#
# Run in a SECOND pod terminal while runpod_path_a_rerun.sh is in tmux session 'path_a_rerun'.

set -uo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability"
DONE_FILE="$OUT_DIR/.PATH_A_DONE"
FAIL_FILE="$OUT_DIR/.PATH_A_STEP_FAILED.txt"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-24}"
GIT_REMOTE_BRANCH="${GIT_REMOTE_BRANCH:-emnlp-perm-edit}"

cd "$ROOT"
echo "Watching $DONE_FILE  (poll ${POLL_INTERVAL_SEC}s; timeout ${TIMEOUT_HOURS}h)"

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "$GIT_REMOTE_BRANCH" ]]; then
  echo "WARNING: branch '$current_branch' != '$GIT_REMOTE_BRANCH'"
fi

max_polls=$(( TIMEOUT_HOURS * 3600 / POLL_INTERVAL_SEC ))
poll=0
while [[ ! -f "$DONE_FILE" ]]; do
  if [[ $poll -ge $max_polls ]]; then echo "TIMEOUT"; exit 2; fi
  poll=$(( poll + 1 ))
  echo "[$(date +%H:%M:%S)] poll $poll/$max_polls — sleeping ${POLL_INTERVAL_SEC}s"
  sleep "$POLL_INTERVAL_SEC"
done

echo ""
echo "[$(date +%H:%M:%S)] Marker detected."
if [[ -f "$FAIL_FILE" ]]; then echo "Failures:"; cat "$FAIL_FILE"; fi

RESULT_FILES=(
  "$OUT_DIR/qwen_edge_ablation_flip_rates_v2.json"
  "$OUT_DIR/qwen_direction_intervention_sweep_all_v2.json"
  "$OUT_DIR/qwen_direction_intervention_sweep_pos1_v2.json"
  "$OUT_DIR/qwen_layer_locator_pos1_coeff1_v2.json"
  "$OUT_DIR/gemma_suprathreshold_antirefuse.json"
)
OPTIONAL_FILES=("$FAIL_FILE")

to_add=()
for f in "${RESULT_FILES[@]}"; do
  if [[ -f "$f" ]]; then to_add+=("$f"); else echo "  (missing: $(basename "$f"))"; fi
done
for f in "${OPTIONAL_FILES[@]}"; do
  if [[ -f "$f" ]]; then to_add+=("$f"); fi
done
if [[ ${#to_add[@]} -eq 0 ]]; then echo "Nothing to commit."; exit 3; fi

COMMIT_MSG_FILE=$(mktemp)
{
  echo "emnlp phase 0 Path A re-run: Qwen thinking-mode + Gemma supra sign fixes"
  echo ""
  echo "Re-runs the 4 Qwen behavioral experiments with enable_thinking=False applied"
  echo "via the patched scripts/pipeline/utils.py format_prompt, plus the 5 Gemma"
  echo "supra-threshold variants with sign-flipped scale (-N) to test anti-refuse"
  echo "direction scaling instead of pro-refuse."
  echo ""
  if [[ -f "$FAIL_FILE" ]]; then
    echo "Step failures:"
    sed 's/^/  - /' "$FAIL_FILE"; echo ""
  else
    echo "All steps OK."; echo ""
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
if git diff --staged --quiet; then echo "Nothing to commit."; rm "$COMMIT_MSG_FILE"; exit 0; fi
git commit -F "$COMMIT_MSG_FILE"
COMMIT_SHA=$(git rev-parse --short HEAD)
rm "$COMMIT_MSG_FILE"
echo "Local commit: $COMMIT_SHA"

if git push origin "HEAD:$GIT_REMOTE_BRANCH"; then
  echo "[$(date +%H:%M:%S)] PUSH OK: $COMMIT_SHA on origin/$GIT_REMOTE_BRANCH"
else
  echo "[$(date +%H:%M:%S)] PUSH FAILED. Commit $COMMIT_SHA is local-only."
  exit 4
fi

echo "DONE."
