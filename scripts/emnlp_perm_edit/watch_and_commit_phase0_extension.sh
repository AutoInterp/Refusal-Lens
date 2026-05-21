#!/usr/bin/env bash
# Watches the Phase 0 extension run for completion, then commits + pushes
# results to GitHub.
#
# Run this in a SEPARATE terminal from the main launcher. Polls every 2 minutes
# for the .PHASE0_EXT_DONE marker file produced at the end of
# runpod_phase0_extension.sh, then runs git add + commit + push for the new
# result artifacts.
#
# Usage (from RunPod, after `runpod_phase0_extension.sh` is running in tmux
# session 'phase0_ext'):
#   bash scripts/emnlp_perm_edit/watch_and_commit_phase0_extension.sh
#
# Environment variables:
#   GIT_REMOTE_BRANCH   : branch to push to (default: emnlp-perm-edit)
#   POLL_INTERVAL_SEC   : seconds between polls (default: 120)
#   TIMEOUT_HOURS       : max hours to wait before giving up (default: 24)
#
# Requires: gh credentials OR pre-configured remote with PAT/SSH. If git push
# fails (auth issue), the script prints instructions and exits with a non-zero
# code without committing-and-failing — the local commit is still on disk.

set -uo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability"
DONE_FILE="$OUT_DIR/.PHASE0_EXT_DONE"
FAIL_FILE="$OUT_DIR/.PHASE0_EXT_STEP_FAILED.txt"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-24}"
GIT_REMOTE_BRANCH="${GIT_REMOTE_BRANCH:-emnlp-perm-edit}"

cd "$ROOT"
echo "============================================================"
echo "Phase 0 extension watcher + auto-commit"
echo "Repo root: $ROOT"
echo "Watching: $DONE_FILE"
echo "Poll: every ${POLL_INTERVAL_SEC}s; timeout ${TIMEOUT_HOURS}h"
echo "Will push to: origin/$GIT_REMOTE_BRANCH"
echo "============================================================"

# Sanity: are we on the right branch?
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "$GIT_REMOTE_BRANCH" ]]; then
  echo "WARNING: current branch is '$current_branch', not '$GIT_REMOTE_BRANCH'."
  echo "  Will commit to '$current_branch' but push to origin/$GIT_REMOTE_BRANCH."
fi

# Wait for the marker file, with timeout
max_polls=$(( TIMEOUT_HOURS * 3600 / POLL_INTERVAL_SEC ))
poll=0
while [[ ! -f "$DONE_FILE" ]]; do
  if [[ $poll -ge $max_polls ]]; then
    echo "TIMEOUT after ${TIMEOUT_HOURS}h. Aborting watcher; the main run may still be in progress."
    exit 2
  fi
  poll=$(( poll + 1 ))
  ts=$(date +"%H:%M:%S")
  echo "[$ts] poll $poll/$max_polls — no marker yet, sleeping ${POLL_INTERVAL_SEC}s..."
  sleep "$POLL_INTERVAL_SEC"
done

echo ""
echo "[$(date +"%H:%M:%S")] Marker $DONE_FILE detected. Preparing commit."

# Summarize step status
if [[ -f "$FAIL_FILE" ]]; then
  echo ""
  echo "WARNING: some steps failed:"
  cat "$FAIL_FILE"
  echo ""
fi

# List what's available
echo "Result artifacts present:"
ls -la "$OUT_DIR"/direction_intervention_sweep_*.json 2>/dev/null
ls -la "$OUT_DIR"/edge_ablation_pos2_flip_rates.json 2>/dev/null
ls -la "$OUT_DIR"/layer_locator_pos2_coeff1.json 2>/dev/null
echo ""

# Build the list of result files to commit (explicit; avoid -A for safety)
RESULT_FILES=(
  "$OUT_DIR/direction_intervention_sweep_all.json"
  "$OUT_DIR/direction_intervention_sweep_pos2.json"
  "$OUT_DIR/edge_ablation_pos2_flip_rates.json"
  "$OUT_DIR/layer_locator_pos2_coeff1.json"
)
OPTIONAL_FILES=(
  "$FAIL_FILE"
)

to_add=()
for f in "${RESULT_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    to_add+=("$f")
  else
    echo "  (missing: $(basename "$f") — skipping in commit)"
  fi
done
for f in "${OPTIONAL_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    to_add+=("$f")
  fi
done

if [[ ${#to_add[@]} -eq 0 ]]; then
  echo "No result files to commit. Aborting watcher."
  exit 3
fi

# Compose commit message
COMMIT_MSG_FILE=$(mktemp)
{
  echo "emnlp phase 0 extension: direction-sweep 2x2 + coeff sweep + layer locator"
  echo ""
  echo "Automated commit by watch_and_commit_phase0_extension.sh on completion of RunPod run."
  echo ""
  echo "Disambiguates magnitude vs position confound in the edge-ablation results"
  echo "by sweeping coefficient * r_hat at L15 under two position-modes:"
  echo "  - all positions, every forward step (Arditi)"
  echo "  - last_prompt_only (seq pos=-2 of prompt encoding pass)"
  echo "Plus edge ablation re-run at pos=-2 only (Cell D of the 2x2),"
  echo "plus a layer locator (coeff=1.0 pos=-2 across 8 non-L15 layers) as a"
  echo "pre-emptive depth profile."
  echo ""
  if [[ -f "$FAIL_FILE" ]]; then
    echo "Step failures (continued past, per script policy):"
    sed 's/^/  - /' "$FAIL_FILE"
    echo ""
  else
    echo "All steps completed successfully."
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

echo "Staging:"
git add "${to_add[@]}"

# Anything to commit?
if git diff --staged --quiet; then
  echo "Nothing to commit (no changes vs HEAD). The results may have already been committed."
  rm "$COMMIT_MSG_FILE"
  exit 0
fi

git commit -F "$COMMIT_MSG_FILE"
COMMIT_SHA=$(git rev-parse --short HEAD)
echo ""
echo "Local commit: $COMMIT_SHA"
rm "$COMMIT_MSG_FILE"

# Push
echo ""
echo "Pushing to origin/$GIT_REMOTE_BRANCH..."
if git push origin "HEAD:$GIT_REMOTE_BRANCH"; then
  echo "[$(date +"%H:%M:%S")] PUSH OK. Commit $COMMIT_SHA is on origin/$GIT_REMOTE_BRANCH."
else
  echo "[$(date +"%H:%M:%S")] PUSH FAILED. The commit $COMMIT_SHA is local-only."
  echo "Likely cause: GitHub auth not configured on this pod."
  echo ""
  echo "To push manually after configuring auth, run:"
  echo "  git push origin HEAD:$GIT_REMOTE_BRANCH"
  exit 4
fi

echo ""
echo "============================================================"
echo "Phase 0 extension watch_and_commit DONE."
echo "============================================================"
