#!/usr/bin/env bash
# Watches the Phase 0 GPU run for completion, then commits + pushes results to GitHub.
#
# Run this in a SEPARATE terminal from the main launcher. Polls every 2 minutes
# for the .PHASE0_DONE marker file produced at the end of runpod_phase0_all.sh,
# then runs git add + commit + push for the result artifacts.
#
# Usage (from RunPod, after `runpod_phase0_all.sh` is running in tmux session 'phase0_all'):
#   bash scripts/emnlp_perm_edit/watch_and_commit_phase0.sh
#
# Environment variables:
#   GIT_REMOTE_BRANCH   : branch to push to (default: emnlp-perm-edit)
#   COMMIT_AUTHOR_EMAIL : git author email override (uses repo config if unset)
#   COMMIT_AUTHOR_NAME  : git author name override (uses repo config if unset)
#   POLL_INTERVAL_SEC   : seconds between polls (default: 120)
#   TIMEOUT_HOURS       : max hours to wait before giving up (default: 24)
#
# Requires: gh credentials OR pre-configured remote with PAT/SSH. If git push
# fails (auth issue), the script prints instructions and exits with a non-zero
# code without committing-and-failing — the local commit is still on disk.

set -uo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability"
DONE_FILE="$OUT_DIR/.PHASE0_DONE"
FAIL_FILE="$OUT_DIR/.PHASE0_STEP_FAILED.txt"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-120}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-24}"
GIT_REMOTE_BRANCH="${GIT_REMOTE_BRANCH:-emnlp-perm-edit}"

cd "$ROOT"
echo "============================================================"
echo "Phase 0 watcher + auto-commit"
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
ls -la "$OUT_DIR"/ 2>/dev/null | grep -v "^total" | grep -v "^d"
echo ""

# Optionally configure git author for this session
if [[ -n "${COMMIT_AUTHOR_EMAIL:-}" ]]; then
  git -c user.email="$COMMIT_AUTHOR_EMAIL" config user.email "$COMMIT_AUTHOR_EMAIL"
fi
if [[ -n "${COMMIT_AUTHOR_NAME:-}" ]]; then
  git -c user.name="$COMMIT_AUTHOR_NAME" config user.name "$COMMIT_AUTHOR_NAME"
fi

# Build the list of result files to commit (explicit; avoid -A to keep safety)
RESULT_FILES=(
  "$OUT_DIR/directdot_drift_audit.json"
  "$OUT_DIR/edge_ablation_flip_rates.json"
  "$OUT_DIR/topk_feature_sweep.json"
  "$OUT_DIR/topk_edge_sweep.json"
  "$OUT_DIR/flip_rate_summary.json"
  "$OUT_DIR/controllability_audit_figure.png"
  "$OUT_DIR/topk_feature_pareto_figure.png"
  "$OUT_DIR/topk_edge_vs_node_figure.png"
  "$OUT_DIR/PHASE0_GPU_SUMMARY.md"
)
# Optional / produced by individual steps; include if present
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

# Compose commit message from step status
COMMIT_MSG_FILE=$(mktemp)
{
  echo "emnlp phase 0 GPU run results: 0b + 0d + 0e"
  echo ""
  echo "Automated commit by watch_and_commit_phase0.sh on completion of RunPod run."
  echo "K_MODE=${K_MODE:-unknown}"
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
  echo "Run aggregation locally for figures + Wilson CIs if not done remotely:"
  echo "  PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_aggregate_phase0_gpu.py"
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
echo "Phase 0 watch_and_commit DONE."
echo "============================================================"
