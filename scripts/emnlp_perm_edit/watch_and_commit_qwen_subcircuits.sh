#!/usr/bin/env bash
# Watch the Qwen subcircuits run and auto-commit + push results, then upload
# the merged subcircuits.json to the HF dataset so the frontend picks it up.
#
# Run in a SECOND tmux window/terminal while runpod_qwen_subcircuits.sh runs
# in tmux session 'qwen_subcircuits':
#   tmux new-window -t qwen_subcircuits -n watcher \
#     "bash scripts/emnlp_perm_edit/watch_and_commit_qwen_subcircuits.sh 2>&1 | tee /tmp/qwen_subcircuits_watcher.log"

set -uo pipefail

ROOT="${ROOT:-$(pwd)}"
RUN_NAME="${RUN_NAME:-run_emnlp_qwen_L18_20260522}"
QWEN_RUN="$ROOT/data/results/pipeline_runs_qwen/$RUN_NAME"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/qwen_subcircuits"
DONE_FILE="$OUT_DIR/.QWEN_SUBCIRCUITS_DONE"
FAIL_FILE="$OUT_DIR/.QWEN_SUBCIRCUITS_STEP_FAILED.txt"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-300}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-36}"
GIT_REMOTE_BRANCH="${GIT_REMOTE_BRANCH:-emnlp-perm-edit}"
HF_DATASET="${HF_DATASET:-moon70/refusal-lens-graphs}"
SKIP_HF_UPLOAD="${SKIP_HF_UPLOAD:-0}"

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
echo "[$(date +%H:%M:%S)] Done marker detected."
if [[ -f "$FAIL_FILE" ]]; then echo "Step failures:"; cat "$FAIL_FILE"; fi

# --- Collect result files (skip anything missing) ---
RESULT_FILES=(
  "$OUT_DIR/topk_sweep_zero_features.json"
  "$OUT_DIR/topk_sweep_proxy_features.json"
  "$OUT_DIR/topk_sweep_proxy_edges.json"
  "$OUT_DIR/pareto_curves.json"
  "$OUT_DIR/pareto_curves.png"
  "$OUT_DIR/subcircuits_frontend.json"
  "$OUT_DIR/QWEN_SUBCIRCUIT_REPORT.md"
  "$QWEN_RUN/02_attribution/attribution_results.json"
  "$QWEN_RUN/04_labels/feature_labels.json"
  "$QWEN_RUN/04_labels/feature_class_sets.json"
  "$QWEN_RUN/04_labels/feature_comparison_labeled.json"
  "$QWEN_RUN/04_labels/label_coverage.json"
  "$QWEN_RUN/04_labels/layer_histogram.json"
  "$QWEN_RUN/07_subcircuits/subcircuits.json"
  "$QWEN_RUN/07_subcircuits/subcircuits_summary.json"
  "$QWEN_RUN/07_subcircuits/SUBCIRCUITS_REPORT.md"
  "$QWEN_RUN/08_ablation/ablation_results.json"
  "$QWEN_RUN/08_ablation/ablation_summary.json"
  "$QWEN_RUN/08_ablation/ABLATION_SUMMARY.md"
  "$QWEN_RUN/08_ablation/dissociation_matrix.png"
  "$QWEN_RUN/08_ablation/positions_comparison.png"
)
OPTIONAL_FILES=("$FAIL_FILE")

to_add=()
for f in "${RESULT_FILES[@]}"; do
  if [[ -f "$f" ]]; then to_add+=("$f"); else echo "  (missing: ${f#$ROOT/})"; fi
done
for f in "${OPTIONAL_FILES[@]}"; do
  if [[ -f "$f" ]]; then to_add+=("$f"); fi
done
if [[ ${#to_add[@]} -eq 0 ]]; then echo "Nothing to commit."; exit 3; fi

COMMIT_MSG_FILE=$(mktemp)
{
  echo "qwen subcircuits: stage 07/08 replication + Top-K sparsity sweep (L18)"
  echo ""
  echo "Rule-based subcircuit identification + ReplacementModel zero-ablation +"
  echo "Top-K Pareto sweep (zero/proxy mechanisms, attribution/activation rankings)"
  echo "per design spec docs/superpowers/specs/2026-06-01-qwen-subcircuits-topk-design.md."
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
  echo "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
} > "$COMMIT_MSG_FILE"

git add -f "${to_add[@]}"
if git diff --staged --quiet; then echo "Nothing to commit."; rm "$COMMIT_MSG_FILE"; exit 0; fi
git commit -F "$COMMIT_MSG_FILE"
COMMIT_SHA=$(git rev-parse --short HEAD)
rm "$COMMIT_MSG_FILE"
echo "Local commit: $COMMIT_SHA"

if git push origin "HEAD:$GIT_REMOTE_BRANCH"; then
  echo "[$(date +%H:%M:%S)] PUSH OK: $COMMIT_SHA on origin/$GIT_REMOTE_BRANCH"
else
  echo "[$(date +%H:%M:%S)] PUSH FAILED. Commit $COMMIT_SHA is local-only."
  echo "Fix auth and run: git push origin HEAD:$GIT_REMOTE_BRANCH"
fi

# --- HF upload: merged subcircuits.json + small derived artifacts ---
if [[ "$SKIP_HF_UPLOAD" == "1" ]]; then
  echo "SKIP_HF_UPLOAD=1 — done."
  exit 0
fi
echo ""
echo "Uploading frontend artifacts to HF dataset $HF_DATASET ..."
python3 - <<PYEOF
import os, sys
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN") or None)
repo = "$HF_DATASET"
run = "$RUN_NAME"
uploads = [
    ("$OUT_DIR/subcircuits_frontend.json", f"runs/{run}/subcircuits.json"),
    ("$OUT_DIR/pareto_curves.json",        f"runs/{run}/pareto_curves.json"),
    ("$OUT_DIR/QWEN_SUBCIRCUIT_REPORT.md", f"runs/{run}/QWEN_SUBCIRCUIT_REPORT.md"),
]
ok = True
for local, remote in uploads:
    p = Path(local)
    if not p.exists():
        print(f"  (skip missing {p.name})"); continue
    try:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=remote,
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"qwen subcircuits run artifacts: {remote}")
        print(f"  uploaded {remote}")
    except Exception as e:
        ok = False
        print(f"  UPLOAD FAILED {remote}: {e}")
sys.exit(0 if ok else 5)
PYEOF
rc=$?
if [[ $rc -eq 0 ]]; then
  echo "HF upload complete. Frontend pickup: python3 scripts/pipeline/fetch_graph_data.py --run $RUN_NAME --dataset-repo $HF_DATASET"
else
  echo "HF upload had failures (exit $rc). Re-run this watcher or upload manually."
fi
echo "DONE."
