#!/usr/bin/env bash
# End-to-end automation: fetch raw .pt from HF → pack to JSON.gz → annotate via
# Stage 05 → push annotated bundle back to HF. Run unattended, check status via
# a DONE/FAIL marker file or tail the log.
#
# Usage:
#   tmux new -s stage05
#   bash scripts/pipeline_qwen/auto_fetch_pack_push.sh [run_name] [mode]
#   # Detach: Ctrl-b d
#   # Reattach later: tmux attach -t stage05
#
# Env: HF_TOKEN must be set (write scope for the target dataset).
# Outputs:
#   /tmp/stage05_auto.log                — live log of all 4 steps
#   /tmp/stage05_auto.DONE  (on success) — contains end time + duration
#   /tmp/stage05_auto.FAIL  (on failure) — contains step that died

set -euo pipefail

RUN_NAME="${1:-run_20260422_015552}"
MODE="${2:-single}"   # single | multi | both
DATASET_REPO="${DATASET_REPO:-moon70/refusal-lens-graphs}"
LOG=/tmp/stage05_auto.log
DONE_MARKER=/tmp/stage05_auto.DONE
FAIL_MARKER=/tmp/stage05_auto.FAIL

# Redirect all output to tee the log + stdout.
exec > >(tee "$LOG") 2>&1

# Reset markers
rm -f "$DONE_MARKER" "$FAIL_MARKER"

# Pin the step name so trap can report which phase failed.
STEP="init"
trap 'echo "[$(date)] FAILED at step: $STEP" > "$FAIL_MARKER"; echo "[$(date)] !! TRAP: exit $? at step $STEP"' ERR

SECONDS=0
echo "============================================================"
echo "[$(date)] Refusal-Lens Stage 05 automation"
echo "  run:         $RUN_NAME"
echo "  mode:        $MODE"
echo "  dataset:     $DATASET_REPO"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "  HF_TOKEN:    set (length=${#HF_TOKEN})"
else
  echo "  HF_TOKEN:    NOT SET (will fail at push)"
fi
echo "============================================================"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
RUN_DIR="data/results/pipeline_runs_qwen/$RUN_NAME"

# ------------------------------------------------------------------
# Step 1: Fetch raw .pt from HF
# ------------------------------------------------------------------
STEP="fetch"
echo ""
echo "[$(date)] [1/4] Fetching .pt files (--mode $MODE)..."
t=$SECONDS
python3 scripts/pipeline_qwen/fetch_raw_graphs.py \
    --run "$RUN_NAME" \
    --dataset-repo "$DATASET_REPO" \
    --subdir 02_attribution/graphs \
    --mode "$MODE"
n_pt=$(ls "$RUN_DIR/02_attribution/graphs/"*.pt 2>/dev/null | wc -l)
size_pt=$(du -sh "$RUN_DIR/02_attribution/graphs/" 2>/dev/null | cut -f1)
echo "[$(date)] [1/4] DONE — $n_pt .pt files ($size_pt) in $(( SECONDS - t ))s"

# ------------------------------------------------------------------
# Step 2: Pack .pt → JSON.gz
# ------------------------------------------------------------------
STEP="pack"
echo ""
echo "[$(date)] [2/4] Packing .pt → JSON.gz (this is the slow step, ~30–60 s/graph)..."
t=$SECONDS
python3 scripts/pipeline_qwen/02c_pack_graphs.py \
    --run-dir "$RUN_DIR"
n_gz=$(ls "$RUN_DIR/graph_data/"*.json.gz 2>/dev/null | wc -l)
size_gz=$(du -sh "$RUN_DIR/graph_data/" 2>/dev/null | cut -f1)
echo "[$(date)] [2/4] DONE — $n_gz .json.gz files ($size_gz) in $(( SECONDS - t ))s"

# ------------------------------------------------------------------
# Step 3: Annotate + stage frontend (Stage 05 --skip-convert)
# ------------------------------------------------------------------
STEP="annotate_stage05"
echo ""
echo "[$(date)] [3/4] Annotating + staging frontend via Stage 05..."
t=$SECONDS
python3 scripts/pipeline_qwen/05_visualize_circuits.py \
    --run-dir "$RUN_DIR" \
    --subcircuits-run "$RUN_DIR" \
    --mode "$MODE" \
    --skip-convert \
    --source-graph-data "$RUN_DIR/graph_data" \
    --gzip
echo "[$(date)] [3/4] DONE in $(( SECONDS - t ))s"

# ------------------------------------------------------------------
# Step 4: Push annotated bundle to HF
# ------------------------------------------------------------------
STEP="push_hf"
echo ""
echo "[$(date)] [4/4] Pushing annotated graph_data + subcircuits.json to HF..."
t=$SECONDS
python3 scripts/pipeline_qwen/push_graph_data.py \
    --run-dir "$RUN_DIR" \
    --dataset-repo "$DATASET_REPO"
echo "[$(date)] [4/4] DONE in $(( SECONDS - t ))s"

# ------------------------------------------------------------------
# Success
# ------------------------------------------------------------------
STEP="finalize"
TOTAL_MIN=$(( SECONDS / 60 ))
cat > "$DONE_MARKER" <<EOF
run: $RUN_NAME
mode: $MODE
total_minutes: $TOTAL_MIN
ended_at: $(date)
browse: https://huggingface.co/datasets/$DATASET_REPO/tree/main/runs/$RUN_NAME/
laptop_fetch: python3 scripts/pipeline_qwen/fetch_graph_data.py --run $RUN_NAME --dataset-repo $DATASET_REPO
EOF

echo ""
echo "============================================================"
echo "[$(date)] ALL DONE — total ${TOTAL_MIN} min"
echo "  Annotated bundle at: https://huggingface.co/datasets/$DATASET_REPO/tree/main/runs/$RUN_NAME/"
echo "  On your laptop:"
echo "    python3 scripts/pipeline_qwen/fetch_graph_data.py --run $RUN_NAME --dataset-repo $DATASET_REPO"
echo "    cd $RUN_DIR/05_frontend && python3 -m http.server 8000"
echo "============================================================"
