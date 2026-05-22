#!/usr/bin/env bash
# Stage 08a — overnight RunPod run with auto-commit on success.
#
# Usage on the pod (after `cd /workspace/Refusal-Lens && source /workspace/venv/bin/activate`):
#   export GHP_TOKEN=ghp_...                # GitHub PAT, contents:write on the repo
#   export RUN_DIR=data/results/pipeline_runs_qwen/run_20260422_015552
#   export BRANCH=l15-refactor              # branch to commit/push to
#   bash scripts/pipeline_qwen/run_stage08_overnight.sh
#
# Flags chosen for overnight:
#   --positions both         comparative analysis (all + anchors)
#   --max-new-tokens 80      cuts wall clock ~2.5× vs default 200; refuse/comply
#                            classification is decided in the first ~30 tokens.
#   --skip-baseline          reuse Stage 06's per-prompt baselines
#   --resume                 picks up after preemption
#   --checkpoint-every 5     small enough that resume cost stays bounded
#
# Wall-clock estimate on a single H100: ~10–14 hours for 50 prompts × 11 conds ×
# 5 ablation sets × 2 positions modes. Adjust by trimming --positions or
# --subcircuits if you want it tighter.
#
# Outputs land in $RUN_DIR/08_ablation/. The watcher commits + pushes when the
# script prints `^DONE!` and exits. On any Traceback / Error / FAIL line it
# stops and refuses to commit (so you can debug instead of pushing a half run).

set -euo pipefail

: "${RUN_DIR:?must set RUN_DIR=data/results/pipeline_runs_qwen/run_<ts>}"
: "${GHP_TOKEN:?must export GHP_TOKEN}"
: "${BRANCH:=l15-refactor}"
: "${SESSION:=stage08}"

LOG=/workspace/stage08_output.log
REPO=/workspace/Refusal-Lens
VENV_ACTIVATE=/workspace/venv/bin/activate

cd "$REPO"

# Sanity: prerequisites must exist
if [[ ! -f "$RUN_DIR/07_subcircuits/subcircuits.json" ]]; then
  echo "ERROR: $RUN_DIR/07_subcircuits/subcircuits.json missing — run Stage 07 first."
  exit 1
fi
if [[ ! -f "$RUN_DIR/06_causal/causal_results.json" ]]; then
  echo "WARNING: $RUN_DIR/06_causal/causal_results.json missing — Stage 06 baselines won't be reused."
  echo "         Either drop --skip-baseline below, or run Stage 06 first."
fi

# Kill any existing tmux session of this name
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Window 1 — the actual run
tmux new-session -d -s "$SESSION" -n run \
  "cd $REPO && source $VENV_ACTIVATE && \
   PYTHONPATH=src python3 -u scripts/pipeline_qwen/08_ablate_subcircuits.py \
     --run-dir $RUN_DIR \
     --positions both \
     --max-new-tokens 80 \
     --skip-baseline \
     --resume \
     --checkpoint-every 5 \
   2>&1 | tee $LOG; \
   echo '[run-window] script exited with $?'; \
   sleep 600"

# Window 2 — watcher with auto-commit/push on success
# - Polls log every 60s, prints heartbeat
# - Trips on Traceback / Error / FAIL → refuses to commit, prints last 50 lines
# - Trips on `^DONE!` (the script's success marker) → commits + pushes
tmux new-window -t "$SESSION" -n watcher "
set -e
L='$LOG'
RUN_DIR='$RUN_DIR'
BRANCH='$BRANCH'
GHP='$GHP_TOKEN'
echo \"[watcher] watching \$L\"
echo \"[watcher] start: \$(date)\"
START_T=\$(date +%s)
while true; do
  if grep -qE 'Traceback|^FAIL|RuntimeError|CUDA out of memory|AssertionError' \$L 2>/dev/null; then
    echo '[watcher] ❌ FAILED — last 50 lines:'
    tail -n 50 \$L
    echo '[watcher] NOT committing. Investigate, then re-run.'
    sleep 1800
    exit 1
  fi
  if grep -q '^DONE!' \$L 2>/dev/null; then
    break
  fi
  NOW=\$(date +%s)
  ELAPSED=\$(( (NOW - START_T) / 60 ))
  TAIL=\$(tail -n 1 \$L 2>/dev/null | cut -c1-110)
  echo \"[\$(date +%H:%M:%S)  +\${ELAPSED}min] tail: \$TAIL\"
  sleep 60
done

echo '[watcher] ✅ run complete — committing'
cd $REPO

# Snapshot the log into the run dir for posterity
cp \$L \$RUN_DIR/08_ablation/run_log.txt 2>/dev/null || true

# Configure git (idempotent — won't overwrite existing config)
git config user.email \"\${GIT_AUTHOR_EMAIL:-mahmoud@local}\" 2>/dev/null || true
git config user.name \"\${GIT_AUTHOR_NAME:-Stage 08 runner}\" 2>/dev/null || true

# Build authenticated remote URL
U=\$(git remote get-url origin | sed \"s|https://|https://x-access-token:\${GHP}@|\")

# Pull + rebase, then add the run dir + commit + push
git fetch \$U \$BRANCH
git checkout \$BRANCH
git pull --rebase \$U \$BRANCH

git add \$RUN_DIR/08_ablation/

if git diff --staged --quiet; then
  echo '[watcher] nothing to commit (no staged changes)'
else
  git commit -m 'stage 08a: full ablation results from overnight runpod run' \
             -m \"run_dir: \$RUN_DIR\" \
             -m \"positions: both, max_new_tokens: 80, baselines reused from 06_causal\"
  git push \$U \$BRANCH
  echo '[watcher] ✅ pushed to '\$BRANCH
fi

echo '[watcher] all done — leaving session up so you can attach and inspect'
sleep 7200
"

echo "[runner] tmux session '$SESSION' started."
echo "[runner] tmux ls; tmux attach -t $SESSION"
echo "[runner]   Ctrl-b n           switch windows (run / watcher)"
echo "[runner]   Ctrl-b d           detach (leaves it running)"
echo "[runner] tail -f $LOG         follow output without tmux"
