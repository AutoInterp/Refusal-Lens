#!/bin/bash
# Qwen Stage 2 attribution across 3 refusal-direction variants (full/outlier/complement),
# unit-normalized targets at L18 pos -1, measurement_hook=hook_resid_post (patched
# circuit-tracer). 50 prompts x 11 conditions. Graphs local (move to HDD at end).
# Qwen transcoders are 160k-wide -> high VRAM; expandable_segments + small batch.
set -u
cd /mnt/c/Users/Georg/Code/Refusal-Lens
PY=.venv/bin/python
NPROMPTS=50
BATCH=96
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

count_done () {
  $PY -c "import json;print(len(json.load(open('data/results/pipeline_runs_qwen/qwen_var_$1/02_attribution/attribution_results.json'))['results']))" 2>/dev/null || echo 0
}

for variant in complement outlier full; do
  RD=data/results/pipeline_runs_qwen/qwen_var_$variant/02_attribution
  mkdir -p "$RD/graphs"
  echo "############ QWEN VARIANT=$variant start $(date) ############"
  for attempt in 1 2 3 4 5 6 7 8; do
    echo "---- $variant attempt $attempt $(date) (done: $(count_done $variant)) ----"
    $PY scripts/pipeline_qwen/02_run_attribution.py \
      --run-dir data/results/pipeline_runs_qwen/qwen_var_$variant \
      --n-prompts $NPROMPTS --skip-multi-graph --target-layer 18 \
      --single-position-target -1 --dtype float32 --batch-size $BATCH --save-graphs --resume
    rc=$?
    echo "---- $variant attempt $attempt exit=$rc done=$(count_done $variant) $(date) ----"
    if [ "$rc" = "0" ]; then echo "$variant ATTRIBUTION COMPLETE"; break; fi
    echo "retry after OOM/crash in 30s..."; sleep 30
  done
done
echo "############ ALL QWEN VARIANT ATTRIBUTION DONE $(date) ############"
