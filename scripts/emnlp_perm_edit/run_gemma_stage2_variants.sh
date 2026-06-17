#!/bin/bash
# Multi-hour Gemma Stage 2 attribution across 3 refusal-direction variants
# (full / outlier / complement), unit-normalized targets at L15 pos -2,
# measurement_hook=hook_resid_post (patched circuit-tracer). 50 prompts x 11 cond.
#
# SPEED FIX: graphs are written to the LOCAL SSD (3 GB/s) during attribution, then
# moved to the external HDD (/mnt/d, 65 MB/s) in the BACKGROUND while the next
# variant computes. Keeps the GPU at full throughput AND lands graphs on the HDD.
# --resume + retry loop survives transient RAM/GPU OOMs.
set -u
cd /mnt/c/Users/Georg/Code/Refusal-Lens
PY=.venv/bin/python
NPROMPTS=50
HDD=/mnt/d/refusal_graphs
BATCH=128
# Avoid CUDA allocator thrashing near 32GB capacity (outlier/full targets activate
# more features than complement and were pinning VRAM at 98% -> ~7min/graph).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! ls /mnt/d >/dev/null 2>&1; then echo "FATAL: /mnt/d not mounted"; exit 2; fi

count_done () {
  $PY -c "import json;print(len(json.load(open('data/results/pipeline_runs/gemma_var_$1/02_attribution/attribution_results.json'))['results']))" 2>/dev/null || echo 0
}

MOVE_PIDS=""
for variant in complement outlier full; do
  RD=data/results/pipeline_runs/gemma_var_$variant/02_attribution
  mkdir -p "$RD/graphs" "$HDD/gemma_var_$variant"
  echo "############ VARIANT=$variant start $(date) (local graphs -> bg move to HDD) ############"
  for attempt in 1 2 3 4 5 6 7 8; do
    echo "---- $variant attempt $attempt $(date) (done: $(count_done $variant)) ----"
    $PY scripts/pipeline/02_run_attribution.py \
      --run-dir data/results/pipeline_runs/gemma_var_$variant \
      --n-prompts $NPROMPTS --skip-multi-graph --target-layer 15 \
      --single-position-target -2 --dtype float32 --batch-size $BATCH --save-graphs --resume
    rc=$?
    echo "---- $variant attempt $attempt exit=$rc done=$(count_done $variant) $(date) ----"
    if [ "$rc" = "0" ]; then echo "$variant ATTRIBUTION COMPLETE"; break; fi
    echo "retry after OOM/crash in 30s..."; sleep 30
  done
  # Background move this variant's graphs to the HDD, freeing local space, while
  # the next variant computes. rsync --remove-source-files only deletes after a
  # verified transfer.
  # HDD moves are DEFERRED to the very end (mv contends badly with attribution on
  # drvfs). Graphs stay on the local SSD (1TB free) until all variants finish.
  echo "[$variant] attribution done; graphs kept local, move deferred to end."
done
echo "############ ALL GEMMA VARIANT ATTRIBUTION DONE $(date) — graphs local, move to HDD next ############"
