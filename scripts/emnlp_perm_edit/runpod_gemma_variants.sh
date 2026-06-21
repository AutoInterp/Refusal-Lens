#!/bin/bash
# Regenerate Gemma variant attribution graphs (complement/full/outlier), gate
# each against Georg's committed nets, pack+annotate, and push to HF.
# Env knobs: DRY_RUN=1 (print plan only), NO_TMUX=1, VARIANTS=..., BATCH=128,
#   NPROMPTS=50, DATASET_REPO=moon70/refusal-lens-graphs.
set -u
cd "$(git rev-parse --show-toplevel)"
PY="${PY:-python3}"
VARIANTS="${VARIANTS:-complement full outlier}"
NPROMPTS="${NPROMPTS:-50}"
BATCH="${BATCH:-128}"
DATASET_REPO="${DATASET_REPO:-moon70/refusal-lens-graphs}"
RUNS=data/results/pipeline_runs
NETS_REF=data/results/emnlp_perm_edit/phase0_controllability/gemma_var_nets.json
DONE_MARKER=$RUNS/.GEMMA_VARIANTS_DONE
FAIL_LOG=$RUNS/.GEMMA_VARIANTS_FAILED.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Self-relaunch into tmux unless told not to.
if [ "${NO_TMUX:-0}" != "1" ] && [ -z "${TMUX:-}" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  exec tmux new-session -s gemma_variants "bash $0"
fi

rm -f "$DONE_MARKER" "$FAIL_LOG"

echo "### ensure variant directions ###"
$PY scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py || { echo "direction build FAILED"; exit 2; }

for v in $VARIANTS; do
  RD=$RUNS/gemma_var_$v
  if [ -f "$RD/.VARIANT_PUSHED" ]; then
    echo "[$v] already pushed (marker present); skipping. rm $RD/.VARIANT_PUSHED to force a re-run."
    continue
  fi
  echo "############ VARIANT=$v  $(date) ############"
  ATTR="$PY scripts/pipeline/02_run_attribution.py --run-dir $RD \
    --n-prompts $NPROMPTS --skip-multi-graph --target-layer 15 \
    --single-position-target -2 --measurement-hook hook_resid_post \
    --backend transformerlens --dtype float32 --batch-size $BATCH \
    --save-graphs --resume"
  GATE="$PY scripts/emnlp_perm_edit/verify_variant_nets.py \
    --attribution-results $RD/02_attribution/attribution_results.json \
    --nets-ref $NETS_REF --variant $v"
  VIZ="$PY scripts/pipeline/05_visualize_circuits.py --run-dir $RD \
    --mode single --skip-subcircuits --gzip"
  PUSH="$PY scripts/pipeline/push_graph_data.py --run-dir $RD --source 05_frontend \
    --dataset-repo $DATASET_REPO --run-name run_gemma_${v}_L15"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY: $ATTR"; echo "DRY: $GATE"; echo "DRY: $VIZ"; echo "DRY: $PUSH"; continue
  fi

  # Attribution with OOM retry.
  ok=0
  for attempt in 1 2 3 4 5; do
    echo "---- $v attribution attempt $attempt $(date) ----"
    eval $ATTR && { ok=1; break; }
    echo "retry after crash/OOM in 30s..."; sleep 30
  done
  [ "$ok" = "1" ] || { echo "$v: attribution FAILED" | tee -a "$FAIL_LOG"; continue; }

  echo "---- $v nets gate ----"
  if ! eval $GATE; then
    echo "$v: NETS GATE FAILED (graphs not trusted; check fork/hook/direction)" | tee -a "$FAIL_LOG"
    continue
  fi

  echo "---- $v pack+annotate (05) ----"; eval $VIZ || { echo "$v: 05 FAILED" | tee -a "$FAIL_LOG"; continue; }
  echo "---- $v push to HF ($DATASET_REPO/runs/run_gemma_${v}_L15) ----"
  eval $PUSH || { echo "$v: push FAILED" | tee -a "$FAIL_LOG"; continue; }

  echo "---- $v purge .pt to free disk ----"
  rm -rf "$RD/02_attribution/graphs"
  touch "$RD/.VARIANT_PUSHED"
  echo "$v DONE $(date)"
done

if [ "${DRY_RUN:-0}" != "1" ]; then
  touch "$DONE_MARKER"
  echo "############ ALL VARIANTS COMPLETE $(date) ############"
  [ -f "$FAIL_LOG" ] && { echo "FAILURES:"; cat "$FAIL_LOG"; }
fi
