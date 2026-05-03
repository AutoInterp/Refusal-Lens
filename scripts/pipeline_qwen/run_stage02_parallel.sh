#!/usr/bin/env bash
# Run Stage 02 attribution across N GPUs in parallel by sharding the prompt list.
#
# Usage:
#   run_stage02_parallel.sh --run-dir <path> --n-gpus 4 [--n-prompts 50] [--batch-size 256] [...other stage 02 flags]
#
# Each GPU i runs its own Stage 02 process against prompts [i*N/N_GPUS, (i+1)*N/N_GPUS)
# with CUDA_VISIBLE_DEVICES=i. Shard-specific checkpoint files live under the same
# 02_attribution/ dir so a crashed shard can --resume cleanly.
#
# After all shards finish, run merge_stage02_shards.py to stitch them into a single
# attribution_results.json.

set -euo pipefail

RUN_DIR=""
N_GPUS=1
N_PROMPTS=50
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)   RUN_DIR="$2"; shift 2 ;;
        --n-gpus)    N_GPUS="$2"; shift 2 ;;
        --n-prompts) N_PROMPTS="$2"; shift 2 ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$RUN_DIR" ]]; then
    echo "Error: --run-dir required" >&2
    exit 1
fi

mkdir -p "$RUN_DIR/02_attribution/logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS=()

echo "Sharding $N_PROMPTS prompts across $N_GPUS GPUs..."
for (( gpu=0; gpu<N_GPUS; gpu++ )); do
    # Ceiling division so the last shard picks up any remainder
    start=$(( gpu * N_PROMPTS / N_GPUS ))
    end=$(( (gpu + 1) * N_PROMPTS / N_GPUS ))
    log_file="$RUN_DIR/02_attribution/logs/shard_gpu${gpu}_${start}_${end}.log"

    echo "  GPU $gpu: prompts [$start, $end) -> $log_file"
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT_DIR/02_run_attribution.py" \
        --run-dir "$RUN_DIR" \
        --n-prompts "$N_PROMPTS" \
        --prompt-start "$start" \
        --prompt-end "$end" \
        "${EXTRA_ARGS[@]}" \
        > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo "All shards launched (PIDs: ${PIDS[*]}). Tail logs with:"
echo "  tail -f $RUN_DIR/02_attribution/logs/shard_gpu*.log"
echo

# Wait for all shards; report failures loudly.
FAILED=()
for (( gpu=0; gpu<N_GPUS; gpu++ )); do
    pid=${PIDS[$gpu]}
    if wait "$pid"; then
        echo "Shard GPU $gpu (pid $pid) OK"
    else
        echo "Shard GPU $gpu (pid $pid) FAILED — see $RUN_DIR/02_attribution/logs/shard_gpu${gpu}_*.log"
        FAILED+=("$gpu")
    fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "ERROR: ${#FAILED[@]} shard(s) failed: ${FAILED[*]}"
    exit 1
fi

echo
echo "All shards complete. Merging shards into attribution_results.json..."
python "$SCRIPT_DIR/merge_stage02_shards.py" --run-dir "$RUN_DIR"

echo "DONE."
