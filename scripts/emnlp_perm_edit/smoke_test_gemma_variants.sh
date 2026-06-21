#!/bin/bash
# 2-prompt GPU smoke for the Gemma variant pipeline: attribution -> nets gate
# (loose) -> 05 pack/annotate -> push --dry-run, into a throwaway run-dir.
set -eu
cd "$(git rev-parse --show-toplevel)"
PY="${PY:-python3}"
NETS_REF=data/results/emnlp_perm_edit/phase0_controllability/gemma_var_nets.json
SMOKE=/tmp/gemma_var_smoke
rm -rf "$SMOKE"; mkdir -p "$SMOKE/01_direction/directions" "$SMOKE/01_direction/positions_L15"

echo "### stage complement direction into smoke run-dir ###"
cp data/results/pipeline_runs/gemma_var_complement/01_direction/directions/layer_15.pt "$SMOKE/01_direction/directions/"
cp data/results/pipeline_runs/gemma_var_complement/01_direction/positions_L15/pos_-2.pt "$SMOKE/01_direction/positions_L15/"

echo "### attribution (2 prompts) ###"
$PY scripts/pipeline/02_run_attribution.py --run-dir "$SMOKE" \
  --n-prompts 2 --skip-multi-graph --target-layer 15 --single-position-target -2 \
  --measurement-hook hook_resid_post --backend transformerlens --dtype float32 \
  --batch-size 64 --save-graphs --resume

echo "### nets gate (loose: 2-prompt, sign + magnitude only) ###"
$PY scripts/emnlp_perm_edit/verify_variant_nets.py \
  --attribution-results "$SMOKE/02_attribution/attribution_results.json" \
  --nets-ref "$NETS_REF" --variant complement --corr-min -1.0 --bare-rel-tol 0.6

echo "### pack + annotate ###"
$PY scripts/pipeline/05_visualize_circuits.py --run-dir "$SMOKE" --mode single --skip-subcircuits --gzip

echo "### push (dry-run) ###"
$PY scripts/pipeline/push_graph_data.py --run-dir "$SMOKE" --source 05_frontend \
  --dataset-repo moon70/refusal-lens-graphs --run-name run_gemma_complement_L15 --dry-run

echo "SMOKE TEST PASSED"
