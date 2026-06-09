#!/usr/bin/env bash
# Smoke test for the Qwen subcircuits + Top-K sweep run. Run this ON THE POD
# (GPU required for steps 4-7) before launching the full job.
#
# Exercises every step with --max-prompts 2 / K=1,5 into an isolated smoke dir
# (never touches the real run dir outputs), then asserts output schemas.
# Expected wall: ~20-30 min, ~$1.50 — dominated by the two ReplacementModel loads.
#
# Usage:  bash scripts/emnlp_perm_edit/smoke_test_qwen_subcircuits.sh
# CPU-only portion (laptop): CPU_ONLY=1 bash scripts/emnlp_perm_edit/smoke_test_qwen_subcircuits.sh

set -uo pipefail

ROOT="${ROOT:-$(pwd)}"
RUN_NAME="${RUN_NAME:-run_emnlp_qwen_L18_20260522}"
QWEN_RUN="$ROOT/data/results/pipeline_runs_qwen/$RUN_NAME"
GRAPH_DIR="${GRAPH_DIR:-$QWEN_RUN/graph_data}"
# Fall back to the locally fetched frontend copy (laptop layout)
if [[ ! -f "$GRAPH_DIR/000_bare_single.json.gz" ]]; then
  ALT="$ROOT/data/results/pipeline_runs/$RUN_NAME/05_frontend/graph_data"
  [[ -f "$ALT/000_bare_single.json.gz" ]] && GRAPH_DIR="$ALT"
fi
SMOKE_RUN="${SMOKE_RUN:-/tmp/qwen_subcircuits_smoke_run}"
SMOKE_OUT="${SMOKE_OUT:-/tmp/qwen_subcircuits_smoke_out}"
RHAT_PATH="$QWEN_RUN/01_direction/positions_L18/pos_-1_unnormalized.pt"
CPU_ONLY="${CPU_ONLY:-0}"
PY="${PY:-python3}"

echo "Smoke: graphs=$GRAPH_DIR  run=$SMOKE_RUN  out=$SMOKE_OUT  CPU_ONLY=$CPU_ONLY"
rm -rf "$SMOKE_RUN" "$SMOKE_OUT"
mkdir -p "$SMOKE_RUN" "$SMOKE_OUT"
cd "$ROOT"

fail() { echo "SMOKE FAILED: $1"; exit 1; }

# Self-stage the packed graphs from HF if absent (~180MB; same staging the launcher does)
if [[ ! -f "$GRAPH_DIR/000_bare_single.json.gz" ]]; then
  echo "Packed graphs not found — pulling from HF (moon70/refusal-lens-graphs)..."
  GRAPH_DIR="$QWEN_RUN/graph_data"
  $PY - <<PYEOF || fail "HF graph pull (check HF_TOKEN / network)"
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil
snap = snapshot_download(repo_id="${HF_DATASET:-moon70/refusal-lens-graphs}", repo_type="dataset",
                         allow_patterns=["runs/$RUN_NAME/graph_data/*"])
src = Path(snap) / "runs" / "$RUN_NAME" / "graph_data"
dst = Path("$GRAPH_DIR"); dst.mkdir(parents=True, exist_ok=True)
n = 0
for f in src.iterdir():
    shutil.copy2(f, dst / f.name); n += 1
print(f"staged {n} graph files into $GRAPH_DIR")
PYEOF
fi
[[ -f "$GRAPH_DIR/000_bare_single.json.gz" ]] || fail "packed graphs not found at $GRAPH_DIR"

# 1. rebuild attribution index (2 prompts)
PYTHONPATH=scripts $PY scripts/emnlp_perm_edit/qwen_rebuild_attribution_index.py \
  --graph-data-dir "$GRAPH_DIR" --run-dir "$SMOKE_RUN" --n-prompts 2 --force \
  || fail "rebuild_attribution_index"
$PY -c "
import json
d=json.load(open('$SMOKE_RUN/02_attribution/attribution_results.json'))
assert d['metadata']['reconstructed'] is True
assert len(d['results'])==2
r=d['results'][0]
assert len(r['conditions'])==11, f\"expected 11 conditions, got {len(r['conditions'])}\"
g=r['conditions']['bare']['graphs']['single']
assert g['top_features'] and g['top50_features']
assert r['feature_comparison']['fiction']['vs_bare']['n_shared'] > 0
print('  [1/8] rebuild index OK')
" || fail "rebuild index assertions"

# 2. Stage 04 bookkeeping
PYTHONPATH=scripts/pipeline_qwen $PY scripts/pipeline_qwen/04_label_features.py \
  --run-dir "$SMOKE_RUN" --skip-download >/dev/null || fail "stage04"
$PY -c "
import json
fl=json.load(open('$SMOKE_RUN/04_labels/feature_labels.json'))
cs=json.load(open('$SMOKE_RUN/04_labels/feature_class_sets.json'))
assert len(fl)>50
v=next(iter(fl.values()))
assert 'conditions_seen' in v and 'layer' in v
assert 'per_condition_top50' in cs and 'by_bucket' in cs
print(f'  [2/8] stage04 OK ({len(fl)} features)')
" || fail "stage04 assertions"

# 3. Stage 07 subcircuits
PYTHONPATH=scripts/pipeline_qwen $PY scripts/pipeline_qwen/07_identify_subcircuits.py \
  --run-dir "$SMOKE_RUN" --graph-mode single >/dev/null || fail "stage07"
$PY -c "
import json
d=json.load(open('$SMOKE_RUN/07_subcircuits/subcircuits.json'))
sc=d['subcircuits']
for name in ('universal_refusal_core','ctrl_shared_refusal','jb_fiction_specific_vs_ctrl'):
    assert name in sc, name
    assert isinstance(sc[name]['features'], list)
assert len(sc['universal_refusal_core']['features'])>0
print(f'  [3/8] stage07 OK ({len(sc)} subcircuits)')
" || fail "stage07 assertions"

if [[ "$CPU_ONLY" == "1" ]]; then
  echo "CPU_ONLY=1 — skipping GPU steps 4-7, running aggregator on partial data."
else
  [[ -f "$RHAT_PATH" ]] || fail "direction file missing: $RHAT_PATH (run launcher prereq staging first)"

  # 4. Stage 08 (1 subcircuit, BOTH position modes, 2 prompts).
  # 'both' is required: the full run leads with 'all' (slice-position
  # interventions), which is a different backend code path than 'anchors' —
  # an anchors-only smoke previously passed while the full run crashed.
  PYTHONPATH=scripts/pipeline_qwen $PY scripts/pipeline_qwen/08_ablate_subcircuits.py \
    --run-dir "$SMOKE_RUN" --graph-mode single --max-prompts 2 \
    --subcircuits universal_refusal_core --positions both --max-new-tokens 40 \
    --backend transformerlens \
    || fail "stage08"
  $PY -c "
import json
d=json.load(open('$SMOKE_RUN/08_ablation/ablation_results.json'))
rows=d['results'] if isinstance(d,dict) else d
assert len(rows)>=1
r0=rows[0]
assert 'baseline' in r0 and 'ablations' in r0
abl=r0['ablations']['universal_refusal_core']
assert 'all' in abl and 'anchors' in abl, f'expected both position modes, got {list(abl)}'
for mode in ('all','anchors'):
    cond=next(iter(abl[mode].values()))
    assert 'cls' in cond and 'changed_vs_baseline' in cond
print('  [4/8] stage08 OK (both position modes)')
" || fail "stage08 assertions"

  # 5-7. Top-K sweeps (2 prompts, K=1,5)
  PYTHONPATH=scripts $PY scripts/emnlp_perm_edit/00_topk_circuit_sweep_qwen.py \
    --mechanism zero --source features --backend transformerlens \
    --graph-data-dir "$GRAPH_DIR" \
    --max-prompts 2 --k-values 1,5 --max-new-tokens 40 \
    --out "$SMOKE_OUT/topk_sweep_zero_features.json" || fail "topk zero"
  PYTHONPATH=scripts $PY scripts/emnlp_perm_edit/00_topk_circuit_sweep_qwen.py \
    --mechanism proxy --source features --graph-data-dir "$GRAPH_DIR" \
    --rhat-path "$RHAT_PATH" --max-prompts 2 --k-values 1,5 --max-new-tokens 40 \
    --out "$SMOKE_OUT/topk_sweep_proxy_features.json" || fail "topk proxy features"
  PYTHONPATH=scripts $PY scripts/emnlp_perm_edit/00_topk_circuit_sweep_qwen.py \
    --mechanism proxy --source edges --graph-data-dir "$GRAPH_DIR" \
    --rhat-path "$RHAT_PATH" --max-prompts 2 --k-values 1,5 --max-new-tokens 40 \
    --skip-baseline --out "$SMOKE_OUT/topk_sweep_proxy_edges.json" || fail "topk proxy edges"
  $PY -c "
import json
for name in ('zero_features','proxy_features','proxy_edges'):
    d=json.load(open(f'$SMOKE_OUT/topk_sweep_{name}.json'))
    cells=d['per_cell']
    assert cells, name
    recs=next(iter(cells.values()))
    r0=recs[0]
    assert {'prompt_idx','condition','n_used','classification','coherent'} <= set(r0)
    if d['metadata']['mechanism']=='proxy' and 'pos_K1' in cells:
        assert 'delta_applied' in cells['pos_K1'][0]
b=json.load(open('$SMOKE_OUT/topk_sweep_proxy_features.json'))['baseline']
assert b.get('records'), 'proxy baseline missing'
z=json.load(open('$SMOKE_OUT/topk_sweep_zero_features.json'))['baseline']
assert z.get('records'), 'zero baseline missing'
print('  [5-7/8] topk sweeps OK')
" || fail "topk sweep assertions"
fi

# 8. aggregate
PYTHONPATH=scripts $PY scripts/emnlp_perm_edit/qwen_subcircuits_aggregate.py \
  --run-dir "$SMOKE_RUN" --out-dir "$SMOKE_OUT" || fail "aggregate"
$PY -c "
import json
p=json.load(open('$SMOKE_OUT/pareto_curves.json'))
sf=json.load(open('$SMOKE_OUT/subcircuits_frontend.json'))
assert 'subcircuits' in sf and 'universal_refusal_core' in sf['subcircuits']
assert any(k.startswith('topk_refusal') for k in sf['subcircuits'])
import os; assert os.path.exists('$SMOKE_OUT/QWEN_SUBCIRCUIT_REPORT.md')
print('  [8/8] aggregate OK')
" || fail "aggregate assertions"

echo ""
echo "SMOKE TEST PASSED. Full run: bash scripts/emnlp_perm_edit/runpod_qwen_subcircuits.sh"
