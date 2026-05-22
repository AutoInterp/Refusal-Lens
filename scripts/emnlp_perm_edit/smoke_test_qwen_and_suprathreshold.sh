#!/usr/bin/env bash
# Smoke test for Qwen full Phase 0 + Gemma supra-threshold.
#
# Verifies (with small subsets):
#   1. Qwen Stage 02 attribution can generate ONE graph at L18 (the riskiest new step)
#   2. Qwen Stage 02c can pack that single graph
#   3. Qwen 0a linearization runs on the packed graph (n=1)
#   4. Qwen 0b edge ablation runs on the linearization output (1 variant × 1 prompt)
#   5. Qwen direction sweep + layer locator run for 3 prompts each
#   6. Gemma supra-threshold (5x variant) runs for 3 prompts
#
# Wall: ~10-15 min on H100 SXM (mostly model loads + 1 attribution graph).
#
# Usage (from RunPod terminal at /workspace/Refusal-Lens):
#   bash scripts/emnlp_perm_edit/smoke_test_qwen_and_suprathreshold.sh

set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-smoke_qwen_L18}"
SMOKE_RUN_DIR="$ROOT/data/results/pipeline_runs_qwen/$SMOKE_RUN_NAME"
SMOKE_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability/smoke_qwen_supra"
QWEN_DIRECTIONS_SRC="$ROOT/data/results/pipeline_runs_qwen/run_20260502_154423/01_direction/directions"
QWEN_DIRECTIONS_RUN="$SMOKE_RUN_DIR/01_direction/directions"

mkdir -p "$SMOKE_DIR" "$QWEN_DIRECTIONS_RUN" "$SMOKE_RUN_DIR/01_direction"

cd "$ROOT"
if [[ ! -f "scripts/emnlp_perm_edit/00_direction_intervention_sweep_qwen.py" ]]; then
  echo "ERROR: expected to run from Refusal-Lens repo root."
  exit 1
fi
if [[ -d ".venv" ]]; then source .venv/bin/activate; fi

# Pull pipeline_qwen if not present
if [[ ! -f "scripts/pipeline_qwen/02_run_attribution.py" ]]; then
  echo "Fetching scripts/pipeline_qwen/ from temp/gemma-vs-qwen-pipeline..."
  git fetch origin temp/gemma-vs-qwen-pipeline 2>/dev/null || true
  git checkout origin/temp/gemma-vs-qwen-pipeline -- scripts/pipeline_qwen/
fi

# Pull Qwen direction files if not present
if [[ ! -f "$QWEN_DIRECTIONS_SRC/layer_18.pt" ]]; then
  echo "Fetching Qwen direction files..."
  git fetch origin temp/gemma-vs-qwen-pipeline 2>/dev/null || true
  mkdir -p "$QWEN_DIRECTIONS_SRC"
  QWEN_SRC_PREFIX="data/results/pipeline_runs_qwen/run_20260502_154423/01_direction/directions"
  for L in $(seq -f "%02g" 0 35); do
    git show origin/temp/gemma-vs-qwen-pipeline:"$QWEN_SRC_PREFIX/layer_${L}.pt" > "$QWEN_DIRECTIONS_SRC/layer_${L}.pt" 2>/dev/null || true
  done
fi
cp -n "$QWEN_DIRECTIONS_SRC"/*.pt "$QWEN_DIRECTIONS_RUN/" 2>/dev/null || true

echo "============================================================"
echo "Qwen + supra-threshold SMOKE TEST"
echo "Smoke output: $SMOKE_DIR"
echo "Smoke Qwen run: $SMOKE_RUN_DIR"
echo "Start: $(date)"
echo "============================================================"

# Smoke 1: Qwen Stage 02 — generate ONE attribution graph at L18
# (This is the highest-risk new step. If this works, the full 550-graph run should too.)
echo ""
echo "[smoke 1/6] Qwen Stage 02 — generate 1 attribution graph at L18 (~2-3 min)"
PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/02_run_attribution.py \
    --run-dir "$SMOKE_RUN_DIR" \
    --target-layer 18 \
    --skip-multi-graph \
    --n-prompts 1 \
    --save-graphs

# Smoke 2: Stage 02c pack
echo ""
echo "[smoke 2/6] Qwen Stage 02c — pack 1 graph into .json.gz"
PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/02c_pack_graphs.py \
    --run-dir "$SMOKE_RUN_DIR"

# Smoke 3: Qwen 0a linearization on the single packed graph
echo ""
echo "[smoke 3/6] Qwen 0a linearization (n=1)"
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_linearization_decomposition.py \
    --graph-data-dir "$SMOKE_RUN_DIR/graph_data" \
    --out-dir "$SMOKE_DIR" \
    --n-prompts 1 \
    --mode single
mv "$SMOKE_DIR/linearization_decomposition.json" "$SMOKE_DIR/smoke_qwen_decomp.json"

# Smoke 4: Qwen 0b edge ablation (1 variant × 1 prompt)
echo ""
echo "[smoke 4/6] Qwen 0b edge ablation, ablate_all_edges only, 1 prompt"
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime_qwen.py \
    --decomposition "$SMOKE_DIR/smoke_qwen_decomp.json" \
    --rhat-path "$QWEN_DIRECTIONS_RUN/layer_18.pt" \
    --variants "ablate_all_edges" \
    --max-prompts 1 \
    --out "$SMOKE_DIR/smoke_qwen_edge_ablation.json"

# Smoke 5: Qwen direction sweep @ L18 all positions, 3 prompts, 2 coefficients
echo ""
echo "[smoke 5/6] Qwen direction sweep @ L18 all positions, 3 prompts, coeffs={0.005, 1.0}"
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep_qwen.py \
    --directions-dir "$QWEN_DIRECTIONS_RUN" \
    --layers 18 \
    --position-mode all \
    --coefficients "0.005,1.0" \
    --max-prompts 3 \
    --out "$SMOKE_DIR/smoke_qwen_direction_all.json"

# Smoke 6: Gemma supra-threshold 5x, 3 prompts
echo ""
echo "[smoke 6/6] Gemma supra-threshold 5x, 3 prompts"
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py \
    --variants "ablate_all_edges_5x" \
    --max-prompts 3 \
    --out "$SMOKE_DIR/smoke_gemma_supra_5x.json"

# Sanity checks
echo ""
echo "============================================================"
echo "Smoke test sanity checks"
echo "============================================================"
SMOKE_DIR="$SMOKE_DIR" SMOKE_RUN_DIR="$SMOKE_RUN_DIR" python3 <<'PYEOF'
import json, os
from pathlib import Path
smoke_dir = Path(os.environ["SMOKE_DIR"])
smoke_run = Path(os.environ["SMOKE_RUN_DIR"])

# Check Stage 02 + 02c produced something
n_packed = len(list((smoke_run / "graph_data").glob("*.json.gz"))) if (smoke_run / "graph_data").exists() else 0
print(f"\n  Stage 02+02c packed graphs in {smoke_run}: {n_packed} (expected: 11 for 1 prompt × 11 conditions)")
if n_packed == 0:
    print("  WARNING: Stage 02 produced no packed graphs. Check the smoke log.")

# Decomposition
decomp = smoke_dir / "smoke_qwen_decomp.json"
if decomp.exists():
    d = json.loads(decomp.read_text())
    n_recs = len(d.get("per_prompt", []))
    print(f"\n  Decomposition records: {n_recs}")
    if n_recs:
        r = d["per_prompt"][0]
        print(f"    sample: prompt={r['prompt_idx']} cond={r['condition']}")
        print(f"            all_signed={r.get('all_signed', 'n/a'):.2f}")

# Qwen edge ablation
edge = smoke_dir / "smoke_qwen_edge_ablation.json"
if edge.exists():
    d = json.loads(edge.read_text())
    for variant, recs in d["per_variant"].items():
        bare = [r for r in recs if r["condition"] == "bare"]
        n_comply = sum(1 for r in bare if r["classification"] == "COMPLY")
        print(f"\n  Qwen 0b {variant} on n=1 bare: COMPLY {n_comply}/{len(bare)}")

# Qwen direction sweep
ds = smoke_dir / "smoke_qwen_direction_all.json"
if ds.exists():
    d = json.loads(ds.read_text())
    for lk, lb in d["per_layer"].items():
        for ck, recs in lb["per_coefficient"].items():
            bare = [r for r in recs if r["condition"] == "bare"]
            n_comply = sum(1 for r in bare if r["classification"] == "COMPLY")
            print(f"  Qwen sweep {lk} {ck}: bare->COMPLY {n_comply}/{len(bare)}")

# Gemma supra
gs = smoke_dir / "smoke_gemma_supra_5x.json"
if gs.exists():
    d = json.loads(gs.read_text())
    for variant, recs in d["per_variant"].items():
        bare = [r for r in recs if r["condition"] == "bare"]
        n_comply = sum(1 for r in bare if r["classification"] == "COMPLY")
        print(f"  Gemma {variant}: bare->COMPLY {n_comply}/{len(bare)}")

print("""
Expected at this scale:
  - Stage 02+02c packed: 11 .json.gz files (1 prompt × 11 conditions)
  - Decomposition records: 11
  - Qwen 0b ablate_all_edges: 0-1/1 COMPLY (small edit, mostly stays refuse)
  - Qwen sweep coeff=1.0 bare: 2-3/3 COMPLY (Arditi-style flips most)
  - Qwen sweep coeff=0.005 bare: 0-1/3 COMPLY (weak edit)
  - Gemma supra 5x: 0-1/3 COMPLY (effective coeff ~0.025, near baseline)

If Stage 02 produced 0 graphs OR Qwen sweep coeff=1.0 shows 0/3 flip,
STOP and investigate before launching the ~32 h full run.
""")
PYEOF

echo ""
echo "[smoke] DONE. If sanity above looks OK, launch full:"
echo "[smoke]   bash scripts/emnlp_perm_edit/runpod_qwen_and_suprathreshold.sh"
