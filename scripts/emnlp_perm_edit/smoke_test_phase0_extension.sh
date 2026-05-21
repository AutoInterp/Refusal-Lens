#!/usr/bin/env bash
# Quick smoke test for the Phase 0 extension (direction-sweep 2x2 + coeff
# sweep + layer locator).
#
# Runs all 4 new code paths against a 3-prompt subset so we can verify:
#   1. fp32 model load + r_hat load works on this pod
#   2. The new position_mode="last_prompt_only" hook fires without crashing
#   3. The --layers arg correctly attaches hook at a non-L15 layer
#   4. All drivers write valid output JSON
#   5. Sanity check: at coeff=1.0 (Arditi magnitude), all-positions should
#      produce ~100% flip on the bare-refuse condition (matches Stage 06).
#      At pos=-2 only, we don't know the answer — that's the Cell B unknown.
#
# Wall: ~8-10 min on H100 SXM fp32.
#
# Usage (from RunPod terminal at /workspace/Refusal-Lens):
#   bash scripts/emnlp_perm_edit/smoke_test_phase0_extension.sh

set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
SMOKE_DIR="$ROOT/data/results/emnlp_perm_edit/phase0_controllability/smoke_extension"
mkdir -p "$SMOKE_DIR"

cd "$ROOT"
if [[ ! -f "scripts/emnlp_perm_edit/00_direction_intervention_sweep.py" ]]; then
  echo "ERROR: expected to run from Refusal-Lens repo root. cwd=$(pwd)"
  exit 1
fi

# Activate venv if present (mirrors runpod_phase0_all.sh)
if [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

echo "============================================================"
echo "Phase 0 extension SMOKE TEST"
echo "Repo root: $ROOT"
echo "Smoke output: $SMOKE_DIR"
echo "Start: $(date)"
echo "============================================================"

# Smoke 1: direction sweep @ L15 all positions, 3 prompts, 2 coefficients (0.005 + 1.0)
echo ""
echo "[smoke 1/4] direction sweep @ L15 all positions, 3 prompts, coeffs={0.005, 1.0}"
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep.py \
    --layers 15 \
    --position-mode all \
    --coefficients "0.005,1.0" \
    --max-prompts 3 \
    --out "$SMOKE_DIR/smoke_direction_all.json"

# Smoke 2: direction sweep @ L15 pos=-2 only, 3 prompts, 2 coefficients
echo ""
echo "[smoke 2/4] direction sweep @ L15 pos=-2 only, 3 prompts, coeffs={0.005, 1.0}"
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep.py \
    --layers 15 \
    --position-mode last_prompt_only \
    --coefficients "0.005,1.0" \
    --max-prompts 3 \
    --out "$SMOKE_DIR/smoke_direction_pos2.json"

# Smoke 3: edge ablation @ pos=-2, 3 prompts, ablate_all_edges only
echo ""
echo "[smoke 3/4] edge ablation @ pos=-2, 3 prompts, ablate_all_edges only"
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_edge_ablation_runtime.py \
    --position-mode last_prompt_only \
    --variants "ablate_all_edges" \
    --max-prompts 3 \
    --out "$SMOKE_DIR/smoke_edge_pos2.json"

# Smoke 4: layer locator @ pos=-2 coeff=1.0, two non-L15 layers (L9 + L18), 3 prompts
echo ""
echo "[smoke 4/4] layer locator @ pos=-2 coeff=1.0, layers={9, 18}, 3 prompts"
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_direction_intervention_sweep.py \
    --layers "9,18" \
    --position-mode last_prompt_only \
    --coefficients "1.0" \
    --max-prompts 3 \
    --out "$SMOKE_DIR/smoke_layer_locator.json"

# Sanity checks
echo ""
echo "============================================================"
echo "Smoke test sanity checks"
echo "============================================================"
SMOKE_DIR="$SMOKE_DIR" python3 <<'PYEOF'
import json
import os
from pathlib import Path

smoke_dir = Path(os.environ["SMOKE_DIR"])

def summarize_direction(path, label):
    d = json.loads(Path(path).read_text())
    print(f"\n  {label}: {path}")
    print(f"  position_mode = {d['metadata']['position_mode']}, layers = {d['metadata']['layers']}")
    for layer_key, layer_block in d["per_layer"].items():
        for coeff_key, recs in layer_block["per_coefficient"].items():
            bare = [r for r in recs if r["condition"] == "bare"]
            jb   = [r for r in recs if r["condition"].startswith("jb_")]
            n_bare_comply = sum(1 for r in bare if r["classification"] == "COMPLY")
            n_jb_comply   = sum(1 for r in jb if r["classification"] == "COMPLY")
            print(f"    {layer_key} {coeff_key:14s}  bare->COMPLY: {n_bare_comply}/{len(bare)}   "
                  f"jb->COMPLY: {n_jb_comply}/{len(jb)}")

def summarize_edge(path, label):
    d = json.loads(Path(path).read_text())
    print(f"\n  {label}: {path}")
    print(f"  position_mode = {d['metadata'].get('position_mode', '?')}")
    for variant, recs in d["per_variant"].items():
        bare = [r for r in recs if r["condition"] == "bare"]
        jb   = [r for r in recs if r["condition"].startswith("jb_")]
        n_bare_comply = sum(1 for r in bare if r["classification"] == "COMPLY")
        n_jb_comply   = sum(1 for r in jb if r["classification"] == "COMPLY")
        print(f"    {variant:22s} bare->COMPLY: {n_bare_comply}/{len(bare)}   "
              f"jb->COMPLY: {n_jb_comply}/{len(jb)}")

summarize_direction(smoke_dir / "smoke_direction_all.json", "L15 direction sweep @ all positions")
summarize_direction(smoke_dir / "smoke_direction_pos2.json", "L15 direction sweep @ pos=-2")
summarize_edge(smoke_dir / "smoke_edge_pos2.json", "edge ablation @ pos=-2")
summarize_direction(smoke_dir / "smoke_layer_locator.json", "layer locator @ pos=-2 coeff=1.0")

print("""
Expected at this scale (3 prompts, only loose sanity):
  - L15 direction_all @ coeff=1.0  : bare-refuse should mostly flip to COMPLY
    (Stage 06 reports 98% on n=50; with n=3 expect 2-3/3).
  - L15 direction_pos2 @ coeff=1.0 : Cell B unknown — this is the actual
    experiment. Could be anywhere from 0/3 to 3/3.
  - direction_*       @ coeff=0.005: weak; mostly NO change vs baseline.
  - edge_pos2 ablate_all_edges     : like the existing all-positions version,
    expect mostly NO flip (Phase 0 reported 6% on n=50).
  - layer_locator L9/L18 @ coeff=1.0: unknown — these are the locator points.
    Mainly we're verifying the --layers arg correctly targets non-L15 modules
    without crashing. Any classification (even 0/3 flip) means the wiring works.

If L15 direction_all @ coeff=1.0 shows ALL refusing on bare (0/3 flip),
something is broken (hook not firing, wrong layer, wrong direction).
Investigate before launching the full run.
""")
PYEOF

echo ""
echo "[smoke] DONE. Inspect outputs in $SMOKE_DIR"
echo "[smoke] If sanity check above looks reasonable, launch full run:"
echo "[smoke]   bash scripts/emnlp_perm_edit/runpod_phase0_extension.sh"
