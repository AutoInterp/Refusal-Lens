#!/usr/bin/env bash
# RunPod launcher — Qwen subcircuit identification (07) + ablation (08) + Top-K sparsity sweep.
#
# Design spec: docs/superpowers/specs/2026-06-01-qwen-subcircuits-topk-design.md
#
# Steps (default "reuse" path — packed L18 graphs already on HF):
#   CPU  1. rebuild attribution index from packed graphs   (skips Stage 02 GPU re-pay)
#   CPU  2. Stage 04 --skip-download                        (set bookkeeping, no labels)
#   CPU  3. Stage 07 --graph-mode single                    (rule-based subcircuits)
#   GPU  4. Stage 08 subcircuit ablation                    (ReplacementModel, ~8.5 h)
#   GPU  5. Top-K sweep, zero mechanism (features)          (ReplacementModel, ~7 h)
#   GPU  6. Top-K sweep, proxy mechanism (features)         (plain model fp32, ~3 h)
#   GPU  7. Top-K sweep, proxy mechanism (edges)            (plain model fp32, ~3 h)
#   CPU  8. aggregate -> Pareto curves + report + frontend subcircuits
#
# Estimated wall on H100 SXM 80GB: ~22 h ± 20% (~$65-80 at $2.99/h).
# REGEN_UPSTREAM=1 additionally re-runs Stage 01 + 02 + 02b + 02c first
# (~+14 h, +$40) — only needed for a fully self-contained reproduction.
#
# Usage:
#   bash scripts/emnlp_perm_edit/runpod_qwen_subcircuits.sh          # tmux-detached
#   DRY_RUN=1 NO_TMUX=1 bash scripts/emnlp_perm_edit/runpod_qwen_subcircuits.sh   # print plan only
#
# Re-running the launcher resumes: rebuild/04/07 are idempotent-or-cheap, and
# 08 + sweeps use --resume / incremental saves.

set -uo pipefail

SESSION="qwen_subcircuits"
LOG="/tmp/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
ROOT="${ROOT:-$(pwd)}"
RUN_NAME="${RUN_NAME:-run_emnlp_qwen_L18_20260522}"
QWEN_RUN="$ROOT/data/results/pipeline_runs_qwen/$RUN_NAME"
GRAPH_DIR="$QWEN_RUN/graph_data"
OUT_DIR="$ROOT/data/results/emnlp_perm_edit/qwen_subcircuits"
HF_DATASET="${HF_DATASET:-moon70/refusal-lens-graphs}"
K_VALUES="${K_VALUES:-1,3,5,10,25,50,100,250}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
# All 5 JB classes + both controls (config default only covers 3 classes).
SUBCIRCUITS="${SUBCIRCUITS:-universal_refusal_core,ctrl_shared_refusal,jb_fiction_specific_vs_ctrl,jb_roleplay_specific_vs_ctrl,jb_analytical_specific_vs_ctrl,jb_completion_specific_vs_ctrl,jb_cognitive_reframe_specific_vs_ctrl}"
REGEN_UPSTREAM="${REGEN_UPSTREAM:-0}"
DRY_RUN="${DRY_RUN:-0}"
RHAT_PATH="$QWEN_RUN/01_direction/positions_L18/pos_-1_unnormalized.pt"
DONE_FILE="$OUT_DIR/.QWEN_SUBCIRCUITS_DONE"
FAIL_FILE="$OUT_DIR/.QWEN_SUBCIRCUITS_STEP_FAILED.txt"

echo "============================================================"
echo "Qwen subcircuits + Top-K sweep launcher"
echo "Run dir:    $QWEN_RUN"
echo "Out dir:    $OUT_DIR"
echo "K values:   $K_VALUES   max_new_tokens: $MAX_NEW_TOKENS"
echo "REGEN_UPSTREAM=$REGEN_UPSTREAM  DRY_RUN=$DRY_RUN"
echo "============================================================"

# --- Self-relaunch into tmux ---
if [[ -z "${TMUX:-}" ]] && [[ "${NO_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' exists. Kill: tmux kill-session -t $SESSION"
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    "K_VALUES='$K_VALUES' MAX_NEW_TOKENS='$MAX_NEW_TOKENS' SUBCIRCUITS='$SUBCIRCUITS' REGEN_UPSTREAM='$REGEN_UPSTREAM' DRY_RUN='$DRY_RUN' RUN_NAME='$RUN_NAME' HF_DATASET='$HF_DATASET' HF_TOKEN='${HF_TOKEN:-}' HF_HOME='${HF_HOME:-}' bash $0 2>&1 | tee $LOG"
  echo "Detached tmux session '$SESSION'. Reattach: tmux attach -t $SESSION"
  echo "Watch log:  tail -f $LOG"
  echo "Watcher (2nd window): bash scripts/emnlp_perm_edit/watch_and_commit_qwen_subcircuits.sh"
  exit 0
fi

cd "$ROOT"
mkdir -p "$OUT_DIR"
rm -f "$DONE_FILE" "$FAIL_FILE"

declare -a STEP_NAMES STEP_RESULTS
run_step() {
  local name="$1"; shift
  echo ""
  echo "########## STEP: $name ##########"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] would run: $*"
    STEP_NAMES+=("$name"); STEP_RESULTS+=("DRY")
    return 0
  fi
  local t0=$(date +%s)
  if "$@"; then
    local t1=$(date +%s)
    echo "########## STEP $name OK ($((t1-t0))s) ##########"
    STEP_NAMES+=("$name"); STEP_RESULTS+=("OK")
  else
    local rc=$?
    local t1=$(date +%s)
    echo "########## STEP $name FAILED (exit $rc, $((t1-t0))s) — continuing ##########"
    echo "$name (exit $rc)" >> "$FAIL_FILE"
    STEP_NAMES+=("$name"); STEP_RESULTS+=("FAILED")
  fi
}

# --- Step A: Environment ---
echo ""
echo "=== Step A: Env ==="
if [[ "$DRY_RUN" != "1" ]]; then
  if [[ ! -d ".venv" ]]; then python3 -m venv .venv; fi
  source .venv/bin/activate
  pip install --upgrade pip -q
  pip install -e . -q
  pip install -e ./vendor/circuit-tracer -q
  python3 -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}')" || {
    echo "FATAL: CUDA not available." | tee -a "$FAIL_FILE"; touch "$DONE_FILE"; exit 1
  }
else
  echo "[DRY_RUN] skipping venv setup + CUDA check"
fi

# --- Step B: Prerequisites ---
echo ""
echo "=== Step B: Prerequisites ==="

# B1: packed graphs (pull from HF if absent; local 05_frontend copy also accepted)
if [[ ! -f "$GRAPH_DIR/000_bare_single.json.gz" ]]; then
  LOCAL_FE="$ROOT/data/results/pipeline_runs/$RUN_NAME/05_frontend/graph_data"
  if [[ -f "$LOCAL_FE/000_bare_single.json.gz" ]]; then
    echo "  Staging graphs from local frontend copy..."
    mkdir -p "$GRAPH_DIR"
    cp "$LOCAL_FE"/*.json.gz "$GRAPH_DIR/" 2>/dev/null
    cp "$LOCAL_FE/graph-metadata.json" "$GRAPH_DIR/" 2>/dev/null || true
  else
    # Staging (~180MB) runs even under DRY_RUN — it's prereq data, not GPU work,
    # and the smoke test needs the graphs before the first real launch.
    echo "  Pulling packed graphs from HF ($HF_DATASET)..."
    python3 - <<PYEOF
from huggingface_hub import snapshot_download
from pathlib import Path
import shutil
snap = snapshot_download(repo_id="$HF_DATASET", repo_type="dataset",
                         allow_patterns=["runs/$RUN_NAME/graph_data/*"])
src = Path(snap) / "runs" / "$RUN_NAME" / "graph_data"
dst = Path("$GRAPH_DIR"); dst.mkdir(parents=True, exist_ok=True)
n = 0
for f in src.iterdir():
    shutil.copy2(f, dst / f.name); n += 1
print(f"  staged {n} graph files")
PYEOF
  fi
fi
N_GRAPHS=$(ls "$GRAPH_DIR"/*.json.gz 2>/dev/null | wc -l)
echo "  Packed graphs present: $N_GRAPHS (expect 550)"
if [[ "$N_GRAPHS" -lt 550 ]] && [[ "$DRY_RUN" != "1" ]] && [[ "$REGEN_UPSTREAM" != "1" ]]; then
  echo "FATAL: incomplete graph set ($N_GRAPHS/550)." | tee -a "$FAIL_FILE"
  touch "$DONE_FILE"; exit 1
fi

# B2: direction files (needed by proxy sweep steps) — Path-A git-show fallback
if [[ ! -f "$RHAT_PATH" ]]; then
  echo "  Pulling direction files from temp/gemma-vs-qwen-pipeline..."
  git fetch origin temp/gemma-vs-qwen-pipeline 2>/dev/null || true
  mkdir -p "$QWEN_RUN/01_direction/positions_L18" "$QWEN_RUN/01_direction/directions"
  SRC_PREFIX="data/results/pipeline_runs_qwen/run_20260502_154423/01_direction"
  git show "origin/temp/gemma-vs-qwen-pipeline:$SRC_PREFIX/direction_metadata.json" \
    > "$QWEN_RUN/01_direction/direction_metadata.json" 2>/dev/null || true
  git show "origin/temp/gemma-vs-qwen-pipeline:$SRC_PREFIX/directions/layer_18.pt" \
    > "$QWEN_RUN/01_direction/directions/layer_18.pt" 2>/dev/null || true
  for P in -1 -2 -3 -4 -5; do
    git show "origin/temp/gemma-vs-qwen-pipeline:$SRC_PREFIX/positions_L18/pos_${P}_unnormalized.pt" \
      > "$QWEN_RUN/01_direction/positions_L18/pos_${P}_unnormalized.pt" 2>/dev/null || true
  done
fi
if [[ -f "$RHAT_PATH" ]]; then
  echo "  Direction r_unnorm present: $RHAT_PATH"
elif [[ "$DRY_RUN" == "1" ]]; then
  echo "  [DRY_RUN] direction file missing — would be fetched via git show"
else
  echo "FATAL: could not stage $RHAT_PATH" | tee -a "$FAIL_FILE"; touch "$DONE_FILE"; exit 1
fi

# B3: dataset
if [[ ! -f "dataset/refusal_lens_controlled_dataset.json" ]]; then
  echo "FATAL: dataset/refusal_lens_controlled_dataset.json missing" | tee -a "$FAIL_FILE"
  [[ "$DRY_RUN" == "1" ]] || { touch "$DONE_FILE"; exit 1; }
fi
echo "  Prerequisites OK."

# --- Optional: full upstream regeneration (REGEN_UPSTREAM=1) ---
if [[ "$REGEN_UPSTREAM" == "1" ]]; then
  run_step "qwen_stage01_direction" \
    bash -c "PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/01_compute_direction.py \
        --run-dir '$QWEN_RUN' --recompute"
  run_step "qwen_stage02_attribution" \
    bash -c "PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/02_run_attribution.py \
        --run-dir '$QWEN_RUN' --target-layer 18 --skip-multi-graph --n-prompts 50 --save-graphs --resume"
  run_step "qwen_stage02b_stats" \
    bash -c "PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/02b_statistical_analysis.py \
        --run-dir '$QWEN_RUN'"
  run_step "qwen_stage02c_pack" \
    bash -c "PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/02c_pack_graphs.py \
        --run-dir '$QWEN_RUN'"
fi

# --- STEP 1: rebuild attribution index (CPU; auto-skips if Stage 02 output exists) ---
run_step "rebuild_attribution_index" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/qwen_rebuild_attribution_index.py \
      --graph-data-dir '$GRAPH_DIR' --run-dir '$QWEN_RUN'"

# --- STEP 2: Stage 04 set bookkeeping (CPU, no HF label downloads) ---
run_step "stage04_bookkeeping" \
  bash -c "PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/04_label_features.py \
      --run-dir '$QWEN_RUN' --skip-download"

# --- STEP 3: Stage 07 subcircuit identification (CPU) ---
run_step "stage07_subcircuits" \
  bash -c "PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/07_identify_subcircuits.py \
      --run-dir '$QWEN_RUN' --graph-mode single"

# --- STEP 4: Stage 08 subcircuit ablation (GPU, ReplacementModel) ---
run_step "stage08_subcircuit_ablation" \
  bash -c "PYTHONPATH=scripts/pipeline_qwen python3 scripts/pipeline_qwen/08_ablate_subcircuits.py \
      --run-dir '$QWEN_RUN' --graph-mode single --positions both \
      --subcircuits '$SUBCIRCUITS' --backend transformerlens \
      --max-new-tokens $MAX_NEW_TOKENS --resume"

# --- STEP 5: Top-K sweep — zero mechanism, features (GPU, ReplacementModel) ---
run_step "topk_zero_features" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_circuit_sweep_qwen.py \
      --mechanism zero --source features --backend transformerlens \
      --graph-data-dir '$GRAPH_DIR' --k-values '$K_VALUES' \
      --max-new-tokens $MAX_NEW_TOKENS --resume \
      --out '$OUT_DIR/topk_sweep_zero_features.json'"

# --- STEP 6: Top-K sweep — proxy mechanism, features (GPU, plain model fp32) ---
run_step "topk_proxy_features" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_circuit_sweep_qwen.py \
      --mechanism proxy --source features \
      --graph-data-dir '$GRAPH_DIR' --rhat-path '$RHAT_PATH' --k-values '$K_VALUES' \
      --max-new-tokens $MAX_NEW_TOKENS --resume \
      --out '$OUT_DIR/topk_sweep_proxy_features.json'"

# --- STEP 7: Top-K sweep — proxy mechanism, edges (GPU; reuses proxy baseline) ---
run_step "topk_proxy_edges" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_circuit_sweep_qwen.py \
      --mechanism proxy --source edges \
      --graph-data-dir '$GRAPH_DIR' --rhat-path '$RHAT_PATH' --k-values '$K_VALUES' \
      --max-new-tokens $MAX_NEW_TOKENS --resume --skip-baseline \
      --out '$OUT_DIR/topk_sweep_proxy_edges.json'"

# --- STEP 8: Aggregate (CPU) ---
run_step "aggregate_report" \
  bash -c "PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/qwen_subcircuits_aggregate.py \
      --run-dir '$QWEN_RUN' --out-dir '$OUT_DIR'"

# --- Final summary ---
echo ""
echo "============================================================"
echo "QWEN SUBCIRCUITS RUN COMPLETE"
echo "============================================================"
for i in "${!STEP_NAMES[@]}"; do
  echo "  ${STEP_NAMES[$i]} : ${STEP_RESULTS[$i]}"
done
if [[ -f "$FAIL_FILE" ]]; then
  echo ""
  echo "Failures:"; cat "$FAIL_FILE"
fi
echo ""
echo "Result files:"
ls -la "$OUT_DIR"/*.json "$OUT_DIR"/*.md 2>/dev/null
ls -la "$QWEN_RUN/07_subcircuits"/subcircuits*.json "$QWEN_RUN/08_ablation"/ablation_summary.json 2>/dev/null

touch "$DONE_FILE"
echo "Touched $DONE_FILE"
echo "Watcher: bash scripts/emnlp_perm_edit/watch_and_commit_qwen_subcircuits.sh"
