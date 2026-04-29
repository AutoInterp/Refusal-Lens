#!/usr/bin/env bash
# scripts/pipeline/run_p7.sh
#
# P7 — full pipeline rerun on RunPod (smoke -> full -> push), tmux-persistent.
#
# Each invocation creates a fresh, fully self-contained run directory: NO
# stages are reused or symlinked from prior runs. Both the smoke and the full
# run recompute Stage 01 (refusal directions) and Stage 06 (causal
# intervention) from scratch — required for end-to-end validation of the
# circuit-tracer measurement_hook patch.
#
# Default: smoke (3 prompts) -> verdict gate -> full (50 prompts) -> HF push
# (raw .pt + packed JSON.gz + run metadata). Self-relaunches into a detached
# tmux session so it survives SSH disconnects.
#
# Stage order in BOTH smoke and full:
#   01 (direction) -> 02 (attribution) -> 02b (stats) -> 03×2 (verify multi+single)
#   -> 04 (labels) -> 02c (pack) -> 07 (subcircuits) -> 06 (causal) -> 08 (ablate)
#
# Usage on RunPod:
#   export HF_TOKEN=hf_...                   # for HF push (or `huggingface-cli login`)
#   export GHP_TOKEN=ghp_...                 # only if --git-push-results
#   cd /workspace/Refusal-Lens
#   git pull                                 # script does NOT auto-pull
#   pip install -e ./vendor/circuit-tracer   # one-time, if measurement_hook missing
#   bash scripts/pipeline/run_p7.sh          # default: smoke+full+push, in tmux
#
# Common variants:
#   bash scripts/pipeline/run_p7.sh --mode smoke           # smoke only (~1.5 h)
#   bash scripts/pipeline/run_p7.sh --mode full            # skip smoke (~22-26 h)
#   bash scripts/pipeline/run_p7.sh --no-tmux              # current shell
#   bash scripts/pipeline/run_p7.sh --git-push-results     # commit small results
#   bash scripts/pipeline/run_p7.sh --no-push-raw          # skip raw .pt push
#
# Logs / markers:
#   /tmp/p7_pipeline_<ts>.log      main log (also at /tmp/p7_pipeline.log via symlink)
#   <run_dir>/_<stage>.log         per-stage logs
#   <run_dir>/.P7_DONE             written on full success
#   <run_dir>/.P7_FAIL             written with phase/step on failure
#
# Tmux ops:
#   Attach to live output:    tmux attach -t p7
#   Detach without stopping:  Ctrl-b d
#   Kill the run:             tmux kill-session -t p7

set -euo pipefail

# ============================================================
# Arg parse
# ============================================================
ORIG_ARGS=("$@")

MODE=both                          # smoke | full | both
SMOKE_PROMPTS=3
FULL_PROMPTS=50
DATASET_REPO=moon70/refusal-lens-graphs
PUSH_RAW=1
PUSH_PACKED=1
PUSH_RUN_META=1
GIT_PUSH_RESULTS=0
POSITIONS=both                     # all | anchors | both
SUBCIRCUITS_FILE=subcircuits_k50_f50.json
USE_TMUX=1
TMUX_SESSION=p7
BATCH_SIZE=256
RUN_ROOT=data/results/pipeline_runs
BRANCH=l15-refactor

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE=$2; shift 2;;
        --smoke-prompts) SMOKE_PROMPTS=$2; shift 2;;
        --full-prompts) FULL_PROMPTS=$2; shift 2;;
        --dataset-repo) DATASET_REPO=$2; shift 2;;
        --no-push-raw) PUSH_RAW=0; shift;;
        --no-push-packed) PUSH_PACKED=0; shift;;
        --no-push-run-meta) PUSH_RUN_META=0; shift;;
        --git-push-results) GIT_PUSH_RESULTS=1; shift;;
        --positions) POSITIONS=$2; shift 2;;
        --subcircuits-file) SUBCIRCUITS_FILE=$2; shift 2;;
        --no-tmux) USE_TMUX=0; shift;;
        --session) TMUX_SESSION=$2; shift 2;;
        --batch-size) BATCH_SIZE=$2; shift 2;;
        --branch) BRANCH=$2; shift 2;;
        -h|--help) awk 'NR==1{print;next} /^[^#]/{exit} {print}' "$0"; exit 0;;
        *) echo "ERROR: unknown arg: $1"; exit 2;;
    esac
done

case "$MODE" in
    smoke|full|both) ;;
    *) echo "ERROR: --mode must be smoke|full|both, got '$MODE'"; exit 2;;
esac
case "$POSITIONS" in
    all|anchors|both) ;;
    *) echo "ERROR: --positions must be all|anchors|both, got '$POSITIONS'"; exit 2;;
esac

# ============================================================
# Tmux self-relaunch
# ============================================================
if [[ $USE_TMUX -eq 1 ]] && [[ -z "${TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux not installed. Either install (apt-get install -y tmux) or pass --no-tmux."
        exit 1
    fi
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "ERROR: tmux session '$TMUX_SESSION' already exists."
        echo "  Attach with: tmux attach -t $TMUX_SESSION"
        echo "  Or pass --session OTHER_NAME to use a different one."
        exit 1
    fi

    SCRIPT_PATH=$(readlink -f "$0" 2>/dev/null || realpath "$0")
    REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)

    inner_args=()
    for a in "${ORIG_ARGS[@]}"; do
        [[ "$a" == "--no-tmux" ]] && continue
        inner_args+=("$a")
    done
    inner_args+=("--no-tmux")

    hft=$(printf '%q' "${HF_TOKEN:-}")
    ghpt=$(printf '%q' "${GHP_TOKEN:-}")
    hfh=$(printf '%q' "${HF_HOME:-}")
    quoted_inner=$(printf '%q ' "${inner_args[@]}")

    tmux new-session -d -s "$TMUX_SESSION" \
        "export HF_TOKEN=$hft GHP_TOKEN=$ghpt HF_HOME=$hfh; cd $(printf '%q' "$REPO_ROOT") && exec bash $(printf '%q' "$SCRIPT_PATH") $quoted_inner"

    echo "============================================================"
    echo "P7 pipeline launched in tmux session '$TMUX_SESSION'"
    echo "  Mode:           $MODE"
    echo "  Smoke prompts:  $SMOKE_PROMPTS"
    echo "  Full prompts:   $FULL_PROMPTS"
    echo "  HF dataset:     $DATASET_REPO"
    echo "  HF push (raw):  $([[ $PUSH_RAW -eq 1 ]] && echo yes || echo no)"
    echo "  Git push:       $([[ $GIT_PUSH_RESULTS -eq 1 ]] && echo yes || echo no)"
    echo "============================================================"
    echo
    echo "Attach to live output:   tmux attach -t $TMUX_SESSION"
    echo "Detach without stopping: Ctrl-b d"
    echo "Tail log instead:        tail -f /tmp/p7_pipeline.log"
    echo "Kill the run:            tmux kill-session -t $TMUX_SESSION"
    exit 0
fi

# ============================================================
# In-process body (we are now inside tmux, or --no-tmux)
# ============================================================
LOG=/tmp/p7_pipeline_$(date +%Y%m%d_%H%M%S).log
ln -sfn "$LOG" /tmp/p7_pipeline.log
exec > >(tee -a "$LOG") 2>&1

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

PHASE="init"
STEP="init"
RUN_DIR_ACTIVE=""

trap '
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo
        echo "============================================================"
        echo "[$(date)] !! FAILED at $PHASE / $STEP (exit $rc)"
        echo "============================================================"
        if [[ -n "${RUN_DIR_ACTIVE:-}" ]]; then
            echo "$PHASE/$STEP exit=$rc at $(date)" > "$RUN_DIR_ACTIVE/.P7_FAIL"
            echo "FAIL marker: $RUN_DIR_ACTIVE/.P7_FAIL"
            sl="$RUN_DIR_ACTIVE/_${STEP}.log"
            if [[ -f "$sl" ]]; then
                echo
                echo "Last 40 lines of $sl:"
                tail -n 40 "$sl"
            fi
        fi
        echo
        echo "Resume hints:"
        echo "  1. Inspect tmux scrollback:  tmux attach -t $TMUX_SESSION  (Ctrl-b [ to scroll)"
        echo "  2. Re-run from scratch:      bash scripts/pipeline/run_p7.sh --mode <smoke|full|both>"
        echo "  3. Per-stage logs:           ls $RUN_DIR_ACTIVE/_*.log"
    fi
' EXIT

run_stage() {
    local label=$1; shift
    STEP=$label
    local logfile="$RUN_DIR_ACTIVE/_${label}.log"
    echo
    echo "------------------------------------------------------------"
    echo "[$(date)] [$PHASE/$label]  $*"
    echo "------------------------------------------------------------"
    local t=$SECONDS
    PYTHONPATH=src "$@" 2>&1 | tee "$logfile"
    echo "[$(date)] [$PHASE/$label] DONE in $(( SECONDS - t ))s"
}

# ============================================================
# Preflight
# ============================================================
PHASE="preflight"
STEP="banner"

echo "============================================================"
echo "[$(date)] Refusal-Lens P7 pipeline"
echo "  Mode:                   $MODE"
echo "  Smoke prompts:          $SMOKE_PROMPTS"
echo "  Full prompts:           $FULL_PROMPTS"
echo "  Reuse / symlink:        none — every stage runs fresh in this run dir"
echo "  Stage 02 batch size:    $BATCH_SIZE"
echo "  Stage 08 positions:     $POSITIONS"
echo "  Stage 08 subcircuits:   $SUBCIRCUITS_FILE"
echo "  HF dataset:             $DATASET_REPO"
echo "  HF push raw .pt:        $([[ $PUSH_RAW -eq 1 ]] && echo yes || echo no)"
echo "  HF push packed JSON.gz: $([[ $PUSH_PACKED -eq 1 ]] && echo yes || echo no)"
echo "  HF push run meta:       $([[ $PUSH_RUN_META -eq 1 ]] && echo yes || echo no)"
echo "  Git push results:       $([[ $GIT_PUSH_RESULTS -eq 1 ]] && echo yes || echo no)"
[[ -n "${HF_TOKEN:-}" ]] && echo "  HF_TOKEN env:           set" || echo "  HF_TOKEN env:           NOT SET (will use ~/.cache/huggingface/token if logged in)"
[[ -n "${GHP_TOKEN:-}" ]] && echo "  GHP_TOKEN env:          set" || echo "  GHP_TOKEN env:          NOT SET"
echo "  Repo root:              $REPO_ROOT"
echo "  Branch:                 $BRANCH"
echo "  Log:                    $LOG"
echo "============================================================"

STEP="circuit_tracer"
echo "[$(date)] Verifying circuit-tracer install..."
python3 -c "
from circuit_tracer.attribution.attribute import attribute
import inspect, sys
ok = 'measurement_hook' in inspect.signature(attribute).parameters
print('  measurement_hook param available:', ok)
sys.exit(0 if ok else 1)
" || {
    echo "ERROR: circuit-tracer missing measurement_hook param."
    echo "  Fix: pip install -e ./vendor/circuit-tracer"
    exit 1
}

STEP="gpu"
python3 -c "
import torch, sys
ok = torch.cuda.is_available()
if ok:
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  GPU: {name} ({mem:.1f} GB)')
sys.exit(0 if ok else 1)
" || { echo "ERROR: no CUDA GPU available"; exit 1; }

STEP="auth_check"
if [[ $GIT_PUSH_RESULTS -eq 1 && -z "${GHP_TOKEN:-}" ]]; then
    echo "ERROR: --git-push-results requires GHP_TOKEN env var (set with contents:write scope on the repo)."
    exit 1
fi
if [[ ($PUSH_RAW -eq 1 || $PUSH_PACKED -eq 1 || $PUSH_RUN_META -eq 1) && -z "${HF_TOKEN:-}" ]]; then
    if [[ ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
        echo "WARNING: HF push enabled, no HF_TOKEN env var, and no cached token at"
        echo "         ${HF_HOME:-$HOME/.cache/huggingface}/token. Push steps will fail."
        echo "         Either: export HF_TOKEN=hf_..., run huggingface-cli login,"
        echo "         or pass --no-push-raw --no-push-packed --no-push-run-meta."
    fi
fi

mkdir -p "$RUN_ROOT"

# ============================================================
# SMOKE phase
# ============================================================
if [[ "$MODE" == "smoke" || "$MODE" == "both" ]]; then
    PHASE="smoke"
    SMOKE=$RUN_ROOT/full_smoke_$(date +%Y%m%d_%H%M%S)
    mkdir -p "$SMOKE"
    RUN_DIR_ACTIVE=$SMOKE

    echo
    echo "============================================================"
    echo "[$(date)] SMOKE PHASE — $SMOKE_PROMPTS-prompt end-to-end, all stages fresh"
    echo "  Smoke run dir: $SMOKE"
    echo "============================================================"

    run_stage "01_smoke" python3 scripts/pipeline/01_compute_direction.py \
        --run-dir "$SMOKE"

    run_stage "02_smoke" python3 scripts/pipeline/02_run_attribution.py \
        --run-dir "$SMOKE" --n-prompts $SMOKE_PROMPTS --batch-size $BATCH_SIZE \
        --save-graphs

    run_stage "03_smoke_multi" python3 scripts/pipeline/03_verify_attribution.py \
        --run-dir "$SMOKE" --graph-mode multi
    run_stage "03_smoke_single" python3 scripts/pipeline/03_verify_attribution.py \
        --run-dir "$SMOKE" --graph-mode single

    run_stage "02b_smoke" python3 scripts/pipeline/02b_statistical_analysis.py \
        --run-dir "$SMOKE"

    run_stage "04_smoke" python3 scripts/pipeline/04_label_features.py \
        --run-dir "$SMOKE"

    run_stage "02c_smoke" python3 scripts/pipeline/02c_pack_graphs.py \
        --run-dir "$SMOKE"

    run_stage "07_smoke" python3 scripts/pipeline/07_identify_subcircuits.py \
        --run-dir "$SMOKE"

    run_stage "06_smoke" python3 scripts/pipeline/06_causal_intervention.py \
        --run-dir "$SMOKE" --max-prompts $SMOKE_PROMPTS

    run_stage "08_smoke" python3 scripts/pipeline/08_ablate_subcircuits.py \
        --run-dir "$SMOKE" --max-prompts $SMOKE_PROMPTS --positions all \
        --max-new-tokens 80 \
        --skip-baseline \
        --subcircuits universal_refusal_core,jb_fiction_specific_vs_ctrl

    STEP="smoke_verdict"
    echo
    echo "============================================================"
    echo "[$(date)] SMOKE VERDICT"
    echo "============================================================"
    err=$(grep -EH "Traceback|RuntimeError|CUDA out of memory|AssertionError|ERROR:" "$SMOKE"/_*.log 2>/dev/null | head -30 || true)
    if [[ -n "$err" ]]; then
        echo "Errors found in smoke logs:"
        echo "$err"
        echo
        echo "Aborting before full run. Fix and re-run with --mode both (or --mode full)."
        exit 1
    fi
    grep -hE "DONE!" "$SMOKE"/_*.log 2>/dev/null | head -20 || true
    echo "+ smoke phase clean (no error markers in any stage log)"

    echo
    echo "Stage 03 hook-aware identity check (expect ratio in [1.5, 2.0]):"
    python3 -c "
import json, glob
for f in sorted(glob.glob('$SMOKE/03_verification/verification_results*.json')):
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f'  {f}: failed to parse ({e})')
        continue
    s = d.get('summary', {})
    r = s.get('attr_to_dot_ratio_mean')
    o_m = s.get('baseline_offset_mean')
    o_s = s.get('baseline_offset_std')
    hook = s.get('measurement_hook', '?')
    print(f'  {f.split(\"/\")[-1]}: hook={hook}, ratio={r}, baseline_mean={o_m}, baseline_std={o_s}')
" || true

    echo
    echo "Stage 07 sweep configs produced:"
    ls -1 "$SMOKE/07_subcircuits/" 2>/dev/null | grep -E "subcircuits.*json$" | head -10 || true

    echo "[$(date)] SMOKE DONE"
fi

if [[ "$MODE" == "smoke" ]]; then
    echo
    echo "============================================================"
    echo "[$(date)] Smoke-only mode complete. To launch full run:"
    echo "  bash scripts/pipeline/run_p7.sh --mode full"
    echo "============================================================"
    exit 0
fi

# ============================================================
# FULL phase
# ============================================================
PHASE="full"
RUN=$RUN_ROOT/run_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
RUN_DIR_ACTIVE=$RUN

echo
echo "============================================================"
echo "[$(date)] FULL PHASE — $FULL_PROMPTS prompts, all stages fresh"
echo "  Full run dir: $RUN"
echo "============================================================"

run_stage "01" python3 scripts/pipeline/01_compute_direction.py --run-dir "$RUN"

run_stage "02" python3 scripts/pipeline/02_run_attribution.py \
    --run-dir "$RUN" --n-prompts $FULL_PROMPTS --batch-size $BATCH_SIZE \
    --save-graphs

run_stage "02b" python3 scripts/pipeline/02b_statistical_analysis.py --run-dir "$RUN"
run_stage "03_multi"  python3 scripts/pipeline/03_verify_attribution.py --run-dir "$RUN" --graph-mode multi
run_stage "03_single" python3 scripts/pipeline/03_verify_attribution.py --run-dir "$RUN" --graph-mode single
run_stage "04"  python3 scripts/pipeline/04_label_features.py --run-dir "$RUN"
run_stage "02c" python3 scripts/pipeline/02c_pack_graphs.py    --run-dir "$RUN"
run_stage "07"  python3 scripts/pipeline/07_identify_subcircuits.py --run-dir "$RUN"

run_stage "06"  python3 scripts/pipeline/06_causal_intervention.py \
    --run-dir "$RUN" --max-prompts $FULL_PROMPTS

run_stage "08" python3 scripts/pipeline/08_ablate_subcircuits.py \
    --run-dir "$RUN" --positions $POSITIONS \
    --max-new-tokens 80 \
    --skip-baseline --resume --checkpoint-every 5 \
    --subcircuits-file $SUBCIRCUITS_FILE

# ============================================================
# PUSH phase
# ============================================================
PHASE="push"
echo
echo "============================================================"
echo "[$(date)] PUSH PHASE"
echo "============================================================"

if [[ $PUSH_RAW -eq 1 ]]; then
    run_stage "push_raw" python3 scripts/pipeline/push_raw_graphs.py \
        --run-dir "$RUN" --dataset-repo "$DATASET_REPO"
fi

if [[ $PUSH_PACKED -eq 1 ]]; then
    run_stage "push_packed" python3 scripts/pipeline/push_graph_data.py \
        --run-dir "$RUN" --source 02c --dataset-repo "$DATASET_REPO"
fi

if [[ $PUSH_RUN_META -eq 1 ]]; then
    run_stage "push_run_meta" python3 scripts/pipeline/push_run.py \
        --run-dir "$RUN" --dataset-repo "$DATASET_REPO" --skip-graphs
fi

if [[ $GIT_PUSH_RESULTS -eq 1 ]]; then
    PHASE="git_push"
    STEP="git_push"
    echo
    echo "------------------------------------------------------------"
    echo "[$(date)] Committing small results to git..."
    echo "------------------------------------------------------------"

    # Don't clobber existing git config
    git config user.email >/dev/null 2>&1 || git config user.email "${GIT_AUTHOR_EMAIL:-runpod@local}"
    git config user.name  >/dev/null 2>&1 || git config user.name  "${GIT_AUTHOR_NAME:-P7 runner}"

    U=$(git remote get-url origin | sed "s|https://|https://x-access-token:${GHP_TOKEN}@|")
    git fetch "$U" "$BRANCH"
    git checkout "$BRANCH"
    git pull --rebase "$U" "$BRANCH"

    # Only the small/JSON dirs. raw .pt and packed .json.gz are gitignored anyway,
    # but list explicitly so we never accidentally stage huge files.
    for d in 02b_stats 03_verification 04_labels 06_causal 07_subcircuits 08_ablation; do
        # -d follows symlinks; here Stage 06 is a real dir since we always run it.
        [[ -d "$RUN/$d" && ! -L "$RUN/$d" ]] && git add "$RUN/$d" || true
    done
    [[ -d "$RUN/02_attribution" ]] && find "$RUN/02_attribution" -maxdepth 1 -name "*.json" -exec git add {} + 2>/dev/null || true
    find "$RUN" -maxdepth 1 -name "*.json" -exec git add {} + 2>/dev/null || true

    if git diff --staged --quiet; then
        echo "No staged changes (likely all paths are gitignored). HF push has the data."
    else
        git commit -m "P7: full-pipeline rerun results in $(basename "$RUN")" \
                   -m "Run dir: $RUN" \
                   -m "Stage 02 prompts: $FULL_PROMPTS, batch: $BATCH_SIZE" \
                   -m "Stage 08 positions: $POSITIONS, subcircuits: $SUBCIRCUITS_FILE" \
                   -m "HF dataset: https://huggingface.co/datasets/$DATASET_REPO/tree/main/runs/$(basename "$RUN")"
        git push "$U" "$BRANCH"
        echo "+ pushed to $BRANCH"
    fi
fi

# ============================================================
# Summary
# ============================================================
PHASE="summary"
STEP="summary"

TOTAL_MIN=$(( SECONDS / 60 ))
DONE_FILE="$RUN/.P7_DONE"
cat > "$DONE_FILE" <<EOF
P7 pipeline completed
ended_at: $(date)
total_minutes: $TOTAL_MIN
mode: $MODE
smoke_prompts: $SMOKE_PROMPTS
full_prompts: $FULL_PROMPTS
run_dir: $RUN
hf_dataset: $DATASET_REPO/runs/$(basename "$RUN")
stage_08_positions: $POSITIONS
stage_08_subcircuits: $SUBCIRCUITS_FILE
git_pushed: $([[ $GIT_PUSH_RESULTS -eq 1 ]] && echo yes || echo no)
EOF

echo
echo "============================================================"
echo "[$(date)] P7 ALL DONE — total ${TOTAL_MIN} min"
echo "============================================================"
echo
echo "Results:"
echo "  Local run dir:  $RUN"
echo "  HF dataset URL: https://huggingface.co/datasets/$DATASET_REPO/tree/main/runs/$(basename "$RUN")"
echo "  DONE marker:    $DONE_FILE"
echo
echo "Headline files for NeurIPS write-up:"
echo "  Stage 03 (verification identity):"
ls -1 "$RUN/03_verification/"*.json 2>/dev/null | sed 's/^/    /' | head -5
echo "  Stage 07 (per-prompt subcircuits):"
ls -1 "$RUN/07_subcircuits/"*.json 2>/dev/null | sed 's/^/    /' | head -10
echo "  Stage 08 (comply-weighted ablation):"
ls -1 "$RUN/08_ablation/"*.json 2>/dev/null | sed 's/^/    /' | head -5
echo
echo "Quick numbers (per_ablation summary from Stage 08):"
python3 -c "
import json, sys
try:
    d = json.load(open('$RUN/08_ablation/ablation_summary.json'))
except Exception as e:
    print(f'  (ablation_summary.json not parseable: {e})')
    sys.exit(0)
pa = d.get('per_ablation', {})
for abl, blocks in pa.items():
    print(f'  {abl}:')
    for mode, body in blocks.get('positions', {}).items():
        w = body.get('weighted', {})
        b = body.get('bare_break_rate')
        jr = w.get('jb_weighted_recovery_rate')
        cr = w.get('ctrl_weighted_break_rate')
        print(f'    {mode:>8s}: bare_break={b}, jb_weighted_recovery={jr}, ctrl_weighted_break={cr}')
" 2>/dev/null || true
