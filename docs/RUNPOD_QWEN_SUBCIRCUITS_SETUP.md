# RunPod Setup — Qwen Subcircuits + Top-K Sweep Run

Step-by-step instructions for launching the Qwen Stage 07/08 + Top-K sparsity
sweep on a RunPod H100, starting from a fresh SSH connection. Design spec:
`docs/superpowers/specs/2026-06-01-qwen-subcircuits-topk-design.md`.

## Cost & time budget (single H100 SXM 80 GB)

Grounded in measured Path-A throughput (plain Qwen3-4B fp32, max_new_tokens=80:
1.6–2.8 s/gen → planned 2.0; ReplacementModel bf16 planned 5 s/gen):

| step | generations | est. wall |
|---|---|---|
| setup + model/transcoder downloads (~70 GB) | — | ~0.7 h |
| CPU steps (rebuild index, 04, 07, aggregate) | — | ~0.4 h |
| Stage 08 ablation (7 subcircuits × 2 pos-modes × 11 conds × 50 + baselines) | 8,250 | ~11.5 h |
| Top-K zero sweep (8 K × 600) | 4,800 | ~6.7 h |
| Top-K proxy sweep, features (+ baseline) | 5,100 | ~2.8 h |
| Top-K proxy sweep, edges | 4,800 | ~2.7 h |
| **Total** | **~23,750** | **~25 h ± 20% (20–30 h)** |

≈ **$60–90** at ~$2.99/h (secure) / ~$2.69/h (community). Smoke test ≈ 30 min ≈ $1.50.

Knobs if you need to shrink it (env vars on the launcher):
- `SUBCIRCUITS=universal_refusal_core,ctrl_shared_refusal,jb_fiction_specific_vs_ctrl,jb_analytical_specific_vs_ctrl,jb_cognitive_reframe_specific_vs_ctrl` → Stage 08 drops to ~8.5 h (−3 h, loses 2 JB classes)
- Skip the edges sweep: comment out STEP 7 (−2.7 h)
- `K_VALUES=1,3,10,25,100,250` (−25% of sweep time)
- `REGEN_UPSTREAM=1` **adds** ~14 h / ~$42 (full Stage 01→02→02c re-run; only for a fully self-contained reproduction — the default reuses the HF graphs)

**VRAM**: ReplacementModel = Qwen3-4B (~8 GB bf16) + 36 transcoders (~60 GB) ≈
70 GB → needs the 80 GB H100 (same stack that generated the graphs).
**Disk**: ≥ 150 GB volume recommended (HF cache ~70 GB + venv + outputs).

---

## 0. Pod spec

- GPU: **1× H100 SXM 80 GB** (80 GB is required, see VRAM above)
- Volume: ≥ 150 GB mounted at `/workspace`
- Template: any recent PyTorch/CUDA image with Python ≥ 3.10

## 1. Clone + branch (skip clone if the volume already has the repo)

```bash
cd /workspace
git clone --recurse-submodules https://github.com/AutoInterp/Refusal-Lens.git
cd Refusal-Lens
git checkout emnlp-perm-edit
git pull origin emnlp-perm-edit
git submodule update --init --recursive    # vendor/circuit-tracer is required
```

## 2. Environment variables

```bash
# HuggingFace — token needs READ+WRITE on moon70/refusal-lens-graphs
export HF_TOKEN="hf_..."
export HF_HOME=/workspace/hf            # keep the 70GB model cache on the volume

# Git push auth (watcher pushes results to the branch) — use a GitHub PAT
git config user.name  "Mahmoud Shabana"
git config user.email "algoversemechinterp@gmail.com"
git remote set-url origin "https://<github-username>:<PAT>@github.com/AutoInterp/Refusal-Lens.git"
```

Add `export HF_TOKEN=... HF_HOME=/workspace/hf` to `~/.bashrc` too — tmux
windows started later must inherit them.

## 3. Preflight checks (run ALL before spending GPU time)

```bash
cd /workspace/Refusal-Lens

# 3a. GPU is the right one
nvidia-smi --query-gpu=name,memory.total --format=csv   # expect H100, ~81559 MiB

# 3b. Disk
df -h /workspace                                         # expect >= 150G total

# 3c. tmux present
tmux -V

# 3d. Python deps install cleanly (one-time, ~5 min; the launcher re-checks)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip -q && pip install -e . -q && pip install -e ./vendor/circuit-tracer -q
python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"
python3 -c "import circuit_tracer, transformers, huggingface_hub; print('imports OK')"

# 3e. HF READ — must print 553
python3 -c "
from huggingface_hub import HfApi
fs=[f for f in HfApi().list_repo_files('moon70/refusal-lens-graphs', repo_type='dataset')
    if f.startswith('runs/run_emnlp_qwen_L18_20260522/')]
print(len(fs))"

# 3f. HF WRITE — uploads + deletes a probe file (must print 'write OK')
python3 -c "
import io, os
from huggingface_hub import HfApi
api=HfApi(token=os.environ.get('HF_TOKEN') or None)
api.upload_file(path_or_fileobj=io.BytesIO(b'probe'), path_in_repo='runs/_preflight_probe.txt',
                repo_id='moon70/refusal-lens-graphs', repo_type='dataset',
                commit_message='preflight write probe')
api.delete_file('runs/_preflight_probe.txt', repo_id='moon70/refusal-lens-graphs',
                repo_type='dataset', commit_message='remove preflight probe')
print('write OK')"

# 3g. Git PUSH — dry run (must end with 'Everything up-to-date' or a ref line, NOT an auth error)
git push --dry-run origin HEAD:emnlp-perm-edit

# 3h. Dataset present in repo
ls dataset/refusal_lens_controlled_dataset.json

# 3i. Orchestration dry-run — prints the full step plan, stages graphs +
#     direction files (downloads ~180MB of graphs from HF if absent), no GPU work
DRY_RUN=1 NO_TMUX=1 bash scripts/emnlp_perm_edit/runpod_qwen_subcircuits.sh
#     Expect: "Packed graphs present: 550", "Direction r_unnorm present: ...",
#     8 steps listed as DRY. Then clear the dry-run marker:
rm -f data/results/emnlp_perm_edit/qwen_subcircuits/.QWEN_SUBCIRCUITS_DONE

# 3j. Local unit tests (CPU, <1 min)
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_qwen_subcircuit_orchestration.py
```

## 4. Smoke test (~30 min, ~$1.50 — REQUIRED before the full run)

Exercises every step including both model loads, on 2 prompts / K∈{1,5}, into
isolated `/tmp` dirs (never touches real outputs), with schema assertions:

```bash
source .venv/bin/activate
bash scripts/emnlp_perm_edit/smoke_test_qwen_subcircuits.sh
# Must end with: SMOKE TEST PASSED
```

If it fails at a GPU step, nothing about the full run will work better — fix
first (the failing step name is printed).

## 5. Launch the full run

```bash
# From repo root, OUTSIDE tmux (it self-relaunches into tmux session 'qwen_subcircuits'):
bash scripts/emnlp_perm_edit/runpod_qwen_subcircuits.sh

# Start the watcher in a second window of the same session:
tmux new-window -t qwen_subcircuits -n watcher \
  "cd /workspace/Refusal-Lens && HF_TOKEN=$HF_TOKEN bash scripts/emnlp_perm_edit/watch_and_commit_qwen_subcircuits.sh 2>&1 | tee /tmp/qwen_subcircuits_watcher.log"
```

The watcher polls every 5 min (36 h timeout); on the DONE marker it commits the
result JSONs/report to `emnlp-perm-edit`, pushes, and uploads
`subcircuits.json` (+ curves + report) to HF `runs/run_emnlp_qwen_L18_20260522/`.

## 6. Monitoring

```bash
tmux attach -t qwen_subcircuits          # main window = run, 'watcher' window = watcher
tail -f /tmp/qwen_subcircuits_*.log      # launcher log (one per start time)
ls -la data/results/emnlp_perm_edit/qwen_subcircuits/          # results appear incrementally
cat data/results/emnlp_perm_edit/qwen_subcircuits/.QWEN_SUBCIRCUITS_STEP_FAILED.txt  # only exists on failures
```

Detach with `Ctrl-b d`. SSH disconnects are safe — everything lives in tmux.

## 7. Failures & resume

- A failed step does NOT abort the run (`run_step` continues; failures are
  recorded in `.QWEN_SUBCIRCUITS_STEP_FAILED.txt` and the final summary).
- **To resume after a crash/preemption: just re-run the launcher.** Every step
  is idempotent or resumable (rebuild/04/07 cheap re-runs; Stage 08 + sweeps
  use `--resume` with incremental checkpoints).
- If only the watcher died, re-run it — if the DONE marker already exists it
  commits/pushes immediately.

## 8. After completion

1. Confirm the push: the branch should have a `qwen subcircuits: ...` commit.
2. Refresh the frontend on any machine to see the subcircuits + Top-K sets in
   the graph viewer panel:
   ```bash
   python3 scripts/pipeline/fetch_graph_data.py \
       --run run_emnlp_qwen_L18_20260522 --dataset-repo moon70/refusal-lens-graphs
   cd data/results/pipeline_runs/run_emnlp_qwen_L18_20260522/05_frontend
   python3 -m http.server 8000     # subcircuit panel now lists topk_* + rule-based sets
   ```
3. Read `data/results/emnlp_perm_edit/qwen_subcircuits/QWEN_SUBCIRCUIT_REPORT.md`
   (dissociation digest + Pareto knees) — the basis for the results report.
4. **Stop the pod.**

## Output inventory

| artifact | path |
|---|---|
| Rule-based subcircuits | `<run>/07_subcircuits/subcircuits.json` |
| Stage 08 dissociation results | `<run>/08_ablation/ablation_{results,summary}.json`, `dissociation_matrix.png` |
| Top-K sweeps | `qwen_subcircuits/topk_sweep_{zero_features,proxy_features,proxy_edges}.json` |
| Pareto curves + knees | `qwen_subcircuits/pareto_curves.{json,png}` |
| Frontend subcircuits (rule-based + topk_*) | `qwen_subcircuits/subcircuits_frontend.json` → HF `runs/<run>/subcircuits.json` |
| Report | `qwen_subcircuits/QWEN_SUBCIRCUIT_REPORT.md` |

(`<run>` = `data/results/pipeline_runs_qwen/run_emnlp_qwen_L18_20260522`)
