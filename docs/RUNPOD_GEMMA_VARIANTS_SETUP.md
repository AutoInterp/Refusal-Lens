# RunPod Setup — Gemma Variant Attribution Graph Regeneration

Step-by-step instructions for regenerating the three Gemma refusal-direction
variant attribution graphs (complement / full / outlier) on a RunPod GPU, starting
from a fresh SSH connection. Each variant attributes at L15 toward a different
decomposition of the unnormalized refusal direction; results are packed and pushed
to HF `moon70/refusal-lens-graphs`.

## Cost & time budget (A40 or A6000 48 GB)

Three variants × 50 prompts each at L15, float32 + save-graphs, plus 05 pack
and HF push per variant:

| step | est. wall |
|---|---|
| setup + model/circuit-tracer fork download | ~0.5 h |
| ensure_gemma_variant_directions (CPU) | < 1 min |
| complement attribution (50 prompts, A40) | ~1–1.5 h |
| full attribution (50 prompts) | ~1–1.5 h |
| outlier attribution (50 prompts) | ~0.5–1 h |
| 05 pack + HF push × 3 | ~0.2 h |
| **Total** | **~3–5 h** |

≈ **$2–6** at ~$0.79–1.19/h (A40/A6000 community). Smoke test ≈ 15 min ≈ $0.20.

**VRAM**: Gemma-3-4B in float32 (~16 GB) + gemma-scope transcoders +
circuit-tracer transformer-lens backend + attribution buffers. Fits comfortably
on a **48 GB card**. An H100 80 GB also works but is overkill and more expensive.

**Disk**: **150 GB network volume** at `/workspace`. Actual footprint ≈ 30 GB
(HF cache ~20 GB + venv ~5 GB + repo ~3 GB + `.pt` graphs per variant ~1–2 GB
purged after push). Use a *network* volume — it persists across pods so a spot
preemption reattaches the model cache instead of re-downloading it.
Requires `HF_HOME=/workspace/hf` (step 2).

### The circuit-tracer fork (critical)

This run uses `--measurement-hook hook_resid_post`. The upstream
`circuit-tracer` package does not expose this hook; the fork at
`vendor/circuit-tracer` (branch `refusal-lens-multi-position-fix`) does.
**Without the fork the attribution graphs are wrong.** Always install with
`uv pip install -e vendor/circuit-tracer` (step 3).

---

## 0. Pod spec

- GPU: **1× A40 48 GB or A6000 48 GB** (48 GB is sufficient; H100 80 GB also works)
- Volume: **150 GB** network volume mounted at `/workspace`
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
export HF_HOME=/workspace/hf            # keep the model cache on the volume

# Git push auth (watcher pushes attribution summaries to the branch) — use a GitHub PAT
git config user.name  "Mahmoud Shabana"
git config user.email "algoversemechinterp@gmail.com"
git remote set-url origin "https://<github-username>:<PAT>@github.com/AutoInterp/Refusal-Lens.git"
```

Add `export HF_TOKEN=... HF_HOME=/workspace/hf` to `~/.bashrc` too — tmux
windows started later must inherit them.

## 3. Install dependencies (one-time, ~5 min)

```bash
cd /workspace/Refusal-Lens
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
# CRITICAL: install the fork, not the upstream package
uv pip install -e vendor/circuit-tracer
python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))"
python3 -c "import circuit_tracer, transformers, huggingface_hub; print('imports OK')"
```

> If `uv` is not available: `pip install uv -q` first, or fall back to
> `pip install -e vendor/circuit-tracer -q`.

## 4. Preflight checks (run ALL before spending GPU time)

```bash
cd /workspace/Refusal-Lens
source .venv/bin/activate

# 4a. GPU is present and is ≥ 48 GB
nvidia-smi --query-gpu=name,memory.total --format=csv
# expect: A40 or A6000, ~49140 MiB (or H100/A100 if on a larger card)

# 4b. Disk
df -h /workspace                          # expect >= 150G total

# 4c. tmux present
tmux -V

# 4d. HF READ — must print 556 (or close; the Qwen run is the read-probe target)
python3 -c "
from huggingface_hub import HfApi
fs=[f for f in HfApi().list_repo_files('moon70/refusal-lens-graphs', repo_type='dataset')
    if f.startswith('runs/run_emnlp_qwen_L18_20260522/')]
print(len(fs))"
# expect: ~556

# 4e. HF WRITE — uploads + deletes a probe file (must print 'write OK')
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

# 4f. Git PUSH — dry run (must end with 'Everything up-to-date' or a ref line, NOT an auth error)
git push --dry-run origin HEAD:emnlp-perm-edit

# 4g. Unit tests (CPU, < 1 min)
PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/tests/test_gemma_variant_pipeline.py

# 4h. Direction check — ensure variant direction files are present / buildable
python3 scripts/emnlp_perm_edit/ensure_gemma_variant_directions.py --check-only
# expect: all three variants report directions present

# 4i. Orchestration dry-run — prints the full step plan per variant, no GPU work
DRY_RUN=1 NO_TMUX=1 bash scripts/emnlp_perm_edit/runpod_gemma_variants.sh
# expect: three DRY blocks (complement / full / outlier), each showing
#   DRY: attribution ... DRY: nets gate ... DRY: viz ... DRY: push
```

All nine checks must pass before proceeding.

## 5. Smoke test (~15 min, ~$0.20 — REQUIRED before the full run)

Exercises the full pipeline (attribution → nets gate → pack → push dry-run) on
2 prompts into an isolated `/tmp` directory. Never touches real outputs.

```bash
source .venv/bin/activate
bash scripts/emnlp_perm_edit/smoke_test_gemma_variants.sh
# Must end with: SMOKE TEST PASSED
```

If it fails at the attribution step, the circuit-tracer fork is likely not
installed correctly (re-run step 3). If it fails at the nets gate, check that
`data/results/emnlp_perm_edit/phase0_controllability/gemma_var_nets.json` exists.
The failing step name is printed — fix before proceeding.

## 6. Launch the full run

```bash
# From repo root, OUTSIDE tmux (it self-relaunches into tmux session 'gemma_variants'):
bash scripts/emnlp_perm_edit/runpod_gemma_variants.sh

# Start the watcher in a second window of the same session:
tmux new-window -t gemma_variants -n watcher \
  "cd /workspace/Refusal-Lens && HF_TOKEN=$HF_TOKEN bash scripts/emnlp_perm_edit/watch_and_commit_gemma_variants.sh 2>&1 | tee /tmp/gemma_variants_watcher.log"
```

The orchestrator runs complement → full → outlier in sequence. For each variant:
1. Builds attribution graphs (50 prompts, L15, `hook_resid_post`, float32).
2. Passes the nets gate — **must print `"ok": true`** for the variant to be trusted.
3. Packs + annotates the frontend bundle (step 05).
4. Pushes to HF `moon70/refusal-lens-graphs/runs/run_gemma_<variant>_L15/`.
5. Purges the raw `.pt` graphs to free disk.

On completion it writes `data/results/pipeline_runs/.GEMMA_VARIANTS_DONE`.
The watcher (8 h timeout, 5 min poll) picks up the marker and commits the
small attribution summary JSONs + packed metadata to `emnlp-perm-edit`.

## 7. Monitoring

```bash
tmux attach -t gemma_variants    # main window = run, 'watcher' window = watcher
# detach with Ctrl-b d — SSH disconnects are safe, everything lives in tmux

# Logs
tail -f /tmp/gemma_variants_*.log 2>/dev/null     # if the launcher echoes there

# Progress
ls -la data/results/pipeline_runs/gemma_var_complement/
ls -la data/results/pipeline_runs/gemma_var_full/
ls -la data/results/pipeline_runs/gemma_var_outlier/

# Failure log (only exists if a step failed)
cat data/results/pipeline_runs/.GEMMA_VARIANTS_FAILED.txt
```

### Per-variant nets gate

Each variant's gate prints a JSON block. The critical field is `"ok": true` —
if any variant prints `"ok": false`, its graphs are not trusted and the variant
is skipped (recorded in `.GEMMA_VARIANTS_FAILED.txt`). Do not proceed to the
assembly step for that variant without investigating.

## 8. Failures & resume

- A failed step does NOT abort the run — the orchestrator continues to the next
  variant. Failures are recorded in `.GEMMA_VARIANTS_FAILED.txt`.
- **To resume after a crash/preemption: just re-run the launcher.**
  Re-running skips any variant that already completed and was pushed to HF
  (detected via a per-variant `.VARIANT_PUSHED` marker written after the purge
  step), so it will NOT hit the Stage-05 missing-graphs error on a completed
  variant. Only incomplete variants are resumed; their attribution uses
  `--resume` (incremental checkpoints). To force a re-run of a completed
  variant: `rm data/results/pipeline_runs/gemma_var_<v>/.VARIANT_PUSHED`.
- If only the watcher died, re-run it — if the DONE marker already exists it
  commits/pushes immediately.

## 9. After completion

1. Confirm the watcher commit on `emnlp-perm-edit`: `git log --oneline -5`.
2. Confirm HF: three run directories should now exist under
   `moon70/refusal-lens-graphs/runs/`:
   - `run_gemma_complement_L15/`
   - `run_gemma_full_L15/`
   - `run_gemma_outlier_L15/`
3. **Stop the pod.**
4. Assemble + serve locally (Task 10):
   ```bash
   python3 scripts/pipeline/assemble_compare_frontend.py
   # then open the frontend to compare complement / full / outlier attribution graphs
   ```

## Output inventory

| artifact | HF path |
|---|---|
| Complement attribution graphs + frontend | `runs/run_gemma_complement_L15/` |
| Full attribution graphs + frontend | `runs/run_gemma_full_L15/` |
| Outlier attribution graphs + frontend | `runs/run_gemma_outlier_L15/` |

Local paths (on the pod, post-run):

| artifact | local path |
|---|---|
| Attribution results (all variants) | `data/results/pipeline_runs/gemma_var_<v>/02_attribution/attribution_results.json` |
| Packed frontend metadata | `data/results/pipeline_runs/gemma_var_<v>/05_frontend/data/graph-metadata.json` |
| DONE marker | `data/results/pipeline_runs/.GEMMA_VARIANTS_DONE` |
| Failure log (if any) | `data/results/pipeline_runs/.GEMMA_VARIANTS_FAILED.txt` |

(`<v>` = `complement`, `full`, or `outlier`)
