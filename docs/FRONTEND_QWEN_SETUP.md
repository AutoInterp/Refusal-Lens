# Viewing the Qwen (and Gemma) Attribution Graphs Locally

Setup guide for research partners to browse the circuit-tracer attribution graphs on
their own machine. No GPU required — this only fetches pre-computed graphs and serves
a static frontend.

The Qwen run is **`run_emnlp_qwen_L18_20260522`**: 550 attribution graphs (50 prompts ×
11 conditions), refusal-direction target at **L18**, Qwen3-4B thinking-mode template,
transcoder `mwhanna/qwen3-4b-transcoders`.

---

## 0. Prerequisites

- Python 3.10+ (the repo's `.venv` already has what you need; otherwise
  `pip install huggingface_hub`).
- **HuggingFace access to the graph dataset.** The graphs live in the **private**
  dataset `moon70/refusal-lens-graphs`. You must either be granted read access to it
  or use a token that can read it. (Mahmoud: share the dataset with each partner's HF
  account, or make it public.)

> Note: `fetch_graph_data.py`'s built-in default repo is `AutoInterp/refusal-lens-graphs`,
> which is currently unreachable (404). **Always pass
> `--dataset-repo moon70/refusal-lens-graphs`** as shown below until the default is
> repointed.

## 1. Clone with submodules

The frontend is a git submodule (`vendor/circuit-tracer`). A plain clone leaves it
empty and the fetch will fail late.

```bash
git clone --recurse-submodules <repo-url>
cd Refusal-Lens
git checkout emnlp-perm-edit          # branch carrying this guide + Qwen fetch

# If you already cloned without submodules:
git submodule update --init --recursive
```

## 2. Authenticate to HuggingFace (private dataset)

```bash
pip install huggingface_hub            # if not already present
hf auth login                          # paste a token with read access to the dataset
```

## 3. Fetch the Qwen graphs

```bash
# List everything available (sanity check your access):
python3 scripts/pipeline/fetch_graph_data.py --list \
    --dataset-repo moon70/refusal-lens-graphs

# Fetch the Qwen L18 run (~180 MB, idempotent — safe to re-run):
python3 scripts/pipeline/fetch_graph_data.py \
    --run run_emnlp_qwen_L18_20260522 \
    --dataset-repo moon70/refusal-lens-graphs
```

This stages everything under
`data/results/pipeline_runs/run_emnlp_qwen_L18_20260522/05_frontend/`.

## 4. Serve and open

```bash
cd data/results/pipeline_runs/run_emnlp_qwen_L18_20260522/05_frontend
python3 -m http.server 8000
```

Then open in a browser:

- **Single-graph viewer:** http://localhost:8000/
- **Compare view (bare ↔ JB side-by-side):** http://localhost:8000/compare.html

Pick a prompt/condition from the graph list (slugs are `NNN_<condition>_single`, e.g.
`000_bare_single`, `000_jb_roleplay_single`). The 11 conditions per prompt are:
`bare`, `jb_{fiction,roleplay,analytical,completion,cognitive_reframe}`, and the
matched `ctrl_*` controls.

---

## What you'll see right now (and what's coming)

- ✅ **Attribution graphs render fully** — nodes (CLT features, embeddings, MLP error,
  the L18 refusal-direction target logit), edges, influence, per-token view.
- ⏳ **Feature labels are not in yet.** CLT feature nodes currently show as raw
  `F#<id>` (e.g. `F#0_17384`). Qwen labels can't be borrowed from Gemma (different
  transcoder: 163,840 features vs Gemma's 16,384), so they're being generated
  separately and baked in next. After that, re-run the fetch in step 3 to pick them up.
- ⏳ **Subcircuit panel is empty for Qwen.** Subcircuits are produced by the Stage-08
  porting work (Task 2); once `subcircuits.json` is published for this run, re-running
  the fetch will populate the panel automatically.

## Gemma graphs (for comparison)

Same flow, different `--run`:

```bash
python3 scripts/pipeline/fetch_graph_data.py \
    --run run_20260430_023247 \
    --dataset-repo moon70/refusal-lens-graphs
```

Other available Gemma runs: `run_20260418_172402`, `run_20260422_015552`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `vendor frontend not found` | `git submodule update --init --recursive` |
| `404 ... Repository Not Found` | You lack access to the dataset, or dropped `--dataset-repo moon70/refusal-lens-graphs`. Run `hf auth login` and confirm access. |
| `huggingface_hub not installed` | `pip install huggingface_hub` (or use the repo `.venv`). |
| Graphs list loads but nodes are blank | Hard-refresh; ensure you're serving from inside the `05_frontend/` dir (relative paths). |
| Re-fetch is slow | It's incremental — unchanged files are skipped by content hash. |
