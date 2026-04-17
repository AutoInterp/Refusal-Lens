# Refusal-Lens Pipeline

End-to-end pipeline for attributing and analyzing the refusal circuit in Gemma-3-4b-it. Ingests harmful/harmless prompts, computes per-layer refusal directions, runs CLT attribution with the vendored circuit-tracer, verifies the attribution arithmetic, and labels every feature using the Gemma Scope HuggingFace dashboards.

For the design philosophy, stage specs, and forward roadmap see **[PIPELINE_PLAN.md](PIPELINE_PLAN.md)**. This README covers the current implementation state, the most recent experiment's numerical results, and deployment.

---

## Stage summary

| # | Script | Role | Status |
|---|---|---|---|
| 01 | `01_compute_direction.py` | Per-layer refusal directions (normalized + unnormalized) at all 34 layers | ✓ Done |
| 02 | `02_run_attribution.py` | CLT attribution graphs for bare + 5 jailbreak classes, per-prompt feature comparison | ✓ Done |
| 02b | `02b_statistical_analysis.py` | Paired stats (Wilcoxon, t-test, Cohen's d, bootstrap CIs), dual-mechanism decomposition, plots, markdown report | ✓ Done |
| 03 | `03_verify_attribution.py` | Verifies `sum(feature attributions) ≈ r · h[L=32]`, reports MLP contribution fraction | ✓ Done |
| 04 | `04_label_features.py` | Labels every unique feature using HuggingFace dashboard binary payloads (top/bottom logits, examples) | ✓ Done |
| 05 | `05_visualize_circuits.py` | Color-coded attribution circuit diagrams | Planned |
| 06 | `06_causal_intervention.py` | Arditi-method causal intervention (Tejas's track) | Planned |

Shared modules: `config.py` (model/layer/class constants), `utils.py` (run-dir helpers, prompt formatting, dataset selection).

---

## Latest experiment — `run_20260417_010035` (50 prompts)

### Stage 01: Direction Computation

| Layer | Separation | Role |
|---|---:|---|
| 0 | 7.4 | — |
| 11 | 1,346 | early MLP buildup |
| 15 | **3,101** | **best causal layer** (Tejas) |
| 24 | 9,384 | JB effects concentrate here → L32 |
| 29 | 18,930 | — |
| 32 | **20,873** | **best separation layer** |
| 33 | 287 | pre-RMSNorm artifact |

- Matches Tejas's prior numbers within noise (L32: 20,827; L15: 3,131).
- **Cosine(L15, L32) = −0.115** — near-orthogonal. The direction rotates meaningfully between the layer where it *separates* and the layer where it's *causally effective*. Flag for mentor.
- The L33 collapse (287) is consistent with the pre-RMSNorm hook artifact.

### Stage 02b: Statistical Analysis

All five jailbreak classes produce highly significant effects on net refusal attribution:

| Class | Δnet | % change | Cohen's d | p (Wilcoxon) | 95% CI | Consistency |
|---|---:|---:|---:|---:|---|---:|
| **Analytical** | −73.7 | −104.6% | −2.37 | 5.3e-15 | [−82.1, −64.9] | 49/50 |
| **Fiction** | −65.3 | −92.7% | −1.57 | 3.0e-13 | [−76.6, −53.8] | 47/50 |
| **Cognitive reframe** | −50.2 | −71.3% | −1.41 | 2.5e-14 | [−60.2, −40.4] | 49/50 |
| **Roleplay** | −38.7 | −54.9% | −0.91 | 1.4e-8 | [−50.5, −27.0] | 42/50 |
| **Completion** | **+5.0** | **+7.2%** | +0.27 | 0.011 | [−0.1, +10.2] | 15/50 |

**Dual-mechanism decomposition** (how each class moves the positive-attribution and negative-attribution halves separately):

| Class | dPos (pro-refusal) | dNeg (anti-refusal) | Dominant |
|---|---:|---:|---|
| Roleplay | −22.5 (−16.7%) | −16.2 (−25.3%) | Balanced |
| Fiction | −43.1 (−32.0%) | −22.2 (−34.6%) | Balanced |
| Analytical | −44.7 (−33.2%) | −29.0 (−45.1%) | Balanced |
| **Completion** | **+19.7 (+14.6%)** | −14.6 (−22.8%) | **Pro-refusal recruitment** |
| Cognitive reframe | −33.8 (−25.1%) | −16.4 (−25.5%) | Dampening-dominant |

Completion is the paradox: 35/50 prompts show *more* refusal attribution after the JB framing. dPos increases by +14.6% — new pro-refusal features are being recruited, not existing ones being amplified.

**Feature comparison sizes** (total unique features per condition pair):

| Class | Bare | JB | Shared % | JB-only % | Sign-flip % |
|---|---:|---:|---:|---:|---:|
| Roleplay | 8,342 | 12,540 | 65.7% | 56.3% | 17.5% |
| **Fiction** | 8,342 | 12,996 | 58.5% | **62.5%** | **25.6%** |
| Analytical | 8,342 | 12,109 | 63.0% | 56.6% | 23.1% |
| Completion | 8,342 | 11,200 | **73.5%** | 45.3% | 16.6% |
| Cognitive reframe | 8,342 | 11,131 | 64.4% | 51.8% | 20.3% |

Fiction restructures the circuit most dramatically — highest sign-flip rate (25.6%) and highest share of JB-only features. Completion preserves the most of the bare circuit (73.5% shared).

### Stage 03: Attribution Verification (M2)

- Full projection `r · h[L=32]` (mean): **17,230.8** (std 1,986)
- Sum of feature attributions (mean): **70.47** (std 17.6)
- **MLP ratio**: **0.404%** of the total refusal signal lives in the transcoded MLP features. The remaining 99.6% is carried by attention heads + embeddings.
- `attr_net_mean=70.47` exactly matches the bare condition in Stage 02b — keys plumb through correctly.
- Per-layer decomposition (10 prompts, 34 layers) shows early-layer buildup — layers 7–11 alone contribute ~2,400 to the projection for a typical prompt.

### Stage 04: Feature Labeling (M4)

- **876 unique features** collected across all prompts and conditions
- **876 labeled (100%)** via HuggingFace Gemma Scope dashboards (`mwhanna/gemma-scope-2-4b-it`)
- **788 priority features** (those appearing in sign-flipped / dampened / amplified-anti buckets) all labeled
- Comparison-bucket sizes: 603 sign-flipped, 115 dampened, 117 amplified-anti
- Caveat: many top-token patches in Gemma Scope are polyglot or byte-level noise. The labels are *correct representations of what the dashboard shows*, but human-interpretable features will require reading activation examples on concrete prompts (Stage 05).

---

## Mechanistic story so far

A running narrative of what the pipeline has taught us about Gemma-3-4b-it's refusal circuit. This section grows as new stages and analyses land — last revised 2026-04-17 after A3 + A4 visualizations.

Plot paths below are relative to repo root; each was produced by the most recent pipeline run (`data/results/pipeline_runs/run_20260417_010035/`).

### 1. Refusal is linear, but the "direction" rotates across layers

The diff-in-means between harmful and harmless activations gives a clean linear readout at every layer from L9 onward, with separation climbing monotonically to L32 (~20,873) before collapsing at L33 to 287 — the known pre/post-RMSNorm artifact.

![Separation by layer](../../data/results/pipeline_runs/run_20260417_010035/02b_stats/separation_by_layer.png)

But **the direction at the best-separation layer (L32) is nearly orthogonal to the direction at Tejas's best-causal layer (L15)**: cosine = −0.115 at n=64. Causal steering works at L15 despite L15 having ~7× less separation than L32 — which means the right causal direction is not simply a scaled version of the right readout direction. The signal rotates as it propagates. (A5 will show the full 34×34 cosine matrix so we can see exactly where the pivot happens.)

### 2. The signal is assembled in two waves, with a dampening layer mid-stream

The per-layer-decomposition plot (10 prompts × 34 layers, aggregated) shows:

![Per-layer contribution](../../data/results/pipeline_runs/run_20260417_010035/03_verification/per_layer_contribution.png)

- **Early wave (L7–L11)** — first buildup, peaks ~850 at L11
- **Mid plateau (L12–L19)** — modest contributions (200–450)
- **L20 actively dampens** the projection (~−200 mean)
- **Late wave (L23–L32)** — the dominant ramp; L29–L30 alone contribute ~1,650 each
- **L33 (post-measurement)** — −17,311, the RMSNorm artifact; not part of the assembled projection

Interpretation: the separation at L32 is mostly *late*-built. Tejas's L15 intervention works because downstream layers then amplify the L15 residual ~4×. L20 is an open question — a "pump-the-brakes" layer actively pulling the projection down.

### 3. MLP transcoders only capture ~0.4% of the refusal signal

Verification against the full residual-stream dot product shows:

| Metric | Mean | Std |
|---|---:|---:|
| `r · h[L=32]` | 17,230 | 1,986 |
| Σ feature attributions | 70.47 | 17.6 |
| **MLP ratio** | **0.404%** | 0.077% |

The ~99.6% gap is carried by **attention heads + embeddings** — a property of the Gemma Scope transcoder set (MLP-only). Implication: any mechanistic intervention relying purely on transcoded MLP features will move <1% of the measured refusal signal. Attention-head attribution is a known gap (task S3).

### 4. Jailbreaks produce strong, statistically real changes

Five jailbreak classes × 50 prompts, paired bare-vs-JB, all stat-significant:

![Class comparison](../../data/results/pipeline_runs/run_20260417_010035/02b_stats/class_comparison.png)
![Effect sizes](../../data/results/pipeline_runs/run_20260417_010035/02b_stats/effect_sizes.png)

Ordering (by effect size):

```
Analytical       d = −2.37  (strongest; 49/50 consistency)
Fiction          d = −1.57
Cognitive reframe d = −1.41
Roleplay         d = −0.91
Completion       d = +0.27  (INVERTED — discussed below)
```

Analytical framing is the most reliable net suppressor (net goes *negative*, i.e. the model becomes anti-refusal-biased). Fiction is the most disruptive to the underlying circuit (see #5). Roleplay is weakest.

### 5. Fiction reorganizes the circuit; completion preserves it

Feature-comparison sizes between each JB condition and bare:

![Feature comparison summary](../../data/results/pipeline_runs/run_20260417_010035/02b_stats/feature_comparison_summary.png)

| Class | Shared with bare | JB-only features | Sign-flipped |
|---|---:|---:|---:|
| **Fiction** | 58.5% | **62.5%** | **25.6%** |
| Analytical | 63.0% | 56.6% | 23.1% |
| Cognitive reframe | 64.4% | 51.8% | 20.3% |
| Roleplay | 65.7% | 56.3% | 17.5% |
| Completion | **73.5%** | 45.3% | 16.6% |

Fiction uses the most novel features and flips the most shared features — it's not suppressing the circuit, it's *rewiring* it. Completion in contrast keeps almost three-quarters of the bare circuit intact — consistent with its paradoxical *strengthening* effect (see #6).

### 6. Completion is the paradox: it recruits *more* refusal

Completion-style JBs ("I cannot help with that, but what I can suggest is…") should *weaken* refusal but instead **strengthen it by +7.2%** (Cohen's d = +0.27). Decomposition into pro-refusal (`dPos`) and anti-refusal (`dNeg`) halves:

![Per-prompt deltas](../../data/results/pipeline_runs/run_20260417_010035/02b_stats/per_prompt_deltas.png)

| Class | dPos | dNeg | Interpretation |
|---|---:|---:|---|
| Roleplay | −22.5 | −16.2 | Balanced dampening |
| Fiction | −43.1 | −22.2 | Balanced dampening |
| Analytical | −44.7 | −29.0 | Balanced dampening |
| **Completion** | **+19.7** | −14.6 | **Pro-refusal recruitment** |
| Cognitive reframe | −33.8 | −16.4 | Dampening-dominant |

Every other class dampens both halves. Completion is the only one that *grows* the pro-refusal half (+14.6%) while weakening the anti-refusal half — two mechanisms pulling in opposite directions, with pro-refusal winning. 35/50 prompts go up, 15/50 go down (the per-prompt-delta plot shows the bimodal pattern clearly). This suggests there's a specific set of features being *recruited* by the completion framing. Identifying which ones is A2.

---

## Gaps & open questions

Items flagged for future pipeline stages or deeper analysis.

| Gap | Status | Where it gets answered |
|---|---|---|
| Does the model actually refuse bare prompts / comply under JB? | Not measured | A1 (Stage 02c — coherence + refusal classification) |
| Which specific features are recruited by completion? | Not identified | A2 (Stage 02b extension — top Δattribution under completion) |
| How much does the direction rotate, layer-by-layer? | Only 6 pairwise cosines stored | A5 (full 34×34 heatmap, next) |
| Bare-vs-JB distribution shape (esp. completion's bimodal pattern) | Only means + CIs reported | A6 (violin/box plot by class) |
| Where do sign-flipped / dampened / amplified-anti features live by layer? | Not binned | A8 (layer histogram per bucket) |
| Class overlap — which features appear in all 5 classes vs one? | Table only | A7 (UpSet plot) |
| What does L20 do? Why is it the only negative-contribution layer? | Open | Dedicated follow-up (not in current plan) |
| 99.6% of the refusal signal lives in attention + embeddings — which heads? | Not attributed | Task S3 (attention-head attribution) |
| Are any two JB classes structurally similar (shared recruited features)? | Not cross-tabulated | Stage 07 subcircuit identification |

---

## Deployment

### Local smoke test (no GPU required)

```bash
PYTHONPATH=src python3 -m pytest scripts/pipeline/tests/test_pipeline_local.py -v -W ignore::DeprecationWarning
```

Expected: 60/60 pass. Uses an existing run directory as a fallback for stages that need heavy compute.

### Full RunPod run (stages 01 → 04)

Prerequisites on the pod:
- `/workspace/Refusal-Lens` clone on the `foundation` branch
- `/workspace/venv` with torch (nightly for Blackwell GPUs: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`), transformers, scipy, and the vendored circuit-tracer installed
- HuggingFace token available (either `~/.cache/huggingface/token` or `HF_TOKEN` env var)
- A GitHub fine-grained PAT exported as `GHP_TOKEN` for auto-push (scope: contents:write on the repo)
- Recommended hardware: **RTX A6000 (48GB)** or **A40 (48GB)** — model + transcoders fit in ~15–20 GB, Blackwell cards need PyTorch nightly

**Step 1 — start the experiment in a detached tmux session:**

```bash
tmux new-session -d -s pipeline -n run "cd /workspace/Refusal-Lens && source /workspace/venv/bin/activate && python -u scripts/pipeline/tests/test_runpod_1_4.py --n-prompts 50 2>&1 | tee /workspace/pipeline_output.log"
```

**Step 2 — export `GHP_TOKEN` in your SSH shell before adding the watcher window** (so the new window inherits it):

```bash
export GHP_TOKEN=ghp_...   # paste your fresh token
echo $GHP_TOKEN            # sanity check
```

**Step 3 — add the watcher window that prints heartbeats and auto-commits on completion:**

```bash
tmux new-window -t pipeline -n watcher "L=/workspace/pipeline_output.log; echo \"[watcher] monitoring \$L\"; until grep -q 'PIPELINE COMPLETE' \$L 2>/dev/null; do grep -q 'Pipeline stopped at Stage' \$L 2>/dev/null && { echo '[watcher] FAILED'; exit 1; }; echo \"[watcher \$(date +%H:%M:%S)] alive — tail: \$(tail -n1 \$L 2>/dev/null | cut -c1-120)\"; sleep 30; done && echo '[watcher] pipeline complete — committing...' && cd /workspace/Refusal-Lens && R=\$(grep -oE 'data/results/pipeline_runs/run_[0-9_]+' \$L | head -1) && echo \"[watcher] run_dir: \$R\" && cp \$L \$R/pipeline_output.log && U=\$(git remote get-url origin | sed \"s|https://|https://x-access-token:\$GHP_TOKEN@|\") && git pull --rebase \$U foundation && git add \$R && git commit -n -m 'pipeline run: stages 01-04 (50 prompts)' && git push \$U foundation && echo '[watcher] DONE — pushed to foundation'"
```

**Step 4 — attach to watch progress:**

```bash
tmux attach -t pipeline     # Ctrl+b n to switch windows, Ctrl+b d to detach
```

**Expected runtimes** (48 GB card, 50 prompts):
- Stage 01: ~1 min
- Stage 02: ~3–4 hours
- Stage 02b: ~1 min
- Stage 03: ~5–10 min
- Stage 04: ~2 min

### Resuming after a crash

Stage 02 checkpoints after every prompt. To resume the attribution stage without redoing stages 01:

```bash
python -u scripts/pipeline/tests/test_runpod_1_4.py \
  --n-prompts 50 \
  --run-dir data/results/pipeline_runs/run_YYYYMMDD_HHMMSS \
  --skip-stage 01 \
  --resume
```

To re-run only the lightweight stages (e.g. after a 02b bug fix):

```bash
python -u scripts/pipeline/tests/test_runpod_1_4.py \
  --n-prompts 50 \
  --run-dir data/results/pipeline_runs/run_YYYYMMDD_HHMMSS \
  --skip-stage 01 02
```

### Security notes

- `GHP_TOKEN` appears in `ps` output during the `git push` — acceptable on a private pod, rotate the token afterward if the pod is shared.
- Prefer fine-grained PATs scoped to this repo's contents, not classic tokens with full-repo access.
- The watcher writes a copy of `pipeline_output.log` into the run directory before committing, so the log is preserved as part of the run artifact.

---

## Output directory layout

```
data/results/pipeline_runs/run_YYYYMMDD_HHMMSS/
├── run_config.json
├── config.json
├── pipeline.txt                       # stage-level log from test runner
├── pipeline_output.txt                # full stdout stream (added by watcher)
├── 01_direction/
│   ├── refusal_direction.pt           # normalized r_hat at best_separation_layer
│   ├── unnormalized_r.pt              # unnormalized r (all layers)
│   ├── direction_metadata.json        # per-layer separation, cosines, best layers
│   └── directions/layer_XX.pt         # normalized r_hat per layer
├── 02_attribution/
│   ├── attribution_results.json       # per-prompt, per-condition feature attributions
│   ├── attribution_checkpoint.json    # resume state
│   └── feature_comparison_aggregate.json
├── 02b_stats/
│   ├── statistical_analysis.json
│   ├── EXPERIMENT_SUMMARY.md
│   ├── class_comparison.png
│   ├── per_prompt_deltas.png
│   ├── effect_sizes.png
│   └── feature_comparison_summary.png
├── 03_verification/
│   ├── verification_results.json
│   └── per_layer_decomposition.json
└── 04_labels/
    ├── feature_labels.json            # {L:F → {top_logits, bottom_logits, examples, ...}}
    ├── feature_labels_cache.json      # raw HF payload cache (survives re-runs)
    ├── feature_comparison_labeled.json
    ├── label_coverage.json
    └── top_features_report.md
```

---

## Known caveats

- **Gemma-3-4b-it is not on Neuronpedia** — labels come from the HuggingFace Gemma Scope dashboards (byte-range HTTP requests against `index.json.gz`). Many top-token patches are polyglot / byte-level noise; semantic interpretation requires activation examples on concrete prompts.
- **Transcoders cover MLP only** — the MLP ratio of ~0.4% means 99.6% of the refusal signal is carried by attention + embeddings. This is a property of the transcoder set, not a bug.
- **Layer 33 anomaly** — forward hooks capture pre-RMSNorm, `hidden_states` captures post-RMSNorm. Stage 01 uses `hidden_states`, which is why L33 separation shows as 287 (not ~20k). Expected.
- **Position -2** is the "model" token in Gemma-3's chat template. Stage 01 and Stage 02 use this position.
- **PyTorch + Blackwell GPUs** — stable PyTorch only supports up to sm_90. RTX PRO 6000 Blackwell (sm_120) needs `torch --pre` from the nightly cu128 index.
