# Refusal-Lens Unified Pipeline Plan

**Created**: 2026-04-14
**Last updated**: 2026-04-25 (Stage 08a PR #1 landed; GPU smoke running on the local 5080)
**Reference run**: `data/results/pipeline_runs/run_20260422_015552/` (50 prompts × 11 conditions, L15 measurement, controlled dataset)
**Working branch**: `l15-refactor`

---

## Goal

Combine Mahmoud's correlational/attribution work with Tejas's causal-intervention work into one reproducible pipeline that tells Georg's research story:

> *"Here are the circuits, they're real, here's what they encode, here's what happens when you manipulate them, and here's how to permanently patch a specific class of jailbreak."*

The full chain — correlational → causal → patching — is what unlocks NeurIPS-grade claims. ICML workshop submission targets the correlational + causal halves (May 4); NeurIPS adds the patching half.

---

## Pipeline Layout

```
scripts/pipeline/
├── config.py                       # Shared constants (D_MODEL=2560, 34 layers, 11 conditions)
├── utils.py                        # Run-dir helpers, prompt formatting, dataset loading,
│                                   # hooks, intervention helpers, Stage 08 cart helpers
├── utils_viz.py                    # Frontend staging, overlap/subcircuit annotation,
│                                   # gzip helpers
├── 01_compute_direction.py         # Per-layer refusal directions (34 layers)         [DONE ✓]
├── 02_run_attribution.py           # CLT attribution graphs (multi + single mode)     [DONE ✓]
├── 02b_statistical_analysis.py     # 30 stats blocks, ctrl-aware effect sizes         [DONE ✓]
├── 02c_pack_graphs.py              # `.pt → JSON.gz` packer for HF distribution       [DONE ✓]
├── 03_verify_attribution.py        # `sum(attr) ≈ r·h[L]` consistency check           [DONE ✓]
├── 04_label_features.py            # HF-dashboard-binary labels for every feature     [DONE ✓]
├── 04b_delphi_labels.py            # Claude-API human-readable labels                 [PLANNED]
├── 05_visualize_circuits.py        # Frontend orchestrator (3-way bare/ctrl/JB)       [DONE ✓]
├── 05_frontend_patches/*.{js,css,html}  # Overlap colors, subcircuit panel,
│                                   # compare.html, gzip-fetch, feature cart
├── 06_causal_intervention.py       # L15 Arditi (pro/anti/benign) (Tejas Script 20)   [DONE ✓]
├── 07_identify_subcircuits.py      # 18 rule-based subcircuits + JB-vs-ctrl contrast  [DONE ✓]
├── 08_ablate_subcircuits.py        # Runtime feature ablation (Stage 08a / PR #1)     [DONE ✓ — GPU validation pending]
├── 08b_direction_ablation.py       # Permanent weight edit (Arditi-style projection)  [PLANNED]
├── 08c_sidecar_build.py            # Input-dependent sidecar wrapper                  [PLANNED]
├── ablation_server.py              # FastAPI demo backend for Stage 05 cart           [DONE ✓]
├── fetch_graph_data.py             # Frontend bundle fetch from HF (~3 GB)
├── fetch_raw_graphs.py             # Raw `.pt` fetch from HF (~80 GB)
├── push_graph_data.py              # Push JSON.gz bundle to HF
├── push_raw_graphs.py              # Push raw `.pt` archive to HF
├── push_run.py                     # One-shot upload of an entire run dir
├── merge_stage02_shards.py         # Multi-GPU shard merger
└── rebuild_graph_metadata.py       # Repair tool for graph-metadata.json
```

Tests: `scripts/pipeline/tests/test_pipeline_local.py` — 13 stages worth of unit tests, ~225+ assertions.

Results land in `data/results/pipeline_runs/run_YYYYMMDD_HHMMSS/`.

---

## Critical Design Decisions (load-bearing)

### Per-layer directions, L15 is causal
Tejas established that the refusal direction **rotates** across layers. L32 has the strongest separation (~21,000) but L15 is causally effective. The 34×34 cosine heatmap (Stage 02b/A5) reveals **three regimes**: A (L5–L13, semantic), B (L14–L17, pivot), C (L18–L33, output). Stage 06 uses unnormalized r at **L15** for intervention; Stage 02 uses **L15 as measurement layer** for attribution graphs.

### D_MODEL = 2560 (Gemma-3-4b-it)
Hidden dim 2560 (not 2304 as an older memory had). Constant in `config.py`.

### Controlled dataset, 11 conditions
`dataset/refusal_lens_controlled_dataset.json` — 50 prompts × {`bare`, 5 × `jb_*`, 5 × `ctrl_*`} = 550 generations. Each `jb_<class>` has a matched `ctrl_<class>` benign prefix. **This is what enables `jb_specific` vs `prefix-induced` separation** — the headline ICML novelty.

### CLT (cross-layer transcoder), zero-positions mask
`mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine`. A feature at source layer `k` writes into layers `k..N-1` (decoder shape `[N-k, d_model]`). Encoder input = `pre_feedforward_layernorm.output`; decoder writes at `post_feedforward_layernorm.output`. Gemma-3-it transcoders force-zero positions `[0:4]` (`<bos>`, `<start_of_turn>`, `user`, `\n`) → captured in `STAGE_08_FIRST_TOKEN_MASK`. **Load-bearing for Stage 08b/08c math.**

### Two-graph attribution scheme
Stage 02 emits two attribution graphs per (prompt, condition):
- `multi`: measurement positions `[-5, -3, -2]` (template-anchor span)
- `single`: measurement position `[-2]` only (final-user-token)

Stages 02b/04/07 read both modes; Stage 05 picks `single` by default. Multi is the canonical for the comparative analysis.

### Two Gemma-3 architectures
`Gemma3ForCausalLM` (plain, `model.layers[L]`) vs. `Gemma3ForConditionalGeneration` (multimodal wrapper, `model.language_model.layers[L]`). Stage 06 uses the multimodal wrapper. **Stage 08b must detect at load time** when projecting decoder directions out of `o_proj` / `down_proj`.

### Producer-side packaging (Stage 02c)
Raw `.pt` graphs are ~1.5 GB each (~80 GB per run). The frontend bundle is gzipped JSON (~3 GB). `02c_pack_graphs.py` decouples `.pt → JSON.gz` from full Stage 05 staging so collaborators can fetch the small bundle without ever downloading raw `.pt`.

### Workflow rule (per Mahmoud's preference)
Claude proposes code → Mahmoud implements/edits by hand → Claude runs local tests + GPU smoke → reports findings. No mocks. Real tests, gated `pytest.importorskip` only when needed.

---

## Reference Run — `run_20260422_015552`

50 prompts × 11 conditions, L15 measurement, controlled dataset. **All numbers below are from this run unless noted.**

| Stage | Output | Headline numbers |
|---|---|---|
| 01 | `01_direction/` | 34 layers; \|r_L15\|=3,123.9 (recompute, in-script bf16); separation L32=20,873, L15=3,101 |
| 02 | `02_attribution/` | 50×11×2 = 1,100 attribution graphs, 0 errors |
| 02b | `02b_stats/` | 30 stats blocks (2 modes × 3 comparisons × 5 classes); cognitive_reframe attr drops -51.9% (d=-2.05) under JB, ctrl prefix moves it +1.4% (n.s.) — **direct correlational evidence JB effect is semantic** |
| 03 | `03_verification/` | 50/50 within tolerance; MLP = 0.02% of signal (rest is attention + embed) |
| 04 | `04_labels/` | 1,353 unique features, **100% HF dashboard coverage** |
| 06 | `06_causal/` | **96.7% pro-flip** (87/90 JB→REFUSE), **100% anti** (49/49 bare→COMPLY), **10/10 benign force-refuse**. 100% coherent. **L15 r IS the refusal axis** — bidirectional, generic. 54.6 min wall clock on H100 |
| 07 | `07_subcircuits/` | 18 subcircuits (11 legacy + 7 ctrl-aware); **`jb_specific_frac` per class**: cognitive_reframe **38.6%**, analytical 34.2%, fiction 34.2%, roleplay 20.0%, completion 18.4% — *up to 82% of "JB features" in prior work are prefix-induced, not JB-semantic* |
| 08 | `08_ablation/` | GPU smoke running locally on 5080 (Apr 25) |

Subcircuit sizes (used by Stage 08):
- `universal_refusal_core` = 116 (positive control)
- `ctrl_shared_refusal` = 50 (negative control)
- `jb_fiction_specific_vs_ctrl` = 52
- `jb_analytical_specific_vs_ctrl` = 69
- `jb_cognitive_reframe_specific_vs_ctrl` = 88

**Peak-layer finding**: every refusal subcircuit peaks at **L14**, one layer before our L15 measurement target — clean mechanistic claim worth calling out.

---

## Stage-by-Stage Reference

### Stage 01 — Compute Refusal Directions [✓]
**Role**: per-layer refusal direction `r` (normalized) + `unnormalized_r` (Tejas's intervention magnitude). All 34 layers @ pos=-2; per-position at L15 (-1..-15) for diagnostics.

**Run**:
```bash
PYTHONPATH=src python3 scripts/pipeline/01_compute_direction.py \
    --output-dir data/results/pipeline_runs/run_<ts>/01_direction \
    --n-prompts 64 --layers all
```

**Outputs**: `refusal_direction.pt`, `unnormalized_r.pt`, `directions/layer_XX.pt`, `direction_metadata.json`.

**Findings**:
- L32 sep=20,873 (strongest separation, pre-RMSNorm)
- L15 sep=3,101 (causally effective)
- Cosine(L15, L32) = **−0.115** (near-orthogonal — direction rotates)
- L33 sep=287 (post-RMSNorm collapse — diagnostic, not a usable layer)

**Open**: `|r_L15|` magnitude gap — we get 3,123.9 vs Tejas's 4,019.7 with same methodology. Both produce the bulletproof 10/10 benign force-refuse, so non-blocking, but worth a dataset diff before publication.

---

### Stage 02 — Run Attribution [✓]
**Role**: CLT attribution graphs at L15 for (prompt, condition, mode). Two-graph scheme: `multi=[-5,-3,-2]` and `single=[-2]`.

**Run** (RunPod or local 5080):
```bash
# Single-GPU
PYTHONPATH=src python3 scripts/pipeline/02_run_attribution.py \
    --run-dir data/results/pipeline_runs/run_<ts> \
    --measurement-layer 15 --modes multi,single \
    --batch-size 256 --save-graphs --resume

# Multi-GPU (H100 cluster)
bash scripts/pipeline/run_stage02_parallel.sh <run_dir> <n_gpus>
```

**Outputs**: `02_attribution/attribution_results.json` (per-prompt feature comparisons), `02_attribution/graphs/{idx}_{cond_name}_{mode}.pt` (raw circuit-tracer graphs), `attribution_checkpoint.json`.

**Schema** (consumed by every downstream stage): `results[i].conditions.<cond>.graphs.{multi,single}.{net, pos_sum, neg_sum, top50_features, comparison.{vs_bare, vs_ctrl, ctrl_vs_bare}, ...}`. See HANDOFF.md "Critical schema changes" for the full layout.

**Findings**: 1,100 graphs land cleanly. `bare.multi.net = 7.287` matches `sum(per_target[i].net)` exactly (multi-target weighted-row fix landed in circuit-tracer patch).

---

### Stage 02b — Statistical Analysis [✓]
**Role**: 30 stats blocks (2 modes × 3 comparisons × 5 classes) — paired Wilcoxon, Cohen's d, bootstrap CIs. Plots: separation curve (A3), 34×34 cosine heatmap (A5), per-class distribution (A6).

**Run**:
```bash
PYTHONPATH=src python3 scripts/pipeline/02b_statistical_analysis.py \
    --run-dir data/results/pipeline_runs/run_<ts>
```

**Outputs**: `02b_stats/statistical_analysis.json`, `EXPERIMENT_SUMMARY.md`, separation/cosine/distribution PNGs.

**Findings** (from reference run):

| Class | jb_*_vs_bare %chg | ctrl_*_vs_bare %chg |
|---|---|---|
| cognitive_reframe | **−51.9%** (d=−2.05) | +1.4% (n.s.) |
| fiction | **−34.6%** (d=−2.21) | +5.2% |
| roleplay | −6.2% (d=−0.43) | −3.9% |
| completion | +4.2% (d=+0.43) | +1.8% (n.s.) |

**The matched ctrl prefixes do NOT reduce attribution to refusal — they slightly *increase* it.** JB attribution drops with huge effect sizes (d>2 for 3/5 classes). Direct correlational evidence the JB effect is *semantic*, not formatting.

---

### Stage 02c — Pack Graphs [✓]
**Role**: `.pt → JSON.gz` packer for HF distribution. Decouples conversion from full Stage 05 staging.

**Run**:
```bash
PYTHONPATH=src python3 scripts/pipeline/02c_pack_graphs.py \
    --run-dir data/results/pipeline_runs/run_<ts>
```

**Outputs**: `<run>/graph_data/*.json.gz` + `graph-metadata.json`.

---

### Stage 03 — Attribution Verification (M2) [✓]
**Role**: confirms `sum(feature_attributions) ≈ r · h[L=15]` per prompt. Reports MLP-only contribution fraction (always small — that's the point).

**Run**:
```bash
PYTHONPATH=src python3 scripts/pipeline/03_verify_attribution.py \
    --run-dir data/results/pipeline_runs/run_<ts>
```

**Outputs**: `03_verification/verification.json`, `per_layer_contribution.png` (A4).

**Findings**: 50/50 verified within tolerance. **MLP = 0.02% of signal** on L15-measurement (was ~0.4% on L32). Transcoders only decompose MLP — the rest of refusal lives in attention + embeddings. Per-layer contribution chart (A4) shows assembly across L7–L11 even when measured at L15.

---

### Stage 04 — Feature Labeling (M4) [✓]
**Role**: every unique feature gets labeled from the HF dashboard binary (`mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine/features/{layer}.bin`). Top/bottom logits, activation examples, conditions seen.

**Run**:
```bash
PYTHONPATH=src python3 scripts/pipeline/04_label_features.py \
    --run-dir data/results/pipeline_runs/run_<ts>
```

**Outputs**: `04_labels/feature_labels.json` (1,353 features, 100% labeled), `feature_class_sets.json` (with `per_condition_top50` block — Stage 07's input), `feature_comparison_labeled.json`, A7 UpSet, A8 layer histogram.

**Findings**: `feature_class_sets.per_condition_top50` has all 11 keys → enables Stage 07 ctrl-aware rules. Many top tokens are polyglot/byte-level (Gemma Scope characteristic) — labels alone don't tell the whole story; Stage 05 visualization + concrete prompts are needed. **04b LLM labels (planned)** would synthesize human-readable strings and inject as `clerp` at staging time.

---

### Stage 05 — Visualize Circuits [✓ code; needs end-to-end refresh on `run_20260422_015552`]
**Role**: frontend orchestrator. Stages a viewer with 3-way bare/ctrl/JB overlap coloring, subcircuit filter panel, ablation cart.

**Run**:
```bash
# 1) (one-time per run) fetch raw .pt from HF
PYTHONPATH=src python3 scripts/pipeline/fetch_raw_graphs.py \
    --run run_<ts> --dataset-repo moon70/refusal-lens-graphs

# 2) (one-time) pack to gzipped JSON + push back to HF
PYTHONPATH=src python3 scripts/pipeline/02c_pack_graphs.py --run-dir data/results/pipeline_runs/run_<ts>
PYTHONPATH=src python3 scripts/pipeline/push_graph_data.py \
    --run-dir data/results/pipeline_runs/run_<ts> --source 02c \
    --dataset-repo moon70/refusal-lens-graphs

# 3) stage the frontend
PYTHONPATH=src python3 scripts/pipeline/05_visualize_circuits.py \
    --run-dir data/results/pipeline_runs/run_<ts> \
    --subcircuits-run data/results/pipeline_runs/run_<ts> \
    --mode single --skip-convert --gzip

# 4) serve
cd data/results/pipeline_runs/run_<ts>/05_frontend && python3 -m http.server 8000
```

**Outputs**: `05_frontend/index.html`, `compare.html` (bare↔JB side-by-side), gzipped graph JSONs, subcircuit panel.

**Frontend patches** (`05_frontend_patches/`): `overlap-colors.css` (9 buckets, gold = `shared_with_ctrl` = prefix-induced), `overlap-annotate.js`, `subcircuit-panel.{js,css}`, `compare.html`, `gzip-fetch.js`, `feature-cart.{js,css}` (Stage 08 cart — shift/cmd-click to pin features).

**Pending**: end-to-end refresh on `run_20260422_015552` — Stage 05 code is validated against synthetic + `run_20260417_010035` but the new attribution graphs haven't been rendered yet.

---

### Stage 06 — Causal Intervention (Arditi, Tejas Script 20 port) [✓]
**Role**: at L15, add unnormalized r → flip JB to REFUSE (Arditi); subtract r → flip bare to COMPLY (symmetry); apply same hook to benign prompts → expect 10/10 force-refuse (Tejas's bulletproof control proving the hook is generic).

**Run**:
```bash
PYTHONPATH=src python3 scripts/pipeline/06_causal_intervention.py \
    --run-dir data/results/pipeline_runs/run_<ts> \
    --r-source recompute       # bf16 in-script — Tejas-exact methodology
```

`--r-source` options: `stage01` (use `01_direction/unnormalized_r.pt`), `tejas-rescale` (rescale Stage 01 to Tejas's reported `|r|`), `recompute` (fresh 64+64 diff-in-means under the same bf16 model — chosen for the headline run).

**Outputs**: `06_causal/causal_results.json`, `causal_summary.json`, `flip_rate_by_class.png`, `intervention_symmetry.png`, `FLIP_RATE_SUMMARY.md`.

**Findings** (run_20260422_015552):
- pro-refusal-add: **87/90 = 96.7%** (Tejas got 90/90)
- anti-refusal-sub: **49/49 = 100%** (our symmetry addition; Tejas didn't test)
- benign force-refuse: **10/10 = 100%** (matches Tejas exactly)
- 100% coherence on all 146 flips
- Per-class: analytical 27/27 (100%), roleplay 9/9 (100%), completion 1/1 (100%), cognitive_reframe 32/33 (97%), **fiction 18/20 (90%)** — fiction is the hardest

**Headline causal claim**: bidirectional symmetry + 10/10 benign establishes that `r_L15` IS the refusal axis, manipulable in both directions; not a one-way push, not JB-specific.

---

### Stage 07 — Identify Subcircuits [✓]
**Role**: rule-based set-logic over `feature_class_sets.json` → 18 named subcircuits (11 legacy + 7 ctrl-aware). Computes `jb_vs_ctrl_contrast` headline ICML metric.

**Run**:
```bash
PYTHONPATH=src python3 scripts/pipeline/07_identify_subcircuits.py \
    --run-dir data/results/pipeline_runs/run_<ts>
```

**Outputs**: `07_subcircuits/subcircuits.json`, `subcircuits_summary.json`, `SUBCIRCUITS_REPORT.md`, `jb_vs_ctrl_contrast.png`, `jb_specific_by_layer.png`, treemap, by-layer chart, overlap chart.

**Subcircuit families**:
- **Legacy** (11): `universal_refusal_core`, `canonical_pro_refusal`, `sign_flip_convergent`, `dampening_specialists`, `anti_refusal_amplifiers`, `late_wave_layer24_32` (empty on L15-measurement runs by construction), 5 × `{class}_exclusive`.
- **Ctrl-aware** (7, new): `ctrl_shared_refusal` (prefix-invariant refusal spine), `ctrl_only`, 5 × `jb_{class}_specific_vs_ctrl` (per-class JB-semantic features).

**Headline metric — `jb_specific_frac` per class** (the ICML novelty):

| Class | JB-specific % | Interpretation |
|---|---|---|
| cognitive_reframe | **38.6%** | deepest JB |
| analytical | 34.2% | strong JB-semantic |
| fiction | 34.2% | strong JB-semantic |
| roleplay | 20.0% | mostly prefix artifact |
| completion | 18.4% | mostly prefix artifact |

**Claim**: *up to 82% of what prior work called "JB features" is prefix-induced, not JB-semantic — only the controlled dataset can measure this.*

**Structural identities**:
- `canonical_pro_refusal ∩ sign_flip_convergent` ≈ 86% — JB-recruited refusal IS sign-flipped refusal
- `universal_refusal_core ∩ dampening_specialists` ≈ 85% — dampening attacks the canonical core
- Anti-refusal amplifiers fire ~2.5× more often than dampening specialists (bypass uses general-purpose features, suppression uses specialized rare ones)
- Top anti-refusal amplifier: `L24:F107` (` ok`, ` okay`) — literal "OK, I'll help" feature

---

### Stage 08 — Subcircuit Ablation & Patching [PR #1 done; PR #2/#3 planned]

The patching half. Three sub-stages, three PRs, with a manual-steering frontend deliverable bundled into PR #1.

#### Stage 08a — Runtime ablation [✓ code + 50-prompt run done 2026-04-26; ⚠️ dissociation gates did NOT pass — revisit needed]
**Role**: zero-ablate Stage 07 features at runtime via `ReplacementModel.feature_intervention_generate`. Produces a dissociation matrix per (subcircuit, class).

**Run**:
```bash
# Smoke (5 prompts, ~5–8 min on H100, ~10–15 min on 5080)
PYTHONPATH=src python3 scripts/pipeline/08_ablate_subcircuits.py \
    --run-dir data/results/pipeline_runs/run_20260422_015552 \
    --subcircuits universal_refusal_core,ctrl_shared_refusal,jb_fiction_specific_vs_ctrl \
    --positions all --max-prompts 5 --skip-baseline --resume

# Full (50 × 11 × 5 × 2 = 5,500 generations + baselines from Stage 06)
PYTHONPATH=src python3 scripts/pipeline/08_ablate_subcircuits.py \
    --run-dir data/results/pipeline_runs/run_20260422_015552 \
    --positions both --skip-baseline --resume --checkpoint-every 10
```

**CLI flags**: `--subcircuits` (comma list, default = `STAGE_08_DEFAULT_SUBCIRCUITS`), `--feature-file cart.json` (manual cart override), `--ablation-name`, `--positions {all|anchors|both}`, `--conditions`, `--max-prompts`, `--prompt-{start,end}`, `--skip-baseline` (reuse Stage 06), `--resume`, `--checkpoint-every`.

**Outputs**: `08_ablation/ablation_results.json` (per-prompt records, mirrors Stage 06 schema), `ablation_summary.json` (dissociation matrix), `dissociation_matrix_{all,anchors}.png`, `positions_comparison.png`, `ABLATION_SUMMARY.md`.

**Validation gates** (must hit all three for the NeurIPS claim):

| Gate | Subcircuit | Threshold | 50-prompt result (`run_20260422_015552`) |
|---|---|---|---|
| Positive control | `universal_refusal_core` (116 feat) | bare REFUSE → ≤30/49 (break_rate ≥ 39%) | **21.7%** ✗ |
| Negative control | `ctrl_shared_refusal` (50 feat) | every JB class flip moves ≤5pp | **33.6% avg recovery** ✗ |
| **Dissociation** | ≥2 of `jb_{fiction,analytical,cognitive_reframe}_specific_vs_ctrl` | each shows **≥+20pp** higher recovery on its own class than other-class avg | **all 6 deltas negative (−7 to −20pp)** ✗ |

##### Stage 08 revisit — REVISIT NEEDED before progressing to 08b/08c

The 50-prompt run shows the Stage 07 subcircuit extraction produces feature sets that are correlationally selective for their target class (fiction set fires 12× more on fiction than other classes; analytical 4×; cognitive_reframe 3–4×) **but ablating those sets does not preferentially break their target class**. All 6 dissociation deltas came out negative.

**Why this is suspicious**: Anthropic's *On the Biology of Large Language Models* (2024) demonstrates that MLP-feature interventions on a CLT can suppress refusal/jailbreak behavior. Our negative result is therefore most likely an artifact of how our subcircuits are defined, **not** a fundamental limitation of MLP-mediated patching. The Stage 03 finding that "MLP carries 0.02% of L15 refusal signal" is a useful priors-check on attribution-to-`r`, but it does NOT preclude MLP ablation from breaking jailbreaks — a feature can carry tiny attribution magnitude and still be causally pivotal (the class-specific circuit can be small).

**Hypotheses for what's wrong with the current rules**:
1. **Corpus-aggregated top-50 set logic is too coarse.** A feature is in `jb_fiction_specific_vs_ctrl` if it appears in fiction's corpus top-50 but not the matched ctrl's. This produces statistical separability but does not guarantee per-prompt activation overlap; on any given fiction prompt, the relevant features may be in the prompt's individual top-N but outside the corpus top-50. Try **per-prompt rules** (feature must appear in ≥X individual fiction prompts' top-N).
2. **Attribution target is logits, not the L15 refusal direction.** Stage 02 attributes to refusal-token logits / final-layer projection. Georg's request is to expose attribution **directly to the L15 refusal direction** (project gradient onto `r_L15`). Re-deriving subcircuits against attribution-to-`r_L15` likely produces a different (and more causal) ranking.
3. **Ctrl-aware subtraction may be removing the wrong features.** The `_specific_vs_ctrl` rule subtracts ctrl-class top-50 from JB-class top-50, on the theory that "shared with ctrl = prefix-induced, not JB-semantic." This may be overcorrecting and removing genuinely causal refusal features that happen to also appear under matched ctrl prefixes.
4. **Layerwise filtering is missing.** All Stage 07 subcircuits draw from layers 0–33 indiscriminately. Refusal subcircuits peak at L14 (per Stage 07 finding); a layer-restricted version (e.g. features in `[L13, L14, L15]` only) may produce a tighter causal set.
5. **Threshold for inclusion is arbitrary.** Top-50 per condition was chosen by analogy to Stage 02 visualization; smaller (top-10/top-20) may be more conservative and causal, larger (top-100) may capture distributed-but-relevant features.
6. **Feature-set size confound.** Bare-break rate correlates with feature count in our run (universal 116→22%, analytical 69→17%, fiction 52→4%). Set sizes should be **matched** before claiming class-specificity — currently not controlled.

**Action items for the revisit** (do these before Stage 08b/08c):
- (a) Re-run Stage 07 with per-prompt selection rules instead of corpus-aggregated top-50, and rerun Stage 08a on the new sets.
- (b) Build the L15-`r`-attribution dashboard variant Georg requested, re-derive Stage 07 subcircuits against attribution-to-`r_L15`, rerun 08a.
- (c) Add a layer-restricted variant (e.g. features only at L13–L15, where the refusal signal peaks) and rerun.
- (d) Add a size-matched random-feature negative control (sample N random features per ablation set, where N = target subcircuit size); the random control's recovery rate is the noise floor against which class-specific deltas should be compared.
- (e) Reproduce a published MLP-suppression result from Anthropic's *On the Biology of Large Language Models* on Gemma-3-4b-it as a sanity check that our 08a harness is wired correctly. If we can replicate their result, the issue is in our rules; if we can't, the issue is in our harness.

**Until at least one of (a)–(d) produces a passing dissociation gate, treat 08a as inconclusive, not negative.** Do not commit GPU time to 08b/08c (which build on top of the same feature sets) until a passing 08a baseline exists. The full report and figures from the failed run are kept under `<run>/08_ablation/` for the paper appendix and as the "before" against which revisits will be compared.

#### Stage 08a — Manual feature steering [✓]
Frontend cart in Stage 05 + FastAPI demo backend.

**Cart**: shift/cmd-click feature nodes → toggles into right-rail cart. Buttons: Export `cart.json`, Copy CLI, Run Ablation (POSTs to `localhost:8080`), Clear.

**Server**:
```bash
pip install fastapi pydantic uvicorn
PYTHONPATH=src python3 scripts/pipeline/ablation_server.py --host 127.0.0.1 --port 8080
# Loads ReplacementModel singleton ~60–120s
```

`POST /ablate` returns `{baseline, baseline_cls, baseline_coherent, ablated, ablated_cls, ablated_coherent, n_features, positions, elapsed_s}`. CORS open for `localhost:8000` (frontend).

#### Stage 08b — Permanent weight editing (Arditi-style) [PLANNED, NeurIPS-load-bearing]
**Role**: project decoder directions of ablated features out of `o_proj` and `down_proj` at every target layer (per CLT). Produces an **edited HF model** that runs without the replacement model at inference.

Approach (CLT-correct):
1. Resolve features `[(k, j), ...]` from subcircuits + optional cart.
2. Per source layer `k`, fetch decoder vectors `D[k, j] ∈ R^{(N-k)×d_model}`. Group by **target layer** `L = k + offset`.
3. Per target `L`: SVD-orthonormalize `D_L`, build projector `P_L = Q Q^T`, apply to `down_proj` and `o_proj` rows: `W ← W − P_L @ W`.

**Pitfalls**: skip `embed_tokens` / `lm_head` (shared, scope creep); leave `post_*_layernorm.weight` (γ) untouched (RMSNorm scalar-multiplicative — projector commutes); detect `Gemma3ForCausalLM` vs `Gemma3ForConditionalGeneration` at load.

**Outputs**: `<run>/08b_direction_ablation/<set_name>/edited_model/` (HF `save_pretrained` dir), `metadata.json` (per-layer Frobenius `‖ΔW‖_F / ‖W‖_F`), `eval/flip_rate_summary.json` (Stage 06 rerun on edited model), `eval/mmlu.json` (300-prompt sanity check).

**Acceptance**: target-class JB flip drops ≥15pp; other classes stable ±5pp; MMLU delta < 3pp.

#### Stage 08c — Input-dependent sidecar [PLANNED]
**Role**: `SidecarWrapper(nn.Module)` wraps a plain HF Gemma and at every layer subtracts the feature reconstructions an MLP would have emitted — mathematically equivalent to 08a's `feature_intervention(value=0.0)` for the same feature set.

Hooks:
- pre-hook on `block[k].pre_feedforward_layernorm.output`: compute `ReLU(W_enc[k,j] @ h + b_enc[k,j])`, zero positions [0:4], cache.
- post-hook on `block[L].post_feedforward_layernorm.output`: for each `(k, j, offset)` with `k+offset == L`, subtract `D[k, offset, j, :] * cache[k][j]`.

**Equivalence test (must pass before claim)**: same prompt, both paths → `torch.allclose(logits_rm, logits_wrap, rtol=1e-3)` (bf16 tolerance).

**Comparison output** (`08b_vs_08c_compare.md`): flip-rate delta per class, MMLU delta, param-count delta, generation speed, deploy-as-single-HF-dir (08b yes / 08c no, custom wrapper).

---

## What's Done vs. What's Left

### Done
- **Correlational pipeline (01–04, 07)**: validated end-to-end on `run_20260422_015552`. ICML headline numbers in hand.
- **Stage 02c** producer-side packaging.
- **Stage 06** causal intervention: 96.7% / 100% / 100% bidirectional + benign control.
- **Stage 07** ctrl-aware subcircuits + `jb_specific_frac` headline metric.
- **Stage 08a** runtime ablation + manual cart + FastAPI backend (PR #1, code complete, 13/13 unit tests pass, **50-prompt GPU run done 2026-04-26 — dissociation gates failed; revisit Stage 07 rules before continuing**).
- **Stage 05** code (frontend patches, 3-way overlap, subcircuit panel, ablation cart).

### Pending (priority order)
1. **Stage 05 frontend refresh** on `run_20260422_015552` — needs the L15 raw graphs from HF reformatted via `02c → push → 05_visualize` and a browser spot-check. Code's ready; just plumbing the round-trip. Visual acceptance criteria in HANDOFF §1.
2. **Stage 08 revisit** — see "Stage 08 revisit" section above. The 50-prompt run failed all 3 validation gates; before any 08b/08c GPU time, re-derive Stage 07 subcircuits with at least one of: per-prompt rules, attribution-to-`r_L15`, layer-restricted, or matched-size random-baseline. Reproduce an Anthropic *Biology of LLMs* MLP-suppression result on Gemma-3-4b-it as a harness sanity check.
3. **Stage 08b** — Arditi-style permanent weight edit. NeurIPS-load-bearing. **Blocked on a passing 08a baseline.**
4. **Stage 08c** — sidecar wrapper. Equivalence test must pass before claim. **Blocked on a passing 08a baseline.**
5. **L15-direction attribution dashboard variant** (per Georg's Apr 26 ask) — expose attribution projected onto `r_L15` as a Stage 02 mode and a Stage 05 viewer toggle. Feeds the Stage 08 revisit (option b).
6. **Stage 04b** — Claude API LLM labels (~$2 Haiku 4.5). Drops human-readable `clerp` strings into the frontend. Post-ICML.
7. **`|r_L15|` magnitude diagnostic** — diff `harmful_train.json` between branches to explain the 3,123.9 vs 4,019.7 gap. Non-blocking; for publication rigor.

---

## Open Questions / Gaps

| # | Question | Where it gets answered |
|---|---|---|
| 1 | Will any revised Stage 07 rule produce a passing Stage 08a dissociation gate? | Stage 08 revisit (rules a–e in `Stage 08a` section above) |
| 2 | Why does fiction flip at only 90% under Stage 06 when the other classes hit 97–100%? Narrative-distributed attention? | Pull the 2 unflipped fiction prompts (`causal_results.json`), look for structural commonalities |
| 3 | Is Stage 08b's Arditi projection mathematically equivalent to 08a's runtime ablation in the limit of all-positions? | 08c equivalence test (must pass `torch.allclose(logits_08a, logits_08c)` on bf16 tolerance) |
| 4 | Does the edited 08b model preserve general capabilities? | MMLU delta < 3pp on 300-prompt subset |
| 5 | `\|r_L15\|` magnitude gap (3,123.9 vs Tejas's 4,019.7) | Dataset diff between `l15-refactor` and `origin/tejas-circuit-experiments` |
| 6 | What does L20 do? Why is it the only negative-contribution layer? | Open — dedicated probe |
| 7 | What features live in Regime A (L5–L13) vs C (L18–L33)? Weakly anti-correlated. | Stage 07 cluster inspection (Approach B with HDBSCAN — deferred) |
| 8 | What happens at the L13→L14 pivot? Specific attention head triggering rotation? | Attention-head attribution (S3 task — not in current plan) |
| 9 | 99.98% of the refusal signal lives in attention + embeddings on L15 measurement — which heads? | Attention-head attribution |
| 10 | Are any two JB classes structurally similar (shared `jb_*_specific_vs_ctrl` features)? | Stage 07 cross-tab — easy follow-up |
| 11 | Will manual cart steering produce coherent ablations the cart's UX implies? | Live demo of `ablation_server.py` — only verifiable post-08a-validation |
| 12 | Generalize to Qwen? | **Ruqiya's track** (not Mahmoud's) |

---

## Workflow Conventions

### Testing
```bash
# Local (no GPU required)
PYTHONPATH=src .venv/bin/python3 scripts/pipeline/tests/test_pipeline_local.py --stage all

# Single stage
PYTHONPATH=src .venv/bin/python3 scripts/pipeline/tests/test_pipeline_local.py --stage 08

# Stages: utils, utils-viz, 01, 01-a5, 02, 02b, 03, 03-a4, 04-a7, 04-a8, 04-schema,
#         06, 07, 07-ctrl, 08, all
```

Local venv: `.venv/` at repo root. CPU torch + numpy + matplotlib + transformers + nnsight + circuit-tracer (via `pip install -e ".[runpod]"` for full GPU stack).

### Local-dev limitations
- Stage 08 unit tests run with **just base venv** (pure Python helpers).
- Stages 01/02/03/06/07's tests need numpy + matplotlib (now installed).
- Stage 04-a7 has a known pre-existing `upsetplot 0.9.0` × `matplotlib 3.10.9` `nan` RGBA bug — unrelated to current work; needs upsetplot upgrade or matplotlib pin.

### Code style
- Heavy imports (torch, circuit_tracer, numpy, matplotlib) inside functions or `main()`, not at module top-level. Keeps test-time imports cheap.
- Each stage mirrors the Phase 0/1/2/3 + checkpoint/resume + summary-MD pattern from Stage 06.

### Git
- All work on `l15-refactor`. Don't touch `main` without approval.
- Submodule `vendor/circuit-tracer` on `refusal-lens-measurement-patch` (commit `b5300ee`).
- Don't commit raw `.pt` graphs (`.gitignore` excludes them); push to HF instead.

---

## Where to Find Things

| What | Path |
|---|---|
| Pipeline scripts | `scripts/pipeline/` |
| Latest validated run | `data/results/pipeline_runs/run_20260422_015552/` |
| Legacy L32 reference run | `data/results/pipeline_runs/run_20260417_010035/` |
| Stage 06 results | `<run>/06_causal/FLIP_RATE_SUMMARY.md` |
| Stage 07 results | `<run>/07_subcircuits/SUBCIRCUITS_REPORT.md` |
| Controlled dataset | `dataset/refusal_lens_controlled_dataset.json` |
| Local tests | `scripts/pipeline/tests/test_pipeline_local.py` |
| Frontend patches | `scripts/pipeline/05_frontend_patches/` |
| Stage 08 plan (full) | `/home/mshab/.claude/plans/curried-discovering-giraffe.md` |
| Tejas's bulletproof source (Stage 06) | `git show origin/tejas-circuit-experiments:data/tejas_experiments/scripts/20_bulletproof_pipeline.py` |
| HF dataset (graphs) | `https://huggingface.co/datasets/moon70/refusal-lens-graphs` |
| Session handoff | `HANDOFF.md` |
| Pipeline README (deployment + findings narrative) | `scripts/pipeline/README.md` |

---

## Final Note

The correlational pipeline (01/02/02b/03/04/07) is complete and validated on real data. The causal pipeline (06) is complete and validated. The patching pipeline's first step (08a) is code-complete and ran end-to-end on 50 prompts; **all three validation gates failed**, but the most plausible read is that the Stage 07 subcircuit-extraction rules — not the harness or the MLP path itself — are the wrong target. Anthropic's *On the Biology of Large Language Models* shows MLP-feature interventions DO suppress refusal/jailbreak behavior, so the path forward is to revisit how features are filtered into subcircuits (per-prompt rules, attribution-to-`r_L15`, layer-restricted, size-matched random control) before committing GPU time to 08b/08c.
