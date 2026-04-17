# Refusal-Lens Unified Pipeline Plan

**Created**: 2026-04-14
**Last updated**: 2026-04-17 (post first successful 50-prompt RunPod run through Stage 04)

## Goal

Combine Mahmoud's correlation/attribution work with Tejas's causal intervention work into one reproducible pipeline that tells Georg's research story: *"Here are the circuits, they're real, here's what they encode, and here's what happens when you manipulate them."*

---

## Pipeline Layout

```
scripts/pipeline/
├── config.py                      # Shared constants (D_MODEL=2560, 34 layers)
├── utils.py                       # Shared utilities
├── 01_compute_direction.py        # Per-layer refusal directions        [DONE ✓]
├── 02_run_attribution.py          # CLT attribution graphs              [DONE ✓]
├── 02b_statistical_analysis.py    # Stats + plots for attribution       [DONE ✓]
├── 03_verify_attribution.py       # M2: attribution = dot product check [DONE ✓]
├── 04_label_features.py           # M4: HuggingFace dashboard labeling  [DONE ✓]
├── 05_visualize_circuits.py       # M3: color-coded graph visualization [NEXT]
├── 06_causal_intervention.py      # Arditi method (Tejas Script 16)
├── 06b_linear_probe.py            # Linear probe readability vs causality
├── 07_identify_subcircuits.py     # Cluster functional subcircuits
└── 08_ablate_subcircuits.py       # Targeted subcircuit ablation

scripts/pipeline/tests/
├── test_pipeline_local.py         # Local validation (60/60 passing)
└── test_runpod_1_4.py             # RunPod integration (stages 01→04)
```

Results land in `data/results/pipeline_runs/run_YYYYMMDD_HHMMSS/`.

---

## Critical Design Decisions

### Per-layer directions
Tejas established that the refusal direction **rotates across layers** — L32 has strongest separation (20,873) but L15 is causally effective (3,101). Cosine(L15, L32) = −0.115 in our n=64 sample, confirming the directions are near-orthogonal. Stage 01 computes `r` at **all 34 layers**; downstream causal stages use `r[layer]` when intervening.

### D_MODEL = 2560
Gemma-3-4b-it hidden dim is 2560 (not 2304 as an older memory had). Pipeline reads this from model config, but the constant is fixed in `config.py`.

### Feature Comparison → Labeling
Stage 02 emits feature-comparison buckets (shared / sign-flipped / dampened / amplified-anti). Stage 04 labels **every unique feature** in the union of top-50 and comparison data, then merges the labels into the comparison buckets (`feature_comparison_labeled.json`).

### Label source: HuggingFace dashboard binary (not Neuronpedia)
Neuronpedia does not index gemma-3-4b-it. We read `mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine/features/{layer}.bin` directly via byte-range HTTP requests using the `index.json.gz` offsets. Each feature's payload (top/bottom logits, activation examples) is used to construct its label.

### Dataset ingestion (future)
Currently random selection via `select_diverse_prompts()`. `utils.load_experiment_dataset()` exists as the future swap point for curated dataset JSON — will be wired in once M1 (clean 50-prompt dataset) is ready.

### Workflow rule
Claude suggests code → Mahmoud implements by hand → Claude runs local tests and reports bugs → Mahmoud fixes → full RunPod run → Claude reviews results. No mocks — real tests, `pytest.importorskip("torch")` for gating when needed.

---

## Stage Status (as of 2026-04-17)

### Stage 01 — Compute Refusal Directions ✓
- **Local**: 26/26 tests passing
- **RunPod (n=64)**: L32 sep=20,873 (prior 20,644; Tejas 20,827); L15 sep=3,101 (Tejas 3,131); L33=287 (pre-RMSNorm artifact captured as expected)
- Cosine(L15,L32) = **−0.115** (near-orthogonal direction rotation) — flag for mentor
- Outputs: `01_direction/refusal_direction.pt`, `unnormalized_r.pt`, `directions/layer_XX.pt`, `direction_metadata.json`

### Stage 02 — Run Attribution ✓
- 50 prompts × 5 JB classes × bare = 300 attribution graphs
- Outputs: `02_attribution/attribution_results.json`, `feature_comparison_aggregate.json`, `attribution_checkpoint.json`

### Stage 02b — Statistical Analysis ✓
- Outputs: `statistical_analysis.json`, `EXPERIMENT_SUMMARY.md`, four PNGs
- **Key results** (50 prompts):

  | Class | Δnet | % | p (Wilcoxon) | Cohen's d | Consistency |
  |---|---|---|---|---|---|
  | Roleplay | −38.7 | −54.9% | 1.4e-8 *** | −0.91 | 42/50 |
  | Fiction | −65.3 | −92.7% | 3.0e-13 *** | −1.57 | 47/50 |
  | Analytical | −73.7 | −104.6% | 5.3e-15 *** | −2.37 | 49/50 |
  | Completion | +5.0 | +7.2% | 0.011 * | +0.27 | 15/50 |
  | Cognitive_reframe | −50.2 | −71.3% | 2.5e-14 *** | −1.41 | 49/50 |

- Completion's dual mechanism: dPos=+19.7 (pro-refusal recruitment), dNeg=−14.6 (anti-refusal weakening) → net positive

### Stage 03 — Attribution Verification (M2) ✓
- MLP ratio = 0.404% (prior 0.35–0.39%) — transcoders decompose ~0.4% of the refusal signal; the rest is attention + embeddings
- attr_net_mean=70.47 matches 02b bare exactly → key paths plumb through correctly
- Per-layer decomposition shows early-layer buildup (L7–L11 contribute ~2400 to dot product) — motivates a dedicated plot (see gap #4)

### Stage 04 — Feature Labeling (M4) ✓
- 876 unique features, **100% labeled** (788 priority features all labeled)
- Sources: top-50 per condition ∪ sign-flipped (603) ∪ dampened (115) ∪ amplified-anti (117)
- Caveat: many top-token patches are polyglot/byte-level noise. Expected for Gemma Scope; means labels alone won't tell the full story — Stage 05 visualization + activation examples on concrete prompts will be needed.

---

## Added Metrics & Plots (from 2026-04-17 post-run review)

These were not in the original plan. They're divided into **scientifically load-bearing** (add before mentor meeting) and **narrative-improving** (nice-to-have).

### Load-bearing — add to 02b or a dedicated stage

**A1. Response coherence + refusal classification per prompt** *(goes into Stage 02)*
- Run `utils.classify_response` and `utils.is_coherent` over bare + each JB generation
- Attach `bare_refused`, `jb_refused`, `bare_coherent`, `jb_coherent` booleans to each result row
- **Why**: attribution deltas are only meaningful if the model's output actually changes — if bare refuses and JB still refuses, the attribution drop is disconnected from behavior. This is the single most load-bearing missing check.
- **Status**: Mahmoud has flagged this as planned for later in the pipeline; tracked here for completeness.

**A2. Completion-paradox feature attribution** *(Stage 02b table + plot)*
- Top-10 features with largest positive Δattribution under completion (features being *recruited* to strengthen refusal)
- Compare to top-10 dampened features under the three suppressive classes
- Render as side-by-side labeled bar charts
- **Why**: we know dPos=+19.7 is happening but not which features cause it. This directly answers task S2 (completion paradox deep-dive) and is the most research-interesting single figure.

### Narrative-improving — 10-minute matplotlib additions to 02b

**A3. Separation-vs-layer curve**
- X: layer index 0–33, Y: separation magnitude
- Already have the data in `01_direction/direction_metadata.json`
- **Why**: makes the L33 pre-RMSNorm collapse (287 vs L32's 20,873) visually undeniable; shows monotonic buildup

**A4. Per-layer attribution contribution bar chart**
- Stage 03 already computes it for 10 prompts but never visualizes
- Aggregate mean contribution per layer across all 50 prompts, render as bar chart
- **Why**: reveals where the refusal signal is *assembled* — our data hints at L7–L11 contributing more than L32 does, which reshapes how we'd design interventions

**A5. Full 34×34 cosine heatmap**
- We currently store only 6 pairwise cosines (L15/18/25/32). Compute full matrix.
- **Why**: Georg explicitly asked about rotation. A heatmap shows exactly where the direction pivots (probably L14–L20) and gives a crisp answer

**A6. Bare-vs-JB net-attribution distribution plot**
- Violin or box plot per class on the same axes
- **Why**: means/CIs don't show the tail behavior. Important for completion especially (35/50 strengthen, 15/50 weaken — a bimodal pattern hidden by the mean)

**A7. Feature-class UpSet / Venn**
- The "Classes" column in 04 already encodes which JB classes each feature appears in
- Render as UpSet (preferred over 5-way Venn)
- **Why**: distinguishes universal refusal-circuit features from JB-class-specific ones — directly serves M3 (circuit visualization)

**A8. Top-features-by-layer histogram**
- Histogram of where sign-flipped / dampened / amplified-anti features concentrate by layer
- **Why**: prior finding said JB effect concentrates in L24–L32. Confirm or refute with the 50-prompt data.

### Suggested placement

| Addition | Stage | Rationale |
|---|---|---|
| A1 coherence | 02 | Needs model generations, so compute at attribution time |
| A2 completion features | 02b | Uses Stage 02's per-condition feature data |
| A3 separation curve | 02b or 01 | Data lives in 01 metadata; plot is a stats/visualization concern |
| A4 per-layer contribution | 03 | 03 already has the data per-prompt; aggregate and plot |
| A5 cosine heatmap | 02b or 01 | Derivable from 01 directions; fits better with 02b report |
| A6 distribution plot | 02b | Standard statistical addition |
| A7 UpSet | 04 or 05 | Belongs with visualization but derivable from 04 data |
| A8 layer histogram | 04 | Uses 04's by-layer grouping already present |

**Recommendation**: bundle A3, A5, A6 into 02b (easy), A4 into 03, A7+A8 into 04. A1 waits for Stage 02 enhancement. A2 is its own short script under 02b or as a sub-analysis.

---

## Downstream Stages (unchanged from original plan)

### Stage 05 — Visualize Attribution Circuits (M3)
**Depends on**: 02, 02b, 04
**Outputs**: color-coded circuit diagrams with labeled nodes, shared-vs-JB-only coloring, subcircuit candidates
**Research story**: "Fiction fundamentally reorganizes the circuit, while completion preserves its skeleton."

### Stage 06 — Causal Intervention (Arditi)
**Depends on**: 01
**Owner**: Tejas
**Source**: Tejas Script 16
**Outputs**: control + jailbreak flip rates per layer; exact-magnitude ablation on all positions, every forward pass, unnormalized r
**Research story**: "Refusal direction is causally sufficient at L15, despite L32 having 7x stronger separation. Separation ≠ causation."

### Stage 06b — Linear Probe
**Depends on**: 01
**Outputs**: per-layer probe accuracy on harmful-vs-harmless
**Research story**: "Refusal is linearly readable everywhere from L9 onward, but only causally mutable at L15."

### Stage 07 — Identify Subcircuits
**Depends on**: 02, 04, 05
**Outputs**: named subcircuit definitions (dampening, tug-of-war, JB-detection)
**Research story**: "Three functional subcircuits with distinct roles."

### Stage 08 — Ablate Subcircuits
**Depends on**: 07, 01
**Outputs**: per-subcircuit ablation results
**Research story**: "Ablating the dampening subcircuit removes X% of fiction's bypass effect."

---

## Execution Order & Dependencies

```
01 ──┬──→ 02 ──→ 02b ──┬──→ 04 ──→ 05
     │                 │
     │                 └──→ 03
     │
     └──→ 06 ──→ 06b

04 + 05 + 06 ──→ 07 ──→ 08
```

---

## Progress Table

| Stage | Status | Local Test | RunPod Test | Notes |
|-------|--------|------------|-------------|-------|
| 01    | ✓ Done | 26/26      | ✓ n=64      | Matches Tejas within noise |
| 02    | ✓ Done | via S03    | ✓ 50 prompts × 5 classes | Checkpoint-resumable |
| 02b   | ✓ Done | Matches reference | ✓ 50 prompts | Stats match prior exactly |
| 03    | ✓ Done | 15/15      | ✓ 50 prompts | MLP ratio 0.404% |
| 04    | ✓ Done | —          | ✓ 876 features, 100% labeled | HF dashboard binary |
| 02→A1 | Planned| —          | —           | Coherence wiring — Mahmoud queued |
| 02b→A2–A6 | Planned | — | —       | Plot additions from 2026-04-17 review |
| 04→A7–A8 | Planned | — | —        | Plot additions from 2026-04-17 review |
| 05    | Planned| —          | —           | M3 visualization |
| 06    | Planned| —          | —           | Tejas's task |
| 06b   | Planned| —          | —           | |
| 07    | Planned| —          | —           | |
| 08    | Planned| —          | —           | |

---

## Next Milestones

1. **Plot additions A3/A5/A6 in 02b + A4 in 03 + A7/A8 in 04** — one batch of matplotlib work, improves narrative without blocking Stage 05
2. **A2 completion-paradox feature attribution** — research-interesting figure, small script
3. **A1 response coherence** — wire into Stage 02 (queued by Mahmoud)
4. **Stage 05 visualization** — once plot additions land
5. **Stage 06 causal intervention** — Tejas's parallel track
6. **M1 clean 50-prompt dataset** — deferred until pipeline is complete
