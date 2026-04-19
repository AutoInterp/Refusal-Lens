# Refusal-Lens Unified Pipeline Plan

**Created**: 2026-04-14
**Last updated**: 2026-04-19 (Stage 07 rule-based subcircuits complete; 10-prompt local run complete; Stage 05 subcircuit filter panel queued)

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

### Stage 05 feature labels via LLM synthesis (proposed 2026-04-18, deferred)
Circuit-tracer's `create_graph_files` leaves `clerp=""` on feature nodes. Anthropic's demos use hand-written labels which doesn't scale to 876 unique features across our 50 prompts. Plan: an offline labeler that, per feature, assembles `(top_logits, bottom_logits, activation_example_quantiles, top-5 incoming attribution edges, top-5 outgoing attribution edges)` → single Claude API call → 5–10 word description → store as `llm_label` in `feature_labels.json` → `utils_viz` injects into each graph's `clerp` at conversion time. Estimated cost: ~$2 total with Haiku 4.5 (876 × one-shot calls, no caching needed). Deferred until after 50-prompt pipeline validated end-to-end and Georg weighs in on whether this is worth the run cost.

### Stage 05 side-by-side comparison viewer (proposed 2026-04-18)
Currently the frontend shows one graph at a time. Research workflow demands simultaneous bare ↔ JB viewing. Plan: new `compare.html` in `05_frontend_patches/` with two iframes + URL-param `?graph=<slug>` support via a small patch to `init-cg.js`. Each iframe is an independent copy of the vanilla viewer — shared data, separate pan/zoom/feature-detail state. Deferred until the 10-prompt RunPod subset validates the core pipeline end-to-end.

### Stage 05 storage strategy (2026-04-18, revisit after Georg feedback)
Attribution graphs are much larger than expected: **~1.5 GB per raw `.pt`, ~40 MB per pruned frontend JSON** (with `node_threshold=0.8`, `edge_threshold=0.98`). For 50 prompts × 6 conditions = 300 graphs: ~430 GB raw `.pt`, ~12 GB pruned JSON.

**Current decision** — optimize for *information fidelity* over storage:
- **No tighter pruning.** Lowering thresholds (e.g. `0.6 / 0.9`) would disproportionately drop low-influence features — exactly the class-exclusive "JB subcircuit" features we want to highlight.
- **Transport compression (gzip) instead of pruning.** `.json` gzips ~5–8× in practice → ~5–8 MB per graph. No information loss.
- **Raw `.pt` files are pod-only for now.** Not committed; not synced. Revisit if Georg asks for them (options at that point: `scp` to local archive, or push to HF datasets under `AutoInterp/refusal-lens-graphs`).
- **Pruned JSONs are the canonical artifact.** Hosted (not committed to git) — planned target: HuggingFace dataset repo, frontend fetches directly. Falls back to local `graph_data/` for local dev.

**If storage is still a problem after gzip:** options in escalation order — (1) HF-hosted JSONs + frontend fetches from HF URL, (2) git-LFS for a committed subset (10 representative prompts), (3) accept stricter thresholds as last resort, carefully documenting what's dropped.

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

**Approach A (rule-based, chosen as default 2026-04-18)** — interpretable, narratively defensible:
- `universal_refusal_core` = features in all 5 JB classes ∩ bare top-50 (~100 features, from A7 analysis)
- `canonical_pro_refusal` = features in 5-class intersection but NOT in bare pool (~50 features; shared suppression target)
- `{class}_exclusive` (×5) = features unique to one JB class (43–107 each, from A7)
- `dampening_specialists` = features in `dampened` bucket across ≥3 classes (~50)
- `anti_refusal_amplifiers` = features in `amplified_anti` bucket across ≥3 classes (~60)
- `late_wave_layer24_32` = all features in L24–L32 band (cross-cuts above; ~630)

Implementation: pure set logic on `feature_class_sets.json` + `layer_histogram.json` + `feature_comparison_labeled.json`. No GPU, no ML fitting. Local test derives all from existing `run_20260417_010035` outputs.

**Approach B (embedding-based, alternative, discuss with Georg)** — deferred, pending mentor input:
- Build per-feature vector: `(class_membership_5d, layer_1d, mean_attribution_1d, max_activation_1d, top_logit_embeddings_Nd)`
  - `top_logit_embeddings` = average of Gemma/Qwen sentence-embedding vectors for the top-5 promoted tokens, capturing semantic content of what the feature encodes
- Dimensionality reduction: PCA to ~20 dims, then UMAP to 2D for visualization
- Cluster with HDBSCAN (no k specified; cluster count determined by density)
- Name clusters post-hoc by inspecting the most-central features per cluster (highest local density)

Trade-offs vs rule-based:
- **Pros**: finds unexpected feature groupings we didn't predict; captures semantic similarity (e.g. "all safety-warning features" clustering together regardless of which classes they appear in); more likely to generalize to new JB classes
- **Cons**: cluster identity is statistical, not interpretable by inspection; hard to explain to Georg without a UMAP plot + handpicked labels; requires careful tuning (HDBSCAN min_cluster_size, min_samples); may produce ~10–20 small clusters instead of 6–8 big ones
- **Open question for Georg**: does he want us to run both and compare? Or commit to one?

**Plan**: ship Approach A first (rule-based). If Georg wants Approach B, add a `--method=embedding` flag to `07_identify_subcircuits.py` that runs HDBSCAN on the same input data and writes a parallel `subcircuits_embedding.json`. Both outputs feed into Stage 08 ablation independently — we can ablate by either subcircuit definition.

### Stage 07 results (2026-04-19)

11 subcircuits identified on `run_20260417_010035` (50 prompts). 834/876 features tagged. Two structural identities surfaced:
- `canonical_pro_refusal ∩ sign_flip_convergent` = 48/56 (**86%**) — "JB-recruited refusal ≡ sign-flipped refusal"
- `universal_refusal_core ∩ dampening_specialists` = 44/52 (**85%**) — "dampening attacks the canonical core, not a separate circuit"
- `canonical_pro_refusal ∩ dampening_specialists` = 2/52 (**4%**) — the two mechanisms use disjoint features, ablatable independently
- Temporal sequence in layer peaks: anti-refusal amplifiers (L25) → universal core (L29) → dampening + sign-flip (L30) → canonical pro-refusal (L32)
- Anti-refusal amplifiers fire 2.5× more often than dampening specialists (0.0078 vs 0.0031 mean activation frequency) — bypass uses general-purpose features, suppression uses specialized rare ones
- `L24:F107` (' ok', ' okay') is the top anti-refusal amplifier — a literal "OK, I'll help" compliance feature the model recruits to bypass refusal
- Top-3 by |attribution| of `universal_refusal_core` == top-3 of `dampening_specialists` — JBs attack the strongest bare-refusal features first

Full results + figures in `README.md` Section 9 + `07_subcircuits/SUBCIRCUITS_REPORT.md`.

### Stage 05 subcircuit filter panel (proposed 2026-04-19, Plan A)

Now that Stage 07 emits 11 named subcircuits, the frontend needs per-subcircuit highlighting in the graph viewer. Three options, implementing A first:

**Plan A (chosen 2026-04-19)** — right-rail checkbox panel, composes with existing overlap coloring:
- `utils_viz.annotate_subcircuits(graph_json, subcircuits_json)` attaches `subcircuits: [...]` membership array to each feature node at stage_frontend time
- `05_frontend_patches/subcircuit-panel.{js,css}` adds a collapsible panel listing the 11 subcircuits with counts; click-to-filter dims non-member nodes, hover highlights members
- Keeps the existing shared/jb_unique/bare overlap colors — subcircuit filter is a dim/highlight layer, not a color conflict
- Works in both `index.html` and `compare.html` (applied to each iframe independently)

**Plan B (later, if requested)** — radio "color by overlap" vs "color by subcircuit". Subcircuit-color mode uses a distinct palette (universal=green, canonical=orange, sign-flip=red, dampening=blue, anti-amp=purple, class-exclusive=shade-per-class). Multi-membership nodes need tie-break (first-match priority order or pie-slice encoding). Deferred — subcircuits overlap, and Plan A's dim/highlight avoids the tie-break problem.

**Plan C (deferred)** — dedicated `subcircuits.html` viewer with left-sidebar list and per-subcircuit preset views. Bigger lift; only pursue if Plan A proves insufficient for Georg's workflow.

### Stage 05 gzip compression approach (2026-04-19)

Two deploy targets differ in fetch semantics:

- **HuggingFace dataset host (preferred for the 50-prompt ship):** HF serves `.json.gz` with `Content-Encoding: gzip` and browsers auto-decode — frontend fetch unchanged. Canonical artifact = `.json.gz` on HF, raw `.json` only in local dev.
- **Local dev (`python -m http.server`):** no `Content-Encoding` header; requires either (a) a tiny `util.js` patch that feeds the response through `DecompressionStream('gzip')` on `.json.gz` URLs (~10 LOC), or (b) keeping raw `.json` locally. Going with (b) initially — `stage_frontend()` stays uncompressed for local testing; a new `build_gzipped_frontend_bundle()` helper produces the HF-ready tree separately.

For the 10-prompt local run: 60 graphs × ~40 MB = ~2.4 GB uncompressed, ~200 MB gzipped. Both are well under practical limits; we keep raw locally, gzip for HF upload.

### 10-prompt local run (`run_20260418_172402`, complete 2026-04-19)

Stage 02 ran locally on RTX 4090 (bf16) with `--save-graphs` → 60 `.pt` files, 83 GB on disk. Directional replication of all 50-prompt findings:

| Class | n=10 mean net | n=10 Δ vs bare | n=50 Δ |
|---|---|---|---|
| bare | +65.78 | — | — |
| completion | +75.62 | **+9.84** | +5.0 |
| roleplay | +41.70 | −24.08 | −38.7 |
| cognitive_reframe | +24.38 | −41.40 | −50.2 |
| analytical | +2.66 | −63.12 | −73.7 |
| fiction | +0.36 | −65.42 | −65.3 |

Completion paradox replicates (+14.9%). Directional agreement with 50-prompt on all 5 classes. Primary use: graph corpus for end-to-end Stage 05 subcircuit-filter-panel iteration without RunPod.

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
| 02    | ✓ Done | 1-prompt bf16 on 4090 | ✓ 50 prompts × 5 classes | Now supports `--save-graphs` + `--dtype` |
| 02b   | ✓ Done | Matches reference | ✓ 50 prompts | Includes A3 sep-curve, A5 cosine heatmap, A6 distribution plots |
| 03    | ✓ Done | 15/15      | ✓ 50 prompts | MLP ratio 0.404%; A4 per-layer contribution added |
| 04    | ✓ Done | 04-a7, 04-a8 pass | ✓ 876 features, 100% labeled | A7 UpSet + A8 layer histogram added |
| A3    | ✓ Done | ✓ | — | Separation-vs-layer plot in 02b |
| A4    | ✓ Done | ✓ | — | Per-layer contribution aggregate in 03 |
| A5    | ✓ Done | ✓ | — | 34×34 cosine heatmap (reveals Regime A/B/C pivot structure) |
| A6    | ✓ Done | ✓ | — | Bare-vs-JB distribution violin plot |
| A7    | ✓ Done | ✓ | — | Feature-class UpSet (N=788; 12.7% universal, 46.1% class-exclusive) |
| A8    | ✓ Done | ✓ | — | Top-features-by-layer histogram (80% of JB-affected features in L24–L32) |
| 05a   | ✓ Done | ✓ bf16 on 4090 | — | Stage 02 `--save-graphs` flag + `.pt` persistence |
| 05b   | ✓ Done | ✓ | — | `utils_viz.py` helpers: convert_pt, annotate_overlap, stage_frontend, gzip_json_files |
| 05c   | ✓ Done | ✓ | — | Frontend overlap coloring (shared/jb-unique/bare) + legend + fetch-override patch |
| 05d   | ✓ Done | ✓ | — | `compare.html` side-by-side viewer (bare ↔ JB iframes, URL-param driven) |
| 05e   | ✓ Done | 10-prompt local bf16 complete 2026-04-19 | — | 60 .pt files, 83 GB; directional match with 50-prompt |
| 05f   | Planned | — | — | Full 50-prompt RunPod run with `--save-graphs` |
| 05g   | **Next** | — | — | Subcircuit filter panel (Plan A) — right-rail highlight/dim |
| 05h   | Planned | — | — | Gzipped HF bundle (`build_gzipped_frontend_bundle`) |
| A1    | Planned | — | — | Coherence wiring in Stage 02 (deferred, tracked) |
| A2    | Planned | — | — | Completion-paradox feature attribution (deferred) |
| LLM labels | Planned | — | — | Claude-API-synthesized `clerp` labels (deferred) |
| 6-way UpSet | Planned | — | — | `feature_labels.json[conditions_seen]` including bare |
| 06    | Planned | — | — | Tejas's task — causal intervention (Arditi method) |
| 06b   | Planned | — | — | Linear probe |
| 07    | ✓ Done | 13/13 tests pass; 11 subcircuits, 834/876 tagged | — | Rule-based; embedding-based Plan B deferred for Georg meeting |
| 08    | Planned | — | — | Subcircuit ablation (priority queue in README §9 Stage 08 ablation queue) |

---

## Next Milestones

1. **Plot additions A3/A5/A6 in 02b + A4 in 03 + A7/A8 in 04** — one batch of matplotlib work, improves narrative without blocking Stage 05
2. **A2 completion-paradox feature attribution** — research-interesting figure, small script
3. **A1 response coherence** — wire into Stage 02 (queued by Mahmoud)
4. **Stage 05 visualization** — once plot additions land
5. **Stage 06 causal intervention** — Tejas's parallel track
6. **M1 clean 50-prompt dataset** — deferred until pipeline is complete
