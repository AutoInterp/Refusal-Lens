# Upstream-Propagation Trace View (v2) — Design Spec

**Date:** 2026-07-04
**Author:** Mahmoud (with Claude)
**Status:** Approved design → ready for implementation plan
**Builds on:** `docs/superpowers/specs/2026-06-28-bare-to-comply-trace-view-design.md` (v1) and its
plan/impl (commits 27f6185..a398ecb). Research framing:
`docs/superpowers/plans/2026-06-28-jailbreak-suppression-circuit-trace.md`.
Memory: [[project_georg_jailbreak_suppression_direction]].

## 1. Purpose

v1 classifies only the **late-layer** features with a *direct* signed edge to the refusal logit
(`refusal_centric` / `suppression` / `amplification`). v2 propagates those classifications
**upstream** — identifying the earlier-layer features that help *compute* each v1 seed — so we can
see how non-refusal features eventually drive (or suppress) the refusal direction. This is still a
**strict correlation instrument**; its output is a **ranked, falsifiable hypothesis ledger** that a
later steering/ablation harness (the source of truth, separate spec) will confirm or refute.

## 2. Scope

- **Model / prompts:** Gemma-complement, the same 4 judge-verified flips as v1 (idx 4 jb_roleplay,
  29 & 41 jb_cognitive_reframe, 39 jb_analytical). No Qwen, no full/outlier variants.
- **Seeds:** the v1 classified features are the seeds, unchanged. v2 does **not** re-derive or relax
  v1's classification — same strictness bar.
- **In scope:** upstream contribution propagation (idea 2), bare↔jb delta decomposition with
  passive/active sub-labels (idea 3), a depth slider (depth 0 == v1), and the hypothesis export.
- **Out of scope (explicit):** super-node collapse of redundant refusal seeds (**v3**); the actual
  causal steering/ablation experiments (**separate future spec**); Qwen; any change to v1's
  `classify_pair` logic.

## 3. Inputs (all on disk — no GPU, no re-fetch)

The same packed graphs v1 uses:
`data/results/compare_3way/run_gemma_complement_L15/05_frontend/graph_data/{NNN}_{cond}_single.json.gz`.
Each graph is a **signed linear attribution DAG**: `nodes` (feature nodes
`feature_type=="cross layer transcoder"`, plus `mlp reconstruction error`, `embedding`, one `logit`)
and `links` with **signed `weight`** = the direct first-order effect of source on target. Node
fields used: `node_id, feature (int), layer (str→int), ctx_idx (int), activation (float),
overlap_bucket, feature_type`. v1's `trace_classifier.classify_pair` supplies the seed node-ids +
per-graph feature aggregates.

## 4. The propagation math

A feature is keyed by `(layer, feature)`; it may have several nodes (positions). All quantities are
aggregated to feature level within a graph (edge = sum over the feature's nodes; activation = max),
reusing v1's `aggregate_features`.

**Seed → analysis mapping (fixed):** `refusal_centric` seeds (stable pro-refusal, not a delta) use
**idea 2**. `suppression` and `amplification` seeds are both bare→jb deltas and use **idea 3**. No
seed is analyzed by both, so each upstream feature gets one unambiguous role.

### 4.1 Idea 2 — class-conditional upstream contribution (refusal_centric seeds)

A seed's *incoming* edges are its input attribution. For upstream feature `u` and seed `s`, define
the **depth-capped signed path-sum**:

```
contrib(u → s) = Σ over directed paths u ⇝ s of length ≤ k   ( Π edge weights along the path )
```

Computed by a single backward pass over `s`'s ancestor sub-DAG (process nodes in reverse
topological order; `val[s]=1`; `val[u] += w(u→m)·val[m]` for each child edge `u→m`, only to depth
`k`). `u` **inherits the seed's class with the path sign**: `contrib>0` → `u` supports the seed's
role; `contrib<0` → `u` opposes it. Record `hop(u,s)` = shortest edge-distance `u`→`s`.

### 4.2 Idea 3 — delta decomposition (suppression & amplification seeds)

Suppression/amplification are bare→jb *deltas*, so propagate the delta. For each upstream parent `u`
of seed `s`, split the change in `u`'s contribution to `s` into two first-order terms:

```
Δ(u → s)  =  Δa_u · w^B(u→s)        (activation-change term)
           +  a^B_u · Δw(u→s)        (edge-weight-change term)
           +  Δa_u · Δw(u→s)         (2nd-order; reported but not used for the label)
where  Δa_u = a^J_u − a^B_u ,  Δw = w^J − w^B ,  B = bare graph, J = jb graph.
```

Deeper-than-one-hop upstream features get the same split applied along the depth-capped path
(product rule, first order): the path's Δ is attributed to whichever single edge on the dominant
path changed most — v2 keeps this to **direct parents (hop 1) for the passive/active label** and
reports deeper contributors only with their signed Δ and hop (no passive/active label past hop 1,
to stay strict). **Mechanism label** (hop-1):

- **passive_cascade** — the **activation-change** term dominates (|Δa·w| ≥ (1+margin)·|a·Δw|) and
  `u` moved consistently with the seed's change (a refusal driver that switched off for a
  suppression seed; an anti-refusal driver that switched on for an amplification seed).
- **active_inhibitor** — the **edge-weight-change** term dominates, or `u` is newly active in jb
  with an opposing-sign edge (`a^B_u≈0, a^J_u>0`).
- **mixed** — neither term dominates by `margin`.

### 4.3 Strictness levers (defaults; all in config)

- **Depth cap** `k = 3`.
- **Contribution threshold** `tau = 0.10`: keep `u` for seed `s` only if `|contrib(u→s)|` (idea 2)
  or `|Δ(u→s)|` (idea 3) ≥ `tau ·` (seed's total incoming |attribution| resp. total |Δ|).
- **Passive/active margin** `margin = 0.25`.
- **Activation gating:** traversal only through features active in the relevant graph (graphs are
  already active-pruned; additionally require nonzero activation).
- **Coverage / honesty:** per seed report `coverage = Σ kept |contrib| / total |contrib|` and
  `unexplained_error_frac` = fraction of the seed's incoming attribution arriving via
  `mlp reconstruction error` nodes. Surfaced in the UI and the ledger.
- **Sign-cancellation / multi-membership:** a feature may feed several seeds, possibly with opposite
  signs. Give each feature a **net signed score per seed-class** = Σ over seeds of its signed
  contribution; label it by the **dominant** class (max |net score|); keep every (seed, contrib,
  hop, mechanism) membership in the ledger row.

### 4.4 Completeness invariant (a testable guarantee)

Because the graph is linear, `Σ_u contrib(u→s) over ALL ancestors (uncapped) + residual = seed's
total input`. The unit tests assert `kept + dropped + error-leakage == total` on synthetic DAGs.

## 5. Output classes & visual encoding

Seeds keep v1 colors at **hop 0**: `refusal_centric` red, `suppression` blue, `amplification`
green. Upstream features (hop ≥ 1) are encoded on three independent visual channels so the palette
stays legible:

- **hue** = the seed class the feature feeds (red / blue / green),
- **fill-opacity** = hop-distance gradient (`opacity = 1/(1+hop)`; fainter = deeper upstream),
- **border** = mechanism: solid = seed or `passive_cascade`; dashed = `active_inhibitor`;
  (idea-2 refusal/amplification upstream with no delta role → solid).

The baked node fields: v1's `rl_trace_class` (seeds) is kept; v2 adds `rl_trace_upstream_class`
(dominant seed-class it feeds, or absent), `rl_trace_hop` (int, 0 for seeds), `rl_trace_mechanism`
(`seed`|`passive_cascade`|`active_inhibitor`|`mixed`|`none`).

## 6. Architecture

Reuses the v1 pipeline; adds one pure module and extends the assembler + frontend.

```
scripts/pipeline/
  trace_propagate.py         # NEW pure module (no I/O): the §4 math
  assemble_trace_frontend.py # EXTEND: run v1 classify → propagate → bake both → write ledger
  05_frontend_patches/
    trace.html               # EXTEND: depth slider + upstream evidence columns
    trace-highlight.js       # EXTEND: opacity/border encoding + hop filtering
    trace-highlight.css      # EXTEND: hop/mechanism styles
    trace_config.json        # EXTEND: k, tau, margin
```

**`trace_propagate.py` (pure, unit-tested) — interfaces:**
- `build_ancestors(graph) -> dict` — child→parents adjacency with signed weights over feature/error/
  embedding/logit nodes, plus a topological order.
- `upstream_contributions(graph, seed_feature_keys, *, k, tau) -> {seed_key: {u_key: {contrib, hop}}, coverage, error_frac}`
  — idea 2 backward path-sum, thresholded. Called with the **refusal_centric** seeds.
- `delta_decompose(bare, jb, seed_feature_keys, *, k, tau, margin) -> {seed_key: {u_key: {delta, act_term, edge_term, mechanism, hop}}, coverage, error_frac}`
  — idea 3.
- `assign_upstream_classes(contrib_map, delta_map, seed_classes) -> {feature_key: {upstream_class, hop, mechanism, memberships:[...]}}`
  — dominant-class resolution + membership list.
- `bake_upstream_classes(graph, feature_class_map) -> graph` — set `rl_trace_upstream_class`,
  `rl_trace_hop`, `rl_trace_mechanism` on the graph's feature nodes (keyed via `(layer,feature)`),
  leaving v1's `rl_trace_class` (seeds) intact.

**Assembler flow (per prompt pair):** v1 `classify_pair` → seed keys per class →
`upstream_contributions` (refusal_centric seeds) + `delta_decompose` (suppression + amplification
seeds) → `assign_upstream_classes` → `bake_trace_classes` (v1 seeds) +
`bake_upstream_classes` (v2) on each of the 8 graphs → write `trace_manifest.json` (extended
evidence) + `trace_hypotheses.json` (§7) + copy `trace.html`.

**Frontend:** `trace-highlight.js` reads `rl_trace_hop`/`rl_trace_upstream_class`/`rl_trace_mechanism`
and applies hue+opacity+border; a **depth slider (0…k)** in `trace.html` sets a max hop and the JS
shows/colors only nodes with `rl_trace_hop ≤ slider` (client-side filter; baked once). Depth 0 = the
exact v1 view. The evidence table gains columns: `target seed`, `hop`, `contrib/Δ`, `mechanism`,
`predicted_effect`, `verification_status`.

## 7. Causal-verification handoff

The assembler emits **`data/results/trace_bare_to_comply/trace_hypotheses.json`**: a flat list, one
row per (upstream-feature → seed) hypothesis:

```json
{
  "prompt_idx": 4, "jb_class": "jb_roleplay",
  "feature": {"layer": 9, "feature": 12345}, "target_seed": {"layer": 15, "feature": 67890},
  "role": "upstream_refusal|upstream_suppression|upstream_amplification",
  "hop": 2, "signed_contribution": -3.14,
  "predicted_effect": -3.14,        // linear prediction: zeroing this feature shifts the seed by this
  "mechanism": "passive_cascade|active_inhibitor|mixed|none",
  "coverage": 0.71, "verification_status": "unverified"
}
```

`predicted_effect` is the falsifiable number a later steering/ablation harness compares its measured
effect against; `verification_status` starts `"unverified"` and is the placeholder that harness will
set to `confirmed`/`refuted`/`partial`. v2 ships the schema; it does **not** run the experiments.
The UI states plainly that highlights are **hypotheses, not proven** (attribution ≠ causation).

## 8. Testing

- **Pure unit tests** `scripts/emnlp_perm_edit/tests/test_trace_propagate.py` on synthetic DAGs:
  two-hop signed path-sum (`w1·w2`), depth cap (a hop-4 path excluded at `k=3`), `tau` threshold,
  the completeness invariant (`kept + dropped + error-leakage == total`), the Δa·w vs a·Δw split and
  `passive_cascade`/`active_inhibitor`/`mixed` assignment incl. the `margin` boundary,
  dominant-class resolution under sign cancellation, and `bake_upstream_classes` field-setting.
- **Structural tests** (append to `test_trace_patches.py`): the depth slider element, the new
  evidence columns, and the hop/opacity/mechanism CSS classes.
- **Integration smoke** `test_trace_assemble.py` (extend): on the real 4 graphs, assert upstream
  features are found at `hop ≥ 1`, per-seed `coverage` is reported, and `trace_hypotheses.json`
  exists with rows whose `verification_status == "unverified"` and a numeric `predicted_effect`.
- **Manual visual smoke:** serve, drag the depth slider 0→k, confirm depth 0 matches v1 and deeper
  settings reveal fainter upstream nodes with correct hue/border.

## 9. Defaults to confirm during implementation
`k=3`, `tau=0.10`, `margin=0.25`, opacity `1/(1+hop)` — all in `trace_config.json`, tunable in the
visual smoke without code changes.
