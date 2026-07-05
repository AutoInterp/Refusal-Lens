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

**Normalized edge weights (required — validated on real graphs).** Raw circuit-tracer weights are
large and signed; multiplying them along paths *explodes* combinatorially (real 3-hop products
reached ~10⁷ and a few huge paths dominated everything — coverage collapsed to 2 features / 23%).
So we propagate **normalized** weights, matching circuit-tracer's own `influence` semantics. For each
target `t`, normalize its incoming edges by the total incoming magnitude *including error leakage*:

```
norm(s → t) = weight(s → t) / ( Σ_{s'} |weight(s' → t)|  +  error_into[t] )
```

Then a **signed path-sum of normalized weights** is a bounded **fractional influence** (the direct
feature-parents' shares sum to ≤ 1; the remainder is the error/unexplained fraction):

```
contrib(u → s) = Σ over directed paths u ⇝ s of length ≤ k   ( Π norm weights along the path )
```

Computed by a single backward pass over `s`'s ancestor sub-DAG (`val[s]=1`;
`val[u] += norm(u→m)·val[m]` for each child edge `u→m`, only to depth `k`). `u` **inherits the
seed's class with the path sign**: `contrib>0` → `u` supports the seed's role; `contrib<0` → opposes
it. Record `hop(u,s)` = shortest edge-distance `u`→`s`.

**Empirical note — the pruned CLT graphs are shallow.** Because cross-layer transcoders write across
many layers, early-layer (L0–L11) features are *direct* (hop-1) parents of the L14/L15 refusal seeds;
on the real graphs hop 2 holds only a handful of ~0.001-magnitude ancestors and hop 3 is empty. v2's
value is therefore realized mostly at **hop 1** (which already contains the early-layer non-refusal
features of interest). The depth slider + 1…k mechanism machinery are still built and correct; deep
cascades would require regenerating denser (less-pruned) graphs — out of scope, a future effort.

### 4.2 Idea 3 — delta decomposition (suppression & amplification seeds)

Suppression/amplification are bare→jb *deltas*, so propagate the delta. **Two separated concerns:**
the **contribution magnitude / threshold** uses the *normalized* fractional influence of §4.1 —
`Δ(u→s) = contrib_jb(u→s) − contrib_bare(u→s)` (difference of two bounded normalized path-sums,
0 where the feature is absent from a graph). The **mechanism label** below is a property of the
*raw* edge (did the activation or the connection change?), so the per-edge split uses raw weights and
activations. For each upstream parent `u` of seed `s`, split the change in `u`'s raw contribution to
`s` into two first-order terms:

```
Δ(u → s)  =  Δa_u · w^B(u→s)        (activation-change term)
           +  a^B_u · Δw(u→s)        (edge-weight-change term)
           +  Δa_u · Δw(u→s)         (2nd-order; reported but not used for the label)
where  Δa_u = a^J_u − a^B_u ,  Δw = w^J − w^B ,  B = bare graph, J = jb graph.
```

**Per-edge mechanism (the atomic unit), for any depth.** The clean two-term split above is a
property of a *single edge*. So we classify **each edge** on a path by which term dominates its
contribution-change:

- edge is **passive** — the **activation-change** term dominates (`|Δa_u · w| ≥ (1+margin)·|a_u · Δw|`):
  the upstream feature itself switched off/on, the connection is unchanged.
- edge is **active** — the **edge-weight-change** term dominates, or the source is newly active in jb
  with an opposing-sign edge (`a^B ≈ 0, a^J > 0`): the connection strength/sign itself changed.

**Feature mechanism = propagate along the dominant path, for hops 1…k (general in k).** For a
depth-`k` path `u ⇝ s`, the feature's mechanism is the composition of its edges' labels along its
**dominant path** (the path carrying the largest |contribution| to `s`):

- **passive_cascade** — every edge on the dominant path is *passive*. The seed's change genuinely
  cascades upstream through features switching off/on, connections intact. (This is the
  "does it cascade?" signal: a passive_cascade at hop 3 means the chain stayed passive all the way.)
- **active_inhibitor** — the dominant path contains an *active* edge; the mechanism becomes active at
  the **first** (closest-to-seed) active edge — that link is where an active mechanism breaks into an
  otherwise-passive chain.
- **mixed** — a feature reaches `s` by two or more paths of comparable |contribution| whose composed
  labels disagree (e.g. one passive_cascade, one active_inhibitor). Reported explicitly, never
  silently collapsed.

At hop 1 this reduces exactly to the single-edge split. The per-edge labels are the same clean,
two-term determinations at every depth — we never try to collapse a whole path's Δ into one split.

### 4.3 Strictness levers (defaults; all in config)

- **Depth cap** `k = 3`.
- **Contribution threshold — ABSOLUTE floor** `tau = 0.05`: keep `u` for seed `s` only if
  `|contrib(u→s)| ≥ tau` (idea 2) or `|Δ(u→s)| ≥ tau` (idea 3). Because contributions are now bounded
  *fractional* influences, an absolute floor is meaningful ("explains ≥ 5 % of the seed's normalized
  input"); a relative-to-total threshold is NOT used (with ~380 tiny contributors it kept nothing).
  Validated on real graphs: `tau=0.05` keeps ~6–10 features/seed-set (tight, high-precision).
- **Passive/active margin** `margin = 0.25`.
- **Activation gating:** traversal only through features active in the relevant graph (graphs are
  already active-pruned; additionally require nonzero activation).
- **Overflow handling (bias-free, no new data).** Hop-1 has hundreds of low-attribution ancestors.
  The dashboard shows the **top-N by |normalized contrib|** (default `top_n_display = 25`) and
  collapses the rest into an **expandable "M more — X % of seed influence" bucket** — never a silent
  drop. Selection/ranking is by attribution + bare→jb delta only (mechanism-agnostic); no semantic
  gate (that would bake in the hypothesis about what the mechanism looks like — see §11).
- **Coverage / honesty:** per seed report `coverage = Σ kept |contrib| / total |contrib|` and
  `error_frac` = fraction of the seed's incoming attribution arriving via `mlp reconstruction error`
  nodes. Surfaced in the UI and the ledger.
- **Sign-cancellation / multi-membership:** a feature may feed several seeds, possibly with opposite
  signs. Aggregate per feature by the **max-|contribution|** single-seed influence (keeping its
  sign) for the node label; label by the **dominant** seed-class; keep every (seed, contrib, hop,
  mechanism) membership as a ledger row.

### 4.4 Completeness invariant (a testable guarantee)

With normalized weights the invariant is per-seed and directly interpretable: a seed's **direct**
(hop-1) normalized feature-parent shares plus its `error_frac` sum to **1** (they are fractions of
the seed's total input). Deeper hops redistribute those shares. The unit tests assert the direct
normalized shares + error fraction ≈ 1, and that `coverage = Σ kept |contrib| / Σ all |contrib|`
on synthetic DAGs with hand-computed normalized values.

## 5. Output classes & visual encoding

Seeds keep v1 colors at **hop 0**: `refusal_centric` red, `suppression` blue, `amplification`
green. Upstream features (hop ≥ 1) are encoded on three independent visual channels so the palette
stays legible:

- **hue** = the seed class the feature feeds (red / blue / green),
- **fill-opacity** = hop-distance gradient (`opacity = 1/(1+hop)`; fainter = deeper upstream),
- **border** = mechanism: solid = seed or `passive_cascade`; dashed = `active_inhibitor`; dotted =
  `mixed`; idea-2 `refusal_centric` upstream (no delta role, mechanism `none`) → solid.

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

**`trace_propagate.py` (pure, unit-tested) — interfaces (see the plan for exact signatures):**
- `build_key_graph(graph)` → feature-key digraph `{parents, act, error_into}` (positions summed,
  error tracked as leakage).
- `normalized_parents(kg)` → per-target normalized parent weights (§4.1 formula).
- `path_sums(parents, seed_key, k)` → `{ancestor_key: (signed_contrib, hop)}` backward path-sum
  (used on normalized parents).
- `upstream_contributions(kg, seed_keys, *, k, tau)` → `{per_feature: {key:{contrib,hop}}, coverage,
  error_frac}` — idea 2 on **refusal_centric** seeds; absolute `tau` floor; per-feature aggregate by
  max-|contrib|.
- `edge_delta_label(w_bare, w_jb, a_bare, a_jb, *, margin)` → per-edge passive/active/ambiguous on
  **raw** weights (idea-3 atomic unit).
- `dominant_path(parents_jb, seed_key, k)` → each ancestor's max-|Π norm| path (seed-adjacent first).
- `delta_decompose(bare_kg, jb_kg, seed_keys, *, k, tau, margin)` → `{per_feature:{key:{delta,hop,
  mechanism}}, coverage, error_frac}` — idea 3; delta = normalized jb path-sum − bare path-sum;
  mechanism via `dominant_path` + `edge_delta_label`.
- `assign_upstream_classes(contrib_by_class, delta_by_class)` → `{key:{upstream_class,hop,mechanism}}`
  — dominant-class resolution.
- `bake_upstream_classes(graph, feature_class_map)` → sets `rl_trace_upstream_class`, `rl_trace_hop`,
  `rl_trace_mechanism` on feature nodes, leaving v1 seed `rl_trace_class` intact.

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
  "hop": 2, "signed_contribution": -0.12,
  "predicted_effect": -0.12,        // normalized fractional influence: zeroing u removes ~this
                                    // fraction of the seed's (normalized) input — the falsifiable
                                    // prediction the ablation/steering test compares against
  "mechanism": "passive_cascade|active_inhibitor|mixed|none",
  "coverage": 0.71, "verification_status": "unverified"
}
```

`predicted_effect` is the falsifiable number a later steering/ablation harness compares its measured
effect against; `verification_status` starts `"unverified"` and is the placeholder that harness will
set to `confirmed`/`refuted`/`partial`. v2 ships the schema; it does **not** run the experiments.
The UI states plainly that highlights are **hypotheses, not proven** (attribution ≠ causation).

## 8. Testing

- **Pure unit tests** `scripts/emnlp_perm_edit/tests/test_trace_propagate.py` on synthetic DAGs with
  hand-computed **normalized** values: `normalized_parents` (shares of `Σ|w|+error` sum to 1),
  two-hop normalized path-sum, depth cap (hop-4 path excluded at `k=3`), the **absolute** `tau` floor,
  `coverage = Σ kept / Σ all`, the per-edge Δa·w vs a·Δw split (raw weights) incl. the `margin`
  boundary, **multi-hop mechanism propagation** (all-passive chain → `passive_cascade`; active edge
  at an intermediate hop → `active_inhibitor` from that edge; disagreeing paths → `mixed`),
  dominant-class resolution under sign cancellation, and `bake_upstream_classes` field-setting.
- **Real-graph smoke (E2E, required per task).** Each pure-function task is additionally exercised on
  an actual Gemma graph by the controller: contributions must be **bounded** (|contrib| ≤ ~1), not
  exploding; `coverage` and `error_frac` sane; kept-count in the expected 6–10 range at `tau=0.05`.
- **Structural tests** (append to `test_trace_patches.py`): depth slider, overflow "M more" control,
  new evidence columns, hop/opacity/mechanism CSS classes.
- **Integration smoke** `test_trace_assemble.py` (extend): on the real 4 graphs, assert upstream
  features found (hop ≥ 1), per-seed `coverage`/`error_frac` reported, and `trace_hypotheses.json`
  exists with `verification_status == "unverified"` and a numeric `predicted_effect` in [−1, 1].
- **Browser E2E (Playwright, Task 7 + final).** Serve, drive `trace.html`: depth slider at 0 matches
  v1; dragging reveals fainter upstream nodes; the "M more" bucket expands; the evidence table shows
  hop/mechanism.

## 9. Defaults to confirm during implementation
`k=3`, `tau=0.05` (absolute floor), `margin=0.25`, `top_n_display=25`, opacity `1/(1+hop)` — all in
`trace_config.json`, tunable in the visual smoke without code changes.

## 10. v2.1 follow-on (out of scope here; flagged) — feature semantics

v2 ships without any semantic content because the graph artifacts have none (`clerp` is empty, the
only logit target is the refusal direction, no local top-activation data). v2.1 is a **coupled**
data-generation + annotation effort:

1. **`clerp` generation** — harvest each transcoder feature's top-activating examples (remote feature
   DB if one exists, else a GPU pass over a corpus) and **LLM-label** the feature space, filling the
   empty `clerp` field. This is independently valuable for reading any of our graphs.
2. **Semantic annotation layer** — use those labels as a **reversible lens, never a gate**:
   (a) annotate/rank, never remove features; (b) *surface* the strong-attribution +
   semantically-*unexpected* features as **novel-mechanism candidates** (e.g. a role-play feature
   suppressing refusal — exactly Georg's target, which a keyword filter would discard); (c) rank
   primarily by attribution + bare→jb delta (mechanism-agnostic), semantics secondary; (d) always
   report hidden mass; (e) open-vocabulary tags, no hard-coded harm/legality/completion categories.

Both share the same top-activation harvest, so they are specced and built together after v2 lands.
