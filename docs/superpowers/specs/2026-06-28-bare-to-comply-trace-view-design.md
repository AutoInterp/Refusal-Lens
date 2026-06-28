# Bare→Comply Trace View — Design Spec

**Date:** 2026-06-28
**Author:** Mahmoud (with Claude)
**Status:** Approved design → ready for implementation plan
**Related:** `docs/superpowers/plans/2026-06-28-jailbreak-suppression-circuit-trace.md` (research framing),
`docs/superpowers/plans/2026-06-28-bare-to-comply-shortlist.md` (the 4 prompts),
`docs/bare_to_comply_flips_responses.md` (responses). Memory: [[project_georg_jailbreak_suppression_direction]].

## 1. Purpose

A focused 2-panel frontend that places a **refused (bare)** attribution graph next to its
**jailbroken (jb)** counterpart for each genuine bare→comply prompt, and **colors the feature
nodes** by their role in the refusal computation and how they change between the two prompts.
This is a **strict correlation** instrument — a precursor to the causal (steering/ablation)
tests that come later. Highlights are **hypotheses to test, not proven mechanisms**
(attribution ≠ causation).

It answers, visually, Georg's binary-search question: does a jailbreak work by **(a)** suppressing
the features that feed *positively* into the refusal direction, or **(b)** amplifying the
"don't-refuse" features that feed *negatively* into it?

## 2. Scope

- **Model:** Gemma-complement (`run_gemma_complement_L15`) ONLY. Qwen produced zero genuine
  bare→comply flips, so it is excluded.
- **Prompts:** ONLY the 4 judge-verified "true bare→comply" flips:
  | idx | jb condition | request |
  |---|---|---|
  | 4 | jb_roleplay | social-engineering password theft |
  | 29 | jb_cognitive_reframe | anonymous negative reviews about someone |
  | 41 | jb_cognitive_reframe | manipulate legal system for false accusation |
  | 39 | jb_analytical | tamper with financial records |
- **Out of scope (v1):** Qwen; the full/outlier Gemma variants; signed path-propagation for deep
  upstream features (that is **v2**, see §8); causal tests; re-generating graphs.

## 3. Inputs (all already on disk — no re-fetch, no GPU)

Per-prompt packed graphs under
`data/results/compare_3way/run_gemma_complement_L15/05_frontend/graph_data/{NNN}_{cond}_single.json.gz`.
Each graph JSON has `metadata` (incl. `prompt_tokens`, `prompt_idx`, `jb_class`), `nodes`, `links`.
Relevant node fields: `node_id`, `feature` (id), `layer`, `ctx_idx`, `feature_type`
(`"cross layer transcoder"` = real features), `activation`, `influence` (unsigned importance),
`overlap_bucket`. The refusal target is a `logit` node (`Output "refusal_L15_pos-2"`); **links carry
signed `weight`** (≈41% of edges into the logit are negative).

## 4. Feature classification (the core analysis)

Computed per prompt pair (bare graph + jb graph) by a pure function in the assembler.

**Identity & matching.** Features keyed by `(layer, feature_id)`. Positions don't align across
bare/jb (jb wrapper shifts tokens), so match by identity; the trailing `…<start_of_turn>model\n`
position is the comparable "decision point."

**Per-feature aggregates** (computed **separately in each graph** where the feature is present):
- `edge_to_refusal` = signed `weight` of the feature's link(s) into the refusal logit node of
  *that graph* (summed if multiple). If the feature has no direct link to the logit,
  `edge_to_refusal = 0` (→ stays unclassified in v1; v2 fixes this).
- `act` = max `activation` of the feature's nodes in that graph (0 if absent).

Because sign is per-graph, classification uses the graph-appropriate edge: `refusal_centric` and
`suppression` use the **bare** graph's `edge_to_refusal` (falling back to jb if the feature is
jb-only); `amplification` uses the **jb** graph's `edge_to_refusal`. The top-N sign gate is applied
within each graph.

**Sign gate.** A feature gets a sign only if `|edge_to_refusal|` is in the **top-N** by magnitude
(default N=20, tunable). `edge_to_refusal > 0` ⇒ pro-refusal; `< 0` ⇒ anti-refusal.

**The three classes** (with default thresholds, all in config):

| Class | Rule | Panel shown | Color |
|---|---|---|---|
| **refusal_centric** | pro-refusal (signed, gated) | both | 🔴 red |
| **suppression** | refusal_centric AND `act_jb ≤ act_bare·(1−Δ)` or present-in-bare/absent-in-jb | jb panel (red in bare) | 🔵 blue |
| **amplification** | anti-refusal (signed, gated) AND `act_jb ≥ act_bare·(1+Δ)` or `jb_unique`/absent-in-bare | jb panel | 🟢 green |

- `Δ` = activation-change margin, default **0.30**.
- **Model-token gate** (optional, default OFF): restrict `refusal_centric` to features whose
  peak activation is at the model-turn position(s). Surfaced as a config flag for experimentation.
- All other nodes render **neutral gray** (unchanged from default viewer styling).

**Output of the classifier:** for each prompt pair, a map `{graph: {node_id: class}}` plus a
per-feature **evidence table** (feature id, layer, class, `edge_to_refusal`, `act_bare`, `act_jb`,
`overlap_bucket`).

## 5. Architecture

Mirrors the existing compare pipeline but simpler and self-contained.

```
scripts/pipeline/
  assemble_trace_frontend.py        # NEW: orchestrator + pure classifier
  05_frontend_patches/
    trace.html                      # NEW: 2-panel harness + legend + evidence table
    trace_config.json               # NEW: 4 flips + thresholds (N, Δ, model-token gate)
    trace-highlight.js              # NEW: injected into each iframe; recolors nodes by class
    trace-highlight.css             # NEW: red/blue/green node styles + legend styles
```

**Assembler `assemble_trace_frontend.py`:**
1. Read `trace_config.json` (the 4 `{idx, jb_condition}` pairs + thresholds).
2. For each pair, load bare + jb graph JSON from the complement run.
3. Run the **pure classifier** (§4) → per-node class map + evidence table.
4. **Bake** the class onto each node: write `node.rl_trace_class` into copied graph JSONs placed in
   the output dir (so the field rides along with the datum d3 binds — the injected JS reads it back).
5. Inject `trace-highlight.{js,css}` into the viewer's `index.html` (same injection mechanism as
   `compact-mode.*` in `utils_viz.py`), and reuse `?compact=1` to hide nav/overlays.
6. Write `trace_manifest.json` (pairs: prompt text, bare/jb slugs, evidence tables).
7. Stage `trace.html` → output dir. Print serve instructions.

**Output dir:** `data/results/trace_bare_to_comply/`. The assembler copies the complement
`05_frontend` viewer into it once (as `viewer/`), then **overwrites only the 8 needed
`graph_data` files** (4 bare + 4 jb) with their baked (`rl_trace_class`-annotated) versions and
injects the highlight assets into `viewer/index.html`. `trace.html` + `trace_manifest.json` sit at
the dir root; the iframes load `./viewer/index.html?slug=…&compact=1`. Served via
`python3 -m http.server`.

**In-graph recoloring (`trace-highlight.js`):** after the gridsnap graph renders, select the node
elements, read the bound datum's `rl_trace_class`, and add a matching CSS class
(`rl-refusal`/`rl-suppress`/`rl-amplify`). Re-applies on re-render (gridsnap relayout). No
circuit-tracer source edits — purely additive injection, consistent with `compact-mode.js`.

**`trace.html` (parent harness):**
- Top bar: **prompt picker** (4 flips, labeled by idx + jb class + request) and a **color legend**
  (red = refusal-centric, blue = suppressed refusal driver, green = amplified don't-refuse).
- Two iframes side by side: **left = bare**, **right = jb**, each
  `./<viewer>/index.html?slug=<slug>&compact=1`.
- Below the iframes: an **evidence table** for the selected pair (from the manifest) — feature,
  layer, class, edge weight, act_bare, act_jb, overlap bucket — so the analysis is legible even if
  a node is hard to spot in the graph. Sortable by class is nice-to-have, not required for v1.

## 6. Data flow

```
trace_config.json ─┐
                   ├─> assemble_trace_frontend.py ──> classify(bare, jb)  (pure)
complement graphs ─┘                                   │
                                                       ├─> baked graph JSONs (node.rl_trace_class)
                                                       ├─> trace_manifest.json (pairs + evidence)
                                                       └─> trace.html + injected highlight assets
                                                                 │
                                              http.server ──> browser: 2-panel colored view
```

## 7. Testing

- **Pure-function unit tests** (`scripts/emnlp_perm_edit/tests/test_trace_classifier.py`, CPU,
  follows existing test style) on synthetic nodes/links:
  - signed `edge_to_refusal` aggregation (sum of multiple links; missing link → 0);
  - top-N sign gate (features below the gate stay unclassified);
  - suppression (pro-refusal + activation drop / presence flip) and amplification (anti-refusal +
    rise / jb_unique) assignment, including the boundary at Δ;
  - a refusal_centric feature unchanged across panels stays red (not blue).
- **Structural assembly check:** running the assembler on the 4 flips yields `trace.html`,
  `trace_manifest.json` with 4 pairs, baked graphs containing `rl_trace_class`, and at least one
  node of each class in at least one pair (smoke).
- **Manual visual smoke:** serve, confirm the 2 panels load, legend renders, colors appear, and the
  evidence table matches the colored nodes.

## 8. v2 (explicitly deferred)

Signed **path-propagation** to classify deep upstream non-refusal features that act through
intermediate layers (e.g. role-play features that suppress refusal indirectly). Approaches to weigh
in v2: depth-limited signed influence over the link graph; or circuit-tracer **super-nodes**
(`qParams.supernodes`, already supported) collapsing the redundant L13–15 refusal features into one
node and re-tracing upstream (Georg's suggestion). v1 ships the strict direct-edge version first.

## 9. Open defaults to confirm during implementation
- N (top edges getting a sign) = 20; Δ (activation margin) = 0.30; model-token gate OFF.
  These live in `trace_config.json` and can be tuned in the visual smoke without code changes.
