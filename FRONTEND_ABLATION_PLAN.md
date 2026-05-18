# Frontend Manual Feature & Supernode Ablation — Plan

**Goal**: enable an interpretability researcher to (a) click feature nodes in the attribution-graph viewer, group them into a "supernode", and (b) run an ablation/steering pass against a chosen prompt with the result rendered live next to the graph — without breaking the existing Stage 05 frontend or the Stage 08 batch ablation pipeline.

**Scope of this plan**: design + phased rollout only. No code in this document.

---

## 1. Inventory of what already exists (do not rebuild)

| Component | Path | Purpose | State |
|---|---|---|---|
| Vendored circuit-tracer frontend | `vendor/circuit-tracer/circuit_tracer/frontend/` | Original Anthropic graph viewer (`index.html?slug=...`). | Unchanged. |
| Stage 05 orchestrator | `scripts/pipeline/05_visualize_circuits.py` | Stages JSON + vendored frontend + 3-way overlap annotation. | Stable. |
| Subcircuit filter panel | `05_frontend_patches/subcircuit-panel.{js,css}` (204 + 131 lines) | Filter graph nodes by Stage 07 subcircuit membership. | Stable. |
| 3-way overlap annotation | `05_frontend_patches/overlap-{annotate.js, colors.css}` | Color nodes by bare/ctrl/jb overlap. | Stable. |
| Side-by-side compare | `05_frontend_patches/compare.html` (251 lines) | Two `index.html` iframes (bare ↔ jb). | Stable. Does **not** load `feature-cart.js`. |
| **Feature cart panel** | `05_frontend_patches/feature-cart.{js,css}` (315 + 156 lines) | Shift-click feature nodes → cart → export JSON or POST to `localhost:8080/ablate`. | **Built, never end-to-end tested.** |
| **Live ablation server** | `scripts/pipeline/ablation_server.py` (224 lines) | FastAPI singleton over `ReplacementModel.feature_intervention_generate`; CORS open for `localhost:8000`. | **Built, never end-to-end tested.** |
| Stage 08 batch ablation | `scripts/pipeline/08_ablate_subcircuits.py` (1,200 lines) | The CLI ablation pipeline; supports `--feature-file cart.json`. | Stable. |
| `utils.load_cart` | `scripts/pipeline/utils.py` | Loads `cart.json` (the format `feature-cart.js` exports). | Stable. |

**Headline**: ~70 % of the live-ablation feature is already implemented. The plan below tests, extends, and documents what exists rather than starting from scratch.

---

## 2. Capability gaps (what's missing)

In priority order:

1. **End-to-end test never run.** No record of `feature-cart.js` ↔ `ablation_server.py` working together against a real graph. We need to confirm the loop closes before extending.
2. **Cart not loaded in `compare.html`.** The cart works only on the standalone `index.html?slug=...` page; the side-by-side compare view (the most useful interpretation surface) doesn't include it.
3. **No supernode (multi-feature group) selection.** Cart accumulates singletons via shift-click; no way to lasso-select a region of the graph or label a group as a named supernode.
4. **No steering, only ablation.** Server accepts a `value` per feature but the frontend hardcodes `value: 0.0`. Positive/negative steering (e.g. add 50 % of natural max activation) needs UI.
5. **No live response diff.** Result panel shows raw text; users can't easily see token-level diffs between baseline and ablated.
6. **No cart import / supernode persistence.** Can export `cart.json` but can't load a saved cart or a Stage 07 subcircuit into the cart UI.
7. **No multi-supernode comparison.** Can't run two supernodes on the same prompt and see them side-by-side.
8. **Server has no grouped-ablation knowledge.** Backend takes a flat list. Grouping is purely a frontend label, with no consequence.

---

## 3. Non-goals (explicit YAGNI)

- **Multi-user / shared-state hosting.** This is a research tool; localhost only.
- **Authentication / authorization.** Same.
- **Replacing the vendored circuit-tracer frontend.** We add patches; we don't fork.
- **Mobile / responsive layout.** Desktop only.
- **Persistent server-side cart history.** Carts live in `localStorage` + downloadable JSON only.
- **Real-time graph re-attribution post-ablation.** That's a Stage 02 problem, not a frontend problem; cost is ~10–30 s per prompt and out of scope here.

---

## 4. Architecture (target end state)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Browser  (localhost:8000)                                                  │
│                                                                            │
│  ┌─ compare.html ────────────────────────────────┐  ┌─ feature-cart panel ┐ │
│  │  ┌─ left iframe ──────┐  ┌─ right iframe ──┐  │  │ supernode list:    │ │
│  │  │ index.html?slug=   │  │ index.html?slug=│  │  │ ▢ pro_refusal (6f) │ │
│  │  │   {idx}_bare       │  │   {idx}_jb_*    │  │  │ ▢ jb_unique  (12f) │ │
│  │  │  [graph]           │  │  [graph]        │  │  │ + new supernode    │ │
│  │  └────────────────────┘  └─────────────────┘  │  │                    │ │
│  │                                               │  │ steering:          │ │
│  │  shift-click → add singleton  to active SN    │  │   value: [+0.0]    │ │
│  │  drag-select → add region     to active SN    │  │   positions: all▼  │ │
│  │  + button   → create new supernode            │  │                    │ │
│  └───────────────────────────────────────────────┘  │ [Run] [Export]     │ │
│                                                     │ result:            │ │
│                                                     │  baseline (REFUSE) │ │
│                                                     │  ablated  (COMPLY) │ │
│                                                     │  diff: …           │ │
│                                                     └────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                │  POST /ablate { groups: [...], prompt }
                                ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ ablation_server.py  (localhost:8080)                                       │
│  ModelHolder.model = ReplacementModel(...)  (singleton, ~60–120 s warmup) │
│  /ablate     → run baseline + ablated, return classifications + responses  │
│  /ablate_multi (NEW) → run baseline + N grouped ablations, return N pairs  │
│  /steer      (NEW)  → sign-aware add/sub at given value                    │
│  /health     → liveness                                                    │
└────────────────────────────────────────────────────────────────────────────┘
```

Key invariants:
- Existing Stage 05 outputs (graph_data/*.json.gz, vendored index.html, single-graph slugs) **continue to work as-is**. Cart panel is additive.
- Existing Stage 08 batch pipeline **continues to work as-is**. `cart.json` schema is preserved; we add an optional `groups` block (backward-compatible).
- Existing `compare.html` **continues to work as-is** when `feature-cart.js` is absent (the cart panel is gated behind a feature flag in the patch).

---

## 5. Phased rollout (each phase is independently shippable + reversible)

Each phase has: **scope · deliverables · risk · test gate · rollback plan**.

### Phase 1 — End-to-end smoke (the missing test)

**Scope**: confirm the existing `feature-cart.js` ↔ `ablation_server.py` loop works against a real run.

**Deliverables**:
- `scripts/pipeline/tests/test_ablation_server_smoke.py` — automated test: starts server, POSTs a known feature set against a known prompt, asserts response shape + a known JB-recovery flip on `L13:F427`.
- Manual verification doc `docs/frontend_ablation_smoke.md` — 5-line README for the human path: `python ablation_server.py --lazy-model` (in tmux) → serve `data/results/pipeline_runs/run_20260430_023247/05_frontend/` on `:8000` → open browser, shift-click a node, click Run.
- Bug fixes (almost certainly required) for issues found during smoke.

**Risk**: low. We're testing what already exists.

**Test gate**: the automated smoke passes; the human path produces a baseline-COMPLY → ablated-REFUSE flip on a known-flipping prompt.

**Rollback plan**: trivial — no shipped artifacts other than a new test file.

**Estimated effort**: 4–6 h (mostly waiting for two model loads).

### Phase 2 — Cart in `compare.html`

**Scope**: make the cart panel work in side-by-side compare view. This is the highest-leverage UI change because compare.html is where users actually do interpretation.

**Deliverables**:
- `compare.html` patched to include `<script src="./feature-cart.js" defer>` (and `feature-cart.css`).
- `feature-cart.js` adapted to handle two iframes: clicks in either iframe add to the same cart. Implementation note: the cart panel sits in the parent `compare.html`; iframes message clicks up via `postMessage`.
- `subcircuit-panel.js` integration: when a Stage 07 subcircuit is selected in the existing filter panel, optionally seed the cart with its features (button: "Add subcircuit to cart").
- Stage 05 orchestrator (`05_visualize_circuits.py`) updated to copy `feature-cart.{js,css}` alongside `compare.html` (it already copies the other patches; this is a one-line addition).

**Risk**: low-medium. The vendored `index.html` is iframe-isolated, so we have to use `postMessage` to bridge clicks. This is well-understood but worth a 30-min experiment first.

**Test gate**: 
- `compare.html` still loads with cart code absent (feature flag).
- Shift-click in left iframe + shift-click in right iframe both populate the same cart.
- Existing `index.html?slug=...` standalone view still works (cart panel still attaches there too — no regression).
- Subcircuit-panel "Add to cart" button populates the cart with the selected subcircuit's features.

**Rollback plan**: revert one HTML and one orchestrator line; no schema changes.

**Estimated effort**: 1 day.

### Phase 3 — Supernodes (named groups)

**Scope**: enable users to create named groups of features (e.g. "pro_refusal_core", "fiction_unique") within the cart and run them as a single ablation unit.

**Deliverables**:
- `feature-cart.js` extended to track groups: `cart = Map<group_name, Set<feature_key>>`. UI: collapsible group list, "+ new group" button, drag-and-drop reassignment.
- Cart JSON schema bumped (backward-compatible): adds an optional `groups` field. v1 schema (flat features list) still works.
  ```json
  {
    "features": [...],          // legacy: flat list, still supported
    "groups": [                  // new: optional named groups
      {"name": "pro_refusal_core", "features": ["L13:F427", "L15:F442", ...]},
      {"name": "anti_refusal", "features": [...]}
    ],
    "source_run": "...",
    "exported_at": "..."
  }
  ```
- `utils.load_cart` updated to read `groups` if present, else fall back to flat features (one synthetic group `"manual"`).
- `08_ablate_subcircuits.py --feature-file cart.json` extended: runs each group as a separate ablation_name. Backward-compatible: a flat-features cart produces one ablation named `manual_cart`.
- Lasso-select in the graph (drag-select with shift held): adds all enclosed feature nodes to the active group. Uses the d3 zoom transform from circuit-tracer to project mouse coords back to graph coords.

**Risk**: medium. Lasso-select in a d3 zoom-pan SVG with iframe boundaries is non-trivial. Mitigation: ship Phase 3a (named groups + drag-and-drop, no lasso) first; lasso is Phase 3b.

**Test gate**:
- Old `cart.json` files (flat) still load.
- Two named groups in one cart → batch ablation produces two `ablation_results` blocks.
- Stage 07 subcircuits importable via the existing "Add subcircuit to cart" button now create a named group, not a flat dump.
- Lasso-select adds the right features (manual verification + automated test on a fixture graph).

**Rollback plan**: groups are an additive optional field; reverting the JS to ignore the `groups` block is safe.

**Estimated effort**: 2 days (3a) + 2 days (3b lasso).

### Phase 4 — Steering (signed values)

**Scope**: replace the hardcoded `value: 0.0` with a per-feature or per-group steering value. Enables positive (force-refuse) and negative (suppress-refuse) interventions, not just zero-ablation.

**Deliverables**:
- Cart UI: per-group "steering value" input (default 0.0 = ablate). Slider with sensible range (e.g. ±max-activation × 2, where max-activation is taken from `feature_labels.json`).
- Per-feature override (advanced): expand a group → edit individual feature values.
- Server `/ablate` endpoint already accepts `value` per feature — no server change needed for Phase 4 if we just route the UI value through.
- Server `/steer` endpoint (NEW): convenience wrapper that takes a single direction `r` (loaded from `01_direction/unnormalized_r.pt`) + sign + magnitude, replicating the L15 r-intervention from Stage 06 in the browser. This is the high-leverage demo: live bidirectional axis manipulation.

**Risk**: low for the value-passthrough; medium for `/steer` because it requires loading the per-layer r tensors at server startup.

**Test gate**:
- Setting `value = 100` on a known pro-refusal feature with baseline-COMPLY produces a refused output.
- `/steer` endpoint replicates Stage 06's 100 % flip rate on a 5-prompt smoke.

**Rollback plan**: gate the steering UI behind a feature flag.

**Estimated effort**: 1 day for value-passthrough; 1 day for `/steer`.

### Phase 5 — Live result rendering & ergonomics

**Scope**: replace the bare `<pre>` result with a structured response card; add cart save/load; add multi-supernode comparison.

**Deliverables**:
- Result card: baseline + ablated side-by-side, classification badges (REFUSE/COMPLY), coherent-flag, elapsed-s, token-level diff highlighting.
- "Save cart to browser" / "Load cart" using `localStorage` (no server-side persistence).
- "Run all groups" button: POSTs to `/ablate_multi` (NEW endpoint) which runs the baseline once + each group's ablation, returning N (baseline, ablated) pairs. Renders a stacked result panel.
- Recent-history strip: last 5 runs visible, click to re-render.
- Shareable URL state: prompt + cart + group selections encoded into URL hash so a user can paste a link to a colleague (assuming colleague is also on the same `localhost:8000` setup — really for self-sharing across browser tabs).

**Risk**: low. UI polish.

**Test gate**:
- All Phase 1–4 tests still pass.
- A new playwright/puppeteer test: load compare.html, add 3 features to a group, run ablation, verify result card has both classifications and a non-empty diff.

**Rollback plan**: each ergonomics feature is independently flagged.

**Estimated effort**: 2 days.

### Phase 6 (optional, post-ICML) — Headless API harness

**Scope**: expose the server's grouped-ablation API as a Jupyter-friendly Python client so headless analysis (no browser) is possible.

**Deliverables**: `scripts/pipeline/ablation_client.py` — thin HTTP wrapper, `pip install`-friendly. Use case: scripted parameter sweeps over (group, prompt, value) without the GPU re-init cost of the CLI batch path.

**Estimated effort**: 0.5 day. Defer until someone asks.

---

## 6. Safeguards against regressions

The existing pipeline + frontend must continue to work unchanged. Concrete guard rails:

1. **Feature flags on every patch.** Each new JS file checks for a global `window.REFUSAL_LENS_DISABLE_CART = true` to disable itself. CSS is namespaced so removal is one selector deletion.
2. **Schema versioning on `cart.json`.** New `groups` field is optional. `utils.load_cart` falls back to legacy flat-features when absent. Add a `cart.json` schema version field (default `"1.0"`).
3. **Server endpoints are additive.** `/ablate` (existing) is untouched in signature. `/ablate_multi` and `/steer` are net-new.
4. **Stage 05 orchestrator changes are additive copies.** Adding `feature-cart.js` to the staged file list is one new line; it doesn't alter existing files.
5. **Existing Stage 08 CLI invocations untouched.** `--subcircuits-file subcircuits.json` and `--subcircuits canonical_pro_refusal` continue to work bit-exact. The grouped `cart.json` is one additional code path.
6. **Test coverage gate.** Each phase ships with at least one regression test in `scripts/pipeline/tests/test_pipeline_local.py` (or a new `test_frontend_ablation.py`) — failures block merge.
7. **No vendored `index.html` modifications.** All extensions go through patch files in `05_frontend_patches/`. The vendored circuit-tracer frontend is read-only.

---

## 7. Rollout sequencing relative to the paper

Given the paper deadlines (ICML mech interp workshop imminent, NeurIPS main ~3 weeks):

- **Phase 1** is a prerequisite for any paper figure that shows live ablation. Must land **this week**.
- **Phase 2** (cart in compare.html) gives us a teaser-quality demo for ICML supplementary. **Optional but high-leverage.**
- **Phase 3a** (named groups, no lasso) is needed if the paper claims grouped ablation as a tool contribution. **NeurIPS-only**, post-ICML.
- **Phase 3b**, **Phase 4**, **Phase 5**, **Phase 6**: post-NeurIPS submission. The paper does not depend on them.

**Critical path for ICML camera-ready demo (if accepted)**: Phase 1 + a 30-second screen recording showing shift-click → cart → run → live flip. No further phases required.

---

## 8. Test strategy

| Test | Where | Run when |
|---|---|---|
| `test_ablation_server_smoke` (Phase 1) | `scripts/pipeline/tests/` | Every push |
| Cart in compare.html headless test (Phase 2) | playwright/puppeteer in `tests/frontend/` | Every push |
| `cart.json` schema backward-compat (Phase 3) | `test_pipeline_local.py::test_cart_v1_load` | Every push |
| Stage 08 grouped-cart ablation (Phase 3) | `test_pipeline_local.py::test_stage_08_grouped_cart` | Every push |
| `/steer` parity with Stage 06 (Phase 4) | manual smoke + 5-prompt automated | Pre-merge |
| Full-flow playwright (Phase 5) | `tests/frontend/` | Pre-release |

**Manual exploratory testing protocol**: every phase requires at least 30 minutes of human exploration on `data/results/pipeline_runs/run_20260430_023247/` against the prompts in §9.7.6 of the report (the L13:F427 examples), to confirm subjective UX matches expectations.

---

## 9. Open design questions

1. **Should grouping be a frontend-only label, or pushed to the server?**
   - Frontend-only (simpler): server runs N ablations, frontend renders them as N groups. No server-side grouping concept.
   - Server-aware (more flexible): server accepts a `groups` payload and returns per-group results in one round-trip.
   - Recommendation: **server-aware** via a new `/ablate_multi` endpoint (Phase 5). Lower per-run overhead because the model load happens once per request, not N times.

2. **Should the cart persist across browser refreshes?**
   - localStorage (simple, single-browser, single-tab).
   - URL-hash encoding (shareable).
   - Recommendation: **both**. localStorage for default persistence, URL-hash for explicit "share this view" intent.

3. **Lasso-select implementation: SVG-native or DOM-overlay?**
   - SVG-native (cleaner, integrates with d3 zoom): more code, more iframe-boundary issues.
   - DOM overlay (simpler): lower fidelity, less elegant.
   - Recommendation: **start with shift-click + drag-rect overlay** (DOM); upgrade to SVG-native if the UX is too clunky. Both are within the same Phase 3b budget.

4. **Should we expose Stage 06's symmetric `r`-intervention as a "one-button ablate-the-whole-axis" button in the cart panel?**
   - Pros: showcase the bidirectional symmetry result live; powerful demo.
   - Cons: blurs the line between feature ablation and direction intervention (paper Outline C territory).
   - Recommendation: **yes, in Phase 4 via `/steer`**, but visually distinct (separate button, separate result card section, label as "axis intervention" not "ablation").

---

## 10. Effort summary

| Phase | Wall | Skill | Blocking? |
|---|---|---|---|
| 1 — End-to-end smoke | 0.5 day | Backend + tests | Yes (must precede 2+) |
| 2 — Cart in compare.html | 1 day | Frontend (postMessage) | Optional for paper |
| 3a — Named groups (no lasso) | 2 days | Frontend + schema | NeurIPS-only |
| 3b — Lasso-select | 2 days | Frontend (d3) | Optional |
| 4 — Steering values + `/steer` | 2 days | Backend + frontend | Optional |
| 5 — Result card + ergonomics | 2 days | Frontend | Optional |
| 6 — Headless client | 0.5 day | Backend | Defer |

**Critical path to a paper-grade demo**: Phase 1 only (~0.5 day). Everything beyond that is either ICML-supplementary, post-acceptance polish, or NeurIPS-second-iteration material.

---

*This plan is intentionally conservative on what's "needed for the paper" and aggressive on what's "would make the tool great". Items marked "optional for paper" can ship after the May-deadline crunch.*
