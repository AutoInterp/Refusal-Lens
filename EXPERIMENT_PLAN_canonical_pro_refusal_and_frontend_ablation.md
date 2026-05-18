# Experiment Plan — `canonical_pro_refusal` Ablation + Live Frontend Ablation

**Status**: drafted 2026-05-01, not yet executed.
**Branch**: `l15-refactor` (do all work here; do not branch off without coordinating).
**Constraint**: do NOT break the existing validated pipeline. All Stage 01–08 scripts pass 244/244 tests; touch them only via additive flags or new files.

This plan covers two distinct but related deliverables:

| # | Deliverable | Wall | Risk to pipeline |
|---|---|---|---|
| **A** | One Stage 08 invocation that ablates `canonical_pro_refusal` (88 features, "all 5 JB top-50 ∖ bare top-50"). | ~3–6 h on H100. | Zero — uses existing `08_ablate_subcircuits.py --subcircuits` flag with no code change. |
| **B** | Live ablation server + frontend feature-cart end-to-end. ~70 % already built (`ablation_server.py` + `feature-cart.js`); needs testing, `compare.html` integration, and a supernode-grouping UI on top. | ~1–2 days of focused work, mostly frontend JS. | Low — all changes are in `scripts/pipeline/ablation_server.py` and `05_frontend_patches/`; the pipeline doesn't import from either. |

Treat A and B as independent: A unblocks a paper claim and is the higher scientific priority; B unblocks ad-hoc human-in-the-loop exploration and is the higher *long-term* leverage (every collaborator can use it during writeup).

---

# Part A — Ablate `canonical_pro_refusal`

## A.1 Motivation

`canonical_pro_refusal` is the only subcircuit in our Stage 07 rule set that is *defined* to be JB-only pro-refusal recruitment: features in the top-50 of all 5 JB classes but NOT in bare's top-50. Conceptually it is the cleanest answer to "what does the model recruit *because* of jailbreak attempts that it doesn't already use on bare harmful prompts?"

It was not in the Stage 08 default set (`STAGE_08_DEFAULT_SUBCIRCUITS` in `config.py`), which selected the 5 most general subcircuits:

```python
STAGE_08_DEFAULT_SUBCIRCUITS = (
    "universal_refusal_core",
    "ctrl_shared_refusal",
    "jb_fiction_specific_vs_ctrl",
    "jb_analytical_specific_vs_ctrl",
    "jb_cognitive_reframe_specific_vs_ctrl",
)
```

The omission is correctable in one CLI invocation. Cost: ~3–6 h on H100.

## A.2 What the result will tell us

| Outcome | Interpretation |
|---|---|
| High JB recovery (>40 %), low bare break (<10 %), high coverage on JB classes | `canonical_pro_refusal` IS the JB-suppression circuit. Big result for the writeup — clean causal isolation of the JB-only mechanism. |
| Moderate JB recovery (20–40 %), moderate bare break (~10 %) | Mixed. Features carry some JB-bypass weight but also affect bare. Compatible with the "redundancy" interpretation already supported by `universal_refusal_core` results. |
| Low JB recovery (<20 %), low bare break | The features fire on JBs but don't carry causal bypass weight — the dampening/sign-flip mechanisms work via mass effect, not via this 88-feature subset. Argues for ablating mass subcircuits. |
| Low JB recovery, high bare break | Negative-dissociation-like result — the features carry bare-refusal weight that's incidentally also active on JBs. Forces a re-frame of the rule's interpretation. |

The **cross-test** is `canonical_pro_refusal` recovery vs `universal_refusal_core` recovery: the universal core hit 26.6 %; if canonical_pro_refusal hits significantly higher (say 40+ %) on JB classes specifically, that's strong evidence that the "JB-only" features are causally more potent than the "shared" core for JB-bypass — a publishable mechanism story.

## A.3 Methodology (zero pipeline change)

Single Stage 08 invocation via the existing `--subcircuits` flag (already supported in `08_ablate_subcircuits.py`):

```bash
# On RunPod H100 (cheapest available; ~$3-10 in compute)
RUN=data/results/pipeline_runs/run_20260430_023247
PYTHONPATH=src python3 scripts/pipeline/08_ablate_subcircuits.py \
    --run-dir $RUN \
    --subcircuits canonical_pro_refusal \
    --subcircuits-file subcircuits_k50_f50.json \
    --positions all \
    --max-new-tokens 80 \
    --skip-baseline \
    --resume --checkpoint-every 5 \
    2>&1 | tee $RUN/_08_canonical_pro_refusal.log
```

Notes:
- `--subcircuits canonical_pro_refusal` (verified at line 90 of `08_ablate_subcircuits.py`) overrides the default subcircuit list to just this one.
- **`08_ablate_subcircuits.py` writes to a hardcoded path** `<run-dir>/08_ablation/ablation_results.json` (line 1158). Re-running on the same run dir WILL overwrite the existing 5-ablation results. Two clean ways to avoid this:
  - **Option A (recommended)**: copy the run dir to a sibling and run there. `cp -r data/results/pipeline_runs/run_20260430_023247 data/results/pipeline_runs/run_20260430_023247_canonical` then run with `--run-dir <new path>`. Disk cost: ~36 MB (the small JSONs are all that exist locally; 02_attribution/graphs/ is on HF only).
  - **Option B**: add a `--output-file` CLI flag (~5-line change). Pattern: `out_path = out_dir / args.output_file` instead of `out_dir / "ablation_results.json"`. Add the flag with `default="ablation_results.json"` so existing runs aren't affected. Then invoke with `--output-file ablation_results_canonical.json`.
  - Choose Option A for "I just want this experiment to finish"; Option B if you want the script to support multiple-run-on-same-dir generally.
- `--skip-baseline` reuses the already-classified baselines from Stage 06 (don't redo the 50 × 11 baseline pass — saves ~50 min).
- `--resume --checkpoint-every 5` is the safety net for pod restarts mid-run.
- `subcircuits_k50_f50.json` is the per-prompt sweep config; canonical_pro_refusal at this config will be smaller than the legacy 88-feature set (per § 7.2 of the report, the per-prompt versions are 5–10× smaller than legacy). For canonical_pro_refusal specifically: **expect roughly 15–25 features at k50_f50** based on the size pattern, vs 88 at legacy. The smaller per-prompt count is the rigorous version; if the result is borderline (~20 % recovery) you can re-run on `subcircuits.json` (legacy 88-feature) for the corpus-aggregate cross-check.

### A.3.1 Pre-flight check

Before kicking off the H100 run, sanity-check the canonical_pro_refusal entry exists in both subcircuit files:

```bash
RUN=data/results/pipeline_runs/run_20260430_023247
python3 -c "
import json
for f in ['subcircuits.json', 'subcircuits_k50_f50.json']:
    d = json.load(open(f'$RUN/07_subcircuits/{f}'))
    cp = d.get('subcircuits', d).get('canonical_pro_refusal', {})
    feats = cp.get('features', cp) if isinstance(cp, dict) else cp
    n = len(feats) if hasattr(feats, '__len__') else 'unknown'
    print(f'{f}: canonical_pro_refusal n_features = {n}')
"
```

Confirm the per-prompt sweep version has a non-trivial number of features (>5) before paying for H100 time.

## A.4 Output schema

The invocation produces:
- `08_ablation/ablation_results_canonical.json` — per-(prompt × condition) records, same schema as the existing `ablation_results.json`.
- `08_ablation/ABLATION_SUMMARY_canonical.md` — generated by passing the new file through the existing summary logic.
- Optional: `08_ablation/canonical_pro_refusal_drilldown.{json,csv}` via `08_recovery_drilldown.py --ablation-results-file ablation_results_canonical.json` (will need `--ablation-results-file` to be added to that script — one-line change).

## A.5 Expected runtime

On H100 SXM with ~85 features × 50 prompts × 11 conditions × 1 position-mode at `--max-new-tokens 80`:

- ~30 s/generation × 550 generations × 1 ablation = ~4.6 h
- Plus 1 h overhead (model load, tokenization, classification post-processing)
- **Budget: 5.5–6 h on H100 SXM**, $5–10 at community pricing.

## A.6 Verification + headline numbers

After completion, capture for the writeup:

1. `ablation_summary_canonical.json[per_ablation][canonical_pro_refusal][positions][all]`:
   - `weighted.jb_weighted_recovery_rate` — the headline
   - `weighted.bare_break_rate` — the specificity floor
   - `weighted.ctrl_weighted_break_rate` — the ctrl break (should be low since canonical_pro_refusal is JB-only by construction)
2. Per-class recovery rates (5 JB classes) — to compare against the universal_refusal_core 5-class breakdown.
3. Activation audit: per-class top-50 hit rate on canonical_pro_refusal features. **Expected**: high on all JB classes (definitionally >50 %), low on bare (definitionally 0 %), variable on ctrl (depends on prefix overlap).

The publishable claim, if the result is positive: **"The 88-feature subcircuit recruited specifically by jailbreaks (in all 5 JB top-50 but not in bare top-50) accounts for X% of jailbreak comply when ablated, while breaking only Y% of bare refusals — clean causal isolation of the JB-only refusal-promoting circuit."**

If the result is negative, the writeup pivot is equally clean: **"Even the cleanest set-logic-defined JB-only circuit (88 features) carries minimal causal weight. JB bypass is mediated by distributed signal in the residual stream that no top-K feature subset captures — see Stage 06 vs Stage 08 § 9.5."**

Either outcome is publishable, which is why this experiment is the highest-priority follow-up.

---

# Part B — Live ablation server + frontend feature-cart

## B.1 What's already in the repo

This is the most important context before scoping new work. Files that already exist:

### `scripts/pipeline/ablation_server.py` (224 lines, FastAPI)
- Loads `ReplacementModel` once at startup (~60–120 s).
- Endpoint `POST /ablate` accepts `{features: [{layer, feat_idx, value}], prompt: str, positions: "all"|"anchors", max_new_tokens?}`.
- Returns `{baseline, baseline_cls, baseline_coherent, ablated, ablated_cls, ablated_coherent, n_features, positions, elapsed_s}`.
- Endpoint `GET /health` reports model load state.
- CORS opened for `http://localhost:8000` (frontend dev server).
- Backend: `nnsight` (because `feature_intervention_generate` lives there; cf. attribution which uses TransformerLens for `hook_resid_post`).

### `scripts/pipeline/05_frontend_patches/feature-cart.js`
- Click-to-cart UI: clicking a feature node in the attribution graph adds `{layer, feat_idx, label, value}` to a `Map`-backed cart.
- "Export cart.json" button: downloads the cart in the same schema `08_ablate_subcircuits.py --feature-file` expects, so cart-curated subcircuits can be ablated offline at scale.
- "Run ablation (live)" button: POSTs the cart to `http://localhost:8080/ablate` and renders the baseline-vs-ablated diff inline.
- Hardcoded `SERVER_URL = http://localhost:8080/ablate`.

### Wiring
- `scripts/pipeline/utils_viz.py` already injects `feature-cart.css` + `feature-cart.js` into the Stage 05 frontend's per-graph HTML. So every regenerated frontend (post `05_visualize_circuits.py`) ships with the cart UI live.

### What's verified to work
- The endpoint is structurally correct (matches `feature_intervention_generate` API).
- The cart UI emits the expected JSON schema (matches `utils.load_cart`).
- CORS is correctly scoped.

### What's NOT verified
- End-to-end: starting both server + frontend server, clicking through the cart, hitting "Run ablation (live)", and getting a sensible response back.
- The `compare.html` two-graph view (a separate frontend page) does NOT yet have the feature-cart panel injected — `utils_viz.py` only patches single-graph pages.
- Supernode (multi-feature group) selection is not implemented. Currently you can only add features one-by-one.
- Cart persistence (save/load named carts) is not implemented.

## B.2 Phased plan

### Phase 1 — End-to-end test (1–2 hours, no code changes)

Goal: confirm the existing infrastructure works as advertised.

```bash
# Terminal 1: stand up the ablation server (needs GPU)
PYTHONPATH=src python3 scripts/pipeline/ablation_server.py \
    --host 127.0.0.1 --port 8080 --dtype bfloat16

# Terminal 2: serve the frontend
cd data/results/pipeline_runs/run_20260430_023247/05_frontend
python3 -m http.server 8000

# Terminal 3 (browser): http://localhost:8000/
# Pick a graph (e.g., 039_jb_fiction_single).
# Open developer console; check for any feature-cart JS errors.
# Click 3-5 high-magnitude feature nodes — confirm they appear in the cart panel.
# Click "Run ablation (live)" — wait ~10-30 s, confirm a response panel renders.
# Confirm the response shows distinct baseline and ablated text.
```

Pass criteria:
- Server endpoint `GET /health` returns `{"status": "ok"}` after model load.
- Cart accumulates features as you click.
- POST `/ablate` returns 200 with non-empty baseline and ablated strings.
- Baseline classification matches what's in the corresponding `06_causal/causal_results.json`'s baseline column for the same prompt.

If any step fails: file a bug in the cart panel and patch in Phase 1b. Most likely failure modes:
- Stage 05 frontend was generated *before* `feature-cart.js` was wired in → regenerate Stage 05 frontend.
- Server import fails on `circuit_tracer` (editable install issue) → `pip install -e ./vendor/circuit-tracer`.
- `feature_intervention_generate` shape mismatch (e.g., interventions list format changed) → diff against `08_ablate_subcircuits.py`'s build_interventions and align.

### Phase 2 — `compare.html` integration (1 day)

Goal: enable the cart in the two-graph compare view, with two carts (one per side) so you can ablate one prompt's features and see how the other prompt's response shifts under the same ablation.

Changes required (all in `05_frontend_patches/`):

1. **`utils_viz.py` patch**: detect when the page being patched is `compare.html` (or any multi-graph page) and inject a *paired* cart panel — one per graph slot, with distinct DOM IDs. ~10 lines.
2. **`feature-cart.js` upgrade**: support multiple cart instances on the same page, keyed by an "owner graph slug." Right now the cart is a singleton (`const cart = new Map()` at module top); change to a `Map<slug, Map<key, feature>>`.
3. **`compare.html` script section**: when the user clicks "Run ablation (live)" on cart A, send the request with `prompt = cart_A.prompt` and the features from `cart_A`; render the response in cart A's panel. Optionally, "Apply A's features to B's prompt" — apply cart_A's feature set to cart_B's prompt to test cross-prompt-class transfer.

Risk: low. `compare.html` already exists in `05_frontend_patches/` (the Stage 05 frontend ships it as a static asset). The cart code is self-contained; the only cross-cutting change is `utils_viz.py`.

### Phase 3 — Supernode (multi-feature) selection (1–2 days)

Goal: select multiple features in one drag, save the group as a named supernode, ablate the supernode as a single unit.

Changes (all in `05_frontend_patches/`):

1. **Lasso-select UI**: add a "Selection mode: feature / supernode / off" toggle. In supernode mode, click+drag draws a bounding box; all features within it are added to the active supernode. ~50 lines of JS.
2. **Named supernode persistence**: a panel listing currently-defined supernodes per graph; "Save as…" prompts for a name; supernodes persist across cart runs. Storage: `localStorage` keyed by graph slug. ~30 lines.
3. **Cart-builds-from-supernode**: dropdown lets you load a supernode into the cart (additive — combines with whatever's already there). ~10 lines.
4. **Server side: no change** — the supernode is just N features in the existing `features` array. Backend doesn't need to know it's "a supernode" vs "N individual features."

### Phase 4 — Ergonomics & data flow (optional, 1 day)

- **"Save run" button**: writes the cart + ablation result + prompt + classifier output to a named JSON file in `<run-dir>/manual_ablations/<timestamp>.json`. Builds a research log.
- **Comparison mode**: side-by-side text diff of baseline vs ablated, with token-level highlighting where the responses diverge. CSS + JS only.
- **Pre-canned subcircuit loading**: button that loads `subcircuits_k50_f50.json[universal_refusal_core]` features into the cart in one click. Useful for "canonical refusal core" interactive ablation experiments.
- **Cross-graph supernode export**: take a supernode defined on prompt A and apply it to prompt B's graph — useful for cross-prompt-class transfer tests.

## B.3 Risk & mitigations

| Risk | Mitigation |
|---|---|
| Frontend changes break the static-asset version that's already at `05_frontend/` of every run dir | Frontend patches are injected at Stage 05 *render time*, not at ship time. Existing run dirs aren't affected unless `05_visualize_circuits.py` is re-run on them. |
| Server changes break `08_ablate_subcircuits.py` | Server doesn't import from `08_ablate_subcircuits.py`. The two share `utils.classify_response / format_prompt / is_coherent` only — a stable interface. |
| Backend differences between TransformerLens and nnsight produce different ablation results | Cross-validate: run `08_ablate_subcircuits.py` on a cart-exported subcircuit (which uses Stage 08's TL/nnsight pipeline) and compare against the live server's response on the same subcircuit. They should match within tokenization/RNG noise. Add a test for this in `scripts/pipeline/tests/test_ablation_server.py`. |
| Concurrent requests to the server cause GPU OOM | The server is single-threaded by default (FastAPI sync endpoints). Confirm `uvicorn` runs with workers=1 (the default). For production, add a request-queue middleware that serializes ablation calls. |
| User selects too many features → server times out / OOM | Add a per-request feature-count cap (e.g., 50) with a warning. Most useful supernodes will be 5–20 features anyway. |

## B.4 Testing strategy

Add `scripts/pipeline/tests/test_ablation_server.py` (skip-on-no-GPU):

1. `test_request_schema` — validate AblateRequest accepts well-formed input, rejects malformed (missing prompt, negative layer, etc.).
2. `test_response_classification` — submit a known-refusing prompt with no features, confirm response is REFUSE; submit a known-comply prompt with no features, confirm COMPLY. Sanity check classifier wiring.
3. `test_zero_ablation_smoke` — submit one feature, confirm response shape and elapsed_s is reasonable. (This needs a GPU; gate with `pytest.importorskip("torch")` and `not torch.cuda.is_available() → skip`.)
4. `test_cart_export_matches_offline_format` — frontend-side fixture: a cart with 3 features → exported JSON matches `utils.load_cart` schema → loadable by `08_ablate_subcircuits.py --feature-file`.
5. `test_supernode_persistence_format` (after Phase 3) — localStorage schema is documented and round-trippable.

The current local test suite is 244/0/1. Target: **add ~5 tests, end at 249/0/1**, none of them GPU-required for CI (real GPU tests gated). Testing strategy avoids breaking the 244-passing baseline.

## B.5 Dependencies

Adds `fastapi`, `pydantic`, `uvicorn` to the dev-only requirements (already imported guarded in `ablation_server.py`). Add to `pyproject.toml` under an `[project.optional-dependencies] frontend` block so the regular pipeline install doesn't pull them in:

```toml
[project.optional-dependencies]
frontend = ["fastapi>=0.100", "pydantic>=2", "uvicorn>=0.20"]
```

Install for development: `pip install -e ".[frontend]"`.

## B.6 Estimated wall

- Phase 1 (end-to-end test): 1–2 hours.
- Phase 2 (compare.html): 1 day (~6 hours).
- Phase 3 (supernodes): 1–2 days (~12 hours).
- Phase 4 (ergonomics, optional): 1 day.

**Total: 2–3 days for a polished deliverable**; 1 day for an MVP that just works with single-graph + supernodes.

## B.7 Coordination notes

- Ruqiya could pick up Phase 3/4 in parallel with Mahmoud doing Part A — the work is disjoint (Part A is a CLI invocation; Part B is frontend JS). No merge conflicts.
- Phase 1 must complete before Phase 2 (need to know the wiring works before extending it).
- Phase 2 unblocks the most useful new UX (compare two prompts under the same ablation), so prioritize over Phase 3 if time-constrained.

---

# Coordination & precedence

| Deadline | Item | Critical path |
|---|---|---|
| **May 4 — ICML workshop abstract** | Headline numbers from `run_20260430_023247` already captured in REPORT. Part A's result *could* land in the abstract if it runs Mon/Tue, but the existing report numbers are sufficient on their own. **Do not block on Part A.** | Read REPORT, draft abstract. |
| **NeurIPS deadline** | Both Part A and Part B add real value. Part A is the headline-experiment-completeness add (the missing canonical_pro_refusal cell); Part B is the methodology-section live-demo add (interactive ablation as a research tool). Both should land before camera-ready. | Part A first (~6 h H100, low risk). Part B testing in parallel on local CPU. |

# Open questions for the user

1. **Output file naming for Part A**: should the canonical_pro_refusal results overwrite `ablation_results.json` (clean schema, replaces existing) or write to `ablation_results_canonical.json` (additive, preserves the 5-ablation original)? **Recommend additive — cheaper to revert.**
2. **Sweep config for Part A**: run on `subcircuits_k50_f50.json` only, or also on legacy `subcircuits.json` (88-feature corpus-aggregate)? **Recommend k50_f50 first; if result is borderline or surprising, re-run on legacy as cross-check.**
3. **Server hosting for Part B**: does the user want the server running on the user's RunPod pod long-term, or only spun up ad-hoc for analysis sessions? Affects whether to add it to `run_p7.sh` as an optional post-pipeline phase.
4. **Frontend hosting**: Stage 05 frontends are currently static and served via `python3 -m http.server`. For a public release, do we want to ship a single deployable bundle (e.g., GitHub Pages) that points at a remote ablation server (e.g., a HF Space)? Out of scope for this plan but worth flagging.
