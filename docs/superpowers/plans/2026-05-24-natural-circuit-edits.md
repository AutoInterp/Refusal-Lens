# Natural Circuit Edits — EMNLP 2026 Phase 1 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Georg's 2026-05-24 paper pivot. Show that circuit-tracer-guided feature edits can match a **natural** baseline (position-wise direction subtraction), where natural = something the model could in principle produce on its own from prompt context. The unnatural Arditi baseline (subtract canonical r̂ at every position) is now demoted; the new headline is **closing the gap between circuit edits and natural direction subtraction**.

**Architecture:** Five sub-experiments split across CPU (analysis) and GPU (intervention runs). Sub-experiments E1-E2 establish the new natural baseline. E3 produces the feature-set candidates (active, deactivated, newly-activated). E4-E5 are the actual circuit-edit interventions that test whether the natural baseline can be matched. All experiments run on both Gemma-3-4B-IT (L15) and Qwen3-4B (L18). Attribution graphs + 0a linearization from Batches 14-17 are reused; no new graph generation required.

**Tech Stack:** Python 3.11, PyTorch 2.x, HF Transformers, matplotlib, statsmodels for Wilson CIs. Reuses existing edge_ablation_hook.py and the Qwen/Gemma drivers, extended with new hook factories for per-position direction subtraction and combined ablation+addition operations.

**Spec reference:** `PAPER_OUTLINE_v2_emnlp.md` § 7 "📋 New experiments" (the 2026-05-24 pivot).

**Branch:** `emnlp-perm-edit` (active branch; all commits land here).

**Estimated total cost:** ~18-22 hr GPU + ~2 hr CPU + ~$65-80 in RunPod credits.

**Timeline target:** 5-7 days from kickoff to all results in hand.

---

## File Structure

```
scripts/emnlp_perm_edit/
├── 00_per_position_alpha_audit.py            (E0 — CPU; new)
├── 00_natural_direction_sweep.py             (E1 — GPU; new; uses positions_LXX/pos_-N.pt per position)
├── 00_feature_delta_clustering.py            (E3 — CPU; new; harmful vs JB-broken active-set diff)
├── 00_combined_feature_ablation.py           (E4 — GPU; new; ablate deactivated + add newly-active)
├── edge_ablation_hook.py                     (extended: per-position direction hook)
└── ...existing files...

data/results/emnlp_perm_edit/phase1_natural_edits/
├── per_position_alpha_audit.json             (E0)
├── natural_direction_sweep_gemma.json        (E1)
├── natural_direction_sweep_qwen.json         (E2; uses E1 driver)
├── feature_delta_clusters.json               (E3)
├── combined_ablation_gemma.json              (E4)
├── combined_ablation_qwen.json               (E5)
├── PHASE1_NATURAL_SUMMARY.md
├── natural_baseline_dose_response_figure.png
├── circuit_vs_natural_comparison_figure.png
└── feature_delta_heatmap_figure.png
```

---

## Sub-experiments

### E0 — Per-position alpha decomposition audit (CPU, ~1-2 hr)

**Motivation:** Georg explicitly flagged this in the 2026-05-24 meeting. Mahmoud said "before I was taking the average across all positions rather than a per position alpha" — code audit revealed that 0a (`linearization_decomposition.json`) emits ONE scalar `all_signed` per (prompt, condition), summed over all source positions feeding into the target node at pos=-2 (Gemma) or pos=-1 (Qwen). Our edge-ablation hook applies this scalar at every position uniformly when `position_mode="all"`. This is "average-y" in spirit and a likely contributor to why 0b results look the same at "all positions" vs "pos=-2 only" (~10% either way).

**What we'll produce:** a refactored 0a that *also* emits per-source-position deltas alongside the existing aggregate. The packed graphs already contain per-edge `ctx_idx` (source position index); we group edge attributions by ctx_idx and sum within each group.

**Output schema** (added to `linearization_decomposition.json`):

```json
{
  "per_prompt": [
    {
      "prompt_idx": 0, "condition": "bare",
      "all_signed": -44033,
      "per_position": {
        "-15": -42, "-14": -103, ..., "-2": -38291, "-1": 0
      },
      ...existing fields...
    }
  ]
}
```

**Tasks:**
- [ ] E0.1 Read 1 Gemma graph, dump unique `ctx_idx` values found across edge sources. Confirm they span positions -15..-1.
- [ ] E0.2 Extend `graph_loader.py:extract_edge_records_to_target` to optionally group by source `ctx_idx`.
- [ ] E0.3 Modify `00_linearization_decomposition.py` to add `per_position` field per record (preserves existing schema, additive only).
- [ ] E0.4 Run 0a on Gemma graphs + verify sum_over_positions(per_position[i]) == all_signed.
- [ ] E0.5 Same for Qwen graphs.
- [ ] E0.6 Sanity-print per-position distribution for a few prompts: which positions carry the bulk of the signal?

**Decision point after E0:** If per-position deltas reveal that pos=-2 (Gemma) / pos=-1 (Qwen) carries near-100% of the signal, then "uniform-application-at-every-position" was effectively equivalent to "pos=-2 only" with a slight tail. If the distribution is broader, the previous edge-ablation experiments were genuinely applying-the-wrong-magnitude-at-most-positions, and the new per-position alpha is meaningfully different.

---

### E1 — Natural direction sweep (Gemma L15) — the new baseline (GPU, ~4 hr, ~$15)

**Motivation:** Per Georg, the natural intervention is subtracting the per-position refusal direction `r̂_pos=-i` at each position `-i` simultaneously. This is "literally the difference between a normal prompt and a jbroken prompt" at each position, expressing what the model could plausibly produce given different prompt context.

**Driver:** `00_natural_direction_sweep.py` — new script. Loads `positions_L15/pos_-N.pt` for N=1..15 (already on disk in `pipeline_runs/run_20260430_023247/01_direction/positions_L15/`).

**Hook factory (new):** `make_per_position_subtraction_hook(positions_to_r: dict[int, Tensor], coeff: float, target_layer: int)`. For each position `i` in the prompt sequence, applies `h[i] -= coeff · r_i` where `r_i = positions_to_r[i]`. Skips positions outside the calibrated range (e.g., if prompt has more than 15 tokens, leave older positions untouched).

**Sweep dimensions:**
- Coefficients: same as EXP 1, {0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0}.
- Subtract at all 15 calibrated positions (pos=-1 through pos=-15). The hook checks the prompt length and applies only to valid positions.

**Tasks:**
- [ ] E1.1 Verify per-position direction files exist for Gemma L15 (positions -1..-15 in `positions_L15/`).
- [ ] E1.2 Implement `make_per_position_subtraction_hook` in `edge_ablation_hook.py`. Unit-test: edit at position N matches single-direction subtraction of `r_N`.
- [ ] E1.3 Write `00_natural_direction_sweep.py` (driver). Defaults: Gemma-3-4B-IT, L15, fp32, all 11 conditions × 50 prompts.
- [ ] E1.4 Smoke test on 3 prompts × {0.005, 1.0} coefficients. Verify bare-flip at coeff=1.0 is **somewhere between** EXP 1 all-positions (100%) and EXP 2 pos=-2-only (20%). Likely in the 40-80% range.
- [ ] E1.5 Run full sweep. Save to `natural_direction_sweep_gemma.json`.
- [ ] E1.6 Aggregate flip rates with Wilson CIs against per-prompt baselines.

**Expected wall:** ~30 min/cell × 8 cells = ~4 hr on H100 SXM fp32.

---

### E2 — Natural direction sweep (Qwen L18) (GPU, ~4 hr, ~$15)

**Motivation:** Cross-model replication of the natural baseline.

**Driver:** Same as E1, with `--model Qwen/Qwen3-4B`, `--target-layer 18`, `--directions-dir positions_L18/`.

**Tasks:**
- [ ] E2.1 Verify Qwen per-position direction files (positions_L18/pos_-1.pt through pos_-15.pt).
- [ ] E2.2 Smoke test (3 prompts × {0.005, 1.0}). Verify bare-flip ranges between Qwen EXP 1 all-positions (95%) and EXP 2 pos=-1-only (45%).
- [ ] E2.3 Run full sweep on Qwen at L18. Save to `natural_direction_sweep_qwen.json`.
- [ ] E2.4 Aggregate flip rates.

**Expected wall:** ~30 min/cell × 8 cells = ~4 hr.

---

### E3 — Feature-delta clustering (CPU, ~1 hr)

**Motivation:** Georg's circuit-edit hypothesis. We've only been ablating features active in the harmful attribution graph. Per his suggestion, we also need to identify:
- **Deactivated set**: features active in *harmful* graph but absent (or below activation threshold) in *JB-broken* graph → these are the "refusal-promoting" features that the jailbreak suppresses.
- **Newly-activated set**: features absent in *harmful* graph but present in *JB-broken* graph → these are the "compliance-supporting" features the jailbreak introduces.

The hypothesis is that active-only ablation underperforms (especially on Gemma where active-only is 6% bare flip) because we're only doing half the operation. A natural JB prompt both suppresses some features and activates others.

**Analysis:** For each (prompt_idx, jb_class) pair, load:
- The harmful attribution graph (bare condition)
- The JB-broken attribution graph (jb_<class> condition)

Compute the set difference of "active features" (any feature with non-zero edge attribution into the target node) between the two graphs.

**Output schema:**

```json
{
  "per_prompt_class": [
    {
      "prompt_idx": 0, "jb_class": "fiction",
      "deactivated_features": [
        {"layer": 13, "feature_idx": 427, "abs_attribution_harmful": 5234.1, "abs_attribution_jb": 0.0},
        ...
      ],
      "newly_activated_features": [
        {"layer": 7, "feature_idx": 891, "abs_attribution_harmful": 0.0, "abs_attribution_jb": 1832.5},
        ...
      ],
      "preserved_features": [...features active in both...]
    },
    ...
  ],
  "summary": {
    "median_deactivated_count_per_pair": 47,
    "median_newly_activated_count_per_pair": 22,
    "feature_overlap_across_classes": {
      "fiction_vs_roleplay": 12, ...
    }
  }
}
```

**Tasks:**
- [ ] E3.1 Implement `00_feature_delta_clustering.py` — reads packed graphs from Gemma + Qwen runs.
- [ ] E3.2 For each (prompt_idx, jb_class), compute active-feature set difference. Record counts + attribution magnitudes.
- [ ] E3.3 Cross-class overlap: which features are in the deactivated set for *all* JB classes (= "universal refusal features" that all JB types disable)? Which are class-specific?
- [ ] E3.4 Symmetric question for newly-activated.
- [ ] E3.5 Output clusters + summary stats. Render `feature_delta_heatmap_figure.png`.

**Decision point after E3:** Are deactivated and newly-activated sets stable across prompts in a class (i.e., a consistent "jailbreak signature")? Or per-prompt idiosyncratic? If stable, the active+newly-active extension should generalize well. If per-prompt, we need per-prompt feature sets in E4.

---

### E4 — Combined ablation + addition (Gemma L15) (GPU, ~4 hr, ~$15)

**Motivation:** The headline new experiment. For each (prompt, condition), simultaneously:
1. Ablate the *deactivated* features (set their activation to zero in the residual). This is what existing 0b does at the algebraic-projection level — we now need to do it at the actual-feature level via the CLT.
2. Inject the *newly-activated* features (add their CLT encoder direction × the JB-graph's activation magnitude to the residual).

**Hook factory (new):** `make_combined_feature_hook(deactivated_features: list, newly_activated_features: list, clt_encoder, target_layer)`. Operates on the L15 residual:
- For each `(layer, feat_idx)` in deactivated: subtract `attribution_harmful · clt_encoder[layer][feat_idx]` from h[L15].
- For each `(layer, feat_idx)` in newly_activated: add `attribution_jb · clt_encoder[layer][feat_idx]` to h[L15].

**Comparison conditions:**
- (a) Active-only ablation (baseline, what existing 0b does): subtract deactivated features only.
- (b) Combined: subtract deactivated + add newly-activated.
- (c) Newly-activated-only (control): add newly-activated features only.

For each condition, compare to:
- Natural direction baseline at the equivalent magnitude (from E1).
- EXP 1 reference dose-response.

**Tasks:**
- [ ] E4.1 Load Gemma CLT encoder weights from `mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine` (or use circuit-tracer's loader).
- [ ] E4.2 Implement `make_combined_feature_hook` in `edge_ablation_hook.py`.
- [ ] E4.3 Write `00_combined_feature_ablation.py` (driver). Uses E3's clusters as input.
- [ ] E4.4 Smoke test on 3 prompts × 1 JB class. Verify the hook produces well-defined classifications (not gibberish), bare-flip rate plausibly higher than active-only (which is 6%).
- [ ] E4.5 Run full experiment on all 50 prompts × 5 JB classes × 3 conditions.
- [ ] E4.6 Aggregate flip rates.

**Expected wall:** ~30 min/condition × 3 conditions × 5 JB classes = ~7.5 hr. Aggressive scoping: run only the 3 most informative JB classes (fiction, roleplay, analytical), reducing to ~4.5 hr.

**Expected outcome decisions:**
- If active+newly-active flips ≥ 60% on bare → matches the natural baseline → headline result lands.
- If still ~10-20% → the natural-baseline ceiling is genuinely unreachable by circuit edits at L15. Either layer's not the right target or features are insufficient.
- If newly-activated-only also flips meaningfully → addition matters as much as ablation; complete reframing of the circuit-edit story.

---

### E5 — Combined ablation + addition (Qwen L18) (GPU, ~3 hr, ~$10)

**Motivation:** Cross-model replication. **Lower priority than E4** because Qwen edge ablation already achieves 95% bare flip with active-only ablation; the active+newly-active extension is less critical there. But still worth verifying the methodology works on Qwen for paper-completeness.

**Tasks:**
- [ ] E5.1 Load Qwen CLT from `mwhanna/qwen3-4b-transcoders`.
- [ ] E5.2 Adapt E4's driver to Qwen paths.
- [ ] E5.3 Run on 50 prompts × 5 JB classes × 3 conditions. Reduced scope if Qwen active-only already hits ~95%.

---

## Cross-cutting tasks

### Aggregation + figures

- [ ] X.1 Extend `00_aggregate_phase0_gpu.py` (or new `00_aggregate_phase1.py`) to produce per-experiment flip-rate tables.
- [ ] X.2 Figure: `natural_baseline_dose_response_figure.png` — E1/E2 dose-response curves with EXP 1 (Arditi) overlay so the gap between "natural" and "Arditi maximum" is visually clear.
- [ ] X.3 Figure: `circuit_vs_natural_comparison_figure.png` — E4/E5 active-only vs combined vs natural baseline.
- [ ] X.4 Figure: `feature_delta_heatmap_figure.png` — E3 cluster visualization (rows = JB classes; cols = feature IDs; cells = #prompts where this feature is deactivated/newly-activated).

### Sanity checks

- [ ] S.1 Re-verify Cell B (Batch 14) interpretation given E0 findings. If per-position deltas are dominated by pos=-2, then Cell B's 20% bare flip is faithful to the linearization identity at that position.
- [ ] S.2 Confirm that natural baseline at coeff=1.0 is strictly less than EXP 1 all-positions coeff=1.0 on both Gemma and Qwen. (If it's *equal*, the "natural vs unnatural" framing collapses.)

---

## Phase boundary

This plan is "Phase 1: Natural Edits." It depends on:
- ✅ Phase 0 results (attribution graphs, 0a linearization on both models)
- ✅ Batches 14, 15, 17 (direction sweep, edge ablation baseline)
- ✅ Per-position direction files saved in Stage 01 outputs

Outputs feed:
- EMNLP paper § 3 (natural-vs-unnatural spectrum) — E1, E2
- EMNLP paper § 4 (circuit-tracer decomposition + feature deltas) — E3
- EMNLP paper § 5 (the headline closure section) — E4, E5
- EMNLP paper § 6 (layer locator — also needs re-running with magnitude normalization, but that's a separate item)

---

## Open questions / decision points

1. **After E0**: how spatially distributed are the per-position deltas? Concentrated at pos=-2 or broadly spread?
2. **After E3**: are deactivated and newly-activated sets stable per JB class, or per-prompt? If per-prompt, E4 needs per-prompt feature lists.
3. **After E4**: does the active+newly-active extension on Gemma actually close the gap to the natural baseline? If yes, we have a clean paper. If no, we need to either:
   - Move to deeper layers (the band L12-L24 from Batch 14)
   - Add more feature classes (errors, embeddings) into the combined operation
   - Acknowledge that some refusal signal isn't captured by the CLT decomposition at this layer

4. **Layer locator redo** (separate; not in this plan): re-run the layer locator with magnitude-normalized coefficients so the comparison across layers is fair. Add as a small follow-up plan if needed.

---

*Plan created 2026-05-24 in response to Georg's pivot from magnitude-gap to natural-circuit-edits thesis.*
