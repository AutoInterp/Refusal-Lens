# Refusal-Lens Refactoring Guide

## Goal

Extend the Refusal-Lens shared research package to compute **attribution circuits to the refusal direction** using Anthropic's circuit-tracer framework, then validate the RP-suppresses-refusal hypothesis through feature steering experiments.

This aligns with the research vision in `Attribution_Circuits_to_Refusal_Direction.pdf`:
- **Primary method**: Full multi-hop attribution graphs from input tokens through CLT features to the refusal projection R = ⟨x^(ℓ\*,c\*), r̂⟩
- **Validation method**: Feature steering experiments (the notebook's 6 experiments) to confirm attribution graph predictions
- **Key hypothesis**: Role-playing features suppress refusal features, and this suppression is the causal mechanism by which role-playing jailbreaks bypass safety alignment

---

## Research Approach (from Mentor's PDF)

### Core Idea (Section 1.3)

Instead of attributing to output logits z_v = x^(L) · W_U[:,v], we attribute to the **refusal projection**:

```
R = ⟨x^(ℓ*,c*), r̂⟩
```

where r̂ is the refusal direction (Arditi et al. 2024) and (ℓ\*, c\*) is the measurement point.

In the CLT local replacement model, R is a **linear function** of all upstream feature activations (because attention patterns and LayerNorm denominators are frozen). This means attribution edges A_{s→R} = a_s · w_{s→R} are **exact**, not approximations.

The virtual weight to the refusal direction is mathematically identical to logit attribution, with r̂ replacing W_U[:,v] as the "readout vector."

### How RP ↔ Refusal Investigation Fits

The attribution graph approach gives **stronger evidence** for the RP-suppresses-refusal hypothesis:

1. **Attribution graph on jailbreak prompt** → RP features appear as negative-attribution edges to R (they reduce the refusal signal). This is exact mechanistic evidence.
2. **Contrastive attribution** → Compare graphs for the same harmful request with/without RP wrapper. The structural difference *is* the suppression mechanism.
3. **Multi-hop pathways** → Trace whether RP features suppress refusal features directly or through intermediate computational steps.
4. **Validation via steering** → The 6 notebook experiments confirm that perturbing high-attribution features changes R as the graph predicts.

### Positioning vs. Prior Work

| Work | Method | Target | Multi-hop? |
|------|--------|--------|------------|
| Anthropic Circuit Tracing (2025) | CLT + attribution graph | Output logits | Yes |
| Paleka et al. (2024) | SAE + gradient of refusal direction | Refusal direction | No (single hop) |
| Marks et al. (2024) | SAE + attribution patching | Logit difference | Yes (SAE features) |
| **Our approach** | **CLT + attribution graph** | **Refusal direction** | **Yes (full graph)** |

---

## Refactoring Phases

### Phase A: Foundation

Shared infrastructure needed by both attribution and steering approaches.

#### Step 1: Model Management Module -- COMPLETE
- [x] Create `src/refusal_lens/model_loader.py`
- [x] Implement `ModelConfig` dataclass (model name, dtype, device_map)
- [x] Implement `load_model()` function
- [x] Implement `tokenize_prompt()` using chat template
- [x] Implement `generate_text()` for greedy decoding
- [x] Implement `display_topk()` for logit comparison
- [x] Implement `get_device_info()` for device/VRAM detection
- [x] Add `steering` optional dependency group in `pyproject.toml`
- [x] Add mypy overrides for torch-related modules
- [x] Export new API in `__init__.py`
- [x] Write tests in `test/test_model_loader.py` (39 tests)
- [x] Verify: 120/120 tests pass (81 existing + 39 new)

#### Step 2: SAE/Transcoder Module -- COMPLETE
- [x] Create `src/refusal_lens/sae.py`
- [x] Port `JumpReLUSAE` class (encode, decode, forward, affine skip connection)
- [x] Port `load_sae()` and `load_sae_flexible()` utilities
- [x] Port `load_sae_set()` batch loading helper
- [x] Port `analyze_width_mismatch()` diagnostic
- [x] Export new API in `__init__.py`
- [x] Fix bugs found during verification (3 bugs + conditional base class + string spacing)
- [x] Write tests in `test/test_sae.py` (50 tests: 24 mock, 23 real torch, 3 real HF)
- [x] Verify: 50/50 tests pass (venv), 24 pass + 26 skipped (system Python)

#### Step 3: Dataset Integration -- COMPLETE
- [x] Create `src/refusal_lens/data_loader.py`
- [x] Implement `load_split()` and `load_processed()` to read from local JSON files
- [x] Implement `create_contrastive_dataset()` for train/eval splits
- [x] Match notebook's N_TRAIN=128, N_EVAL=64, seed=42 pattern
- [x] Implement `list_available_splits()` and `list_available_processed()` discovery
- [x] Export new API in `__init__.py`
- [x] Fix bugs found during verification (5 bugs: typos, wrong directory, missing function)
- [x] Write tests in `test/test_data_loader.py` (52 tests, all real data)
- [x] Verify: 196/196 tests pass (81 existing + 39 model_loader + 24 sae + 52 data_loader)

#### Step 4: Refusal Direction Computation -- COMPLETE
- [x] Create `src/refusal_lens/refusal_directions.py`
- [x] Implement `compute_refusal_directions()` (difference-in-means + PCA, from Arditi et al.)
- [x] Implement `collect_resid_acts_multipos()` (multi-position activation collection)
- [x] Implement `gather_residual_activations()` and `get_model_layers()` (hook-based capture)
- [x] Implement `find_best_layer()` (layer with strongest separation)
- [x] Implement serialization: `save_directions()` / `load_directions()` (.pt + summary.json)
- [x] Export new API in `__init__.py`
- [x] Fix bugs found during verification (5 bugs: HAD_TORCH typo, len(seq_len), last_K case, results=[], diff.mean)
- [x] Write tests in `test/test_refusal_directions.py` (40 tests, real torch with FakeModel)
- [x] Verify: 280/280 tests pass (venv), 196 pass + 45 skipped (system Python)

#### Step 4b: Centralized Configuration -- COMPLETE
- [x] Create `src/refusal_lens/config.py`
- [x] Consolidate all constants: `MODEL_NAME`, `DEVICE`, `REFUSAL_LAYERS`, `MEASUREMENT_LAYER`, dataset paths, hyperparameters
- [x] Move scattered constants from `sae.py` (layers, widths) and future modules into config
- [x] Export in `__init__.py`
- [x] Write tests in `test/test_config.py` (29 tests, all real)
- [x] Verify: 225/225 tests pass (system Python), 45 skipped (no torch)

#### Step 5: Extend PromptTemplateLibrary -- COMPLETE
- [x] Add `ROLEPLAY_JAILBREAK`, `MATCHED_HARMFUL`, `COMPLEX_BENIGN` to `PromptCategory` enum
- [x] Add 4 prompt lists as ClassVars: ROLEPLAY_JAILBREAK_PROMPTS, MATCHED_HARMFUL_PROMPTS, COMPLEX_BENIGN_PROMPTS, CLEAN_HARMLESS_PROMPTS (8 each)
- [x] Add `get_matched_pairs()` for contrastive attribution (jailbreak vs bare harmful)
- [x] Add `get_experimental_prompts()` returning all 4 prompt sets by name
- [x] Update `_get_expected_outcome()` and `DEFAULT_TEMPLATES` for new categories
- [x] Backward compatibility preserved (original 4 categories, framing lists, `get_standard_prompts`)
- [x] Write tests in `test/test_prompt_template.py` (36 tests, all real)
- [x] Verify: 261/261 tests pass (system Python), 45 skipped (no torch)

#### Step 6: Bridge Supernode/Neuronpedia Formats -- COMPLETE
- [x] Add `Feature` NamedTuple (layer, feature_idx, token_pos) to `supernode_analyzer.py`
- [x] Add `parse_neuronpedia_url()` — parses graph URLs into named Feature groups
- [x] Add `get_transcoder_for_feature()` — maps Feature to correct 16k/262k transcoder
- [x] Add `supernodes_to_features()` — converts SupernodeData to Feature lists
- [x] Add `NEURONPEDIA_GRAPH_URL`, `ROLEPLAY_FEATURES`, `REFUSAL_FEATURES` constants
- [x] Export all new symbols in `__init__.py` and `__all__`
- [x] Write tests in `test/test_neuronpedia_bridge.py` (44 tests, all real)
- [x] Verify: 305/305 tests pass (system Python), 45 skipped (no torch)

---

### Phase B: Circuit-Tracer Integration

The core of the mentor's vision: modify Anthropic's circuit-tracer to attribute to the refusal direction.

#### Step 7: Circuit-Tracer Setup -- COMPLETE
- [x] Study circuit-tracer library: `attribute()`, `ReplacementModel`, `Graph`, `CustomTarget`, `prune_graph`
- [x] Discover: `CustomTarget` API already supports arbitrary direction attribution (no internals modification needed)
- [x] Discover: Gemma 3 4B IT supported via `nnsight` backend with transcoders at `mwhanna/gemma-scope-2-4b-it`
- [x] Create `src/refusal_lens/clt.py` with guarded imports (`HAS_CIRCUIT_TRACER`)
- [x] Implement `load_replacement_model()` with defaults from config
- [x] Implement `make_refusal_target()` — wraps `CustomTarget` with validation
- [x] Implement `attribute_to_refusal()` — core function, with `layer`/`position` params (warns for intermediate-layer, falls back to last-layer until Step 8 patch)
- [x] Implement `attribute_to_refusal_sweep()` — layer-sweep analysis over `config.REFUSAL_LAYERS`
- [x] Implement `prune_refusal_graph()` — wraps `prune_graph` with sensible defaults
- [x] Implement `extract_top_features()` — reads adjacency matrix, ranks features by |A_{s→R}|
- [x] Export all new symbols in `__init__.py` and `__all__`
- [x] Write tests in `test/test_clt.py` (45 tests: 24 always-run + 21 circuit-tracer-gated)
- [x] Verify: 329/329 tests pass (system Python), 66 skipped (no torch/circuit-tracer)
- [x] Document: intermediate-layer attribution deferred to Step 8 (circuit-tracer patch)

#### Step 8: Intermediate-Layer Attribution (`attribution.py` + vendored patch) -- COMPLETE
- [x] Vendor circuit-tracer under `vendor/circuit-tracer/` for direct modification
- [x] Patch `vendor/.../attribute.py`: add `measurement_layer`/`measurement_position` params, thread to backends
- [x] Patch `vendor/.../attribute_nnsight.py`: add params to `attribute()` and `_run_attribution()`, replace Phase 3 hardcoded `n_layers`/`n_pos-1` with `_ml`/`_mp` variables
- [x] Patch `vendor/.../attribute_transformerlens.py`: same Phase 3 patch as nnsight
- [x] Create `src/refusal_lens/attribution.py` with `attribute_to_direction()` and `validate_measurement_point()`
- [x] Update `src/refusal_lens/clt.py`: `attribute_to_refusal()` delegates to `attribute_to_direction()` (warning removed)
- [x] Export `attribute_to_direction`, `validate_measurement_point` in `__init__.py`
- [x] Write tests in `test/test_attribution.py` (66 tests: validation, signatures, vendored patch verification, delegation, backward compat)
- [x] Update `test/test_clt.py` (45 tests: warning tests replaced with delegation tests)
- [x] Verify: 500/500 tests pass (full suite), 0 skipped
- [x] Key design: `None` defaults preserve original behavior; Phase 4 (feature-to-feature edges) unchanged
- [x] **Demo checkpoint**: Pipeline is end-to-end ready for proof-of-concept attribution to refusal direction

#### Step 9: Attribution Graph Pipeline
- [ ] Create `src/refusal_lens/attribution_pipeline.py`
- [ ] Implement `compute_refusal_graph(prompt, refusal_dir, layer, position)` — end-to-end pipeline
- [ ] Implement `compare_graphs(graph_a, graph_b)` — contrastive attribution between prompts
- [ ] Implement graph pruning wrapper (threshold by |A_{s→R}|)
- [ ] Implement graph export for visualization (JSON format compatible with circuit-tracer frontend)
- [ ] Label output node as "Refusal Direction" in visualization
- [ ] Write tests in `test/test_attribution_pipeline.py`
- [ ] Verify: pipeline runs on harmful/harmless prompts, produces valid graphs

---

### Phase C: Feature Discovery & RP Analysis

Use attribution graphs to discover which features drive refusal and investigate the RP mechanism.

#### Step 10: Feature Discovery from Attribution
- [ ] Create `src/refusal_lens/feature_discovery.py`
- [ ] Implement `discover_refusal_features(graphs)` — rank features by |A_{s→R}| across prompts
- [ ] Implement `discover_rp_features(jailbreak_graphs, harmful_graphs)` — find features with negative attribution on jailbreak but not on bare harmful
- [ ] Implement `compare_with_neuronpedia(discovered, neuronpedia_features)` — do discovered features match the Neuronpedia circuit trace?
- [ ] Implement contrastive analysis: which features appear in jailbreak graphs but not harmful graphs?
- [ ] Write tests in `test/test_feature_discovery.py`
- [ ] Verify: discovered features are sensible, overlap analysis with Neuronpedia works

#### Step 11: RP-Refusal Attribution Analysis
- [ ] Implement `analyze_rp_suppression(jailbreak_graph, harmful_graph)` — quantify how RP features reduce R
- [ ] Implement `trace_suppression_pathway(graph, rp_features, refusal_features)` — find multi-hop paths from RP features to refusal features
- [ ] Implement cross-feature attribution: for each RP feature, which refusal features does it suppress (and through what intermediates)?
- [ ] Implement aggregate statistics over prompt sets
- [ ] Write tests
- [ ] Verify: suppression pathways are identified, statistics are meaningful

---

### Phase D: Validation via Steering

Reposition the notebook's 6 experiments as validation of attribution graph predictions.

#### Step 12: Activation Hooks Module
- [ ] Create `src/refusal_lens/hooks.py`
- [ ] Port `get_model_layers()` (multi-path layer discovery)
- [ ] Port `gather_residual_activations()` and `gather_mlp_io()`
- [ ] Port `make_transcoder_steering_hook()` (MLP replacement)
- [ ] Port `make_sae_feature_steering_hook()` (residual-stream steering)
- [ ] Port `make_multipos_ablation_hook()` for direction ablation
- [ ] Write tests in `test/test_hooks.py`
- [ ] Verify: hooks register/remove cleanly, interventions modify outputs

#### Step 13: Enhance RefusalClassifier
- [ ] Merge notebook's refusal patterns into `refusal_classifier.py`
- [ ] Add `estimate_refusal_prob_logit()` as a fast-path method
- [ ] Add `compute_perplexity()` as a validation method
- [ ] Add `judge_refusal_generation()` for generation-based evaluation
- [ ] Preserve backward compatibility of existing API
- [ ] Update existing tests, add new tests for new methods
- [ ] Verify: all existing tests still pass, new methods work correctly

#### Step 14: Feature Intervention Module
- [ ] Create `src/refusal_lens/intervention.py`
- [ ] Port `InterventionResult` dataclass
- [ ] Port `run_intervention()` function
- [ ] Port `measure_feature_activations()` for empirical distributions
- [ ] Port empirical clamping logic (99th percentile bounds)
- [ ] Write tests in `test/test_intervention.py`
- [ ] Verify: intervention results are structured correctly, clamping works

#### Step 15: Validation Experiment Runner
- [ ] Create `src/refusal_lens/validation_experiments.py`
- [ ] Implement `ValidationExperimentConfig` for each of the 6 experiments
- [ ] Implement `ValidationExperimentRunner` that:
  - Takes attribution graph predictions as input
  - Runs steering experiments to validate predictions
  - Compares observed behavioral changes with predicted attribution magnitudes
- [ ] Implement the 6 validation experiments:
  1. **Ablate RP on Jailbreak** → high-attribution RP features zeroed → R should increase (model refuses)
  2. **Inject RP on Harmful** → inject RP features → R should decrease (refusal bypassed)
  3. **Ablate Refusal** → zero high-attribution refusal features → R should decrease
  4. **Cross-Feature Correlation** → correlate attribution magnitudes with activation correlations
  5. **Combined RP+Refusal** → inject RP + ablate refusal → strongest R decrease
  6. **Control (Complex-Benign)** → inject non-RP features → R should not decrease
- [ ] Implement aggregate summary with validity assessment (PPL ratio < 5, generation judge, attribution-prediction correlation)
- [ ] Write tests in `test/test_validation_experiments.py`
- [ ] Verify: all 6 experiments can be configured and executed

---

## How the RP Investigation Works in This Plan

The RP-suppresses-refusal hypothesis is investigated through **two complementary lenses**:

### Lens 1: Attribution (Steps 10-11)
- Compute attribution graphs on jailbreak prompts vs. bare harmful prompts
- **If RP features suppress refusal**, they will have **negative attribution edges** to R on jailbreak prompts
- The multi-hop graph reveals the *mechanism*: do RP features suppress refusal features directly (same layer) or through intermediate features (cross-layer)?
- Quantify suppression: Σ A_{rp→R} on jailbreak vs. harmful gives the total RP contribution to reducing R

### Lens 2: Steering Validation (Step 15)
- Take features identified by attribution and perturb them
- **Exp 1**: If attribution says RP features reduce R, then ablating them should increase R → model refuses
- **Exp 2**: If attribution says RP features reduce R, then injecting them into harmful prompts should decrease R → model complies
- **Exp 6 (Control)**: Features NOT identified by attribution (complex-benign) should have no effect on R

### Why both lenses together are stronger
- Attribution alone is correlational (which features co-vary with R). Steering establishes causation.
- Steering alone shows *that* features matter but not *how the model computes R*. Attribution shows the computational pathway.
- The combination — attribution predicts, steering validates — is the standard in Anthropic's framework (PDF Section 3.5).

---

## Mentor Feedback & Tejas Cross-Reference (2026-03-29)

After the Step 8 demo, Georg provided feedback and Tejas ran independent experiments on the `tejas-circuit-experiments` branch. This section tracks what was asked, what's been answered, what's missing, and how to integrate Tejas's findings.

### Georg's Feedback — Status

| # | Question / Request | Status | Where to Address |
|---|---|---|---|
| G1 | Share Neuronpedia graph link for interactive exploration | **NOT DONE** | Step 9 `export_graph_json()` produces the JSON; need to upload to Neuronpedia |
| G2 | Sanity check: does higher r̂ coefficient = refuse or not-refuse? | **NOT IN REPO** | Need `project_onto_refusal_direction()` function. Tejas did this (script 01) — confirm harmful > harmless |
| G3 | Measure that jailbreak actually activates r̂ less | **NOT IN REPO** | Need to compute ⟨x^(ℓ,c), r̂⟩ per prompt, compare conditions. Tejas found minimal difference (~-0.095) |
| G4 | Look at features that differ between harmful and jailbreak | **PARTIAL** | Our `compare_graphs()` does top-k comparison. Need full active-feature-set comparison |
| G5 | Look at features that flip positive → negative attribution | **PARTIAL** | Our `compare_graphs()` tracks `sign_flipped`. Demo found 0; Tejas also found 0. Mechanism is magnitude, not sign flip |
| G6 | Look at features inactive in harmful but active in jailbreak | **NOT DONE** | Need to compare full `graph.active_features` tensors, not just top-k |
| G7 | Discovery: what do these features encode? Why are they active? | **NOT DONE** | Phase C (Steps 10-11) + Neuronpedia feature lookup |

### What Tejas Did (tejas-circuit-experiments branch)

Tejas ran 6 experiment scripts on RunPod (RTX 6000 Ada, 48GB) using gemma-3-4b-pt with PLT transcoders. Key work:

1. **Computed refusal direction** at layer 13 via difference-in-means (64 harmful + 64 harmless)
2. **Ran sanity check** (G2/G3): projected harmful, harmless, and jailbreak prompts onto r̂ — found nearly identical refusal projection for all three conditions
3. **Ran attribution** with `CustomTarget` targeting r̂ on 30 matched pairs
4. **Ran steering experiments**: alpha sweep × layer sweep on successful jailbreaks
5. **Tested novel jailbreaks**: designed 8 analytical framings, 7/8 bypassed refusal
6. **Key finding: two jailbreak mechanisms**:
   - **Dampening** (RP jailbreaks): both pro-refusal and anti-refusal features decrease. Circuit disengages. Steerable via r̂.
   - **Tug-of-war** (fiction/analytical jailbreaks): both sides INCREASE massively. Circuit engages more but cancels out. Immune to r̂ steering.

### What Tejas Did Incorrectly / Sub-optimally

1. **Attribution ran at LAST LAYER, not layer 13**: His branch uses the **unpatched** upstream circuit-tracer (`safety-research/circuit-tracer` commit `6d64f60`). His `attribution.py` passes `measurement_layer=13` to `attribute()`, but the upstream `attribute()` does NOT accept that kwarg — it would raise `TypeError` (or be silently ignored depending on version). This is why his script 02 switched to calling `attribute()` directly without measurement params. **All his attribution experiments measure at the unembed layer, not at layer 13 where the refusal direction was computed.**
   - **Fix**: Re-run Tejas's experiments with our patched fork (`AutoInterp/circuit-tracer`, branch `refusal-lens-measurement-patch`) which adds `measurement_layer`/`measurement_position` support.

2. **Used gemma-3-4b-pt (pretrained, not instruction-tuned)**: The base model hasn't been safety-trained, so its feature activations around refusal may differ from the IT model. The refusal direction was computed from the IT model but attribution ran on the base model. This mismatch could explain why refusal projection was nearly identical across conditions.
   - **Note**: This may actually be the intended approach (transcoders are trained on base model activations), but needs validation.

3. **Script 02 bypasses our wrapper**: Calls `circuit_tracer.attribute()` directly instead of `attribute_to_direction()`, losing the measurement-layer routing.

### Experiments to Add (Post Step 9)

Based on Georg's feedback and gaps in Tejas's work, add these experiments:

#### Experiment A: Refusal Direction Polarity Sanity Check (G2 + G3)
- **What**: Compute ⟨x^(ℓ,c), r̂⟩ for harmful, harmless, and jailbreak prompts at the best refusal layer
- **Why**: Confirm the direction is oriented correctly (harmful > harmless), and measure whether jailbreaks reduce the projection
- **How**: Use `gather_residual_activations()` at best layer, dot-product with r̂, compare distributions
- **Implementation**: Add `project_onto_direction(model, tokenizer, prompts, direction, layer)` to `refusal_directions.py` or `attribution_pipeline.py`
- **Expected result**: harmful > harmless (confirms polarity); jailbreak ≈ harmful or slightly less (Tejas found ~equal)

#### Experiment B: Full Feature-Set Comparison (G4 + G5 + G6)
- **What**: Compare the FULL set of active features between harmful and jailbreak graphs, not just top-k
- **Why**: Top-k misses features that are low-attribution in one condition but high in another, or completely inactive vs active
- **How**: For each graph, get `graph.active_features` (all features, not just top-k). Compute set intersection/difference at the (layer, feature_idx) level. For shared features, compute the attribution delta.
- **Implementation**: Add `compare_full_feature_sets(graph_a, graph_b)` to `attribution_pipeline.py`
- **Output**:
  - Features active in jailbreak but inactive in harmful (G6)
  - Features with sign-flipped attribution (G5)
  - Features with largest |attribution_delta| (G4)

#### Experiment C: Re-run Tejas's Experiments with Measurement Layer Patch
- **What**: Re-run the 30-pair attribution experiment with `measurement_layer=13` using our patched circuit-tracer
- **Why**: Tejas's results are at the last layer; attribution at layer 13 (where r̂ was computed) may show different feature rankings and clearer dampening vs tug-of-war patterns
- **How**: Use our `attribute_to_refusal(prompt, model, r_hat, layer=13)` which now works end-to-end

#### Experiment D: Feature Identity Discovery (G7)
- **What**: For the top features identified in Experiments B/C, look them up on Neuronpedia to understand what they encode
- **Why**: Georg wants to understand *why* features flip/activate under jailbreak — need semantic labels
- **How**: Use `get_transcoder_for_feature()` and `parse_neuronpedia_url()` from `supernode_analyzer.py` to generate Neuronpedia dashboard links for discovered features
- **Implementation**: Add `lookup_features_neuronpedia(features)` to `feature_discovery.py` (Step 10)

#### Experiment E: Dampening vs Tug-of-War Classification
- **What**: Reproduce and extend Tejas's key finding — classify jailbreaks by mechanism type
- **Why**: This is a novel finding: RP jailbreaks dampen the refusal circuit, fiction jailbreaks amplify both sides. Understanding this distinction is critical for the research.
- **How**: For each jailbreak type, compute `sum(positive_attributions)` and `sum(negative_attributions)` separately. Dampening: both decrease. Tug-of-war: both increase.
- **Implementation**: Add to Step 11 (`analyze_rp_suppression`)

### Integration with Tejas's Work

Tejas's 6 scripts and results are in `data/tejas_experiments/` on his branch. To integrate:

1. **Merge his results data** into our `data/results/` structure (JSON files, figures)
2. **Re-run his experiments** using our patched circuit-tracer fork with `measurement_layer=13`
3. **Compare last-layer vs layer-13 attribution** — does the measurement point change the dampening/tug-of-war finding?
4. **Add his novel jailbreak prompts** to our `PromptTemplateLibrary` (8 analytical framings)
5. **Validate his steering immunity finding** using our Phase D steering framework

---

## Conference Experiment Procedure (April 2026)

Refactoring is paused at Step 9. Focus shifts to experiment results for a conference submission. The existing scripts (`run_meeting_experiments.py`, `validate_tejas_replication.py`) serve as the infrastructure. Once experiments are complete, these scripts inform the refactoring of Steps 9–15 (the pipeline code will formalize patterns proven in the experiment scripts).

### Priority 1: Scale the Dataset

**Goal**: Move from 5–10 prompts per jailbreak class to 30–50 for statistical power.

| Task | Status | Notes |
|------|--------|-------|
| P1.1: Curate 50 diverse harmful prompts across topic categories (cybercrime, weapons, fraud, social engineering, etc.) | NOT DONE | Draw from `harmful_train.json` (260 available), ensure topic diversity |
| P1.2: Write 5+ jailbreak prefix variants per class (RP, fiction, analytical, completion + any new classes) | NOT DONE | Current script has 1 prefix per class in `JB_CLASSES` dict |
| P1.3: Add jailbreak classes from literature (DAN, few-shot, encoding, multi-turn) if feasible | NOT DONE | Check if single-turn prefixes can capture these; multi-turn may need different infra |
| P1.4: Run full attribution on 50 prompts × 5 classes = 250 attribution graphs | NOT DONE | ~5 min per graph on A40 → ~21 hours. May need to batch across RunPod sessions |
| P1.5: Compute per-class statistics with confidence intervals | NOT DONE | Bootstrap CI on mean net attribution per class |
| P1.6: Paired statistical tests (Wilcoxon signed-rank or paired t-test) per class vs bare | NOT DONE | Need p-values for each class's effect |

**Script**: Extend `run_meeting_experiments.py` Phase 5, or create `scripts/run_scaled_experiments.py`
**Refactoring link**: Results inform `attribution_pipeline.py` (`batch_attribute`, `aggregate_features`) and `feature_discovery.py` design

### Priority 2: Causal Intervention (Experiment B from report)

**Goal**: Confirm that top features are *causally necessary* for refusal, not just correlated.

| Task | Status | Notes |
|------|--------|-------|
| P2.1: Implement feature activation clamping via forward hooks on the ReplacementModel or IT model | NOT DONE | Hook `model.language_model.layers[L]` to clamp specific transcoder feature activations |
| P2.2: For top 5 pro-refusal features (L29:F1066, L28:F305, L29:F6752, L31:F498, L25:F963), clamp activation to 0 on bare harmful prompt → measure if refusal decreases | NOT DONE | Single-feature ablation |
| P2.3: For top 5 features, clamp activation to bare-harmful value during jailbreak prompt → measure if refusal is restored | NOT DONE | Activation patching: force jailbreak to use bare activations |
| P2.4: Multi-feature ablation: clamp top-5 simultaneously → stronger effect? | NOT DONE | Test necessity of the set, not just individuals |
| P2.5: Measure effect via both (a) refusal direction projection change and (b) generation-based refusal detection | NOT DONE | Need both metrics — projection is internal, generation is behavioral |
| P2.6: Report effect sizes and compare to baseline (no intervention) | NOT DONE | |

**Script**: Create `scripts/run_causal_intervention.py`
**Refactoring link**: Results inform `intervention.py` (Step 14) and `validation_experiments.py` (Step 15) design. Tejas's `steer_and_generate()` hook pattern is the starting point.

### Priority 3: Feature Interpretation (Experiment E from report)

**Goal**: Label top features with human-readable descriptions so the paper tells a mechanistic story.

| Task | Status | Notes |
|------|--------|-------|
| P3.1: Generate Neuronpedia URLs for top-20 features from Phase 3 results | NOT DONE | Use `get_transcoder_for_feature()` from `supernode_analyzer.py` to build dashboard links |
| P3.2: Look up max-activating examples on Neuronpedia for each feature | NOT DONE | May need to check if Gemma-3-4b-it features are indexed on Neuronpedia |
| P3.3: If not on Neuronpedia, compute max-activating examples from our dataset | NOT DONE | Run all 260 harmful prompts, record which features fire most strongly |
| P3.4: Write human-readable labels for top-20 features (e.g., "harmful content detector", "role-play frame detector") | NOT DONE | Manual labeling based on max-activating examples |
| P3.5: Create a feature reference table for the paper | NOT DONE | Feature ID, layer, mean attribution, label, example activating prompt |

**Script**: Create `scripts/lookup_features.py`
**Refactoring link**: Results inform `feature_discovery.py` (Step 10) design

### Secondary Priorities (after P1–P3)

#### S1: Statistical Rigor Addition
- Add p-values (paired Wilcoxon) to existing Phase 5 results
- Bootstrap 95% CI on all class means
- Compute Cohen's d effect sizes per jailbreak class vs bare
- Add to existing experiment report or generate supplementary stats table

#### S2: Completion Paradox Deep-Dive
- Compare full feature activation profiles of completion-framed vs bare prompts
- Identify features uniquely recruited by "Complete this manual entry:" framing
- Investigate if the model has a "jailbreak detection" circuit that completion triggers
- Could be a standalone finding for the paper

#### S3: Attention Head Attribution
- Extract attention head edges from existing graphs (already in adjacency matrix, just not analyzed)
- Quantify attention vs MLP contribution to refusal per layer
- Determine if jailbreaks primarily affect attention or MLP pathways
- Addresses the "0.4% MLP" caveat

#### S4: Cross-Model Generalization
- Run pipeline on Gemma-3-12B-IT or another model with CLT transcoders
- Compare feature-level findings across model sizes
- Deferred until after first paper draft

### Experiment Results Tracking

| Run | Date | Script | Config | Output Dir | Key Result |
|-----|------|--------|--------|-----------|-----------|
| Demo (gemma-2-2b-it) | 2026-03-26 | `demo_attribution.py` | 16 pairs, layers 7-15 | `data/results/demo/runs/20260326_110155` | Pipeline works e2e |
| Demo comprehensive | 2026-03-26 | `demo_attribution.py` | 64 dir, 8 pairs, layers 7-15 | `data/results/demo/runs/20260326_111432` | 15 suppression candidates |
| Tejas validation | 2026-04-03 | `validate_tejas_replication.py` | 10 pairs, layer 32 | `data/results/meeting_experiments` | Matches Tejas exactly |
| Meeting experiments | 2026-04-05 | `run_meeting_experiments.py` | 10 pairs, 5 classes, layer 32 | `data/results/meeting_experiments` | Fiction -59.8%, Completion +11.4% |
| Scaled run | TBD | `run_scaled_experiments.py` | 50 prompts × 5 classes | TBD | Pending P1 |
| Causal intervention | TBD | `run_causal_intervention.py` | Top-5 features | TBD | Pending P2 |

---

## Overlap Reconciliation

| Concern | Notebook | Mentor's PDF | Resolution |
|---------|----------|-------------|------------|
| Primary method | Feature steering | CLT attribution graphs | Attribution first, steering as validation |
| Transcoder type | Gemma Scope 2 (single-layer) | Cross-Layer Transcoder (CLT) | Use CLTs from circuit-tracer |
| Feature identification | Pre-selected from Neuronpedia URL | Discovered via attribution to R | Attribution discovery, compare with Neuronpedia |
| Model framework | Standard model + PyTorch hooks | CLT replacement model (frozen attn/LN) | Use circuit-tracer's ReplacementModel |
| Refusal direction | Difference-in-means (matches Arditi) | Difference-in-means (matches Arditi) | Same approach, shared implementation |
| Refusal patterns | 8 refusal + 3 compliance | Not specified | Merge notebook patterns into repo's lists |
| Classification tiers | Binary (refusal/compliance) | Not specified | Keep repo's 3-tier, add logit proxy |
| Prompt generation | Hardcoded lists | Not specified | Extend template library with new categories |
| Datasets | HF runtime download | Not specified | Use repo's local splits |
| Supernode data | Neuronpedia URL parsing | Not specified | Bridge both formats |
| Validation | Standalone experiments | Validate attribution predictions | Experiments validate graph predictions |

---

## Verification Strategy

For each step:
1. **Real tests** -- test with actual libraries and real data (no mocks); skip tests when deps are missing
2. **Integration check** -- verify existing tests still pass (`pytest test/`)
3. **Import check** -- verify the module imports without requiring GPU/model
4. **Alignment check** -- confirm the code serves the attribution-to-refusal-direction goal

Phase-level verification:
- **Phase A complete**: Can compute refusal directions from contrastive datasets
- **Phase B complete**: Can compute attribution graphs targeting refusal direction R
- **Phase C complete**: Can discover features upstream of refusal, identify RP suppression pathways
- **Phase D complete**: Steering experiments validate attribution graph predictions

Final verification: Create reproducible scripts in `scripts/` directory:
- `scripts/compute_directions.py` — Step 1: compute r̂ from contrastive data, save to `data/results/computed_directions/`
- `scripts/compute_circuits.py` — Step 2: build attribution graph to R, save to `data/results/circuits/`
- `scripts/explore_supernodes.py` — Step 3: supernode exploration
- `scripts/run_validation.py` — Step 4: run all 6 steering validation experiments, save to `data/results/validation/`

Output directory structure:
```
data/results/
├── computed_directions/   # .pt files per layer + summary.json
├── circuits/              # Attribution graph outputs
├── features/              # Discovered feature rankings
└── validation/            # Steering experiment results
```

---

## Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `src/refusal_lens/model_loader.py` | A | Model loading, tokenization, generation |
| `src/refusal_lens/sae.py` | A | JumpReLU SAE/Transcoder class and loading |
| `src/refusal_lens/data_loader.py` | A | Dataset loading from local JSON splits |
| `src/refusal_lens/refusal_directions.py` | A | Refusal direction r̂ computation |
| `src/refusal_lens/config.py` | A | Centralized configuration (model, layers, paths, hyperparams) |
| `src/refusal_lens/attribution.py` | B | `attribute_to_direction()` — core contribution |
| `src/refusal_lens/attribution_pipeline.py` | B | End-to-end graph computation pipeline |
| `src/refusal_lens/feature_discovery.py` | C | Feature ranking and RP analysis from graphs |
| `src/refusal_lens/hooks.py` | D | PyTorch hooks for steering validation |
| `src/refusal_lens/intervention.py` | D | Unified intervention runner with perplexity |
| `src/refusal_lens/validation_experiments.py` | D | 6 validation experiments |
| `scripts/compute_directions.py` | Final | Reproducible script for computing r̂ |
| `scripts/compute_circuits.py` | Final | Reproducible script for attribution graphs |
| `scripts/run_validation.py` | Final | Reproducible script for steering validation |

## Files to Modify

| File | Changes |
|------|---------|
| `src/refusal_lens/__init__.py` | Export new modules in public API |
| `src/refusal_lens/refusal_classifier.py` | Add notebook patterns, logit proxy, perplexity |
| `src/refusal_lens/supernode_analyzer.py` | Add Neuronpedia URL parsing, Feature namedtuple |
| `src/refusal_lens/prompt_template.py` | Add new prompt categories |
| `pyproject.toml` | Add circuit-tracer and other optional dependencies |

## Key Dependencies

| Package | Purpose | Phase |
|---------|---------|-------|
| `torch`, `transformers`, `accelerate` | Model loading and inference | A |
| `safetensors`, `huggingface_hub` | SAE/Transcoder weight loading | A |
| `circuit-tracer` | CLT replacement model, attribution graphs | B |
| `einops` | Tensor operations | A |

## Deployment: Running Experiments on GPU

The experiments (especially Phase B attribution graphs and Phase D steering) require a GPU with enough VRAM to hold Gemma 3 4B IT in bfloat16 (~8 GB) plus transcoders and activation buffers. Two deployment options:

### Option A: Local GPU

If you have a local NVIDIA GPU (RTX 3090/4090 with 24 GB, or A6000 with 48 GB):

1. Clone the repo and create a venv:
   ```bash
   git clone <repo-url> && cd Refusal-Lens
   python3 -m venv venv && source venv/bin/activate
   pip install -e ".[steering]"   # installs torch, transformers, etc.
   ```
2. Run scripts directly:
   ```bash
   python scripts/compute_directions.py
   python scripts/compute_circuits.py
   python scripts/run_validation.py
   ```
3. Results land in `data/results/`.

### Option B: RunPod (Cloud GPU)

RunPod provides on-demand GPU instances. Recommended workflow:

#### 1. Choose a pod template
- **GPU**: A100 40 GB or A100 80 GB (best price/perf for 4B model + transcoders)
- **Cheaper alternative**: RTX A6000 48 GB or RTX 4090 24 GB (tight but workable for 4B)
- **Template**: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` (or latest PyTorch template)
- **Disk**: 50 GB container disk (model weights get cached to HF hub cache)

#### 2. Setup script (run once when pod starts)
```bash
# Clone repo
git clone <repo-url> /workspace/Refusal-Lens
cd /workspace/Refusal-Lens

# Install package with all deps
pip install -e ".[steering]"

# Pre-download model and transcoder weights (avoids re-downloading on restart)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('google/gemma-3-4b-it')
AutoModelForCausalLM.from_pretrained('google/gemma-3-4b-it')
"

# Verify GPU is available
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

#### 3. Run experiments
```bash
cd /workspace/Refusal-Lens

# Phase A: Compute refusal directions (~5-10 min)
python scripts/compute_directions.py

# Phase B: Build attribution graphs (~30-60 min depending on prompt count)
python scripts/compute_circuits.py

# Phase D: Steering validation (~30-60 min)
python scripts/run_validation.py
```

#### 4. Save results before terminating pod
```bash
# Option 1: Push results to git
cd /workspace/Refusal-Lens
git add data/results/
git commit -m "Add experiment results from RunPod run"
git push

# Option 2: Download via RunPod UI or rsync
# RunPod provides a file browser in the web terminal, or use:
# rsync -avz /workspace/Refusal-Lens/data/results/ <local-destination>
```

#### 5. Cost optimization tips
- Use **Spot/Interruptible** instances for ~60% cost savings (experiments are resumable since each script saves incrementally)
- Use **Community Cloud** tier for cheaper rates
- **Stop the pod** (don't terminate) if you need to pause — disk persists, GPU billing stops
- Pre-download model weights to a **Network Volume** ($0.07/GB/month) to avoid re-downloading on new pods

### GPU VRAM Requirements (Estimates)

| Experiment | Min VRAM | Recommended |
|-----------|----------|-------------|
| Compute r̂ (Step 4) | 16 GB | 24 GB |
| Attribution graphs (Steps 8-9) | 24 GB | 40 GB |
| Steering validation (Step 15) | 16 GB | 24 GB |
| Full pipeline (all phases) | 24 GB | 40-80 GB |

*Note: VRAM estimates may change once circuit-tracer integration is tested. The CLT replacement model adds overhead beyond the base model.*

### Intermediate Demo: Proof-of-Concept (after Step 8)

**Script**: `scripts/demo_attribution.py`
**Model**: `google/gemma-2-2b-it` (smaller, fits on local 24GB GPU)
**Transcoders**: Gemma Scope 2 2B IT (16k width)
**Goal**: End-to-end proof that the pipeline works — show mentor preliminary results

#### What the demo does:
1. Load contrastive dataset (16 harmful + 16 harmless pairs)
2. Compute refusal direction r̂ via difference-in-means
3. Load ReplacementModel with transcoders
4. Run attribution on 1 harmful prompt → extract top-20 features by |A_{s→R}|
5. Run attribution on 1 RP-jailbreak prompt → extract top-20 features
6. Compare: which features appear in jailbreak but not harmful? (RP candidates)
7. Output: table of features + attribution values, optional matplotlib plots

#### Expected outputs:
- `data/results/demo/top_features_harmful.json` — top-20 features for harmful prompt
- `data/results/demo/top_features_jailbreak.json` — top-20 features for jailbreak prompt
- `data/results/demo/comparison.json` — contrastive analysis (shared, harmful-only, jailbreak-only)
- `data/results/demo/attribution_bar_chart.png` — bar chart of A_{s→R} for top features
- `data/results/demo/layer_sweep.png` — attribution magnitude across layers (if sweep is run)

#### VRAM requirement:
- gemma-2-2b-it in bfloat16 ≈ 4 GB + transcoders ≈ 2 GB + buffers ≈ 4 GB → ~10 GB minimum
- Should fit on RTX 3060 12GB (tight) or any 24GB+ card comfortably

### Deployment Checklist (to fill in after Phase D is complete)
- [ ] Create `scripts/setup_runpod.sh` — one-command pod setup
- [ ] Add `HF_TOKEN` handling for gated models (if Gemma requires it)
- [ ] Add progress logging so long runs can be monitored via `tail -f`
- [ ] Add checkpoint/resume logic to scripts so interrupted runs can continue
- [ ] Test full pipeline end-to-end on RunPod before final experiments
- [ ] Document expected wall-clock time per script on different GPU tiers

---

## References

- **Ameisen et al. (2025)** — Circuit tracing framework (CLT, attribution graphs)
- **Arditi et al. (2024)** — Refusal direction (difference-in-means)
- **Paleka, Chlenski, Arditi (MATS 2024)** — Finding features causally upstream of refusal (closest prior work, single-hop)
- **Marks et al. (2024)** — Sparse feature circuits (SAE attribution patching)
- `circuit-tracer` library: https://github.com/safety-research/circuit-tracer
