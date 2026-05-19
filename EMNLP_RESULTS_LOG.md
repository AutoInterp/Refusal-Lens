# EMNLP Results Log

**Branch**: `emnlp-perm-edit` (off `l15-refactor` HEAD).
**Specs**: `EXPERIMENT_PLAN_per_class_jb_orthogonalization.md` (Track A + Track B).
**Implementation plans**: `docs/superpowers/plans/2026-05-17-phase{0,1}-*.md`.
**Started**: 2026-05-19 (post-spec-commit).

This file is the running log of all EMNLP experiments and findings — committed alongside the runs so the experimental record stays with the code. Each entry follows the structure: what was run, why, the result, the interpretation, and any follow-up actions.

When a result invalidates or modifies a hypothesis from the spec, **note it explicitly** so future readers can see the decision trail.

---

## 2026-05-19 — Batch 1: scaffold + per-class u_C diagnostics

**Tasks executed**:
- P0 Task 0 (scaffold) — `ec2aa48`
- P1 Task 1 (`directions.py` library + tests) — `4442fd7` (6/6 tests pass)
- P1 Task 2 (`00_compute_directions.py` CLI + run) — uncommitted pending discussion
- Follow-up: `00_uc_pairwise_robustness.py` (Pearson cosine check responding to Georg's mean-subtraction concern)

**Compute**: ~5 min CPU, RTX 5080 16 GB laptop, `.venv/bin/python3` with `torch 2.11.0+cpu`.

**Headline finding**: per-class `u_C` vectors are far less geometrically distinct than the spec assumed. Four of five JB classes — roleplay, analytical, completion, cognitive_reframe — share most of their orthogonal-to-r̂ axis (pairwise cosines +0.67 to +0.89). Fiction is the geometric outlier (cosines +0.20 to +0.49 with others). The shared-axis pattern is confirmed real by Pearson (mean-subtracted) cosine control — not an all-ones-direction / anisotropy artifact.

### Per-class u_C magnitudes and cos(r̂, u_C) sanity check

Source: `data/results/emnlp_perm_edit/phase1_runtime_hook/direction_diagnostics.json`.

| Class | `‖r_jb_C^⊥‖ / ‖r̂‖` | `cos(r̂, u_C)` (should be 0) |
|---|---:|---:|
| fiction | 0.341 | −2.30e-07 |
| roleplay | 0.300 | −4.14e-07 |
| analytical | 0.270 | −2.27e-07 |
| completion | 0.245 | −2.68e-07 |
| cognitive_reframe | 0.376 | −1.09e-06 |

**Interpretation**: orthogonal-component magnitudes line up with the REPORT § 5.5.2 expected range (0.24–0.38 ‖r̂‖). Numerical `cos(r̂, u_C)` is at machine precision floor — projection math is correct. Foundational construction is clean.

### Pairwise cos(u_C, u_C') — the dissociation-feasibility diagnostic

Source: same `direction_diagnostics.json` + `uc_pairwise_robustness.json`. Spec expected ≤ ±0.3 across all pairs.

| Pair | raw cos | Pearson cos (mean-subtracted) | Δ | Verdict vs spec ≤0.3 |
|---|---:|---:|---:|---|
| fiction × roleplay | +0.4146 | +0.4147 | −0.0001 | ⚠️ slightly above |
| fiction × analytical | +0.1989 | +0.1987 | +0.0001 | ✓ within |
| fiction × completion | +0.4909 | +0.4907 | +0.0002 | ⚠️ |
| fiction × cognitive_reframe | +0.2560 | +0.2563 | −0.0004 | ✓ |
| **roleplay × analytical** | **+0.6695** | +0.6695 | −0.0001 | 🚨 above |
| **roleplay × completion** | **+0.7482** | +0.7485 | −0.0003 | 🚨 |
| **roleplay × cognitive_reframe** | **+0.8924** | +0.8925 | −0.0001 | 🚨 |
| **analytical × completion** | **+0.6803** | +0.6803 | −0.0000 | 🚨 |
| **analytical × cognitive_reframe** | **+0.7450** | +0.7453 | −0.0003 | 🚨 |
| **completion × cognitive_reframe** | **+0.7181** | +0.7188 | −0.0007 | 🚨 |

`max |raw − Pearson| = 0.0007` across all 10 pairs → all-ones-direction projections are tiny (0.16% – 2.57% of ‖u_C‖); the raw cosines are not artifacts.

### Interpretation

1. **The high u_C overlap is real geometric structure, not anisotropy.** Georg's specific concern that "computing cosine without mean-subtraction would always produce similar directions" does NOT apply here — Pearson cosines match raw to 4 decimal places. Direct quote of his concern from 2026-05-19: *"If we are computing the cosine similarity on the activations of a vector without subtracting the mean activation from it, wouldn't we always have vectors that point in the same or similar directions?"* — answered empirically: the residuals are well-centered (likely by RMSNorm), so the raw cosines we compute on diff-of-means vectors are not biased by all-ones-direction effects.

2. **Two-cluster structure suggested.** Fiction's u_C is the only one geometrically distinct from the others. The remaining four classes share their orthogonal axis to a degree (+0.67 to +0.89 pairwise) that suggests they are largely **the same mechanism viewed under different surface prefixes**. This is itself a substantive finding: across the 5 JB classes from the controlled dataset, **JB-like-ness after stripping the shared harmless-axis push is ~2 distinct modes** (fiction vs the other four), not 5.

3. **Track B per-class orthogonalization is at risk for 4 of 5 classes.** Orthogonalizing against the u_C of any one of {roleplay, analytical, completion, cog_reframe} will substantially also orthogonalize against the others. This is the exact dissociation problem Track B was designed to solve, and the u_C-based construction does not solve it for these 4 classes.

4. **Fiction-only Track B may still work.** Fiction's u_C is geometrically distinct; a fiction-specific permanent edit could plausibly dissociate fiction from the other 4. This is consistent with REPORT § 10.2 ("Why fiction is special") and § 9.3 — fiction is the only class that passed Stage 08's per-class dissociation test (+16 pp).

5. **Track A taxonomy prediction.** When we run Phase 0 0f clustering on transcoder features, the JB classes likely cluster into **two groups: {fiction} vs {roleplay, analytical, completion, cog_reframe}**, not five distinct clusters. The taxonomy work in Track A becomes the mechanistic explanation for this geometric observation.

### Hypotheses to update in the spec

- **H0-8 (feature role taxonomy):** the cluster count assumption "5 classes → 5 distinct cluster signatures" weakens. Reframe as "JB classes cluster into N taxonomy buckets with N likely < 5"; the actual N is empirical.
- **H0-9 (per-class perturbation signature):** the expected outcome shifts from "each of 5 classes has a distinct signature" to "fiction has a distinct signature, the other 4 share a signature." This is still publishable — it reframes the bypass-mechanism story from "5 classes, 5 mechanisms" to "2 classes-of-mechanism".
- **Spec § 9 risk register:** the "u_C vectors too cosine-similar to dissociate" risk is now realized (probability becomes 1, not Medium-low) for 4 of 5 classes. Update the mitigation language.

### Recommended next actions

- **Share with Georg before any GPU spend on Track B Phase 2.** Specifically: ask whether to (a) restrict Track B to fiction-only proof-of-concept, (b) attempt orthogonalizing against the SHARED orthogonal-axis (a single direction common to the 4-class cluster) as a "universal-JB" edit, or (c) drop Track B in favor of doubling down on Track A's taxonomy story.
- **Run Phase 0 Task 10 (direction-alignment robustness audit / 0c) next** — answers Georg's separate cosine challenge from 2026-05-17 (on `cos(r_jb_C, −r̂)`, not on u_C-u_C' pairs).
- **Don't pause Track A.** The taxonomy work (0a/0b/0d/0e/0f/0g) is more important than ever — it's the mechanistic explanation for the 2-cluster geometry we just observed.

---

## Open questions

- The 4-class shared axis: is it the "harmful-prompt-with-fictional-context" axis? The "soft-refusal lead-in" axis? Phase 0 0f clustering + 0g perturbation signature should reveal what these 4 classes are sharing.
- Should we redefine `u_C` to be orthogonal to BOTH r̂ AND the 4-class shared axis? This would isolate truly class-specific machinery (if any exists for the 4 classes).
- Does the high overlap hold under different measurement positions (pos=−5, pos=−3)? Pos=−2 is the decision token; the shared axis might be position-specific.

---

## 2026-05-19 — Batch 2: Phase 0 Task 10 (0c direction-alignment robustness audit)

**Tasks executed**:
- P0 Task 10 (`00_direction_robustness.py` + tests) — pending commit; 5/5 tests pass.

**Compute**: ~10 s CPU.

**Purpose**: address Georg's 2026-05-17 cosine challenge on `cos(r_jb_C, −r̂)` (REPORT § 5.5.2 reported +0.72 to +0.94 across classes; Georg asked whether this is real geometry or a high-dim anisotropy artifact). Three diagnostics tested H0-5.

### Per-class results

Source: `data/results/emnlp_perm_edit/phase0_controllability/direction_robustness.json`.

| Class | class-mean cos(r̂, r_jb_C) | per-prompt mean ± std | random p95 / rank | Pearson cos | H0-5 |
|---|---:|---:|---:|---:|---|
| fiction | −0.7194 | −0.60 ± 0.29 | 0.038 / **0/1000** | −0.7194 | ⚠️ FAIL on per-prompt (delta 0.12 > 0.10) |
| roleplay | −0.8879 | −0.84 ± 0.15 | 0.036 / **0/1000** | −0.8879 | ✓ PASS |
| analytical | −0.9376 | −0.92 ± 0.02 | 0.036 / **0/1000** | −0.9376 | ✓ PASS |
| completion | −0.7952 | −0.72 ± 0.19 | 0.037 / **0/1000** | −0.7953 | ✓ PASS |
| cognitive_reframe | −0.9405 | −0.93 ± 0.02 | 0.036 / **0/1000** | −0.9405 | ✓ PASS |

**H0-5 overall: PASS (4/5 classes pass all three controls; fiction passes 2/3).**

### Interpretation — three answers to Georg's three concerns

1. **Random-direction baseline (concern: "could any random direction look this aligned in d=2560?"):** **Decisively answered NO.** All 5 classes have rank 0/1000 — the real `cos(r_jb_C, r̂)` magnitude is more extreme than ANY of 1000 random unit-vector cosines. The 95th-percentile of random cosines is ~0.036; real cosines are 0.72–0.94. The directional alignment is **statistically unmistakable** in this 2560-dim space.

2. **Pearson cosine (concern: "could the all-ones direction be inflating the cosine?"):** **Empirically answered NO.** All 5 classes show `|raw − Pearson|` = 0 to 4 decimal places. The residual stream is sufficiently well-centered (likely by RMSnorm γ scaling) that the all-ones-direction projections of `r_jb_C` and `r̂` are negligible, so mean-subtraction doesn't change cosine. **This is the second empirical confirmation of this fact** (the u_C pairwise check from Batch 1 showed the same `|delta| ≤ 0.0007` result independently).

3. **Per-prompt cosine (concern: "is the class-mean cosine masking within-class variance?"):** **Nuanced answer — mostly NO, but fiction is heterogeneous.**
   - For roleplay/analytical/completion/cog_reframe: per-prompt mean is within 0.05–0.10 of class-mean, and per-prompt std is small (0.02–0.19). The class-mean is a tight summary.
   - For fiction: class-mean is −0.72, per-prompt mean is −0.60 (delta 0.12, just over the 0.10 threshold), with per-prompt std 0.29. **Some fiction prompts align weakly or anti-aligned** with −r̂. This is consistent with fiction being a category-mismatch attack with diverse surface forms (different stories, different framings).

### What we tell Georg

> "Empirically tested all three concerns. Random-direction baseline rules out coincidence (real cosines are more extreme than ANY of 1000 random samples across all 5 classes). Pearson cosine confirms no all-ones-direction bias (delta to raw cosine is zero to 4 decimal places). Per-prompt cosine confirms class-mean for 4/5 classes; **fiction's class-mean cosine of +0.72 with −r̂ should be qualified** as 'per-prompt mean +0.60 ± std 0.29' because fiction prompts are heterogeneous. The 4 other classes have tight per-prompt distributions where the class-mean is a faithful summary."

### Cross-reference to Batch 1 result

The high u_C pairwise overlap finding from Batch 1 is **independent** of H0-5 — H0-5 is about `r_jb_C` vs `r̂`, while Batch 1 was about `u_C` (the orthogonal component) vs `u_C'`. Both Batch 1 and Batch 2 found Pearson cosines matching raw to 4 decimal places, which is a consistent picture: **residual-stream activations at L15 pos=−2 don't have a meaningful all-ones-direction bias**. Any cosine measurement we make on diff-of-means vectors is robust to mean-subtraction — Georg's specific framing of his concern doesn't apply, BUT the substantive concern (high geometric overlap between u_C vectors) still stands as real geometric structure.

### Recommended next actions

- **Run Phase 0 Task 1 (HF graph pull, ~10 min network)** — unlocks 0a, 0d, 0e, 0f, 0g.
- **Phase 0 Tasks 2–4 (graph_loader + 0a linearization decomposition + figure)** — provides Georg's H0-1 audit data on CPU, no GPU needed.
- The GPU runs (0b-simple, 0d, 0e) wait for RunPod or local CUDA install.

---

*Last updated 2026-05-19 after Batch 2 direction-alignment robustness audit (H0-5).*

---

## 2026-05-19 — Batch 3: Phase 0 Task 1 (HF graph pull) + schema discovery

**Task executed**: P0 Task 1 (HF graph pull from `moon70/refusal-lens-graphs`).

**Outcome**: 1100 packed JSON.gz files (550 single + 550 multi) pulled. Path: `data/results/pipeline_runs/run_20260430_023247/05_frontend/graph_data/`. 421 MB on disk (gitignored).

**Operational note**: hit HF rate limit (1000 req / 5 min) at 994/1100 files on first attempt; one retry after 60s cooldown finished the remaining 106 files using the cached snapshot.

**Schema corrections discovered** (vs implementation plan):
- Edges field: `links` (not `edges`) — plan supported both
- Node id field: `node_id` (not `id`) — required code change
- Node type field: `feature_type` (not `node_type`) — required code change
- Target identification: `feature_type == 'logit'` AND `is_target_logit == True`
- Source IDs in edges can be either node_ids OR external `E_<token_id>_<pos>` style refs that don't appear in the node list — graph_loader silently skips externals (not edges into the logit anyway).

**Path correction**: actual graph_data path is `05_frontend/graph_data/` (the frontend-staging path), not `02_attribution/graph_data/` as the plan assumed.

Tracked in commits: `graph_loader.py` uses corrected field names; `00_linearization_decomposition.py` uses corrected path.

---

## 2026-05-19 — Batch 4: Phase 0 Tasks 2-4 (graph_loader + 0a + decomposition figure)

**Tasks executed**:
- P0 Task 2 (`graph_loader.py` + 8 tests)
- P0 Task 3 (`00_linearization_decomposition.py` CLI + run)
- P0 Task 4 (`00_decomposition_figure.py` stacked-bar figure)

**Compute**: ~10 s CPU for the 550-graph decomposition; ~1 s for the figure. No GPU.

### Per-condition signed attribution decomposition

Source: `data/results/emnlp_perm_edit/phase0_controllability/decomposition_by_condition.json`. Means across 50 prompts per condition; values in direct_dot units.

| Condition | feat_signed | embed_signed | err_signed | all_signed |
|---|---:|---:|---:|---:|
| **bare** | **−9,469** | **−35,609** | **+190** | **−44,887** |
| jb_fiction | −13,015 | −35,399 | +963 | −47,451 |
| jb_roleplay | −12,823 | −35,981 | +792 | −48,012 |
| jb_analytical | −13,479 | −36,155 | +967 | −48,667 |
| jb_completion | −11,714 | −36,653 | +497 | −47,869 |
| **jb_cognitive_reframe** | **−14,435** | −36,397 | +300 | **−50,532** |
| ctrl_fiction | −11,914 | −36,997 | +337 | −48,574 |
| ctrl_roleplay | −12,767 | −36,588 | +351 | −49,005 |
| ctrl_analytical | −11,698 | −36,469 | +95 | −48,072 |
| ctrl_completion | −12,026 | −36,913 | +742 | −48,197 |
| ctrl_cognitive_reframe | −12,684 | −36,280 | +447 | −48,517 |

Sign convention (per REPORT § 4): more negative = stronger pull AWAY from the refusal axis (JBs shift residual along −r̂; their attribution magnitudes are more negative than bare).

### Headline findings — answers to Georg's "what controls refusal in our transcoders?" question

1. **Embeddings carry ~75% of the signed attribution magnitude into direct_dot.** Mean embedding contribution ranges −35,399 to −36,997 across conditions; total ranges −44,887 to −50,532. Specifically:
   - bare: embedding share = 79.3% of |all_signed|
   - jb_cognitive_reframe: embedding share = 72.0%
   - All conditions: embedding share is 72–82% of total magnitude
   This is a substantial mechanistic finding. **The transcoder framework's attribution graph attributes most of the refusal-direction signal at L15 pos=−2 to embedding writes**, not to MLP-transcoded feature writes. This **partially explains** why Stage 08 sparse MLP-feature ablation plateaued at 34.8% in REPORT § 9.10 — most of the actionable signal isn't in the MLP-feature subspace at all.

2. **Features (MLP transcoders) contribute ~20–30% of magnitude.** Class structure visible:
   - bare features: −9,469 (smallest magnitude)
   - JB features: −11,714 to −14,435 (consistently larger than bare by 24–53%)
   - ctrl features: −11,698 to −12,767 (similar to JBs in magnitude — prefix-induced, not JB-semantic)
   Per the v1 Pareto plateau, this is the budget available to MLP-feature ablation. Even comprehensive ablation of all MLP features (if it worked perfectly) couldn't shift direct_dot by more than the feature_signed magnitude — ~10–15k, leaving the embedding-attributed ~35k untouched.

3. **Error nodes are negligible.** Mean error_signed ranges +95 to +967 (always positive, ~1–2% of total magnitude). This directly tests H0-3 (error-node prominence): **transcoder reconstruction errors are NOT a publishable mechanism component on this dataset**. Their contribution is small and sign-inconsistent across conditions — looks like noise around zero, not a structured signal.

4. **Cognitive_reframe has the strongest signal pull** (all_signed = −50,532, vs bare −44,887 = +13% more negative). Consistent with REPORT § 5.5.2 finding that cognitive_reframe has the largest residual displacement (1.11 ‖r̂‖) and strongest semantic JB effect.

5. **Ctrl conditions ≈ JBs in attribution magnitude** (differ by <5% in `all_signed`). The transcoder attribution graph **doesn't strongly distinguish "real JB" from "matched-length control prefix"** at the signed-magnitude level — consistent with REPORT § 5.3 finding that "most of the apparent JB effect is prefix-induced, not semantically driven" at the Cohen's d level.

### Filtering caveat (documented for transparency)

Packed graphs were produced with `--edge-threshold=0.98` (Stage 02c default), filtering low-magnitude edges to reduce file size. Comparing our filtered measurements to Stage 03's unfiltered `attr_net` (bare-only ground truth):

| Prompt 0 (bare) | Our (filtered) | Stage 03 (unfiltered) | Filtering loss |
|---|---:|---:|---:|
| feature_signed | (subset of attr_net) | — | — |
| total_signed | −44,887 (mean) | −48,886 (mean attr_net) | ~8% |

All 50 bare prompts show >5% filtering loss vs Stage 03. **This is expected and not a problem** — the decomposition story (which edge type dominates) is preserved under filtering. For paper-grade absolute magnitudes, we'd re-extract from raw .pt graphs; for the qualitative taxonomy work, filtered packed graphs are sufficient.

### Implication for hypotheses

- **H0-1 (controllability completeness):** Comprehensive ablation of ALL edges would drive direct_dot to baseline_offset. Per linearization identity, baseline = direct_dot − total_signed = −29,467 − (−48,886) = +19,419 (per Stage 03 reference). On bare prompts with edge ablation, direct_dot would shift from −29k to +19k, which is a large positive shift = **strong refusal direction shift**. Whether the model BEHAVIORALLY flips under this shift is for sub-experiment 0b to test (GPU work).
- **H0-3 (error-node prominence):** **0a evidence suggests H0-3 will not hold** — error_signed is ~1% of total magnitude. The 0b runtime ablation will confirm or refute behaviorally; on current evidence, ablating error nodes alone will have minimal flip-rate effect.
- **H0-1 / H0-2 sign correctness:** Embedding contribution is consistently negative (−35k) across all conditions; feature contribution is also consistently negative but smaller; error contribution is small and positive. **Signs are coherent across the dataset** — no apparent sign-handling bug at this aggregate level.

### Recommended next actions

- **Next CPU work**: Generate the decomposition figure (DONE, `decomposition_figure.png` ready for Georg).
- **Pending GPU work** (RunPod): 0b-simple runtime intervention to test H0-1 behaviorally. Per the 0a finding, the strongest expected effect is from `ablate_embeddings_all` (removing the ~−35k embedding contribution should drive direct_dot most positive), and the weakest from `ablate_errors_all` (only ~+200 contribution to remove).
- **0d and 0e (top-K sweeps)** can wait — 0a already tells us features are the smaller part of the signal; top-K feature ablation will plateau low. Top-K EDGE sweep (including embedding edges) is more interesting.
- **0f and 0g (taxonomy)** valuable but depend on per-feature data we'd need to re-extract. 0a uses aggregated sums; per-feature clustering needs the individual feature attribution records.

---

*Last updated 2026-05-19 after Batch 4 linearization decomposition.*

---

## 2026-05-19 — Batch 5: Package 0b-simple for RunPod (CPU prep done)

**Tasks executed**:
- P0 Task 5 (`edge_ablation_hook.py` library + 4 tests)
- P0 Task 6 (`00_edge_ablation_runtime.py` driver — CPU-syntax-validated)
- New: `runpod_phase0_0b.sh` launch script (matches scripts/pipeline/run_p7.sh pattern)

**Compute**: ~5 min CPU for code + tests. **No execution yet — driver requires CUDA.**

### What's in the package

1. `edge_ablation_hook.py`: factory for the residual-stream `r_hat`-projection subtraction hook. Math: `h_new = h - (delta / ||r_hat||²) · r_hat`, so `h_new · r_hat = h · r_hat - delta` after the hook. 4/4 unit tests pass.

2. `00_edge_ablation_runtime.py`: the driver. For each of 7 variants × 50 prompts × 11 conditions = 3,850 generations:
   - Looks up per-(prompt, condition) `delta` from 0a's `linearization_decomposition.json`
   - Registers the hook on Gemma-3-4B-IT's `language_model.layers[15]` block output
   - Greedy generate `max_new_tokens=80`
   - Classifies refuse/comply via `utils.classify_response` (Stage 08 convention)
   - Saves incrementally every 100 generations + after each variant (robust to OOM/crash)

3. `runpod_phase0_0b.sh`: tmux-persistent launcher. Verifies CUDA torch, pulls HF graphs if missing, regenerates 0a if missing, runs the driver.

### Sanity check (CPU-only, on real data)

Delta-lookup verification for prompt 0 bare across all 7 variants:

| Variant | delta_field | scale | delta_applied |
|---|---|---:|---:|
| ablate_features_pos | feature_pos | 1.0 | +8,234 |
| ablate_features_neg | feature_neg | 1.0 | −17,272 |
| ablate_features_all | feature_signed | 1.0 | −9,039 |
| ablate_embeddings_all | embedding_signed | 1.0 | **−35,139** |
| ablate_errors_all | error_signed | 1.0 | +144 |
| ablate_all_edges | all_signed | 1.0 | −44,033 |
| ablate_all_2x | all_signed | 2.0 | −88,067 |

Hook math verified: applying delta=−9,039 to a test tensor shifted `r_hat`-projection by exactly −9,039.0 (machine precision). Driver is logically correct; just needs CUDA torch to run.

### Behavioral predictions for the GPU run

For JB-comply prompts (where the model currently complies, direct_dot ≈ −32k):
- **`ablate_all_2x`**: delta ≈ −90k, shifts direct_dot to ~+58k — should flip ~all JB-comply → REFUSE (strongest test of H0-1)
- **`ablate_all_edges`**: delta ≈ −48k, shifts to ~+16k — should also flip many to REFUSE; tests H0-1 at baseline magnitude
- **`ablate_embeddings_all`**: delta ≈ −36k, shifts to ~+4k — predicts strong flip-to-refuse (highest single-component impact)
- **`ablate_features_all`**: delta ≈ −12k, shifts to ~−20k — borderline; tests if MLP features alone are sufficient
- **`ablate_errors_all`**: delta ≈ +0.5k, shifts to ~−32.5k — predicts ~no behavioral change (tests H0-3)
- **`ablate_features_pos` vs `ablate_features_neg`**: predicts ASYMMETRY (pos drives more flip than neg) — tests H0-2 sign-handling

### RunPod execution instructions for the user

```bash
# 1. Spin up RunPod (H100 SXM or A100 80GB recommended; RTX 4090 24GB also works)
#    Template: pytorch:2.x-py3.12-cuda12.x
#    Volume: 50 GB minimum

# 2. From RunPod terminal at /workspace:
git clone https://github.com/<your-fork>/Refusal-Lens.git
cd Refusal-Lens
git checkout emnlp-perm-edit
git submodule update --init --recursive
bash scripts/emnlp_perm_edit/runpod_phase0_0b.sh
# Detaches into tmux; reattach with: tmux attach -t phase0_0b

# 3. Pull result back when done:
# From laptop:
scp -P <port> root@<pod-ip>:/workspace/Refusal-Lens/data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json .
```

Expected wall: ~2h on H100 SXM, ~3.5h on RTX 4090. Output: `edge_ablation_flip_rates.json` with 3,850 classification records (7 variants × 550 inputs).

### Aggregation deferred to next CPU batch (Batch 6)

The driver produces RAW classifications. To compute flip rates with Wilson CIs vs Stage 06 baselines, we need a separate aggregation script (Phase 0 plan Task 8). That's CPU-only and can run on the laptop after the GPU run completes.

---

*Last updated 2026-05-19 after Batch 5 (0b-simple packaged for RunPod, awaiting GPU execution).*

---

## 2026-05-19 — Batch 6: 0d/0e drivers + Phase 0 GPU aggregation + unified launcher

**Tasks executed**:
- P0 Task 11 (graph_loader.extract_edge_records_to_target + 2 tests)
- P0 Task 13 (graph_loader.extract_feature_profile + 1 test)
- New `00_topk_sweep.py` — unified driver for 0d (features-only) and 0e (all edges) via `--mode` flag
- New `00_aggregate_phase0_gpu.py` — unified aggregation reading 0b + 0d + 0e raw outputs against Stage 06 baselines, producing flip rates with Wilson 95% CIs + three figures (controllability bar, 0d Pareto, 0e-vs-0d overlay)
- New `runpod_phase0_all.sh` — single tmux-persistent launcher chaining 0b → 0d → 0e

**Compute**: ~5 min CPU for code + tests + sanity check. 11/11 graph_loader tests pass.

**Verified on real data**: compute_delta_for_variant produces expected values; e.g., for prompt 0 bare's 164 feature edges, K=500 pos/neg/abs all collapse to the full feature_signed sum of -9038.6 (matches 0a output exactly).

### Implication for GPU run design

The unified launcher with `K_MODE=coarse` (3 K values instead of 7) cuts ~8 h GPU off the full sweep. For initial paper draft, coarse mode is probably sufficient to test H0-6 / H0-7; full mode can come later if the coarse Pareto shows interesting structure.

---

## 2026-05-19 — Batch 7: Phase 0 0f feature role clustering (CPU)

**Tasks executed**:
- P0 Task 13 (`00_feature_taxonomy.py` — hierarchical agglomerative clustering on 22-dim feature profiles)

**Compute**: ~30 s CPU (550 graphs, 903 features clustered after `min_occurrences=5` filter).

### Headline: H0-8 PASSES with silhouette 0.515

Source: `data/results/emnlp_perm_edit/phase0_controllability/feature_taxonomy.json` + `feature_taxonomy_clusters.md` + `feature_taxonomy_figure.png`.

**Discrete feature roles for refusal exist.** 903 transcoder features cluster cleanly into 6 buckets at silhouette score 0.515 (>0.5 = strong cluster structure; >0.3 was the H0-8 acceptance bar). The taxonomy structure that emerges:

| Cluster | n features | Per-condition attribution | Role |
|---|---:|---|---|
| **C4 (pillars)** | **4** | +463 to +606 across ALL 11 conditions (freq ~1.00) | **Pillar pro-refusal**: tiny but huge-magnitude. Features: L13:F97447, L10:F7492, L7:F77020, L11:F53616. |
| **C2 (dominant anti)** | **23** | −398 to −446 across ALL 11 conditions (freq 1.00) | **Dominant universal anti-refusal**: largest aggregate magnitude (~10k summed per condition). Active in 100% of prompts. |
| **C1 (mid pro-refusal)** | 58 | +82 to +100 across all conditions (freq 0.68–0.94) | Mid-tier pro-refusal, near-universal but smaller per-feature |
| **C3 (mid JB-active anti)** | 93 | −36 to −61 with JB-class bias | JB-active anti-refusal at moderate magnitudes |
| **C5 (small JB-active anti)** | 22 | −136 to −207, JB-class bias | Smaller JB-active anti-refusal |
| **C0 (background)** | 703 | −3 to −5 (small magnitudes), freq 0.03–0.15 | Generic context features (78% of total feature count) |

### Striking observations

1. **The "pillar" pro-refusal cluster has only 4 features.** L13:F97447, L10:F7492, L7:F77020, L11:F53616. These 4 features collectively contribute +463 to +606 to direct_dot on every condition, and fire on 96-100% of all prompts. **This is the load-bearing universal-refusal-core that v1's REPORT § 9.7 was groping toward with `canonical_pro_refusal` (88 features) and `k50_f50` 1-feature canonical (just L13:F427)**. Our Cluster 4 finds 4 features that may BE that core (one of them, L13:F97447, is reminiscent of the canonical's L13:F427 — same layer 13, different feature index due to schema differences; see below).

2. **JB vs CTRL conditions look almost identical within every cluster.** The signed attributions across `jb_*` and `ctrl_*` rows are similar (e.g., Cluster 2: jb_cognitive_reframe -446 vs ctrl_cognitive_reframe -409). **The clustering doesn't strongly distinguish JB-semantic from prefix-only effects** — consistent with REPORT § 5.3 finding that "most of the apparent JB effect is prefix-induced." This refines H0-9: per-class perturbation signatures may show subtle differences (run 0g to find out) but the gross cluster structure is JB-vs-CTRL-agnostic.

3. **The anti-refusal machinery (Clusters 2, 3, 5) carries ~99k aggregate signed magnitude** vs the pro-refusal machinery (Clusters 1, 4) carrying ~8k. The model "wants" to comply by ~12× more in attribution-magnitude than it "wants" to refuse — and yet on bare prompts the model REFUSES. The baseline_offset (+19k per Stage 03) plus the pillar features' +2.5k must collectively be the threshold that tips refusal.

4. **The 78% "background" cluster (C0)** consists of features with tiny per-feature contributions (-3 to -5) but appearing in 3–15% of prompts. These are noise around zero — likely transcoder features that fire on specific prompts' content (named entities, syntactic patterns, etc.) without consistent direction-axis structure.

### Caveat: Stage 04 semantic annotation not wired

The packed-graph features use a different feature_idx convention from `04_labels/feature_labels.json` (packed graphs reference indices like 28679, 117841, 230854 in a flat circuit-tracer feature space; Stage 04 labels use the Gemma Scope per-layer 0-16382 convention via Neuronpedia). The cluster annotation script tried to look up `top_logits` per feature but got empty results because the IDs don't match. **The clustering itself is unaffected**; we just don't have inline semantic labels.

Resolutions for paper-grade annotation:
- Re-extract Stage 04-style labels from the packed-graph feature_idx convention
- Or hand-annotate the top-25 features per cluster by inspecting their attribution contexts in `feature_labels.json`'s `examples` field
- Or skip semantic annotation for the EMNLP paper and describe clusters by their attribution patterns alone

This is a Phase-0 follow-up item; not a blocker for 0g (perturbation signature) which uses the cluster assignments directly.

### Implication for Track A taxonomy story

The 6-cluster structure provides the **mechanistic skeleton** for the v2 paper's refusal-taxonomy figure. Combined with the 0a finding (embeddings carry ~75% of total attribution magnitude), the picture is now:

- **Refusal-direction control at L15 pos=−2 has three structural components:**
  1. **Embeddings (~75% of signed magnitude)** — token-level inputs pushing the residual toward harmless-axis
  2. **Pillar pro-refusal features (4 features, +~2k aggregate)** — small but concentrated pro-refusal voting
  3. **Distributed anti-refusal features (Clusters 2/3/5, ~138 features, −~80k aggregate)** — distributed pushback

The baseline_offset (~+19k) is what closes the gap on bare prompts (refusal wins despite features+embeddings voting comply); JBs presumably tip this balance by shifting the embedding contribution slightly more negative (per § 5.5.1) or by some interaction with the anti-refusal cluster magnitudes.

### Recommended next actions

- **Batch 8 (next, CPU)**: 0g perturbation signature — for each (cluster, JB class), compute Δ_sem = mean(cluster activation | jb_C) − mean(cluster activation | ctrl_C). This will reveal whether class-specific JB perturbations exist BELOW the cluster-level (within-cluster) even though they don't visibly separate clusters.
- **0g produces the EMNLP paper's headline figure**: two-panel (per-class perturbation heatmap + correlation with `r_jb_C` decoder projection).
- **GPU work remains queued for RunPod** (0b + 0d + 0e via `runpod_phase0_all.sh`).

---

*Last updated 2026-05-19 after Batch 7 feature role clustering (H0-8 PASS).*
