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
