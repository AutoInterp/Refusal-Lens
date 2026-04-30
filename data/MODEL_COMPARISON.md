# Refusal Mechanisms: Gemma-3-4B-IT vs Qwen3-4B

Side-by-side comparison of refusal-circuit findings across the two models.

- **Gemma data:** populated from `data/tejas_experiments/results_v2/`.
- **Qwen data:** populated from `data/qwen_experiments/results_v2/` (this run).

| | Gemma-3-4B-IT | Qwen3-4B |
|---|---|---|
| Model ID | `google/gemma-3-4b-it` | `Qwen/Qwen3-4B` |
| Transcoders | `mwhanna/gemma-scope-2-4b-it` | `mwhanna/qwen3-4b-transcoders` |
| Attention SAEs | `mwhanna/gemma-scope-attn-saes-16k` | **not published** |
| Num transformer blocks | 34 | 36 |
| Hidden size | 2560 | 2560 |
| Architecture wrapper | Multimodal (`text_config`) | Flat causal LM |
| Chat template ends with | `<start_of_turn>model\n` | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` |

> Qwen's chat template always emits an empty `<think></think>` block even with
> `enable_thinking=False`. This shifts the trailing-token semantics: position -2
> on Qwen is `\n` after `</think>`, not the `model` token as on Gemma.

---

## 1. Refusal Direction (sanity check)

Difference-in-means at the optimal position × layer, computed on 64 harmful +
64 harmless prompts with float64 accumulation and left padding.

| Metric | Gemma-3-4B-IT | Qwen3-4B |
|---|---|---|
| Best position | **-2** (the `model` token) | **-1** (`\n` after `</think>`) |
| Best layer | **32** | **34** |
| Best layer / depth | 32/34 = 0.94 | 34/36 = 0.94 |
| Causal layer | **15** | **18** |
| Causal layer / depth | 15/34 = 0.44 | 18/36 = 0.50 |

> Best-layer / depth ratio is preserved (~0.94). Absolute index and template-
> dependent position differ. Re-tuning is mandatory; do NOT assume Gemma's
> indices on Qwen.

---

## 2. Causal Intervention — Arditi Method

Add unnormalized r at all positions, every step (script 16). Tested at L15
(Gemma's causal layer) + L18 + best-layer.

### Qwen3-4B per-layer results

| Layer | \|r\| | Control benign refuse | JB flip rate | JB coherent | Verdict |
|---|---|---|---|---|---|
| L15 | 7.7 | 0/10 ✓ | 5/48 (10%) | 5/5 | weak — wrong layer |
| **L18** | **15.1** | **0/10 ✓** | **42/48 (88%)** | **42/42 (100%)** | **causal layer** |
| L34 | 260.1 | 5/10 ✗ | 44/48 (92%) | 31/42 (74%) | over-strong, breaks model |

### Per-class flip rate at causal layer

| JB class | Gemma comply | Gemma flipped | Qwen comply | Qwen flipped |
|---|---|---|---|---|
| roleplay | 9/50 | 9/9 (100%) | 11/19 | 7/11 (64%) |
| fiction | 20/50 | 20/20 (100%) | 11/19 | 11/11 (100%) |
| analytical | 27/50 | 27/27 (100%) | 14/19 | 12/14 (86%) |
| completion | 0/50 | – | 5/19 | 5/5 (100%) |
| cognitive_reframe | 34/50 | 34/34 (100%) | 7/19 | 7/7 (100%) |
| **TOTAL** | **90/250** | **90/90 (100%)** | **48/95** | **42/48 (88%)** |

> Qwen reaches 88% (vs Gemma's 100%), with one class (roleplay) at 64%. The
> 100%-coherent rate matches Gemma's qualitative result. Confirms that Qwen's
> refusal is causally mediated by a single mid-network direction.

---

## 3. Mechanism Comparison (script 11)

Refusal-direction projection at the best layer for bare prompts vs three
jailbreak prefixes. Tests whether jailbreaks dampen the projection uniformly
or by class-specific amounts.

### Qwen3-4B

| Class | bare → jailbreak | reduction |
|---|---|---|
| LOCK (roleplay) | +17,663 → +14,890 | **−16%** (mild dampening) |
| HACK (analytical) | +24,250 → +2,531 | **−90%** (near-total bypass) |
| PHISH | +24,954 → +4,026 | **−84%** (near-total bypass) |
| FICTION | +24,955 → +130 | **−99.5%** (sign-flip on some prompts) |

> Qualitative pattern (analytical / PHISH bypass strongest; roleplay weakest)
> reproduces on Qwen. Direct numerical comparison with Gemma requires
> re-running the same script on Gemma in this format; the Tejas mechanism
> table (Section 5) uses different units (attribution net) and is not
> directly comparable.

---

## 4. L<causal> projections: where do JB prompts sit?

Projection of activations onto the (normalized) refusal direction at the
causal layer. Negative = harmless side, positive = harmful side.

### Gemma (L15)

| Condition | Mean projection | n |
|---|---|---|
| Bare (refused) | **−29,466** | 50 |
| ctrl_roleplay | −31,305 | 50 |
| ctrl_fiction | −30,158 | 50 |
| ctrl_analytical | −30,870 | 50 |
| ctrl_completion | −30,936 | 50 |
| ctrl_cognitive_reframe | −31,351 | 50 |
| jb_roleplay (complied) | **−32,001** | 9 |
| jb_fiction (complied) | **−30,753** | 20 |
| jb_analytical (complied) | **−32,164** | 27 |
| jb_cognitive_reframe (complied) | **−33,013** | 34 |

> Gemma finding: jailbroken prompts that successfully comply sit *deeper* in
> the harmless range than even the matched controls.

### Qwen (L18)

_Not yet computed in this format. Section 3 above shows the projection at the
best layer (L34) for bare-vs-jailbreak; the matched-control comparison
requires Qwen-specific token-matched prefixes (Gemma's `controlled_dataset.json`
prefix pairs are token-matched for Gemma's tokenizer and do not match on
Qwen's vocabulary)._

---

## 5. Per-position direction structure (Q5)

Cosine similarity between refusal directions extracted at adjacent positions.
On Gemma, adjacent positions at the causal layer are strongly anti-correlated.

### Gemma (causal layer)

| | cos |
|---|---|
| cos(-2, -3) | **≈ −0.78** |

### Qwen3-4B at L18 (causal layer)

| | -5 | -4 | -3 | -2 | -1 |
|---|---|---|---|---|---|
| **-5** | — | +0.25 | **+0.80** | +0.33 | **+0.66** |
| **-4** | | — | +0.20 | +0.37 | +0.22 |
| **-3** | | | — | +0.35 | **+0.74** |
| **-2** | | | | — | +0.36 |
| **-1** | | | | | — |

### Qwen3-4B at L34 (best separation layer)

All pairs in [+0.04, +0.34]. Directions across positions are **near-independent**
at this depth.

> **Key divergence:** the per-position anti-correlation observed on Gemma is
> NOT reproduced on Qwen. All cosine values across positions are non-negative
> on Qwen at every layer tested. Identifies the anti-correlation pattern as a
> Gemma-specific artifact rather than a generalizable property.

> A secondary structural observation on Qwen L18: positions -5, -3, -1 form
> a tightly correlated cluster (cos > 0.66), while position -4 is decoupled
> (cos ≤ 0.37 with all others). This is template-driven — `<think>` is a
> special token whose direction diverges from the surrounding newlines.

---

## 6. Negative Result — Georg's Exact-Magnitude Method

Set the refusal projection to a fixed target value at position -1
(script 17). Tested at scales 0.1 to 1.0.

### Target projections vs intervention magnitudes (Qwen)

| Layer | target_proj | \|r\| | ratio |
|---|---|---|---|
| L15 | +99.4 | 7.7 | 13× |
| L18 | +262.4 | 15.1 | 17× |
| L34 | +61,587 | 260.1 | 237× |

### Scaling sweep at L18 (Qwen)

| scale | benign coherent | benign refuse | JB flip | JB coherent |
|---|---|---|---|---|
| 0.1 | 0/5 | 0/5 | 1/5 | 0/5 |
| 0.2 | 0/5 | 0/5 | 0/5 | 0/5 |
| 0.3 | 0/5 | 0/5 | 0/5 | 0/5 |
| 0.5+ | 0/5 | (collapsed) | — | — |

All scales produced token-loop outputs (`ifiedified...`, `_____...`,
`I I I...`) — full mode collapse.

> **Comparison:** on Gemma, Georg's method works at scale=1.0 with no collapse.
> On Qwen, every scale tested collapses. Displacement-based interventions
> (Arditi, add r) port across architectures; exact-magnitude interventions
> (Georg, set proj=target) do not. Architectural divergence in residual-
> stream geometry, despite shared direction-based refusal.

---

## 7. Mechanism: Two MLP Jailbreak Classes (BLOCKED)

Net circuit-tracer attribution (positive sum + negative sum) to the refusal
direction, by topic and jailbreak type. Gemma reveals two mechanistically
distinct patterns; we wanted to know if Qwen does too.

### Gemma

| Topic | Bare (pos / neg / net) | RP (pos / neg / net) | Fiction (pos / neg / net) |
|---|---|---|---|
| Lock | 93 / −89 / **+4** | 59 / −118 / **−60** | 61 / −91 / **−30** |
| Hack | 117 / −63 / **+54** | 44 / −96 / **−52** | 71 / −106 / **−34** |
| Phish | 119 / −58 / **+62** | 60 / −129 / **−69** | 74 / −102 / **−28** |

- **Role-play (dampening):** pro-refusal features drop *more* aggressively
  (e.g., 117→44 on hack). The pro-refusal circuit disengages.
- **Fiction (tug-of-war):** pro-refusal drops *less* (117→71) but anti-refusal
  grows *more* (−63→−106). Both forces engage harder; anti wins.

### Qwen

**Not run.** Requires Stage 02 attribution. MLP transcoders for Qwen are
available (`mwhanna/qwen3-4b-transcoders`, 36 layers × 163,840 features).
Attention SAEs for Qwen are NOT published — only Gemma has them. Tejas's
reference scripts `02_attribution_30pairs.py` etc. are 0-byte placeholders, so
no port reference. Estimated effort: write from scratch against
circuit-tracer docs.

---

## 8. 10-pair attribution (corrected direction, all features) (BLOCKED)

| Metric | Gemma | Qwen |
|---|---|---|
| Bare net mean | +75.5 | _not run (Stage 02 blocked)_ |
| JB net mean | +56.7 | _not run_ |
| Mean diff (JB − bare) | **−18.7** | _not run_ |
| JB lower than matched bare | 9/10 | _not run_ |

---

## 9. Attention vs MLP split (BLOCKED)

| Metric | Gemma | Qwen |
|---|---|---|
| Attribution sum (MLPs only) | ~75 | _not run_ |
| Dot product (full residual) | ~18,322 | _not run_ |
| MLP fraction of refusal signal | **0.4%** | _structurally blocked_ |
| Attention + embedding fraction | **99.6%** | _no Qwen attention SAEs_ |

> Even with MLP-only attribution, this comparison cannot be completed for
> Qwen — the attention component requires SAEs that mwhanna has not published.

---

## Summary: Cross-Model Hypotheses

| Finding (from Gemma) | Holds on Qwen? | Source |
|---|---|---|
| Refusal computable as a single direction | **YES** | Section 1 |
| Best position is the first generation token | **PARTIAL** — different token (newline after `</think>` vs `model`) but same role | Section 1 |
| Best layer near the top of the network | **YES** — both at ~94% of depth | Section 1 |
| Causal layer is mid-network (≠ best layer) | **YES** — L18 (50% depth) on Qwen, L15 (44%) on Gemma | Section 1, 2 |
| Two MLP jailbreak mechanisms (dampen vs tug-of-war) | **UNKNOWN** | Section 7 (blocked) |
| Attention carries ~99%+ of the refusal signal | **UNKNOWN** | Section 9 (blocked) |
| Arditi intervention flips ≥95% of jailbroken prompts | **PARTIAL** — 88% on Qwen vs 100% on Gemma; 100% coherent | Section 2 |
| Per-position directions anti-correlated at causal layer | **NO** — Qwen shows positive correlations only | Section 5 |
| Exact-magnitude (Georg) intervention works | **NO** — mode collapse at every scale on Qwen | Section 6 |

### Coverage summary

- **Stages 01, 03, 06 complete on Qwen.** 4 of 5 mentor questions answerable.
- **Stages 02, 04, 05, 07 blocked.** MLP transcoders are available but
  attention SAEs are not, and the reference attribution scripts are empty
  placeholders.
- **Two new findings vs Gemma:** (a) per-position anti-correlation does not
  reproduce; (b) exact-magnitude intervention causes mode collapse.

---

*Sources:*
- Gemma data: `data/tejas_experiments/results_v2/{sanity_check_v2.json, separation_table.json, v2_attribution_10pairs.json, v2_mechanism_comparison.json, bulletproof/final_summary.json}` and `data/tejas_experiments/README.md`
- Qwen data: `data/qwen_experiments/results_v2/{refusal_direction_v2.pt, separation_table.json, sanity_check_v2.json, v2_mechanism_comparison.json, causal_arditi/full_results.json}`
- Qwen scripts: `data/qwen_experiments/scripts/{01,11,15,16,17}*.py`