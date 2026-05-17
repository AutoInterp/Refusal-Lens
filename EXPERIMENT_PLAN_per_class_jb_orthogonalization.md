# Experiment Plan — Per-Class Jailbreak Direction Orthogonalization (EMNLP 2026 Main Track)

**Status**: drafted 2026-05-17, not yet executed.
**Branch**: `emnlp-perm-edit` — to be created off `l15-refactor` HEAD. **Do not commit EMNLP work to `l15-refactor`** (frozen as the ICML submission reference).
**Headline contribution**: a permanent per-class model edit that surgically eliminates one jailbreak class's success rate while preserving base refusal, benign helpfulness, and other classes' jailbreak susceptibility.
**Supporting contribution**: mechanistic decomposition of the direction-vs-ablation gap (carries v2 paper's Framing A in supporting role).

---

## 0. One-sentence thesis

> By orthogonalizing Gemma-3-4B-IT's `o_proj` and `down_proj` weight matrices against the *class-specific orthogonal component* `r_jb_C^⊥ = r_jb_C − proj_r̂(r_jb_C)` of each jailbreak class's empirical residual-stream displacement, we produce a permanent model edit that (i) drops the target class's jailbreak success rate to ≤10% on the controlled 50×11 dataset, (ii) preserves base refusal at ≥48/50, (iii) leaves cross-class jailbreak rates within ±10 pp of unedited baseline, and (iv) degrades helpfulness by ≤5% on a standard benchmark. The construction succeeds where sparse MLP-feature ablation plateaued at 34.8 % because the edit operates at the basis where refusal lives (residual-stream direction), not the basis where attribution graphs map it (transcoder features).

---

## 1. Background and motivation

The ICML 2026 workshop submission (`PAPER_OUTLINES_v1.md`, code on `l15-refactor`) established:

| Claim | Method | Result |
|---|---|---|
| 1-D additive `r̂` intervention at L15 fully recovers jailbreak-induced compliance | Stage 06 forward hook on `hook_resid_post[15]` | **100 %** (89/89) flip JB-comply → REFUSE |
| Per-class `r_jb_C` (Ball/Wang convention, native magnitude) recovers most of it | Runtime subtract `r_jb_C` from prompts of class C | **93.3 %** (83/89), with analytical 100 % and cognitive_reframe 97 % (§ 5.7 REPORT) |
| Strongest sparse MLP-feature ablation plateaus at ~35 % | Stage 08 per-prompt top-50 ablation | **34.8 %** [Wilson 25.7, 45.2] |
| Class-specific feature subcircuits do not dissociate | Stage 08 `jb_{class}_specific_vs_ctrl` | `jb_fiction_specific_vs_ctrl` recovers **0 %** on fiction, 22 % on roleplay |

The unresolved question — and Georg's stated "strong contribution to get working" for a main-conference paper — is whether the per-class directional intervention can be compiled into a **permanent model edit** that dissociates classes, rather than a runtime hook. This is what the original Stage 08b/08c plan envisioned via CLT decoder vectors but never implemented, and what the v2 paper outline (`PAPER_OUTLINE_v2_emnlp.md`) does not yet specify in implementable detail.

This plan specifies that implementation, using `r_jb_C^⊥` (not CLT decoders, which Georg flagged as basis-mismatched in his 2026-04-26 mentor feedback) and addressing the Gemma-3 post-LayerNorm complication that the naive Arditi recipe doesn't handle.

---

## 2. Direction construction

For each class C ∈ {fiction, roleplay, analytical, completion, cognitive_reframe}:

```python
# Inputs: existing run_20260430_023247 outputs
r_hat        = torch.load("01_direction/unnormalized_r.pt")[15].float()      # shape (2560,)
residuals    = torch.load("02b_stats/residuals_L15_per_cond.pt")             # dict: cond → (n_prompts, 3, 2560)
h_bare       = residuals["bare"][:, 2, :].mean(0)                            # pos=-2, mean across 50 prompts
h_jb_C       = residuals[f"jb_{C}"][:, 2, :].mean(0)

# Full per-class JB direction (Ball/Wang convention; points toward jailbreak)
r_jb_C       = h_jb_C - h_bare

# Class-specific orthogonal component (removes the shared harmless-axis component)
proj_r_hat   = (r_jb_C @ r_hat) / (r_hat @ r_hat)                            # scalar
r_jb_C_perp  = r_jb_C - proj_r_hat * r_hat

# Unit direction for projection operations
u_C          = r_jb_C_perp / r_jb_C_perp.norm()
```

**Diagnostics to record before any intervention:**

| Quantity | Expected range | Sanity check |
|---|---|---|
| `‖r_jb_C^⊥‖ / ‖r̂‖` | 0.24–0.38 (per § 5.5.2 of REPORT) | If <0.1: the orthogonal component is too small to carry causal weight; methodology may be vacuous for this class. |
| `cos(r̂, u_C)` | exactly 0.0 (by construction) | Numerical floor: <1e-6 |
| Pairwise `cos(u_C, u_{C'})` across 5 classes | small (≤±0.3 expected) | Large positive cosine means classes share orthogonal machinery → dissociation will be hard. Large negative cosine means classes' orthogonal axes are anti-aligned → unexpected, investigate. |

The pairwise cosines are the load-bearing diagnostic for whether dissociation is achievable at all. We have not measured these yet — Phase 1 begins with this diagnostic before running any intervention.

---

## 3. Phase 1 — Runtime hook validation

### 3.1 Implementation

The canonical projection hook function (used by all three variants below; the differences are *where* in the model graph it's attached):

```python
def make_orthogonal_projection_hook(u_C: torch.Tensor):
    u_C = u_C.to(dtype=torch.float32, device="cuda")  # do projection in fp32 for stability
    def hook(module, inputs, output):
        h = output  # (batch, seq, d_model), bfloat16
        h_f32 = h.float()
        proj = (h_f32 * u_C).sum(-1, keepdim=True)    # (batch, seq, 1)
        h_new = h_f32 - proj * u_C
        return h_new.to(h.dtype)
    return hook
```

**Layer choice — three variants tested in Phase 1, each with a specific hook target:**

- **Variant 1A — single-layer residual-stream hook at L15**: hook attached to `model.language_model.layers[15]` block output (equivalently `hook_resid_post[15]` in TransformerLens), projecting `u_C` out of the *full residual stream* at L15 once. Matches the Stage 06 + § 5.7 jb_vector_intervention convention. The lightest intervention; the cleanest paper story if it dissociates, but does NOT mirror the Phase 2 weight edit's per-sublayer write-removal structure.

- **Variant 1B — multi-layer sublayer-output hooks at L=15..L=33**: hook attached to `post_attention_layernorm.output` and `post_feedforward_layernorm.output` at each of the 19 layers from 15 onwards. This is the equivalence baseline for Phase 2 because each hook only removes `u_C` from THIS layer's sublayer write (matching what the γ-corrected weight edit does), while the residual stream pass-through is untouched. Use this for Level 1–5 equivalence verification in § 5.

- **Variant 1C — per-layer sweep**: re-runs Variant 1A's residual-stream hook at each layer L ∈ {0, 11, 15, 19, 25, 33} individually (one at a time). Mechanism-diagnostic — identifies which layers matter most for `u_C` removal, supporting the v2 paper's gap-decomposition framing (Framing A).

**Why hook on post-LN sublayer outputs in 1B, not on `o_proj`/`down_proj` outputs directly:** the Phase 2 weight edit produces `post_attn_LN(o_proj(x))` with zero `u_C` component (after the γ-corrected projection). Hooking on the *post-LN* output and projecting out `u_C` (no γ correction needed) produces the same residual update. Hooking on the *pre-LN* `o_proj` output and projecting out `u_C` directly (without γ correction) is *wrong* — it produces a different result from the weight edit. The post-LN hook is both simpler and correct.

The headline result is whichever variant produces the cleanest dissociation. If 1A succeeds, it's the strongest claim (most surgical — single intervention point). If 1A fails but 1B succeeds, the paper's "permanent edit" story uses 1B → Phase 2 equivalence. If neither succeeds, pivot per § 8 risk register.

**Positions:** apply at all positions (matches Stage 06 + Stage 08 `--positions all`). The weight edit in Phase 2 is structurally position-invariant, so the hook must be too for direct equivalence.

### 3.2 Evaluation conditions per class C

For each of 6 hooks (5 per-class + universal `r̂` control + random-direction control) × existing 50×11 controlled dataset, generate with `max_new_tokens=80` greedy, classify response (refuse/comply/unclear) using the same classifier as Stage 08, compare to Stage 06 baseline. The random-direction control samples a unit vector from the L15 residual subspace with seed=42, matched in magnitude to the mean `‖u_C‖` across the 5 classes — this is the negative control for "any random projection would dissociate."

### 3.3 Phase 1 outputs

```
data/results/emnlp_perm_edit/phase1_runtime_hook/
    direction_diagnostics.json          # u_C norms, pairwise cosines, projection magnitudes
    flip_rates_per_hook.json            # (hook_class × eval_class) flip rate matrix + Wilson CIs
    layer_sweep_results.json            # if L sweep is run
    dissociation_matrix.png             # main figure (5 classes × 5 classes + controls)
    PHASE1_SUMMARY.md
```

### 3.4 Phase 1 acceptance bar (gate to Phase 2)

**Primary bar — full reversal:** target class C's JB-comply → REFUSE flip rate ≥ 90 % (i.e., JB success rate drops to ≤ 10 %), AND every other class's flip rate is within ±10 pp of unedited baseline (≤ 0 % expected since no JB hook is applied), AND bare refusal preservation ≥ 48/50.

**Fallback bar — pp dissociation:** dissociation Δ ≥ +30 pp (target class flip rate minus mean-of-other-classes flip rate) on ≥ 3 of 5 classes, with bare refusal preservation ≥ 48/50.

**Failure mode:** if neither bar holds even for the cleanest class, the methodology is not viable — pivot to attention-head-mediated subcircuit (the other option from our brainstorm) or revisit the direction construction (e.g., switch to `r_jb_sem_C^⊥` controlled for prefix). Do not advance to Phase 2 without one of the two bars cleared.

### 3.5 Phase 1 compute estimate

| Run | Wall on RTX 4090 |
|---|---|
| Direction diagnostics (no generation) | <1 min |
| Variant 1A — single-layer L15 hook: 7 hooks × 550 prompts × 80 tokens | ~3.5 h |
| Variant 1B — multi-layer L15..L33 hook (same hooks, more attachment points; same gen cost) | ~3.5 h |
| Variant 1C — per-layer sweep (6 layers × 7 hooks × 550 prompts × 80 tokens) | ~21 h |
| **Phase 1 total (1A + 1B + 1C)** | **~28 h** |
| **Phase 1 minimum viable (1A + 1B only)** | **~7 h** |

---

## 4. Phase 2 — Permanent weight edit

### 4.1 Corrected math (handles Gemma-3 post-LayerNorm)

Gemma-3 architecture has `post_attention_layernorm` between `o_proj` and the residual add, and `post_feedforward_layernorm` between `down_proj` and the residual add. The naive Arditi recipe of `W_new = (I − u_C u_C^T) W` projects `u_C` out of the **pre-LayerNorm** output of the sublayer — but the residual update is the **post-LayerNorm** output, and γ-scaling per dimension re-introduces a component along `u_C`.

The correct derivation for attention sublayer:

Let `y = o_proj.input @ W^T` where `W = o_proj.weight` of shape `(d_model, d_head·n_heads)`. The residual update is:
```
Δh = post_attention_layernorm(y)
   = γ_post_attn ⊙ (y / RMS(y))
   = c · (γ_post_attn ⊙ y)              where c = 1/RMS(y), a scalar per token position
```

For `u_C^T Δh = 0` to hold for all `y`:
```
u_C^T (γ_post_attn ⊙ y) = 0
(γ_post_attn ⊙ u_C)^T y = 0              (element-wise commutativity)
v_attn^T y = 0                            where v_attn = γ_post_attn ⊙ u_C
```

For this to hold for all `o_proj.input`, we need `v_attn^T W^T = 0`. Apply the left-projection to `W`:
```
v̂_attn = v_attn / ‖v_attn‖
W_new = (I − v̂_attn v̂_attn^T) W
```

Verification: `v_attn^T W_new = v_attn^T (I − v̂_attn v̂_attn^T) W = (v_attn^T − ‖v_attn‖ v̂_attn^T) W = 0` ✓.

**Same logic for `down_proj`** with `v_ff = γ_post_ff ⊙ u_C`:
```
v̂_ff[L] = (γ_post_ff[L] ⊙ u_C) / ‖γ_post_ff[L] ⊙ u_C‖
down_proj.weight[L]_new = (I − v̂_ff[L] v̂_ff[L]^T) · down_proj.weight[L]
```

**Per-layer `γ` correction:** `γ_post_attn[L]` and `γ_post_ff[L]` are *different parameters per layer*. The projectors `v̂_attn[L]` and `v̂_ff[L]` must be recomputed for each of Gemma-3's 34 layers. They are not interchangeable.

**Gemma-3 RMSNorm parameterization caveat:** Gemma's `RMSNorm.forward` computes `output * (1.0 + self.weight)` (not `output * self.weight`). The effective `γ` to use in the derivation is `(1 + post_attention_layernorm.weight)`, not `post_attention_layernorm.weight` directly. Easy to get wrong.

### 4.2 Weight matrices in scope

| Matrix | Phase | Per-layer | Justification |
|---|---|---|---|
| `model.language_model.layers[L].self_attn.o_proj.weight` | 2a | ✓ (34 layers × 5 classes) | Attention output write to residual; passes through `post_attention_layernorm`. |
| `model.language_model.layers[L].mlp.down_proj.weight` | 2a | ✓ (34 layers × 5 classes) | MLP output write to residual; passes through `post_feedforward_layernorm`. |
| `model.embed_tokens.weight` (= `lm_head.weight`, tied) | 2b (extension) | once per class | Token embedding write to residual at input; no γ scaling (no LayerNorm immediately after embed in Gemma-3, just × √d_model). **Deferred to 2b** because embed/lm_head tying causes the edit to also modify unembedding, which may degrade output token distribution. Adding only if Phase 2a alone fails the equivalence bar. |

Excluded: input layer norms, query/key/value projections (these don't write to residual; they read), MLP up_proj/gate_proj (these write to MLP intermediate, not residual).

### 4.3 Phase 2a implementation outline

```python
def edit_model_per_class(model, u_C: torch.Tensor) -> dict:
    """Returns a dict of {param_name: (old_weight, new_weight)} for rollback."""
    rollback = {}
    for L in range(34):
        layer = model.language_model.layers[L]

        # o_proj edit using γ_post_attn[L]
        gamma_attn = 1.0 + layer.post_attention_layernorm.weight.float()
        v_attn = gamma_attn * u_C
        v_hat_attn = v_attn / v_attn.norm()
        W_o = layer.self_attn.o_proj.weight
        rollback[f"layers.{L}.self_attn.o_proj.weight"] = W_o.clone()
        proj_attn = torch.eye(W_o.shape[0], device=W_o.device, dtype=torch.float32) - torch.outer(v_hat_attn, v_hat_attn)
        layer.self_attn.o_proj.weight.copy_((proj_attn.to(W_o.dtype) @ W_o))

        # down_proj edit using γ_post_ff[L]
        gamma_ff = 1.0 + layer.post_feedforward_layernorm.weight.float()
        v_ff = gamma_ff * u_C
        v_hat_ff = v_ff / v_ff.norm()
        W_d = layer.mlp.down_proj.weight
        rollback[f"layers.{L}.mlp.down_proj.weight"] = W_d.clone()
        proj_ff = torch.eye(W_d.shape[0], device=W_d.device, dtype=torch.float32) - torch.outer(v_hat_ff, v_hat_ff)
        layer.mlp.down_proj.weight.copy_((proj_ff.to(W_d.dtype) @ W_d))

    return rollback
```

(Sketch only; production code will live in `scripts/pipeline/08b_direction_edit.py` and have proper dtype handling, dry-run mode, checkpoint saving of the edited model.)

### 4.4 Phase 2 outputs

```
data/results/emnlp_perm_edit/phase2_weight_edit/
    edited_models/                       # 5 per-class edited checkpoints (or LoRA-style delta files)
        gemma3_4b_orthogonalized_fiction/
        gemma3_4b_orthogonalized_roleplay/
        ...
    flip_rates_per_class.json
    equivalence_verification.json        # see § 5 below
    PHASE2_SUMMARY.md
```

### 4.5 Phase 2 acceptance bar

Same primary/fallback bars as Phase 1 — but now applied to the **weight-edited model running without any hook**. Additionally, the equivalence verification protocol in § 5 must pass.

### 4.6 Phase 2 compute estimate

| Run | Wall on RTX 4090 |
|---|---|
| Apply weight edit (5 classes × 34 layers × 2 matrices) | <5 min total (matrix ops only) |
| Re-evaluate edited models on 50×11 (5 classes × 550 prompts × 80 tokens) | ~2.5 h |
| Equivalence verification (capture residuals on 50×11 for both hook + weight) | ~1 h |
| **Phase 2 total** | **~3.5 h** |

---

## 5. Equivalence verification protocol (load-bearing for paper claim)

The paper's "permanent edit" claim depends on demonstrating that the Phase 2 weight-edited model behaves equivalently to the Phase 1 **Variant 1B** runtime-hooked model (multi-layer sublayer-output hooks at L=15..L=33). Variant 1A (single-layer L15 residual-stream hook) is *not* the equivalence baseline — it's structurally different from the weight edit. We define five levels of equivalence, in decreasing order of mechanistic rigor:

### Level 1 — Residual-stream cosine equivalence

For each of 50 prompts × 11 conditions × 34 layers × 3 measurement positions [−5, −3, −2] = ~56,000 measurement points, compute:

```
cos_L_pos = cos( h_runtime_hook[L, pos], h_weight_edit[L, pos] )
```

**Bar:**
- Mean across all 56k points: **≥ 0.99**
- Minimum across all 56k points: **≥ 0.95**
- Reported per-layer mean cosine plotted as a line chart (34 points) — should be flat near 1.0 across layers.

### Level 2 — Projection-onto-`u_C` verification

For each (input, layer), compute the residual-stream projection magnitude onto `u_C`:

```
proj_magnitude = |h[L, pos=-2] · u_C| / ‖h[L, pos=-2]‖
```

Both runtime-hook and weight-edit variants should have this near zero at all layers L ≥ initial_hook_layer.

**Bar:**
- Both variants: `proj_magnitude ≤ 0.01` (i.e., `u_C` component is < 1 % of total residual magnitude) at L ≥ 15.
- Variants agree within 5× of each other on this metric (i.e., if hook gives 0.001 and weight gives 0.004, that's fine; if hook gives 0.001 and weight gives 0.05, the γ correction is incomplete).

### Level 3 — Output token equivalence

Generate `max_new_tokens=80` greedy under both variants for the same 550 inputs. Compute token-level agreement rate:

```
token_agreement = mean over (input × position) of [tok_hook == tok_weight]
```

**Bar:** ≥ 95 % token-level agreement, averaged across 550 inputs. Mismatches concentrate at semantically equivalent rephrasings (acceptable) rather than refusal/comply boundary flips (would invalidate equivalence).

### Level 4 — Classification equivalence

Classify each generation as REFUSE/COMPLY/UNCLEAR using the existing Stage 08 classifier. Compute classification agreement rate:

```
classification_agreement = mean over inputs of [class_hook == class_weight]
```

**Bar:** ≥ 98 % classification agreement, with any disagreements documented per-prompt.

### Level 5 — Aggregate dissociation matrix equivalence

The full dissociation matrix (5 hook classes × 5 eval classes + bare + ctrl_avg = 7 × 5 = 35 cells) under both variants. Each cell is a flip rate with Wilson 95 % CI.

**Bar:** Every cell's flip rate matches within **± 2 pp absolute** between variants. Wilson CIs overlap on every cell.

### Reporting in the paper

The headline equivalence sentence:

> Across 50 prompts × 11 conditions × 34 layers × 3 positions, the weight-edited model produces residual streams cosine-similar to the runtime-hooked model with mean **≥ 0.99** (minimum **≥ 0.95**); token-level output agreement is **≥ 95 %**; classification agreement is **≥ 98 %**; and the full per-class dissociation matrix matches within **± 2 pp** on every cell with overlapping Wilson 95 % CIs.

If this passes, we have published "a true permanent weight edit." If only Levels 4–5 pass but Levels 1–3 fail, we report "behavioral equivalence (output-level)" honestly and weaken the mechanism claim. If Level 5 fails, the weight edit is not a substitute for the hook; we fall back to framing as "deployable inference-time circuit modification."

---

## 6. Phase 3 — Generalization and helpfulness

### 6.1 Out-of-distribution jailbreak set

**Dataset:** HarmBench's 40 standard behaviors × 5 JB templates (the same templates as our controlled dataset, applied to new harmful requests). If HarmBench setup is friction-heavy, fall back to AdvBench's untemplated harmful requests as a sanity check (tests whether the edit generalizes beyond our specific templates).

**Bar per class C:** on the OOD JB set restricted to class C templates, the edit produces flip-rate ≥ 80 % (allowing 10 pp slippage from in-distribution due to domain shift).

**Bar cross-class:** within ±15 pp of unedited baseline (allowing some additional slippage on OOD vs ID).

**Compute:** ~3 h per class on RTX 4090.

### 6.2 Helpfulness benchmark

**Dataset:** 100 benign single-turn prompts from MT-Bench's first turn (categories: writing, roleplay, reasoning, math, coding, extraction, STEM, humanities). Generate from unedited and per-class-edited models.

**Metric:** LM judge (Claude Sonnet 4.6, `claude-sonnet-4-6` — or GPT-4o as cross-validator) on a 1–10 quality scale comparing edited vs unedited outputs per prompt, plus output length distribution check.

**Bar:** mean quality score within **≤ 0.5** of unedited (on 1–10 scale). No more than 5 % of prompts get a "significantly worse" verdict from the judge.

**Compute:** ~1 h per class on RTX 4090 + ~30 min LM judge API calls per class.

### 6.3 Phase 3 outputs

```
data/results/emnlp_perm_edit/phase3_generalization/
    harmbench_ood_results.json
    helpfulness_results.json
    PHASE3_SUMMARY.md
```

---

## 7. Branch strategy and timeline

### 7.1 Branch

Create `emnlp-perm-edit` off `l15-refactor` HEAD:

```bash
git checkout l15-refactor
git pull
git checkout -b emnlp-perm-edit
git push -u origin emnlp-perm-edit
```

**Do not commit EMNLP work to `l15-refactor`.** That branch is the ICML submission's frozen reference. The new branch can freely add `scripts/pipeline/08b_direction_edit.py`, `scripts/pipeline/08c_equivalence_verify.py`, and dedicated test files.

### 7.2 Timeline (assuming mid-June EMNLP deadline, 4 weeks from 2026-05-17)

| Week | Goal | Deliverable |
|---|---|---|
| **Week 1** (5/18–5/24) | Phase 1: direction diagnostics + L15 hook + layer sweep + control conditions | `phase1_runtime_hook/` populated; dissociation matrix figure; go/no-go decision on Phase 2 |
| **Week 2** (5/25–5/31) | Phase 2a: implement γ-corrected weight edit on o_proj + down_proj; equivalence verification protocol | Edited model checkpoints; equivalence verification report |
| **Week 3** (6/1–6/7) | Phase 3: OOD JB + helpfulness benchmarks; (parallel) attention-head attribution support story for Framing A | Generalization results; helpfulness results; supporting mechanism figure |
| **Week 4** (6/8–6/14) | Paper drafting, figure polish, supplementary materials | EMNLP submission draft |

If Phase 1 fails the gate (5/24): pivot to attention-head-mediated subcircuit methodology (see brainstorming session for the other option). Phase 1 gate decision is the major fork.

If Phase 2 equivalence fails (5/31): fall back to "deployable inference-time circuit modification" framing — still publishable, less novel for EMNLP main.

### 7.3 Compute budget

| Phase | Wall | Cumulative |
|---|---|---|
| Phase 1 minimum (Variants 1A + 1B only, no per-layer sweep) | ~7 h | 7 h |
| Phase 1 full (Variants 1A + 1B + 1C) | ~28 h | 28 h |
| Phase 2 (weight edit + re-eval + equivalence check) | ~3.5 h | 10.5–31.5 h |
| Phase 3 (OOD JB + helpfulness, 5 classes) | ~20 h | 30.5–51.5 h |

All on local RTX 4090. No RunPod / H100 required. Variant 1C (per-layer sweep) is the swing-cost item — it's mechanism-supporting data for Framing A and can be deferred to Week 3 if 1A or 1B already passes the gate. Phase 2 is structurally fast (matrix ops + re-evaluation); the only compute risk is if equivalence verification reveals problems that require iteration on the γ correction.

---

## 8. Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| Phase 1 dissociation fails primary AND fallback bars | Medium-low | Phase 1 is cheap (~25 h). If it fails, the failure mode itself is informative — pivot to attention-head-mediated subcircuit and re-use the existing per-class direction infrastructure. |
| Phase 2 γ-corrected edit has residual `u_C` component | Medium | Level 2 of equivalence verification catches this directly. If proj_magnitude > 0.01 ‖h‖, debug per-layer (likely Gemma's `1 + weight` parameterization handled wrong, or multi-layer interaction not anticipated). |
| Embed/lm_head tying breaks output quality when `W_E` is included | Medium-high | `W_E` deferred to Phase 2b. Default Phase 2a excludes it. Re-test in 2b only if 2a alone fails Level 5 equivalence. |
| Per-class `u_C` vectors are too cosine-similar to dissociate | Medium-low | The direction diagnostic in § 2 measures this BEFORE any intervention. If pairwise `cos(u_C, u_{C'})` > 0.5, dissociation will be hard; flag in Phase 1 plan adjustment. |
| Cross-class promiscuity persists despite the orthogonal-component design | Medium | Our prior data (§ 9.8.3 of REPORT) shows the **activation-selectivity ≠ causal-selectivity** problem at the MLP-feature level. We don't yet have evidence whether `r_jb_C^⊥` resolves it. This is the central empirical question of Phase 1; both outcomes are publishable (positive: surgical edit works; negative: even direction-level construction inherits promiscuity, refines the mechanistic claim). |
| OOD JB set requires unanticipated setup | Low | HarmBench has a standard `harmbench-evaluator` interface; if integration is friction-heavy, AdvBench is plug-and-play. |
| LM judge for helpfulness is biased / unreliable | Low | Run two independent judges (Claude Sonnet 4.6 + GPT-4o); report inter-judge agreement; for headline number use the more conservative judge. |
| Timeline slips past EMNLP deadline | Medium | Phases 1 + 2 alone constitute a publishable result if 3 slips. EMNLP submission can omit helpfulness benchmark with a "limitations" paragraph and still tell a complete in-distribution story. NeurIPS December cycle is the backup. |

---

## 9. Open questions for Georg

1. **`W_E` inclusion** — does Georg agree with deferring `W_E` to Phase 2b, given embed/lm_head tying in Gemma-3? Or should we accept the unembedding side-effect from the start as part of a more complete edit?

2. **Multi-layer `u_C` extraction** — the design extracts `u_C` once at L15 and applies it (with per-layer γ correction) at every layer's `o_proj` and `down_proj`. An alternative is to extract `u_C[L]` per-layer (different direction at each layer's residual). Per Stage 03 layer story, the per-class machinery probably has different orientation at L0–L11 (anti-refusal accumulator) vs L33 (pro-refusal flip). Is single-direction enough, or should Phase 1 also sweep this?

3. **Phase 1 fallback to attention-head-mediated subcircuit** — if r_jb_C^⊥ fails the bar, the brainstorm identified attention-head attribution + targeted W_O edit as the next-best option. Worth implementing in parallel as a backup, or wait for Phase 1 gate?

4. **EMNLP timeline vs NeurIPS December cycle** — if Phase 3 helpfulness benchmarks slip, submit to EMNLP with a limitations note, or move target to NeurIPS for a more complete story?

5. **Cross-model replication (Qwen3, Gemma-2-9B)** — listed in v2 paper outline §4.6/§4.7 as cross-model support. For this permanent-edit specifically: should Phase 1 replicate on Qwen3-4B once Ruqiya's pipeline lands, or is one-model evidence sufficient for the EMNLP submission?

---

## 10. Where things live (paths and conventions)

| Artifact | Path |
|---|---|
| Existing per-class direction infrastructure | `scripts/analysis/jb_vector_intervention_per_class.py`, `02b_stats/residuals_L15_per_cond.pt`, `01_direction/unnormalized_r.pt` |
| Stage 06 hook helper (reused) | `scripts/pipeline/utils.py::make_intervention_hook` |
| Stage 08 classifier (reused for Phase 1/2/3) | `scripts/pipeline/utils.py::classify_response`, `is_coherent` |
| New Phase 1 entrypoint | `scripts/emnlp_perm_edit/01_runtime_hook_dissociation.py` |
| New Phase 2 entrypoint | `scripts/emnlp_perm_edit/02_weight_edit.py` |
| New equivalence verification | `scripts/emnlp_perm_edit/03_equivalence_verify.py` |
| New Phase 3 entrypoints | `scripts/emnlp_perm_edit/04_harmbench_ood.py`, `05_helpfulness_judge.py` |
| Phase outputs | `data/results/emnlp_perm_edit/phase{1,2,3}_*/` |

---

*Drafted 2026-05-17 from brainstorming session with Mahmoud. Phase 1 implementation planned to begin Week 1 (5/18–5/24) once the design doc is approved.*
