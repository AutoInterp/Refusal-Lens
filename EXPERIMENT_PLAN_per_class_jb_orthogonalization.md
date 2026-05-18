# Experiment Plan — Transcoder Controllability Audit + Per-Class Jailbreak Direction Orthogonalization (EMNLP 2026 Main Track)

**Status**: drafted 2026-05-17, updated 2026-05-17 to incorporate Georg's foundational edge-ablation track.
**Branch**: `emnlp-perm-edit` (off `l15-refactor` HEAD). **Do not commit EMNLP work to `l15-refactor`** (frozen as the ICML submission reference).
**Headline contribution (Track B)**: a permanent per-class model edit that surgically eliminates one jailbreak class's success rate while preserving base refusal, benign helpfulness, and other classes' jailbreak susceptibility.
**Foundational contribution (Track A — Georg's foundational ask)**: a comprehensive controllability audit of the transcoder framework, decomposing the 35% sparse-feature plateau into named edge-type components (feature edges, embedding edges, error nodes, signed) and testing whether comprehensive edge ablation recovers the full direction-level control the linearization identity guarantees.
**Both tracks feed the same EMNLP paper**: Track A becomes the mechanistic-decomposition pillar (Framing A in `PAPER_OUTLINE_v2_emnlp.md`); Track B becomes the surgical-edit pillar (Framing B).

---

## 0. One-sentence thesis (both tracks)

> **Track A:** Comprehensive ablation of all edges feeding the refusal direction at L15 pos=−2 — including feature edges, embedding edges, error nodes, and both positive and negative attributions — recovers ≥X% of direction-level control predicted by the linearization identity, partitioning the previously-observed 35% sparse-ablation plateau into named structural components.
>
> **Track B:** Orthogonalizing Gemma-3-4B-IT's `o_proj` and `down_proj` weight matrices against the class-specific orthogonal component `r_jb_C^⊥ = r_jb_C − proj_r̂(r_jb_C)` of each jailbreak class's empirical residual-stream displacement produces a permanent model edit that (i) drops the target class's jailbreak success rate to ≤10% on the controlled 50×11 dataset, (ii) preserves base refusal at ≥48/50, (iii) leaves cross-class jailbreak rates within ±10 pp of unedited baseline, and (iv) degrades helpfulness by ≤5% on a standard benchmark.

The two tracks are scientifically complementary: Track A asks "how much control do we have in principle via the transcoder framework?" — Track B asks "given some controllability, can we make it class-selective and bake it into weights?". They share infrastructure (the existing run_20260430_023247 attribution graphs, residual tensors, and Stage 06 baselines) and can execute in parallel.

---

## 1. Background and motivation

The ICML 2026 workshop submission (`PAPER_OUTLINES_v1.md`, code on `l15-refactor`) established:

| Claim | Method | Result |
|---|---|---|
| 1-D additive `r̂` intervention at L15 fully recovers jailbreak-induced compliance | Stage 06 forward hook on `hook_resid_post[15]` | **100 %** (89/89) flip JB-comply → REFUSE |
| Per-class `r_jb_C` (Ball/Wang convention, native magnitude) recovers most of it | Runtime subtract `r_jb_C` from prompts of class C | **93.3 %** (83/89), with analytical 100 % and cognitive_reframe 97 % (§ 5.7 REPORT) |
| Strongest sparse MLP-feature ablation plateaus at ~35 % | Stage 08 per-prompt top-50 ablation | **34.8 %** [Wilson 25.7, 45.2] |
| Class-specific feature subcircuits do not dissociate | Stage 08 `jb_{class}_specific_vs_ctrl` | `jb_fiction_specific_vs_ctrl` recovers **0 %** on fiction, 22 % on roleplay |
| Linearization identity holds bit-exact under corrected basis | Stage 03 verification (`measurement_hook="hook_resid_post"`) | `Σ edges + baseline_offset = direct_dot` within <0.4 % per prompt |

Two unresolved questions motivate this plan, one per track:

**Track A — Georg's foundational ask (2026-05-17 mentor exchange):** the 35% Pareto plateau on sparse MLP-feature ablation could be (i) the transcoder framework's structural limit (signal genuinely lives outside features in attention/embeddings/error nodes), (ii) a methodology artifact of *node-level* ablation (which destroys feature signal everywhere it propagates) where *edge-level* ablation would do better, or (iii) a sign-handling or attribution bug in our current pipeline. Georg's framing: "in principle, our expectation should be that we have complete control over the refusal direction — if we don't, we have to figure out why." Phase 0 makes this question quantitatively answerable.

**Track B — Georg's earlier ask ("strong contribution for a main-conference paper"):** whether the per-class directional intervention can be compiled into a **permanent model edit** that dissociates classes, rather than a runtime hook. The original Stage 08b/08c plan envisioned this via CLT decoder vectors but never implemented it; the v2 paper outline does not yet specify it in implementable detail. This plan does, using `r_jb_C^⊥` (not CLT decoders, which Georg flagged as basis-mismatched in his 2026-04-26 mentor feedback) and addressing the Gemma-3 post-LayerNorm complication.

---

## 2. Phase 0 — Transcoder Controllability Audit (Track A, Georg's foundational experiment)

This phase is intentionally orthogonal to the per-class direction approach. It tests how much of the refusal-direction projection (`direct_dot = h[L15, pos=-2] · r̂`) we can causally manipulate by intervening on the transcoder framework's accounting — and where the missing control, if any, lives.

### 2.1 Hypotheses

**H0-1 (controllability completeness):** Comprehensively ablating ALL edges feeding `direct_dot` — including feature edges, embedding edges, error_node edges, and both positive and negative attributions — drives `direct_dot` to the empirically-measured `baseline_offset` (per the Stage 03 linearization identity), and to zero or beyond when ablation is scaled by 2×. Equivalently: the transcoder framework gives us complete control over the refusal-direction projection at L15.

**H0-2 (signed attribution correctness):** Negative-attribution sources (features/embeddings/error nodes that push refusal in the *opposite* direction at L15) should, when ablated, push `direct_dot` in the predicted positive direction. If ablating negative-only sources doesn't shift `direct_dot` in the expected direction, sign-handling has a bug to find before any further claim is published.

**H0-3 (error-node prominence):** Transcoder error nodes carry a measurable but bounded fraction of `direct_dot`. Per the current Stage 03 audit, error_node Σ is implicit in `baseline_offset`; an explicit error-node-only ablation should isolate this fraction and quantify it.

**H0-4 (edge ≠ node):** Comprehensive edge ablation recovers strictly more `direct_dot` drive than node-level ablation at matched feature scope, because edge ablation is surgical and node ablation destroys signal globally. Expected outcome: edge ablation closes a measurable fraction of the v1 35%-plateau gap.

**Hypothesis-outcome table:**

| Outcome | Interpretation | EMNLP paper implication |
|---|---|---|
| H0-1 holds → comprehensive ablation drives `direct_dot` to baseline | Transcoder framework is mechanistically complete; 35% plateau is node-vs-edge artifact | Framing A strengthened — we explain the missing 65% as methodology, not structure |
| H0-1 fails → `direct_dot` stays well above baseline after comprehensive ablation | Signal lives in something the linearization doesn't capture (e.g., attention paths not in the transcoder graph at all) | Framing A pivot — paper publishes a structural negative result with localization |
| H0-2 fails → negative ablations don't flip sign | Sign-handling bug somewhere in pipeline | Pause Track B; debug attribution math before any further claim |
| H0-3 → error nodes carry >10% of `direct_dot` | Transcoder reconstruction is imperfect; error nodes are a publishable mechanism component | Add error-node-only ablation result as paper figure |
| H0-4 holds → edge > node by ≥10 pp | Methodology lever, immediately publishable | Reframe v1 35% claim with edge-level number |

### 2.2 Sub-experiment 0a — Offline linearization decomposition (no GPU)

**Goal:** quantify per-edge-type contribution to `direct_dot` for every (prompt, condition) using only the saved attribution graphs. CPU-only arithmetic on the JSON.gz packed graphs (already on HF; pull locally via `scripts/pipeline/fetch_graph_data.py` if not present).

**Method:**

For each of 550 (prompt, condition) instances:
1. Load the packed attribution graph from `02_attribution/graph_data/<prompt_id>__<condition>.json.gz`.
2. Categorize all source nodes by type: `{feature, embedding, error_node}` (circuit-tracer exposes these in its node-type field).
3. Extract each source node's signed attribution to `direct_dot` at the measurement target (L15, pos=−2).
4. Aggregate per (prompt, condition):
   - `Σ_features_pos`, `Σ_features_neg`, `Σ_features_signed`
   - `Σ_embeddings_pos`, `Σ_embeddings_neg`, `Σ_embeddings_signed`
   - `Σ_errors_pos`, `Σ_errors_neg`, `Σ_errors_signed`
   - `Σ_all_signed = Σ_features_signed + Σ_embeddings_signed + Σ_errors_signed`
   - `baseline_offset = direct_dot − Σ_all_signed` (per linearization identity; should match Stage 03's reported number within 0.4 %)
5. Verify the identity holds within Stage 03's tolerance (<1 % reconstruction error on 550/550 inputs).

**Outputs:**
- `data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json` — per-(prompt, condition) full breakdown
- `data/results/emnlp_perm_edit/phase0_controllability/decomposition_by_condition.json` — per-condition aggregates (mean, std, min, max) of each component
- `data/results/emnlp_perm_edit/phase0_controllability/decomposition_figure.png` — stacked-bar chart per condition showing feature / embedding / error_node / baseline contributions to `direct_dot`

**Acceptance:**
- Linearization identity `direct_dot = Σ_all_signed + baseline_offset` reconstructs to <1 % error on 550/550 inputs.
- Per-edge-type contributions reported with per-condition means and stds.
- Negative-attribution sums reported separately from positive sums (this is the diagnostic for H0-2 sign-handling).

**Compute:** ~30 min wall on CPU (file IO bound). No GPU needed.

### 2.3 Sub-experiment 0b — Comprehensive edge-ablation runtime intervention (GPU)

**Goal:** causally test the linearization decomposition. By zeroing the contribution of each edge type at runtime (and combinations including over-ablation), measure how much `direct_dot` can actually be controlled and whether the model's refuse/comply classification follows.

Two implementation paths, run sequentially:

**0b-simple (residual-stream r̂-projection modulation, fast):** A runtime hook at L15 `hook_resid_post` that subtracts a per-(prompt, condition) scalar × `r̂_unit` from the residual, with the scalar chosen to zero out the target edge type's contribution to `direct_dot`. Mathematically: if we want to remove `delta` units from `direct_dot`, the hook computes `h_new = h − (delta / ‖r̂‖²) × r̂`. After the hook, `h_new · r̂ = h · r̂ − delta`.

Variants per (prompt, condition), with `delta` chosen from sub-experiment 0a's pre-computed sums:

| Variant | delta subtracted | Target post-intervention `direct_dot` |
|---|---|---|
| `ablate_features_pos` | `Σ_features_pos` | drops by Σ_features_pos |
| `ablate_features_neg` | `Σ_features_neg` (negative number → pushes opposite way) | rises by |Σ_features_neg| — H0-2 sign check |
| `ablate_features_all` | `Σ_features_signed` | `direct_dot − Σ_features_signed` |
| `ablate_embeddings_all` | `Σ_embeddings_signed` | `direct_dot − Σ_embeddings_signed` |
| `ablate_errors_all` | `Σ_errors_signed` | `direct_dot − Σ_errors_signed` |
| `ablate_all_edges` | `Σ_all_signed` | `baseline_offset` (full edge ablation) |
| `ablate_all_2x` | `2 × Σ_all_signed` | `baseline_offset − Σ_all_signed` (over-ablation) |

For each variant × 50 prompts × 11 conditions = 3,850 generations. Greedy `max_new_tokens=80` (Stage 08 convention). Classify each generation as refuse/comply via `utils.classify_response`. Compute flip rates vs Stage 06 baseline.

This approach operates on the residual stream directly, NOT through the transcoder graph. Its purpose is to test whether the linearization decomposition is *behaviorally meaningful*: if we take away the predicted amount of `r̂`-projection per edge-type bucket, does the model flip refusal as the linearization identity predicts?

**0b-rigorous (true edge ablation in vendor/circuit-tracer, if 0b-simple results warrant):** Modify `vendor/circuit-tracer` to support edge-level ablation in the linearization framework — replacing each chosen edge's contribution to the target node with its baseline (zero-input transcoder activation) value, propagating through, and regenerating. Per `PAPER_OUTLINE_v2_emnlp.md` §4.1, this is ~3 person-days of patch + tests. Deferred to Week 2 if 0b-simple results indicate it's needed (e.g., if 0b-simple shows surprising behavior that requires "true" edge ablation to disambiguate).

**Outputs (0b):**
- `data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json` — per-variant flip-rate matrix
- `data/results/emnlp_perm_edit/phase0_controllability/controllability_audit_figure.png` — main figure showing `direct_dot` shift achieved per variant vs the variant's predicted shift from 0a
- `data/results/emnlp_perm_edit/phase0_controllability/sign_audit.md` — focused report on H0-2 (negative-ablation sign correctness)

**Acceptance criteria for Phase 0 as a whole:**
- 0a linearization identity verified on 550/550 inputs (<1% reconstruction error).
- 0b H0-1 test: `ablate_all_edges` drives `direct_dot` measurably toward baseline_offset (within ±10% of predicted, measured by re-extracting `direct_dot` from a forward pass with hook active). `ablate_all_2x` drives `direct_dot` past zero on bare prompts (i.e., flips the sign), and the model's REFUSE→COMPLY flip rate on bare-refuse exceeds 90% under this over-ablation.
- 0b H0-2 test: `ablate_features_neg` shifts `direct_dot` in the OPPOSITE direction from `ablate_features_pos`. Sign correctness verified.
- 0b H0-3 test: per-component flip rates reported. If `ablate_errors_all` alone produces >5% flip rate on bare-refuse or JB-comply, error nodes carry publishable mechanism weight.

If H0-1 or H0-2 fails, Track B can still proceed but Phase 0 results become a paper-grade negative result that reshapes Framing A. If both hold, the v2 paper has a clean "transcoder framework gives complete refusal-direction control" pillar.

### 2.4 Phase 0 outputs (summary)

```
data/results/emnlp_perm_edit/phase0_controllability/
    linearization_decomposition.json        # 0a — per-(prompt, condition) decomposition
    decomposition_by_condition.json         # 0a — per-condition aggregates
    decomposition_figure.png                # 0a — stacked-bar component figure
    edge_ablation_flip_rates.json           # 0b — per-variant flip rates
    controllability_audit_figure.png        # 0b — main figure
    sign_audit.md                           # 0b — H0-2 negative-ablation correctness
    PHASE0_SUMMARY.md                       # human-readable headline
```

### 2.5 Phase 0 compute estimate

| Run | Wall on RTX 5080 16 GB |
|---|---|
| 0a — offline decomposition (CPU only) | ~30 min |
| 0b-simple — 7 variants × 550 prompts × 80 tokens, single L15 hook | ~3.5 h |
| 0b-rigorous (if needed) — vendor/circuit-tracer patch + re-eval | ~3 person-days + ~6 h GPU |
| **Phase 0 minimum (0a + 0b-simple)** | **~4 h** |
| **Phase 0 full (0a + 0b-simple + 0b-rigorous)** | **3+ person-days + ~10 h GPU** |

---

## 3. Per-class direction construction (Track B foundational diagnostics)

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

The pairwise cosines are the load-bearing diagnostic for whether dissociation is achievable at all. Phase 1 begins with this diagnostic before running any intervention.

---

## 4. Phase 1 — Per-class runtime hook validation (Track B)

### 4.1 Implementation

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

- **Variant 1B — multi-layer sublayer-output hooks at L=15..L=33**: hook attached to `post_attention_layernorm.output` and `post_feedforward_layernorm.output` at each of the 19 layers from 15 onwards. This is the equivalence baseline for Phase 2 because each hook only removes `u_C` from THIS layer's sublayer write (matching what the γ-corrected weight edit does), while the residual stream pass-through is untouched. Use this for Level 1–5 equivalence verification in § 6.

- **Variant 1C — per-layer sweep**: re-runs Variant 1A's residual-stream hook at each layer L ∈ {0, 11, 15, 19, 25, 33} individually (one at a time). Mechanism-diagnostic — identifies which layers matter most for `u_C` removal, supporting the v2 paper's gap-decomposition framing (Framing A).

**Why hook on post-LN sublayer outputs in 1B, not on `o_proj`/`down_proj` outputs directly:** the Phase 2 weight edit produces `post_attn_LN(o_proj(x))` with zero `u_C` component (after the γ-corrected projection). Hooking on the *post-LN* output and projecting out `u_C` (no γ correction needed) produces the same residual update. Hooking on the *pre-LN* `o_proj` output and projecting out `u_C` directly (without γ correction) is *wrong* — it produces a different result from the weight edit. The post-LN hook is both simpler and correct.

The headline result is whichever variant produces the cleanest dissociation. If 1A succeeds, it's the strongest claim (most surgical — single intervention point). If 1A fails but 1B succeeds, the paper's "permanent edit" story uses 1B → Phase 2 equivalence. If neither succeeds, pivot per § 9 risk register.

**Positions:** apply at all positions (matches Stage 06 + Stage 08 `--positions all`). The weight edit in Phase 2 is structurally position-invariant, so the hook must be too for direct equivalence.

### 4.2 Evaluation conditions per class C

For each of 6 hooks (5 per-class + universal `r̂` control + random-direction control) × existing 50×11 controlled dataset, generate with `max_new_tokens=80` greedy, classify response (refuse/comply/unclear) using the same classifier as Stage 08, compare to Stage 06 baseline. The random-direction control samples a unit vector from the L15 residual subspace with seed=42, matched in magnitude to the mean `‖u_C‖` across the 5 classes — this is the negative control for "any random projection would dissociate."

### 4.3 Phase 1 outputs

```
data/results/emnlp_perm_edit/phase1_runtime_hook/
    direction_diagnostics.json          # u_C norms, pairwise cosines, projection magnitudes
    flip_rates_per_hook.json            # (hook_class × eval_class) flip rate matrix + Wilson CIs
    layer_sweep_results.json            # if L sweep is run
    dissociation_matrix.png             # main figure (5 classes × 5 classes + controls)
    PHASE1_SUMMARY.md
```

### 4.4 Phase 1 acceptance bar (gate to Phase 2)

**Primary bar — full reversal:** target class C's JB-comply → REFUSE flip rate ≥ 90 % (i.e., JB success rate drops to ≤ 10 %), AND every other class's flip rate is within ±10 pp of unedited baseline (≤ 0 % expected since no JB hook is applied), AND bare refusal preservation ≥ 48/50.

**Fallback bar — pp dissociation:** dissociation Δ ≥ +30 pp (target class flip rate minus mean-of-other-classes flip rate) on ≥ 3 of 5 classes, with bare refusal preservation ≥ 48/50.

**Failure mode:** if neither bar holds even for the cleanest class, the methodology is not viable — pivot to attention-head-mediated subcircuit (the other option from our brainstorm) or revisit the direction construction (e.g., switch to `r_jb_sem_C^⊥` controlled for prefix). Do not advance to Phase 2 without one of the two bars cleared.

### 4.5 Phase 1 compute estimate

| Run | Wall on RTX 5080 16 GB |
|---|---|
| Direction diagnostics (no generation) | <1 min |
| Variant 1A — single-layer L15 hook: 7 hooks × 550 prompts × 80 tokens | ~3.5 h |
| Variant 1B — multi-layer L15..L33 hook (same hooks, more attachment points; same gen cost) | ~3.5 h |
| Variant 1C — per-layer sweep (6 layers × 7 hooks × 550 prompts × 80 tokens) | ~21 h |
| **Phase 1 total (1A + 1B + 1C)** | **~28 h** |
| **Phase 1 minimum viable (1A + 1B only)** | **~7 h** |

---

## 5. Phase 2 — Permanent weight edit (Track B)

### 5.1 Corrected math (handles Gemma-3 post-LayerNorm)

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

### 5.2 Weight matrices in scope

| Matrix | Phase | Per-layer | Justification |
|---|---|---|---|
| `model.language_model.layers[L].self_attn.o_proj.weight` | 2a | ✓ (34 layers × 5 classes) | Attention output write to residual; passes through `post_attention_layernorm`. |
| `model.language_model.layers[L].mlp.down_proj.weight` | 2a | ✓ (34 layers × 5 classes) | MLP output write to residual; passes through `post_feedforward_layernorm`. |
| `model.embed_tokens.weight` (= `lm_head.weight`, tied) | 2b (extension) | once per class | Token embedding write to residual at input; no γ scaling (no LayerNorm immediately after embed in Gemma-3, just × √d_model). **Deferred to 2b** because embed/lm_head tying causes the edit to also modify unembedding, which may degrade output token distribution. Adding only if Phase 2a alone fails the equivalence bar. |

Excluded: input layer norms, query/key/value projections (these don't write to residual; they read), MLP up_proj/gate_proj (these write to MLP intermediate, not residual).

### 5.3 Phase 2a implementation outline

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

(Sketch only; production code will live in `scripts/emnlp_perm_edit/08b_direction_edit.py` and have proper dtype handling, dry-run mode, checkpoint saving of the edited model.)

### 5.4 Phase 2 outputs

```
data/results/emnlp_perm_edit/phase2_weight_edit/
    edited_models/                       # 5 per-class edited checkpoints (or LoRA-style delta files)
        gemma3_4b_orthogonalized_fiction/
        gemma3_4b_orthogonalized_roleplay/
        ...
    flip_rates_per_class.json
    equivalence_verification.json        # see § 6 below
    PHASE2_SUMMARY.md
```

### 5.5 Phase 2 acceptance bar

Same primary/fallback bars as Phase 1 — but now applied to the **weight-edited model running without any hook**. Additionally, the equivalence verification protocol in § 6 must pass.

### 5.6 Phase 2 compute estimate

| Run | Wall on RTX 5080 16 GB |
|---|---|
| Apply weight edit (5 classes × 34 layers × 2 matrices) | <5 min total (matrix ops only) |
| Re-evaluate edited models on 50×11 (5 classes × 550 prompts × 80 tokens) | ~2.5 h |
| Equivalence verification (capture residuals on 50×11 for both hook + weight) | ~1 h |
| **Phase 2 total** | **~3.5 h** |

**VRAM note:** Phase 2 keeps two model copies in memory during equivalence verification (unedited + edited). On 16 GB, this is tight. If needed, capture residuals from each model in separate runs and compare offline rather than holding both in VRAM simultaneously.

---

## 6. Equivalence verification protocol (load-bearing for Track B "permanent edit" claim)

The paper's "permanent edit" claim depends on demonstrating that the Phase 2 weight-edited model behaves equivalently to the Phase 1 **Variant 1B** runtime-hooked model (multi-layer sublayer-output hooks at L=15..L=33). Variant 1A (single-layer L15 residual-stream hook) is *not* the equivalence baseline — it's structurally different from the weight edit. We define five levels of equivalence, in decreasing order of mechanistic rigor:

### 6.1 Level 1 — Residual-stream cosine equivalence

For each of 50 prompts × 11 conditions × 34 layers × 3 measurement positions [−5, −3, −2] = ~56,000 measurement points, compute:

```
cos_L_pos = cos( h_runtime_hook[L, pos], h_weight_edit[L, pos] )
```

**Bar:**
- Mean across all 56k points: **≥ 0.99**
- Minimum across all 56k points: **≥ 0.95**
- Reported per-layer mean cosine plotted as a line chart (34 points) — should be flat near 1.0 across layers.

### 6.2 Level 2 — Projection-onto-`u_C` verification

For each (input, layer), compute the residual-stream projection magnitude onto `u_C`:

```
proj_magnitude = |h[L, pos=-2] · u_C| / ‖h[L, pos=-2]‖
```

Both runtime-hook and weight-edit variants should have this near zero at all layers L ≥ initial_hook_layer.

**Bar:**
- Both variants: `proj_magnitude ≤ 0.01` (i.e., `u_C` component is < 1 % of total residual magnitude) at L ≥ 15.
- Variants agree within 5× of each other on this metric (i.e., if hook gives 0.001 and weight gives 0.004, that's fine; if hook gives 0.001 and weight gives 0.05, the γ correction is incomplete).

### 6.3 Level 3 — Output token equivalence

Generate `max_new_tokens=80` greedy under both variants for the same 550 inputs. Compute token-level agreement rate:

```
token_agreement = mean over (input × position) of [tok_hook == tok_weight]
```

**Bar:** ≥ 95 % token-level agreement, averaged across 550 inputs. Mismatches concentrate at semantically equivalent rephrasings (acceptable) rather than refusal/comply boundary flips (would invalidate equivalence).

### 6.4 Level 4 — Classification equivalence

Classify each generation as REFUSE/COMPLY/UNCLEAR using the existing Stage 08 classifier. Compute classification agreement rate:

```
classification_agreement = mean over inputs of [class_hook == class_weight]
```

**Bar:** ≥ 98 % classification agreement, with any disagreements documented per-prompt.

### 6.5 Level 5 — Aggregate dissociation matrix equivalence

The full dissociation matrix (5 hook classes × 5 eval classes + bare + ctrl_avg = 7 × 5 = 35 cells) under both variants. Each cell is a flip rate with Wilson 95 % CI.

**Bar:** Every cell's flip rate matches within **± 2 pp absolute** between variants. Wilson CIs overlap on every cell.

### 6.6 Reporting in the paper

The headline equivalence sentence:

> Across 50 prompts × 11 conditions × 34 layers × 3 positions, the weight-edited model produces residual streams cosine-similar to the runtime-hooked model with mean **≥ 0.99** (minimum **≥ 0.95**); token-level output agreement is **≥ 95 %**; classification agreement is **≥ 98 %**; and the full per-class dissociation matrix matches within **± 2 pp** on every cell with overlapping Wilson 95 % CIs.

If this passes, we have published "a true permanent weight edit." If only Levels 4–5 pass but Levels 1–3 fail, we report "behavioral equivalence (output-level)" honestly and weaken the mechanism claim. If Level 5 fails, the weight edit is not a substitute for the hook; we fall back to framing as "deployable inference-time circuit modification."

---

## 7. Phase 3 — Generalization and helpfulness (Track B)

### 7.1 Out-of-distribution jailbreak set

**Dataset:** HarmBench's 40 standard behaviors × 5 JB templates (the same templates as our controlled dataset, applied to new harmful requests). If HarmBench setup is friction-heavy, fall back to AdvBench's untemplated harmful requests as a sanity check (tests whether the edit generalizes beyond our specific templates).

**Bar per class C:** on the OOD JB set restricted to class C templates, the edit produces flip-rate ≥ 80 % (allowing 10 pp slippage from in-distribution due to domain shift).

**Bar cross-class:** within ±15 pp of unedited baseline (allowing some additional slippage on OOD vs ID).

**Compute:** ~3 h per class on RTX 5080 16 GB.

### 7.2 Helpfulness benchmark

**Dataset:** 100 benign single-turn prompts from MT-Bench's first turn (categories: writing, roleplay, reasoning, math, coding, extraction, STEM, humanities). Generate from unedited and per-class-edited models.

**Metric:** LM judge (Claude Sonnet 4.6, `claude-sonnet-4-6` — or GPT-4o as cross-validator) on a 1–10 quality scale comparing edited vs unedited outputs per prompt, plus output length distribution check.

**Bar:** mean quality score within **≤ 0.5** of unedited (on 1–10 scale). No more than 5 % of prompts get a "significantly worse" verdict from the judge.

**Compute:** ~1 h per class on RTX 5080 16 GB + ~30 min LM judge API calls per class.

### 7.3 Phase 3 outputs

```
data/results/emnlp_perm_edit/phase3_generalization/
    harmbench_ood_results.json
    helpfulness_results.json
    PHASE3_SUMMARY.md
```

---

## 8. Branch strategy and parallel-track timeline

### 8.1 Branch

`emnlp-perm-edit` (already created off `l15-refactor` HEAD, this spec is committed as the branch's first non-trivial work).

**Do not commit EMNLP work to `l15-refactor`.** That branch is the ICML submission's frozen reference. The new branch can freely add `scripts/emnlp_perm_edit/...` and dedicated test files.

### 8.2 Parallel-track timeline (assuming mid-June EMNLP deadline)

Track A (Phase 0) and Track B (Phases 1–3) execute **in parallel**, with Track A given slight priority so results come sooner for Georg's foundational question. The tracks share no code paths and have no GPU contention if run sequentially on the same machine; if running on the same GPU, alternate Track A's CPU-heavy 0a work with Track B's GPU work to avoid idle time.

| Week | Track A (Phase 0) | Track B (Phases 1–3) | Joint deliverable |
|---|---|---|---|
| **Week 1** (5/18–5/24) | **PRIORITY**: Implement & run 0a (offline linearization decomposition, CPU-bound). Implement 0b-simple driver. Run `ablate_features_pos`, `ablate_features_neg` (H0-2 sign check) and `ablate_all_edges` (H0-1 controllability). | Phase 1 direction diagnostics (CPU, <1 min) + Variant 1A single-layer L15 hook on 5 classes + controls (~4 h). | End of Week 1: Phase 0 acceptance verdict (H0-1 / H0-2 results), Phase 1A dissociation matrix. |
| **Week 2** (5/25–5/31) | Run remaining 0b-simple variants (per-edge-type, 2× over-ablation). If H0-1 fails, decide whether to invest in 0b-rigorous vendor/circuit-tracer patch. | Variant 1B multi-layer hook (~4 h) + acceptance check on 1A + 1B. If primary or fallback bar clears: begin Phase 2 weight edit implementation. | End of Week 2: Both tracks have go/no-go verdicts. Variant 1C deferred to Week 3 if 1A or 1B already passed. |
| **Week 3** (6/1–6/7) | (If needed) 0b-rigorous edge ablation in vendor/circuit-tracer (~3 person-days). Variant 1C per-layer sweep also feeds Track A's mechanism story. | Phase 2 weight-edit implementation + equivalence verification (~4 h GPU + iteration time). | End of Week 3: paper-grade Track A and Track B headline numbers. |
| **Week 4** (6/8–6/14) | Generalization + helpfulness benchmarks for Track B (Phase 3, ~20 h). Track A finalization. | Paper drafting, figure polish, supplementary materials. | EMNLP submission draft. |

If Phase 0's H0-1 fails strongly (i.e., comprehensive edge ablation doesn't drive `direct_dot` toward baseline_offset): pause Track B Phase 2 weight-edit work and prioritize understanding the gap. Track A's negative result becomes a paper in its own right and Track B's per-class permanent-edit claim may need to be reframed.

If Phase 1's primary/fallback bars BOTH fail: pivot Track B to attention-head-mediated subcircuit (the parallel option from the brainstorm, noted for future Georg discussion). Track A continues regardless.

### 8.3 Compute budget

| Phase | Wall (RTX 5080 16 GB) | Cumulative |
|---|---|---|
| Phase 0 minimum (0a + 0b-simple) | ~4 h | 4 h |
| Phase 1 minimum (1A + 1B only) | ~7 h | 11 h |
| Phase 1 full (1A + 1B + 1C) | ~28 h | 32 h |
| Phase 2 (weight edit + re-eval + equivalence) | ~3.5 h | 14.5–35.5 h |
| Phase 3 (OOD JB + helpfulness, 5 classes) | ~20 h | 34.5–55.5 h |

All on local RTX 5080 16 GB. No RunPod / H100 required. **VRAM caveats** on 16 GB:
- Single-stream sequential generation (batch size 1) is the safe pattern for Gemma-3-4B-IT bf16 (weights ~7.5 GB + KV cache).
- Phase 2 equivalence verification benefits from holding both unedited and edited model copies — likely OOMs on 16 GB. Capture residuals in two passes and compare offline.

---

## 9. Risk register

| Risk | Probability | Mitigation |
|---|---|---|
| Phase 0 H0-1 fails (comprehensive edge ablation does NOT drive `direct_dot` to baseline) | Medium | This is itself a paper-grade publishable negative result: localizes where the gap lives (which edge type is missing). Phase 0b-rigorous (true circuit-tracer edge ablation) becomes the disambiguator if 0b-simple's residual-stream proxy doesn't tell the full story. |
| Phase 0 H0-2 fails (negative-attribution ablations don't flip sign as expected) | Low-medium | Critical — signals a sign-handling or basis bug. PAUSE all downstream claims and debug attribution math. The Stage 03 linearization identity holds to <0.4 % per prompt, so a sign bug would manifest as a per-prompt error correlated with signed-attribution distribution. |
| Phase 0 0a packed graphs not locally available (only on HF) | Low | Pull via `scripts/pipeline/fetch_graph_data.py --source 02c`. Bandwidth-dependent; ~485 MB total for the JSON.gz packed graphs. |
| Phase 1 dissociation fails primary AND fallback bars | Medium-low | Phase 1 is cheap (~7 h minimum). If it fails, the failure mode itself is informative — pivot to attention-head-mediated subcircuit and re-use the existing per-class direction infrastructure. Track A continues regardless. |
| Phase 2 γ-corrected edit has residual `u_C` component | Medium | Level 2 of equivalence verification catches this directly. If proj_magnitude > 0.01 ‖h‖, debug per-layer (likely Gemma's `1 + weight` parameterization handled wrong, or multi-layer interaction not anticipated). |
| Embed/lm_head tying breaks output quality when `W_E` is included | Medium-high | `W_E` deferred to Phase 2b. Default Phase 2a excludes it. Re-test in 2b only if 2a alone fails Level 5 equivalence. |
| Per-class `u_C` vectors are too cosine-similar to dissociate | Medium-low | The direction diagnostic in § 3 measures this BEFORE any intervention. If pairwise `cos(u_C, u_{C'})` > 0.5, dissociation will be hard; flag in Phase 1 plan adjustment. |
| Cross-class promiscuity persists despite the orthogonal-component design | Medium | Our prior data (§ 9.8.3 of REPORT) shows the **activation-selectivity ≠ causal-selectivity** problem at the MLP-feature level. We don't yet have evidence whether `r_jb_C^⊥` resolves it. This is the central empirical question of Phase 1; both outcomes are publishable. Phase 0's H0-4 result also informs this — if edge ablation is strictly more surgical than node ablation in Track A, the same lever applies in Track B's design. |
| RTX 5080 16 GB VRAM constrains Phase 2 equivalence verification (need 2 model copies) | Medium | Capture residuals from each model in separate passes; compare offline. Adds ~30 min wall but avoids OOM. |
| OOD JB set requires unanticipated setup | Low | HarmBench has a standard `harmbench-evaluator` interface; if integration is friction-heavy, AdvBench is plug-and-play. |
| LM judge for helpfulness is biased / unreliable | Low | Run two independent judges (Claude Sonnet 4.6 + GPT-4o); report inter-judge agreement; for headline number use the more conservative judge. |
| Timeline slips past EMNLP deadline | Medium | Phase 0 + Phase 1 + Phase 2 alone constitute a publishable result if Phase 3 slips. EMNLP submission can omit helpfulness benchmark with a "limitations" paragraph and still tell a complete in-distribution story. NeurIPS December cycle is the backup. |

---

## 10. Status of prior open questions (originally for Georg, mostly addressed by 2026-05-17 mentor exchange)

1. **`W_E` inclusion** — Phase 2a defers `W_E` due to embed/lm_head tying. Adds in Phase 2b if Phase 2a alone misses Level 5 equivalence. *Resolution: keep deferred unless empirics force the change.*

2. **Multi-layer `u_C` extraction** — current plan extracts `u_C` once at L15 and applies it (with per-layer γ correction) at every layer. Variant 1C per-layer sweep diagnoses whether per-layer `u_C[L]` is needed; deferred to Week 3. *Resolution: handled by 1C; not blocking.*

3. **Phase 1 fallback to attention-head-mediated subcircuit** — if `r_jb_C^⊥` fails the bar, the brainstorm identified attention-head attribution + targeted W_O edit as the next-best option. *Resolution: documented in § 9 risk register; implement only if Phase 1 fails.*

4. **EMNLP timeline vs NeurIPS December cycle** — if Phase 3 helpfulness benchmarks slip, submit to EMNLP with a limitations note. NeurIPS is the backup. *Resolution: in § 9.*

5. **Cross-model replication (Qwen3, Gemma-2-9B)** — listed in v2 paper outline §4.6/§4.7. *Resolution: out of scope for this EMNLP plan; ride Ruqiya's Qwen3 pipeline rebase as a follow-up.*

6. **Track A (Phase 0) edge-ablation methodology — Georg's 2026-05-17 ask** — *resolution: incorporated as Section 2. Phase 0 is the foundational track running in parallel with Phase 1, with slight priority on early results for Georg.*

---

## 11. Where things live (paths and conventions)

| Artifact | Path |
|---|---|
| Existing per-class direction infrastructure | `scripts/analysis/jb_vector_intervention_per_class.py`, `02b_stats/residuals_L15_per_cond.pt`, `01_direction/unnormalized_r.pt` |
| Existing attribution graph data (Phase 0 input) | `data/results/pipeline_runs/run_20260430_023247/02_attribution/graph_data/<prompt_id>__<condition>.json.gz` (pull locally via `scripts/pipeline/fetch_graph_data.py` if missing) |
| Stage 06 hook helper (reused) | `scripts/pipeline/utils.py::make_intervention_hook` |
| Stage 08 classifier (reused for Phase 0, 1, 2, 3) | `scripts/pipeline/utils.py::classify_response`, `is_coherent` |
| **New Phase 0 entrypoints** | `scripts/emnlp_perm_edit/00_linearization_decomposition.py` (0a), `scripts/emnlp_perm_edit/00_edge_ablation_runtime.py` (0b-simple), `scripts/emnlp_perm_edit/00_edge_ablation_rigorous.py` (0b-rigorous, if needed) |
| New Phase 1 entrypoints | `scripts/emnlp_perm_edit/01_compute_directions.py`, `scripts/emnlp_perm_edit/01_runtime_hook_v{1A,1B,1C}.py`, `scripts/emnlp_perm_edit/01_runtime_hook_controls.py` |
| New Phase 2 entrypoint | `scripts/emnlp_perm_edit/02_weight_edit.py` |
| New equivalence verification | `scripts/emnlp_perm_edit/03_equivalence_verify.py` |
| New Phase 3 entrypoints | `scripts/emnlp_perm_edit/04_harmbench_ood.py`, `scripts/emnlp_perm_edit/05_helpfulness_judge.py` |
| Phase outputs | `data/results/emnlp_perm_edit/phase{0,1,2,3}_*/` |

---

*Drafted 2026-05-17 from brainstorming session with Mahmoud (Track B); updated 2026-05-17 to incorporate Georg's foundational controllability-audit ask as Phase 0 / Track A and to anchor compute estimates on RTX 5080 16 GB VRAM (laptop). Both tracks begin Week 1; Track A prioritized for early results.*
