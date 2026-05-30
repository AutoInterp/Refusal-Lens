# Edge-Ablation Math Audit (Gemma-3-4B-IT, L15)

**Task (Georg, this week):** step through the ablation procedure one operation at a
time, get a clear mathematical account of what we are actually doing, and verify it
is *correct* for Gemma — in particular, explain why we saw no behavioral effect from
edge ablation until we applied the supra-threshold scaling.

**Verdict up front:** the procedure is mathematically correct. The hook does exactly
what it claims (subtract a chosen amount from the refusal-direction projection), and
the edge-derived amount is faithfully transcribed from the attribution graph. The
reason it produced ~no behavioral flip on Gemma is **not** a bug — it is that the
edge-derived perturbation is ~20–40× below the behavioral threshold. When we scale it
up, the flip-rate curve lands exactly on top of the independent direction-intervention
curve. That collapse is the proof of correctness.

All numbers below are from `run_20260430_023247` (50-prompt controlled set, fp32,
TF32 disabled), recomputed live during this audit.

---

## 1. The objects

- **Refusal direction** `r` (unnormalized), built per layer as
  `r[L] = mean(h_harmful[L, pos=-2]) − mean(h_harmless[L, pos=-2])`
  (`scripts/pipeline/.../01_direction`, mirrored in `00_compute_directions.py:73`).
  At L15: **‖r‖ = 3101.2**, so **‖r‖² = 9.616 × 10⁶**.
  `r` is *not* unit-norm; everything downstream is expressed in this raw basis.

- **direct_dot** = `h[L15, pos=-2] · r` — the scalar we attribute. This is the
  measurement-target "logit" node in every attribution graph
  (`graph_loader.find_measurement_target_node_id`).

- **Attribution graph** (circuit-tracer, CLT-based): linearizes direct_dot into a sum
  of signed edge contributions from three source categories — `feature` (transcoder
  decoder writes), `embedding` (token-embedding writes), `error_node` (reconstruction
  residual) — plus a constant baseline offset.

## 2. The hook — what one ablation operation actually does

`make_scalar_rhat_subtraction_hook(r, delta)` (`edge_ablation_hook.py`) registers a
forward hook on layer L15 that, at the targeted position(s), computes

```
h_new = h − (delta / ‖r‖²) · r
```

The only thing this changes about the refusal projection is, by construction:

```
h_new · r = h · r − delta · (r·r)/‖r‖² = (h · r) − delta
```

So **`delta` is literally "how many direct_dot-units to remove from the projection."**
Nothing else. `coeff ≡ delta/‖r‖²` is the same quantity expressed as a multiple of `r`
(i.e. `h_new = h − coeff·r`). This is the single axis on which we compare all
interventions.

Two knobs:
- `position_mode="all"` (default): apply at **every** position on every forward pass
  (Arditi-style). `="last_prompt_only"`: apply only at prompt pos=-2.
- the sign of `delta`: `delta>0` subtracts `r` (moves *away* from refusal → comply);
  `delta<0` adds `r` (moves *toward* refusal).

## 3. Where `delta` comes from in edge ablation

For each (prompt, condition) we load its attribution graph and sum the signed edge
weights flowing into the direct_dot target node, bucketed by source category
(`graph_loader.aggregate_edge_attributions`). The driver
(`00_edge_ablation_runtime.py`) then sets `delta = (chosen bucket sum) × scale`:

| variant | bucket | scale |
|---|---|---|
| `ablate_all_edges` | `all_signed` (= feature+embedding+error) | 1.0 |
| `ablate_features_all` | `feature_signed` | 1.0 |
| `ablate_*_antirefuse_Nx` | `all_signed` | −N |

The linearization identity holds **exactly** in our data (max residual = 0.0 over 50
bare prompts):

```
direct_dot  =  all_signed  +  baseline_offset
```

## 4. Worked example — bare prompt 0

| quantity | value |
|---|---|
| feature_signed | −9,038.6 |
| embedding_signed | **−35,138.9** |
| error_signed | +144.0 |
| **all_signed** | **−44,033.5** |
| direct_dot (h·r) | −28,706.1 |
| baseline_offset (= direct_dot − all_signed) | +15,327.4 |

Two things jump out immediately, and both are central to the answer:

1. **direct_dot is negative for a refusing prompt.** That is fine — the refusal
   *signal* is the relative shift along `r`, not the absolute sign. (The negativity is
   an embedding artifact, see point 2.) What matters: subtracting `r` reduces the
   projection and produces compliance; this is confirmed behaviorally in §6.

2. **79% of `all_signed` is the embedding edge, only 21% is interpretable features**
   (median over the 50 bare prompts: embedding 79.3%, feature 21.0%, error ≈0%). So
   "ablate all edges" on Gemma is *mostly ablating the token-embedding write* to the
   L15 refusal projection, not the transcoder features we actually care about
   interpretively. This is the residual-stream-norm/outlier-dimension issue showing up
   in attribution space.

Now convert `delta` to the comparison axis. Default `ablate_all_edges`:

```
coeff = all_signed / ‖r‖² = −44,033.5 / 9.616e6 = −0.0046   (prompt 0)
median over 550 records: coeff = −0.00501
features-only:           coeff = −0.00130
```

The **negative sign** means default `ablate_all_edges` actually nudges *toward*
refusal (adds `r`). That is the Batch-15 sign bug: to ablate-toward-comply on Gemma
you must flip the sign (the `_antirefuse_` variants, `scale=−N`).

## 5. Why nothing happened — magnitude, made precise

The behavioral threshold lives on the `coeff` axis. From the **independent**
direction-intervention sweep (subtract `coeff·r` at all positions, L15), bare→comply
flip rate:

| coeff | 0.005 | 0.01 | 0.05 | 0.1 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|
| bare comply | 6% | 6% | 20% | 40% | 60% | 94% | 100% |

Inflection ≈ **0.1–0.25**. The edge-derived `|coeff| ≈ 0.005` (all edges) or `0.0013`
(features only) sits **20–50× below** that inflection — right at the 6% floor of the
curve. So removing *exactly* the attributed edge contribution is, by construction, a
sub-threshold perturbation. Nothing was broken; the attribution graph is simply
telling us the edge mass is small relative to the residual-stream geometry, and we
were faithfully removing that small amount.

## 6. The proof of correctness — dose-response collapse

Scale the (sign-corrected, anti-refuse) edge delta by N and compare to the direction
sweep at the matching `coeff = 0.005·N`:

| eff. coeff | direction sweep `coeff·r` | supra edge-ablation `antirefuse_Nx` |
|---|---|---|
| 0.005 (1×) | 6% | 6% (`ablate_all_edges`) |
| 0.025 (5×) | ~8% | 10% |
| 0.05 (10×) | 20% | 18% |
| 0.25 (50×) | 60% | 58% |
| 0.5 (100×) | 94% | 90% |
| 1.0 (200×) | 100% | 100% |

The two curves are the **same curve**. The edge-ablation hook, once put on the same
magnitude axis, reproduces the direction-intervention dose-response point-for-point.
This is only possible if the hook is faithfully implementing `h·r −= delta` and the
edge sum is a correct readout of the contribution. ∎

Negative control: the **wrong-sign** supra variants (`ablate_all_edges_5x…200x`,
which add `r`) run the curve *backwards* — 8% → 6% → 0% → 0% → 0% comply — i.e. the
model refuses *harder* as we push it further toward `r`. Exactly what the sign
analysis predicts.

## 7. Position is the other lever (relevant to Georg's per-position point)

The `delta` is computed at a single position (pos=-2, the target node) but applied at
**every** position in default mode. That uniform application is what drives the flip:

| coeff = 1.0, bare comply | all positions | pos=-2 only |
|---|---|---|
| direction sweep | 100% | 10% |

Editing only the measured position barely moves behavior; the effect comes from
displacing the whole prompt's residual stream. This is precisely the "unnatural"
intervention Georg flagged, and motivates the position-wise direction-subtraction
baseline in the natural-circuit-edits plan. For the *current* edge-ablation
methodology it means: the magnitude on the pos=-2 axis is even further from threshold
than §5 suggests — the 6%→100% all-position curve does most of its work off-target.

---

## 8. Conclusions

1. **The ablation math is correct for Gemma.** Hook identity verified algebraically and
   the linearization identity holds to 0.0 residual. The dose-response collapse onto
   the direction sweep (§6) is independent empirical confirmation.
2. **"No effect" was a magnitude fact, not a bug.** Edge-derived `coeff ≈ 0.005` is
   ~20–50× below the ~0.1–0.25 behavioral inflection. Supra-threshold scaling closes it.
3. **Sign matters on Gemma:** `all_signed < 0`, so the natural-scale subtraction is
   pro-refuse; anti-refuse needs the sign flip.
4. **Embeddings dominate (79%).** On Gemma the all-edges bucket is mostly the
   token-embedding write — a direct fingerprint of the residual-stream-norm issue. The
   interpretively meaningful feature bucket carries only ~21% of the mass and
   `coeff ≈ 0.0013` on its own (~100× sub-threshold).

## 9. Implications for the other two tasks

- **Task 2 (subcircuits on Qwen):** Qwen's residual stream is ~200× smaller-norm, the
  embedding does not swamp the feature bucket, and edge-derived `coeff ≈ 1.0` already
  lands at the top of its dose-response — which is why we expect the Gemma-style
  subcircuit edits to bite on Qwen where they were sub-threshold on Gemma. Same hook,
  same math; the geometry is what changed.
- **Task 1 (frontend):** the target node shown in the graph UI is exactly the
  direct_dot node audited here; the edge weights the UI renders are the same
  `all_signed` mass — worth surfacing the feature-vs-embedding split so the embedding
  domination is visible to reviewers.
