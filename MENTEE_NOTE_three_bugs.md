# Three bugs in the attribution pipeline

We tracked these down while trying to verify whether the sum of all attribution
scores (features + errors + embeddings) actually reconstructs `r̂·h_residual`
on a single prompt. The answer ended up being "yes, modulo a baseline" —
but only after fixing three independent bugs that were stacked on top of each
other.

## TL;DR (for Slack)

> Found 3 stacked bugs in our attribution pipeline that were jointly making
> the methodology look broken:
>
> 1. **Stage 03 was comparing against the wrong residual stream point** —
>    `hidden_states[L+1]` (post-block, norm ≈ 30,000) instead of
>    `pre_feedforward_layernorm.output[L]` (post-RMSNorm pre-MLP, norm ≈ 18).
>    The "50% MLP ratio" the script reported was meaningless.
> 2. **Stage 02 wasn't actually passing `measurement_layer` /
>    `measurement_position` to `circuit-tracer.attribute()`** — it
>    hardcoded the metadata but the actual attribute call defaulted to
>    post-stack at pos=-1. Metadata lied about reality.
> 3. **The refusal direction was being computed at the residual stream
>    (Arditi-style, `hidden_states[L+1]`) but applied as a cotangent at
>    a totally different point** (post-RMSNorm pre-MLP, ~1700× different
>    magnitude). We patched circuit-tracer to support a new
>    `measurement_hook="hook_resid_post"` parameter that injects the
>    cotangent at the same point where the direction was computed.
>
> After fixes: methodology check passes — `Σ edges + baseline = direct_dot`
> exactly. MPS = CUDA to 4 decimal places (ruling out the suspected backend
> bug). Three real-but-orthogonal issues.

---

## Bug 1: Stage 03 verification was comparing different residual-stream points

### What the bug is

`scripts/pipeline/03_verify_attribution.py` reads the recorded attribution sum
from Stage 02 and compares it to the residual-stream projection
`r̂·h` *but at the wrong point in the model's computation*. It used HF's
`hidden_states[L+1]`, which is the **post-block residual** (after all of
attention + MLP + both layernorms + residual adds). Circuit-tracer's
attribution actually measures at the **input to the MLP** (post-RMSNorm
pre-MLP, inside the block).

### How it occurred

In Gemma-3, every transformer block has *four* RMSNorms (input_layernorm,
post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm).
Two of them sandwich the MLP. The transcoder is hooked at
`pre_feedforward_layernorm.output` (post-RMSNorm pre-MLP), where its norm is
~18, but `hidden_states[L+1]` is the residual *after the whole block*
including post_feedforward_layernorm and the residual add — norm ~30,000.

```python
# 03_verify_attribution.py (BEFORE)
out = model(**inputs, output_hidden_states=True)
act = out.hidden_states[best_layer + 1][0, best_pos, :]   # norm ≈ 30,000
dot = (act @ r_hat).item()                                 # = -29,580
ratio = attr_net / dot                                     # ≈ 0.0023, NONSENSE
```

`hidden_states[L+1]` and `pre_feedforward_layernorm.output[L]` differ by
1,700× in norm, opposite signs in the projection, and represent totally
different stages of the block.

### Consequences if unfixed

The script reports a "MLP transcoders explain 50% of the refusal signal"
or similar ratio that is *entirely meaningless* — it's a magnitude mismatch
between two different residual-stream points, not actual transcoder coverage.
Anyone reading this output would draw wrong conclusions about how much of
the refusal signal the SAEs/transcoders actually capture.

### How we fixed it

Capture the residual at the *exact* hook circuit-tracer measures via a forward
hook on `pre_feedforward_layernorm` of the measurement layer:

```python
# 03_verify_attribution.py (AFTER)
captured = {}
def hook_fn(module, inputs, output):
    captured["act"] = output.detach()

measurement_module = model.model.language_model.layers[best_layer].pre_feedforward_layernorm
handle = measurement_module.register_forward_hook(hook_fn)
try:
    with torch.no_grad():
        model(**inputs, output_hidden_states=True)
finally:
    handle.remove()

act = captured["act"][0, best_pos, :]   # norm ≈ 18, the right tensor
dot = (act @ r_hat).item()
```

Now the ratio compares the same scalar at the same physical location.

---

## Bug 2: Stage 02 wasn't actually passing measurement settings to circuit-tracer

### What the bug is

`scripts/pipeline/02_run_attribution.py` records `metadata.measurement_layer = 32`
in the output JSON, but the actual `attribute()` call doesn't pass
`measurement_layer` or `measurement_position`. Circuit-tracer falls back to
its defaults (post-transformer layer + last token position). The metadata
field is purely descriptive — it doesn't drive the computation.

### How it occurred

```python
# 02_run_attribution.py (BEFORE)
g = attribute(
    prompt=formatted,
    model=model,
    attribution_targets=[target],
    batch_size=args.batch_size,
    max_feature_nodes=args.max_features,
    verbose=False,
    # ← measurement_layer and measurement_position NOT passed
)
...
metadata = {"measurement_layer": config.BEST_SEPARATION_LAYER,
            "measurement_position": -2}                              # ← lies
```

Circuit-tracer's defaults:
```python
_ml = n_layers if measurement_layer is None else measurement_layer   # = 34
_mp = n_pos - 1 if measurement_position is None else measurement_position  # = -1
```

So every attribution actually measured at *post-stack output, last token* —
not at L32 pos=-2 like the metadata claimed.

### Consequences if unfixed

Two big ones:
1. Stage 03 reads the metadata to decide where to compute the direct dot
   product. It reads "L32 pos=-2" and dutifully measures there — but the
   recorded values were actually computed at post-stack pos=-1. Stage 03's
   verification ratio is then garbage on top of the Bug 1 garbage.
2. Anyone reading the JSON thinks they're looking at L32 pos=-2 attributions.
   They aren't.

### How we fixed it

Pass the parameters explicitly:

```python
# 02_run_attribution.py (AFTER)
g = attribute(
    prompt=formatted,
    model=model,
    attribution_targets=[target],
    batch_size=args.batch_size,
    max_feature_nodes=args.max_features,
    measurement_layer=config.MEASUREMENT_LAYER,        # = 32
    measurement_position=config.MEASUREMENT_POSITION,  # = -2
    verbose=False,
)
```

Now the metadata field reflects what was actually computed.

---

## Bug 3: Refusal direction was extracted at one point but used at another

### What the bug is

This is the deep one and what spawned a circuit-tracer library patch.

Stage 01 (`01_compute_direction.py`) extracts the refusal direction
`r̂_L = (mean_harmful − mean_harmless).normalize()` from
`hidden_states[L+1]` — the **residual stream** value at the end of layer L.
This matches Arditi et al.'s "Refusal in LLMs is mediated by a single
direction" methodology: refusal direction lives in the residual stream.

Stage 02 then *uses* `r̂_L` as the cotangent in `attribute()`. But
circuit-tracer's `measurement_layer=L` injects the cotangent at the
post-RMSNorm pre-MLP point (`pre_feedforward_layernorm.output[L]`,
because the transcoder's `feature_input_hook` is `mlp.hook_in`).

So the **direction was computed in one basis (residual stream) and applied
in a different basis (post-RMSNorm)**. The two points differ by:
- ~1700× in magnitude
- one MLP block's contribution
- one RMSNorm transformation (which is *non-linear* in the per-token RMS
  and therefore can't be undone by simple rescaling)

### How it occurred

```python
# 01_compute_direction.py
act = out.hidden_states[layer + 1][j, pos, :]   # post-block residual, norm ≈ 30,000
means[layer] += act / n
# → direction r̂_L lives at hidden_states[L+1] (residual stream)
```

```python
# 02_run_attribution.py
target = CustomTarget(token_str=..., prob=1.0, vec=r_hat)
g = attribute(..., attribution_targets=[target], measurement_layer=L)
# → r_hat injected at pre_feedforward_layernorm.output[L] (post-RMSNorm, norm ≈ 18)
```

Same vector `r_hat`, but the residual it's projected against is on a totally
different scale and represents a different stage of the block.

### Consequences if unfixed

The attribution methodology becomes interpretively muddled. The recorded
"net" value is `r̂_L · h_post_norm[L]` minus a baseline — but `h_post_norm`
isn't the residual stream you reasoned about when defining the direction.
The "refusal signal" you're attributing isn't quite the refusal signal
Arditi described. Causal interventions (activation addition / directional
ablation, which work in the residual stream) and attribution (which would
work in the post-norm basis) become un-comparable.

### How we fixed it

We patched `circuit-tracer` to accept a new `measurement_hook` parameter
that controls *where* in the model the cotangent is injected — independently
of where the transcoder's `feature_input_hook` lives. With
`measurement_hook="hook_resid_post"`, the cotangent gets injected at
`hook_resid_post[L]` (= `hidden_states[L+1]`, the residual stream — exactly
where Stage 01 extracted the direction).

Files patched in `vendor/circuit-tracer`:

```
attribution/attribute.py                  + measurement_hook parameter (dispatcher)
attribution/attribute_transformerlens.py  + thread param + use_measurement_cache
attribution/context_transformerlens.py    + _measurement_resid_activations cache
attribution/attribute_nnsight.py          + thread param (also patched but
                                            blocked by nnsight's .grad-on-non-
                                            module-output limitation)
attribution/context_nnsight.py            + matching cache (idem)
replacement_model/replacement_model_*.py  + get_measurement_loc()
utils/tl_nnsight_mapping.py               + hook_resid_pre entry for Gemma-3
                                            (used to resolve hook_resid_post
                                            via next layer's input_layernorm)
```

Usage:

```python
from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.attribution.targets import CustomTarget

model = ReplacementModel.from_pretrained(
    "google/gemma-3-4b-it",
    "mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine",
    backend="transformerlens",   # nnsight backend has unrelated `.grad` issues
                                 # for residual-stream measurement; TL works.
    device=torch.device("mps"),
    dtype=torch.float32,
)

graph = attribute(
    prompt=formatted_prompt,
    model=model,
    attribution_targets=[CustomTarget("refusal", 1.0, r_hat_15)],
    measurement_layer=15,
    measurement_position=-2,
    measurement_hook="hook_resid_post",   # ← NEW: injects cotangent at the
                                          # residual stream, matching where
                                          # r_hat_15 was extracted in Stage 01
)
```

After this fix, the methodology check passes:

```
Σ edges (features + errors + embeds + logits)  +  baseline  =  direct_dot
                                              -51,205        +21,625        -29,580
```

The `baseline` is `r̂ · h(at sources=0)` in the linearized model — small at
the post-final-norm point (~−10) but large at intermediate residual-stream
points (~+21,625) because accumulated transcoder `b_dec` terms propagate
through 15 frozen-attention layers without being damped by the final norm.

This is exactly what Anthropic's circuit-tracing methodology says should
happen: `Σ edges = direct_dot − baseline`. The Anthropic papers bypass the
baseline issue by using *demeaned logit targets* at the unembed point —
demeaning makes the baseline cancel by orthogonality. For raw refusal-
direction targets at intermediate layers, the baseline is non-zero and has
to be either (a) measured separately and subtracted, or (b) tolerated as a
constant offset that doesn't affect relative feature ranking.

---

## What's NOT a bug (we suspected, then ruled out)

- **MPS-vs-CUDA gradient mismatch.** We saw an apparent ~8× gap between MPS
  and CUDA at one point. Ruled out: when run with the *same* settings, MPS
  and CUDA produce identical attribution sums to 4 decimal places. The
  apparent gap was an artifact of comparing two different measurement points
  across runs.
- **Transcoder dtype downcast on MPS.** Suspected based on a subagent
  hypothesis, but the actual checkpoints are float32 and the einsum runs in
  float32 on MPS. No silent precision loss.
- **The methodology itself.** `Σ edges = r̂·h − baseline` holds by
  construction (linearization theorem) and is reproduced empirically once
  the three bugs above are fixed.
