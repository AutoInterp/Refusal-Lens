# Gemma-3-4B-IT → Qwen3-4B Migration Notes

What changed when porting `data/tejas_experiments/scripts/` to Qwen, and why.

## 1. Model & Transcoders

| | Gemma | Qwen |
|---|---|---|
| Model ID | `google/gemma-3-4b-it` | `Qwen/Qwen3-4B` |
| Transcoder ID | `mwhanna/gemma-scope-2-4b-it` | `mwhanna/qwen3-4b-transcoders` |
| Transcoder subpath | `transcoder_all/width_16k_l0_small_affine` | TBD — check HF repo |
| Architecture style | Multimodal (text + vision) | Text-only causal LM |
| Hidden size | 2560 | 2560 (same) |
| Num transformer blocks | 34 | 36 |
| Attention | GQA | GQA (different head count) |

## 2. Code-level differences

### Config access (multimodal vs flat)

Gemma 3 wraps everything in `text_config`:

```python
# Gemma:
d_model = model.config.text_config.hidden_size

# Qwen:
d_model = model.config.hidden_size
```

CONFIG.py provides `get_hidden_size(model)` which handles both.

### Decoder layer access

Gemma 3 nests the language model inside a multimodal wrapper:

```python
# Gemma:
model.model.language_model.layers[LAYER]

# Qwen:
model.model.layers[LAYER]
```

CONFIG.py provides `get_decoder_layers(model)`.

### Chat template

Different special tokens entirely. Gemma:

```
<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
```

Qwen:

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

The Gemma scripts assumed position **-2 = the literal `model` token**, which
was the position with strongest separation. For Qwen, position -2 is the `\n`
after `assistant`. The semantically analogous "first generation token" position
is different, so **the optimal position must be rediscovered**.

## 3. Hyperparameters that need re-tuning

| Hyperparameter | Gemma value | Why it might differ for Qwen |
|---|---|---|
| Best layer (separation) | 32 | Different depth (36 vs 34) and training |
| Best position | -2 | Different chat template tokens |
| Causal layer | 15 | Mid-network position varies by architecture |
| Per-position anti-correlation | cos(-2, -3) ≈ -0.78 | Token semantics differ |

Run `01_compute_direction_and_sanity.py` to discover position + best layer.
Run a sweep over candidate causal layers in `16_causal_arditi.py` (or test
each layer L ∈ {10, 13, 15, 18, 20, 25} as Gemma did) to find the causal one.

## 4. Hyperparameters that DO carry over (Georg's corrections)

These are numerical-recipe choices that should work for any model:

- **Float64 accumulation** of mean activations (never bfloat16)
- **Left padding** (right padding misaligns the trailing positions used for direction extraction)
- **Multi-position sweep** at `[-5, -4, -3, -2, -1]`
- **No filter** on attribution features (use all ~14k+ active, not capped 3k)
- **Use IT model + IT transcoders** (not PT)

## 5. Path conventions

The Gemma scripts hardcoded `/workspace/...` paths from the remote training
pod. The Qwen scripts use `pathlib.Path` rooted at the repo via CONFIG:

```python
RESULTS_V2_DIR = PROJECT_ROOT / "data" / "qwen_experiments" / "results_v2"
```

This makes the scripts portable across machines.

## 6. Open questions for Qwen

Things the Gemma analysis answered that we don't yet know for Qwen:

1. Does Qwen also have **two distinct MLP jailbreak classes** (RP = dampening,
   fiction = tug-of-war)?
2. Does the same **99.6% attention / 0.4% MLP** split hold?
3. Is Qwen's refusal also **causally mediated by a single direction** at one
   specific layer, or distributed across layers?
4. Are the **circuit-informed jailbreaks** (analytical, completion, meta-grading)
   that bypass Gemma also effective on Qwen?
5. Does Qwen exhibit the **per-position anti-correlation** pattern at its
   causal layer?

The point of the comparison is to determine which findings are **architectural**
(generalize) vs **model-specific** (Gemma-only artifact).
