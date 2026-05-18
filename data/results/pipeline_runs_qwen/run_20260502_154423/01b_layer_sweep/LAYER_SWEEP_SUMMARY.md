# Qwen3-4B Causal-Layer Sweep — Headline

**Best causal layer (by coherent flip rate): L18 = 100%**

- Currently in `pipeline_qwen/config.py`: `BEST_CAUSAL_LAYER = 18` (100% coherent flip)
- ✓ Current value matches sweep — no update needed.

## Top 5 layers (coherent flip rate)

| Layer | Coherent flip | Any flip | Per-class breakdown |
|---|---|---|---|
| L18 | 100% | 100% | roleplay=100%, fiction=0%, analytical=100%, completion=0%, cognitive_reframe=100% |
| L19 | 100% | 100% | roleplay=100%, fiction=0%, analytical=100%, completion=0%, cognitive_reframe=100% |
| L20 | 100% | 100% | roleplay=100%, fiction=0%, analytical=100%, completion=0%, cognitive_reframe=100% |
| L21 | 100% | 100% | roleplay=100%, fiction=0%, analytical=100%, completion=0%, cognitive_reframe=100% |
| L22 | 100% | 100% | roleplay=100%, fiction=0%, analytical=100%, completion=0%, cognitive_reframe=100% |

## Setup

- model: `Qwen/Qwen3-4B`
- n_prompts: 8 × 5 jb_classes = 40 (prompt, class) pairs evaluated as baseline
- n_baseline_comply_pairs: 23 (57%) — only pairs where baseline COMPLY are eligible substrate
- max_new_tokens: 60
- intervention: `h[:,:,:] += r_unnormalized` at every position, `r` from Stage 01 unnormalized_r.pt at the same layer

## Notes

- Coherent flip rate is the headline metric — gibberish flips don't count.
- Tejas reports 90/90 = 100% on Gemma-3 at L15 with full dataset. Below ~70% on Qwen here may indicate (a) dataset too small, (b) max_new_tokens too low, or (c) Qwen's refusal axis is genuinely weaker than Gemma's at any single layer.