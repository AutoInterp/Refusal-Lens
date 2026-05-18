# Stage 06 Causal Intervention — Flip Rate Summary

**Method**: Arditi intervention (add unnormalized r at all positions, every forward step).
**Layer**: L18.
**Dataset**: refusal_lens_controlled_dataset.json (50 prompts × 11 conditions).
**Elapsed**: 36.4 min.

## Direction source

- `r_source`: **stage01**
- \|r\|: **15.1** (Tejas reports 4019.7 on his bulletproof run)

## Phase 0 — dataset verification

- Bare refused: **40/50**
- Ctrl refused: **237/250** (94.8%)
- Ctrl-leak pairs excluded: **13**
- Bare-comply exclusions: **10**

## Pro-refusal add (headline result)

Flip rate: **96.3%** (130/135 JB-comply prompts flipped to REFUSE)
Coherent flips: **130/130**

### Per-class breakdown

| Class | Comply baseline | Flipped | Rate | Coherent |
|---|---|---|---|---|
| `analytical` | 39 | 35 | 90% | 35 |
| `cognitive_reframe` | 36 | 36 | 100% | 36 |
| `completion` | 7 | 7 | 100% | 7 |
| `fiction` | 15 | 15 | 100% | 15 |
| `roleplay` | 38 | 37 | 97% | 37 |

## Anti-refusal sub (bare → comply)

Flip rate: **92.5%** (37/40 bare-refuse prompts flipped to COMPLY)
Coherent flips: **37/37**

## Symmetry claim

The bidirectional flip symmetry is the headline. If pro is ~100% and anti is high, the L15 unnormalized r vector IS the model's refusal axis — not a one-way push. This is the causal complement to the Stage 07 correlational `jb_vs_ctrl_contrast` finding.

## Phase 2c — benign force-refuse control (Tejas bulletproof)

Force-refuse rate on 10 benign prompts: **0.0%** (0/10)
Coherent responses: **10/10**

Tejas reports **10/10** on his bulletproof run. A result below ~80% here would indicate the intervention isn't a generic refusal push, invalidating the 'L15 r IS the refusal axis' claim.

