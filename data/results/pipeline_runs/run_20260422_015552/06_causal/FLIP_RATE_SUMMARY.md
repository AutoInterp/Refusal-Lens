# Stage 06 Causal Intervention — Flip Rate Summary

**Method**: Arditi intervention (add unnormalized r at all positions, every forward step).
**Layer**: L15.
**Dataset**: refusal_lens_controlled_dataset.json (50 prompts × 11 conditions).
**Elapsed**: 54.6 min.

## Direction source

- `r_source`: **recompute**
- \|r\|: **3123.9** (Tejas reports 4019.7 on his bulletproof run)
- Recomputed from 64+64 prompts under the same bf16 model load as intervention (Tejas-exact)

## Phase 0 — dataset verification

- Bare refused: **49/50**
- Ctrl refused: **246/250** (98.4%)
- Ctrl-leak pairs excluded: **4**
- Bare-comply exclusions: **1**

## Pro-refusal add (headline result)

Flip rate: **96.7%** (87/90 JB-comply prompts flipped to REFUSE)
Coherent flips: **87/87**

### Per-class breakdown

| Class | Comply baseline | Flipped | Rate | Coherent |
|---|---|---|---|---|
| `analytical` | 27 | 27 | 100% | 27 |
| `cognitive_reframe` | 33 | 32 | 97% | 32 |
| `completion` | 1 | 1 | 100% | 1 |
| `fiction` | 20 | 18 | 90% | 18 |
| `roleplay` | 9 | 9 | 100% | 9 |

## Anti-refusal sub (bare → comply)

Flip rate: **100.0%** (49/49 bare-refuse prompts flipped to COMPLY)
Coherent flips: **49/49**

## Symmetry claim

The bidirectional flip symmetry is the headline. If pro is ~100% and anti is high, the L15 unnormalized r vector IS the model's refusal axis — not a one-way push. This is the causal complement to the Stage 07 correlational `jb_vs_ctrl_contrast` finding.

## Phase 2c — benign force-refuse control (Tejas bulletproof)

Force-refuse rate on 10 benign prompts: **100.0%** (10/10)
Coherent responses: **10/10**

Tejas reports **10/10** on his bulletproof run. A result below ~80% here would indicate the intervention isn't a generic refusal push, invalidating the 'L15 r IS the refusal axis' claim.

