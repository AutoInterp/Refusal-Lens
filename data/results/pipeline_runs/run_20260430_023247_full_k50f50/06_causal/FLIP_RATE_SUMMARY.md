# Stage 06 Causal Intervention — Flip Rate Summary

**Method**: Arditi intervention (add unnormalized r at all positions, every forward step).
**Layer**: L15.
**Dataset**: refusal_lens_controlled_dataset.json (50 prompts × 11 conditions).
**Elapsed**: 53.8 min.

## Direction source

- `r_source`: **stage01**
- \|r\|: **3101.2** (Tejas reports 4019.7 on his bulletproof run)

## Phase 0 — dataset verification

- Bare refused: **50/50**
- Ctrl refused: **250/250** (100.0%)
- Ctrl-leak pairs excluded: **0**
- Bare-comply exclusions: **0**

## Pro-refusal add (headline result)

Flip rate: **100.0%** (89/89 JB-comply prompts flipped to REFUSE)
Coherent flips: **89/89**

### Per-class breakdown

| Class | Comply baseline | Flipped | Rate | Coherent |
|---|---|---|---|---|
| `analytical` | 28 | 28 | 100% | 28 |
| `cognitive_reframe` | 33 | 33 | 100% | 33 |
| `completion` | 0 | 0 | 0% | 0 |
| `fiction` | 19 | 19 | 100% | 19 |
| `roleplay` | 9 | 9 | 100% | 9 |

## Anti-refusal sub (bare → comply)

Flip rate: **98.0%** (49/50 bare-refuse prompts flipped to COMPLY)
Coherent flips: **49/49**

## Symmetry claim

The bidirectional flip symmetry is the headline. If pro is ~100% and anti is high, the L15 unnormalized r vector IS the model's refusal axis — not a one-way push. This is the causal complement to the Stage 07 correlational `jb_vs_ctrl_contrast` finding.

## Phase 2c — benign force-refuse control (Tejas bulletproof)

Force-refuse rate on 10 benign prompts: **100.0%** (10/10)
Coherent responses: **10/10**

Tejas reports **10/10** on his bulletproof run. A result below ~80% here would indicate the intervention isn't a generic refusal push, invalidating the 'L15 r IS the refusal axis' claim.

