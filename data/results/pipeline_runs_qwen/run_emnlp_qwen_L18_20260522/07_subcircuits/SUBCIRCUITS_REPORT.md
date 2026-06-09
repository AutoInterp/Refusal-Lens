# Subcircuits Report (Rule-Based)

Each subcircuit is defined by a precise set-logic rule over the features observed across bare + 5 JB classes (original rules) and bare + 5 jb_* + 5 ctrl_* conditions (new Apr 22 ctrl-aware rules). No ML fitting — fully interpretable.

## Summary table

| Subcircuit | Size | Peak layer | n_layers occupied | Mean act. freq. |
|---|---|---|---|---|
| `roleplay_exclusive` | 158 | L0 (×20) | 18 | N/A |
| `sign_flip_convergent` | 139 | L5 (×25) | 14 | N/A |
| `cognitive_reframe_exclusive` | 135 | L0 (×23) | 18 | N/A |
| `universal_refusal_core` | 122 | L8 (×16) | 16 | N/A |
| `fiction_exclusive` | 105 | L0 (×19) | 17 | N/A |
| `completion_exclusive` | 98 | L8 (×12) | 19 | N/A |
| `ctrl_shared_refusal` | 98 | L8 (×15) | 16 | N/A |
| `dampening_specialists` | 94 | L10 (×13) | 15 | N/A |
| `analytical_exclusive` | 80 | L5 (×10) | 19 | N/A |
| `anti_refusal_amplifiers` | 39 | L13 (×6) | 15 | N/A |
| `jb_completion_specific_vs_ctrl` | 36 | L8 (×11) | 14 | N/A |
| `canonical_pro_refusal` | 35 | L0 (×9) | 13 | N/A |
| `jb_cognitive_reframe_specific_vs_ctrl` | 33 | L4 (×4) | 14 | N/A |
| `jb_analytical_specific_vs_ctrl` | 31 | L7 (×4) | 14 | N/A |
| `jb_fiction_specific_vs_ctrl` | 31 | L8 (×8) | 13 | N/A |
| `jb_roleplay_specific_vs_ctrl` | 25 | L7 (×3) | 14 | N/A |
| `late_wave_layer24_32` | 0 | — | 0 | N/A |
| `ctrl_only` | 0 | — | 0 | N/A |

## JB-vs-Ctrl recruitment contrast (NEW — Task 10)

For each JB class, how much of the corpus-level top-50 recruitment is **genuinely JB-semantic** vs. **prefix-induced** (also triggered by the matched benign ctrl prefix)? Old L32 data could not compute this; it's the headline new finding enabled by the 11-condition ctrl-balanced dataset.

| Class | \|jb_top50\| | \|ctrl_top50\| | Intersection | JB-specific | Ctrl-specific | **JB-specific %** | Overlap % |
|---|---|---|---|---|---|---|---|
| `analytical` | 204 | 298 | 173 | 31 | 125 | **15%** | 85% |
| `cognitive_reframe` | 167 | 232 | 134 | 33 | 98 | **20%** | 80% |
| `completion` | 229 | 283 | 193 | 36 | 90 | **16%** | 84% |
| `fiction` | 166 | 248 | 135 | 31 | 113 | **19%** | 81% |
| `roleplay` | 177 | 243 | 152 | 25 | 91 | **14%** | 86% |

**Reading**: `JB-specific %` close to 100 → JB recruits mechanisms the benign prefix does NOT (strong JB-semantic signal). Close to 0 → JB's effect is mostly a prefix-length artifact, not genuine semantic mechanism.


## Subcircuit definitions and top features

### `roleplay_exclusive` — n=158

Features seen in **only** the `roleplay` JB class (no bare, no other JB). Candidates for `roleplay`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L8:F152608` | L8 | 0.113 |  |
| `L12:F3556` | L12 | 0.072 |  |
| `L0:F67783` | L0 | 0.070 |  |
| `L10:F149039` | L10 | 0.065 |  |
| `L13:F151341` | L13 | 0.065 |  |

### `sign_flip_convergent` — n=139

Features in the **sign_flipped** bucket of ≥3 JB classes. Robustly reverse attribution sign under JB — the highest-confidence mechanism-change features.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L0:F127509` | L0 | 0.058 |  |
| `L10:F80002` | L10 | 0.041 |  |
| `L5:F122628` | L5 | 0.027 |  |
| `L12:F124617` | L12 | 0.026 |  |
| `L9:F45111` | L9 | 0.023 |  |

### `cognitive_reframe_exclusive` — n=135

Features seen in **only** the `cognitive_reframe` JB class (no bare, no other JB). Candidates for `cognitive_reframe`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L5:F58115` | L5 | 0.115 |  |
| `L0:F5435` | L0 | 0.103 |  |
| `L7:F8016` | L7 | 0.096 |  |
| `L17:F40372` | L17 | 0.092 |  |
| `L12:F72549` | L12 | 0.086 |  |

### `universal_refusal_core` — n=122

Features seen in **bare + all 5 JB classes**. The canonical refusal core — present in both harmful-alone and every jailbreak. Ablation control baseline.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L17:F83241` | L17 | 1.440 |  |
| `L18:F146597` | L18 | 1.359 |  |
| `L14:F97787` | L14 | 1.236 |  |
| `L15:F63157` | L15 | 0.976 |  |
| `L18:F81277` | L18 | 0.702 |  |

### `fiction_exclusive` — n=105

Features seen in **only** the `fiction` JB class (no bare, no other JB). Candidates for `fiction`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L8:F162014` | L8 | 0.093 |  |
| `L10:F146008` | L10 | 0.069 |  |
| `L8:F129759` | L8 | 0.069 |  |
| `L8:F54449` | L8 | 0.068 |  |
| `L14:F135770` | L14 | 0.068 |  |

### `completion_exclusive` — n=98

Features seen in **only** the `completion` JB class (no bare, no other JB). Candidates for `completion`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L0:F51603` | L0 | 0.133 |  |
| `L8:F43582` | L8 | 0.099 |  |
| `L15:F43607` | L15 | 0.098 |  |
| `L8:F121013` | L8 | 0.088 |  |
| `L8:F6396` | L8 | 0.078 |  |

### `ctrl_shared_refusal` — n=98

Features in **bare ∩ all 5 ctrl_*_top50** but **NOT in all 5 jb_*_top50**. The prefix-invariant refusal spine: machinery the refusal circuit uses regardless of whether the prefix carries JB-semantics or matched benign content. These are NOT JB-semantic — they define the baseline that survives a long-prefix perturbation without JB intent.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L18:F141167` | L18 | 0.933 |  |
| `L18:F81277` | L18 | 0.702 |  |
| `L18:F110876` | L18 | 0.617 |  |
| `L18:F17980` | L18 | 0.389 |  |
| `L18:F113352` | L18 | 0.379 |  |

### `dampening_specialists` — n=94

Features in the **dampened** bucket of ≥3 JB classes. Pro-refusal features whose contribution to the refusal direction weakens across most JB types.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L17:F83241` | L17 | 1.440 |  |
| `L18:F146597` | L18 | 1.359 |  |
| `L14:F97787` | L14 | 1.236 |  |
| `L15:F63157` | L15 | 0.976 |  |
| `L18:F141167` | L18 | 0.933 |  |

### `analytical_exclusive` — n=80

Features seen in **only** the `analytical` JB class (no bare, no other JB). Candidates for `analytical`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L7:F34044` | L7 | 0.122 |  |
| `L17:F78757` | L17 | 0.096 |  |
| `L6:F60377` | L6 | 0.086 |  |
| `L17:F58715` | L17 | 0.065 |  |
| `L2:F74986` | L2 | 0.063 |  |

### `anti_refusal_amplifiers` — n=39

Features in the **amplified_anti** bucket of ≥3 JB classes. Anti-refusal features that grow in magnitude across most JB types — the bypass signal.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L17:F82119` | L17 | 0.624 |  |
| `L16:F35905` | L16 | 0.580 |  |
| `L16:F132756` | L16 | 0.204 |  |
| `L16:F45664` | L16 | 0.200 |  |
| `L11:F44587` | L11 | 0.186 |  |

### `jb_completion_specific_vs_ctrl` — n=36

Features in **jb_completion_top50 − ctrl_completion_top50**. The cleanest JB-semantic subcircuit for `completion`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L18:F4806` | L18 | 0.193 |  |
| `L18:F18474` | L18 | 0.175 |  |
| `L0:F51603` | L0 | 0.133 |  |
| `L13:F104173` | L13 | 0.103 |  |
| `L15:F94050` | L15 | 0.102 |  |

### `canonical_pro_refusal` — n=35

Features seen in **all 5 JB classes but NOT bare**. Recruited specifically under jailbreak — interpretable as a shared JB-suppression / pro-refusal response.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L16:F117182` | L16 | 0.115 |  |
| `L13:F101420` | L13 | 0.095 |  |
| `L8:F24881` | L8 | 0.085 |  |
| `L8:F52135` | L8 | 0.080 |  |
| `L5:F122628` | L5 | 0.027 |  |

### `jb_cognitive_reframe_specific_vs_ctrl` — n=33

Features in **jb_cognitive_reframe_top50 − ctrl_cognitive_reframe_top50**. The cleanest JB-semantic subcircuit for `cognitive_reframe`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L16:F56404` | L16 | 0.132 |  |
| `L5:F15544` | L5 | 0.130 |  |
| `L5:F58115` | L5 | 0.115 |  |
| `L0:F5435` | L0 | 0.103 |  |
| `L13:F85261` | L13 | 0.099 |  |

### `jb_analytical_specific_vs_ctrl` — n=31

Features in **jb_analytical_top50 − ctrl_analytical_top50**. The cleanest JB-semantic subcircuit for `analytical`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L16:F56404` | L16 | 0.132 |  |
| `L7:F34044` | L7 | 0.122 |  |
| `L13:F104173` | L13 | 0.103 |  |
| `L13:F85261` | L13 | 0.099 |  |
| `L6:F60377` | L6 | 0.086 |  |

### `jb_fiction_specific_vs_ctrl` — n=31

Features in **jb_fiction_top50 − ctrl_fiction_top50**. The cleanest JB-semantic subcircuit for `fiction`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L12:F30188` | L12 | 0.093 |  |
| `L17:F100284` | L17 | 0.091 |  |
| `L8:F52135` | L8 | 0.080 |  |
| `L7:F56075` | L7 | 0.076 |  |
| `L8:F81362` | L8 | 0.072 |  |

### `jb_roleplay_specific_vs_ctrl` — n=25

Features in **jb_roleplay_top50 − ctrl_roleplay_top50**. The cleanest JB-semantic subcircuit for `roleplay`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L16:F56404` | L16 | 0.132 |  |
| `L5:F15544` | L5 | 0.130 |  |
| `L8:F152608` | L8 | 0.113 |  |
| `L15:F94050` | L15 | 0.102 |  |
| `L17:F21148` | L17 | 0.096 |  |

### `late_wave_layer24_32` — n=0

All tagged features in layers **24–32** — the JB-impact band identified in A8. Layer-based cross-cut; overlaps other subcircuits.

### `ctrl_only` — n=0

Features in **all 5 ctrl_*_top50** but not in bare or any jb_*_top50. Usually tiny; if non-empty, it signals that matched benign prefixes recruit features neither bare-harmful nor any jailbreak uses — typically benign-content semantic features triggered by the ctrl prefix text itself.

## Pairwise overlap (top 10 by normalized intersection)

Normalized overlap = |A ∩ B| / min(|A|, |B|). High values mean the smaller set is largely contained in the larger. `late_wave_layer24_32` naturally absorbs many.

| A | B | norm. overlap |
|---|---|---|
| `canonical_pro_refusal` | `sign_flip_convergent` | 0.89 |
| `universal_refusal_core` | `dampening_specialists` | 0.82 |
| `completion_exclusive` | `jb_completion_specific_vs_ctrl` | 0.58 |
| `fiction_exclusive` | `jb_fiction_specific_vs_ctrl` | 0.52 |
| `analytical_exclusive` | `jb_analytical_specific_vs_ctrl` | 0.45 |
| `jb_cognitive_reframe_specific_vs_ctrl` | `jb_roleplay_specific_vs_ctrl` | 0.40 |
| `dampening_specialists` | `ctrl_shared_refusal` | 0.39 |
| `universal_refusal_core` | `ctrl_shared_refusal` | 0.34 |
| `universal_refusal_core` | `anti_refusal_amplifiers` | 0.33 |
| `cognitive_reframe_exclusive` | `jb_cognitive_reframe_specific_vs_ctrl` | 0.27 |

## Suggested Stage 08 ablation targets (causal-impact order)

1. `canonical_pro_refusal` — JB-specific pro-refusal recruitment. Ablation should *strengthen* JB bypass (removes the JB-only refusal boost).
2. `jb_{cls}_specific_vs_ctrl` (per class) — the cleanest per-class JB-semantic mechanism. Ablating one should selectively restore ctrl-like behavior on that class (dissociation test).
3. `sign_flip_convergent` — robust direction reversals. Ablation should partially restore bare behavior under JB.
4. `dampening_specialists` — weakened pro-refusal features. Restoring them to bare strength should counter fiction/analytical bypass.
5. `anti_refusal_amplifiers` — JB-amplified bypass signal. Suppressing them should increase refusal under JB.
6. `ctrl_shared_refusal` — the prefix-invariant spine. Ablation should break refusal on BOTH ctrl and bare — a negative control proving these aren't JB-specific.
7. `universal_refusal_core` — shared baseline. Ablation should break refusal on bare *and* JB (control — proves the subcircuits matter).

