# Subcircuits Report (Rule-Based)

Each subcircuit is defined by a precise set-logic rule over the features observed across bare + 5 JB classes (original rules) and bare + 5 jb_* + 5 ctrl_* conditions (new Apr 22 ctrl-aware rules). No ML fitting — fully interpretable.

## Summary table

| Subcircuit | Size | Peak layer | n_layers occupied | Mean act. freq. |
|---|---|---|---|---|
| `roleplay_exclusive` | 49 | L11 (×7) | 15 | 0.0189 |
| `universal_refusal_core` | 39 | L0 (×6) | 14 | 0.0110 |
| `fiction_exclusive` | 36 | L12 (×5) | 12 | 0.0163 |
| `dampening_specialists` | 26 | L13 (×4) | 11 | 0.0144 |
| `cognitive_reframe_exclusive` | 26 | L14 (×6) | 11 | 0.0181 |
| `completion_exclusive` | 24 | L14 (×4) | 11 | 0.0114 |
| `analytical_exclusive` | 22 | L14 (×7) | 9 | 0.0171 |
| `canonical_pro_refusal` | 20 | L14 (×5) | 10 | 0.0131 |
| `anti_refusal_amplifiers` | 19 | L14 (×7) | 6 | 0.0127 |
| `jb_analytical_specific_vs_ctrl` | 18 | L14 (×5) | 11 | 0.0300 |
| `jb_cognitive_reframe_specific_vs_ctrl` | 16 | L14 (×5) | 8 | 0.0225 |
| `sign_flip_convergent` | 15 | L13 (×3) | 9 | 0.0308 |
| `jb_fiction_specific_vs_ctrl` | 14 | L14 (×4) | 7 | 0.0179 |
| `jb_completion_specific_vs_ctrl` | 13 | L15 (×3) | 9 | 0.0260 |
| `jb_roleplay_specific_vs_ctrl` | 12 | L1 (×2) | 7 | 0.0276 |
| `ctrl_shared_refusal` | 9 | L0 (×2) | 6 | 0.0127 |
| `late_wave_layer24_32` | 0 | — | 0 | N/A |
| `ctrl_only` | 0 | — | 0 | N/A |

## JB-vs-Ctrl recruitment contrast (NEW — Task 10)

For each JB class, how much of the corpus-level top-50 recruitment is **genuinely JB-semantic** vs. **prefix-induced** (also triggered by the matched benign ctrl prefix)? Old L32 data could not compute this; it's the headline new finding enabled by the 11-condition ctrl-balanced dataset.

| Class | \|jb_top50\| | \|ctrl_top50\| | Intersection | JB-specific | Ctrl-specific | **JB-specific %** | Overlap % |
|---|---|---|---|---|---|---|---|
| `analytical` | 68 | 74 | 50 | 18 | 24 | **26%** | 74% |
| `cognitive_reframe` | 70 | 72 | 54 | 16 | 18 | **23%** | 77% |
| `completion` | 71 | 76 | 58 | 13 | 18 | **18%** | 82% |
| `fiction` | 66 | 72 | 52 | 14 | 20 | **21%** | 79% |
| `roleplay` | 74 | 75 | 62 | 12 | 13 | **16%** | 84% |

**Reading**: `JB-specific %` close to 100 → JB recruits mechanisms the benign prefix does NOT (strong JB-semantic signal). Close to 0 → JB's effect is mostly a prefix-length artifact, not genuine semantic mechanism.


## Subcircuit definitions and top features

### `roleplay_exclusive` — n=49

Features seen in **only** the `roleplay` JB class (no bare, no other JB). Candidates for `roleplay`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L3:F9041` | L3 | 236.490 | 's', 'alanine', ' capitals' |
| `L1:F516` | L1 | 199.723 | 'ب', 'ের', '㑹' |
| `L0:F454` | L0 | 186.203 | '৭', ' covariance', '𝗻' |
| `L7:F220` | L7 | 167.002 | '避免', ' pondered', ' ensuring' |
| `L1:F5158` | L1 | 163.197 | ' fuss', 'ENCE', 'usam' |

### `universal_refusal_core` — n=39

Features seen in **bare + all 5 JB classes**. The canonical refusal core — present in both harmful-alone and every jailbreak. Ablation control baseline.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L0:F144` | L0 | 2833.904 | 'ம்', '이', 'oja' |
| `L0:F66` | L0 | 1047.576 | 'ে', 'יות', 'er' |
| `L10:F111` | L10 | 757.336 | ' Tämä', ' wielu', ' várias' |
| `L0:F460` | L0 | 715.359 | 'quired', 'CH', 'wg' |
| `L13:F471` | L13 | 688.782 | ' Nobel', 'Let', ' hadn' |

### `fiction_exclusive` — n=36

Features seen in **only** the `fiction` JB class (no bare, no other JB). Candidates for `fiction`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L15:F410` | L15 | 279.532 | ' Versions', ' Atau', 'പാട്' |
| `L2:F295` | L2 | 243.842 | 'mě', 'দের', 'iendo' |
| `L3:F1218` | L3 | 223.514 | 'ea', 'een', 'orems' |
| `L14:F467` | L14 | 216.664 | ' gebruikers', ' housewives', ' espectadores' |
| `L3:F428` | L3 | 194.185 | 't', 'ше', 'tela' |

### `dampening_specialists` — n=26

Features in the **dampened** bucket of ≥3 JB classes. Pro-refusal features whose contribution to the refusal direction weakens across most JB types.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L13:F427` | L13 | 923.159 | ' amic', ' Descent', ' Company' |
| `L11:F315` | L11 | 461.124 | '3', 'About', "'" |
| `L13:F419` | L13 | 330.741 | ' silam', ' solução', ' soluzione' |
| `L2:F343` | L2 | 317.416 | 'া', 'ldquo', '样的' |
| `L15:F383` | L15 | 311.893 | ' várias', 'Ан', 'Nope' |

### `cognitive_reframe_exclusive` — n=26

Features seen in **only** the `cognitive_reframe` JB class (no bare, no other JB). Candidates for `cognitive_reframe`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L10:F198` | L10 | 237.200 | 'Tienes', 'Saturation', 'ឡិច' |
| `L14:F598` | L14 | 196.484 | ' Without', ' Although', ' However' |
| `L12:F10313` | L12 | 166.228 | 'L', 'ab', 'av' |
| `L15:F1541` | L15 | 165.605 | ' conductas', ' legally', ' опас' |
| `L0:F290` | L0 | 161.399 | 'l', 'selves', ' supposing' |

### `completion_exclusive` — n=24

Features seen in **only** the `completion` JB class (no bare, no other JB). Candidates for `completion`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L3:F546` | L3 | 376.671 | 't', ' berjudul', ' Diakses' |
| `L3:F13491` | L3 | 235.190 | 'س', 'on', ' patria' |
| `L14:F85` | L14 | 169.583 | ' cappuccino', ' convivial', ' powdery' |
| `L15:F180` | L15 | 166.586 | '自我', '自己', 'തല്ല' |
| `L13:F2820` | L13 | 149.246 | ' فقط', ' dapibus', ' głównie' |

### `analytical_exclusive` — n=22

Features seen in **only** the `analytical` JB class (no bare, no other JB). Candidates for `analytical`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L11:F258` | L11 | 257.443 | ' orientação', ' уйна', ' kammam' |
| `L14:F257` | L14 | 228.277 | ' limits', ' Limits', ' அதற்கு' |
| `L0:F208` | L0 | 199.155 | 'ية', 'entropic', ' गोइंग' |
| `L14:F669` | L14 | 183.888 | ' onders', ' venc', ' straightforward' |
| `L14:F365` | L14 | 176.887 | 'resultados', 'kelijk', 'ቬ' |

### `canonical_pro_refusal` — n=20

Features seen in **all 5 JB classes but NOT bare**. Recruited specifically under jailbreak — interpretable as a shared JB-suppression / pro-refusal response.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L13:F427` | L13 | 923.159 | ' amic', ' Descent', ' Company' |
| `L11:F99` | L11 | 854.103 | '📘', 'unków', 'ಿದ್ದೇನೆ' |
| `L13:F43` | L13 | 746.737 | 'Fol', 'Julia', ' दुर्ग' |
| `L15:F446` | L15 | 660.201 | 'ко', 'kari', 'кции' |
| `L10:F219` | L10 | 468.099 | '迳', ' könnte', ' Tomcat' |

### `anti_refusal_amplifiers` — n=19

Features in the **amplified_anti** bucket of ≥3 JB classes. Anti-refusal features that grow in magnitude across most JB types — the bypass signal.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L13:F43` | L13 | 746.737 | 'Fol', 'Julia', ' दुर्ग' |
| `L0:F460` | L0 | 715.359 | 'quired', 'CH', 'wg' |
| `L13:F471` | L13 | 688.782 | ' Nobel', 'Let', ' hadn' |
| `L13:F51` | L13 | 534.440 | ' vicenda', ' señala', 'elesaian' |
| `L14:F42` | L14 | 431.494 | '琊', ' Adolph', ' establecido' |

### `jb_analytical_specific_vs_ctrl` — n=18

Features in **jb_analytical_top50 − ctrl_analytical_top50**. The cleanest JB-semantic subcircuit for `analytical`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L10:F219` | L10 | 468.099 | '迳', ' könnte', ' Tomcat' |
| `L9:F237` | L9 | 451.193 | ' IAU', ' $(<', 'Downloading' |
| `L7:F141` | L7 | 394.351 | ' niche', ' niches', '++;' |
| `L8:F219` | L8 | 359.860 | 'Sorry', 'نے', ' встреча' |
| `L13:F419` | L13 | 330.741 | ' silam', ' solução', ' soluzione' |

### `jb_cognitive_reframe_specific_vs_ctrl` — n=16

Features in **jb_cognitive_reframe_top50 − ctrl_cognitive_reframe_top50**. The cleanest JB-semantic subcircuit for `cognitive_reframe`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F524` | L14 | 371.051 | ' komp', ' pitfalls', '㹮' |
| `L13:F419` | L13 | 330.741 | ' silam', ' solução', ' soluzione' |
| `L2:F343` | L2 | 317.416 | 'া', 'ldquo', '样的' |
| `L14:F187` | L14 | 292.990 | ' seemingly', ' loosened', ',”' |
| `L15:F132` | L15 | 276.167 | 'forEach', ' ràng', 'iprofloxacin' |

### `sign_flip_convergent` — n=15

Features in the **sign_flipped** bucket of ≥3 JB classes. Robustly reverse attribution sign under JB — the highest-confidence mechanism-change features.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L9:F137` | L9 | 189.848 | ' DeFi', 'ய்டா', '🏟' |
| `L13:F970` | L13 | 133.754 | 'mr', ' ένα', 'Mr' |
| `L10:F439` | L10 | 130.509 | ' calcular', 'Argb', 'Calculate' |
| `L8:F221` | L8 | 110.689 | 'Snippet', 'tooltip', 'твра' |
| `L2:F542` | L2 | 100.280 | '특별시', ' بأن', '그리고' |

### `jb_fiction_specific_vs_ctrl` — n=14

Features in **jb_fiction_top50 − ctrl_fiction_top50**. The cleanest JB-semantic subcircuit for `fiction`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L8:F199` | L8 | 373.516 | ' করির', 'မြင်', 'စျေး' |
| `L14:F187` | L14 | 292.990 | ' seemingly', ' loosened', ',”' |
| `L15:F410` | L15 | 279.532 | ' Versions', ' Atau', 'പാട്' |
| `L2:F295` | L2 | 243.842 | 'mě', 'দের', 'iendo' |
| `L14:F167` | L14 | 239.816 | '้ง', 'DUCTION', 'ware' |

### `jb_completion_specific_vs_ctrl` — n=13

Features in **jb_completion_top50 − ctrl_completion_top50**. The cleanest JB-semantic subcircuit for `completion`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L3:F546` | L3 | 376.671 | 't', ' berjudul', ' Diakses' |
| `L8:F199` | L8 | 373.516 | ' করির', 'မြင်', 'စျေး' |
| `L13:F419` | L13 | 330.741 | ' silam', ' solução', ' soluzione' |
| `L9:F500` | L9 | 265.312 | ' hjust', ' dificuldades', 'hjust' |
| `L3:F13491` | L3 | 235.190 | 'س', 'on', ' patria' |

### `jb_roleplay_specific_vs_ctrl` — n=12

Features in **jb_roleplay_top50 − ctrl_roleplay_top50**. The cleanest JB-semantic subcircuit for `roleplay`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F187` | L14 | 292.990 | ' seemingly', ' loosened', ',”' |
| `L3:F494` | L3 | 247.571 | ' counselling', 'tion', 'tions' |
| `L15:F369` | L15 | 245.349 | ' overtly', 'િલ્', ' gering' |
| `L9:F1080` | L9 | 227.250 | ' Ipsum', ' descrizione', ' femenina' |
| `L9:F349` | L9 | 225.577 | 'Career', 'رمپ', 'และการ' |

### `ctrl_shared_refusal` — n=9

Features in **bare ∩ all 5 ctrl_*_top50** but **NOT in all 5 jb_*_top50**. The prefix-invariant refusal spine: machinery the refusal circuit uses regardless of whether the prefix carries JB-semantics or matched benign content. These are NOT JB-semantic — they define the baseline that survives a long-prefix perturbation without JB intent.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F426` | L14 | 416.053 | ' huống', '!.', ' family' |
| `L15:F383` | L15 | 311.893 | ' várias', 'Ан', 'Nope' |
| `L0:F338` | L0 | 293.983 | 'би', 'e', ' ounce' |
| `L12:F1017` | L12 | 288.933 | 'mensaje', 'viridis', '侷' |
| `L13:F500` | L13 | 261.795 | ' UnwrapRef', 'ৃত্তি', '𒄑' |

### `late_wave_layer24_32` — n=0

All tagged features in layers **24–32** — the JB-impact band identified in A8. Layer-based cross-cut; overlaps other subcircuits.

### `ctrl_only` — n=0

Features in **all 5 ctrl_*_top50** but not in bare or any jb_*_top50. Usually tiny; if non-empty, it signals that matched benign prefixes recruit features neither bare-harmful nor any jailbreak uses — typically benign-content semantic features triggered by the ctrl prefix text itself.

## Pairwise overlap (top 10 by normalized intersection)

Normalized overlap = |A ∩ B| / min(|A|, |B|). High values mean the smaller set is largely contained in the larger. `late_wave_layer24_32` naturally absorbs many.

| A | B | norm. overlap |
|---|---|---|
| `fiction_exclusive` | `jb_fiction_specific_vs_ctrl` | 0.57 |
| `universal_refusal_core` | `ctrl_shared_refusal` | 0.33 |
| `canonical_pro_refusal` | `jb_analytical_specific_vs_ctrl` | 0.33 |
| `universal_refusal_core` | `anti_refusal_amplifiers` | 0.32 |
| `universal_refusal_core` | `dampening_specialists` | 0.31 |
| `completion_exclusive` | `jb_completion_specific_vs_ctrl` | 0.31 |
| `anti_refusal_amplifiers` | `jb_fiction_specific_vs_ctrl` | 0.29 |
| `analytical_exclusive` | `jb_analytical_specific_vs_ctrl` | 0.28 |
| `cognitive_reframe_exclusive` | `jb_cognitive_reframe_specific_vs_ctrl` | 0.25 |
| `canonical_pro_refusal` | `dampening_specialists` | 0.25 |

## Suggested Stage 08 ablation targets (causal-impact order)

1. `canonical_pro_refusal` — JB-specific pro-refusal recruitment. Ablation should *strengthen* JB bypass (removes the JB-only refusal boost).
2. `jb_{cls}_specific_vs_ctrl` (per class) — the cleanest per-class JB-semantic mechanism. Ablating one should selectively restore ctrl-like behavior on that class (dissociation test).
3. `sign_flip_convergent` — robust direction reversals. Ablation should partially restore bare behavior under JB.
4. `dampening_specialists` — weakened pro-refusal features. Restoring them to bare strength should counter fiction/analytical bypass.
5. `anti_refusal_amplifiers` — JB-amplified bypass signal. Suppressing them should increase refusal under JB.
6. `ctrl_shared_refusal` — the prefix-invariant spine. Ablation should break refusal on BOTH ctrl and bare — a negative control proving these aren't JB-specific.
7. `universal_refusal_core` — shared baseline. Ablation should break refusal on bare *and* JB (control — proves the subcircuits matter).

