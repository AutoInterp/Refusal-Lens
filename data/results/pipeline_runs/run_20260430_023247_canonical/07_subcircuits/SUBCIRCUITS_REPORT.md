# Subcircuits Report (Rule-Based)

Each subcircuit is defined by a precise set-logic rule over the features observed across bare + 5 JB classes (original rules) and bare + 5 jb_* + 5 ctrl_* conditions (new Apr 22 ctrl-aware rules). No ML fitting — fully interpretable.

## Summary table

| Subcircuit | Size | Peak layer | n_layers occupied | Mean act. freq. |
|---|---|---|---|---|
| `sign_flip_convergent` | 155 | L14 (×35) | 15 | 0.0229 |
| `roleplay_exclusive` | 140 | L13 (×19) | 16 | 0.0113 |
| `dampening_specialists` | 122 | L14 (×24) | 14 | 0.0197 |
| `universal_refusal_core` | 98 | L14 (×24) | 15 | 0.0160 |
| `canonical_pro_refusal` | 88 | L11 (×12) | 15 | 0.0235 |
| `completion_exclusive` | 85 | L13 (×12) | 16 | 0.0105 |
| `cognitive_reframe_exclusive` | 80 | L14 (×22) | 15 | 0.0127 |
| `fiction_exclusive` | 77 | L14 (×11) | 16 | 0.0104 |
| `anti_refusal_amplifiers` | 64 | L14 (×18) | 11 | 0.0169 |
| `analytical_exclusive` | 53 | L13 (×11) | 13 | 0.0120 |
| `jb_fiction_specific_vs_ctrl` | 37 | L14 (×9) | 10 | 0.0155 |
| `jb_roleplay_specific_vs_ctrl` | 36 | L14 (×7) | 12 | 0.0168 |
| `jb_analytical_specific_vs_ctrl` | 32 | L14 (×11) | 10 | 0.0221 |
| `jb_cognitive_reframe_specific_vs_ctrl` | 30 | L14 (×8) | 9 | 0.0261 |
| `ctrl_shared_refusal` | 25 | L14 (×5) | 11 | 0.0200 |
| `jb_completion_specific_vs_ctrl` | 15 | L15 (×5) | 8 | 0.0180 |
| `late_wave_layer24_32` | 0 | — | 0 | N/A |
| `ctrl_only` | 0 | — | 0 | N/A |

## JB-vs-Ctrl recruitment contrast (NEW — Task 10)

For each JB class, how much of the corpus-level top-50 recruitment is **genuinely JB-semantic** vs. **prefix-induced** (also triggered by the matched benign ctrl prefix)? Old L32 data could not compute this; it's the headline new finding enabled by the 11-condition ctrl-balanced dataset.

| Class | \|jb_top50\| | \|ctrl_top50\| | Intersection | JB-specific | Ctrl-specific | **JB-specific %** | Overlap % |
|---|---|---|---|---|---|---|---|
| `analytical` | 112 | 132 | 80 | 32 | 52 | **29%** | 71% |
| `cognitive_reframe` | 124 | 122 | 94 | 30 | 28 | **24%** | 76% |
| `completion` | 119 | 137 | 104 | 15 | 33 | **13%** | 87% |
| `fiction` | 113 | 126 | 76 | 37 | 50 | **33%** | 67% |
| `roleplay` | 137 | 127 | 101 | 36 | 26 | **26%** | 74% |

**Reading**: `JB-specific %` close to 100 → JB recruits mechanisms the benign prefix does NOT (strong JB-semantic signal). Close to 0 → JB's effect is mostly a prefix-length artifact, not genuine semantic mechanism.


## Subcircuit definitions and top features

### `sign_flip_convergent` — n=155

Features in the **sign_flipped** bucket of ≥3 JB classes. Robustly reverse attribution sign under JB — the highest-confidence mechanism-change features.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F178` | L14 | 650.650 | ' fruitless', ' hojas', 'astotal' |
| `L14:F524` | L14 | 529.200 | ' komp', ' pitfalls', '㹮' |
| `L14:F566` | L14 | 348.337 | ' gimm', ' अनावश्यक', 'AppendLine' |
| `L13:F500` | L13 | 341.058 | ' UnwrapRef', 'ৃত্তি', '𒄑' |
| `L10:F234` | L10 | 310.379 | ' möglichst', ' absorbance', ' voltages' |

### `roleplay_exclusive` — n=140

Features seen in **only** the `roleplay` JB class (no bare, no other JB). Candidates for `roleplay`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L13:F1599` | L13 | 256.423 | ' commemoration', ' rosso', ' tuxedo' |
| `L1:F516` | L1 | 199.723 | 'ب', 'ের', '㑹' |
| `L1:F5158` | L1 | 187.433 | ' fuss', 'ENCE', 'usam' |
| `L15:F1165` | L15 | 180.585 | ' scrumptious', ' comfy', ' nyaman' |
| `L3:F301` | L3 | 175.783 | 'нде', ' toán', 'মনসিংহ' |

### `dampening_specialists` — n=122

Features in the **dampened** bucket of ≥3 JB classes. Pro-refusal features whose contribution to the refusal direction weakens across most JB types.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L13:F427` | L13 | 987.291 | ' amic', ' Descent', ' Company' |
| `L14:F178` | L14 | 650.650 | ' fruitless', ' hojas', 'astotal' |
| `L14:F524` | L14 | 529.200 | ' komp', ' pitfalls', '㹮' |
| `L11:F315` | L11 | 501.452 | '3', 'About', "'" |
| `L13:F419` | L13 | 392.199 | ' silam', ' solução', ' soluzione' |

### `universal_refusal_core` — n=98

Features seen in **bare + all 5 JB classes**. The canonical refusal core — present in both harmful-alone and every jailbreak. Ablation control baseline.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L0:F144` | L0 | 3021.694 | 'ம்', '이', 'oja' |
| `L0:F66` | L0 | 1108.372 | 'ে', 'יות', 'er' |
| `L13:F427` | L13 | 987.291 | ' amic', ' Descent', ' Company' |
| `L13:F43` | L13 | 827.943 | 'Fol', 'Julia', ' दुर्ग' |
| `L13:F471` | L13 | 789.785 | ' Nobel', 'Let', ' hadn' |

### `canonical_pro_refusal` — n=88

Features seen in **all 5 JB classes but NOT bare**. Recruited specifically under jailbreak — interpretable as a shared JB-suppression / pro-refusal response.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L11:F99` | L11 | 868.505 | '📘', 'unków', 'ಿದ್ದೇನೆ' |
| `L15:F446` | L15 | 703.379 | 'ко', 'kari', 'кции' |
| `L10:F219` | L10 | 495.312 | '迳', ' könnte', ' Tomcat' |
| `L9:F237` | L9 | 468.193 | ' IAU', ' $(<', 'Downloading' |
| `L7:F141` | L7 | 408.989 | ' niche', ' niches', '++;' |

### `completion_exclusive` — n=85

Features seen in **only** the `completion` JB class (no bare, no other JB). Candidates for `completion`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L11:F6997` | L11 | 282.717 | '_______________', ' _____', ' ________' |
| `L9:F11835` | L9 | 204.020 | ' deprec', ' ferram', '😷' |
| `L3:F7886` | L3 | 201.804 | 'posium', 'ים', 'ер' |
| `L15:F4325` | L15 | 164.654 | ' HMRC', ' recognisable', ' fridge' |
| `L15:F34` | L15 | 161.338 | ' tarmac', ' stort', ' mediums' |

### `cognitive_reframe_exclusive` — n=80

Features seen in **only** the `cognitive_reframe` JB class (no bare, no other JB). Candidates for `cognitive_reframe`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L11:F5805` | L11 | 277.303 | ' відповід', ' süd', ' öst' |
| `L15:F632` | L15 | 202.578 | ' fraudulent', 'драт', ' सें' |
| `L11:F3229` | L11 | 163.029 | 'Jenis', 'Joined', 'Activity' |
| `L14:F7153` | L14 | 157.519 | ' बचाव', ' anti', 'Anti' |
| `L14:F11639` | L14 | 156.150 | ' üzerine', 'ből', ' 엔진' |

### `fiction_exclusive` — n=77

Features seen in **only** the `fiction` JB class (no bare, no other JB). Candidates for `fiction`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L15:F410` | L15 | 321.971 | ' Versions', ' Atau', 'പാട്' |
| `L2:F295` | L2 | 288.555 | 'mě', 'দের', 'iendo' |
| `L11:F4093` | L11 | 276.008 | '使用的', ' anvä', 'Sho' |
| `L3:F1218` | L3 | 258.898 | 'ea', 'een', 'orems' |
| `L3:F428` | L3 | 240.698 | 't', 'ше', 'tela' |

### `anti_refusal_amplifiers` — n=64

Features in the **amplified_anti** bucket of ≥3 JB classes. Anti-refusal features that grow in magnitude across most JB types — the bypass signal.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L0:F66` | L0 | 1108.372 | 'ে', 'יות', 'er' |
| `L13:F43` | L13 | 827.943 | 'Fol', 'Julia', ' दुर्ग' |
| `L13:F471` | L13 | 789.785 | ' Nobel', 'Let', ' hadn' |
| `L0:F460` | L0 | 784.006 | 'quired', 'CH', 'wg' |
| `L15:F107` | L15 | 591.443 | ' unspecified', 'Square', ' relevant' |

### `analytical_exclusive` — n=53

Features seen in **only** the `analytical` JB class (no bare, no other JB). Candidates for `analytical`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L0:F208` | L0 | 257.036 | 'ية', 'entropic', ' गोइंग' |
| `L11:F11472` | L11 | 248.284 | 'Fighting', ' संदर्भित', ' обознача' |
| `L15:F238` | L15 | 240.600 | ' morti', ' kuvvet', ' businessmen' |
| `L13:F7050` | L13 | 230.377 | ' deoarece', 'Broken', ' çünkü' |
| `L15:F365` | L15 | 194.750 | ' caso', ' bataille', ' Как' |

### `jb_fiction_specific_vs_ctrl` — n=37

Features in **jb_fiction_top50 − ctrl_fiction_top50**. The cleanest JB-semantic subcircuit for `fiction`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L15:F410` | L15 | 321.971 | ' Versions', ' Atau', 'പാട്' |
| `L14:F167` | L14 | 320.643 | '้ง', 'DUCTION', 'ware' |
| `L2:F295` | L2 | 288.555 | 'mě', 'দের', 'iendo' |
| `L11:F4093` | L11 | 276.008 | '使用的', ' anvä', 'Sho' |
| `L14:F467` | L14 | 271.232 | ' gebruikers', ' housewives', ' espectadores' |

### `jb_roleplay_specific_vs_ctrl` — n=36

Features in **jb_roleplay_top50 − ctrl_roleplay_top50**. The cleanest JB-semantic subcircuit for `roleplay`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L8:F199` | L8 | 450.648 | ' করির', 'မြင်', 'စျေး' |
| `L14:F257` | L14 | 319.936 | ' limits', ' Limits', ' அதற்கு' |
| `L14:F785` | L14 | 297.334 | ' אין', ' decept', 'markets' |
| `L13:F442` | L13 | 294.423 | ' materiales', '\uf4a0', ' frutas' |
| `L11:F155` | L11 | 260.536 | ' praw', ' साजिश', ' stran' |

### `jb_analytical_specific_vs_ctrl` — n=32

Features in **jb_analytical_top50 − ctrl_analytical_top50**. The cleanest JB-semantic subcircuit for `analytical`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L11:F258` | L11 | 329.122 | ' orientação', ' уйна', ' kammam' |
| `L14:F464` | L14 | 321.365 | 'िकास', 'াকে', 'გან' |
| `L0:F208` | L0 | 257.036 | 'ية', 'entropic', ' गोइंग' |
| `L15:F180` | L15 | 249.203 | '自我', '自己', 'തല്ല' |
| `L11:F11472` | L11 | 248.284 | 'Fighting', ' संदर्भित', ' обознача' |

### `jb_cognitive_reframe_specific_vs_ctrl` — n=30

Features in **jb_cognitive_reframe_top50 − ctrl_cognitive_reframe_top50**. The cleanest JB-semantic subcircuit for `cognitive_reframe`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L8:F199` | L8 | 450.648 | ' করির', 'မြင်', 'စျေး' |
| `L11:F258` | L11 | 329.122 | ' orientação', ' уйна', ' kammam' |
| `L14:F464` | L14 | 321.365 | 'िकास', 'াকে', 'გან' |
| `L13:F442` | L13 | 294.423 | ' materiales', '\uf4a0', ' frutas' |
| `L11:F5805` | L11 | 277.303 | ' відповід', ' süd', ' öst' |

### `ctrl_shared_refusal` — n=25

Features in **bare ∩ all 5 ctrl_*_top50** but **NOT in all 5 jb_*_top50**. The prefix-invariant refusal spine: machinery the refusal circuit uses regardless of whether the prefix carries JB-semantics or matched benign content. These are NOT JB-semantic — they define the baseline that survives a long-prefix perturbation without JB intent.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F178` | L14 | 650.650 | ' fruitless', ' hojas', 'astotal' |
| `L14:F524` | L14 | 529.200 | ' komp', ' pitfalls', '㹮' |
| `L2:F343` | L2 | 372.324 | 'া', 'ldquo', '样的' |
| `L12:F1017` | L12 | 364.338 | 'mensaje', 'viridis', '侷' |
| `L15:F383` | L15 | 353.112 | ' várias', 'Ан', 'Nope' |

### `jb_completion_specific_vs_ctrl` — n=15

Features in **jb_completion_top50 − ctrl_completion_top50**. The cleanest JB-semantic subcircuit for `completion`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L8:F199` | L8 | 450.648 | ' করির', 'မြင်', 'စျေး' |
| `L12:F66` | L12 | 298.096 | ' blueberries', ' frustrating', ' entom' |
| `L3:F13491` | L3 | 292.642 | 'س', 'on', ' patria' |
| `L11:F6997` | L11 | 282.717 | '_______________', ' _____', ' ________' |
| `L15:F180` | L15 | 249.203 | '自我', '自己', 'തല്ല' |

### `late_wave_layer24_32` — n=0

All tagged features in layers **24–32** — the JB-impact band identified in A8. Layer-based cross-cut; overlaps other subcircuits.

### `ctrl_only` — n=0

Features in **all 5 ctrl_*_top50** but not in bare or any jb_*_top50. Usually tiny; if non-empty, it signals that matched benign prefixes recruit features neither bare-harmful nor any jailbreak uses — typically benign-content semantic features triggered by the ctrl prefix text itself.

## Pairwise overlap (top 10 by normalized intersection)

Normalized overlap = |A ∩ B| / min(|A|, |B|). High values mean the smaller set is largely contained in the larger. `late_wave_layer24_32` naturally absorbs many.

| A | B | norm. overlap |
|---|---|---|
| `universal_refusal_core` | `ctrl_shared_refusal` | 0.68 |
| `canonical_pro_refusal` | `sign_flip_convergent` | 0.56 |
| `fiction_exclusive` | `jb_fiction_specific_vs_ctrl` | 0.54 |
| `universal_refusal_core` | `dampening_specialists` | 0.46 |
| `dampening_specialists` | `ctrl_shared_refusal` | 0.44 |
| `dampening_specialists` | `jb_roleplay_specific_vs_ctrl` | 0.42 |
| `analytical_exclusive` | `jb_analytical_specific_vs_ctrl` | 0.41 |
| `universal_refusal_core` | `anti_refusal_amplifiers` | 0.38 |
| `canonical_pro_refusal` | `dampening_specialists` | 0.34 |
| `universal_refusal_core` | `jb_roleplay_specific_vs_ctrl` | 0.31 |

## Suggested Stage 08 ablation targets (causal-impact order)

1. `canonical_pro_refusal` — JB-specific pro-refusal recruitment. Ablation should *strengthen* JB bypass (removes the JB-only refusal boost).
2. `jb_{cls}_specific_vs_ctrl` (per class) — the cleanest per-class JB-semantic mechanism. Ablating one should selectively restore ctrl-like behavior on that class (dissociation test).
3. `sign_flip_convergent` — robust direction reversals. Ablation should partially restore bare behavior under JB.
4. `dampening_specialists` — weakened pro-refusal features. Restoring them to bare strength should counter fiction/analytical bypass.
5. `anti_refusal_amplifiers` — JB-amplified bypass signal. Suppressing them should increase refusal under JB.
6. `ctrl_shared_refusal` — the prefix-invariant spine. Ablation should break refusal on BOTH ctrl and bare — a negative control proving these aren't JB-specific.
7. `universal_refusal_core` — shared baseline. Ablation should break refusal on bare *and* JB (control — proves the subcircuits matter).

