# Subcircuits Report (Rule-Based)

Each subcircuit is defined by a precise set-logic rule over the features observed across bare + 5 JB classes (original rules) and bare + 5 jb_* + 5 ctrl_* conditions (new Apr 22 ctrl-aware rules). No ML fitting — fully interpretable.

## Summary table

| Subcircuit | Size | Peak layer | n_layers occupied | Mean act. freq. |
|---|---|---|---|---|
| `roleplay_exclusive` | 203 | L7 (×21) | 15 | 0.0068 |
| `sign_flip_convergent` | 160 | L14 (×36) | 14 | 0.0110 |
| `cognitive_reframe_exclusive` | 128 | L13 (×22) | 15 | 0.0064 |
| `universal_refusal_core` | 116 | L14 (×23) | 14 | 0.0050 |
| `jb_cognitive_reframe_specific_vs_ctrl` | 88 | L14 (×13) | 13 | 0.0031 |
| `fiction_exclusive` | 86 | L14 (×13) | 15 | 0.0057 |
| `completion_exclusive` | 83 | L13 (×12) | 14 | 0.0065 |
| `dampening_specialists` | 77 | L14 (×19) | 10 | 0.0065 |
| `analytical_exclusive` | 76 | L13 (×11) | 13 | 0.0073 |
| `jb_analytical_specific_vs_ctrl` | 69 | L14 (×14) | 14 | 0.0057 |
| `jb_fiction_specific_vs_ctrl` | 52 | L14 (×13) | 11 | 0.0043 |
| `ctrl_shared_refusal` | 50 | L14 (×15) | 12 | 0.0056 |
| `canonical_pro_refusal` | 40 | L14 (×7) | 11 | 0.0146 |
| `jb_roleplay_specific_vs_ctrl` | 35 | L5 (×7) | 9 | 0.0046 |
| `jb_completion_specific_vs_ctrl` | 32 | L11 (×6) | 14 | 0.0033 |
| `anti_refusal_amplifiers` | 10 | L14 (×4) | 5 | 0.0047 |
| `ctrl_only` | 1 | L14 (×1) | 1 | 0.0025 |
| `late_wave_layer24_32` | 0 | — | 0 | N/A |

## JB-vs-Ctrl recruitment contrast (NEW — Task 10)

For each JB class, how much of the corpus-level top-50 recruitment is **genuinely JB-semantic** vs. **prefix-induced** (also triggered by the matched benign ctrl prefix)? Old L32 data could not compute this; it's the headline new finding enabled by the 11-condition ctrl-balanced dataset.

| Class | \|jb_top50\| | \|ctrl_top50\| | Intersection | JB-specific | Ctrl-specific | **JB-specific %** | Overlap % |
|---|---|---|---|---|---|---|---|
| `analytical` | 202 | 188 | 133 | 69 | 55 | **34%** | 66% |
| `cognitive_reframe` | 228 | 190 | 140 | 88 | 50 | **39%** | 61% |
| `completion` | 174 | 183 | 142 | 32 | 41 | **18%** | 82% |
| `fiction` | 152 | 172 | 100 | 52 | 72 | **34%** | 66% |
| `roleplay` | 175 | 187 | 140 | 35 | 47 | **20%** | 80% |

**Reading**: `JB-specific %` close to 100 → JB recruits mechanisms the benign prefix does NOT (strong JB-semantic signal). Close to 0 → JB's effect is mostly a prefix-length artifact, not genuine semantic mechanism.


## Subcircuit definitions and top features

### `roleplay_exclusive` — n=203

Features seen in **only** the `roleplay` JB class (no bare, no other JB). Candidates for `roleplay`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L13:F1599` | L13 | 0.025 | ' commemoration', ' rosso', ' tuxedo' |
| `L14:F2768` | L14 | 0.022 | ' directly', ' overtly', ' 직접' |
| `L8:F11998` | L8 | 0.018 | ' unstable', 'collapse', 'soluble' |
| `L5:F1533` | L5 | 0.016 | '一个', 'dwell', 'waitFor' |
| `L9:F15908` | L9 | 0.014 | '傚', ' щодо', ' rendez' |

### `sign_flip_convergent` — n=160

Features in the **sign_flipped** bucket of ≥3 JB classes. Robustly reverse attribution sign under JB — the highest-confidence mechanism-change features.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F566` | L14 | 0.063 | ' gimm', ' अनावश्यक', 'AppendLine' |
| `L14:F14772` | L14 | 0.040 | '‘’', 'intent', ' menghilangkan' |
| `L11:F6224` | L11 | 0.039 | 'ostrat', 'prefixes', ' ഒറ്റ' |
| `L11:F2952` | L11 | 0.030 | 'ijnen', ' attacker', ' inverno' |
| `L14:F15763` | L14 | 0.030 | ' advice', ' Advice', ' guidance' |

### `cognitive_reframe_exclusive` — n=128

Features seen in **only** the `cognitive_reframe` JB class (no bare, no other JB). Candidates for `cognitive_reframe`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L7:F7418` | L7 | 0.030 | ' techniques', ' spoonful', ' biasa' |
| `L14:F2816` | L14 | 0.028 | ' immoral', ' unethical', ' morality' |
| `L9:F1849` | L9 | 0.027 | ' አስፈላጊ', ' maddenin', 'inairement' |
| `L13:F16236` | L13 | 0.026 | 'ではありません', ' نمی', ' نمی\u200c' |
| `L9:F12337` | L9 | 0.026 | ' FXMLLoader', ' contag', ' emocional' |

### `universal_refusal_core` — n=116

Features seen in **bare + all 5 JB classes**. The canonical refusal core — present in both harmful-alone and every jailbreak. Ablation control baseline.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F4933` | L14 | 0.362 | 'ésus', ' щоб', ' conformément' |
| `L13:F427` | L13 | 0.344 | ' amic', ' Descent', ' Company' |
| `L14:F619` | L14 | 0.324 | ' polymorphism', ' 많이', ' попробовать' |
| `L14:F480` | L14 | 0.216 | ' محدود', ' overhauled', ' cromosoma' |
| `L14:F189` | L14 | 0.209 | '然而', ' illicit', 'Illegal' |

### `jb_cognitive_reframe_specific_vs_ctrl` — n=88

Features in **jb_cognitive_reframe_top50 − ctrl_cognitive_reframe_top50**. The cleanest JB-semantic subcircuit for `cognitive_reframe`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L13:F11240` | L13 | 0.072 | 'arganya', ' שה', ' whereupon' |
| `L5:F13142` | L5 | 0.068 | 'ر', 'ेच्छा', ' bulat' |
| `L10:F3865` | L10 | 0.035 | 'Facade', ' hide', 'Hide' |
| `L8:F10675` | L8 | 0.034 | 'Stats', 'get', 'vict' |
| `L8:F7110` | L8 | 0.032 | 'اهده', 'летним', 'Scenic' |

### `fiction_exclusive` — n=86

Features seen in **only** the `fiction` JB class (no bare, no other JB). Candidates for `fiction`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F6744` | L14 | 0.025 | ' bay', 'fram', ' goog' |
| `L11:F3404` | L11 | 0.023 | ' प्रेमी', ' sufferers', ' перева' |
| `L14:F1753` | L14 | 0.023 | ' tedious', ' questionnaires', ' বেশ' |
| `L14:F11754` | L14 | 0.023 | 'শ্বাস', ' शस्त्र', ' saker' |
| `L13:F11024` | L13 | 0.020 | 'ඃ', 'etimes', 'ឹក' |

### `completion_exclusive` — n=83

Features seen in **only** the `completion` JB class (no bare, no other JB). Candidates for `completion`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L11:F5644` | L11 | 0.035 | 'cloud', ' cloud', 'builder' |
| `L10:F7222` | L10 | 0.021 | '리와', ' đẹp', ' Enthusi' |
| `L14:F7968` | L14 | 0.021 | ' própria', ' próprias', '쯔' |
| `L11:F5643` | L11 | 0.018 | ' L', ' Szcz', ' Fort' |
| `L14:F8186` | L14 | 0.017 | '手册', ' Handbook', ' handbook' |

### `dampening_specialists` — n=77

Features in the **dampened** bucket of ≥3 JB classes. Pro-refusal features whose contribution to the refusal direction weakens across most JB types.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F4933` | L14 | 0.362 | 'ésus', ' щоб', ' conformément' |
| `L13:F427` | L13 | 0.344 | ' amic', ' Descent', ' Company' |
| `L14:F619` | L14 | 0.324 | ' polymorphism', ' 많이', ' попробовать' |
| `L14:F189` | L14 | 0.209 | '然而', ' illicit', 'Illegal' |
| `L13:F4722` | L13 | 0.169 | ' nefarious', ' bastard', ' racist' |

### `analytical_exclusive` — n=76

Features seen in **only** the `analytical` JB class (no bare, no other JB). Candidates for `analytical`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F11329` | L14 | 0.025 | 'Attempt', ' attempted', 'attempts' |
| `L11:F14883` | L11 | 0.023 | 'fake', ' misinformation', 'PB' |
| `L12:F7372` | L12 | 0.023 | '答え', ' answer', ' jawaban' |
| `L11:F15217` | L11 | 0.022 | 'หรือไม่', 'putBoolean', '能不能' |
| `L8:F4457` | L8 | 0.022 | ' whiskey', ' drunken', ' alcohol' |

### `jb_analytical_specific_vs_ctrl` — n=69

Features in **jb_analytical_top50 − ctrl_analytical_top50**. The cleanest JB-semantic subcircuit for `analytical`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L5:F13142` | L5 | 0.068 | 'ر', 'ेच्छा', ' bulat' |
| `L14:F2372` | L14 | 0.034 | ' spiking', ' অনিবার্য', ' fração' |
| `L11:F10284` | L11 | 0.028 | 'abstract', 'Generic', 'формация' |
| `L12:F15468` | L12 | 0.026 | ' intermission', ' initialized', 'ηση' |
| `L11:F3575` | L11 | 0.026 | ' itr', ' buvo', ' toolbar' |

### `jb_fiction_specific_vs_ctrl` — n=52

Features in **jb_fiction_top50 − ctrl_fiction_top50**. The cleanest JB-semantic subcircuit for `fiction`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L5:F13142` | L5 | 0.068 | 'ر', 'ेच्छा', ' bulat' |
| `L11:F11519` | L11 | 0.050 | '্যাথ', 'estamps', ' équipes' |
| `L13:F6386` | L13 | 0.037 | ' молит', 'decrement', 'ظ' |
| `L8:F360` | L8 | 0.031 | 'මු', '๎', ' fla' |
| `L10:F9878` | L10 | 0.029 | 'osexuality', ' pornography', 'Sexual' |

### `ctrl_shared_refusal` — n=50

Features in **bare ∩ all 5 ctrl_*_top50** but **NOT in all 5 jb_*_top50**. The prefix-invariant refusal spine: machinery the refusal circuit uses regardless of whether the prefix carries JB-semantics or matched benign content. These are NOT JB-semantic — they define the baseline that survives a long-prefix perturbation without JB intent.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L13:F5745` | L13 | 0.073 | ' any', ' qualquer', ' żad' |
| `L14:F785` | L14 | 0.070 | ' אין', ' decept', 'markets' |
| `L14:F2590` | L14 | 0.068 | ' create', ' depict', ' surviving' |
| `L13:F2820` | L13 | 0.057 | ' فقط', ' dapibus', ' głównie' |
| `L13:F13354` | L13 | 0.050 | ' snowfall', ' antibiotics', ' Christmas' |

### `canonical_pro_refusal` — n=40

Features seen in **all 5 JB classes but NOT bare**. Recruited specifically under jailbreak — interpretable as a shared JB-suppression / pro-refusal response.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L12:F257` | L12 | 0.052 | ' sap', ' bu', ' semantics' |
| `L14:F15763` | L14 | 0.030 | ' advice', ' Advice', ' guidance' |
| `L13:F653` | L13 | 0.014 | 'Neces', 'needs', ' postura' |
| `L14:F15531` | L14 | 0.013 | ' abuses', ' abuse', ' insults' |
| `L14:F3238` | L14 | 0.012 | ' アク', ' pr', '一切' |

### `jb_roleplay_specific_vs_ctrl` — n=35

Features in **jb_roleplay_top50 − ctrl_roleplay_top50**. The cleanest JB-semantic subcircuit for `roleplay`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F3699` | L14 | 0.055 | ' suicide', ' Suicide', ' suicidal' |
| `L11:F155` | L11 | 0.032 | ' praw', ' साजिश', ' stran' |
| `L8:F360` | L8 | 0.031 | 'මු', '๎', ' fla' |
| `L11:F2952` | L11 | 0.030 | 'ijnen', ' attacker', ' inverno' |
| `L10:F733` | L10 | 0.026 | ' copyrights', ' divulge', ' infring' |

### `jb_completion_specific_vs_ctrl` — n=32

Features in **jb_completion_top50 − ctrl_completion_top50**. The cleanest JB-semantic subcircuit for `completion`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L11:F11519` | L11 | 0.050 | '্যাথ', 'estamps', ' équipes' |
| `L13:F6386` | L13 | 0.037 | ' молит', 'decrement', 'ظ' |
| `L11:F5644` | L11 | 0.035 | 'cloud', ' cloud', 'builder' |
| `L14:F2372` | L14 | 0.034 | ' spiking', ' অনিবার্য', ' fração' |
| `L9:F1080` | L9 | 0.033 | ' Ipsum', ' descrizione', ' femenina' |

### `anti_refusal_amplifiers` — n=10

Features in the **amplified_anti** bucket of ≥3 JB classes. Anti-refusal features that grow in magnitude across most JB types — the bypass signal.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F480` | L14 | 0.216 | ' محدود', ' overhauled', ' cromosoma' |
| `L13:F43` | L13 | 0.068 | 'Fol', 'Julia', ' दुर्ग' |
| `L14:F2590` | L14 | 0.068 | ' create', ' depict', ' surviving' |
| `L13:F3426` | L13 | 0.049 | ' Вар', ' burning', ' remembrance' |
| `L13:F1254` | L13 | 0.029 | ' različ', ' différentes', ' various' |

### `ctrl_only` — n=1

Features in **all 5 ctrl_*_top50** but not in bare or any jb_*_top50. Usually tiny; if non-empty, it signals that matched benign prefixes recruit features neither bare-harmful nor any jailbreak uses — typically benign-content semantic features triggered by the ctrl prefix text itself.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L14:F7968` | L14 | 0.021 | ' própria', ' próprias', '쯔' |

### `late_wave_layer24_32` — n=0

All tagged features in layers **24–32** — the JB-impact band identified in A8. Layer-based cross-cut; overlaps other subcircuits.

## Pairwise overlap (top 10 by normalized intersection)

Normalized overlap = |A ∩ B| / min(|A|, |B|). High values mean the smaller set is largely contained in the larger. `late_wave_layer24_32` naturally absorbs many.

| A | B | norm. overlap |
|---|---|---|
| `completion_exclusive` | `ctrl_only` | 1.00 |
| `canonical_pro_refusal` | `sign_flip_convergent` | 0.97 |
| `universal_refusal_core` | `dampening_specialists` | 0.82 |
| `jb_cognitive_reframe_specific_vs_ctrl` | `jb_roleplay_specific_vs_ctrl` | 0.51 |
| `universal_refusal_core` | `anti_refusal_amplifiers` | 0.50 |
| `fiction_exclusive` | `jb_fiction_specific_vs_ctrl` | 0.48 |
| `jb_analytical_specific_vs_ctrl` | `jb_completion_specific_vs_ctrl` | 0.47 |
| `universal_refusal_core` | `ctrl_shared_refusal` | 0.32 |
| `dampening_specialists` | `ctrl_shared_refusal` | 0.32 |
| `analytical_exclusive` | `jb_analytical_specific_vs_ctrl` | 0.32 |

## Suggested Stage 08 ablation targets (causal-impact order)

1. `canonical_pro_refusal` — JB-specific pro-refusal recruitment. Ablation should *strengthen* JB bypass (removes the JB-only refusal boost).
2. `jb_{cls}_specific_vs_ctrl` (per class) — the cleanest per-class JB-semantic mechanism. Ablating one should selectively restore ctrl-like behavior on that class (dissociation test).
3. `sign_flip_convergent` — robust direction reversals. Ablation should partially restore bare behavior under JB.
4. `dampening_specialists` — weakened pro-refusal features. Restoring them to bare strength should counter fiction/analytical bypass.
5. `anti_refusal_amplifiers` — JB-amplified bypass signal. Suppressing them should increase refusal under JB.
6. `ctrl_shared_refusal` — the prefix-invariant spine. Ablation should break refusal on BOTH ctrl and bare — a negative control proving these aren't JB-specific.
7. `universal_refusal_core` — shared baseline. Ablation should break refusal on bare *and* JB (control — proves the subcircuits matter).

