# Subcircuits Report (Rule-Based)

Each subcircuit is defined by a precise set-logic rule over the features observed across bare + 5 JB classes. No ML fitting — fully interpretable.

## Summary table

| Subcircuit | Size | Peak layer | n_layers occupied | Mean act. freq. |
|---|---|---|---|---|
| `late_wave_layer24_32` | 689 | L30 (×115) | 9 | 0.0060 |
| `sign_flip_convergent` | 179 | L30 (×34) | 17 | 0.0072 |
| `roleplay_exclusive` | 104 | L30 (×14) | 18 | 0.0051 |
| `fiction_exclusive` | 97 | L28 (×14) | 19 | 0.0060 |
| `universal_refusal_core` | 83 | L29 (×9) | 16 | 0.0045 |
| `cognitive_reframe_exclusive` | 77 | L30 (×9) | 19 | 0.0060 |
| `completion_exclusive` | 68 | L31 (×10) | 15 | 0.0074 |
| `canonical_pro_refusal` | 56 | L32 (×12) | 16 | 0.0057 |
| `analytical_exclusive` | 54 | L26 (×7) | 18 | 0.0046 |
| `dampening_specialists` | 52 | L30 (×8) | 13 | 0.0031 |
| `anti_refusal_amplifiers` | 50 | L25 (×8) | 14 | 0.0078 |

## Subcircuit definitions and top features

### `late_wave_layer24_32` — n=689

All tagged features in layers **24–32** — the JB-impact band identified in A8. Layer-based cross-cut; overlaps other subcircuits.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L29:F1066` | L29 | 8.037 | ' لیک', ' carr', ' evolved' |
| `L24:F1304` | L24 | 4.394 | 'ureau', ' sentidos', 'anea' |
| `L24:F107` | L24 | 3.881 | ' ok', ' okay', ' OK' |
| `L25:F963` | L25 | 3.838 | '級', 'ận', 'omeres' |
| `L26:F319` | L26 | 3.750 | ' ho', ' хе', ' hur' |

### `sign_flip_convergent` — n=179

Features in the **sign_flipped** bucket of ≥3 JB classes. Robustly reverse attribution sign under JB — the highest-confidence mechanism-change features.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L29:F10018` | L29 | 1.277 | ' Sp', ' sp', 'Sp' |
| `L31:F153` | L31 | 1.215 | ' ullamcorper', '֥', ' an' |
| `L25:F663` | L25 | 1.133 | ' I', 'I', ' আমি' |
| `L22:F143` | L22 | 1.128 | ' by', 'hi', ' closed' |
| `L32:F295` | L32 | 0.894 | 'aadhar', 'aldehyde', 'nemouth' |

### `roleplay_exclusive` — n=104

Features seen in **only** the `roleplay` JB class (no bare, no other JB). Candidates for `roleplay`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L28:F7691` | L28 | 0.689 | 'ba', 'দৃ', ' শীর্ষ' |
| `L29:F2550` | L29 | 0.679 | ' Many', 'many', ' many' |
| `L26:F2203` | L26 | 0.662 | ' ok', 'ipkan', ' okay' |
| `L20:F360` | L20 | 0.496 | ' wow', 'wow', ' sorry' |
| `L26:F5310` | L26 | 0.405 | 'ेष', ' cysts', 'เร' |

### `fiction_exclusive` — n=97

Features seen in **only** the `fiction` JB class (no bare, no other JB). Candidates for `fiction`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L26:F13289` | L26 | 1.073 | '\\*', ' dada', ' realizzato' |
| `L27:F11285` | L27 | 0.784 | 'ogun', 'amo', 'jde' |
| `L22:F1929` | L22 | 0.716 | ' Another', ' another', ' ANOTHER' |
| `L25:F7197` | L25 | 0.678 | ' styling', 'Stage', ' choreography' |
| `L22:F12901` | L22 | 0.649 | ' پنجاب', 'वाईसी', 'గర' |

### `universal_refusal_core` — n=83

Features seen in **bare + all 5 JB classes**. The canonical refusal core — present in both harmful-alone and every jailbreak. Ablation control baseline.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L29:F1066` | L29 | 8.037 | ' لیک', ' carr', ' evolved' |
| `L24:F1304` | L24 | 4.394 | 'ureau', ' sentidos', 'anea' |
| `L24:F107` | L24 | 3.881 | ' ok', ' okay', ' OK' |
| `L25:F963` | L25 | 3.838 | '級', 'ận', 'omeres' |
| `L26:F319` | L26 | 3.750 | ' ho', ' хе', ' hur' |

### `cognitive_reframe_exclusive` — n=77

Features seen in **only** the `cognitive_reframe` JB class (no bare, no other JB). Candidates for `cognitive_reframe`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L33:F4561` | L33 | 0.968 | 'pl', 'Por', ' Plut' |
| `L33:F15282` | L33 | 0.965 | ' audience', ' audiences', 'audience' |
| `L30:F574` | L30 | 0.810 | ' comment', ' peep', ' insult' |
| `L32:F235` | L32 | 0.645 | ' string', ' bool', ' char' |
| `L31:F2998` | L31 | 0.635 | ' 등', ' 等', '等' |

### `completion_exclusive` — n=68

Features seen in **only** the `completion` JB class (no bare, no other JB). Candidates for `completion`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L30:F14640` | L30 | 0.244 | ' Kardash', '淑', 'oka' |
| `L31:F3857` | L31 | 0.160 | ' mu', ' Mu', ' mus' |
| `L30:F13371` | L30 | 0.143 | ' Tra', 'Tra', ' tra' |
| `L31:F1561` | L31 | 0.142 | ' d', '菽', ' statement' |
| `L25:F11691` | L25 | 0.137 | ' labs', 'igkeits', ' facades' |

### `canonical_pro_refusal` — n=56

Features seen in **all 5 JB classes but NOT bare**. Recruited specifically under jailbreak — interpretable as a shared JB-suppression / pro-refusal response.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L25:F5301` | L25 | 1.381 | '<h2>', '##', '</h2>' |
| `L22:F143` | L22 | 1.128 | ' by', 'hi', ' closed' |
| `L28:F7710` | L28 | 1.054 | '往', 'ಚಿತ', 'ந்தி' |
| `L20:F256` | L20 | 1.036 | '<h2>', '</h2>', ' flue' |
| `L32:F295` | L32 | 0.894 | 'aadhar', 'aldehyde', 'nemouth' |

### `analytical_exclusive` — n=54

Features seen in **only** the `analytical` JB class (no bare, no other JB). Candidates for `analytical`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L29:F1917` | L29 | 1.071 | ' зал', ' اتا', ' ne' |
| `L25:F7029` | L25 | 0.774 | 'তখন', 'स्ताव', 'ப்பினர்' |
| `L26:F8980` | L26 | 0.733 | ' whether', ' Whether', 'Whether' |
| `L33:F1042` | L33 | 0.642 | ' சுண்ணாம்பு', ' தாம்', ' Ackerman' |
| `L26:F1568` | L26 | 0.634 | ' ਨ', ' ન', ' न' |

### `dampening_specialists` — n=52

Features in the **dampened** bucket of ≥3 JB classes. Pro-refusal features whose contribution to the refusal direction weakens across most JB types.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L29:F1066` | L29 | 8.037 | ' لیک', ' carr', ' evolved' |
| `L24:F1304` | L24 | 4.394 | 'ureau', ' sentidos', 'anea' |
| `L25:F963` | L25 | 3.838 | '級', 'ận', 'omeres' |
| `L28:F209` | L28 | 3.643 | ' matar', '               ', ' наве' |
| `L31:F498` | L31 | 3.477 | ' methods', ' method', ' 方法' |

### `anti_refusal_amplifiers` — n=50

Features in the **amplified_anti** bucket of ≥3 JB classes. Anti-refusal features that grow in magnitude across most JB types — the bypass signal.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L24:F107` | L24 | 3.881 | ' ok', ' okay', ' OK' |
| `L26:F319` | L26 | 3.750 | ' ho', ' хе', ' hur' |
| `L28:F395` | L28 | 3.212 | ' Int', 'Int', 'Ent' |
| `L29:F2642` | L29 | 3.059 | 'Hew', ' lie', ' atelier' |
| `L29:F257` | L29 | 3.036 | ' utilised', ' frontier', '厉' |

## Pairwise overlap (top 10 by normalized intersection)

Normalized overlap = |A ∩ B| / min(|A|, |B|). High values mean the smaller set is largely contained in the larger. `late_wave_layer24_32` naturally absorbs many.

| A | B | norm. overlap |
|---|---|---|
| `sign_flip_convergent` | `late_wave_layer24_32` | 0.89 |
| `dampening_specialists` | `late_wave_layer24_32` | 0.87 |
| `canonical_pro_refusal` | `sign_flip_convergent` | 0.86 |
| `universal_refusal_core` | `dampening_specialists` | 0.85 |
| `canonical_pro_refusal` | `late_wave_layer24_32` | 0.82 |
| `universal_refusal_core` | `late_wave_layer24_32` | 0.81 |
| `late_wave_layer24_32` | `analytical_exclusive` | 0.80 |
| `late_wave_layer24_32` | `completion_exclusive` | 0.79 |
| `late_wave_layer24_32` | `roleplay_exclusive` | 0.77 |
| `anti_refusal_amplifiers` | `late_wave_layer24_32` | 0.74 |

## Suggested Stage 08 ablation targets (causal-impact order)

1. `canonical_pro_refusal` — JB-specific pro-refusal recruitment. Ablation should *strengthen* JB bypass (removes the JB-only refusal boost).
2. `sign_flip_convergent` — robust direction reversals. Ablation should partially restore bare behavior under JB.
3. `dampening_specialists` — weakened pro-refusal features. Restoring them to bare strength should counter fiction/analytical bypass.
4. `anti_refusal_amplifiers` — JB-amplified bypass signal. Suppressing them should increase refusal under JB.
5. `universal_refusal_core` — shared baseline. Ablation should break refusal on bare *and* JB (control — proves the subcircuits matter).

