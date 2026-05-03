# Subcircuits Report (Rule-Based)

Each subcircuit is defined by a precise set-logic rule over the features observed across bare + 5 JB classes (original rules) and bare + 5 jb_* + 5 ctrl_* conditions (new Apr 22 ctrl-aware rules). No ML fitting — fully interpretable.

## Summary table

| Subcircuit | Size | Peak layer | n_layers occupied | Mean act. freq. |
|---|---|---|---|---|
| `late_wave_layer24_32` | 211 | L31 (×34) | 9 | 0.0029 |
| `fiction_exclusive` | 85 | L26 (×7) | 25 | 0.0017 |
| `roleplay_exclusive` | 51 | L33 (×4) | 27 | 0.0047 |
| `analytical_exclusive` | 42 | L15 (×3) | 25 | 0.0031 |
| `universal_refusal_core` | 40 | L28 (×7) | 12 | 0.0040 |
| `completion_exclusive` | 35 | L22 (×4) | 20 | 0.0015 |
| `cognitive_reframe_exclusive` | 30 | L5 (×3) | 21 | 0.0013 |
| `dampening_specialists` | 26 | L29 (×6) | 9 | 0.0034 |
| `sign_flip_convergent` | 19 | L27 (×3) | 12 | 0.0044 |
| `anti_refusal_amplifiers` | 11 | L16 (×2) | 10 | 0.0020 |
| `canonical_pro_refusal` | 8 | L18 (×2) | 7 | 0.0059 |
| `ctrl_shared_refusal` | 0 | — | 0 | N/A |
| `ctrl_only` | 0 | — | 0 | N/A |
| `jb_analytical_specific_vs_ctrl` | 0 | — | 0 | N/A |
| `jb_cognitive_reframe_specific_vs_ctrl` | 0 | — | 0 | N/A |
| `jb_completion_specific_vs_ctrl` | 0 | — | 0 | N/A |
| `jb_fiction_specific_vs_ctrl` | 0 | — | 0 | N/A |
| `jb_roleplay_specific_vs_ctrl` | 0 | — | 0 | N/A |

## Subcircuit definitions and top features

### `late_wave_layer24_32` — n=211

All tagged features in layers **24–32** — the JB-impact band identified in A8. Layer-based cross-cut; overlaps other subcircuits.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L32:F132684` | L32 | 1.116 | ' Sol', ' Making', 'Navig' |
| `L30:F9168` | L30 | 0.799 | ' sor', '非常', ' reg' |
| `L32:F50401` | L32 | 0.758 | 'Ｉ', '_I', '-I' |
| `L29:F18221` | L29 | 0.695 | 'Unfortunately', '遗憾', ' Unfortunately' |
| `L31:F49541` | L31 | 0.529 | '我会', '我也', 'I' |

### `fiction_exclusive` — n=85

Features seen in **only** the `fiction` JB class (no bare, no other JB). Candidates for `fiction`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L31:F114183` | L31 | 0.155 | '*M', '*C', ' (*' |
| `L31:F116648` | L31 | 0.136 | 'sure', '一定', ' certainly' |
| `L27:F95667` | L27 | 0.089 | 'Once', '话说', ' Once' |
| `L25:F13130` | L25 | 0.080 | '我一直', '我不', '我觉得' |
| `L20:F104315` | L20 | 0.077 | '我不是', 'mere', '并不代表' |

### `roleplay_exclusive` — n=51

Features seen in **only** the `roleplay` JB class (no bare, no other JB). Candidates for `roleplay`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L33:F36808` | L33 | 0.142 | 'Hack', '霸', 'ifle' |
| `L31:F121167` | L31 | 0.111 | '逆', 'Reverse', 'reverse' |
| `L30:F85543` | L30 | 0.093 | ' Understanding', ' understood', ' understand' |
| `L30:F42561` | L30 | 0.089 | '→when', ' When', 'when' |
| `L31:F27096` | L31 | 0.088 | ' investigating', ' trying', ' studying' |

### `analytical_exclusive` — n=42

Features seen in **only** the `analytical` JB class (no bare, no other JB). Candidates for `analytical`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L26:F139999` | L26 | 0.236 | '否', ' false', '“No' |
| `L29:F38514` | L29 | 0.092 | ' disgust', '腐', '恶心' |
| `L30:F31985` | L30 | 0.087 | ' spite', 'ErrMsg', 'Instead' |
| `L26:F126188` | L26 | 0.087 | 'true', ' True', '>true' |
| `L31:F132419` | L31 | 0.083 | 'No', '_no', '-no' |

### `universal_refusal_core` — n=40

Features seen in **bare + all 5 JB classes**. The canonical refusal core — present in both harmful-alone and every jailbreak. Ablation control baseline.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L32:F132684` | L32 | 1.116 | ' Sol', ' Making', 'Navig' |
| `L32:F50401` | L32 | 0.758 | 'Ｉ', '_I', '-I' |
| `L29:F18221` | L29 | 0.695 | 'Unfortunately', '遗憾', ' Unfortunately' |
| `L31:F49541` | L31 | 0.529 | '我会', '我也', 'I' |
| `L32:F158432` | L32 | 0.481 | '创造', 'create', '创' |

### `completion_exclusive` — n=35

Features seen in **only** the `completion` JB class (no bare, no other JB). Candidates for `completion`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L31:F129293` | L31 | 0.088 | ' Feel', ' Content', ' Kind' |
| `L28:F29864` | L28 | 0.077 | ' apolog', ' apology', ' sorry' |
| `L31:F32127` | L31 | 0.074 | '-exc', 'Exc', ' Exc' |
| `L22:F125946` | L22 | 0.069 | 'manufacturer', 'ndx', 'usk' |
| `L26:F28935` | L26 | 0.059 | 'usu', 'zet', 'ично' |

### `cognitive_reframe_exclusive` — n=30

Features seen in **only** the `cognitive_reframe` JB class (no bare, no other JB). Candidates for `cognitive_reframe`-specific jailbreak mechanism.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L8:F152608` | L8 | 0.118 | 'aine', 'NotSupportedException', ' rin' |
| `L32:F147668` | L32 | 0.097 | ' Maint', ' Determin', ' Bec' |
| `L29:F148070` | L29 | 0.073 | ' mitig', ' renew', ' synchron' |
| `L26:F112379` | L26 | 0.069 | 'statements', 'Statements', ' Statements' |
| `L32:F9590` | L32 | 0.018 | 'Sp', 'sp', ' Sp' |

### `dampening_specialists` — n=26

Features in the **dampened** bucket of ≥3 JB classes. Pro-refusal features whose contribution to the refusal direction weakens across most JB types.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L32:F132684` | L32 | 1.116 | ' Sol', ' Making', 'Navig' |
| `L30:F9168` | L30 | 0.799 | ' sor', '非常', ' reg' |
| `L32:F50401` | L32 | 0.758 | 'Ｉ', '_I', '-I' |
| `L29:F18221` | L29 | 0.695 | 'Unfortunately', '遗憾', ' Unfortunately' |
| `L31:F49541` | L31 | 0.529 | '我会', '我也', 'I' |

### `sign_flip_convergent` — n=19

Features in the **sign_flipped** bucket of ≥3 JB classes. Robustly reverse attribution sign under JB — the highest-confidence mechanism-change features.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L33:F80077` | L33 | 0.085 | 'Sp', 'sp', '_sp' |
| `L20:F110580` | L20 | 0.023 | '榛', '禄', 'ata' |
| `L31:F86700` | L31 | 0.018 |  |
| `L31:F111734` | L31 | 0.015 | ' PROM', 'Prom', ' Prom' |
| `L27:F27345` | L27 | 0.015 | 'Prompt', ' prompts', 'prompt' |

### `anti_refusal_amplifiers` — n=11

Features in the **amplified_anti** bucket of ≥3 JB classes. Anti-refusal features that grow in magnitude across most JB types — the bypass signal.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L29:F114615` | L29 | 0.350 | '肯定', ' absolutely', ' Definitely' |
| `L32:F36682` | L32 | 0.169 | 'onte', 'ü', 'indrome' |
| `L33:F97879` | L33 | 0.163 | 'validation', 'ccb', ' Converted' |
| `L16:F35905` | L16 | 0.093 | 'rates', ' Attention', 'uil' |
| `L24:F81123` | L24 | 0.076 | '当然', 'Sure', 'Certainly' |

### `canonical_pro_refusal` — n=8

Features seen in **all 5 JB classes but NOT bare**. Recruited specifically under jailbreak — interpretable as a shared JB-suppression / pro-refusal response.

**Top 5 by |attribution|:**

| Feature | Layer | \|attr\| | Top logits |
|---|---|---|---|
| `L30:F81884` | L30 | 0.116 | '它是', ' It', 'It' |
| `L33:F39236` | L33 | 0.105 | 'Too', 'But', '但它' |
| `L16:F35905` | L16 | 0.093 | 'rates', ' Attention', 'uil' |
| `L18:F148614` | L18 | 0.008 | ' Ug', 'iques', ' Eag' |
| `L13:F65439` | L13 | 0.005 | 'ackson', 'ilon', ' Orlando' |

### `ctrl_shared_refusal` — n=0

Features in **bare ∩ all 5 ctrl_*_top50** but **NOT in all 5 jb_*_top50**. The prefix-invariant refusal spine: machinery the refusal circuit uses regardless of whether the prefix carries JB-semantics or matched benign content. These are NOT JB-semantic — they define the baseline that survives a long-prefix perturbation without JB intent.

### `ctrl_only` — n=0

Features in **all 5 ctrl_*_top50** but not in bare or any jb_*_top50. Usually tiny; if non-empty, it signals that matched benign prefixes recruit features neither bare-harmful nor any jailbreak uses — typically benign-content semantic features triggered by the ctrl prefix text itself.

### `jb_analytical_specific_vs_ctrl` — n=0

Features in **jb_analytical_top50 − ctrl_analytical_top50**. The cleanest JB-semantic subcircuit for `analytical`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

### `jb_cognitive_reframe_specific_vs_ctrl` — n=0

Features in **jb_cognitive_reframe_top50 − ctrl_cognitive_reframe_top50**. The cleanest JB-semantic subcircuit for `cognitive_reframe`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

### `jb_completion_specific_vs_ctrl` — n=0

Features in **jb_completion_top50 − ctrl_completion_top50**. The cleanest JB-semantic subcircuit for `completion`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

### `jb_fiction_specific_vs_ctrl` — n=0

Features in **jb_fiction_top50 − ctrl_fiction_top50**. The cleanest JB-semantic subcircuit for `fiction`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

### `jb_roleplay_specific_vs_ctrl` — n=0

Features in **jb_roleplay_top50 − ctrl_roleplay_top50**. The cleanest JB-semantic subcircuit for `roleplay`: after controlling for prefix length/structure via the matched benign ctrl prefix, what remains is features the JB *semantic* content genuinely recruits. Complements canonical_pro_refusal (which finds cross-class intersection) by isolating per-class mechanism.

## Pairwise overlap (top 10 by normalized intersection)

Normalized overlap = |A ∩ B| / min(|A|, |B|). High values mean the smaller set is largely contained in the larger. `late_wave_layer24_32` naturally absorbs many.

| A | B | norm. overlap |
|---|---|---|
| `dampening_specialists` | `late_wave_layer24_32` | 0.88 |
| `universal_refusal_core` | `late_wave_layer24_32` | 0.82 |
| `universal_refusal_core` | `dampening_specialists` | 0.77 |
| `late_wave_layer24_32` | `fiction_exclusive` | 0.52 |
| `canonical_pro_refusal` | `sign_flip_convergent` | 0.38 |
| `canonical_pro_refusal` | `anti_refusal_amplifiers` | 0.38 |
| `late_wave_layer24_32` | `roleplay_exclusive` | 0.37 |
| `sign_flip_convergent` | `late_wave_layer24_32` | 0.37 |
| `anti_refusal_amplifiers` | `late_wave_layer24_32` | 0.36 |
| `late_wave_layer24_32` | `analytical_exclusive` | 0.29 |

## Suggested Stage 08 ablation targets (causal-impact order)

1. `canonical_pro_refusal` — JB-specific pro-refusal recruitment. Ablation should *strengthen* JB bypass (removes the JB-only refusal boost).
2. `jb_{cls}_specific_vs_ctrl` (per class) — the cleanest per-class JB-semantic mechanism. Ablating one should selectively restore ctrl-like behavior on that class (dissociation test).
3. `sign_flip_convergent` — robust direction reversals. Ablation should partially restore bare behavior under JB.
4. `dampening_specialists` — weakened pro-refusal features. Restoring them to bare strength should counter fiction/analytical bypass.
5. `anti_refusal_amplifiers` — JB-amplified bypass signal. Suppressing them should increase refusal under JB.
6. `ctrl_shared_refusal` — the prefix-invariant spine. Ablation should break refusal on BOTH ctrl and bare — a negative control proving these aren't JB-specific.
7. `universal_refusal_core` — shared baseline. Ablation should break refusal on bare *and* JB (control — proves the subcircuits matter).

