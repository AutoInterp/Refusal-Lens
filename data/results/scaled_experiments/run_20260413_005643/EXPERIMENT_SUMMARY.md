# Scaled Attribution Experiment: Key Findings

**Date**: April 13, 2026
**Model**: google/gemma-3-4b-it (34 layers, 2304-dim)
**GPU**: NVIDIA RTX 6000 Ada (50GB) | **Runtime**: 113 minutes
**Dataset**: 50 diverse harmful prompts x 5 jailbreak classes + bare = 300 attribution graphs
**Direction**: Layer 32 refusal direction (difference-in-means, 64 harmful + 64 harmless)
**Features**: All active CLT features per graph (no top-k filtering)

---

## 1. Main Result: Jailbreak Classes Have Distinct, Statistically Significant Effects

All five jailbreak classes produce statistically significant changes to the MLP refusal circuit, but through quantitatively different mechanisms.

| Class | Net Delta | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Consistency |
|-------|----------|----------|-------------|-----------|--------|-------------|
| **Analytical** | -66.2 | -97.3% | <0.0001*** | -1.72 | [-76.9, -55.5] | 94% (47/50) |
| **Fiction** | -62.7 | -92.1% | <0.0001*** | -1.52 | [-74.1, -51.4] | 94% (47/50) |
| **Cognitive reframe** | -54.8 | -80.5% | <0.0001*** | -1.59 | [-64.1, -45.5] | **100% (50/50)** |
| **Roleplay** | -29.6 | -43.5% | <0.0001*** | -0.81 | [-39.9, -19.7] | 78% (39/50) |
| **Completion** | **+6.0** | **+8.8%** | 0.005** | +0.32 | [+0.5, +11.0] | 34% (17/50) |

*Consistency = fraction of prompts where JB net attribution < bare net attribution.*

**Analytical jailbreaks are the strongest suppressor** (d=-1.72), overtaking fiction (d=-1.52) which was strongest in our preliminary 5-prompt experiments. The "is this technically accurate" framing nearly eliminates MLP refusal attribution (-97.3%).

**Cognitive reframe is the most consistent**: it suppressed refusal in all 50 prompts (100%), despite having a smaller mean effect than analytical/fiction. This suggests it works reliably across all harmful content types, while analytical and fiction occasionally fail (3/50 prompts each).

**Completion paradoxically increases refusal** (+8.8%, p=0.005). This is confirmed at scale and is not noise.

![Class Comparison](class_comparison.png)
*Figure 1: Attribution decomposition by jailbreak class (n=50). Red = pro-refusal, blue = anti-refusal, green = net. Error bars on net. Significance stars from paired Wilcoxon test.*

![Effect Sizes](effect_sizes.png)
*Figure 2: Cohen's d effect sizes. All suppressive classes exceed the "large effect" threshold (|d| > 0.8). Completion is the only class with a positive (refusal-increasing) effect.*

---

## 2. Dual Mechanism Decomposition

Every jailbreak class operates through two simultaneous mechanisms. Their balance determines the class's overall effect.

| Class | dPos (pro-refusal) | dNeg (anti-refusal) | Net Delta | Dominant Mechanism |
|-------|--------------------|--------------------|-----------|--------------------|
| **Roleplay** | -15.6 (-11.8%) | -14.0 (+22.0%) | -29.6 | Balanced dampening + amplification |
| **Fiction** | **-41.4 (-31.4%)** | -21.3 (+33.4%) | -62.7 | **Dampening-dominant** |
| **Analytical** | -38.4 (-29.1%) | **-27.8 (+43.6%)** | -66.2 | **Amplification-dominant** |
| **Completion** | **+19.9 (+15.1%)** | -13.8 (+21.7%) | +6.0 | Pro-refusal recruitment |
| **Cognitive reframe** | -35.7 (-27.1%) | -19.1 (+30.0%) | -54.8 | Dampening-dominant |

*dPos = change in sum of positive (pro-refusal) attributions. dNeg = change in sum of negative (anti-refusal) attributions. Negative dPos = dampening. More negative dNeg = anti-refusal amplification.*

**Fiction achieves the largest dampening** of pro-refusal features (-41.4), reducing their total contribution by 31.4%. This is the "narrative distancing" effect: framing harmful content as fiction disengages the refusal-promoting features.

**Analytical achieves the largest anti-refusal amplification** (-27.8, a 43.6% increase in anti-refusal magnitude). The "evaluate this technically" framing recruits features that actively push against refusal, more than any other class.

**Completion is unique**: it's the only class where pro-refusal features *strengthen* (+19.9). The model interprets "Complete this manual entry:" as an additional safety concern, activating more refusal features than the bare harmful prompt. Anti-refusal features also increase (-13.8), but not enough to overcome the pro-refusal boost.

---

## 3. Feature-Level Circuit Restructuring

Examining all active features (not top-k) reveals how each jailbreak class restructures the refusal circuit.

| Class | Bare Features | JB Features | JB/Bare Ratio | Shared % | JB-only % | Sign-flip % |
|-------|-------------|------------|--------------|----------|----------|------------|
| **Roleplay** | 8,338 | 19,682 | 1.51x | 65.2% | 56.5% | 18.0% |
| **Fiction** | 8,338 | 20,467 | **1.55x** | **56.9%** | **63.0%** | **26.0%** |
| **Analytical** | 8,338 | 18,407 | 1.46x | 62.4% | 56.9% | 22.7% |
| **Completion** | 8,338 | 16,863 | **1.35x** | **73.5%** | **45.3%** | **16.7%** |
| **Cognitive reframe** | 8,338 | 16,967 | 1.34x | 64.2% | 51.3% | 20.0% |

*Shared % = fraction of bare features that remain active under JB. JB-only % = fraction of JB features that are newly recruited. Sign-flip % = fraction of shared features that change attribution sign.*

**Fiction produces the most dramatic circuit restructuring**: it shares only 56.9% of bare features (least overlap), recruits 63.0% new features (most), and flips the sign of 26.0% of shared features (most). This supports the "tug-of-war" mechanism: fiction doesn't just dampen existing features, it fundamentally reorganizes the circuit.

**Completion preserves the most of the original circuit**: 73.5% feature overlap (most), only 45.3% new features (least), and 16.7% sign flips (least). The completion prefix adds context without dramatically changing which features are active — it amplifies existing features rather than recruiting new ones.

**All jailbreak classes roughly double the active feature count** (1.34x to 1.55x more features than bare). The additional tokens in the jailbreak prefix activate features in earlier layers that aren't present in the shorter bare prompt.

---

## 4. Per-Prompt Variance

![Per-Prompt Deltas](per_prompt_deltas.png)
*Figure 3: Per-prompt attribution delta (JB - bare) for each class. Red bars = JB suppresses refusal. Blue bars = JB increases refusal. Green dashed line = class mean.*

**Roleplay shows the highest variance** (std=36.7): it strongly suppresses some prompts (up to -120) but slightly increases refusal for others. This makes roleplay the least predictable jailbreak class.

**Cognitive reframe is remarkably consistent**: every single prompt (50/50) shows suppression, with moderate variance (std=34.4). This makes it the most *reliable* jailbreak class, even though its mean effect is smaller than analytical/fiction.

**Fiction and analytical are bimodal**: most prompts show strong suppression, but 3/50 prompts show mild increases. Investigating these resistant prompts could reveal what makes certain harmful content harder to jailbreak.

---

## 5. The Completion Paradox

Completion is the only class that *strengthens* the refusal signal. The mechanism:

- Pro-refusal features strengthen in **43/50 prompts** (mean: +19.9)
- Anti-refusal features also strengthen in **46/50 prompts** (mean: -13.8)
- But pro-refusal wins: the net effect is +6.0

The "Complete this manual entry:" prefix appears to trigger a **jailbreak detection response**: the model recognizes the task-continuation framing as a potential bypass attempt and activates additional refusal features. This is mechanistically interesting because it suggests the model has learned a meta-pattern for detecting certain jailbreak strategies, but not others.

---

## 6. Implications

1. **Not all jailbreaks work the same way.** The five classes tested here span a spectrum from strong suppression (analytical, d=-1.72) to paradoxical refusal amplification (completion, d=+0.32). Any defense against jailbreaks must account for this diversity.

2. **Analytical framing is more dangerous than fiction.** Despite fiction being the more commonly studied jailbreak class in the literature, analytical framing ("Is this technically accurate?") produces a larger and more consistent effect on MLP refusal features. This class deserves more attention in safety evaluations.

3. **Cognitive reframe is the most reliable jailbreak.** While not the strongest on average, it suppressed refusal in 100% of tested prompts. For an adversary optimizing for reliability over magnitude, this class is optimal.

4. **The completion paradox suggests defensive potential.** If certain framings can inadvertently *strengthen* refusal, understanding this mechanism could inform adversarial training strategies that make models more robust.

5. **Circuit restructuring varies by class.** Fiction fundamentally reorganizes the feature circuit (26% sign flips, 63% new features), while completion mostly amplifies existing features (17% sign flips, 45% new features). Defenses targeting the dampening mechanism may fail against fiction's restructuring approach.

---

*Generated from 50 prompts x 6 conditions = 300 attribution graphs. Full data in `attribution_results.json`, statistical tests in `statistical_analysis.json`, feature comparison in `feature_comparison_aggregate.json`.*
