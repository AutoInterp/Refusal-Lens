# Refusal-Lens Pipeline

End-to-end pipeline for attributing and analyzing the refusal circuit in Gemma-3-4b-it. Ingests harmful/harmless prompts, computes per-layer refusal directions, runs CLT attribution with the vendored circuit-tracer, verifies the attribution arithmetic, and labels every feature using the Gemma Scope HuggingFace dashboards.

For the design philosophy, stage specs, and forward roadmap see **[PIPELINE_PLAN.md](PIPELINE_PLAN.md)**. This README covers the current implementation state, the most recent experiment's numerical results, and deployment.

---

## Stage summary

| # | Script | Role | Status |
|---|---|---|---|
| 01 | `01_compute_direction.py` | Per-layer refusal directions (normalized + unnormalized) at all 34 layers | ✓ Done |
| 02 | `02_run_attribution.py` | CLT attribution graphs for bare + 5 jailbreak classes, per-prompt feature comparison | ✓ Done |
| 02b | `02b_statistical_analysis.py` | Paired stats (Wilcoxon, t-test, Cohen's d, bootstrap CIs), dual-mechanism decomposition, plots, markdown report | ✓ Done |
| 03 | `03_verify_attribution.py` | Verifies `sum(feature attributions) ≈ r · h[L=32]`, reports MLP contribution fraction | ✓ Done |
| 04 | `04_label_features.py` | Labels every unique feature using HuggingFace dashboard binary payloads (top/bottom logits, examples) | ✓ Done |
| 05 | `05_visualize_circuits.py` | Color-coded attribution circuit diagrams | Planned |
| 06 | `06_causal_intervention.py` | Arditi-method causal intervention (Tejas's track) | Planned |

Shared modules: `config.py` (model/layer/class constants), `utils.py` (run-dir helpers, prompt formatting, dataset selection).

---

## Latest experiment — `run_20260417_010035` (50 prompts)

### Stage 01 · Direction computation

<table>
<thead>
<tr><th>Layer</th><th align="right">Separation</th><th>Role</th></tr>
</thead>
<tbody>
<tr><td><code>L0</code></td><td align="right"><code>7.4</code></td><td>—</td></tr>
<tr><td><code>L11</code></td><td align="right"><code>1,346</code></td><td>early MLP buildup</td></tr>
<tr><td><code><strong>L15</strong></code></td><td align="right"><code><strong>3,101</strong></code></td><td><strong>best causal layer</strong> (Tejas)</td></tr>
<tr><td><code>L24</code></td><td align="right"><code>9,384</code></td><td>JB effects concentrate here → L32</td></tr>
<tr><td><code>L29</code></td><td align="right"><code>18,930</code></td><td>—</td></tr>
<tr><td><code><strong>L32</strong></code></td><td align="right"><code><strong>20,873</strong></code></td><td><strong>best separation layer</strong></td></tr>
<tr><td><code>L33</code></td><td align="right"><code>287</code></td><td>pre-RMSNorm artifact</td></tr>
</tbody>
</table>

- Matches Tejas's prior numbers within noise (L32: `20,827`; L15: `3,131`).
- **`cos(L15, L32) = −0.115`** — near-orthogonal. The full 34×34 cosine matrix (see "Mechanistic story" below) resolves this into three regimes separated by two pivots at L13→L14 and L17→L18.
- **`cos(L32, L33) = +0.83`** — the L33 separation collapse (`287` vs `20,873`) is a *magnitude* collapse only, not a directional rotation. RMSNorm preserves the direction but rescales it away.

### Stage 02b · Statistical analysis

All five jailbreak classes produce highly significant effects on net refusal attribution.

<table>
<thead>
<tr>
  <th>Class</th>
  <th align="right">Δnet</th>
  <th align="right">% change</th>
  <th align="right">Cohen's d</th>
  <th align="right">p (Wilcoxon)</th>
  <th>95% CI</th>
  <th align="right">Consistency</th>
</tr>
</thead>
<tbody>
<tr><td><strong>Analytical</strong></td><td align="right"><code>−73.7</code></td><td align="right"><code>−104.6%</code></td><td align="right"><code>−2.37</code></td><td align="right"><code>5.3e-15</code></td><td><code>[−82.1, −64.9]</code></td><td align="right"><code>49/50</code></td></tr>
<tr><td><strong>Fiction</strong></td><td align="right"><code>−65.3</code></td><td align="right"><code>−92.7%</code></td><td align="right"><code>−1.57</code></td><td align="right"><code>3.0e-13</code></td><td><code>[−76.6, −53.8]</code></td><td align="right"><code>47/50</code></td></tr>
<tr><td><strong>Cognitive reframe</strong></td><td align="right"><code>−50.2</code></td><td align="right"><code>−71.3%</code></td><td align="right"><code>−1.41</code></td><td align="right"><code>2.5e-14</code></td><td><code>[−60.2, −40.4]</code></td><td align="right"><code>49/50</code></td></tr>
<tr><td><strong>Roleplay</strong></td><td align="right"><code>−38.7</code></td><td align="right"><code>−54.9%</code></td><td align="right"><code>−0.91</code></td><td align="right"><code>1.4e-8</code></td><td><code>[−50.5, −27.0]</code></td><td align="right"><code>42/50</code></td></tr>
<tr><td><strong>Completion</strong></td><td align="right"><code><strong>+5.0</strong></code></td><td align="right"><code><strong>+7.2%</strong></code></td><td align="right"><code>+0.27</code></td><td align="right"><code>0.011</code></td><td><code>[−0.1, +10.2]</code></td><td align="right"><code>15/50</code></td></tr>
</tbody>
</table>

**Dual-mechanism decomposition** — how each class moves the positive-attribution (`dPos`) and negative-attribution (`dNeg`) halves separately:

<table>
<thead>
<tr><th>Class</th><th align="right">dPos (pro-refusal)</th><th align="right">dNeg (anti-refusal)</th><th>Dominant</th></tr>
</thead>
<tbody>
<tr><td>Roleplay</td><td align="right"><code>−22.5 (−16.7%)</code></td><td align="right"><code>−16.2 (−25.3%)</code></td><td>Balanced</td></tr>
<tr><td>Fiction</td><td align="right"><code>−43.1 (−32.0%)</code></td><td align="right"><code>−22.2 (−34.6%)</code></td><td>Balanced</td></tr>
<tr><td>Analytical</td><td align="right"><code>−44.7 (−33.2%)</code></td><td align="right"><code>−29.0 (−45.1%)</code></td><td>Balanced</td></tr>
<tr><td><strong>Completion</strong></td><td align="right"><code><strong>+19.7 (+14.6%)</strong></code></td><td align="right"><code>−14.6 (−22.8%)</code></td><td><strong>Pro-refusal recruitment</strong></td></tr>
<tr><td>Cognitive reframe</td><td align="right"><code>−33.8 (−25.1%)</code></td><td align="right"><code>−16.4 (−25.5%)</code></td><td>Dampening-dominant</td></tr>
</tbody>
</table>

Completion is the paradox: `35/50` prompts show *more* refusal attribution after the JB framing. `dPos` increases by `+14.6%` — new pro-refusal features are being recruited, not existing ones being amplified.

**Feature comparison sizes** — total unique features per bare-vs-JB pair:

<table>
<thead>
<tr><th>Class</th><th align="right">Bare</th><th align="right">JB</th><th align="right">Shared %</th><th align="right">JB-only %</th><th align="right">Sign-flip %</th></tr>
</thead>
<tbody>
<tr><td>Roleplay</td><td align="right"><code>8,342</code></td><td align="right"><code>12,540</code></td><td align="right"><code>65.7%</code></td><td align="right"><code>56.3%</code></td><td align="right"><code>17.5%</code></td></tr>
<tr><td><strong>Fiction</strong></td><td align="right"><code>8,342</code></td><td align="right"><code>12,996</code></td><td align="right"><code>58.5%</code></td><td align="right"><code><strong>62.5%</strong></code></td><td align="right"><code><strong>25.6%</strong></code></td></tr>
<tr><td>Analytical</td><td align="right"><code>8,342</code></td><td align="right"><code>12,109</code></td><td align="right"><code>63.0%</code></td><td align="right"><code>56.6%</code></td><td align="right"><code>23.1%</code></td></tr>
<tr><td>Completion</td><td align="right"><code>8,342</code></td><td align="right"><code>11,200</code></td><td align="right"><code><strong>73.5%</strong></code></td><td align="right"><code>45.3%</code></td><td align="right"><code>16.6%</code></td></tr>
<tr><td>Cognitive reframe</td><td align="right"><code>8,342</code></td><td align="right"><code>11,131</code></td><td align="right"><code>64.4%</code></td><td align="right"><code>51.8%</code></td><td align="right"><code>20.3%</code></td></tr>
</tbody>
</table>

Fiction restructures the circuit most dramatically — highest sign-flip rate (`25.6%`) and highest share of JB-only features. Completion preserves the most of the bare circuit (`73.5%` shared).

### Stage 03 · Attribution verification (M2)

<table>
<thead>
<tr><th>Metric</th><th align="right">Mean</th><th align="right">Std</th></tr>
</thead>
<tbody>
<tr><td>Full projection <code>r · h[L=32]</code></td><td align="right"><code>17,230.8</code></td><td align="right"><code>1,986</code></td></tr>
<tr><td>Σ feature attributions (MLP only)</td><td align="right"><code>70.47</code></td><td align="right"><code>17.6</code></td></tr>
<tr><td><strong>MLP ratio</strong></td><td align="right"><code><strong>0.404%</strong></code></td><td align="right"><code>0.077%</code></td></tr>
</tbody>
</table>

- `attr_net_mean = 70.47` exactly matches the bare condition in Stage 02b — keys plumb through correctly end-to-end.
- Per-layer decomposition (10 prompts × 34 layers) shows early-layer buildup — layers 7–11 alone contribute `~2,400` to the projection for a typical prompt.
- The remaining `99.6%` is carried by **attention heads + embeddings** — a property of the Gemma Scope transcoder set (MLP-only).

### Stage 04 · Feature labeling (M4)

<table>
<thead>
<tr><th>Metric</th><th align="right">Value</th></tr>
</thead>
<tbody>
<tr><td>Unique features collected</td><td align="right"><code>876</code></td></tr>
<tr><td>Labeled via HF Gemma Scope</td><td align="right"><code>876 (100%)</code></td></tr>
<tr><td>Priority features (in sign-flipped / dampened / amplified-anti buckets)</td><td align="right"><code>788 / 788</code></td></tr>
<tr><td>Sign-flipped features</td><td align="right"><code>603</code></td></tr>
<tr><td>Dampened features</td><td align="right"><code>115</code></td></tr>
<tr><td>Amplified-anti features</td><td align="right"><code>117</code></td></tr>
</tbody>
</table>

Labels come from `mwhanna/gemma-scope-2-4b-it` (byte-range HTTP against the dashboard binary).

> **Caveat.** Many top-token patches in Gemma Scope are polyglot or byte-level noise. Labels are correct representations of what the dashboard shows, but human-interpretable features will require reading activation examples on concrete prompts (Stage 05).

---

## Mechanistic story so far

A running narrative of what the pipeline has taught us about Gemma-3-4b-it's refusal circuit. This section grows as new stages and analyses land — last revised 2026-04-17 after A3 + A4 + A5 + A6 + A7 + A8 visualizations.

Plot paths below are relative to repo root; each was produced by the most recent pipeline run (`data/results/pipeline_runs/run_20260417_010035/`).

### 1. Refusal is linear, but the "direction" rotates in two sharp pivots

The diff-in-means between harmful and harmless activations gives a clean linear readout at every layer from `L9` onward, with separation climbing monotonically to `L32 (~20,873)` before collapsing at `L33` to `287`.

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/02b_stats/separation_by_layer.png" alt="Refusal-direction separation as a function of layer, with L15 and L32 marked and the L33 pre-RMSNorm artifact annotated" width="800">
<figcaption><em><strong>Figure 1.</strong> Per-layer separation <code>|μ(harmful) − μ(harmless)|</code>. Monotonic climb through <code>L32</code>, collapse at <code>L33</code> from the pre-RMSNorm artifact.</em></figcaption>
</figure>

The full 34×34 cosine matrix between per-layer directions exposes three distinct *regimes* separated by two pivots:

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/02b_stats/cosine_heatmap.png" alt="34 by 34 cosine similarity matrix between per-layer refusal directions, showing three blocks" width="800">
<figcaption><em><strong>Figure 2.</strong> Cosine similarity <code>cos(r̂_i, r̂_j)</code> for all layer pairs. Red = aligned, blue = anti-aligned. Grey gridlines mark <code>L15</code>, <code>L25</code>, <code>L32</code>.</em></figcaption>
</figure>

<table>
<thead>
<tr><th>Regime</th><th>Layers</th><th>Internal coherence</th><th>Example cosine</th></tr>
</thead>
<tbody>
<tr><td>A — early "proto-refusal"</td><td><code>L5–L13</code></td><td>Strong (<code>~0.5–0.9</code>)</td><td><code>cos(L5, L13) = +0.39</code></td></tr>
<tr><td><strong>Pivot 1 (sharp)</strong></td><td><code>L13 → L14</code></td><td>—</td><td><code><strong>cos(L13, L14) = −0.21</strong></code></td></tr>
<tr><td>B — causal band</td><td><code>L14–L17</code></td><td>Very strong (<code>~0.94</code>)</td><td><code><strong>cos(L14, L15) = +0.94</strong></code></td></tr>
<tr><td><strong>Pivot 2 (gentler)</strong></td><td><code>L17 → L18</code></td><td>—</td><td>gradual drop</td></tr>
<tr><td>C — late readout</td><td><code>L18–L33</code></td><td>Increasing toward <code>L32</code></td><td><code>cos(L25, L32) = +0.31</code></td></tr>
</tbody>
</table>

This explains the L15/L32 paradox: **L15 (Tejas's best-causal layer) and L32 (best-separation layer) live in different regimes**. `cos(L15, L32) = −0.115` — near-orthogonal. Causal steering works at L15 because downstream layers rewrite the direction; causal intervention applied to the L32 direction at L15 misses the then-current geometry.

A second observation from the heatmap: **`cos(L32, L33) = +0.83`**, so the L33 collapse is almost entirely a *magnitude* collapse, not a directional rotation. RMSNorm rescales the projection but preserves the direction. This tightens the L33 caveat — it's not that the refusal direction "disappears" at L33, it's that the residual stream is renormalized and the dot product shrinks by `~70×`.

Finally, Regime A (`L5–L13`) and Regime C (`L18–L33`) correlate weakly-to-negatively (`−0.5 to −0.1` in the heatmap's blue strips). The early "proto-refusal" is **not** just a noisier version of the final readout — they're distinct features rewritten at the L13→L14 pivot.

### 2. The signal is assembled in two waves, with a dampening layer mid-stream

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/03_verification/per_layer_contribution.png" alt="Bar chart of mean per-layer contribution to the refusal-direction projection across 10 prompts" width="800">
<figcaption><em><strong>Figure 3.</strong> Mean per-layer contribution <code>(h[L+1] − h[L]) · r̂</code> to the L32 projection (n=10 prompts). Layers 0–32 shown; post-measurement L33 is the RMSNorm artifact and is reported in the inset.</em></figcaption>
</figure>

<table>
<thead>
<tr><th>Phase</th><th>Layers</th><th align="right">Mean contribution</th><th>Behaviour</th></tr>
</thead>
<tbody>
<tr><td>Early wave</td><td><code>L7–L11</code></td><td align="right"><code>peaks ~850 at L11</code></td><td>first buildup</td></tr>
<tr><td>Mid plateau</td><td><code>L12–L19</code></td><td align="right"><code>200–450</code></td><td>modest contributions</td></tr>
<tr><td><strong>Dampener</strong></td><td><code><strong>L20</strong></code></td><td align="right"><code><strong>~−200</strong></code></td><td><strong>pulls projection down</strong></td></tr>
<tr><td>Late wave</td><td><code>L23–L32</code></td><td align="right"><code>L29–L30 ≈ 1,650 each</code></td><td>dominant ramp</td></tr>
<tr><td>Post-measurement</td><td><code>L33</code></td><td align="right"><code>−17,311</code></td><td>RMSNorm artifact, not part of projection</td></tr>
</tbody>
</table>

**Interpretation.** The separation at `L32` is mostly *late*-built. Tejas's L15 intervention works because downstream layers then amplify the L15 residual `~4×`. `L20` is an open question — a "pump-the-brakes" layer actively pulling the projection down.

### 3. MLP transcoders only capture ~0.4% of the refusal signal

<table>
<thead>
<tr><th>Metric</th><th align="right">Mean</th><th align="right">Std</th></tr>
</thead>
<tbody>
<tr><td>Full projection <code>r · h[L=32]</code></td><td align="right"><code>17,230</code></td><td align="right"><code>1,986</code></td></tr>
<tr><td>Σ feature attributions</td><td align="right"><code>70.47</code></td><td align="right"><code>17.6</code></td></tr>
<tr><td><strong>MLP ratio</strong></td><td align="right"><code><strong>0.404%</strong></code></td><td align="right"><code>0.077%</code></td></tr>
</tbody>
</table>

The `~99.6%` gap is carried by **attention heads + embeddings** — a property of the Gemma Scope transcoder set (MLP-only). Implication: any mechanistic intervention relying purely on transcoded MLP features will move `<1%` of the measured refusal signal. Attention-head attribution is a known gap (task S3).

### 4. Jailbreaks produce strong, statistically real changes

Five jailbreak classes × 50 prompts, paired bare-vs-JB, all statistically significant.

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/02b_stats/class_comparison.png" alt="Bar chart comparing positive, net, and negative attribution across bare and five jailbreak classes with significance stars" width="800">
<figcaption><em><strong>Figure 4.</strong> Positive (pro-refusal), net, and negative (anti-refusal) attribution per class. Error bars are ±1 std on net; *, **, *** are Wilcoxon p-value significance thresholds.</em></figcaption>
</figure>

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/02b_stats/distribution_by_class.png" alt="Violin plots with inner box plots and jittered scatter points showing the net-attribution distribution per class, with a dashed reference line at the bare mean" width="900">
<figcaption><em><strong>Figure 5.</strong> Net-attribution distribution by class (<code>n=50</code> prompts). Violin shape + inner box + jittered per-prompt points. Dashed grey line at bare mean (<code>+70.5</code>).</em></figcaption>
</figure>

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/02b_stats/effect_sizes.png" alt="Horizontal bar chart of Cohen's d for each jailbreak class with small, medium, and large threshold guides" width="700">
<figcaption><em><strong>Figure 6.</strong> Cohen's d per class. Dotted/dashed/solid grey verticals mark small/medium/large effect thresholds.</em></figcaption>
</figure>

Ordering by effect size:

```text
Analytical       d = −2.37   (strongest; 49/50 consistency)
Fiction          d = −1.57
Cognitive reframe d = −1.41
Roleplay         d = −0.91
Completion       d = +0.27   (INVERTED — discussed below)
```

The distribution plot (Figure 5) adds shape evidence to the effect-size summary:

- **Bare is remarkably tight** — IQR roughly `60–95`, no long tail. The model's default refusal behaviour is very consistent across our 50 prompts. **Every JB class broadens this distribution substantially** — jailbreaks *destabilize* the circuit, not just shift its mean.
- **Analytical is the only class whose median goes firmly negative.** Not just dampening — it flips the net sign on most prompts. This is what "most jailbreakable" actually looks like at the distribution level.
- **Fiction has the longest lower tail (`~−55`)** but retains a small residual cluster above `+50` — some prompts *resist* the fiction framing. Worth asking later whether that's a prompt-category pattern (A7 / A8 can answer).
- **Roleplay has the widest IQR** (`~−10 to +70`) despite only a `−38.7` mean shift — it's the most volatile class, a mix of total-flip and no-effect prompts.
- **Completion's mass sits *above* the bare-mean line** with a narrow tail down to ~0 — the "high-and-tight with a minority tail" shape. This visually previews the recruitment mechanism unpacked in #6.

### 5. Fiction reorganizes the circuit; completion preserves it

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/02b_stats/feature_comparison_summary.png" alt="Grouped bar chart showing shared, bare-only, JB-only, sign-flipped, dampened, and amplified-anti feature counts per class" width="800">
<figcaption><em><strong>Figure 7.</strong> Mean feature counts per JB class across six comparison buckets (shared / bare-only / JB-only / sign-flipped / dampened / amplified-anti).</em></figcaption>
</figure>

<table>
<thead>
<tr><th>Class</th><th align="right">Shared with bare</th><th align="right">JB-only features</th><th align="right">Sign-flipped</th></tr>
</thead>
<tbody>
<tr><td><strong>Fiction</strong></td><td align="right"><code>58.5%</code></td><td align="right"><code><strong>62.5%</strong></code></td><td align="right"><code><strong>25.6%</strong></code></td></tr>
<tr><td>Analytical</td><td align="right"><code>63.0%</code></td><td align="right"><code>56.6%</code></td><td align="right"><code>23.1%</code></td></tr>
<tr><td>Cognitive reframe</td><td align="right"><code>64.4%</code></td><td align="right"><code>51.8%</code></td><td align="right"><code>20.3%</code></td></tr>
<tr><td>Roleplay</td><td align="right"><code>65.7%</code></td><td align="right"><code>56.3%</code></td><td align="right"><code>17.5%</code></td></tr>
<tr><td>Completion</td><td align="right"><code><strong>73.5%</strong></code></td><td align="right"><code>45.3%</code></td><td align="right"><code>16.6%</code></td></tr>
</tbody>
</table>

Fiction uses the most novel features and flips the most shared features — it's not suppressing the circuit, it's *rewiring* it. Completion in contrast keeps almost three-quarters of the bare circuit intact — consistent with its paradoxical *strengthening* effect (see #6).

### 6. Completion is the paradox: it recruits *more* refusal

Completion-style JBs ("I cannot help with that, but what I can suggest is…") should *weaken* refusal but instead **strengthen it by `+7.2%`** (Cohen's d `= +0.27`).

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/02b_stats/per_prompt_deltas.png" alt="Per-prompt delta (JB − bare) bar plots for each of the five jailbreak classes" width="900">
<figcaption><em><strong>Figure 8.</strong> Per-prompt deltas (<code>JB − bare</code>) for each class. Red bars suppress refusal, blue bars strengthen. Completion is visibly bimodal: 35 positive, 15 negative.</em></figcaption>
</figure>

<table>
<thead>
<tr><th>Class</th><th align="right">dPos</th><th align="right">dNeg</th><th>Interpretation</th></tr>
</thead>
<tbody>
<tr><td>Roleplay</td><td align="right"><code>−22.5</code></td><td align="right"><code>−16.2</code></td><td>Balanced dampening</td></tr>
<tr><td>Fiction</td><td align="right"><code>−43.1</code></td><td align="right"><code>−22.2</code></td><td>Balanced dampening</td></tr>
<tr><td>Analytical</td><td align="right"><code>−44.7</code></td><td align="right"><code>−29.0</code></td><td>Balanced dampening</td></tr>
<tr><td><strong>Completion</strong></td><td align="right"><code><strong>+19.7</strong></code></td><td align="right"><code>−14.6</code></td><td><strong>Pro-refusal recruitment</strong></td></tr>
<tr><td>Cognitive reframe</td><td align="right"><code>−33.8</code></td><td align="right"><code>−16.4</code></td><td>Dampening-dominant</td></tr>
</tbody>
</table>

Every other class dampens both halves. Completion is the only one that *grows* the pro-refusal half (`+14.6%`) while weakening the anti-refusal half — two mechanisms pulling in opposite directions, with pro-refusal winning. `35/50` prompts go up, `15/50` go down.

Two independent views converge on this:
- **Figure 5** (distribution by class) shows completion's mass sitting *above* the bare-mean line, with a thin tail down toward zero — "high-and-tight with a minority tail."
- **Figure 8** (per-prompt deltas) shows the same pattern in delta space: a cluster of blue (strengthening) bars with a minority of red (weakening) bars — visibly bimodal.

Both views suggest there's a specific set of features being *recruited* by the completion framing. Identifying which ones is `A2` — next in the analysis plan.

### 7. JB-affected features concentrate in the late regime (L24–L32)

Counting where each comparison bucket's features live by layer reveals a strikingly consistent spatial signature across all three mechanisms:

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/04_labels/features_by_layer.png" alt="Three-row histogram of feature counts by layer for sign-flipped, dampened, and amplified-anti buckets, with the L24–L32 band shaded" width="900">
<figcaption><em><strong>Figure 9.</strong> Feature counts by layer for the three comparison buckets. Shaded band (<code>L24–L32</code>) is the predicted peak window. All three bucket totals match <code>label_coverage.json</code> exactly (<code>603 / 115 / 117</code>).</em></figcaption>
</figure>

<table>
<thead>
<tr><th>Bucket</th><th align="right">Total</th><th align="right">% in L24–L32</th><th align="right">% in L0–L14</th><th>Peak layer</th><th>Top 3 layers</th></tr>
</thead>
<tbody>
<tr><td>Sign-flipped</td><td align="right"><code>603</code></td><td align="right"><code><strong>83.1%</strong></code></td><td align="right"><code>3.3%</code></td><td><code>L30 (89)</code></td><td><code>L30, L31, L32</code></td></tr>
<tr><td>Dampened</td><td align="right"><code>115</code></td><td align="right"><code><strong>80.0%</strong></code></td><td align="right"><code>0.0%</code></td><td><code>L30 (16)</code></td><td><code>L30, L31, L29</code></td></tr>
<tr><td>Amplified-anti</td><td align="right"><code>117</code></td><td align="right"><code><strong>70.9%</strong></code></td><td align="right"><code>1.7%</code></td><td><code>L27 (15)</code></td><td><code>L27, L29, L25</code></td></tr>
</tbody>
</table>

Key observations:

- **~`80%` of all JB-affected features live in the `L24–L32` peak band.** Not just "late layers" — a specific 9-layer window ending at the measurement layer. This **exactly matches** the "late wave" signal-assembly band from #2 (`L23–L32`), and the Regime C cluster from the cosine heatmap in #1.
- **Dampening is a pure late-layer phenomenon.** Zero dampened features below `L15`. Pro-refusal features only *exist* in the late readout regime, so there's nothing to weaken earlier.
- **Amplified-anti peaks earlier** (`L27`) than sign-flipped and dampened (both `L30`). Anti-refusal recruitment happens slightly further from the measurement layer — a candidate for subcircuit decomposition in Stage 07.
- **Sign-flipping follows the regime pivot.** Only `3.3%` of sign-flipped features live in `L0–L14`; the ramp starts at `L15` (coincident with Pivot 1 from #1's cosine matrix) and accelerates through `L24–L32`. Flipping a feature's sign only becomes meaningful once the direction has rotated into Regime C.
- **Amplified-anti leaks into `L33`** (`9.4%`) vs `0%` sign-flipped and `3.5%` dampened. This is an open thread: some anti-refusal activations survive the RMSNorm magnitude collapse, unlike the other two mechanisms. Suggests anti-refusal recruitment happens partly in attention/embeddings at the last layer.

The takeaway: **every JB mechanism we've catalogued exploits the same `L24–L32` band where refusal is actively assembled**. If we want to intervene on the circuit rather than steer the direction, targeted interventions in this window should be substantially more effective than broad-layer interventions.

### 8. Only 12.7% of JB-affected features are universal — each class recruits its own subset

**What these features are.** Every feature in this analysis is an **MLP transcoder feature** from the Gemma Scope `transcoder_all/width_16k_l0_small_affine` set, keyed as `L{layer}:F{feature_idx}` (e.g. `L29:F1066`). A feature enters the analysis when it appears in the top-50 attribution list for at least one `(prompt, condition)` pair across the 50-prompt run. Per Section 7, roughly `80%` of the 788 unique features sit in the `L24–L32` band — late-layer MLP features. The UpSet below aggregates across all three comparison buckets (sign-flipped / dampened / amplified-anti) for each of the 5 JB classes.

**Important caveat — `bare` is not a column in this plot.** Every feature shown here is already "bare-relative" by construction: the comparison buckets are defined *relative to bare*, so bare is implicit. A separate pool in `feature_labels.json`'s `conditions_seen` field tracks which of the 6 conditions (`bare` + 5 JBs) had each feature in its own top-50, giving a striking complementary view:

<table>
<thead>
<tr><th>Condition</th><th align="right">Features ever in top-50</th></tr>
</thead>
<tbody>
<tr><td>roleplay</td><td align="right"><code>434</code></td></tr>
<tr><td>cognitive_reframe</td><td align="right"><code>426</code></td></tr>
<tr><td>fiction</td><td align="right"><code>401</code></td></tr>
<tr><td>completion</td><td align="right"><code>384</code></td></tr>
<tr><td>analytical</td><td align="right"><code>362</code></td></tr>
<tr><td><strong>bare</strong></td><td align="right"><code><strong>100</strong></code></td></tr>
</tbody>
</table>

Under bare refusal, only ~`100` MLP features ever carry top-50 attribution weight. Every JB class recruits **3–4× more features** into the top-50. That's a real finding in its own right: **bare refusal is a focused, concentrated computation; jailbreaks *broaden* the refusal circuit, pulling hundreds of additional MLP features into significant attribution**. Whether that broadening *helps* refusal (completion) or *scatters* it (roleplay) is what the UpSet below disambiguates — but the raw broadening itself is consistent across all 5 JBs. Building a `bare`-inclusive 6-way UpSet is a tracked follow-up.

<figure>
<img src="../../data/results/pipeline_runs/run_20260417_010035/04_labels/feature_class_upset.png" alt="UpSet plot of feature membership across the five jailbreak classes, showing intersection sizes ranked by cardinality" width="900">
<figcaption><em><strong>Figure 10.</strong> Feature–class overlap UpSet across the 5 JB classes (bare is implicit in the comparison-bucket definition). Each column is a class-subset; bar height is the number of unique features affected in exactly that combination. Left-side horizontal bars show per-class totals (features appearing in that class's bucket, any type).</em></figcaption>
</figure>

<table>
<thead>
<tr><th>Class-set size</th><th align="right">Features</th><th align="right">% of 788</th><th>Interpretation</th></tr>
</thead>
<tbody>
<tr><td>1 (class-exclusive)</td><td align="right"><code>363</code></td><td align="right"><code><strong>46.1%</strong></code></td><td>Single-class-specific</td></tr>
<tr><td>2</td><td align="right"><code>147</code></td><td align="right"><code>18.7%</code></td><td>Shared by a pair</td></tr>
<tr><td>3</td><td align="right"><code>103</code></td><td align="right"><code>13.1%</code></td><td>Shared by a triple</td></tr>
<tr><td>4</td><td align="right"><code>75</code></td><td align="right"><code>9.5%</code></td><td>Shared by four classes</td></tr>
<tr><td><strong>5 (universal)</strong></td><td align="right"><code><strong>100</strong></code></td><td align="right"><code><strong>12.7%</strong></code></td><td><strong>Canonical refusal circuit — all 5 JBs</strong></td></tr>
</tbody>
</table>

Per-class totals paired with their effect sizes from #4:

<table>
<thead>
<tr><th>Class</th><th align="right">Total features</th><th align="right">Class-exclusive</th><th align="right">Cohen's d</th></tr>
</thead>
<tbody>
<tr><td>Roleplay</td><td align="right"><code>396</code></td><td align="right"><code><strong>107</strong></code></td><td align="right"><code>−0.91</code></td></tr>
<tr><td>Cognitive reframe</td><td align="right"><code>371</code></td><td align="right"><code>66</code></td><td align="right"><code>−1.41</code></td></tr>
<tr><td>Completion</td><td align="right"><code>363</code></td><td align="right"><code>78</code></td><td align="right"><code>+0.27</code></td></tr>
<tr><td>Fiction</td><td align="right"><code>336</code></td><td align="right"><code>69</code></td><td align="right"><code>−1.57</code></td></tr>
<tr><td>Analytical</td><td align="right"><code><strong>300</strong></code></td><td align="right"><code><strong>43</strong></code></td><td align="right"><code><strong>−2.37</strong></code></td></tr>
</tbody>
</table>

Observations:

- **A canonical refusal circuit exists.** The 100-feature 5-class intersection is the subset consistently affected regardless of JB framing. This is the natural anchor for Stage 07 subcircuit identification — intervening on these 100 should disrupt all five jailbreaks simultaneously.
- **But class-specificity dominates.** `46.1%` of all JB-affected features appear in only ONE class. Each jailbreak framing recruits its own specialized subset — it's not "more or less of a shared circuit."
- **Breadth is inversely related to strength.** Analytical affects the **fewest** features (`300`) but has the **strongest** effect (`d = −2.37`); roleplay affects the **most** (`396`) but the **weakest** (`d = −0.91`). Analytical operates on canonical pro-refusal features with precision; roleplay scatters effects across many targets — which is why its distribution in #4 has the widest IQR.
- **Roleplay is the most idiosyncratic class.** 107 class-exclusive features — far more than any other. Combined with its wide distribution shape, roleplay appears to trigger a diverse, prompt-dependent feature soup rather than a clean mechanism.
- **Largest pair-intersection is `completion + roleplay`** (26 features). Plausibly reflects "social/conversational" features the other three framings don't touch.
- **Completion's 78 class-exclusive features** are a direct A2 target — these are the specific features being *recruited* to produce the paradoxical +7.2% strengthening.

This sharpens the Stage 07 research question: separate the canonical 100 (affected universally) from each class's exclusive subset, and ask whether the class-exclusives have coherent functional roles (e.g. roleplay-exclusives all encoding "persona" features, completion-exclusives all encoding "refusal template" features, etc.).

---

## Gaps & open questions

Items flagged for future pipeline stages or deeper analysis.

<table>
<thead>
<tr><th>Gap</th><th>Status</th><th>Where it gets answered</th></tr>
</thead>
<tbody>
<tr><td>Does the model actually refuse bare prompts / comply under JB?</td><td>Not measured</td><td><code>A1</code> · Stage 02c — coherence + refusal classification</td></tr>
<tr><td>Which specific features are recruited by completion?</td><td>Not identified</td><td><code>A2</code> · Stage 02b extension — top <code>Δattribution</code> under completion</td></tr>
<tr><td>Why does bare refusal use only ~100 features while every JB recruits 3–4× more? Which specific bare features survive / disappear / get replaced under each JB?</td><td>Counts known, identities not mapped</td><td><code>A7+</code> · 6-way UpSet built from <code>feature_labels.json[conditions_seen]</code></td></tr>
<tr><td>What does <code>L20</code> do? Why is it the only negative-contribution layer?</td><td>Open</td><td>Dedicated follow-up (not in current plan)</td></tr>
<tr><td>What features live in Regime A (<code>L5–L13</code>) vs Regime C (<code>L18–L33</code>)? They're weakly anti-correlated</td><td>Open</td><td>Stage 07 subcircuit identification</td></tr>
<tr><td>What happens at the <code>L13→L14</code> pivot? Is there a specific attention head / MLP that triggers the rotation?</td><td>Open</td><td>Task <code>S3</code> (attention-head attribution) + follow-up probe</td></tr>
<tr><td><code>99.6%</code> of the refusal signal lives in attention + embeddings — which heads?</td><td>Not attributed</td><td>Task <code>S3</code> (attention-head attribution)</td></tr>
<tr><td>Are any two JB classes structurally similar (shared recruited features)?</td><td>Not cross-tabulated</td><td>Stage 07 subcircuit identification</td></tr>
</tbody>
</table>

---

## Deployment

### Local smoke test (no GPU required)

```bash
PYTHONPATH=src python3 -m pytest scripts/pipeline/tests/test_pipeline_local.py -v -W ignore::DeprecationWarning
```

Expected: 60/60 pass. Uses an existing run directory as a fallback for stages that need heavy compute.

### Full RunPod run (stages 01 → 04)

Prerequisites on the pod:
- `/workspace/Refusal-Lens` clone on the `foundation` branch
- `/workspace/venv` with torch (nightly for Blackwell GPUs: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`), transformers, scipy, and the vendored circuit-tracer installed
- HuggingFace token available (either `~/.cache/huggingface/token` or `HF_TOKEN` env var)
- A GitHub fine-grained PAT exported as `GHP_TOKEN` for auto-push (scope: contents:write on the repo)
- Recommended hardware: **RTX A6000 (48GB)** or **A40 (48GB)** — model + transcoders fit in ~15–20 GB, Blackwell cards need PyTorch nightly

**Step 1 — start the experiment in a detached tmux session:**

```bash
tmux new-session -d -s pipeline -n run "cd /workspace/Refusal-Lens && source /workspace/venv/bin/activate && python -u scripts/pipeline/tests/test_runpod_1_4.py --n-prompts 50 2>&1 | tee /workspace/pipeline_output.log"
```

**Step 2 — export `GHP_TOKEN` in your SSH shell before adding the watcher window** (so the new window inherits it):

```bash
export GHP_TOKEN=ghp_...   # paste your fresh token
echo $GHP_TOKEN            # sanity check
```

**Step 3 — add the watcher window that prints heartbeats and auto-commits on completion:**

```bash
tmux new-window -t pipeline -n watcher "L=/workspace/pipeline_output.log; echo \"[watcher] monitoring \$L\"; until grep -q 'PIPELINE COMPLETE' \$L 2>/dev/null; do grep -q 'Pipeline stopped at Stage' \$L 2>/dev/null && { echo '[watcher] FAILED'; exit 1; }; echo \"[watcher \$(date +%H:%M:%S)] alive — tail: \$(tail -n1 \$L 2>/dev/null | cut -c1-120)\"; sleep 30; done && echo '[watcher] pipeline complete — committing...' && cd /workspace/Refusal-Lens && R=\$(grep -oE 'data/results/pipeline_runs/run_[0-9_]+' \$L | head -1) && echo \"[watcher] run_dir: \$R\" && cp \$L \$R/pipeline_output.log && U=\$(git remote get-url origin | sed \"s|https://|https://x-access-token:\$GHP_TOKEN@|\") && git pull --rebase \$U foundation && git add \$R && git commit -n -m 'pipeline run: stages 01-04 (50 prompts)' && git push \$U foundation && echo '[watcher] DONE — pushed to foundation'"
```

**Step 4 — attach to watch progress:**

```bash
tmux attach -t pipeline     # Ctrl+b n to switch windows, Ctrl+b d to detach
```

**Expected runtimes** (48 GB card, 50 prompts):
- Stage 01: ~1 min
- Stage 02: ~3–4 hours
- Stage 02b: ~1 min
- Stage 03: ~5–10 min
- Stage 04: ~2 min

### Resuming after a crash

Stage 02 checkpoints after every prompt. To resume the attribution stage without redoing stages 01:

```bash
python -u scripts/pipeline/tests/test_runpod_1_4.py \
  --n-prompts 50 \
  --run-dir data/results/pipeline_runs/run_YYYYMMDD_HHMMSS \
  --skip-stage 01 \
  --resume
```

To re-run only the lightweight stages (e.g. after a 02b bug fix):

```bash
python -u scripts/pipeline/tests/test_runpod_1_4.py \
  --n-prompts 50 \
  --run-dir data/results/pipeline_runs/run_YYYYMMDD_HHMMSS \
  --skip-stage 01 02
```

### Security notes

- `GHP_TOKEN` appears in `ps` output during the `git push` — acceptable on a private pod, rotate the token afterward if the pod is shared.
- Prefer fine-grained PATs scoped to this repo's contents, not classic tokens with full-repo access.
- The watcher writes a copy of `pipeline_output.log` into the run directory before committing, so the log is preserved as part of the run artifact.

---

## Output directory layout

```
data/results/pipeline_runs/run_YYYYMMDD_HHMMSS/
├── run_config.json
├── config.json
├── pipeline.txt                       # stage-level log from test runner
├── pipeline_output.txt                # full stdout stream (added by watcher)
├── 01_direction/
│   ├── refusal_direction.pt           # normalized r_hat at best_separation_layer
│   ├── unnormalized_r.pt              # unnormalized r (all layers)
│   ├── direction_metadata.json        # per-layer separation, cosines, best layers
│   └── directions/layer_XX.pt         # normalized r_hat per layer
├── 02_attribution/
│   ├── attribution_results.json       # per-prompt, per-condition feature attributions
│   ├── attribution_checkpoint.json    # resume state
│   └── feature_comparison_aggregate.json
├── 02b_stats/
│   ├── statistical_analysis.json
│   ├── EXPERIMENT_SUMMARY.md
│   ├── class_comparison.png
│   ├── distribution_by_class.png
│   ├── per_prompt_deltas.png
│   ├── effect_sizes.png
│   ├── feature_comparison_summary.png
│   ├── separation_by_layer.png
│   └── cosine_heatmap.png
├── 03_verification/
│   ├── verification_results.json       # includes per_layer_aggregate (A4)
│   ├── per_layer_decomposition.json
│   └── per_layer_contribution.png      # A4 bar chart
└── 04_labels/
    ├── feature_labels.json            # {L:F → {top_logits, bottom_logits, examples, ...}}
    ├── feature_labels_cache.json      # raw HF payload cache (survives re-runs)
    ├── feature_comparison_labeled.json
    ├── label_coverage.json
    ├── layer_histogram.json           # A8 per-bucket layer counts
    ├── features_by_layer.png          # A8 histogram plot
    ├── feature_class_sets.json        # A7 per-class feature membership
    ├── feature_class_upset.png        # A7 UpSet overlap plot
    └── top_features_report.md
```

---

## Known caveats

- **Gemma-3-4b-it is not on Neuronpedia** — labels come from the HuggingFace Gemma Scope dashboards (byte-range HTTP requests against `index.json.gz`). Many top-token patches are polyglot / byte-level noise; semantic interpretation requires activation examples on concrete prompts.
- **Transcoders cover MLP only** — the MLP ratio of ~0.4% means 99.6% of the refusal signal is carried by attention + embeddings. This is a property of the transcoder set, not a bug.
- **Layer 33 anomaly** — forward hooks capture pre-RMSNorm, `hidden_states` captures post-RMSNorm. Stage 01 uses `hidden_states`, which is why L33 separation shows as 287 (not ~20k). Expected.
- **Position -2** is the "model" token in Gemma-3's chat template. Stage 01 and Stage 02 use this position.
- **PyTorch + Blackwell GPUs** — stable PyTorch only supports up to sm_90. RTX PRO 6000 Blackwell (sm_120) needs `torch --pre` from the nightly cu128 index.
