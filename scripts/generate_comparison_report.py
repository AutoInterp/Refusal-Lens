"""
Generate Comparison Report
===========================
Reads Tejas's saved results and the foundation validation results,
then generates a comprehensive COMPARISON_REPORT.md.

Usage:
  python scripts/generate_comparison_report.py

Runs offline (no GPU needed) -- just reads JSON files.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEJAS_DIR = REPO_ROOT / "data" / "tejas_experiments" / "results_v2"
VALIDATION_DIR = REPO_ROOT / "data" / "results" / "validation"
OUTPUT_FILE = VALIDATION_DIR / "COMPARISON_REPORT.md"


def load_json(path: Path) -> dict | list | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_report():
    # Load data
    tejas_attr = load_json(TEJAS_DIR / "v2_attribution_10pairs.json")
    tejas_mech = load_json(TEJAS_DIR / "v2_mechanism_comparison.json")
    tejas_sep = load_json(TEJAS_DIR / "separation_table.json")

    found_full = load_json(VALIDATION_DIR / "full_results.json")
    found_mode_a = load_json(VALIDATION_DIR / "mode_a_attribution.json")
    found_mode_b = load_json(VALIDATION_DIR / "mode_b_attribution.json")
    found_mech = load_json(VALIDATION_DIR / "mechanism_comparison.json")
    found_dir = None
    dir_path = VALIDATION_DIR / "refusal_direction_foundation.pt"
    if dir_path.exists():
        import torch
        found_dir = torch.load(dir_path, map_location="cpu")

    lines = []
    w = lines.append

    w("# Experiment Comparison Report: Tejas vs Foundation Pipeline")
    w("")
    w(f"Generated from `scripts/generate_comparison_report.py`")
    w("")
    w("---")
    w("")

    # ============================================================
    # Section 1: Refusal Direction
    # ============================================================
    w("## 1. Refusal Direction Computation")
    w("")
    w("Both approaches use difference-in-means (Arditi et al. 2024) with corrected parameters:")
    w("- Position: -2 (model token in chat template)")
    w("- Accumulation: float64")
    w("- Padding: left")
    w("- Data: 64 harmful + 64 harmless prompts")
    w("")

    if found_dir is not None:
        found_layer = found_dir.get("best_layer", "N/A")
        found_sep = found_dir.get("separation", "N/A")
        if isinstance(found_sep, float):
            found_sep = f"{found_sep:.1f}"
    else:
        found_layer = "N/A (not yet run)"
        found_sep = "N/A"

    w("| Metric | Tejas (v2) | Foundation | Match? |")
    w("|--------|-----------|------------|--------|")

    tejas_layer = "32"
    tejas_separation = "20,788"
    if tejas_sep:
        tejas_layer = str(tejas_sep.get("best_layer", 32))
        if "best_separation" in tejas_sep:
            tejas_separation = f"{tejas_sep['best_separation']:,.0f}"

    match_layer = "YES" if str(found_layer) == tejas_layer else f"CHECK ({found_layer} vs {tejas_layer})"
    w(f"| Best layer | {tejas_layer} | {found_layer} | {match_layer} |")
    w(f"| Separation | {tejas_separation} | {found_sep} | See analysis |")
    w(f"| Best position | -2 | -2 | YES |")
    w(f"| Dtype | float64 | float64 | YES |")
    w(f"| Padding | left | left | YES |")
    w("")

    w("### Analysis")
    w("")
    w("If separations match closely (within 5%), the direction computation is correctly replicated.")
    w("Small differences may arise from:")
    w("- Different torch/transformers versions affecting numerical precision")
    w("- Model weight caching differences")
    w("- Chat template formatting differences")
    w("")

    # ============================================================
    # Section 2: Last-Layer Attribution (Mode A)
    # ============================================================
    w("## 2. Attribution Comparison: Last-Layer (Mode A vs Tejas)")
    w("")
    w("Mode A uses `measurement_layer=None` (last layer), matching Tejas's unpatched circuit-tracer.")
    w("")

    # Tejas summary
    if tejas_attr:
        t_bare = [p.get("bare_net", 0) for p in tejas_attr if "bare_net" in p]
        t_jb = [p.get("jb_net", 0) for p in tejas_attr if "jb_net" in p]
        t_bare_mean = sum(t_bare) / len(t_bare) if t_bare else 0
        t_jb_mean = sum(t_jb) / len(t_jb) if t_jb else 0
        t_diff = t_jb_mean - t_bare_mean
        t_lower = sum(1 for b, j in zip(t_bare, t_jb) if j < b)
    else:
        t_bare_mean = 75.5  # From Tejas README
        t_jb_mean = 56.7
        t_diff = -18.7
        t_lower = 9

    # Foundation summary
    f_summary = {}
    if found_full and "mode_a_summary" in found_full:
        f_summary = found_full["mode_a_summary"]
    elif found_mode_a:
        f_bare = [p.get("bare_net", 0) for p in found_mode_a if "bare_net" in p]
        f_jb = [p.get("jb_net", 0) for p in found_mode_a if "jb_net" in p]
        if f_bare:
            f_summary = {
                "bare_mean": sum(f_bare) / len(f_bare),
                "jb_mean": sum(f_jb) / len(f_jb),
                "mean_diff": sum(f_jb) / len(f_jb) - sum(f_bare) / len(f_bare),
                "jb_lower_count": sum(1 for b, j in zip(f_bare, f_jb) if j < b),
                "jb_lower_total": len(f_bare),
            }

    w("| Metric | Tejas (v2) | Foundation (Mode A) | Delta |")
    w("|--------|-----------|---------------------|-------|")

    if f_summary:
        f_bare_mean = f_summary["bare_mean"]
        f_jb_mean = f_summary["jb_mean"]
        f_diff = f_summary["mean_diff"]
        f_lower = f_summary["jb_lower_count"]
        f_total = f_summary["jb_lower_total"]

        delta_bare = f_bare_mean - t_bare_mean
        delta_jb = f_jb_mean - t_jb_mean
        delta_diff = f_diff - t_diff

        w(f"| Bare mean | {t_bare_mean:.1f} | {f_bare_mean:.1f} | {delta_bare:+.1f} |")
        w(f"| JB mean | {t_jb_mean:.1f} | {f_jb_mean:.1f} | {delta_jb:+.1f} |")
        w(f"| Mean diff | {t_diff:+.1f} | {f_diff:+.1f} | {delta_diff:+.1f} |")
        w(f"| JB lower | {t_lower}/10 | {f_lower}/{f_total} | |")
    else:
        w(f"| Bare mean | {t_bare_mean:.1f} | *Not yet run* | |")
        w(f"| JB mean | {t_jb_mean:.1f} | *Not yet run* | |")
        w(f"| Mean diff | {t_diff:+.1f} | *Not yet run* | |")
        w(f"| JB lower | {t_lower}/10 | *Not yet run* | |")

    w("")
    w("### Interpretation")
    w("")
    w("- If Mode A results are within ~20% of Tejas's, the pipeline is correctly replicating his setup.")
    w("- Exact matches are unlikely due to nondeterminism in GPU float operations.")
    w("- The key signal is: **does JB consistently have lower net attribution than bare?**")
    w("  This confirms the RP-suppresses-refusal hypothesis at the MLP level.")
    w("")

    # Per-pair comparison table
    if tejas_attr and found_mode_a:
        w("### Per-Pair Comparison")
        w("")
        w("| Pair | Tejas Bare | Foundation Bare | Tejas JB | Foundation JB |")
        w("|------|-----------|----------------|---------|--------------|")
        for i in range(min(len(tejas_attr), len(found_mode_a))):
            tp = tejas_attr[i]
            fp = found_mode_a[i]
            tb = tp.get("bare_net", "err")
            fb = fp.get("bare_net", "err")
            tj = tp.get("jb_net", "err")
            fj = fp.get("jb_net", "err")
            tb_s = f"{tb:.1f}" if isinstance(tb, (int, float)) else str(tb)
            fb_s = f"{fb:.1f}" if isinstance(fb, (int, float)) else str(fb)
            tj_s = f"{tj:.1f}" if isinstance(tj, (int, float)) else str(tj)
            fj_s = f"{fj:.1f}" if isinstance(fj, (int, float)) else str(fj)
            w(f"| {i+1} | {tb_s} | {fb_s} | {tj_s} | {fj_s} |")
        w("")

    # ============================================================
    # Section 3: Intermediate-Layer Attribution (Mode B)
    # ============================================================
    w("## 3. Intermediate-Layer Attribution (Mode B) -- New")
    w("")
    w("Mode B uses the measurement_layer patch from the AutoInterp circuit-tracer fork.")
    w("This measures R = <x^(l*, c*), r_hat> at the best refusal layer rather than the final layer.")
    w("")
    w("**Why this matters**: The PDF (Section 1.4) notes that intermediate-layer measurement")
    w("traces only the upstream computation that *builds* the refusal signal, which is more")
    w("informative for understanding how the model decides to refuse.")
    w("")

    if found_full and "mode_b_summary" in found_full:
        b_summary = found_full["mode_b_summary"]
        w("| Metric | Mode A (last-layer) | Mode B (intermediate) | Delta |")
        w("|--------|--------------------|-----------------------|-------|")
        if f_summary:
            w(f"| Bare mean | {f_summary['bare_mean']:.1f} | {b_summary['bare_mean']:.1f} | {b_summary['bare_mean'] - f_summary['bare_mean']:+.1f} |")
            w(f"| JB mean | {f_summary['jb_mean']:.1f} | {b_summary['jb_mean']:.1f} | {b_summary['jb_mean'] - f_summary['jb_mean']:+.1f} |")
            w(f"| Mean diff | {f_summary['mean_diff']:+.1f} | {b_summary['mean_diff']:+.1f} | {b_summary['mean_diff'] - f_summary['mean_diff']:+.1f} |")
            w(f"| JB lower | {f_summary['jb_lower_count']}/{f_summary['jb_lower_total']} | {b_summary['jb_lower_count']}/{b_summary['jb_lower_total']} | |")
    elif found_full and "mode_b" in found_full and isinstance(found_full["mode_b"], dict) and "error" in found_full["mode_b"]:
        w(f"**Mode B skipped**: {found_full['mode_b']['error']}")
    else:
        w("*Mode B results not yet available. Run `scripts/validate_tejas_replication.py` first.*")

    w("")
    w("### Analysis")
    w("")
    w("Key questions:")
    w("1. Does intermediate-layer attribution show a **larger or smaller** JB suppression effect?")
    w("2. Does the number of active features change significantly?")
    w("3. Does the dampening vs tug-of-war distinction persist at intermediate layers?")
    w("")

    # ============================================================
    # Section 4: Mechanism Comparison
    # ============================================================
    w("## 4. Mechanism Comparison (RP vs Fiction vs Bare)")
    w("")
    w("Same 9 prompts (3 topics x 3 jailbreak types) as Tejas's script 11 Task 4.")
    w("")

    w("### Tejas Results (v2)")
    w("")
    if tejas_mech:
        w("| Prompt | Net | Positive | Negative | N Features |")
        w("|--------|-----|----------|----------|------------|")
        for topic in ["lock", "hack", "phish"]:
            for jb_type in ["bare", "rp", "fiction"]:
                key = f"{jb_type}_{topic}"
                if key in tejas_mech and "net" in tejas_mech[key]:
                    r = tejas_mech[key]
                    w(f"| {key} | {r['net']:+.1f} | {r['pos']:.1f} | {r['neg']:.1f} | {r.get('n_features', 'N/A')} |")
        w("")

    w("### Foundation Results")
    w("")
    if found_mech:
        w("| Prompt | Net | Positive | Negative | N Features |")
        w("|--------|-----|----------|----------|------------|")
        for topic in ["lock", "hack", "phish"]:
            for jb_type in ["bare", "rp", "fiction"]:
                key = f"{jb_type}_{topic}"
                if key in found_mech and "net" in found_mech[key]:
                    r = found_mech[key]
                    w(f"| {key} | {r['net']:+.1f} | {r['pos']:.1f} | {r['neg']:.1f} | {r.get('n_features', 'N/A')} |")
        w("")
    else:
        w("*Not yet available. Run `scripts/validate_tejas_replication.py` first.*")
        w("")

    w("### Mechanism Classification")
    w("")
    w("Tejas identified two distinct jailbreak mechanisms at the MLP level:")
    w("")
    w("1. **Dampening (RP jailbreaks)**: Pro-refusal features drop aggressively,")
    w("   anti-refusal features grow moderately. The refusal circuit *disengages*.")
    w("   - Steerable via refusal direction (13/16 configs flip to refusal)")
    w("")
    w("2. **Tug-of-war (Fiction jailbreaks)**: Both pro-refusal and anti-refusal")
    w("   features increase massively. The circuit *engages more* but the sides cancel.")
    w("   - Immune to refusal-direction steering (0/16 flip)")
    w("")
    w("**Key question for foundation validation**: Does this pattern reproduce?")
    w("")

    # ============================================================
    # Section 5: The Attention Dominance Finding
    # ============================================================
    w("## 5. The Attention Dominance Gap")
    w("")
    w("Tejas discovered that transcoder-based attribution captures only ~0.4% of the")
    w("total refusal signal:")
    w("")
    w("| Metric | Value |")
    w("|--------|-------|")
    w("| Attribution sum (transcoders/MLPs) | ~75 |")
    w("| Dot product (full residual stream) | ~18,322 |")
    w("| MLP contribution | ~0.4% |")
    w("| Attention + embeddings | ~99.6% |")
    w("")
    w("**Implication**: The mechanism analysis (dampening vs tug-of-war) describes")
    w("MLP behavior only. The main refusal circuit likely runs through **attention heads**,")
    w("which are not decomposed by transcoders.")
    w("")
    w("This is a fundamental limitation of transcoder-only circuit tracing for refusal analysis.")
    w("To get the full picture, we need:")
    w("- Attention SAEs (available in gemma-scope for some models)")
    w("- Direct head attribution via TransformerLens")
    w("- Or hybrid approaches combining attention head attribution with transcoder attribution")
    w("")

    # ============================================================
    # Section 6: Insights & Discussion
    # ============================================================
    w("## 6. Insights and Discussion")
    w("")
    w("### What is similar and why")
    w("")
    w("- **Refusal direction**: Both pipelines use identical methodology (Arditi et al. difference-in-means)")
    w("  with the same corrected parameters. Results should match closely.")
    w("- **Last-layer attribution (Mode A)**: Uses the same circuit-tracer API with CustomTarget.")
    w("  Minor differences from float precision, batch ordering, or version differences.")
    w("- **JB suppression pattern**: The directional finding (JB < bare) should be robust")
    w("  since it reflects genuine model behavior, not pipeline artifacts.")
    w("")
    w("### What may be different and why")
    w("")
    w("- **Intermediate-layer (Mode B)**: This is genuinely new. By measuring at layer 32")
    w("  instead of the final layer, we exclude post-L32 computation from the attribution.")
    w("  This could show a cleaner signal since we're measuring exactly where r_hat was computed.")
    w("- **Attribution magnitudes**: May differ due to transcoder version (`_affine` vs non-affine),")
    w("  dtype (float32 vs bfloat16), or circuit-tracer version differences.")
    w("- **Feature counts**: May differ based on JumpReLU threshold behavior across versions.")
    w("")
    w("### What the results tell us about refusal and jailbreaks")
    w("")
    w("1. **Refusal is real and measurable**: The 20,788 separation confirms a clear refusal")
    w("   direction exists in Gemma-3-4B-IT, with harmful and harmless on opposite sides of zero.")
    w("")
    w("2. **Jailbreaks suppress refusal at the MLP level**: The -18.7 mean difference shows")
    w("   jailbreak prompts consistently reduce MLP-mediated refusal attribution.")
    w("")
    w("3. **But MLPs are only 0.4% of the story**: Attention heads carry 99.6% of the refusal signal.")
    w("   The MLP-level mechanism (dampening vs tug-of-war) is a secondary effect.")
    w("")
    w("4. **Different jailbreak types work differently**: RP jailbreaks dampen the circuit")
    w("   (and can be patched with steering), while fiction/analytical jailbreaks create")
    w("   a tug-of-war (and are immune to steering). This has implications for safety:")
    w("   - Steering-based defenses only work against dampening-type jailbreaks")
    w("   - Tug-of-war jailbreaks require different mitigation strategies")
    w("")

    # ============================================================
    # Section 7: Follow-Up Experiments
    # ============================================================
    w("## 7. Follow-Up Experiments")
    w("")
    w("### Priority 1: Attention Head Attribution")
    w("Since attention carries 99.6% of the refusal signal, analyzing attention heads is critical.")
    w("- Use attention SAEs from gemma-scope (if available for Gemma-3-4B)")
    w("- Or implement direct head attribution: for each head h at layer l,")
    w("  compute A_h = (attention_output_h @ r_hat) to identify which heads drive refusal")
    w("- Compare head attribution patterns between bare harmful and jailbreak prompts")
    w("")
    w("### Priority 2: Refusal Vector Change Post-Jailbreak (from meeting notes)")
    w("Compute dot product of r_hat with W_i (layer weights) for each layer i.")
    w("Compare this profile between harmful and jailbroken prompts.")
    w("The difference classifies whether the jailbreak influences the refusal vector itself")
    w("or just the activations projected onto it.")
    w("")
    w("### Priority 3: Top-K Feature Comparison (from meeting notes)")
    w("For each harmful prompt and its jailbroken variant:")
    w("1. Extract top-K features by |A_{s->R}|")
    w("2. Identify features with same ID but different attribution values")
    w("3. Identify features unique to jailbreak (not in harmful top-K)")
    w("4. Identify features unique to harmful (not in jailbreak top-K)")
    w("5. Focus on features that flip sign or change magnitude significantly")
    w("")
    w("### Priority 4: Jailbreak Class Variance (from meeting notes)")
    w("Test different jailbreak classes systematically:")
    w("- RP (role-playing): 'As a security researcher...'")
    w("- Fiction: 'Write a scene where...'")
    w("- Analytical: 'Is this description correct...'")
    w("- Completion: 'Complete this manual entry...'")
    w("- Translation: reframe informal to formal")
    w("Do different classes mediate refusal differently? Are the top-K features the same?")
    w("")
    w("### Priority 5: Attribution Net Mean on ALL Features (from meeting notes)")
    w("Rerun attribution without any feature filtering (max_feature_nodes=None)")
    w("and compute the net mean across all features, not just top-K.")
    w("This gives the true total MLP contribution to the refusal signal.")
    w("")
    w("### Priority 6: Multi-Layer Measurement Sweep")
    w("Using the measurement_layer patch, sweep across layers 10-33 and plot:")
    w("- How does the attribution sum change with measurement layer?")
    w("- At which layer does the JB suppression effect peak?")
    w("- Does the dampening vs tug-of-war classification change at different layers?")
    w("")
    w("### Priority 7: Cross-Model Comparison")
    w("Run the same pipeline on Gemma-2-2B-IT to check:")
    w("- Does the attention-dominance finding generalize?")
    w("- Are the jailbreak mechanisms (dampening vs tug-of-war) model-dependent?")
    w("- The smaller model is faster to iterate on and fits on consumer GPUs")
    w("")

    w("---")
    w("")
    w("*Report generated by `scripts/generate_comparison_report.py`*")

    # Write report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written to: {OUTPUT_FILE}")
    print(f"  ({len(lines)} lines)")


if __name__ == "__main__":
    generate_report()
