"""                                                                                                                                                       
Stage 02b: Statistical Analysis + Plots                                                                                                                   
========================================                                                                                                                  
Computes paired statistics and generates publication-quality plots
from Stage 02 attribution results.                                                                                                                        
                                                                                                                                                        
Statistics: Wilcoxon signed-rank, paired t-test, Cohen's d, bootstrap 95% CIs                                                                             
Dual mechanism: dPos (pro-refusal change), dNeg (anti-refusal change)                                                                                     
Plots: class comparison, per-prompt deltas, effect sizes, feature comparison                                                                              
                                                                                                                                                        
Inputs:  02_attribution/attribution_results.json, feature_comparison_aggregate.json                                                                       
Outputs: 02b_stats/                                                                                                                                       
"""                                                                                                                                                       
from __future__ import annotations                                                                                                                        
                                                                                                                                                        
import argparse                                                                                                                                           
import sys                                                                                                                                                
from pathlib import Path                                                                                                                                  

import numpy as np                                                                                                                                        
                                        
sys.path.insert(0, str(Path(__file__).resolve().parent))                                                                                                  
import config                                                                                                                                             
from utils import save_json, load_json, get_stage_dir                                                                                                     
                                                                                                                                                        
                                                                                                                                                        
def parse_args():                      
    parser = argparse.ArgumentParser(description="Statistical analysis + plots")
    parser.add_argument("--run-dir", type=Path, required=True)                                                                                            
    parser.add_argument(                                                                                                                                  
        "--n-bootstrap", type=int, default=10000,                                                                                                         
        help="Number of bootstrap resamples for CIs",                                                                                                     
    )                                                                                                                                                     
    return parser.parse_args()
                                                                                                                                                        
                                        
def main():
    from scipy import stats as scipy_stats

    args = parse_args()                                                                                                                                   
    run_dir = args.run_dir
    out_dir = get_stage_dir(run_dir, "02b_stats")                                                                                                         
                                        
    print("=" * 60)                                                                                                                                       
    print("STAGE 02b: Statistical Analysis + Plots")
    print("=" * 60)                                                                                                                                       
                                        
    # Load attribution results
    attr_path = run_dir / "02_attribution" / "attribution_results.json"
    if not attr_path.exists():
        # Fallback to existing scaled experiment results                                                                                                  
        fallback = list(
            (config.REPO_ROOT / "data" / "results" / "scaled_experiments").glob(                                                                          
                "run_*/attribution_results.json"                                                                                                          
            )                                                                                                                                             
        )                                                                                                                                                 
        if not fallback:               
            print("  ERROR: No attribution results found.")
            sys.exit(1)                                                                                                                                   
        attr_path = sorted(fallback)[-1]
        print(f"  Using fallback: {attr_path}")                                                                                                           
                                                                                                                                                        
    raw = load_json(attr_path)
    results = raw if isinstance(raw, list) else raw["results"]                                                                                            
    print(f"  Loaded {len(results)} prompt results")                                                                                                      

    # Load feature comparison aggregate if available                                                                                                      
    feat_agg_path = attr_path.parent / "feature_comparison_aggregate.json"
    feat_agg = load_json(feat_agg_path) if feat_agg_path.exists() else {}                                                                                 
                                                                                                                                                        
    classes = list(config.JB_CLASSES.keys())                                                                                                              
                                                                                                                                                        
    # ============================================================                                                                                        
    # Paired statistical tests
    # ============================================================                                                                                        
    print("\n  Computing paired statistics...")
    paired_data = {}                                                                                                                                      

    for cls in classes:                                                                                                                                   
        bare_nets = []                 
        cls_nets = []
        bare_pos = []                                                                                                                                     
        cls_pos = []
        bare_neg = []                                                                                                                                     
        cls_neg = []                   

        for row in results:
            conds = row.get("conditions", row)
            if "bare" in conds and cls in conds:                                                                                                          
                b = conds["bare"]
                c = conds[cls]                                                                                                                            
                if "error" not in b and "error" not in c:
                    bare_nets.append(b["net"])                                                                                                            
                    cls_nets.append(c["net"])
                    bare_pos.append(b["pos_sum"])                                                                                                         
                    cls_pos.append(c["pos_sum"])
                    bare_neg.append(b["neg_sum"])                                                                                                         
                    cls_neg.append(c["neg_sum"])
                                                                                                                                                        
        n = len(bare_nets)             
        if n < 3:
            print(f"    {cls}: insufficient data ({n} pairs)")                                                                                            
            continue
                                                                                                                                                        
        bare_nets = np.array(bare_nets)
        cls_nets = np.array(cls_nets)
        deltas = cls_nets - bare_nets                                                                                                                     

        # Wilcoxon signed-rank (non-parametric)                                                                                                           
        try:                           
            w_stat, w_pval = scipy_stats.wilcoxon(bare_nets, cls_nets)
        except ValueError:                                                                                                                                
            w_stat, w_pval = float("nan"), float("nan")
                                                                                                                                                        
        # Paired t-test                
        t_stat, t_pval = scipy_stats.ttest_rel(bare_nets, cls_nets)                                                                                       
                                        
        # Cohen's d (paired)
        d_mean = np.mean(deltas)
        d_std = np.std(deltas, ddof=1)                                                                                                                    
        cohens_d = d_mean / d_std if d_std > 0 else 0.0
                                                                                                                                                        
        # Bootstrap 95% CI             
        rng = np.random.RandomState(42)                                                                                                                   
        boot_means = np.array(         
            [rng.choice(deltas, size=n, replace=True).mean()                                                                                              
            for _ in range(args.n_bootstrap)]
        )                                                                                                                                                 
        ci_low = float(np.percentile(boot_means, 2.5))                                                                                                    
        ci_high = float(np.percentile(boot_means, 97.5))
                                                                                                                                                        
        # Dual mechanism decomposition                                                                                                                    
        d_pos = float(np.mean(cls_pos) - np.mean(bare_pos))  # change in pro-refusal
        d_neg = float(np.mean(cls_neg) - np.mean(bare_neg))  # change in anti-refusal                                                                     
                                                                                                                                                        
        paired_data[cls] = {
            "n_pairs": int(n),                                                                                                                            
            "bare_mean_net": float(bare_nets.mean()),
            "bare_std_net": float(bare_nets.std()),                                                                                                       
            "cls_mean_net": float(cls_nets.mean()),
            "cls_std_net": float(cls_nets.std()),                                                                                                         
            "mean_delta": float(d_mean),
            "std_delta": float(d_std),                                                                                                                    
            "pct_change": float(d_mean / bare_nets.mean() * 100)
            if bare_nets.mean() != 0 else 0.0,                                                                                                            
            "wilcoxon_stat": float(w_stat),                                                                                                               
            "wilcoxon_pval": float(w_pval),                                                                                                               
            "ttest_stat": float(t_stat),                                                                                                                  
            "ttest_pval": float(t_pval),                                                                                                                  
            "cohens_d": float(cohens_d),
            "ci_95_low": ci_low,                                                                                                                          
            "ci_95_high": ci_high,     
            "n_cls_lower": int((cls_nets < bare_nets).sum()),                                                                                             
            "bare_mean_pos": float(np.mean(bare_pos)),                                                                                                    
            "cls_mean_pos": float(np.mean(cls_pos)),                                                                                                      
            "bare_mean_neg": float(np.mean(bare_neg)),                                                                                                    
            "cls_mean_neg": float(np.mean(cls_neg)),
            "d_pos": d_pos,                                                                                                                               
            "d_neg": d_neg,            
            "d_pos_pct": float(d_pos / np.mean(bare_pos) * 100)                                                                                           
            if np.mean(bare_pos) != 0 else 0.0,
            "d_neg_pct": float(d_neg / abs(np.mean(bare_neg)) * 100)                                                                                      
            if np.mean(bare_neg) != 0 else 0.0,
        }                                                                                                                                                 
                                        
    # Print results table                                                                                                                                 
    print(f"\n  {'Class':>15} | {'N':>3} | {'Bare':>8} | {'JB':>8} | {'Delta':>8} | "
        f"{'%Chg':>7} | {'p(W)':>10} | {'d':>7} | {'95% CI':>18}")                                                                                      
    print(f"  {'-'*15}-+-{'-'*3}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-"                                                                                        
        f"{'-'*7}-+-{'-'*10}-+-{'-'*7}-+-{'-'*18}")                                                                                                     
                                                                                                                                                        
    for cls in classes:                                                                                                                                   
        if cls not in paired_data:     
            continue                                                                                                                                      
        d = paired_data[cls]
        sig = "***" if d["wilcoxon_pval"] < 0.001 else "**" if d["wilcoxon_pval"] < 0.01 else "*" if d["wilcoxon_pval"] < 0.05 else "ns"                                                                                                  
        print(f"  {cls:>15} | {d['n_pairs']:3d} | {d['bare_mean_net']:+8.1f} | "
            f"{d['cls_mean_net']:+8.1f} | {d['mean_delta']:+8.1f} | "                                                                                   
            f"{d['pct_change']:+6.1f}% | {d['wilcoxon_pval']:10.6f}{sig:>3} | "                                                                         
            f"{d['cohens_d']:+7.2f} | [{d['ci_95_low']:+.1f}, {d['ci_95_high']:+.1f}]")                                                                 
                                                                                                                                                        
    # Dual mechanism table                                                                                                                                
    print(f"\n  Dual Mechanism Decomposition:")                                                                                                           
    print(f"  {'Class':>15} | {'dPos':>10} | {'dNeg':>10} | {'Net':>8} | Dominant")                                                                       
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*25}")                                                                                       
    for cls in classes:                                                                                                                                   
        if cls not in paired_data:                                                                                                                        
            continue                                                                                                                                      
        d = paired_data[cls]           
        if abs(d["d_pos"]) > 2 * abs(d["d_neg"]):
            dominant = "Dampening-dominant"                                                                                                               
        elif abs(d["d_neg"]) > 2 * abs(d["d_pos"]):                                                                                                       
            dominant = "Amplification-dominant"                                                                                                           
        elif d["d_pos"] > 0:                                                                                                                              
            dominant = "Pro-refusal recruitment"                                                                                                          
        else:
            dominant = "Balanced"                                                                                                                         
        print(f"  {cls:>15} | {d['d_pos']:+10.1f} | {d['d_neg']:+10.1f} | "
            f"{d['mean_delta']:+8.1f} | {dominant}")                                                                                                    

    save_json(paired_data, out_dir / "statistical_analysis.json")                                                                                         
    print(f"\n  Saved statistical_analysis.json")
                                                                                                                                                        
    # ============================================================
    # Plots
    # ============================================================                                                                                        
    print("\n  Generating plots...")
    import matplotlib                                                                                                                                     
    matplotlib.use("Agg")              
    import matplotlib.pyplot as plt

    # --- Plot 1: Class comparison bar chart ---                                                                                                          
    fig, ax = plt.subplots(figsize=(12, 7))
    all_classes = ["bare"] + classes                                                                                                                      
    means, stds, pos_means, neg_means = [], [], [], []                                                                                                    

    for cls in all_classes:                                                                                                                               
        nets, poss, negs = [], [], []  
        for row in results:                                                                                                                               
            conds = row.get("conditions", row)
            if cls in conds and "error" not in conds[cls]:
                c = conds[cls]                                                                                                                            
                nets.append(c["net"])
                poss.append(c["pos_sum"])                                                                                                                 
                negs.append(c["neg_sum"])
        means.append(np.mean(nets) if nets else 0)                                                                                                        
        stds.append(np.std(nets) if nets else 0)
        pos_means.append(np.mean(poss) if poss else 0)                                                                                                    
        neg_means.append(np.mean(negs) if negs else 0)
                                                                                                                                                        
    x = np.arange(len(all_classes))    
    w = 0.25                                                                                                                                              
    ax.bar(x - w, pos_means, w, label="Positive (pro-refusal)", color="#d9534f", alpha=0.8)
    ax.bar(x, means, w, label="Net", color="#5cb85c", alpha=0.8, yerr=stds, capsize=4)                                                                    
    ax.bar(x + w, neg_means, w, label="Negative (anti-refusal)", color="#5bc0de", alpha=0.8)                                                              
                                                                                                                                                        
    for i, cls in enumerate(all_classes):                                                                                                                 
        if cls in paired_data and paired_data[cls]["wilcoxon_pval"] < 0.05:                                                                               
            sig = "***" if paired_data[cls]["wilcoxon_pval"] < 0.001 else "**" if paired_data[cls]["wilcoxon_pval"] < 0.01 else "*"                                                                               
            ax.text(i, means[i] + stds[i] + 5, sig,
                    ha="center", fontsize=12, fontweight="bold")                                                                                          
                                                                                                                                                        
    ax.set_xticks(x)                                                                                                                                      
    ax.set_xticklabels([c.upper() for c in all_classes], fontsize=10)                                                                                     
    ax.set_ylabel("Attribution Sum")                                                                                                                      
    ax.set_title(f"Jailbreak Class Comparison: Attribution to Refusal Direction (n={len(results)})")
    ax.legend()                                                                                                                                           
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()                                                                                                                                    
    plt.savefig(out_dir / "class_comparison.png", dpi=150)
    plt.close()                                                                                                                                           
    print("    Saved class_comparison.png")
                                                                                                                                                        
    # --- Plot 2: Per-prompt delta --- 
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))                                                                                                      
    axes = axes.flatten()                                                                                                                                 

    for idx, cls in enumerate(classes):                                                                                                                   
        ax = axes[idx]                 
        deltas = []
        for row in results:
            conds = row.get("conditions", row)
            if "bare" in conds and cls in conds:
                b, c = conds["bare"], conds[cls]                                                                                                          
                if "error" not in b and "error" not in c:
                    deltas.append(c["net"] - b["net"])                                                                                                    
        if deltas:                                                                                                                                        
            colors = ["#d9534f" if d < 0 else "#5bc0de" for d in deltas]
            ax.bar(range(len(deltas)), deltas, color=colors, alpha=0.7)                                                                                   
            ax.axhline(0, color="black", linewidth=0.5)                                                                                                   
            ax.axhline(np.mean(deltas), color="green", linewidth=2,
                        linestyle="--", label=f"mean={np.mean(deltas):+.1f}")                                                                              
            ax.set_title(f"{cls.upper()} (n={len(deltas)})")                                                                                              
            ax.set_xlabel("Prompt index")                                                                                                                 
            ax.set_ylabel("Net delta (JB - bare)")                                                                                                        
            ax.legend(fontsize=8)                                                                                                                         
                                                                                                                                                        
    for j in range(len(classes), 6):                                                                                                                      
        axes[j].set_visible(False)     
                                                                                                                                                        
    fig.suptitle("Per-Prompt Attribution Delta by Jailbreak Class", fontsize=14)                                                                          
    plt.tight_layout()
    plt.savefig(out_dir / "per_prompt_deltas.png", dpi=150)                                                                                               
    plt.close()                        
    print("    Saved per_prompt_deltas.png")                                                                                                              

    # --- Plot 3: Effect sizes ---                                                                                                                        
    fig, ax = plt.subplots(figsize=(10, 6))
    cls_names = [c for c in classes if c in paired_data]
    ds = [paired_data[c]["cohens_d"] for c in cls_names]                                                                                                  
    colors = ["#d9534f" if d < 0 else "#5bc0de" for d in ds]
                                                                                                                                                        
    ax.barh(range(len(cls_names)), ds, color=colors, alpha=0.8)
    ax.set_yticks(range(len(cls_names)))                                                                                                                  
    ax.set_yticklabels([c.upper() for c in cls_names])                                                                                                    
    ax.set_xlabel("Cohen's d (negative = suppresses refusal)")
    ax.set_title("Effect Size by Jailbreak Class")                                                                                                        
    ax.axvline(0, color="black", linewidth=0.5)
    for threshold, style, label in [                                                                                                                      
        (-0.2, ":", "small"), (-0.5, "--", "medium"), (-0.8, "-", "large")                                                                                
    ]:                                                                                                                                                    
        ax.axvline(threshold, color="gray", linewidth=0.5, linestyle=style)                                                                               
        ax.axvline(-threshold, color="gray", linewidth=0.5, linestyle=style)                                                                              
    plt.tight_layout()                                                                                                                                    
    plt.savefig(out_dir / "effect_sizes.png", dpi=150)                                                                                                    
    plt.close()                                                                                                                                           
    print("    Saved effect_sizes.png")
                                                                                                                                                        
    # --- Plot 4: Feature comparison summary ---
    if feat_agg:                                                                                                                                          
        fig, ax = plt.subplots(figsize=(12, 6))
        cls_names = [c for c in classes if c in feat_agg]
        metrics = ["n_shared", "n_bare_only", "n_cls_only", "n_sign_flipped",                                                                             
                    "n_dampened", "n_amplified_anti"]                                                                                                     
        metric_labels = ["Shared", "Bare-only", "JB-only", "Sign-flipped",                                                                                
                        "Dampened", "Amplified anti"]                                                                                                    
        colors_feat = ["#7f7f7f", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
                                                                                                                                                        
        x = np.arange(len(cls_names))  
        width = 0.13                                                                                                                                      
        for j, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors_feat)):
            vals = [feat_agg[c][metric]["mean"] for c in cls_names]                                                                                       
            ax.bar(x + j * width, vals, width, label=label, color=color, alpha=0.8)                                                                       
                                                                                                                                                        
        ax.set_xticks(x + width * 2.5)                                                                                                                    
        ax.set_xticklabels([c.upper() for c in cls_names])
        ax.set_ylabel("Mean Feature Count")                                                                                                               
        ax.set_title("Feature Comparison by Jailbreak Class")
        ax.legend(fontsize=8)                                                                                                                             
        plt.tight_layout()
        plt.savefig(out_dir / "feature_comparison_summary.png", dpi=150)                                                                                  
        plt.close()                                                                                                                                       
        print("    Saved feature_comparison_summary.png")
                                                                                                                                                        
    # ============================================================
    # Summary report
    # ============================================================
    print("\n  Generating summary report...")
                                                                                                                                                        
    report_lines = [
        f"# Attribution Experiment: Statistical Analysis",                                                                                                
        f"",                                                                                                                                              
        f"**Prompts**: {len(results)} | **Model**: {config.MODEL_NAME}",
        f"**Direction**: Layer {config.BEST_SEPARATION_LAYER} (best separation)",                                                                         
        f"",                           
        f"## Main Results",                                                                                                                               
        f"",                           
        f"| Class | Net Delta | % Change | p (Wilcoxon) | Cohen's d | 95% CI | Consistency |",                                                            
        f"|-------|----------|----------|-------------|-----------|--------|-------------|",                                                              
    ]                                                                                                                                                     
    for cls in classes:                                                                                                                                   
        if cls not in paired_data:                                                                                                                        
            continue                   
        d = paired_data[cls]
        sig = "***" if d["wilcoxon_pval"] < 0.001 else "**" if d["wilcoxon_pval"] < 0.01 else "*" if d["wilcoxon_pval"] < 0.05 else "ns"
        consistency = d["n_cls_lower"]                                                                                                                    
        report_lines.append(           
            f"| **{cls.capitalize()}** | {d['mean_delta']:+.1f} | "                                                                                       
            f"{d['pct_change']:+.1f}% | {d['wilcoxon_pval']:.4g}{sig} | "                                                                                 
            f"{d['cohens_d']:+.2f} | [{d['ci_95_low']:+.1f}, {d['ci_95_high']:+.1f}] | "                                                                  
            f"{consistency}/{d['n_pairs']} |"                                                                                                             
        )                                                                                                                                                 
                                                                                                                                                        
    report_lines.extend([              
        f"",
        f"## Dual Mechanism Decomposition",
        f"",                                                                                                                                              
        f"| Class | dPos (pro-refusal) | dNeg (anti-refusal) | Net | Dominant |",
        f"|-------|--------------------|--------------------|----|----------|",                                                                           
    ])                                                                                                                                                    
    for cls in classes:
        if cls not in paired_data:                                                                                                                        
            continue                   
        d = paired_data[cls]
        if abs(d["d_pos"]) > 2 * abs(d["d_neg"]):
            dom = "Dampening-dominant"                                                                                                                    
        elif abs(d["d_neg"]) > 2 * abs(d["d_pos"]):
            dom = "Amplification-dominant"                                                                                                                
        elif d["d_pos"] > 0:           
            dom = "Pro-refusal recruitment"                                                                                                               
        else:
            dom = "Balanced"                                                                                                                              
        report_lines.append(           
            f"| **{cls.capitalize()}** | {d['d_pos']:+.1f} ({d['d_pos_pct']:+.1f}%) | "
            f"{d['d_neg']:+.1f} ({d['d_neg_pct']:+.1f}%) | {d['mean_delta']:+.1f} | {dom} |"                                                              
        )                                                                                                                                                 
                                                                                                                                                        
    if feat_agg:                                                                                                                                          
        report_lines.extend([          
            f"",
            f"## Feature Comparison",
            f"",                                                                                                                                          
            f"| Class | Bare | JB | Shared % | JB-only % | Sign-flip % |",
            f"|-------|------|-----|----------|-----------|------------|",                                                                                
        ])                             
        for cls in classes:                                                                                                                               
            if cls not in feat_agg:    
                continue
            a = feat_agg[cls]
            n_bare = a["n_bare"]["mean"]                                                                                                                  
            n_cls = a["n_cls"]["mean"]
            shared_pct = a["n_shared"]["mean"] / n_bare * 100 if n_bare else 0                                                                            
            cls_only_pct = a["n_cls_only"]["mean"] / n_cls * 100 if n_cls else 0                                                                          
            flip_pct = a["n_sign_flipped"]["mean"] / a["n_shared"]["mean"] * 100 if a["n_shared"]["mean"] else 0                                                                                                           
            report_lines.append(                                                                                                                          
                f"| **{cls.capitalize()}** | {n_bare:.0f} | {n_cls:.0f} | "                                                                               
                f"{shared_pct:.1f}% | {cls_only_pct:.1f}% | {flip_pct:.1f}% |"                                                                            
            )
                                                                                                                                                        
    report_text = "\n".join(report_lines) + "\n"                                                                                                          
    with open(out_dir / "EXPERIMENT_SUMMARY.md", "w") as f:
        f.write(report_text)                                                                                                                              
    print("    Saved EXPERIMENT_SUMMARY.md")
                                                                                                                                                        
    print(f"\n  All outputs saved to {out_dir}/")
    print("DONE!")                                                                                                                                        
                                                                                                                                                        

if __name__ == "__main__":                                                                                                                                
    main()