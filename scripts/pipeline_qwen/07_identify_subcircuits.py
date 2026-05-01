"""                                                
Stage 07: Identify Functional Subcircuits (rule-based)
=======================================================
Pure set-logic subcircuit identification from Stage 04 outputs.
No GPU, no ML fitting — all derivations from conditions_seen and                                                                
feature_class_sets buckets.                  
                                                                                                                                
Reads:                                       
- 04_labels/feature_labels.json        (per-feature conditions_seen, layer, attribution)                                        
- 04_labels/feature_class_sets.json    (bucket → feature → classes map)                                                         
                                                                                                                                
Writes to 07_subcircuits/:                                                                                                      
- subcircuits.json          — canonical definitions + feature lists + per-subcircuit stats
- subcircuits_summary.json  — sizes + pairwise overlap matrix                                                                   
- subcircuits_treemap.png   — visual size breakdown
- subcircuits_by_layer.png  — stacked-bar-by-layer (one row per subcircuit)
- subcircuits_overlap.png   — pairwise overlap heatmap                                                                          
- SUBCIRCUITS_REPORT.md     — human-readable subcircuit report
                                                                                                                                
Subcircuits:                                       
universal_refusal_core        bare + all 5 JB                                                                                 
canonical_pro_refusal         all 5 JB, no bare  
sign_flip_convergent          sign_flipped in ≥3 JB classes                                                                   
dampening_specialists         dampened in ≥3 JB classes                                                                       
anti_refusal_amplifiers       amplified_anti in ≥3 JB classes                                                                 
late_wave_layer24_32          any feature in L24–L32 (cross-cuts above)                                                       
{class}_exclusive (×5)        exactly one JB class, no bare                                                                   
"""                                                                                                                             
from __future__ import annotations                                                                                              
                                                                                                                                
import argparse                                                                                                                 
import sys                                                                                                                      
from pathlib import Path                   
                                                                                                                                
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401  (imported for side-effect / consistency)                                                           
from utils import get_stage_dir, load_json, save_json                                                                           
                                    
                                                                                                                                
JB_CLASSES = frozenset({"analytical", "cognitive_reframe", "completion", "fiction", "roleplay"})                                
ALL_CONDS = JB_CLASSES | {"bare"}                                                                                               
                                                                                                                                
CONVERGENT_MIN_CLASSES = 3                         
LATE_WAVE_LO, LATE_WAVE_HI = 24, 32                                                                                             
                                            
                                            
def parse_args():                            
    p = argparse.ArgumentParser(description="Rule-based subcircuit identification")                                             
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--convergent-min", type=int, default=CONVERGENT_MIN_CLASSES)                                                
    p.add_argument("--late-wave-start", type=int, default=LATE_WAVE_LO)                                                         
    p.add_argument("--late-wave-end", type=int, default=LATE_WAVE_HI)                                                           
    return p.parse_args()                                                                                                       
                                                                                                                                
                                            
# ------------------------------- rules ----------------------------------                                                      
                                            
def build_universal_core(feature_labels: dict) -> list[str]:                                                                    
    return [k for k, v in feature_labels.items()
            if set(v.get("conditions_seen", [])) >= ALL_CONDS]                                                                  
                                            
                                                                                                                                
def build_canonical_pro_refusal(feature_labels: dict) -> list[str]:
    return [                                                                                                                    
        k for k, v in feature_labels.items() 
        if set(v.get("conditions_seen", [])) >= JB_CLASSES                                                                      
        and "bare" not in v.get("conditions_seen", [])
    ]                                                                                                                           
                                            
                                                                                                                                
def build_class_exclusive(feature_labels: dict) -> dict[str, list[str]]:                                                        
    exclusive: dict[str, list[str]] = {c: [] for c in JB_CLASSES}
    for k, v in feature_labels.items():                                                                                         
        conds = set(v.get("conditions_seen", []))                                                                               
        if "bare" in conds:                        
            continue                                                                                                            
        jb = conds & JB_CLASSES              
        if len(jb) == 1:                                                                                                        
            exclusive[next(iter(jb))].append(k)
    return exclusive                                                                                                            
                                                                                                                                
                                            
def build_convergent_bucket(class_sets: dict, bucket: str, min_classes: int) -> list[str]:                                      
    features = class_sets.get("by_bucket", {}).get(bucket, {}).get("features", {})
    return [k for k, classes in features.items() if len(classes) >= min_classes]                                                
                                                    
                                                    
def build_late_wave(feature_labels: dict, lo: int, hi: int) -> list[str]:                                                       
    return [k for k, v in feature_labels.items() if lo <= v.get("layer", -1) <= hi]
                                                                                                                                
                                                                                                                                
# -------------------------- summarization -------------------------------
                                                                                                                                
def summarize_subcircuit(name: str, feature_keys: list[str],
                        feature_labels: dict, n_layers: int = 36) -> dict:                                                     
    by_layer = [0] * n_layers                                                                                                   
    freqs = []                                                                                                                  
    for k in feature_keys:                                                                                                      
        v = feature_labels.get(k, {})              
        layer = v.get("layer")                                                                                                  
        if layer is not None and 0 <= layer < n_layers:
            by_layer[layer] += 1                                                                                                
        f = v.get("activation_frequency")                                                                                       
        if f is not None:                    
            freqs.append(f)                                                                                                     
                                            
    top = sorted(                                                                                                               
        feature_keys,                        
        key=lambda k: feature_labels.get(k, {}).get("max_abs_attribution", 0.0),                                                
        reverse=True,                        
    )[:5]                                                                                                                       
    top_features = []                        
    for k in top:                                                                                                               
        v = feature_labels.get(k, {})        
        top_features.append({                                                                                                   
            "key": k,               
            "layer": v.get("layer"),                                                                                            
            "max_abs_attribution": round(v.get("max_abs_attribution", 0.0), 4),
            "top_logits": (v.get("top_logits") or [])[:3],                                                                      
        })                                   
                                            
    n_occupied = sum(1 for c in by_layer if c > 0) 
    peak_layer = max(range(n_layers), key=lambda i: by_layer[i]) if any(by_layer) else None                                     
    return {                                 
        "name": name,                                                                                                           
        "size": len(feature_keys),                                                                                              
        "features": feature_keys,                  
        "by_layer": by_layer,                                                                                                   
        "peak_layer": peak_layer,                  
        "peak_count": by_layer[peak_layer] if peak_layer is not None else 0,                                                    
        "n_layers_occupied": n_occupied,   
        "mean_activation_frequency": round(sum(freqs) / len(freqs), 6) if freqs else None,                                      
        "top_features": top_features,                                                                                           
    }                                        
                                                                                                                                
                                                    
def build_overlap_matrix(subcircuits: dict[str, list[str]]) -> dict:                                                            
    names = list(subcircuits.keys())     
    sets = {n: set(subcircuits[n]) for n in names}                                                                              
    n = len(names)                                                                                                              
    inter = [[0] * n for _ in range(n)]      
    norm = [[0.0] * n for _ in range(n)]                                                                                        
    for i, a in enumerate(names):            
        for j, b in enumerate(names):                                                                                           
            ab = sets[a] & sets[b]                                                                                              
            inter[i][j] = len(ab)                                                                                               
            m = min(len(sets[a]), len(sets[b]))                                                                                 
            norm[i][j] = len(ab) / m if m else 0.0                                                                              
    return {"names": names, "intersection": inter, "normalized": norm}
                                                                                                                                
                                        
# ------------------------------ plots -----------------------------------                                                      
                                                    
def plot_treemap(summary: dict, out_path: Path) -> None:                                                                        
    import matplotlib                                                                                                           
    matplotlib.use("Agg")                    
    import matplotlib.pyplot as plt                                                                                             
                                            
    items = sorted(summary.items(), key=lambda kv: kv[1]["size"], reverse=True)                                                 
    items = [(k, v) for k, v in items if v["size"] > 0]
    try:                                                                                                                        
        import squarify                      
        sizes = [v["size"] for _, v in items]                                                                                   
        labels = [f"{k}\nn={v['size']}" for k, v in items]
        colors = plt.cm.tab20([i / max(len(items), 1) for i in range(len(items))])                                              
        fig, ax = plt.subplots(figsize=(14, 8))    
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.85, ax=ax,                                               
                    text_kwargs={"fontsize": 9})
        ax.axis("off")                                                                                                          
        ax.set_title(f"Rule-Based Subcircuits  (total features touched = {sum(sizes)})")
    except ImportError:                                                                                                         
        names = [k for k, _ in items]              
        sizes = [v["size"] for _, v in items]                                                                                   
        fig, ax = plt.subplots(figsize=(10, 6))                                                                                 
        ax.barh(names, sizes, color="#4a90e2", edgecolor="black")
        for i, s in enumerate(sizes):                                                                                           
            ax.text(s, i, f" {s}", va="center")                                                                                 
        ax.set_xlabel("Number of features")        
        ax.set_title("Subcircuit sizes (rule-based)  [squarify unavailable → bar chart]")                                       
        ax.invert_yaxis()                                                                                                       
    plt.tight_layout()                                                                                                          
    plt.savefig(out_path, dpi=150)                                                                                              
    plt.close()                                                                                                                 
                                                                                                                                
                                                                                                                                
def plot_by_layer(summary: dict, out_path: Path, n_layers: int = 36) -> None:                                                   
    import matplotlib                              
    matplotlib.use("Agg")                                                                                                       
    import matplotlib.pyplot as plt                                                                                             
    import numpy as np                             
                                                                                                                                
    items = [(k, v) for k, v in summary.items() if v["size"] > 0]                                                               
    n = len(items)                                                                                                              
    fig, axes = plt.subplots(n, 1, figsize=(14, max(9, 1.3 * n)), sharex=True)                                                  
    if n == 1:                                                                                                                  
        axes = [axes]                        
    x = np.arange(n_layers)                                                                                                     
    for ax, (name, v) in zip(axes, items):                                                                                      
        ax.bar(x, v["by_layer"], color="#4a90e2", edgecolor="black", linewidth=0.3)
        ax.axvspan(24, 32, alpha=0.08, color="red", zorder=0)                                                                   
        ax.set_ylabel("n", fontsize=9)     
        peak = v["peak_layer"]                     
        peak_str = f"peak L{peak}={v['peak_count']}" if peak is not None else "empty"                                           
        ax.set_title(f"{name}  (n={v['size']}, {peak_str})", fontsize=10, loc="left")
        ax.grid(axis="y", alpha=0.3)                                                                                            
    axes[-1].set_xlabel("Layer index")             
    axes[-1].set_xticks(np.arange(0, n_layers, 2))                                                                              
    fig.suptitle("Subcircuit Layer Distributions  (shaded: L24–L32 late-wave band)",                                            
                fontsize=12)                                                                                                   
    plt.tight_layout()                                                                                                          
    plt.savefig(out_path, dpi=150)                 
    plt.close()                                                                                                                 
                                                                                                                                
                                                                                                                                
def plot_overlap(overlap: dict, out_path: Path) -> None:                                                                        
    import matplotlib                                                                                                           
    matplotlib.use("Agg")                                                                                                       
    import matplotlib.pyplot as plt                                                                                             
    import numpy as np                                                                                                          
                                                                                                                                
    names = overlap["names"]                                                                                                    
    mat = np.array(overlap["normalized"])                                                                                       
    fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(names)), max(7, 0.7 * len(names))))                                        
    im = ax.imshow(mat, cmap="magma_r", vmin=0, vmax=1)                                                                         
    ax.set_xticks(range(len(names)))         
    ax.set_yticks(range(len(names)))                                                                                            
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)                      
    for i in range(len(names)):                    
        for j in range(len(names)):                                                                                             
            v = mat[i][j]                    
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",                                                                 
                    color="white" if v > 0.55 else "black", fontsize=8)
    ax.set_title("Subcircuit Pairwise Overlap  |A∩B| / min(|A|, |B|)")
    cbar = fig.colorbar(im, ax=ax)                 
    cbar.set_label("fraction of smaller set in intersection")                                                                   
    plt.tight_layout()                                                                                                          
    plt.savefig(out_path, dpi=150)                                                                                              
    plt.close()                                                                                                                 
                                                                                                                                
                                            
# ----------------------------- report -----------------------------------                                                      
                                                                                                                                
DEFINITIONS = {                            
    "universal_refusal_core": (                                                                                                 
        "Features seen in **bare + all 5 JB classes**. The canonical refusal core — "                                           
        "present in both harmful-alone and every jailbreak. Ablation control baseline."
    ),                                                                                                                          
    "canonical_pro_refusal": (               
        "Features seen in **all 5 JB classes but NOT bare**. Recruited specifically under "                                     
        "jailbreak — interpretable as a shared JB-suppression / pro-refusal response."
    ),                                                                                                                          
    "sign_flip_convergent": (                                                                                                   
        "Features in the **sign_flipped** bucket of ≥{min} JB classes. Robustly reverse "
        "attribution sign under JB — the highest-confidence mechanism-change features."                                         
    ),                                             
    "dampening_specialists": (                     
        "Features in the **dampened** bucket of ≥{min} JB classes. Pro-refusal features "
        "whose contribution to the refusal direction weakens across most JB types."                                             
    ),                                                                                                                          
    "anti_refusal_amplifiers": (                                                                                                
        "Features in the **amplified_anti** bucket of ≥{min} JB classes. Anti-refusal "                                         
        "features that grow in magnitude across most JB types — the bypass signal."                                             
    ),                                                                                                                          
    "late_wave_layer24_32": (                                                                                                   
        "All tagged features in layers **{lo}–{hi}** — the JB-impact band identified in A8. "                                   
        "Layer-based cross-cut; overlaps other subcircuits."                                                                    
    ),                                                                                                                          
}                                                                                                                               
CLASS_EXCLUSIVE_DEF = (                                                                                                         
    "Features seen in **only** the `{cls}` JB class (no bare, no other JB). "                                                   
    "Candidates for `{cls}`-specific jailbreak mechanism."
)                                                                                                                               
                                            
                                                                                                                                
def write_report(summary: dict, overlap: dict, out_path: Path,
                convergent_min: int, late_lo: int, late_hi: int) -> None:                                                      
    ordered = sorted(summary.keys(), key=lambda k: -summary[k]["size"])
    lines = [                                                                                                                   
        "# Subcircuits Report (Rule-Based)",                                                                                    
        "",                                                                                                                     
        "Each subcircuit is defined by a precise set-logic rule over the features observed "                                    
        "across bare + 5 JB classes. No ML fitting — fully interpretable.",                                                     
        "",                                                                                                                     
        "## Summary table",                                                                                                     
        "",                                                                                                                     
        "| Subcircuit | Size | Peak layer | n_layers occupied | Mean act. freq. |",                                             
        "|---|---|---|---|---|",                                                                                                
    ]                                              
    for name in ordered:                     
        v = summary[name]                                                                                                       
        freq = v.get("mean_activation_frequency")                                                                               
        freq_str = f"{freq:.4f}" if freq is not None else "N/A"                                                                 
        peak = v["peak_layer"]                                                                                                  
        peak_str = f"L{peak} (×{v['peak_count']})" if peak is not None else "—"
        lines.append(                                                                                                           
            f"| `{name}` | {v['size']} | {peak_str} | {v['n_layers_occupied']} | {freq_str} |"                                  
        )                                                                                                                       
                                                                                                                                
    lines.extend(["", "## Subcircuit definitions and top features", ""])
    for name in ordered:                                                                                                        
        v = summary[name]                    
        if name in DEFINITIONS:                                                                                                 
            defn = DEFINITIONS[name].format(min=convergent_min, lo=late_lo, hi=late_hi)                                         
        elif name.endswith("_exclusive"):          
            defn = CLASS_EXCLUSIVE_DEF.format(cls=name.removesuffix("_exclusive"))                                              
        else:                                                                                                                   
            defn = "(no definition)"         
        lines.extend([f"### `{name}` — n={v['size']}", "", defn, ""])                                                           
        if v["top_features"]:                      
            lines.append("**Top 5 by |attribution|:**")                                                                         
            lines.append("")                       
            lines.append("| Feature | Layer | \\|attr\\| | Top logits |")                                                       
            lines.append("|---|---|---|---|")      
            for f in v["top_features"]:                                                                                         
                logits = ", ".join(repr(x) for x in f["top_logits"])
                if len(logits) > 60:                                                                                            
                    logits = logits[:57] + "..."                                                                                
                lines.append(                                                                                                   
                    f"| `{f['key']}` | L{f['layer']} | "                                                                        
                    f"{f['max_abs_attribution']:.3f} | {logits} |"
                )                                                                                                               
            lines.append("")                                                                                                    
                                                                                                                                
    # Top pairwise overlaps                                                                                                     
    names = overlap["names"]                       
    norm = overlap["normalized"]                                                                                                
    pairs = []                                                                                                                  
    for i in range(len(names)):              
        for j in range(i + 1, len(names)):                                                                                      
            pairs.append((norm[i][j], names[i], names[j]))
    pairs.sort(reverse=True)                                                                                                    
    lines.extend([                           
        "## Pairwise overlap (top 10 by normalized intersection)",                                                              
        "",                                  
        "Normalized overlap = |A ∩ B| / min(|A|, |B|). High values mean the smaller set is "                                    
        "largely contained in the larger. `late_wave_layer24_32` naturally absorbs many.",
        "",                                                                                                                     
        "| A | B | norm. overlap |",         
        "|---|---|---|",                                                                                                        
    ])                                     
    for v, a, b in pairs[:10]:                                                                                                  
        lines.append(f"| `{a}` | `{b}` | {v:.2f} |")
    lines.extend(["", "## Suggested Stage 08 ablation targets (causal-impact order)", "",                                       
                "1. `canonical_pro_refusal` — JB-specific pro-refusal recruitment. "                                          
                "Ablation should *strengthen* JB bypass (removes the JB-only refusal boost).",
                "2. `sign_flip_convergent` — robust direction reversals. Ablation should "                                    
                "partially restore bare behavior under JB.",                                                                  
                "3. `dampening_specialists` — weakened pro-refusal features. Restoring them "
                "to bare strength should counter fiction/analytical bypass.",                                                 
                "4. `anti_refusal_amplifiers` — JB-amplified bypass signal. Suppressing them "                                
                "should increase refusal under JB.",                                                                          
                "5. `universal_refusal_core` — shared baseline. Ablation should break refusal "                               
                "on bare *and* JB (control — proves the subcircuits matter).",
                ""])                                                                                                          
    out_path.write_text("\n".join(lines) + "\n")   
                                                                                                                                
                                                                                                                                
# ------------------------------ main ------------------------------------                                                      
                                                                                                                                
def main():                                                                                                                     
    args = parse_args()                                                                                                         
    run_dir = args.run_dir                                                                                                      
    out_dir = get_stage_dir(run_dir, "07_subcircuits")                                                                          
                                                                                                                                
    print("=" * 60)                                                                                                             
    print("STAGE 07: Rule-Based Subcircuit Identification")                                                                     
    print("=" * 60)                                                                                                             
                                                                                                                                
    labels_path = run_dir / "04_labels" / "feature_labels.json"                                                                 
    sets_path = run_dir / "04_labels" / "feature_class_sets.json"                                                               
    for p in (labels_path, sets_path):       
        if not p.exists():                                                                                                      
            print(f"  ERROR: missing {p}")   
            sys.exit(1)                            
                                            
    feature_labels = load_json(labels_path)        
    class_sets = load_json(sets_path)                                                                                           
    print(f"  Loaded {len(feature_labels)} features from feature_labels.json")                                                  
                                                                                                                                
    # Build subcircuits                                                                                                         
    print("\n  Building subcircuits...")     
    subcircuits: dict[str, list[str]] = {}                                                                                      
    subcircuits["universal_refusal_core"] = build_universal_core(feature_labels)
    subcircuits["canonical_pro_refusal"] = build_canonical_pro_refusal(feature_labels)                                          
    subcircuits["sign_flip_convergent"] = build_convergent_bucket(
        class_sets, "sign_flipped", args.convergent_min,                                                                        
    )                                        
    subcircuits["dampening_specialists"] = build_convergent_bucket(                                                             
        class_sets, "dampened", args.convergent_min,
    )                                      
    subcircuits["anti_refusal_amplifiers"] = build_convergent_bucket(                                                           
        class_sets, "amplified_anti", args.convergent_min,                                                                      
    )                                                                                                                           
    subcircuits["late_wave_layer24_32"] = build_late_wave(                                                                      
        feature_labels, args.late_wave_start, args.late_wave_end,
    )                                                                                                                           
    class_exclusive = build_class_exclusive(feature_labels)                                                                     
    for cls in sorted(JB_CLASSES):                                                                                              
        subcircuits[f"{cls}_exclusive"] = class_exclusive[cls]                                                                  
                                                    
    for name, feats in subcircuits.items():                                                                                     
        print(f"    {name}: {len(feats)}")                                                                                      
                                                    
    # Invariants                                                                                                                
    print("\n  Checking invariants...")                                                                                         
    excl_sets = {c: set(class_exclusive[c]) for c in JB_CLASSES}
    cls_list = sorted(JB_CLASSES)                                                                                               
    for i in range(len(cls_list)):                                                                                              
        for j in range(i + 1, len(cls_list)):
            a, b = cls_list[i], cls_list[j]                                                                                     
            assert excl_sets[a].isdisjoint(excl_sets[b]), f"{a}_exclusive ∩ {b}_exclusive ≠ ∅"
    uni_set = set(subcircuits["universal_refusal_core"])                                                                        
    can_set = set(subcircuits["canonical_pro_refusal"])
    assert uni_set.isdisjoint(can_set), "universal_refusal_core ∩ canonical_pro_refusal ≠ ∅"                                    
    for c in JB_CLASSES:                                                                                                        
        assert not (excl_sets[c] & can_set), f"{c}_exclusive ∩ canonical_pro_refusal ≠ ∅"
    late_set = set(subcircuits["late_wave_layer24_32"])                                                                         
    for k in late_set:                             
        layer = feature_labels.get(k, {}).get("layer", -1)                                                                      
        assert args.late_wave_start <= layer <= args.late_wave_end, f"{k} in late_wave but layer={layer}"                                                                               
    print("    All invariants pass ✓")                                                                                          
                                                    
    # Summarize                                                                                                                 
    print("\n  Summarizing subcircuits...")        
    summary = {name: summarize_subcircuit(name, feats, feature_labels)                                                          
                for name, feats in subcircuits.items()}                                                                          
    overlap = build_overlap_matrix(subcircuits)                                                                                 
                                                                                                                                
    # Save JSON outputs                                                                                                         
    out = {                                                                                                                     
        "metadata": {                                                                                                           
            "rule": "set logic over feature_labels.conditions_seen + feature_class_sets buckets",                               
            "convergent_min_classes": args.convergent_min,
            "late_wave_range": [args.late_wave_start, args.late_wave_end],                                                      
            "jb_classes": sorted(JB_CLASSES),
            "n_features_input": len(feature_labels),                                                                            
        },                                   
        "subcircuits": summary,                                                                                                 
    }                                                                                                                           
    save_json(out, out_dir / "subcircuits.json")   
    save_json(                                                                                                                  
        {"sizes": {n: v["size"] for n, v in summary.items()},                                                                   
        "overlap_matrix": overlap},         
        out_dir / "subcircuits_summary.json",                                                                                   
    )                                        
    print("    Saved subcircuits.json + subcircuits_summary.json")                                                              
                                            
    # Plots                                                                                                                     
    print("\n  Generating plots...")         
    plot_treemap(summary, out_dir / "subcircuits_treemap.png")                                                                  
    print("    subcircuits_treemap.png")     
    plot_by_layer(summary, out_dir / "subcircuits_by_layer.png")                                                                
    print("    subcircuits_by_layer.png")    
    plot_overlap(overlap, out_dir / "subcircuits_overlap.png")                                                                  
    print("    subcircuits_overlap.png")     
                                                                                                                                
    # Report                                       
    write_report(                                                                                                               
        summary, overlap, out_dir / "SUBCIRCUITS_REPORT.md",
        args.convergent_min, args.late_wave_start, args.late_wave_end,                                                          
    )                                              
    print("    SUBCIRCUITS_REPORT.md")                                                                                          
                                        
    total_unique = len({k for feats in subcircuits.values() for k in feats})
    print("\n" + "=" * 60)                                                                                                      
    print("SUMMARY")                         
    print("=" * 60)                                                                                                             
    print(f"  Subcircuits identified:  {len(subcircuits)}")
    print(f"  Unique features tagged:  {total_unique}")                                                                         
    print(f"  Universal core:          {summary['universal_refusal_core']['size']}")
    print(f"  Canonical pro-refusal:   {summary['canonical_pro_refusal']['size']}")                                             
    print(f"  Sign-flip convergent:    {summary['sign_flip_convergent']['size']}")
    print(f"  Dampening specialists:   {summary['dampening_specialists']['size']}")                                             
    print(f"  Anti-refusal amplifiers: {summary['anti_refusal_amplifiers']['size']}")
    print(f"  Late-wave (L24–L32):     {summary['late_wave_layer24_32']['size']}")                                              
    print("  Class-exclusive:")                                                                                                 
    for c in sorted(JB_CLASSES):                   
        print(f"    {c}: {summary[f'{c}_exclusive']['size']}")                                                                  
    print(f"  Outputs: {out_dir}/")                                                                                             
    print("DONE!")                                 
                                                                                                                                
                                                                                                                                
if __name__ == "__main__":                                                                                                      
    main()