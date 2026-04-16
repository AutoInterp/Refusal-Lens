"""                                                                                                                                                       
Stage 04: Feature Labeling via HuggingFace Dashboard Data                                                                                               
==========================================================                                                                                                
Labels ALL unique transcoder features from Stage 02 attribution results                                                                                   
using pre-computed dashboard data from HuggingFace.                                                                                                       
                                                                                                                                                        
For each feature, retrieves:                                                                                                                              
- Top/bottom logits (which tokens the feature promotes/suppresses)                                                                                        
- Top activating examples (text where the feature fires strongest)                                                                                        
- Activation frequency and statistics                                                                                                                     
                                                                                                                                                        
These are loaded from the binary feature files on HuggingFace:                                                                                            
mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine/features/                                                                          
                                                                                                                                                        
The binary format is: 4-byte header + gzip-compressed JSON per feature,                                                                                   
indexed by index.json.gz which gives byte offsets per layer/feature.                                                                                      
                                                                                                                                                        
Inputs:  02_attribution/attribution_results.json, feature_comparison_aggregate.json                                                                       
Outputs: 04_labels/                                                                                                                                       
"""                                    
from __future__ import annotations                                                                                                                        
                                        
import argparse                                                                                                                                           
import gzip
import json                                                                                                                                               
import sys                             
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))                                                                                                  
import config
from utils import save_json, load_json, get_stage_dir                                                                                                     
                                                                                                                                                        
                                                                                                                                                        
# HuggingFace feature data config                                                                                                                         
HF_REPO = "mwhanna/gemma-scope-2-4b-it"                                                                                                                   
HF_FEATURES_PATH = "transcoder_all/width_16k_l0_small_affine/features"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FEATURES_PATH}"                                                                         
REQUEST_DELAY = 0.05  # seconds between HF requests (be polite)                                                                                           
                                                                                                                                                        
                                                                                                                                                        
def parse_args():                                                                                                                                         
    parser = argparse.ArgumentParser(description="Label features via HuggingFace dashboard data")
    parser.add_argument("--run-dir", type=Path, required=True)                                                                                            
    parser.add_argument(
        "--skip-download", action="store_true",                                                                                                           
        help="Skip HuggingFace downloads (use cache only)",
    )                                                                                                                                                     
    parser.add_argument(               
        "--max-features", type=int, default=None,                                                                                                         
        help="Limit number of features to download (for testing). None = all.",
    )                                                                                                                                                     
    parser.add_argument(
        "--n-examples", type=int, default=3,                                                                                                              
        help="Number of activation examples to store per feature (default: 3)",
    )                                                                                                                                                     
    parser.add_argument(
        "--n-logits", type=int, default=10,                                                                                                               
        help="Number of top/bottom logits to store per feature (default: 10)",
    )                                                                                                                                                     
    return parser.parse_args()
                                                                                                                                                        
                                                                                                                                                        
def load_hf_index():
    """Download and parse the feature index from HuggingFace."""                                                                                          
    from huggingface_hub import hf_hub_download

    print("  Downloading feature index from HuggingFace...")                                                                                              
    idx_path = hf_hub_download(
        HF_REPO,                                                                                                                                          
        f"{HF_FEATURES_PATH}/index.json.gz",
    )                                                                                                                                                     
    with gzip.open(idx_path, "rt") as f:
        index = json.load(f)                                                                                                                              
    # Index has version, format, and per-layer entries
    n_layers = sum(1 for k in index if k.isdigit())                                                                                                       
    print(f"    Index loaded: {n_layers} layers, format={index.get('format')}")                                                                           
    return index                                                                                                                                          
                                                                                                                                                        
                                                                                                                                                        
def fetch_feature_data(index: dict, layer: int, feat_idx: int) -> dict | None:
    """                                                                                                                                                   
    Fetch a single feature's dashboard data from HuggingFace via byte-range request.
                                                                                                                                                        
    Returns parsed JSON with keys: transcoder_id, index, examples_quantiles,
    top_logits, bottom_logits, act_min, act_max, quantile_values, histogram,                                                                              
    activation_frequency.                                                                                                                                 
    """                                                                                                                                                   
    layer_key = str(layer)                                                                                                                                
    if layer_key not in index:         
        return None

    offsets = index[layer_key]["offsets"]                                                                                                                 
    if feat_idx >= len(offsets) - 1:
        return None                                                                                                                                       
                                        
    start = offsets[feat_idx]
    end = offsets[feat_idx + 1]
    if start == end:                                                                                                                                      
        return None  # Empty feature
                                                                                                                                                        
    filename = index[layer_key]["filename"]                                                                                                               
    url = f"{HF_BASE_URL}/{filename}"
                                                                                                                                                        
    try:                               
        resp = requests.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=15)
        if resp.status_code not in (200, 206):                                                                                                            
            return None
                                                                                                                                                        
        raw = resp.content             
        gzip_start = raw.find(b"\x1f\x8b")                                                                                                                
        if gzip_start < 0:             
            return None

        return json.loads(gzip.decompress(raw[gzip_start:]))                                                                                              
    except Exception as e:
        print(f"      Error fetching L{layer}:F{feat_idx}: {e}")                                                                                          
        return None                    

                                                                                                                                                        
def extract_label(data: dict, n_examples: int = 3, n_logits: int = 10) -> dict:
    """                                                                                                                                                   
    Extract a concise label from feature dashboard data.
                                                                                                                                                        
    Returns a dict with top/bottom logits, activation examples, and statistics.
    """                                                                                                                                                   
    label = {                          
        "top_logits": data.get("top_logits", [])[:n_logits],                                                                                              
        "bottom_logits": data.get("bottom_logits", [])[:n_logits],
        "activation_frequency": data.get("activation_frequency"),                                                                                         
        "act_min": data.get("act_min"),
        "act_max": data.get("act_max"),                                                                                                                   
    }                                                                                                                                                     

    # Extract top activating examples                                                                                                                     
    examples = []                      
    for quantile in data.get("examples_quantiles", []):
        for ex in quantile.get("examples", [])[:n_examples]:                                                                                              
            tokens = ex.get("tokens", [])
            acts = ex.get("tokens_acts_list", [])                                                                                                         
            trigger_idx = ex.get("train_token_ind", 0)
                                                                                                                                                        
            # Get context around trigger token
            start = max(0, trigger_idx - 8)                                                                                                               
            end = min(len(tokens), trigger_idx + 8)
            context_tokens = tokens[start:end]
            trigger_token = tokens[trigger_idx] if trigger_idx < len(tokens) else ""                                                                      
            trigger_act = acts[trigger_idx] if trigger_idx < len(acts) else 0.0
                                                                                                                                                        
            examples.append({          
                "trigger_token": trigger_token,                                                                                                           
                "trigger_activation": round(trigger_act, 2),
                "context": "".join(context_tokens),                                                                                                       
                "quantile": quantile.get("quantile_name", ""),
            })                                                                                                                                            
        if examples:                   
            break  # Only take from top quantile                                                                                                          
                                                                                                                                                        
    label["examples"] = examples[:n_examples]
    return label                                                                                                                                          
                                        
                                                                                                                                                        
def _parse_feature_key(key: str) -> tuple[int, int]:
    """Parse 'L29:F1066' into (29, 1066)."""
    parts = key.split(":")
    return int(parts[0][1:]), int(parts[1][1:])


def _ensure_feature(features: dict, key: str, attr_val: float = 0.0) -> None:
    """Add a feature to the dict if not already present."""
    if key not in features:
        layer, feat_idx = _parse_feature_key(key)
        features[key] = {
            "layer": layer,
            "feature_idx": feat_idx,
            "max_abs_attribution": abs(attr_val),
            "conditions_seen": set(),
        }


def collect_all_features(results: list[dict]) -> dict[str, dict]:
    """
    Collect ALL unique features from both top50 attribution data AND
    feature comparison data (sign-flipped, dampened, amplified-anti).

    This ensures features that appear in comparisons but aren't in the
    top-50 by attribution magnitude still get labeled.
    """
    features = {}

    for row in results:
        # Source 1: top50_features per condition
        conds = row.get("conditions", row)
        for cls_name, cond in conds.items():
            if isinstance(cond, dict) and "error" not in cond:
                top50 = cond.get("top50_features", {})
                for key, attr_val in top50.items():
                    _ensure_feature(features, key, attr_val)
                    features[key]["max_abs_attribution"] = max(
                        features[key]["max_abs_attribution"], abs(attr_val)
                    )
                    features[key]["conditions_seen"].add(cls_name)

        # Source 2: feature comparison data (sign-flipped, dampened, amplified-anti)
        comp = row.get("feature_comparison", {})
        for cls_name, cls_comp in comp.items():
            for feat in cls_comp.get("top_sign_flipped", []):
                key = feat["key"]
                attr_val = max(abs(feat.get("bare_attr", 0)), abs(feat.get("cls_attr", 0)))
                _ensure_feature(features, key, attr_val)
                features[key]["max_abs_attribution"] = max(
                    features[key]["max_abs_attribution"], attr_val
                )
                features[key]["conditions_seen"].add(cls_name)
            for feat in cls_comp.get("top_dampened", []):
                key = feat["key"]
                _ensure_feature(features, key)
                features[key]["conditions_seen"].add(cls_name)
            for feat in cls_comp.get("top_amplified_anti", []):
                key = feat["key"]
                _ensure_feature(features, key)
                features[key]["conditions_seen"].add(cls_name)

    for key in features:
        features[key]["conditions_seen"] = sorted(features[key]["conditions_seen"])
    return features                                                                                                                                       

                                                                                                                                                        
def collect_comparison_features(results: list[dict]) -> dict:
    """Collect mechanistically interesting features from comparisons."""
    categories = {                                                                                                                                        
        "sign_flipped": defaultdict(lambda: {"count": 0, "classes": set(), "examples": []}),
        "dampened": defaultdict(lambda: {"count": 0, "classes": set(), "examples": []}),                                                                  
        "amplified_anti": defaultdict(lambda: {"count": 0, "classes": set(), "examples": []}),                                                            
    }                                                                                                                                                     
                                                                                                                                                        
    for row in results:                                                                                                                                   
        comp = row.get("feature_comparison", {})
        for cls_name, cls_comp in comp.items():
            for feat in cls_comp.get("top_sign_flipped", []):                                                                                             
                key = feat["key"]                                                                                                                         
                categories["sign_flipped"][key]["count"] += 1                                                                                             
                categories["sign_flipped"][key]["classes"].add(cls_name)                                                                                  
                categories["sign_flipped"][key]["examples"].append({                                                                                      
                    "class": cls_name,
                    "bare_attr": feat["bare_attr"],                                                                                                       
                    "cls_attr": feat["cls_attr"],                                                                                                         
                })
            for feat in cls_comp.get("top_dampened", []):                                                                                                 
                key = feat["key"]      
                categories["dampened"][key]["count"] += 1
                categories["dampened"][key]["classes"].add(cls_name)                                                                                      
                categories["dampened"][key]["examples"].append({
                    "class": cls_name, "delta": feat["delta"],                                                                                            
                })                                                                                                                                        
            for feat in cls_comp.get("top_amplified_anti", []):
                key = feat["key"]                                                                                                                         
                categories["amplified_anti"][key]["count"] += 1
                categories["amplified_anti"][key]["classes"].add(cls_name)
                categories["amplified_anti"][key]["examples"].append({                                                                                    
                    "class": cls_name, "delta": feat["delta"],
                })                                                                                                                                        
                                        
    result = {}
    for cat_name, cat_data in categories.items():
        result[cat_name] = {}                                                                                                                             
        for key, info in cat_data.items():
            result[cat_name][key] = {                                                                                                                     
                "count": info["count"],
                "classes": sorted(info["classes"]),
                "examples": info["examples"],                                                                                                             
            }
    return result                                                                                                                                         
                                        

def main():
    args = parse_args()
    run_dir = args.run_dir
    out_dir = get_stage_dir(run_dir, "04_labels")
                                                                                                                                                        
    print("=" * 60)
    print("STAGE 04: Feature Labeling (HuggingFace Dashboard Data)")                                                                                      
    print("=" * 60)                                                                                                                                       

    # Load attribution results                                                                                                                            
    attr_path = run_dir / "02_attribution" / "attribution_results.json"
    if not attr_path.exists():
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
                                                                                                                                                        
    # Step 1: Collect all unique features
    print("\n  Step 1: Collecting all unique features...")
    all_features = collect_all_features(results)                                                                                                          
    print(f"    Found {len(all_features)} unique features")
                                                                                                                                                        
    # Step 2: Collect comparison features (priority set)                                                                                                  
    print("  Step 2: Collecting comparison features...")
    comparison_features = collect_comparison_features(results)                                                                                            
    priority_features = set()                                                                                                                             
    for cat_data in comparison_features.values():
        priority_features.update(cat_data.keys())                                                                                                         
    print(f"    Priority features (in comparisons): {len(priority_features)}")
    print(f"    Sign-flipped: {len(comparison_features['sign_flipped'])}, "                                                                               
        f"Dampened: {len(comparison_features['dampened'])}, "                                                                                           
        f"Amplified-anti: {len(comparison_features['amplified_anti'])}")                                                                                
                                                                                                                                                        
    # Step 3: Load HF index and fetch feature data
    cache_path = out_dir / "feature_labels_cache.json"                                                                                                    
    if cache_path.exists():                                                                                                                               
        cache = load_json(cache_path)
        print(f"  Step 3: Loaded {len(cache)} cached labels")                                                                                             
    else:                                                                                                                                                 
        cache = {}
                                                                                                                                                        
    if not args.skip_download:         
        index = load_hf_index()
                                                                                                                                                        
        # Sort: priority features first, then by attribution magnitude
        sorted_features = sorted(                                                                                                                         
            all_features.items(),      
            key=lambda x: (x[0] not in priority_features, -x[1]["max_abs_attribution"]),                                                                  
        )                                                                                                                                                 
                                                                                                                                                        
        n_total = len(sorted_features)                                                                                                                    
        if args.max_features is not None:
            n_total = min(n_total, args.max_features)
                                                                                                                                                        
        print(f"  Step 3: Fetching {n_total} features from HuggingFace...")
        n_fetched = 0                                                                                                                                     
        n_found = 0                    
        n_cached = 0                                                                                                                                      

        for i, (key, info) in enumerate(sorted_features[:n_total]):                                                                                       
            if key in cache:           
                n_cached += 1                                                                                                                             
                if cache[key] is not None:
                    n_found += 1                                                                                                                          
                continue               

            data = fetch_feature_data(index, info["layer"], info["feature_idx"])                                                                          
            if data is not None:
                cache[key] = extract_label(data, args.n_examples, args.n_logits)                                                                          
                n_found += 1                                                                                                                              
            else:
                cache[key] = None                                                                                                                         
                                        
            n_fetched += 1
            if n_fetched % 50 == 0:
                print(f"    Fetched {n_fetched}, found {n_found}, "                                                                                       
                    f"cached {n_cached} ({i+1}/{n_total})")
                save_json(cache, cache_path)                                                                                                              
                                        
            time.sleep(REQUEST_DELAY)                                                                                                                     
                                        
        save_json(cache, cache_path)                                                                                                                      
        print(f"    Done: {n_fetched} fetched, {n_found} total found, {n_cached} from cache")
    else:                                                                                                                                                 
        print("  Step 3: Skipping downloads (--skip-download)")
                                                                                                                                                        
    # Step 4: Build feature labels                                                                                                                        
    print("  Step 4: Building feature labels...")
    feature_labels = {}                                                                                                                                   
    n_labeled = 0                      

    for key, info in all_features.items():
        cached = cache.get(key)
        feature_labels[key] = {                                                                                                                           
            "layer": info["layer"],
            "feature_idx": info["feature_idx"],                                                                                                           
            "max_abs_attribution": info["max_abs_attribution"],
            "conditions_seen": info["conditions_seen"],                                                                                                   
            "top_logits": cached["top_logits"] if cached else None,
            "bottom_logits": cached["bottom_logits"] if cached else None,                                                                                 
            "activation_frequency": cached["activation_frequency"] if cached else None,                                                                   
            "examples": cached["examples"] if cached else None,
            "labeled": cached is not None,                                                                                                                
        }                                                                                                                                                 
        if cached is not None:
            n_labeled += 1                                                                                                                                
                                        
    save_json(feature_labels, out_dir / "feature_labels.json")                                                                                            

    # Step 5: Merge with comparison data                                                                                                                  
    print("  Step 5: Merging labels with feature comparison data...")
    comparison_labeled = {}
    for category, features in comparison_features.items():
        labeled_list = []                                                                                                                                 
        for key, info in sorted(features.items(),
                                key=lambda x: x[1]["count"], reverse=True):                                                                               
            label = feature_labels.get(key, {})
            labeled_list.append({                                                                                                                         
                "key": key,            
                "layer": label.get("layer"),                                                                                                              
                "feature_idx": label.get("feature_idx"),                                                                                                  
                "top_logits": label.get("top_logits"),
                "bottom_logits": label.get("bottom_logits"),                                                                                              
                "activation_frequency": label.get("activation_frequency"),
                "examples": label.get("examples"),                                                                                                        
                "count": info["count"],
                "classes": info["classes"],                                                                                                               
                "comparison_examples": info["examples"][:5],
            })                                                                                                                                            
        comparison_labeled[category] = labeled_list
                                                                                                                                                        
    save_json(comparison_labeled, out_dir / "feature_comparison_labeled.json")                                                                            

    # Step 6: Coverage stats                                                                                                                              
    coverage_pct = round(n_labeled / len(feature_labels) * 100, 1) if feature_labels else 0
    coverage = {                                                                                                                                          
        "total_features": len(feature_labels),
        "labeled": n_labeled,                                                                                                                             
        "unlabeled": len(feature_labels) - n_labeled,
        "coverage_pct": coverage_pct,                                                                                                                     
        "source": f"HuggingFace: {HF_REPO}",                                                                                                              
        "priority_features": len(priority_features),
        "priority_labeled": sum(                                                                                                                          
            1 for k in priority_features
            if feature_labels.get(k, {}).get("labeled", False)                                                                                            
        ),                                                                                                                                                
        "comparison_counts": {
            "sign_flipped": len(comparison_features["sign_flipped"]),                                                                                     
            "dampened": len(comparison_features["dampened"]),
            "amplified_anti": len(comparison_features["amplified_anti"]),                                                                                 
        },
    }                                                                                                                                                     
    save_json(coverage, out_dir / "label_coverage.json")

    # Step 7: Top features report                                                                                                                         
    print("  Step 6: Generating top features report...")
    report_lines = [                                                                                                                                      
        "# Feature Labels Report",     
        "",                                                                                                                                               
        f"**Total unique features**: {len(feature_labels)}",
        f"**Labeled**: {n_labeled} ({coverage_pct}%)",                                                                                                    
        f"**Source**: {HF_REPO} dashboard data (top logits + activation examples)",                                                                       
        "",                                                                                                                                               
    ]                                                                                                                                                     
                                                                                                                                                        
    for category, cat_label in [                                                                                                                          
        ("sign_flipped", "Sign-Flipped Features"),
        ("dampened", "Dampened Features (pro-refusal weakened by JB)"),                                                                                   
        ("amplified_anti", "Amplified Anti-Refusal Features"),                                                                                            
    ]:
        report_lines.extend([f"## {cat_label}", ""])                                                                                                      
        items = comparison_labeled.get(category, [])[:15]                                                                                                 
        if items:                                                                                                                                         
            report_lines.append("| Feature | Top Logits | Freq | Count | Classes |")                                                                      
            report_lines.append("|---------|-----------|------|-------|---------|")                                                                       
            for item in items:         
                logits = item.get("top_logits") or []                                                                                                     
                logit_str = ", ".join(repr(l) for l in logits[:5])                                                                                        
                if len(logit_str) > 50:
                    logit_str = logit_str[:47] + "..."                                                                                                    
                freq = item.get("activation_frequency")
                freq_str = f"{freq:.4f}" if freq is not None else "N/A"                                                                                   
                report_lines.append(                                                                                                                      
                    f"| {item['key']} | {logit_str} | {freq_str} | "
                    f"{item['count']} | {', '.join(item['classes'])} |"                                                                                   
                )                      
        report_lines.append("")                                                                                                                           
                                        
    with open(out_dir / "top_features_report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print("    Saved top_features_report.md")                                                                                                             
                                                                                                                                                        
    # Summary                                                                                                                                             
    print(f"\n{'='*60}")                                                                                                                                  
    print("SUMMARY")                   
    print(f"{'='*60}")
    print(f"  Total features:    {len(feature_labels)}")
    print(f"  Labeled:           {n_labeled} ({coverage_pct}%)")                                                                                          
    print(f"  Priority features: {len(priority_features)}")
    print(f"  Outputs: {out_dir}/")                                                                                                                       
    print("DONE!")                     
                                                                                                                                                        
                                        
if __name__ == "__main__":
    main()