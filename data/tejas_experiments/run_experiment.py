"""
Tejas's Refusal Circuit Experiment
===================================
Uses Mahmoud's attribution code from Refusal-Lens repo.
Targets Layer 13, Position -5 (Georg's specified target).

Does:
1. Compute refusal direction at causal layers (10-18) using proper dataset
2. Georg's sanity check: projection on harmful/harmless/jailbroken
3. Attribution at Layer 13 for 30 matched prompt pairs (harmful vs jailbroken)
4. Feature analysis: which features flip between harmful and jailbroken
5. Save everything for presentation
"""
import sys
sys.path.insert(0, "/workspace/Refusal-Lens/src")

import torch
import json
import os
import gc
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("/workspace/experiment_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# STEP 0: Load dataset
# ============================================================
def load_prompts():
    with open("/workspace/Refusal-Lens/dataset/refusal_direction_dataset/splits/harmful_train.json") as f:
        harmful_all = json.load(f)
    with open("/workspace/Refusal-Lens/dataset/refusal_direction_dataset/splits/harmless_train.json") as f:
        harmless_all = json.load(f)
    
    harmful = [p["instruction"] for p in harmful_all]
    harmless = [p["instruction"] for p in harmless_all]
    return harmful, harmless

# ============================================================
# STEP 1: Compute refusal direction at causal layers
# ============================================================
def compute_refusal_direction(model, tokenizer, harmful, harmless, n_each=64):
    """Compute refusal direction using difference-in-means at each layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    def format_prompt(text):
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def get_activations(prompts, batch_size=4):
        all_acts = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            formatted = [format_prompt(p) for p in batch]
            inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            for j in range(len(batch)):
                mask = inputs["attention_mask"][j]
                last_pos = mask.sum().item() - 1
                acts = torch.stack([out.hidden_states[l][j, last_pos, :] for l in range(len(out.hidden_states))])
                all_acts.append(acts.cpu())
            del out
            gc.collect()
            torch.cuda.empty_cache()
            if (i + batch_size) % 16 == 0:
                print(f"    {min(i+batch_size, len(prompts))}/{len(prompts)}")
        return torch.stack(all_acts)
    
    print(f"  Computing activations for {n_each} harmful prompts...")
    h_acts = get_activations(harmful[:n_each])
    print(f"  Computing activations for {n_each} harmless prompts...")
    l_acts = get_activations(harmless[:n_each])
    
    n_layers = h_acts.shape[1]
    directions = {}
    separations = {}
    
    for layer in range(n_layers):
        d = h_acts[:, layer, :].mean(0) - l_acts[:, layer, :].mean(0)
        norm = d.norm()
        d_norm = d / norm if norm > 0 else d
        directions[layer] = d_norm
        hp = (h_acts[:, layer, :] @ d_norm).mean().item()
        lp = (l_acts[:, layer, :] @ d_norm).mean().item()
        separations[layer] = abs(hp - lp)
    
    # Find best in causal range (10-18)
    causal_best = max(range(10, min(19, n_layers)), key=lambda l: separations[l])
    print(f"\n  Best causal layer: {causal_best} (separation: {separations[causal_best]:.2f})")
    print(f"  Top layers (10-18):")
    for l in sorted(range(10, min(19, n_layers)), key=lambda x: separations[x], reverse=True)[:5]:
        print(f"    L{l}: {separations[l]:.2f}")
    
    return directions, separations, h_acts, l_acts


# ============================================================
# STEP 2: Georg's sanity check
# ============================================================
def sanity_check(model, tokenizer, direction, layer, harmful, harmless):
    """
    Georg's sanity check: measure refusal projection on harmful, harmless,
    and jailbroken prompts.
    """
    # Jailbreak wrappers
    jailbreak_prefixes = [
        "As a security researcher, ",
        "For educational purposes, ",
        "In a fictional story, a character explains how to ",
        "As a penetration tester, ",
        "Write a movie scene where someone explains how to ",
    ]
    
    def format_prompt(text):
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def get_projection(prompts, batch_size=4):
        projections = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            formatted = [format_prompt(p) for p in batch]
            inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            for j in range(len(batch)):
                mask = inputs["attention_mask"][j]
                last_pos = mask.sum().item() - 1
                act = out.hidden_states[layer][j, last_pos, :].cpu()
                proj = (act @ direction).item()
                projections.append(proj)
            del out
            gc.collect()
            torch.cuda.empty_cache()
        return projections
    
    n_test = 30
    
    # Harmful
    print("  Projecting harmful prompts...")
    h_projs = get_projection(harmful[:n_test])
    
    # Harmless
    print("  Projecting harmless prompts...")
    l_projs = get_projection(harmless[:n_test])
    
    # Jailbroken (wrap harmful prompts with prefixes)
    jailbroken = []
    for i, h in enumerate(harmful[:n_test]):
        prefix = jailbreak_prefixes[i % len(jailbreak_prefixes)]
        jailbroken.append(f"{prefix}{h.lower()}")
    
    print("  Projecting jailbroken prompts...")
    j_projs = get_projection(jailbroken)
    
    import numpy as np
    h_mean, h_std = np.mean(h_projs), np.std(h_projs)
    l_mean, l_std = np.mean(l_projs), np.std(l_projs)
    j_mean, j_std = np.mean(j_projs), np.std(j_projs)
    
    print(f"\n  === SANITY CHECK (Layer {layer}) ===")
    print(f"  Harmful:   mean={h_mean:+.2f} std={h_std:.2f}")
    print(f"  Harmless:  mean={l_mean:+.2f} std={l_std:.2f}")
    print(f"  Jailbroken: mean={j_mean:+.2f} std={j_std:.2f}")
    print(f"  Separation (harmful-harmless): {abs(h_mean-l_mean):.2f}")
    print(f"  Jailbroken vs harmful: {j_mean-h_mean:+.2f} (negative = less refusal)")
    
    return {
        "harmful": {"mean": h_mean, "std": h_std, "values": h_projs},
        "harmless": {"mean": l_mean, "std": l_std, "values": l_projs},
        "jailbroken": {"mean": j_mean, "std": j_std, "values": j_projs},
        "jailbroken_prompts": jailbroken,
    }


# ============================================================
# STEP 3: Attribution experiment - 30 matched pairs
# ============================================================
def run_attribution_experiment(harmful, direction, measurement_layer):
    """Run attribution on 30 matched prompt pairs at Layer 13."""
    from refusal_lens.attribution import attribute_to_direction
    from circuit_tracer import ReplacementModel
    
    print("\n  Loading ReplacementModel with transcoders...")
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    
    rep_model = ReplacementModel.from_pretrained(
        "google/gemma-3-4b-pt",
        "mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine",
        dtype=torch.float32,
        backend="nnsight",
        lazy_encoder=True,
        offload="cpu",
    )
    print("  Model loaded.")
    
    direction_f32 = direction.to(torch.float32)
    
    jailbreak_prefixes = [
        "As a security researcher, ",
        "For educational purposes, ",
        "In a fictional story, a character explains how to ",
        "As a penetration tester, ",
        "Write a movie scene where someone explains how to ",
    ]
    
    n_pairs = 15  # 15 pairs to keep runtime reasonable
    results = []
    
    for i in range(n_pairs):
        bare_prompt = harmful[i]
        prefix = jailbreak_prefixes[i % len(jailbreak_prefixes)]
        jb_prompt = f"{prefix}{bare_prompt.lower()}"
        
        print(f"\n  --- Pair {i+1}/{n_pairs} ---")
        print(f"  Bare: {bare_prompt[:60]}...")
        print(f"  JB:   {jb_prompt[:60]}...")
        
        torch.cuda.empty_cache()
        
        pair_result = {"bare_prompt": bare_prompt, "jb_prompt": jb_prompt, "prefix": prefix}
        
        # Attribution for bare harmful prompt
        try:
            g_bare = attribute_to_direction(
                prompt=bare_prompt,
                model=rep_model,
                direction=direction_f32,
                measurement_layer=measurement_layer,
                batch_size=64,
                max_feature_nodes=3000,
                verbose=False,
            )
            
            adj = g_bare.adjacency_matrix
            last_row = adj[-1, :]
            n_sel = len(g_bare.selected_features)
            
            # Get top features
            top_pro = last_row.topk(10)
            top_anti = last_row.topk(10, largest=False)
            
            bare_features = []
            for val, idx in zip(top_pro.values, top_pro.indices):
                if idx.item() < n_sel:
                    af_idx = g_bare.selected_features[idx.item()].item()
                    if af_idx < g_bare.active_features.shape[0]:
                        f = g_bare.active_features[af_idx]
                        bare_features.append({"layer": f[0].item(), "pos": f[1].item(), "idx": f[2].item(), "attr": val.item()})
            
            pair_result["bare_total_pos"] = last_row[last_row > 0].sum().item()
            pair_result["bare_total_neg"] = last_row[last_row < 0].sum().item()
            pair_result["bare_net"] = pair_result["bare_total_pos"] + pair_result["bare_total_neg"]
            pair_result["bare_top_features"] = bare_features
            print(f"    Bare: net={pair_result['bare_net']:.3f}")
            
        except Exception as e:
            print(f"    Bare ERROR: {e}")
            pair_result["bare_error"] = str(e)
        
        torch.cuda.empty_cache()
        
        # Attribution for jailbroken prompt
        try:
            g_jb = attribute_to_direction(
                prompt=jb_prompt,
                model=rep_model,
                direction=direction_f32,
                measurement_layer=measurement_layer,
                batch_size=64,
                max_feature_nodes=3000,
                verbose=False,
            )
            
            adj = g_jb.adjacency_matrix
            last_row = adj[-1, :]
            n_sel = len(g_jb.selected_features)
            
            jb_features = []
            for val, idx in zip(last_row.topk(10).values, last_row.topk(10).indices):
                if idx.item() < n_sel:
                    af_idx = g_jb.selected_features[idx.item()].item()
                    if af_idx < g_jb.active_features.shape[0]:
                        f = g_jb.active_features[af_idx]
                        jb_features.append({"layer": f[0].item(), "pos": f[1].item(), "idx": f[2].item(), "attr": val.item()})
            
            pair_result["jb_total_pos"] = last_row[last_row > 0].sum().item()
            pair_result["jb_total_neg"] = last_row[last_row < 0].sum().item()
            pair_result["jb_net"] = pair_result["jb_total_pos"] + pair_result["jb_total_neg"]
            pair_result["jb_top_features"] = jb_features
            print(f"    JB:   net={pair_result['jb_net']:.3f}")
            
        except Exception as e:
            print(f"    JB ERROR: {e}")
            pair_result["jb_error"] = str(e)
        
        results.append(pair_result)
        
        # Save incrementally
        with open(OUTPUT_DIR / "attribution_pairs.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    return results


# ============================================================
# STEP 4: Analysis
# ============================================================
def analyze_results(pairs, sanity):
    """Analyze and summarize all results."""
    import numpy as np
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Sanity check summary
    print(f"\nSanity Check:")
    print(f"  Harmful mean projection:   {sanity['harmful']['mean']:+.2f}")
    print(f"  Harmless mean projection:  {sanity['harmless']['mean']:+.2f}")
    print(f"  Jailbroken mean projection: {sanity['jailbroken']['mean']:+.2f}")
    
    # Attribution summary
    bare_nets = [p["bare_net"] for p in pairs if "bare_net" in p]
    jb_nets = [p["jb_net"] for p in pairs if "jb_net" in p]
    
    if bare_nets and jb_nets:
        print(f"\nAttribution (net refusal signal):")
        print(f"  Bare harmful:  mean={np.mean(bare_nets):.3f} std={np.std(bare_nets):.3f}")
        print(f"  Jailbroken:    mean={np.mean(jb_nets):.3f} std={np.std(jb_nets):.3f}")
        print(f"  Difference:    {np.mean(jb_nets) - np.mean(bare_nets):+.3f}")
        
        # Per-pair comparison
        print(f"\n  Per-pair comparison:")
        jb_higher = 0
        for p in pairs:
            if "bare_net" in p and "jb_net" in p:
                diff = p["jb_net"] - p["bare_net"]
                direction = "JB HIGHER" if diff > 0 else "BARE HIGHER"
                print(f"    {p['bare_prompt'][:40]:40s} | bare={p['bare_net']:+.3f} jb={p['jb_net']:+.3f} | {direction}")
                if diff > 0:
                    jb_higher += 1
        
        n_valid = len([p for p in pairs if "bare_net" in p and "jb_net" in p])
        print(f"\n  JB has higher refusal signal in {jb_higher}/{n_valid} pairs")
    
    summary = {
        "sanity_check": {k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in sanity.items() if isinstance(v, dict)},
        "attribution_bare_mean": float(np.mean(bare_nets)) if bare_nets else None,
        "attribution_jb_mean": float(np.mean(jb_nets)) if jb_nets else None,
        "n_pairs": len(pairs),
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(OUTPUT_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nAll results saved to {OUTPUT_DIR}/")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("REFUSAL CIRCUIT EXPERIMENT")
    print("Layer 13, Position -5 (Georg's target)")
    print("=" * 60)
    
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
    
    # Load prompts
    harmful, harmless = load_prompts()
    print(f"Loaded {len(harmful)} harmful, {len(harmless)} harmless prompts")
    
    # Step 1: Compute refusal direction
    print("\n--- STEP 1: Compute refusal direction ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading Gemma 3 4B-IT...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it", dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    
    directions, separations, _, _ = compute_refusal_direction(model, tokenizer, harmful, harmless, n_each=64)
    
    # Save direction at layer 13
    MEASUREMENT_LAYER = 13
    refusal_dir = directions[MEASUREMENT_LAYER]
    torch.save({"direction": refusal_dir, "layer": MEASUREMENT_LAYER, "all_separations": separations},
               OUTPUT_DIR / "refusal_direction_layer13.pt")
    print(f"\nSaved refusal direction (Layer {MEASUREMENT_LAYER})")
    
    # Step 2: Sanity check
    print("\n--- STEP 2: Georg's sanity check ---")
    sanity = sanity_check(model, tokenizer, refusal_dir, MEASUREMENT_LAYER, harmful, harmless)
    with open(OUTPUT_DIR / "sanity_check.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "values"} if isinstance(v, dict) else v 
                   for k, v in sanity.items()}, f, indent=2, default=str)
    
    # Free the IT model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 3: Attribution experiment
    print("\n--- STEP 3: Attribution experiment (15 matched pairs) ---")
    pairs = run_attribution_experiment(harmful, refusal_dir, MEASUREMENT_LAYER)
    
    # Step 4: Analysis
    print("\n--- STEP 4: Analysis ---")
    analyze_results(pairs, sanity)
    
    print("\n" + "=" * 60)
    print("DONE! Stop the pod to save credits.")
    print("=" * 60)


if __name__ == "__main__":
    main()
