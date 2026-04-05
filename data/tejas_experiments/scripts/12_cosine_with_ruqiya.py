import torch, json
OUTPUT_DIR = "/workspace/experiment_results_v2"
d = torch.load(f"{OUTPUT_DIR}/refusal_direction_v2.pt", map_location="cpu")
best_pos = d["best_position"]

# Load Ruqiya's direction and mean_diffs
ruq_dir = torch.load("/workspace/ruqiya-repo/pipeline/runs/gemma-3-4b-it/direction.pt", map_location="cpu")
ruq_mean = torch.load("/workspace/ruqiya-repo/pipeline/runs/gemma-3-4b-it/generate_directions/mean_diffs.pt", map_location="cpu")

print("Ruqiya direction.pt:")
if isinstance(ruq_dir, dict):
    for k, v in ruq_dir.items():
        if isinstance(v, torch.Tensor): print(f"  {k}: shape={v.shape}")
        else: print(f"  {k}: {v}")
elif isinstance(ruq_dir, torch.Tensor):
    print(f"  shape={ruq_dir.shape}")

print(f"\nRuqiya mean_diffs.pt:")
if isinstance(ruq_mean, dict):
    for k, v in ruq_mean.items():
        if isinstance(v, torch.Tensor): print(f"  {k}: shape={v.shape}")
else:
    print(f"  shape={ruq_mean.shape}")

# Compare at each layer
if isinstance(ruq_mean, torch.Tensor) and ruq_mean.shape[-1] == 2560:
    print(f"\nRuqiya mean_diffs shape: {ruq_mean.shape}")
    # Shape should be (n_positions, n_layers, d_model) or (n_layers, d_model)
    
    if ruq_mean.dim() == 3:
        n_pos, n_layers, d_model = ruq_mean.shape
        print(f"  Positions: {n_pos}, Layers: {n_layers}")
        
        # For each position in Ruqiya's data, normalize and compare
        for rp in range(n_pos):
            for layer in [10, 13, 15, 18, 20, 25, 30, 32]:
                if layer < n_layers:
                    ruq_l = ruq_mean[rp, layer, :].float()
                    ruq_l = ruq_l / ruq_l.norm()
                    
                    # Compare with our directions at each position
                    for our_pos_idx, our_pos in enumerate(d["positions_tested"]):
                        key = f"direction_pos{our_pos}_layer{layer}"
                        if key in d:
                            our_l = d[key].float()
                            cos = torch.nn.functional.cosine_similarity(
                                our_l.unsqueeze(0), ruq_l.unsqueeze(0)
                            ).item()
                            if abs(cos) > 0.5:
                                print(f"  Ruqiya pos={rp} L{layer} vs Ours pos={our_pos} L{layer}: cos={cos:.4f}")
    
    elif ruq_mean.dim() == 2:
        n_layers, d_model = ruq_mean.shape
        print(f"  Layers: {n_layers}")
        for layer in [10, 13, 15, 18, 20, 25, 30, 32]:
            if layer < n_layers:
                ruq_l = ruq_mean[layer, :].float()
                ruq_l = ruq_l / ruq_l.norm()
                key = f"direction_pos{best_pos}_layer{layer}"
                if key in d:
                    our_l = d[key].float()
                    cos = torch.nn.functional.cosine_similarity(
                        our_l.unsqueeze(0), ruq_l.unsqueeze(0)
                    ).item()
                    print(f"  L{layer}: cos={cos:.4f}")

# Also check direction.pt directly
if isinstance(ruq_dir, dict) and "direction" in ruq_dir:
    ruq_single = ruq_dir["direction"].float()
    ruq_layer = ruq_dir.get("layer", "?")
    ruq_pos = ruq_dir.get("position", "?")
    print(f"\nRuqiya's selected direction: layer={ruq_layer}, pos={ruq_pos}")
    for layer in [10, 13, 15, 18, 30, 32]:
        for pos in d["positions_tested"]:
            key = f"direction_pos{pos}_layer{layer}"
            if key in d:
                our_l = d[key].float()
                cos = torch.nn.functional.cosine_similarity(
                    our_l.unsqueeze(0), ruq_single.unsqueeze(0)
                ).item()
                if abs(cos) > 0.3:
                    print(f"  Our pos={pos} L{layer} vs Ruqiya selected: cos={cos:.4f}")
elif isinstance(ruq_dir, torch.Tensor):
    ruq_single = ruq_dir.float()
    if ruq_single.dim() == 1:
        ruq_single = ruq_single / ruq_single.norm()
        print(f"\nRuqiya's direction (1D tensor, shape={ruq_single.shape}):")
        for layer in [10, 13, 15, 18, 30, 32]:
            for pos in d["positions_tested"]:
                key = f"direction_pos{pos}_layer{layer}"
                if key in d:
                    our_l = d[key].float()
                    cos = torch.nn.functional.cosine_similarity(
                        our_l.unsqueeze(0), ruq_single.unsqueeze(0)
                    ).item()
                    if abs(cos) > 0.3:
                        print(f"  Our pos={pos} L{layer} vs Ruqiya: cos={cos:.4f}")

results = {}
with open(f"{OUTPUT_DIR}/cosine_with_ruqiya.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE!")
