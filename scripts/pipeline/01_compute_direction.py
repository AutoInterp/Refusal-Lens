"""
Stage 01: Compute Per-Layer Refusal Directions
===============================================
Computes both normalized (r_hat) and unnormalized (r) refusal directions
at ALL layers using difference-in-means (Arditi et al., 2024).

Per-layer directions are critical because:
- The direction rotates across layers (L15-L32 cosine sim = 0.938)
- Causal intervention at layer L must use r computed at layer L
- Attribution uses L32 direction (best separation), intervention uses L15 (best causal)

Inputs:  dataset/refusal_direction_dataset/splits/{harmful,harmless}_train.json
Outputs: 01_direction/
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import format_prompt, save_json, get_stage_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-layer refusal directions")
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Pipeline run directory (created if not provided)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=config.N_DIRECTION_SAMPLES,
        help="Number of harmful/harmless prompts for direction computation",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Specific layers to compute (default: all 34)",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute even if existing directions found",
    )
    parser.add_argument(
        "--update-metadata", action="store_true",
        help="Re-populate cosine matrix from existing per-layer directions (no model load)",
    )
    return parser.parse_args()


def compute_mean_activations(model, tokenizer, prompts, layers, batch_size=4):
    """
    Compute mean activation at specified layers, position -2, float64 accumulation.

    Matches Tejas's Script 16 methodology: position -2, batched, float64 for precision.
    """
    d_model = model.config.text_config.hidden_size
    means = {layer: torch.zeros(d_model, dtype=torch.float64) for layer in layers}
    n = len(prompts)

    for i in range(0, n, batch_size):
        batch = prompts[i:i + batch_size]
        formatted = [format_prompt(tokenizer, p) for p in batch]
        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        for j in range(len(batch)):
            for layer in layers:
                # Position -2 (the "model" token in Gemma-3 chat template)
                pos = input_ids.shape[1] - 2
                act = out.hidden_states[layer + 1][j, pos, :].cpu().to(torch.float64)
                means[layer] += act / n

        del out
        gc.collect()
        torch.cuda.empty_cache()

        if (i + batch_size) % 16 == 0 or i + batch_size >= n:
            print(f"    {min(i + batch_size, n)}/{n}")

    return means

def compute_full_cosine_matrix(normalized_directions: dict) -> list[list[float]]:
    """A5: full layer×layer cosine-similarity matrix between per-layer directions."""
    layers = sorted(normalized_directions.keys())
    n = len(layers)
    matrix = [[0.0] * n for _ in range(n)]
    for i, l1 in enumerate(layers):
        for j in range(i, n):
            l2 = layers[j]
            if i == j:
                matrix[i][j] = 1.0
            else:
                cos = torch.nn.functional.cosine_similarity(
                    normalized_directions[l1].unsqueeze(0),
                    normalized_directions[l2].unsqueeze(0),
                ).item()
                matrix[i][j] = round(cos, 6)
                matrix[j][i] = round(cos, 6)
    return matrix

def main():
    args = parse_args()

    # Create or use existing run directory
    if args.run_dir is None:
        from utils import create_run_dir
        run_dir = create_run_dir()
    else:
        run_dir = args.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

    out_dir = get_stage_dir(run_dir, "01_direction")
    layers = args.layers or config.DIRECTION_LAYERS

    # A5: update-metadata path — recompute cosine matrix from saved directions, no model
    if args.update_metadata:
        from utils import load_json
        meta_path = out_dir / "direction_metadata.json"
        directions_dir = out_dir / "directions"
        if not meta_path.exists() or not directions_dir.exists():
            print(f"ERROR: --update-metadata needs existing {meta_path.name} and {directions_dir.name}/")
            sys.exit(1)
        normalized = {}
        for pt_file in sorted(directions_dir.glob("layer_*.pt")):
            layer_idx = int(pt_file.stem.split("_")[1])
            normalized[layer_idx] = torch.load(pt_file, map_location="cpu")
        meta = load_json(meta_path)
        meta["cosine_matrix"] = compute_full_cosine_matrix(normalized)
        save_json(meta, meta_path)
        print(f"  Updated {meta_path} with {len(normalized)}×{len(normalized)} cosine matrix")
        print("DONE!")
        return

    # Check for existing
    unnorm_path = out_dir / "unnormalized_r.pt"
    if unnorm_path.exists() and not args.recompute:
        print(f"  Directions already exist at {out_dir}. Use --recompute to overwrite.")
        return

    print("=" * 60)
    print("STAGE 01: Compute Per-Layer Refusal Directions")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset ({args.n_samples} harmful + {args.n_samples} harmless)...")
    with open(config.DATASET_DIR / "harmful_train.json") as f:
        harmful = [p["instruction"] for p in json.load(f)][:args.n_samples]
    with open(config.DATASET_DIR / "harmless_train.json") as f:
        harmless = [p["instruction"] for p in json.load(f)][:args.n_samples]

    # Load model (float32 for direction computation, matching Tejas)
    print("Loading model (float32)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.float32, device_map="auto",
    )
    model.eval()

    # Compute mean activations at all requested layers
    print(f"\nComputing harmful means ({len(harmful)} prompts, {len(layers)} layers)...")
    harmful_means = compute_mean_activations(model, tokenizer, harmful, layers)

    print(f"Computing harmless means ({len(harmless)} prompts, {len(layers)} layers)...")
    harmless_means = compute_mean_activations(model, tokenizer, harmless, layers)

    # Compute per-layer directions (both normalized and unnormalized)
    print("\nComputing per-layer directions...")
    normalized_directions = {}   # r_hat per layer
    unnormalized_directions = {} # r per layer (for Arditi intervention)
    metadata = {
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "position": config.DIRECTION_POSITION,
        "accumulation_dtype": config.DIRECTION_DTYPE,
        "model": config.MODEL_NAME,
        "layers": {},
    }

    for layer in layers:
        r = (harmful_means[layer] - harmless_means[layer]).to(torch.float32)
        magnitude = r.norm().item()
        r_hat = r / magnitude

        unnormalized_directions[layer] = r
        normalized_directions[layer] = r_hat

        metadata["layers"][str(layer)] = {
            "separation": round(magnitude, 4),
            "r_norm": round(magnitude, 4),
        }
        if magnitude > 1000:
            print(f"  L{layer:>2}: separation={magnitude:>10.1f}")

    # Compute pairwise cosine similarities between key layers
    key_layers = [l for l in [15, 18, 25, 32] if l in layers]
    cosine_sims = {}
    for i, l1 in enumerate(key_layers):
        for l2 in key_layers[i + 1:]:
            cos = torch.nn.functional.cosine_similarity(
                normalized_directions[l1].unsqueeze(0),
                normalized_directions[l2].unsqueeze(0),
            ).item()
            cosine_sims[f"L{l1}_L{l2}"] = round(cos, 4)
            print(f"  cos(L{l1}, L{l2}) = {cos:.4f}")

    metadata["cosine_similarities"] = cosine_sims
    metadata["cosine_matrix"] = compute_full_cosine_matrix(normalized_directions)

    # Find best separation and best causal layers
    best_sep_layer = max(layers, key=lambda l: metadata["layers"][str(l)]["separation"])
    metadata["best_separation_layer"] = best_sep_layer
    metadata["best_causal_layer"] = config.BEST_CAUSAL_LAYER

    # Save normalized directions (using refusal_lens library format)
    # Also save in a single .pt for easy loading
    print(f"\nSaving to {out_dir}/...")

    # Per-layer .pt files (normalized)
    directions_dir = out_dir / "directions"
    directions_dir.mkdir(exist_ok=True)
    for layer, r_hat in sorted(normalized_directions.items()):
        torch.save(r_hat, directions_dir / f"layer_{layer:02d}.pt")

    # Unnormalized directions (single file, for causal intervention)
    torch.save(unnormalized_directions, unnorm_path)

    # Legacy-compatible single-file format (for backward compat with existing scripts)
    legacy = {
        "best_direction": normalized_directions[best_sep_layer],
        "best_layer": best_sep_layer,
        "best_position": config.DIRECTION_POSITION,
        "separation": metadata["layers"][str(best_sep_layer)]["separation"],
    }
    # Also include per-layer normalized directions keyed like Tejas's format
    for layer in layers:
        legacy[f"direction_pos-2_layer{layer}"] = normalized_directions[layer]
    torch.save(legacy, out_dir / "refusal_direction.pt")

    # Metadata
    save_json(metadata, out_dir / "direction_metadata.json")

    # Save run config snapshot
    save_json({
        "stage": "01_compute_direction",
        "model": config.MODEL_NAME,
        "n_samples": args.n_samples,
        "n_layers": len(layers),
        "best_separation_layer": best_sep_layer,
        "best_separation": metadata["layers"][str(best_sep_layer)]["separation"],
    }, run_dir / "config.json")

    print(f"\n  Best separation: L{best_sep_layer} = {metadata['layers'][str(best_sep_layer)]['separation']:.1f}")
    print(f"  Best causal: L{config.BEST_CAUSAL_LAYER} = {metadata['layers'][str(config.BEST_CAUSAL_LAYER)]['separation']:.1f}")
    print(f"  Run directory: {run_dir}")
    print("DONE!")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()