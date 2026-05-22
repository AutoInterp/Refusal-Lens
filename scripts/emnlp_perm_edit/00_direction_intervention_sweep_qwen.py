"""Phase 0 extension on Qwen3-4B — direction-intervention coefficient sweep.

Mirror of 00_direction_intervention_sweep.py but adapted for Qwen3-4B:
  - r_hat loaded from Ruqiya's per-layer files (layer_XX.pt format) instead of
    a single dict file (unnormalized_r.pt).
  - Default position convention pos=-1 (Ruqiya's direction was constructed at
    pos=-1; Gemma's at pos=-2).
  - Default probe layer L18 (Ruqiya's Stage 06 lever; 92.5% bare flip at
    coeff=1.0). L34 is peak-separation but unused as primary by Ruqiya.
  - Model arch: Qwen3 uses model.model.layers[i] (no .language_model wrapper).
    The hook's existing fallback handles this automatically.

Tests whether the magnitude-gap finding from Gemma-3-4B (Batch 14) generalizes:
  - Dose-response curve at L18 all-positions: expect 100% flip at coeff=1.0,
    inflection at coeff ≈ ?
  - Dose-response at L18 pos=-1 only: expect weaker, but supra-baseline
  - Layer locator at coeff=1.0 pos=-1: expect band L?-L34 with peak near L34

Output JSON structure matches Gemma driver: per_layer[L]/per_coefficient[coeff].
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from edge_ablation_hook import make_scalar_rhat_subtraction_hook  # noqa: E402
from utils import classify_response, format_prompt, is_coherent, load_controlled_dataset  # noqa: E402


DEFAULT_COEFFS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
CONDITIONS = [
    "bare",
    "jb_fiction", "jb_roleplay", "jb_analytical", "jb_completion", "jb_cognitive_reframe",
    "ctrl_fiction", "ctrl_roleplay", "ctrl_analytical", "ctrl_completion", "ctrl_cognitive_reframe",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--directions-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs_qwen/run_20260502_154423/01_direction/directions",
                   help="Directory containing layer_XX.pt files (Ruqiya's Qwen format).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--layers", default="18",
                   help="Comma-separated layer indices. Default 18 (Ruqiya's Stage 06 lever).")
    p.add_argument("--coefficients", default=",".join(str(c) for c in DEFAULT_COEFFS))
    p.add_argument("--position-mode", required=True, choices=["all", "last_prompt_only"])
    p.add_argument("--target-position", type=int, default=-1,
                   help="Position to edit in last_prompt_only mode. Default -1 for Qwen.")
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    return p.parse_args()


def load_qwen_directions(directions_dir: Path, layers: list[int]) -> dict[int, torch.Tensor]:
    """Load Ruqiya's per-layer Qwen direction files.

    Each file is a tensor (presumably saved as `torch.save(r_layer, "layer_XX.pt")`).
    Falls back to wrapped dict format if the file contains one.
    """
    out = {}
    for L in layers:
        path = directions_dir / f"layer_{L:02d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Direction file not found: {path}")
        obj = torch.load(path, weights_only=False, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            out[L] = obj.float()
        elif isinstance(obj, dict):
            # In case Ruqiya saved as {"direction": tensor, "metadata": ...}
            if "direction" in obj:
                out[L] = obj["direction"].float()
            elif L in obj:
                out[L] = obj[L].float()
            else:
                raise ValueError(f"Unexpected dict structure in {path}: keys={list(obj.keys())}")
        else:
            raise ValueError(f"Unexpected type in {path}: {type(obj)}")
    return out


def main():
    args = parse_args()
    layers_to_run = [int(L.strip()) for L in args.layers.split(",")]
    coefficients = [float(c.strip()) for c in args.coefficients.split(",")]

    print(f"[qwen_sweep] position_mode={args.position_mode} target_position={args.target_position}")
    print(f"[qwen_sweep] layers={layers_to_run}")
    print(f"[qwen_sweep] coefficients={coefficients}")

    # Disable TF32 for true fp32 precision (matches Gemma protocol)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"[qwen_sweep] loading r_hat for layers {layers_to_run} from {args.directions_dir}")
    r_per_layer = load_qwen_directions(args.directions_dir, layers_to_run)
    for L in layers_to_run:
        print(f"  ||r_hat[L{L}]|| = {r_per_layer[L].norm().item():.4f}  shape={tuple(r_per_layer[L].shape)}")

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"[qwen_sweep] loading model {args.model} in {args.dtype}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="cuda",
    )
    model.eval()
    # Qwen path: model.model.layers (no .language_model wrapper)
    if hasattr(model.model, "language_model"):
        all_layers = model.model.language_model.layers
    else:
        all_layers = model.model.layers
    print(f"  loaded in {time.time()-t0:.1f}s, n_layers={len(all_layers)}")

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
    pad_id = tokenizer.eos_token_id

    results = {
        "metadata": {
            "layers": layers_to_run, "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "coefficients": coefficients,
            "position_mode": args.position_mode,
            "target_position": args.target_position,
            "n_prompts": len(dataset),
            "r_hat_norms": {f"L{L}": r_per_layer[L].norm().item() for L in layers_to_run},
            "dtype": args.dtype,
            "method": f"qwen_direction_subtract_coeff_times_r_at_{args.position_mode}",
        },
        "per_layer": {f"L{L}": {"per_coefficient": {}} for L in layers_to_run},
    }

    t_total = time.time()
    n_total_cells = len(layers_to_run) * len(coefficients)
    cell_idx = 0
    for layer in layers_to_run:
        r_hat = r_per_layer[layer]
        r_hat_norm_sq = (r_hat @ r_hat).item()
        target_layer = all_layers[layer]
        layer_key = f"L{layer}"

        for coeff in coefficients:
            cell_idx += 1
            coeff_key = f"coeff_{coeff}"
            delta = coeff * r_hat_norm_sq
            hook_fn = make_scalar_rhat_subtraction_hook(
                r_hat, delta,
                position_mode=args.position_mode,
                target_position=args.target_position,
            )
            print(f"\n[qwen_sweep] [{cell_idx}/{n_total_cells}] layer={layer} coeff={coeff} "
                  f"(delta={delta:.4f}, per-element edit ~ {coeff * r_hat.abs().mean().item():.5f})")
            t_v = time.time()

            per_records = []
            n_done = 0
            n_target = len(dataset) * len(CONDITIONS)
            for prompt_idx, prompt in enumerate(dataset):
                for cond, blob in prompt["conditions"].items():
                    text = blob["text"]
                    formatted = format_prompt(tokenizer, text)
                    ids = tokenizer(formatted, return_tensors="pt").to(model.device)
                    prompt_len = ids.input_ids.shape[1]

                    handle = target_layer.register_forward_hook(hook_fn)
                    try:
                        with torch.no_grad():
                            out = model.generate(
                                **ids, do_sample=False,
                                max_new_tokens=args.max_new_tokens,
                                pad_token_id=pad_id,
                            )
                        resp = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
                    finally:
                        handle.remove()

                    per_records.append({
                        "prompt_idx": prompt_idx, "condition": cond,
                        "layer": layer, "coefficient": coeff, "delta_applied": delta,
                        "response": resp[:300],
                        "classification": classify_response(resp),
                        "coherent": is_coherent(resp),
                    })
                    n_done += 1
                    if n_done % 50 == 0:
                        elapsed = time.time() - t_v
                        eta = elapsed / n_done * (n_target - n_done)
                        print(f"  [{n_done}/{n_target}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

            results["per_layer"][layer_key]["per_coefficient"][coeff_key] = per_records
            elapsed = time.time() - t_v
            print(f"  L{layer} coeff={coeff} done in {elapsed:.0f}s ({len(per_records)} gens)")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2))

    elapsed_total = time.time() - t_total
    print(f"\n[qwen_sweep] complete in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
