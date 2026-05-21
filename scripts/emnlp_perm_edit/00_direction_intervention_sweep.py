"""Phase 0 extension — direction-intervention coefficient sweep.

Subtracts `coefficient * r_hat[layer]` from the residual at the chosen layer(s)
under one of two position-modes:
    - all              : every position, every forward step (Arditi style)
    - last_prompt_only : only seq pos=-2 of the prompt encoding pass

For each (layer, coefficient, prompt, condition), generates max_new_tokens=80
greedy and classifies refuse/comply.

Tests two related questions Georg raised:

  1. Magnitude vs position confound (single layer L15 sweep): does the
     edge-ablation result reflect L15:r_hat's true causal role, or is it just
     under-magnitude / mis-positioned vs the canonical Arditi-style direction
     intervention? Run with --layers 15 + full coefficient list.

  2. Layer locator (single coefficient, multiple layers, pos=-2 only): if
     L15-at-pos=-2 isn't the lever, which layer is? Run with
     --layers 0,3,6,9,12,18,21,24 + --coefficients 1.0 + --position-mode
     last_prompt_only.

The hook reuses make_scalar_rhat_subtraction_hook with delta = coeff * ||r||^2,
so that h_new = h - coeff * r at the targeted positions.

Runs in fp32 by default for consistency with the rest of the Phase 0 GPU
suite. TF32 matmul is explicitly disabled so low-coefficient cells (where
per-element edits are small) aren't biased by TF32's reduced mantissa.

Output structure:
    {
      "metadata": {layers, coefficients, position_mode, ...},
      "per_layer": {
        "L{N}": {
          "per_coefficient": {
            "coeff_{C}": [ {prompt_idx, condition, classification, ...}, ... ]
          }
        }
      }
    }
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
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON path.")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--layers", default="15",
                   help="Comma-separated layer indices. Default 15 (the v1 probe layer).")
    p.add_argument("--coefficients", default=",".join(str(c) for c in DEFAULT_COEFFS),
                   help="Comma-separated coefficients. Each cell subtracts coeff * r_hat[layer].")
    p.add_argument("--position-mode", required=True, choices=["all", "last_prompt_only"],
                   help="'all' = every position every forward step (Arditi). "
                        "'last_prompt_only' = pos=-2 of prompt only.")
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Smoke test: limit to first N prompts.")
    p.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"],
                   help="Model dtype. Default float32 for consistency with the rest of the "
                        "Phase 0 GPU suite (bf16 loses small per-element hook edits).")
    return p.parse_args()


def main():
    args = parse_args()
    layers_to_run = [int(L.strip()) for L in args.layers.split(",")]
    coefficients = [float(c.strip()) for c in args.coefficients.split(",")]

    print(f"[direction_sweep] position_mode={args.position_mode}")
    print(f"[direction_sweep] layers={layers_to_run}")
    print(f"[direction_sweep] coefficients={coefficients}")

    # Disable TF32 so low-coefficient cells aren't silently shaved to ~10-bit precision
    # on Ampere/Hopper/Ada. Matters most when coeff <= 0.01.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"[direction_sweep] loading r_hat for layers {layers_to_run}")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_per_layer = {}
    for L in layers_to_run:
        if L not in r_dict:
            raise KeyError(f"Layer {L} missing from unnormalized_r.pt (have {sorted(r_dict.keys())})")
        r_per_layer[L] = r_dict[L].float()
        print(f"  ||r_hat[L{L}]|| = {r_per_layer[L].norm().item():.2f}")

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"[direction_sweep] loading model {args.model} in {args.dtype}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="cuda",
    )
    model.eval()
    if hasattr(model.model, "language_model"):
        all_layers = model.model.language_model.layers
    else:
        all_layers = model.model.layers
    print(f"  loaded in {time.time()-t0:.1f}s")

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
            "n_prompts": len(dataset),
            "r_hat_norms": {f"L{L}": r_per_layer[L].norm().item() for L in layers_to_run},
            "dtype": args.dtype,
            "method": f"direction_subtract_coeff_times_r_at_{args.position_mode}",
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
                r_hat, delta, position_mode=args.position_mode)
            print(f"\n[direction_sweep] [{cell_idx}/{n_total_cells}] layer={layer} coeff={coeff} "
                  f"(delta={delta:.1f}, per-element edit ~ {coeff * r_hat.abs().mean().item():.3f})")
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
            print(f"  L{layer} coeff={coeff} done in {elapsed:.0f}s ({len(per_records)} gens, "
                  f"~{elapsed/max(len(per_records),1):.1f}s/gen)")
            # Save after each cell so a mid-sweep crash doesn't lose everything
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=2))

    elapsed_total = time.time() - t_total
    print(f"\n[direction_sweep] complete in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
