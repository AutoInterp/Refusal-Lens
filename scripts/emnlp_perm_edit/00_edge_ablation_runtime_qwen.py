"""Phase 0 — Sub-experiment 0b-simple on Qwen3-4B at L18.

Mirror of 00_edge_ablation_runtime.py but adapted for Qwen3-4B:
  - LAYER = 18 (Ruqiya's causal layer, validated via Stage 06: 92.5% bare flip)
  - r_hat loaded from Ruqiya's per-layer .pt file (layer_18.pt) instead of dict
  - Decomposition path = Qwen's 0a output (computed from L18-target graphs)

The hook factory (make_scalar_rhat_subtraction_hook) and edge-derived deltas
work identically — only the target layer, r_hat source path, and model
arch path differ. Model layer access uses the no-`language_model`-wrapper
fallback in the existing code (works for Qwen3 out of the box).
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


LAYER = 18
VARIANT_TO_DELTA_FIELD = {
    "ablate_features_pos": ("feature_pos", 1.0),
    "ablate_features_neg": ("feature_neg", 1.0),
    "ablate_features_all": ("feature_signed", 1.0),
    "ablate_embeddings_all": ("embedding_signed", 1.0),
    "ablate_errors_all": ("error_signed", 1.0),
    "ablate_all_edges": ("all_signed", 1.0),
    "ablate_all_2x": ("all_signed", 2.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decomposition", type=Path, required=True,
                   help="Qwen 0a linearization_decomposition.json (computed from L18-target graphs). "
                        "Deltas are in NORMALIZED-r-projection units (since Stage 02 used the "
                        "normalized direction). We convert to unnormalized basis at hook time.")
    p.add_argument("--rhat-path", type=Path,
                   default=REPO / "data/results/pipeline_runs_qwen/run_20260502_154423/01_direction/positions_L18/pos_-1_unnormalized.pt",
                   help="Path to Qwen UNNORMALIZED r_hat[L18, pos=-1] (||r||≈15.14). Matches "
                        "Ruqiya's Stage 06 Arditi convention so coeff=1.0 in our sweep is "
                        "equivalent to her anti-refuse-sub at coeff=1.0.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--variants", default=",".join(VARIANT_TO_DELTA_FIELD.keys()),
                   help="Comma-separated variant names to run.")
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    p.add_argument("--position-mode", default="all", choices=["all", "last_prompt_only"])
    p.add_argument("--target-position", type=int, default=-1,
                   help="Position for last_prompt_only mode. Default -1 for Qwen.")
    return p.parse_args()


def main():
    args = parse_args()
    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants_to_run:
        assert v in VARIANT_TO_DELTA_FIELD, f"unknown variant: {v}"

    # Disable TF32 for true fp32 precision (consistent with Gemma protocol)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"[qwen-0b] loading UNNORMALIZED r_hat[L{LAYER}] from {args.rhat_path}")
    r_hat_obj = torch.load(args.rhat_path, weights_only=False, map_location="cpu")
    if isinstance(r_hat_obj, torch.Tensor):
        r_hat = r_hat_obj.float()
    elif isinstance(r_hat_obj, dict) and "direction" in r_hat_obj:
        r_hat = r_hat_obj["direction"].float()
    else:
        raise ValueError(f"Unexpected r_hat format in {args.rhat_path}: {type(r_hat_obj)}")
    r_hat_norm = r_hat.norm().item()
    print(f"  ||r_hat_unnorm|| = {r_hat_norm:.4f}  shape={tuple(r_hat.shape)}")
    if r_hat_norm < 5.0:
        print(f"  WARNING: expected ||r_hat|| ≈ 15.14 for Qwen L18 unnormalized. "
              f"Got {r_hat_norm:.4f}. Did you accidentally pass the normalized file?")

    print(f"[qwen-0b] loading decomposition from {args.decomposition}")
    decomp = json.loads(args.decomposition.read_text())
    per_prompt = {(r["prompt_idx"], r["condition"]): r for r in decomp["per_prompt"]}
    print(f"  {len(per_prompt)} (prompt, condition) entries")

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"[qwen-0b] loading model {args.model} in {args.dtype}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="cuda",
    )
    model.eval()
    # Qwen3: model.model.layers; the fallback in the hook handles this automatically
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    target_layer = layers[LAYER]
    print(f"  loaded in {time.time()-t0:.1f}s, n_layers={len(layers)}")

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
        per_prompt = {k: v for k, v in per_prompt.items() if k[0] < args.max_prompts}

    pad_id = tokenizer.eos_token_id

    results = {
        "metadata": {
            "layer": LAYER, "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "n_prompts": len(dataset),
            "variants": variants_to_run,
            "r_hat_norm": r_hat.norm().item(),
            "position_mode": args.position_mode,
            "target_position": args.target_position,
        },
        "per_variant": {v: [] for v in variants_to_run},
    }

    t_total = time.time()
    for variant in variants_to_run:
        delta_field, scale = VARIANT_TO_DELTA_FIELD[variant]
        print(f"\n[qwen-0b] variant={variant} (delta_field={delta_field}, scale={scale})")
        n_done = 0
        n_target = len(dataset) * 11
        t_v = time.time()
        for prompt_idx, prompt in enumerate(dataset):
            for cond, blob in prompt["conditions"].items():
                decomp_rec = per_prompt.get((prompt_idx, cond))
                if decomp_rec is None:
                    continue
                # Decomposition deltas are in NORMALIZED-r-projection units (Stage 02 used
                # normalized r). To apply equivalent edits with unnormalized r_hat in the
                # hook, we scale: delta_unnorm = ||r_unnorm|| * delta_norm.
                # Hook math: h_new = h - (delta_unnorm/||r_unnorm||²)·r_unnorm
                #         = h - (delta_norm/||r_unnorm||)·r_unnorm
                #         = h - delta_norm·(r_unnorm/||r_unnorm||)·1
                #         = h - delta_norm·r_normalized  (same edit as the normalized basis)
                # This preserves the linearization identity: h_new·r_normalized = h·r_normalized - delta_norm.
                delta_norm = float(decomp_rec[delta_field]) * scale
                delta = delta_norm * r_hat_norm  # convert to unnormalized basis
                hook_fn = make_scalar_rhat_subtraction_hook(
                    r_hat, delta,
                    position_mode=args.position_mode,
                    target_position=args.target_position,
                )

                text = blob["text"]
                # enable_thinking=False is load-bearing for Qwen3-4B (Ruqiya's convention;
                # her directions were constructed in non-thinking mode). See format_prompt
                # docstring for details.
                formatted = format_prompt(tokenizer, text, enable_thinking=False)
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

                results["per_variant"][variant].append({
                    "prompt_idx": prompt_idx, "condition": cond,
                    "delta_applied": delta,            # unnormalized basis
                    "delta_norm_basis": delta_norm,    # for reference / debugging
                    "response": resp[:300],
                    "classification": classify_response(resp),
                    "coherent": is_coherent(resp),
                })
                n_done += 1
                if n_done % 50 == 0:
                    elapsed = time.time() - t_v
                    eta = elapsed / n_done * (n_target - n_done)
                    print(f"  [{n_done}/{n_target}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

                if n_done % 100 == 0:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(results, indent=2))
        print(f"  variant={variant} done in {time.time()-t_v:.0f}s")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))

    elapsed_total = time.time() - t_total
    print(f"\n[qwen-0b] all variants done in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
