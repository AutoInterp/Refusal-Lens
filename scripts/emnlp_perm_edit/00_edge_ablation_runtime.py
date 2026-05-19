"""Phase 0 — Sub-experiment 0b-simple: runtime edge-ablation across 7 variants.

For each (prompt, condition), looks up the precomputed edge-type sums from 0a's
linearization_decomposition.json and registers a forward hook on L15 that
subtracts the chosen scalar * r_hat_unit from the residual at every position.
Generates max_new_tokens=80 greedy and classifies refuse/comply.

This is a behavioral-proxy test of edge ablation -- it operates on the residual
stream directly, NOT through circuit-tracer's transcoder graph. Its purpose is
to test whether the linearization decomposition's predictions are causally
meaningful: when we remove the predicted amount of r_hat-projection
contribution, does the model flip refusal as expected?

The "rigorous" version (true transcoder edge ablation via vendor/circuit-tracer
patches) is deferred per spec § 2.3.

Tests H0-1 (controllability completeness), H0-2 (signed-attribution
correctness), H0-3 (error-node prominence), H0-4 (edge != node) behaviorally.
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


LAYER = 15
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
    p.add_argument("--decomposition", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json")
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--out", type=Path,
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/edge_ablation_flip_rates.json")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--variants", default=",".join(VARIANT_TO_DELTA_FIELD.keys()),
                   help="Comma-separated variant names to run.")
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Smoke test: limit to first N prompts.")
    p.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"],
                   help="Model dtype. Default float32 because bf16 loses ~95%% of small "
                        "per-element hook deltas (drift verification confirmed 2026-05-19).")
    return p.parse_args()


def main():
    args = parse_args()
    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants_to_run:
        assert v in VARIANT_TO_DELTA_FIELD, f"unknown variant: {v}"

    print(f"[0b-simple] loading r_hat[L{LAYER}]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float()
    print(f"  ||r_hat|| = {r_hat.norm().item():.2f}")

    print(f"[0b-simple] loading decomposition from {args.decomposition}")
    decomp = json.loads(args.decomposition.read_text())
    per_prompt = {(r["prompt_idx"], r["condition"]): r for r in decomp["per_prompt"]}
    print(f"  {len(per_prompt)} (prompt, condition) entries")

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"[0b-simple] loading model {args.model} in {args.dtype}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="cuda",
    )
    model.eval()
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    target_layer = layers[LAYER]
    print(f"  loaded in {time.time()-t0:.1f}s")

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
        per_prompt = {k: v for k, v in per_prompt.items() if k[0] < args.max_prompts}

    pad_id = tokenizer.eos_token_id

    results = {
        "metadata": {
            "layer": LAYER,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "n_prompts": len(dataset),
            "variants": variants_to_run,
            "r_hat_norm": r_hat.norm().item(),
        },
        "per_variant": {v: [] for v in variants_to_run},
    }

    t_total = time.time()
    for variant in variants_to_run:
        delta_field, scale = VARIANT_TO_DELTA_FIELD[variant]
        print(f"\n[0b-simple] variant={variant} (delta_field={delta_field}, scale={scale})")
        n_done = 0
        n_target = len(dataset) * 11
        t_v = time.time()
        for prompt_idx, prompt in enumerate(dataset):
            for cond, blob in prompt["conditions"].items():
                decomp_rec = per_prompt.get((prompt_idx, cond))
                if decomp_rec is None:
                    continue
                delta = float(decomp_rec[delta_field]) * scale
                hook_fn = make_scalar_rhat_subtraction_hook(r_hat, delta)

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

                results["per_variant"][variant].append({
                    "prompt_idx": prompt_idx, "condition": cond,
                    "delta_applied": delta,
                    "response": resp[:300],
                    "classification": classify_response(resp),
                    "coherent": is_coherent(resp),
                })
                n_done += 1
                if n_done % 50 == 0:
                    elapsed = time.time() - t_v
                    eta = elapsed / n_done * (n_target - n_done)
                    print(f"  [{n_done}/{n_target}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

                # Save incremental progress every 100 generations to recover from OOMs / crashes
                if n_done % 100 == 0:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(results, indent=2))
        print(f"  variant={variant} done in {time.time()-t_v:.0f}s")
        # Save after each variant completes (full variant block stable)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))

    elapsed_total = time.time() - t_total
    print(f"\n[0b-simple] all variants done in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
