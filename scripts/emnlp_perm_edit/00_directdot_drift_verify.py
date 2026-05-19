"""Phase 0 — direct_dot drift verification (Task 7 sanity check).

For a small subset of inputs (5 prompts × all conditions), runs unedited and
hook-edited forward passes and measures the actual direct_dot drift at
L15 pos=-2. Verifies the hook achieves the predicted delta within numerical
tolerance.

If drift is wildly off the prediction, the hook math, the linearization
decomposition (0a), or the model loading has an issue to debug before
trusting 0b/0d/0e flip rates.

Runs on GPU (loads Gemma-3-4B-IT). ~3 min wall on H100.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from edge_ablation_hook import make_scalar_rhat_subtraction_hook  # noqa: E402
from utils import format_prompt, load_controlled_dataset  # noqa: E402


LAYER = 15
MEASUREMENT_POSITION = -2

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
                   default=REPO / "data/results/emnlp_perm_edit/phase0_controllability/directdot_drift_audit.json")
    p.add_argument("--n-prompts", type=int, default=5)
    p.add_argument("--variants", default="ablate_features_all,ablate_embeddings_all,ablate_all_edges,ablate_all_2x")
    p.add_argument("--tolerance", type=float, default=50.0,
                   help="Allowed |actual_drift - predicted_drift| in direct_dot units.")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"],
                   help="Model dtype. Default float32 to avoid bf16 precision loss on small "
                        "per-element hook deltas (per analysis 2026-05-19: bf16 ulp at typical "
                        "residual values ~0.8 vs hook per-element change ~5e-5; bf16 loses 95%% "
                        "of the intervention). Use bfloat16 only for speed validation.")
    return p.parse_args()


def capture_direct_dot_at_target(model, tokenizer, text: str, r_hat: torch.Tensor,
                                 target_layer, hook_fn=None) -> float:
    """Run forward pass and return h[L15, pos=-2] . r_hat as a Python float."""
    formatted = format_prompt(tokenizer, text)
    ids = tokenizer(formatted, return_tensors="pt").to(model.device)
    captured = {}

    def capture_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[:, MEASUREMENT_POSITION, :].detach().float().cpu()

    handles = []
    if hook_fn is not None:
        handles.append(target_layer.register_forward_hook(hook_fn))
    handles.append(target_layer.register_forward_hook(capture_hook))

    try:
        with torch.no_grad():
            model(**ids)
    finally:
        for h in handles:
            h.remove()

    h_vec = captured["h"][0]
    return (h_vec @ r_hat).item()


def main():
    args = parse_args()
    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants_to_run:
        assert v in VARIANT_TO_DELTA_FIELD, f"unknown variant: {v}"

    print(f"[drift] loading r_hat[L{LAYER}]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float()

    print(f"[drift] loading decomposition from {args.decomposition}")
    decomp = json.loads(args.decomposition.read_text())
    per_prompt = {(r["prompt_idx"], r["condition"]): r for r in decomp["per_prompt"]}

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"[drift] loading model {args.model} in {args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="cuda")
    model.eval()
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    target_layer = layers[LAYER]

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    dataset = dataset[:args.n_prompts]

    audit = {
        "metadata": {"n_prompts": args.n_prompts, "tolerance": args.tolerance,
                     "variants": variants_to_run},
        "per_check": [],
    }
    n_passing = 0
    n_failing = 0

    for prompt_idx, prompt in enumerate(dataset):
        for cond, blob in prompt["conditions"].items():
            decomp_rec = per_prompt.get((prompt_idx, cond))
            if decomp_rec is None:
                continue
            text = blob["text"]
            dd_unedited = capture_direct_dot_at_target(
                model, tokenizer, text, r_hat,
                target_layer=target_layer, hook_fn=None)

            for variant in variants_to_run:
                delta_field, scale = VARIANT_TO_DELTA_FIELD[variant]
                delta = float(decomp_rec[delta_field]) * scale
                hook_fn = make_scalar_rhat_subtraction_hook(r_hat, delta)
                dd_edited = capture_direct_dot_at_target(
                    model, tokenizer, text, r_hat,
                    target_layer=target_layer, hook_fn=hook_fn)
                actual_drift = dd_unedited - dd_edited
                err = abs(actual_drift - delta)
                passing = err <= args.tolerance
                if passing:
                    n_passing += 1
                else:
                    n_failing += 1
                audit["per_check"].append({
                    "prompt_idx": prompt_idx, "condition": cond, "variant": variant,
                    "delta_predicted": delta,
                    "drift_measured": actual_drift,
                    "abs_error": err,
                    "passing": passing,
                })

    audit["summary"] = {
        "n_passing": n_passing, "n_failing": n_failing,
        "pass_rate": n_passing / max(n_passing + n_failing, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2))
    print(f"[drift] {n_passing}/{n_passing+n_failing} checks pass within tolerance {args.tolerance}")
    print(f"[drift] wrote {args.out}")
    if n_failing > 0:
        print(f"[drift] WARNING: {n_failing} drift checks failed. Inspect directdot_drift_audit.json.")
        print(f"  Likely causes: hook math error, decomposition staleness, dtype precision, or model load issue.")


if __name__ == "__main__":
    main()
