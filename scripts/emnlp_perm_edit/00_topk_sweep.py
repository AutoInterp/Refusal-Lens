"""Phase 0 — Sub-experiments 0d (top-K features) + 0e (top-K edges) Pareto sweep.

Single driver, two modes:
  --mode features   : ranks ONLY feature edges into the target (0d)
  --mode edges      : ranks ALL edges into the target (features + embeddings + error_nodes) (0e)

Three variants per mode (signed structure of top-K):
  pos-K   : top-K records by signed_attribution descending (most positive first)
  neg-K   : top-K records by signed_attribution ascending (most negative first)
  abs-K   : top-K records by |signed_attribution| descending (mixed signs)

For each (mode, variant, K, prompt, condition), sums the top-K signed
attributions to get delta_K, then registers the 0b-simple runtime hook with
that delta and generates max_new_tokens=80 greedy.

Implements P0 Tasks 11 (0d) + 12 (0e) per the impl plan.
Tests H0-6 (refusal-signal sparsity / Pareto knee) and H0-7 (edge > node Pareto).
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
from graph_loader import (  # noqa: E402
    extract_edge_records_to_target,
    find_measurement_target_node_id,
    load_packed_graph,
)
from utils import classify_response, format_prompt, is_coherent, load_controlled_dataset  # noqa: E402


LAYER = 15
DEFAULT_K_VALUES_FEATURES = [1, 5, 10, 20, 50, 100, 500]
DEFAULT_K_VALUES_EDGES = [1, 5, 10, 50, 100, 500, 1000]
VARIANTS = ("pos", "neg", "abs")
CONDITIONS = [
    "bare",
    "jb_fiction", "jb_roleplay", "jb_analytical", "jb_completion", "jb_cognitive_reframe",
    "ctrl_fiction", "ctrl_roleplay", "ctrl_analytical", "ctrl_completion", "ctrl_cognitive_reframe",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["features", "edges"],
                   help="features: 0d (feature edges only); edges: 0e (all edge types)")
    p.add_argument("--graph-data-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247/05_frontend/graph_data")
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--out", type=Path, default=None,
                   help="Default: 0d -> topk_feature_sweep.json, 0e -> topk_edge_sweep.json")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--k-values", default=None,
                   help="Default: 1,5,10,20,50,100,500 for features; 1,5,10,50,100,500,1000 for edges")
    p.add_argument("--graph-mode", default="single", choices=["single", "multi"])
    p.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"],
                   help="Model dtype. Default float32 because bf16 loses ~95%% of small "
                        "per-element hook deltas (drift verification confirmed 2026-05-19).")
    return p.parse_args()


def compute_delta_for_variant(records: list[dict], variant: str, K: int) -> tuple[float, int]:
    """Sum top-K signed attributions under the given variant. Returns (delta, n_used)."""
    if not records:
        return 0.0, 0
    if variant == "pos":
        chosen = records[:K]
    elif variant == "neg":
        chosen = list(reversed(records))[:K]
    elif variant == "abs":
        chosen = sorted(records, key=lambda r: abs(r["signed_attribution"]), reverse=True)[:K]
    else:
        raise ValueError(f"unknown variant: {variant}")
    return sum(r["signed_attribution"] for r in chosen), len(chosen)


def main():
    args = parse_args()
    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.k_values:
        k_values = [int(k.strip()) for k in args.k_values.split(",")]
    else:
        k_values = DEFAULT_K_VALUES_FEATURES if args.mode == "features" else DEFAULT_K_VALUES_EDGES

    out_path = args.out
    if out_path is None:
        fname = "topk_feature_sweep.json" if args.mode == "features" else "topk_edge_sweep.json"
        out_path = REPO / "data/results/emnlp_perm_edit/phase0_controllability" / fname
    filter_category = "feature" if args.mode == "features" else None
    mode_letter = "d" if args.mode == "features" else "e"

    print(f"[0{mode_letter}] mode={args.mode} K_values={k_values}")
    print(f"[0{mode_letter}] loading r_hat[L{LAYER}]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float()
    print(f"  ||r_hat|| = {r_hat.norm().item():.2f}")

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    print(f"[0{mode_letter}] loading model {args.model} in {args.dtype}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map="cuda")
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
    pad_id = tokenizer.eos_token_id

    print(f"[0{mode_letter}] pre-computing per-(prompt, condition) {args.mode} rankings...")
    rankings = {}
    skipped = []
    for prompt_idx in range(len(dataset)):
        slug_prompt = f"{prompt_idx:03d}"
        for cond in CONDITIONS:
            slug = f"{slug_prompt}_{cond}_{args.graph_mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                skipped.append(slug)
                continue
            graph = load_packed_graph(path)
            target_id = find_measurement_target_node_id(graph)
            records = extract_edge_records_to_target(graph, target_id, filter_category=filter_category)
            rankings[(prompt_idx, cond)] = records
    print(f"  ranked {args.mode} for {len(rankings)} inputs ({len(skipped)} skipped)")

    results = {
        "metadata": {
            "mode": args.mode, "layer": LAYER, "model": args.model,
            "variants": variants_to_run, "k_values": k_values,
            "n_prompts": len(dataset), "graph_mode": args.graph_mode,
            "filter_category": filter_category,
        },
        "per_variant_K": {},
    }

    t_total = time.time()
    n_total_combos = len(variants_to_run) * len(k_values)
    combo_idx = 0
    for variant in variants_to_run:
        for K in k_values:
            combo_idx += 1
            key = f"{variant}_K{K}"
            print(f"\n[0{mode_letter}] [{combo_idx}/{n_total_combos}] variant={variant} K={K}")
            t_v = time.time()
            per_records = []
            for (prompt_idx, cond), records in rankings.items():
                if len(records) == 0:
                    continue
                delta, n_used = compute_delta_for_variant(records, variant, K)
                hook_fn = make_scalar_rhat_subtraction_hook(r_hat, delta)

                text = dataset[prompt_idx]["conditions"][cond]["text"]
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
                    "delta_applied": delta, "n_records_used": n_used,
                    "response": resp[:300],
                    "classification": classify_response(resp),
                    "coherent": is_coherent(resp),
                })
            results["per_variant_K"][key] = per_records
            elapsed = time.time() - t_v
            print(f"  {key} done in {elapsed:.0f}s ({len(per_records)} gens, "
                  f"~{elapsed/max(len(per_records),1):.1f}s/gen)")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(results, indent=2))

    elapsed = time.time() - t_total
    print(f"\n[0{mode_letter}] sweep complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"[0{mode_letter}] wrote {out_path}")


if __name__ == "__main__":
    main()
