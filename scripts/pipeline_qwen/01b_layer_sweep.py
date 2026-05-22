"""
Stage 01b: Causal-layer sweep for Qwen3-4B
==========================================
Validates BEST_CAUSAL_LAYER by running pro_refusal_add intervention
(Arditi-style h += r) at every Qwen3 layer (0..35) and measuring per-layer
flip rate on a small set of (prompt, jb_class) pairs whose baseline COMPLIES.

The layer that best flips COMPLY → REFUSE is the empirical causal layer.
Update `pipeline_qwen/config.py::BEST_CAUSAL_LAYER` accordingly before
running Stage 06.

Inputs:
    --run-dir <run>            Pipeline run dir containing 01_direction/unnormalized_r.pt
    --dataset <path>           controlled dataset (default: dataset/refusal_lens_controlled_dataset.json)
    --n-prompts <int>          Number of prompts to use (default: 8)
    --layers <list>            Specific layers to sweep (default: all 0..N_LAYERS-1)
    --jb-classes <list>        JB classes to evaluate (default: all 5)
    --max-new-tokens <int>     Generation budget per call (default: 60 for sweep speed)

Outputs to <run-dir>/01b_layer_sweep/:
    layer_sweep_results.json   Per-(layer, prompt, class) records + summary
    flip_rate_by_layer.png     Bar chart, one bar per layer
    LAYER_SWEEP_SUMMARY.md     Human-readable headline + recommended BEST_CAUSAL_LAYER

Usage:
    PYTHONPATH=src python3 scripts/pipeline_qwen/01b_layer_sweep.py \\
        --run-dir data/results/pipeline_runs_qwen/run_20260502_154423

Smoke (2 prompts, 4 layers):
    python3 scripts/pipeline_qwen/01b_layer_sweep.py \\
        --run-dir <run> --n-prompts 2 --layers 12 18 22 28
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import (
    classify_response,
    generate_baseline,
    generate_with_hook,
    get_stage_dir,
    is_coherent,
    load_controlled_dataset,
    load_unnormalized_r,
    make_intervention_hook,
    save_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 01b: causal-layer sweep")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run dir with 01_direction/unnormalized_r.pt")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Controlled dataset JSON path")
    p.add_argument("--n-prompts", type=int, default=8)
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Specific layers (default: all 0..N_LAYERS-1)")
    p.add_argument("--jb-classes", type=str, nargs="+", default=None,
                   help="JB classes to evaluate (default: all)")
    p.add_argument("--max-new-tokens", type=int, default=60,
                   help="Lower than full Stage 06 (200) for sweep throughput")
    return p.parse_args()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    out_dir = get_stage_dir(args.run_dir, "01b_layer_sweep")

    layers = args.layers or list(range(config.N_LAYERS))
    jb_classes = args.jb_classes or list(config.JB_CLASSES.keys())

    print("=" * 60)
    print("STAGE 01b: Qwen3 Causal-Layer Sweep")
    print("=" * 60)
    print(f"  layers   : {layers}")
    print(f"  classes  : {jb_classes}")
    print(f"  prompts  : {args.n_prompts}")
    print(f"  max_new  : {args.max_new_tokens}")
    print(f"  out_dir  : {out_dir}")

    rows = load_controlled_dataset(args.dataset, n_prompts=args.n_prompts)

    print("\nLoading per-layer unnormalized r...")
    r_by_layer = load_unnormalized_r(args.run_dir / "01_direction", layers)

    print(f"Loading {config.MODEL_NAME} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    # Phase 0: baselines for every (prompt, jb_class) — keep only COMPLY pairs
    print("\n[PHASE 0] Baselines...")
    baselines: dict[tuple[int, str], dict] = {}
    comply_pairs: list[tuple[int, str]] = []
    t0 = time.time()
    for r_idx, row in enumerate(rows):
        for cls in jb_classes:
            jb_text = row["conditions"][f"jb_{cls}"]["text"]
            resp = generate_baseline(model, tokenizer, jb_text, args.max_new_tokens)
            cls_label = classify_response(resp)
            baselines[(row["id"], cls)] = {
                "cls": cls_label, "coherent": is_coherent(resp),
                "response": resp[:300],
            }
            if cls_label == "COMPLY":
                comply_pairs.append((row["id"], cls))
        print(f"  prompt {r_idx+1}/{len(rows)} done "
              f"(comply_so_far={len(comply_pairs)})")
    print(f"\n  Baseline COMPLY pairs: {len(comply_pairs)}/{len(rows)*len(jb_classes)} "
          f"in {time.time()-t0:.0f}s")

    if not comply_pairs:
        print("\n  ❌ No baseline-COMPLY pairs found. Cannot measure flip rate.")
        print("  Either Qwen refuses all JB framings (good for safety, "
              "bad for sweep) or dataset is too small.")
        save_json({
            "n_prompts": len(rows), "jb_classes": jb_classes,
            "baselines": [{"prompt_id": pid, "class": cls, **b}
                          for (pid, cls), b in baselines.items()],
            "comply_pairs": [], "per_layer": {},
        }, out_dir / "layer_sweep_results.json")
        return

    # Phase 1: per-layer pro_refusal_add on each COMPLY pair
    print(f"\n[PHASE 1] Sweep {len(layers)} layers × {len(comply_pairs)} pairs "
          f"= {len(layers)*len(comply_pairs)} interventions...")
    per_layer: dict[str, dict] = {}
    pair_to_text = {(row["id"], cls): row["conditions"][f"jb_{cls}"]["text"]
                    for row in rows for cls in jb_classes}

    for li, layer in enumerate(layers):
        r_vec = r_by_layer[layer].to(model.device)
        add_hook = make_intervention_hook(r_vec, sign="add")
        per_pair = []
        n_flip = 0
        n_coherent_flip = 0
        per_class_flip = {c: 0 for c in jb_classes}
        per_class_total = {c: 0 for c in jb_classes}

        t_layer = time.time()
        for pid, cls in comply_pairs:
            resp = generate_with_hook(
                model, tokenizer, pair_to_text[(pid, cls)],
                layer, add_hook, args.max_new_tokens,
            )
            cls_label = classify_response(resp)
            coherent = is_coherent(resp)
            flipped = cls_label == "REFUSE"
            per_pair.append({
                "prompt_id": pid, "class": cls, "cls": cls_label,
                "coherent": coherent, "flipped": flipped, "response": resp[:300],
            })
            per_class_total[cls] += 1
            if flipped:
                n_flip += 1
                per_class_flip[cls] += 1
                if coherent:
                    n_coherent_flip += 1

        flip_rate = n_flip / len(comply_pairs)
        per_layer[str(layer)] = {
            "n_pairs": len(comply_pairs),
            "n_flipped": n_flip,
            "n_coherent_flipped": n_coherent_flip,
            "flip_rate": round(flip_rate, 3),
            "coherent_flip_rate": round(n_coherent_flip / len(comply_pairs), 3),
            "per_class": {
                c: {"flipped": per_class_flip[c], "total": per_class_total[c],
                    "rate": round(per_class_flip[c] / per_class_total[c], 3)
                          if per_class_total[c] else 0.0}
                for c in jb_classes
            },
            "per_pair": per_pair,
        }
        print(f"  L{layer:>2} ({li+1}/{len(layers)}): "
              f"flip {n_flip}/{len(comply_pairs)} = {flip_rate:.0%}  "
              f"(coherent: {n_coherent_flip}/{len(comply_pairs)}) "
              f"[{time.time()-t_layer:.0f}s]")

        del r_vec
        gc.collect()
        torch.cuda.empty_cache()

    # Pick best causal layer (max coherent flip rate, ties → lower layer = simpler)
    best_layer = max(layers, key=lambda l: (
        per_layer[str(l)]["coherent_flip_rate"],
        per_layer[str(l)]["flip_rate"],
        -l,
    ))
    best_rate = per_layer[str(best_layer)]["coherent_flip_rate"]

    # Save full results
    results = {
        "model": config.MODEL_NAME,
        "n_layers_swept": len(layers),
        "layers_swept": layers,
        "n_prompts": len(rows),
        "jb_classes": jb_classes,
        "max_new_tokens": args.max_new_tokens,
        "n_baseline_comply_pairs": len(comply_pairs),
        "best_causal_layer": best_layer,
        "best_coherent_flip_rate": best_rate,
        "current_config_BEST_CAUSAL_LAYER": config.BEST_CAUSAL_LAYER,
        "comply_pairs": [{"prompt_id": pid, "class": cls} for pid, cls in comply_pairs],
        "baselines": [{"prompt_id": pid, "class": cls, **b}
                      for (pid, cls), b in baselines.items()],
        "per_layer": per_layer,
    }
    save_json(results, out_dir / "layer_sweep_results.json")
    print(f"\n  Saved layer_sweep_results.json")

    # Plot flip rate by layer
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        flip_rates = [per_layer[str(l)]["flip_rate"] for l in layers]
        coh_rates = [per_layer[str(l)]["coherent_flip_rate"] for l in layers]
        fig, ax = plt.subplots(figsize=(14, 5.5))
        x = list(range(len(layers)))
        width = 0.4
        ax.bar([i - width / 2 for i in x], [r * 100 for r in flip_rates],
               width, label="any flip", color="#2e7d32", alpha=0.7)
        ax.bar([i + width / 2 for i in x], [r * 100 for r in coh_rates],
               width, label="coherent flip", color="#1976d2")
        ax.axvline(layers.index(best_layer), color="red", linestyle="--", alpha=0.5,
                   label=f"best L{best_layer}")
        if config.BEST_CAUSAL_LAYER in layers:
            ax.axvline(layers.index(config.BEST_CAUSAL_LAYER), color="orange",
                       linestyle=":", alpha=0.5,
                       label=f"config L{config.BEST_CAUSAL_LAYER}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers], rotation=60, fontsize=8)
        ax.set_ylabel("Flip rate (% of COMPLY → REFUSE)")
        ax.set_xlabel(f"Layer (n_baseline_comply_pairs = {len(comply_pairs)})")
        ax.set_title(f"Qwen3-4B causal-layer sweep — best L{best_layer} "
                     f"({best_rate:.0%} coherent flip)")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "flip_rate_by_layer.png", dpi=140)
        plt.close()
        print(f"  Saved flip_rate_by_layer.png")
    except Exception as e:
        print(f"  Plot failed (non-fatal): {e}")

    # Headline summary
    sorted_layers = sorted(layers, key=lambda l: -per_layer[str(l)]["coherent_flip_rate"])
    top5 = sorted_layers[:5]
    summary_lines = [
        f"# Qwen3-4B Causal-Layer Sweep — Headline\n",
        f"**Best causal layer (by coherent flip rate): L{best_layer} = {best_rate:.0%}**\n",
        f"- Currently in `pipeline_qwen/config.py`: `BEST_CAUSAL_LAYER = "
        f"{config.BEST_CAUSAL_LAYER}` "
        f"({per_layer[str(config.BEST_CAUSAL_LAYER)]['coherent_flip_rate']:.0%} "
        f"coherent flip)" if config.BEST_CAUSAL_LAYER in layers else
        f"- `BEST_CAUSAL_LAYER={config.BEST_CAUSAL_LAYER}` was not in the swept layer set",
        f"- {'✓ Update required: ' if best_layer != config.BEST_CAUSAL_LAYER else '✓ Current value matches sweep — no update needed.'}"
        f"{'set BEST_CAUSAL_LAYER = ' + str(best_layer) if best_layer != config.BEST_CAUSAL_LAYER else ''}",
        "",
        f"## Top 5 layers (coherent flip rate)\n",
        "| Layer | Coherent flip | Any flip | Per-class breakdown |",
        "|---|---|---|---|",
    ]
    for l in top5:
        pl = per_layer[str(l)]
        per_cls = ", ".join(f"{c}={pl['per_class'][c]['rate']:.0%}"
                            for c in jb_classes)
        summary_lines.append(
            f"| L{l} | {pl['coherent_flip_rate']:.0%} | {pl['flip_rate']:.0%} "
            f"| {per_cls} |"
        )
    summary_lines.extend([
        "",
        f"## Setup\n",
        f"- model: `{config.MODEL_NAME}`",
        f"- n_prompts: {len(rows)} × {len(jb_classes)} jb_classes "
        f"= {len(rows)*len(jb_classes)} (prompt, class) pairs evaluated as baseline",
        f"- n_baseline_comply_pairs: {len(comply_pairs)} "
        f"({len(comply_pairs)/(len(rows)*len(jb_classes)):.0%}) — "
        f"only pairs where baseline COMPLY are eligible substrate",
        f"- max_new_tokens: {args.max_new_tokens}",
        f"- intervention: `h[:,:,:] += r_unnormalized` at every position, "
        f"`r` from Stage 01 unnormalized_r.pt at the same layer",
        "",
        "## Notes\n",
        "- Coherent flip rate is the headline metric — gibberish flips don't count.",
        f"- Tejas reports 90/90 = 100% on Gemma-3 at L15 with full dataset. "
        f"Below ~70% on Qwen here may indicate (a) dataset too small, "
        f"(b) max_new_tokens too low, or (c) Qwen's refusal axis is "
        f"genuinely weaker than Gemma's at any single layer.",
    ])
    summary_path = out_dir / "LAYER_SWEEP_SUMMARY.md"
    summary_path.write_text("\n".join(summary_lines))
    print(f"  Saved LAYER_SWEEP_SUMMARY.md\n")
    print(f"DONE!  Best causal layer: L{best_layer} ({best_rate:.0%} coherent flip)")


if __name__ == "__main__":
    main()
