"""
Stage 03: Verify Attribution Gap
================================
Validates that CLT attribution sums correspond to the MLP contribution
of the residual-stream dot product with the refusal direction.

Computes:
1. Full dot product <x^(L32, pos-2), r_hat> for each prompt
2. Per-layer decomposition: (resid[L+1] - resid[L]) @ r_hat
3. MLP vs attention breakdown
4. Comparison with saved attribution net values

Inputs:  02_attribution/attribution_results.json, 01_direction/refusal_direction.pt
Outputs: 03_verification/verification_results.json, per_layer_decomposition.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import numpy as np

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import format_prompt, save_json, load_json, get_stage_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Verify attribution gap (M2)")
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to existing pipeline run directory",
    )
    parser.add_argument(
        "--n-decompose", type=int, default=5,
        help="Number of prompts for full per-layer decomposition (slow)",
    )
    parser.add_argument(
        "--graph-mode", choices=["multi", "single"], default="multi",
        help="Which Stage 02 graph mode to verify against. 'multi' targets "
             "L15 at the template positions [-5, -3, -2]; 'single' targets "
             "L15 @ pos=-2 only. Default: multi (the headline graph).",
    )
    parser.add_argument(
        "--target-layer", type=int, default=config.MEASUREMENT_LAYER,
        help="Layer at which attribution is measured (default: 15).",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip model verification; re-aggregate per-layer stats + plot from existing JSON",
    )
    return parser.parse_args()


# -------------------- schema + direction helpers --------------------

def _get_bare_net(row: dict, mode: str) -> float | None:
    """Return the bare condition's `net` under graphs[mode], handling legacy flat."""
    conds = row.get("conditions", row)
    bare = conds.get("bare")
    if not isinstance(bare, dict):
        return None
    if "graphs" in bare:
        g = bare["graphs"].get(mode)
        if not isinstance(g, dict) or "error" in g:
            return None
        return float(g.get("net", 0.0))
    # Legacy flat: only valid for mode="single" (pre-refactor = single-target)
    if mode != "single" or "error" in bare or "net" not in bare:
        return None
    return float(bare["net"])


def _load_mode_directions(
    run_dir: Path, target_layer: int, mode: str,
) -> dict[int, torch.Tensor]:
    """Load the per-position L{target_layer} directions used by Stage 02 for
    the given mode. Returns {position: r_hat tensor}.

    Falls back to ``01_direction/directions/layer_{L}.pt`` at pos=-2 when the
    per-position dir is absent (legacy runs).
    """
    pos_dir = run_dir / "01_direction" / f"positions_L{target_layer:02d}"

    # Pick positions for this mode from config — matches Stage 02 defaults.
    if mode == "multi":
        wanted = sorted(config.TARGET_POSITIONS_MULTI)
    else:
        wanted = list(config.TARGET_POSITIONS_SINGLE)

    out: dict[int, torch.Tensor] = {}
    if pos_dir.exists():
        for pos in wanted:
            fp = pos_dir / f"pos_{pos:+d}.pt"
            if fp.exists():
                t = torch.load(fp, map_location="cpu", weights_only=False)
                if isinstance(t, dict):
                    t = t.get("direction") or next(iter(t.values()))
                out[pos] = t.to(torch.float32)

    if out:
        return out

    # Legacy fallback: per-layer direction at pos=-2.
    layer_pt = run_dir / "01_direction" / "directions" / f"layer_{target_layer:02d}.pt"
    if not layer_pt.exists():
        raise FileNotFoundError(
            f"No per-position directions at {pos_dir} and no layer_{target_layer:02d}.pt fallback."
        )
    t = torch.load(layer_pt, map_location="cpu", weights_only=False)
    if isinstance(t, dict):
        t = t.get("direction") or next(iter(t.values()))
    out[-2] = t.to(torch.float32)
    print(f"  (fallback) loaded single direction at L{target_layer} pos=-2 only")
    return out


def _compute_target_scalar(
    model, tokenizer, r_hats: dict[int, torch.Tensor], prompt: str,
    target_layer: int, positions: list[int],
) -> tuple[float, dict[int, float]]:
    """Compute sum_{p in positions} <r_hats[p], h_{target_layer}[p]> for one prompt.
    Returns (total_scalar, per_position_dict).
    """
    formatted = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    seq_len = int(inputs["attention_mask"].sum().item())
    total_tokens = inputs["input_ids"].shape[1]
    per_pos = {}
    total = 0.0
    for pos in positions:
        if abs(pos) > seq_len:
            continue  # out of range for this prompt
        idx = total_tokens + pos  # pos is negative
        act = out.hidden_states[target_layer + 1][0, idx, :].to(torch.float32)
        r_hat_dev = r_hats[pos].to(model.device)
        dot = float((act @ r_hat_dev).item())
        per_pos[pos] = dot
        total += dot

    del out
    return total, per_pos

def aggregate_and_plot(decomposition_results: list, verification_summary: dict, out_dir: Path) -> dict:
    """A4: aggregate per-layer contributions across prompts and emit bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = config.N_LAYERS
    emb_contribs = np.array([r["embedding_dot"] for r in decomposition_results])

    layer_matrix = np.zeros((len(decomposition_results), n_layers))
    for i, r in enumerate(decomposition_results):
        for lc in r["layer_contributions"]:
            layer_matrix[i, lc["layer"]] = lc["contribution"]

    layer_mean = layer_matrix.mean(axis=0)
    layer_std = layer_matrix.std(axis=0)

    aggregate = {
        "n_prompts_decomposed": len(decomposition_results),
        "embedding": {"mean": float(emb_contribs.mean()), "std": float(emb_contribs.std())},
        "layers": [
            {"layer": i, "mean": float(layer_mean[i]), "std": float(layer_std[i])}
            for i in range(n_layers)
        ],
    }

    # Plot only layers 0..measurement_layer; L33 is post-measurement (RMSNorm artifact)                                         
    ml = verification_summary.get("measurement_layer", 32)                                                                      
    plot_max_layer = ml + 1  # inclusive
    x = np.arange(plot_max_layer)                                                                                               
    mean_in = layer_mean[:plot_max_layer]
    std_in = layer_std[:plot_max_layer]                                                                                         
    colors = ["#5cb85c" if m >= 0 else "#d9534f" for m in mean_in]
                                                                                                                                
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, mean_in, yerr=std_in, color=colors, alpha=0.85,                                                                   
            capsize=2, edgecolor="black", linewidth=0.3)                                                                         
    ax.axvline(ml, color="red", linestyle="--", alpha=0.6,                                                                      
                label=f"Measurement layer (L{ml})")                                                                              
                                                                                                                                
    # Post-measurement layers summary (annotation, not a bar)                                                                   
    post_layers = [l for l in aggregate["layers"] if l["layer"] > ml]                                                           
    post_text = ""                                                                                                              
    if post_layers:                                                                                                             
        lines = [
            f"L{l['layer']}: {l['mean']:+.0f} ± {l['std']:.0f}"                                                                 
            for l in post_layers                                                                                                
        ]
        post_text = "Post-L{0} (RMSNorm artifact, not part of projection):\n".format(ml) + "\n".join(lines)                     
                                                                                                                                
    emb_text = (
        f"Embedding: {aggregate['embedding']['mean']:+.2f} "                                                                    
        f"± {aggregate['embedding']['std']:.2f}"                                                                                
    )
    annot = emb_text + ("\n\n" + post_text if post_text else "")                                                                
    ax.text(                                                                                                                    
        0.99, 0.97, annot,
        transform=ax.transAxes, va="top", ha="right", fontsize=9,                                                               
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),                                                             
    )                                                                                                                           
                                                                                                                                
    ax.set_xlabel("Layer index")
    ax.set_ylabel(f"Mean contribution to r · h[L={ml}]")
    ax.set_title(
        f"Per-Layer Contribution to Refusal-Direction Projection "
        f"(n={len(decomposition_results)} prompts, layers 0–{ml})"
    )                                                                                                                           
    ax.axhline(0, color="black", linewidth=0.5)                                                                                 
    ax.legend(loc="upper left")                                                                                                 
    ax.grid(axis="y", alpha=0.3)                                                                                                
    plt.tight_layout()
    plt.savefig(out_dir / "per_layer_contribution.png", dpi=150)                                                                
    plt.close()

    return aggregate

def main():
    args = parse_args()
    run_dir = args.run_dir
    out_dir = get_stage_dir(run_dir, "03_verification")

    # A4: aggregate-only path — regenerate per-layer aggregate + plot without model
    if args.aggregate_only:
        verif_path = out_dir / "verification_results.json"
        decomp_path = out_dir / "per_layer_decomposition.json"
        if not verif_path.exists() or not decomp_path.exists():
            print(f"ERROR: --aggregate-only needs existing {verif_path.name} and {decomp_path.name}")
            sys.exit(1)
        verif = load_json(verif_path)
        decomp = load_json(decomp_path)
        verif["per_layer_aggregate"] = aggregate_and_plot(decomp, verif["summary"], out_dir)
        save_json(verif, verif_path)
        print(f"Aggregate + plot regenerated in {out_dir}/")
        print("DONE!")
        return
    # ----------------------------------------------------------
    # Load inputs from previous stages
    # ----------------------------------------------------------
    target_layer = args.target_layer
    mode = args.graph_mode
    print(f"Verifying Stage 02 `{mode}` graphs against L{target_layer} direct dot products...")

    print("Loading refusal directions...")
    r_hats = _load_mode_directions(run_dir, target_layer, mode)
    positions = sorted(r_hats.keys())
    print(
        f"  Loaded {len(positions)} direction(s) at L{target_layer}: "
        f"positions={positions}"
    )

    print("Loading attribution results...")
    attr_path = run_dir / "02_attribution" / "attribution_results.json"
    if not attr_path.exists():
        print(f"ERROR: {attr_path} not found. Run Stage 02 first.")
        sys.exit(1)
    raw = load_json(attr_path)
    results_list = raw if isinstance(raw, list) else raw["results"]

    # Extract bare prompts and their saved net attributions for the selected graph mode.
    prompts_and_nets = []
    for entry in results_list:
        prompt = entry["prompt"]
        bare_net = _get_bare_net(entry, mode)
        if bare_net is None:
            continue  # this prompt's bare graph errored or is missing for this mode
        prompts_and_nets.append({"prompt": prompt, "attr_net": bare_net})
    if not prompts_and_nets:
        print(f"ERROR: no valid bare `{mode}` graphs found in {attr_path.name}.")
        sys.exit(1)
    print(f"  Found {len(prompts_and_nets)} prompts with bare `{mode}` attribution")

    # ----------------------------------------------------------
    # Load model (float32 for exact dot products)
    # ----------------------------------------------------------
    print("\nLoading model (float32 for exact computation)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.float32, device_map="auto"
    )
    model.eval()

    # ----------------------------------------------------------
    # CHECK 1: Full dot product vs attribution sum (all prompts).
    #   For mode=single: dot = <r_{L,pos=-2}, h_L[pos=-2]>
    #   For mode=multi:  dot = sum_p <r_{L,pos=p}, h_L[pos=p]>   (matches Stage 02's sum-over-targets)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"CHECK 1: Direct dot product vs attribution sum  —  mode={mode}")
    print("=" * 60)

    verification_results = []
    for i, entry in enumerate(prompts_and_nets):
        prompt = entry["prompt"]
        attr_net = entry["attr_net"]

        total_dot, per_pos_dots = _compute_target_scalar(
            model, tokenizer, r_hats, prompt, target_layer, positions,
        )
        ratio = attr_net / total_dot if total_dot != 0 else float("nan")

        verification_results.append({
            "prompt": prompt[:80],
            "target_positions": sorted(per_pos_dots.keys()),
            "per_position_dot": per_pos_dots,
            "total_dot": total_dot,
            "attr_net": attr_net,
            "difference": total_dot - attr_net,
            "mlp_ratio": ratio,
        })

        print(
            f"  [{i+1:>3}/{len(prompts_and_nets)}] "
            f"dot={total_dot:>10.2f}  attr={attr_net:>8.2f}  "
            f"ratio={ratio:.4f}  | {prompt[:45]}..."
        )

        gc.collect()
        torch.cuda.empty_cache()

    # Summary statistics
    dots = np.array([r["total_dot"] for r in verification_results])
    attrs = np.array([r["attr_net"] for r in verification_results])
    ratios = np.array([r["mlp_ratio"] for r in verification_results])

    summary = {
        "n_prompts": len(verification_results),
        "graph_mode": mode,
        "target_layer": target_layer,
        "target_positions": positions,
        "dot_product_mean": float(dots.mean()),
        "dot_product_std": float(dots.std()),
        "attr_net_mean": float(attrs.mean()),
        "attr_net_std": float(attrs.std()),
        "mlp_ratio_mean": float(ratios.mean()),
        "mlp_ratio_std": float(ratios.std()),
        "mlp_pct_mean": float(ratios.mean() * 100),
        "attention_pct_mean": float((1 - ratios.mean()) * 100),
        # Legacy key name (kept for downstream tools that read it)
        "measurement_layer": target_layer,
        "measurement_position": positions[0] if len(positions) == 1 else None,
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY: Attribution Gap Analysis ({len(verification_results)} prompts)")
    print(f"{'='*60}")
    print(f"  Full dot product <x, r_hat>:  {summary['dot_product_mean']:>10.2f} ± {summary['dot_product_std']:.2f}")
    print(f"  MLP attribution sum:          {summary['attr_net_mean']:>10.2f} ± {summary['attr_net_std']:.2f}")
    print(f"  MLP ratio:                    {summary['mlp_pct_mean']:>10.2f}% ± {summary['mlp_ratio_std']*100:.2f}%")
    print(f"  Attention + embedding:        {summary['attention_pct_mean']:>10.2f}%")
    print(f"\n  Interpretation: Transcoders decompose {summary['mlp_pct_mean']:.1f}% of")
    print(f"  the refusal signal. The remaining {summary['attention_pct_mean']:.1f}% is carried")
    print(f"  by attention heads and token embeddings.")

    # ----------------------------------------------------------
    # CHECK 2: Per-layer decomposition (subset of prompts)
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"CHECK 2: Per-layer decomposition ({args.n_decompose} prompts, L{target_layer} @ pos=-2)")
    print(f"{'='*60}")
    # Per-layer decomposition is always at pos=-2 (the causal position) against
    # r_{L,pos=-2} — this tells the layer-buildup story for the causal target.
    # For multi-position mode, this is one projection of the multi-target scalar.
    decomp_pos = -2
    if decomp_pos not in r_hats:
        print(f"  WARN: pos=-2 not loaded (positions: {positions}). Skipping per-layer decomposition.")
        decomposition_results = []
    else:
        r_hat_decomp = r_hats[decomp_pos].to(model.device)
        decomposition_results = []
        for i in range(min(args.n_decompose, len(prompts_and_nets))):
            prompt = prompts_and_nets[i]["prompt"]
            formatted = format_prompt(tokenizer, prompt)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            total_tokens = inputs["input_ids"].shape[1]
            pos_idx = total_tokens + decomp_pos  # pos is negative

            # Embedding contribution
            emb_act = out.hidden_states[0][0, pos_idx, :].to(torch.float32)
            emb_dot = (emb_act @ r_hat_decomp).item()

            # Per-layer contributions
            layer_contribs = []
            for layer in range(config.N_LAYERS):
                before = out.hidden_states[layer][0, pos_idx, :].to(torch.float32)
                after = out.hidden_states[layer + 1][0, pos_idx, :].to(torch.float32)
                contrib = ((after - before) @ r_hat_decomp).item()
                layer_contribs.append({"layer": layer, "contribution": contrib})

            # Verify: embedding + sum(layer_contribs[:target_layer+1]) == full dot product
            full_dot = (
                out.hidden_states[target_layer + 1][0, pos_idx, :].to(torch.float32) @ r_hat_decomp
            ).item()
            sum_contribs = emb_dot + sum(
                lc["contribution"] for lc in layer_contribs[:target_layer + 1]
            )

            sorted_layers = sorted(
                layer_contribs[:target_layer + 1],
                key=lambda x: abs(x["contribution"]), reverse=True,
            )

            print(f"\n  Prompt: {prompt[:55]}...")
            print(f"    Embedding dot:        {emb_dot:>10.2f}")
            print(f"    Sum layer contribs:   {sum_contribs:>10.2f}")
            print(f"    Full dot product:     {full_dot:>10.2f}")
            print(f"    Reconstruction error: {abs(full_dot - sum_contribs):.6f}")
            print(f"    Top 5 layers:")
            for lc in sorted_layers[:5]:
                print(f"      L{lc['layer']:>2}: {lc['contribution']:>+10.2f}")

            decomposition_results.append({
                "prompt": prompt[:80],
                "decomposition_position": decomp_pos,
                "embedding_dot": emb_dot,
                "layer_contributions": layer_contribs,
                "full_dot_product": full_dot,
                "sum_check": sum_contribs,
                "reconstruction_error": abs(full_dot - sum_contribs),
            })

            del out
            gc.collect()
            torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------
    per_layer_aggregate = aggregate_and_plot(decomposition_results, summary, out_dir)

    output = {
        "summary": summary,
        "per_prompt": verification_results,
        "per_layer_aggregate": per_layer_aggregate,
    }
    save_json(output, out_dir / "verification_results.json")
    save_json(decomposition_results, out_dir / "per_layer_decomposition.json")

    print(f"\nResults saved to {out_dir}/")
    print("DONE!")


if __name__ == "__main__":
    main()