"""Top-K circuit sparsity sweep on Qwen3-4B @ L18.

For each (prompt, condition), ranks that prompt's attribution-graph
features/edges and ablates the top-K for K in a sweep schedule, to locate the
Pareto knee: how many features (or edges) must be removed to (a) break the
refusal mechanism and (b) remove the jailbreak mechanism.

Two mechanisms (design spec 2026-06-01-qwen-subcircuits-topk-design.md §3.2):

  --mechanism proxy   Residual-attribution proxy: sum the top-K records'
                      signed attribution (normalized-r units from the packed
                      graphs), convert delta = delta_norm * ||r_unnorm||, and
                      subtract along r via make_scalar_rhat_subtraction_hook
                      at L18 (all positions). Plain HF model, fp32, TF32 off.
                      Sources: features (aggregated) or edges (per-edge).

  --mechanism zero    True feature zeroing via circuit-tracer
                      ReplacementModel.feature_intervention_generate:
                      (layer, slice(None), feat_idx, 0.0) per top-K feature —
                      the Stage-08 'all'-positions convention. bf16.
                      Source: features only.

Rankings (per source) and the conditions each runs on:

  features: pos        most pro-refusal attribution    -> bare        (break refusal)
            neg        most anti-refusal attribution   -> jb_*        (remove jailbreak)
            activation strongest-activated features    -> bare + jb_* (both directions)
  edges:    pos / neg / abs (signed-attribution ranks, matching the Gemma
            0d/0e sweep naming) -> pos: bare, neg: jb_*, abs: bare + jb_*

A no-intervention baseline block (all sweep conditions) is generated for flip
computation by the aggregator. K is clamped per prompt to available records
(n_used recorded). Incremental save per cell; --resume skips completed cells.

Usage (full):
    PYTHONPATH=scripts python3 scripts/emnlp_perm_edit/00_topk_circuit_sweep_qwen.py \
        --mechanism proxy --source features \
        --graph-data-dir <run>/graph_data \
        --rhat-path <run>/01_direction/positions_L18/pos_-1_unnormalized.pt \
        --out data/results/emnlp_perm_edit/qwen_subcircuits/topk_sweep_proxy_features.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "scripts" / "pipeline_qwen"))
from graph_loader import (  # noqa: E402
    extract_edge_records_to_target,
    find_measurement_target_node_id,
    load_packed_graph,
)
# Qwen pipeline utils: format_prompt bakes enable_thinking=False (load-bearing).
from utils import classify_response, format_prompt, is_coherent, load_controlled_dataset  # noqa: E402

LAYER = 18
QWEN_RUN_DIR_DEFAULT = REPO / "data/results/pipeline_runs_qwen/run_emnlp_qwen_L18_20260522"
DEFAULT_K_VALUES = [1, 3, 5, 10, 25, 50, 100, 250]
JB_CONDITIONS = ["jb_fiction", "jb_roleplay", "jb_analytical", "jb_completion", "jb_cognitive_reframe"]
BARE_CONDITIONS = ["bare"]
RANKINGS_BY_SOURCE = {
    "features": ("pos", "neg", "activation"),
    "edges": ("pos", "neg", "abs"),
}
# Which conditions each ranking is evaluated on (design spec §3.2 table).
RANKING_CONDITIONS = {
    "pos": BARE_CONDITIONS,
    "neg": JB_CONDITIONS,
    "activation": BARE_CONDITIONS + JB_CONDITIONS,
    "abs": BARE_CONDITIONS + JB_CONDITIONS,
}
BASELINE_CONDITIONS = BARE_CONDITIONS + JB_CONDITIONS


def feature_key(layer: int, feat_idx: int) -> str:
    return f"L{layer}:F{feat_idx}"


def aggregate_feature_records(records: list[dict]) -> list[dict]:
    """Per-edge records -> per-(layer, feature) entries.

    signed_attribution summed across position instances, activation = max.
    Returned sorted by signed_attribution DESCENDING (most pro-refusal first),
    the canonical order the ranking selectors index into.
    """
    agg: dict[tuple[int, int], dict] = {}
    for r in records:
        if r.get("category", "feature") != "feature":
            continue
        if r["layer"] is None or r["feature"] is None:
            continue
        lf = (r["layer"], r["feature"])
        entry = agg.get(lf)
        if entry is None:
            agg[lf] = {
                "layer": r["layer"],
                "feature": r["feature"],
                "signed_attribution": r["signed_attribution"],
                "activation": r.get("activation", 0.0) or 0.0,
            }
        else:
            entry["signed_attribution"] += r["signed_attribution"]
            entry["activation"] = max(entry["activation"], r.get("activation", 0.0) or 0.0)
    return sorted(agg.values(), key=lambda e: e["signed_attribution"], reverse=True)


def select_topk(records: list[dict], ranking: str, K: int) -> list[dict]:
    """Select the top-K records under a ranking.

    `records` must be sorted by signed_attribution descending (the order
    aggregate_feature_records / extract_edge_records_to_target return).
    """
    if not records:
        return []
    if ranking == "pos":
        chosen = records[:K]
    elif ranking == "neg":
        chosen = list(reversed(records))[:K]
    elif ranking == "abs":
        chosen = sorted(records, key=lambda r: abs(r["signed_attribution"]), reverse=True)[:K]
    elif ranking == "activation":
        chosen = sorted(records, key=lambda r: r.get("activation", 0.0) or 0.0, reverse=True)[:K]
    else:
        raise ValueError(f"unknown ranking: {ranking}")
    return chosen


def summarize_selection(chosen: list[dict]) -> dict:
    """Delta + bookkeeping for a selected top-K set."""
    delta_norm = sum(r["signed_attribution"] for r in chosen)
    return {
        "n_used": len(chosen),
        "delta_norm_basis": delta_norm,
        "sum_abs_attribution": sum(abs(r["signed_attribution"]) for r in chosen),
    }


def build_zero_interventions(chosen: list[dict]) -> list[tuple]:
    """(layer, slice(None), feat_idx, 0.0) per unique feature — Stage 08
    'all'-positions convention."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple] = []
    for r in chosen:
        lf = (r["layer"], r["feature"])
        if lf in seen:
            continue
        seen.add(lf)
        out.append((r["layer"], slice(None), r["feature"], 0.0))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Top-K circuit sparsity sweep (Qwen3-4B L18)")
    p.add_argument("--mechanism", required=True, choices=["proxy", "zero"])
    p.add_argument("--source", required=True, choices=["features", "edges"])
    p.add_argument("--graph-data-dir", type=Path,
                   default=QWEN_RUN_DIR_DEFAULT / "graph_data")
    p.add_argument("--graph-mode", default="single", choices=["single", "multi"])
    p.add_argument("--rhat-path", type=Path,
                   default=QWEN_RUN_DIR_DEFAULT / "01_direction/positions_L18/pos_-1_unnormalized.pt",
                   help="UNNORMALIZED r[L18, pos=-1] (||r||≈15.14). proxy mechanism only.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoders", default="mwhanna/qwen3-4b-transcoders",
                   help="zero mechanism only")
    p.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    p.add_argument("--rankings", default=None,
                   help="Comma-separated; default per source "
                        "(features: pos,neg,activation; edges: pos,neg,abs)")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip the no-intervention baseline block (e.g. when another "
                        "sweep of the same mechanism family already produced it).")
    p.add_argument("--dtype", default=None, choices=["bfloat16", "float32"],
                   help="Default: float32 for proxy (precision protocol), bfloat16 for zero.")
    p.add_argument("--resume", action="store_true",
                   help="Skip cells already complete in --out.")
    return p.parse_args()


def load_rankings(args, dataset) -> dict:
    """Precompute per-(prompt_idx, condition) sorted record lists from packed graphs."""
    conditions_needed = sorted(
        {c for r in (args.rankings_list or RANKINGS_BY_SOURCE[args.source])
         for c in RANKING_CONDITIONS[r]} | set(BASELINE_CONDITIONS)
    )
    filter_category = "feature" if args.source == "features" else None
    rankings = {}
    skipped = []
    for prompt_idx in range(len(dataset)):
        for cond in conditions_needed:
            slug = f"{prompt_idx:03d}_{cond}_{args.graph_mode}"
            path = args.graph_data_dir / f"{slug}.json.gz"
            if not path.exists():
                skipped.append(slug)
                continue
            graph = load_packed_graph(path)
            target_id = find_measurement_target_node_id(graph)
            records = extract_edge_records_to_target(graph, target_id, filter_category=filter_category)
            if args.source == "features":
                records = aggregate_feature_records(records)
            rankings[(prompt_idx, cond)] = records
    return rankings, skipped


def main():
    args = parse_args()
    if args.mechanism == "zero" and args.source == "edges":
        raise SystemExit("zero mechanism supports --source features only "
                         "(true edge zeroing is unsupported; use proxy for edges).")
    args.rankings_list = ([r.strip() for r in args.rankings.split(",")] if args.rankings
                          else list(RANKINGS_BY_SOURCE[args.source]))
    for r in args.rankings_list:
        if r not in RANKING_CONDITIONS:
            raise SystemExit(f"unknown ranking {r!r}")
        if r == "activation" and args.source == "edges":
            raise SystemExit("'activation' ranking is features-only; use 'abs' for edges.")
    k_values = [int(k) for k in args.k_values.split(",")]
    dtype = args.dtype or ("float32" if args.mechanism == "proxy" else "bfloat16")

    import torch  # heavy import deferred so unit tests can import this module CPU-only
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]

    print(f"[topk-{args.mechanism}-{args.source}] rankings={args.rankings_list} K={k_values}")
    print(f"[topk] precomputing rankings from {args.graph_data_dir}")
    rankings, skipped = load_rankings(args, dataset)
    print(f"  {len(rankings)} (prompt, condition) ranked ({len(skipped)} graphs missing)")

    # ---- model / hook machinery, per mechanism ----
    r_hat = None
    r_hat_norm = None
    rm = None
    model = None
    tokenizer = None
    t0 = time.time()
    if args.mechanism == "proxy":
        from edge_ablation_hook import make_scalar_rhat_subtraction_hook  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        r_obj = torch.load(args.rhat_path, weights_only=False, map_location="cpu")
        r_hat = (r_obj if isinstance(r_obj, torch.Tensor) else r_obj["direction"]).float()
        r_hat_norm = r_hat.norm().item()
        print(f"[topk] ||r_unnorm[L{LAYER}]|| = {r_hat_norm:.4f}")
        if r_hat_norm < 5.0:
            print(f"  WARNING: expected ||r|| ≈ 15.14 for Qwen L18 unnormalized; got {r_hat_norm:.4f}. "
                  f"Did you pass the normalized file?")
        print(f"[topk] loading {args.model} in {dtype}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype_map[dtype], device_map="cuda")
        model.eval()
        layers = (model.model.language_model.layers
                  if hasattr(model.model, "language_model") else model.model.layers)
        target_layer = layers[LAYER]
    else:
        from circuit_tracer import ReplacementModel
        print(f"[topk] loading ReplacementModel {args.model} + {args.transcoders} in {dtype}")
        rm = ReplacementModel.from_pretrained(
            args.model, args.transcoders,
            dtype=dtype_map[dtype], backend="nnsight", lazy_encoder=False)
        tokenizer = rm.tokenizer
    print(f"  model ready in {time.time()-t0:.1f}s")

    pad_id = tokenizer.eos_token_id

    def generate_proxy(text: str, delta: float | None) -> str:
        from edge_ablation_hook import make_scalar_rhat_subtraction_hook
        formatted = format_prompt(tokenizer, text)
        ids = tokenizer(formatted, return_tensors="pt").to(model.device)
        prompt_len = ids.input_ids.shape[1]
        handle = None
        if delta is not None:
            handle = target_layer.register_forward_hook(
                make_scalar_rhat_subtraction_hook(r_hat, delta, position_mode="all"))
        try:
            with torch.no_grad():
                out = model.generate(**ids, do_sample=False,
                                     max_new_tokens=args.max_new_tokens, pad_token_id=pad_id)
            return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        finally:
            if handle is not None:
                handle.remove()

    def generate_zero(text: str, interventions: list) -> str:
        formatted = format_prompt(tokenizer, text)
        decoded, _logits, _cache = rm.feature_intervention_generate(
            formatted, interventions, max_new_tokens=args.max_new_tokens,
            return_activations=False, do_sample=False)
        if decoded.startswith(formatted):
            return decoded[len(formatted):].strip()
        idx = decoded.rfind("<|im_start|>assistant\n")
        return decoded[idx + len("<|im_start|>assistant\n"):].strip() if idx >= 0 else decoded.strip()

    # ---- resume ----
    results = {
        "metadata": {
            "mechanism": args.mechanism, "source": args.source, "layer": LAYER,
            "model": args.model, "graph_mode": args.graph_mode,
            "k_values": k_values, "rankings": args.rankings_list,
            "ranking_conditions": {r: RANKING_CONDITIONS[r] for r in args.rankings_list},
            "n_prompts": len(dataset), "max_new_tokens": args.max_new_tokens,
            "dtype": dtype, "r_hat_norm": r_hat_norm,
            "delta_convention": "delta = delta_norm_basis * ||r_unnorm|| (proxy only)",
            "positions_convention": "all positions (proxy hook + zero interventions)",
        },
        "baseline": {},
        "per_cell": {},
    }
    if args.resume and args.out.exists():
        prev = json.loads(args.out.read_text())
        if (prev.get("metadata", {}).get("mechanism") == args.mechanism
                and prev["metadata"].get("source") == args.source):
            results["baseline"] = prev.get("baseline", {})
            results["per_cell"] = prev.get("per_cell", {})
            print(f"[topk] resume: {len(results['per_cell'])} cells already present")

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))

    def expected_cell_size(ranking: str) -> int:
        return len(dataset) * len(RANKING_CONDITIONS[ranking])

    # ---- baseline block ----
    if not args.skip_baseline and len(results["baseline"].get("records", [])) < len(dataset) * len(BASELINE_CONDITIONS):
        print(f"\n[topk] baseline block ({len(dataset)} prompts × {len(BASELINE_CONDITIONS)} conditions)")
        t_b = time.time()
        recs = []
        for prompt_idx, prompt in enumerate(dataset):
            for cond in BASELINE_CONDITIONS:
                text = prompt["conditions"][cond]["text"]
                resp = (generate_proxy(text, None) if args.mechanism == "proxy"
                        else generate_zero(text, []))
                recs.append({"prompt_idx": prompt_idx, "condition": cond,
                             "response": resp[:300],
                             "classification": classify_response(resp),
                             "coherent": is_coherent(resp)})
            if (prompt_idx + 1) % 10 == 0:
                save()
        results["baseline"] = {"model_family": args.mechanism, "records": recs}
        save()
        print(f"  baseline done in {time.time()-t_b:.0f}s")

    # ---- sweep cells ----
    n_cells = len(args.rankings_list) * len(k_values)
    cell_idx = 0
    t_total = time.time()
    for ranking in args.rankings_list:
        conds = RANKING_CONDITIONS[ranking]
        for K in k_values:
            cell_idx += 1
            key = f"{ranking}_K{K}"
            if args.resume and len(results["per_cell"].get(key, [])) >= expected_cell_size(ranking):
                print(f"[topk] [{cell_idx}/{n_cells}] {key} — complete, skipping")
                continue
            print(f"\n[topk] [{cell_idx}/{n_cells}] ranking={ranking} K={K} conds={conds}")
            t_v = time.time()
            per_records = []
            for prompt_idx, prompt in enumerate(dataset):
                for cond in conds:
                    records = rankings.get((prompt_idx, cond))
                    if not records:
                        continue
                    chosen = select_topk(records, ranking, K)
                    sel = summarize_selection(chosen)
                    text = prompt["conditions"][cond]["text"]
                    if args.mechanism == "proxy":
                        delta = sel["delta_norm_basis"] * r_hat_norm
                        resp = generate_proxy(text, delta)
                        sel["delta_applied"] = delta
                    else:
                        resp = generate_zero(text, build_zero_interventions(chosen))
                    per_records.append({
                        "prompt_idx": prompt_idx, "condition": cond, **sel,
                        "response": resp[:300],
                        "classification": classify_response(resp),
                        "coherent": is_coherent(resp),
                    })
                    if len(per_records) % 100 == 0:
                        results["per_cell"][key] = per_records
                        save()
            results["per_cell"][key] = per_records
            save()
            elapsed = time.time() - t_v
            print(f"  {key} done in {elapsed:.0f}s ({len(per_records)} gens, "
                  f"~{elapsed/max(len(per_records),1):.1f}s/gen)")

    print(f"\n[topk] sweep complete in {(time.time()-t_total)/60:.1f} min — wrote {args.out}")


if __name__ == "__main__":
    main()
