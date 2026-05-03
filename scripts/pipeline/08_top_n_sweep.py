"""
Stage 08b: per-prompt top-N feature ablation sweep + random-feature control.
==============================================================================
The recovery-vs-features Pareto curve that defends Pillar 3 of the paper
("expression resists localization").

Differs from `08_ablate_subcircuits.py` in one essential way: the ablation
features are **per-prompt**, not global. For each prompt, we pick its top-N
features by max absolute attribution (aggregated across conditions) and
ablate THAT prompt with THOSE features.

Builds 7 per-prompt ablation sets:
  per_prompt_top_1, top_5, top_10, top_20, top_50, top_100  (Pareto curve)
  per_prompt_random_6                                       (random control)

Output schema matches Stage 08 (`ablation_results.json` + `ablation_summary.json`),
so the downstream renormalization (`08_renorm_baselines.py`) works on this output
identically.

Usage:
    PYTHONPATH=src python3 scripts/pipeline/08_top_n_sweep.py \\
        --run-dir data/results/pipeline_runs/run_20260430_023247_topN \\
        --max-new-tokens 80

Wall on RTX 4090: ~50 prompts × 11 conditions × 8 generations × 7 s/gen ≈ 8.6 h.
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Reuse helpers from Stage 08
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    classify_response,
    format_prompt,
    is_coherent,
    load_controlled_dataset,
    load_json,
    save_json,
)

# Re-import Stage 08 helpers (the script ships them as module-level funcs)
import importlib.util
_stage08_spec = importlib.util.spec_from_file_location(
    "stage08", str(Path(__file__).resolve().parent / "08_ablate_subcircuits.py")
)
_stage08 = importlib.util.module_from_spec(_stage08_spec)
_stage08_spec.loader.exec_module(_stage08)
build_per_prompt_top_index = _stage08.build_per_prompt_top_index
build_interventions = _stage08.build_interventions
generate_ablated = _stage08.generate_ablated
generate_baseline_rm = _stage08.generate_baseline_rm
classify_record = _stage08.classify_record
aggregate_summary = _stage08.aggregate_summary
aggregate_weighted_summary = _stage08.aggregate_weighted_summary

JB_CLASSES = ("analytical", "cognitive_reframe", "completion", "fiction", "roleplay")
DEFAULT_N_VALUES = (1, 5, 10, 20, 50, 100)
DEFAULT_RANDOM_N = 6
DEFAULT_RANDOM_SEED = 42


def parse_feature_key(key: str) -> tuple[int, int]:
    """Parse 'L{layer}:F{feat_idx}' to (layer, feat_idx)."""
    layer = int(key.split(":")[0][1:])
    feat = int(key.split(":")[1][1:])
    return (layer, feat)


def build_per_prompt_top_n(
    per_prompt_top: dict, n_values: tuple[int, ...]
) -> dict[str, dict[int, list[tuple[int, int]]]]:
    """For each N, build {prompt_id: [(L, F)]} by aggregating per-condition top
    features across conditions and taking top-N by max abs attribution.

    Returns ``{ablation_name: {prompt_id: [features]}}``.
    """
    out: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for n in n_values:
        ablation_name = f"per_prompt_top_{n}"
        per_prompt_features: dict[int, list[tuple[int, int]]] = {}
        for pid, per_cond in per_prompt_top.items():
            # Aggregate across conditions: take max |attr| per feature
            feat_max: dict[str, float] = {}
            for cond, feats in per_cond.items():
                for fk, attr in feats.items():
                    if attr > feat_max.get(fk, 0.0):
                        feat_max[fk] = attr
            sorted_feats = sorted(feat_max.items(), key=lambda x: -x[1])
            top_n = [parse_feature_key(fk) for fk, _ in sorted_feats[:n]]
            per_prompt_features[pid] = top_n
        out[ablation_name] = per_prompt_features
    return out


def build_per_prompt_random(
    per_prompt_top: dict, n: int, seed: int
) -> dict[str, dict[int, list[tuple[int, int]]]]:
    """Build per-prompt random feature ablation. Random features are sampled
    from the union of all features that appear in ANY prompt's top features
    (i.e. the universe of attribution-relevant transcoder features for this run).
    Each prompt gets a different random N features (different seed per prompt).
    """
    # Union of all features observed across all prompts × conditions
    universe: set[str] = set()
    for pid, per_cond in per_prompt_top.items():
        for cond, feats in per_cond.items():
            universe.update(feats.keys())
    universe_list = sorted(universe)  # deterministic order

    per_prompt_features: dict[int, list[tuple[int, int]]] = {}
    rng = random.Random(seed)
    for pid in per_prompt_top:
        # Per-prompt seed for reproducibility
        local_rng = random.Random(rng.randint(0, 2**31))
        chosen_keys = local_rng.sample(universe_list, k=min(n, len(universe_list)))
        per_prompt_features[pid] = [parse_feature_key(fk) for fk in chosen_keys]

    return {f"per_prompt_random_{n}": per_prompt_features}


def process_prompt(rm, tokenizer, row, ablation_sets_per_prompt, conditions,
                   max_new_tokens: int) -> dict:
    """Per-prompt ablation pass with per-prompt feature lists.

    `ablation_sets_per_prompt`: ``{ablation_name: {prompt_id: [(L, F)]}}``
    """
    pid = row["id"]
    result: dict = {
        "prompt_idx": row.get("idx"),
        "prompt_id": pid,
        "topic": row.get("topic"),
        "baseline": {},
        "ablations": {},
    }

    # Baselines (regenerate fresh — matches canonical sweep)
    for cond in conditions:
        formatted = format_prompt(tokenizer, row["conditions"][cond]["text"])
        resp = generate_baseline_rm(rm, tokenizer, formatted, max_new_tokens)
        result["baseline"][cond] = classify_record(resp)

    # Ablations: one per (ablation_name) using THIS prompt's features
    for abl_name, per_prompt_features in ablation_sets_per_prompt.items():
        features = per_prompt_features.get(pid, [])
        if not features:
            continue
        result["ablations"][abl_name] = {"n_features": len(features)}
        per_cond: dict = {}
        for cond in conditions:
            formatted = format_prompt(tokenizer, row["conditions"][cond]["text"])
            tokenized_length = len(tokenizer(formatted)["input_ids"])
            interventions = build_interventions(features, "all", tokenized_length)
            resp = generate_ablated(rm, tokenizer, formatted, interventions, max_new_tokens)
            per_cond[cond] = classify_record(resp)
            per_cond[cond]["changed_vs_baseline"] = (
                per_cond[cond]["cls"] != result["baseline"][cond]["cls"]
            )
        result["ablations"][abl_name]["all"] = per_cond

    return result


def main():
    p = argparse.ArgumentParser(description="Per-prompt top-N + random-N ablation sweep.")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Sister run directory (must already contain 02_attribution/, 06_causal/, etc.)")
    p.add_argument("--n-values", type=str, default=",".join(map(str, DEFAULT_N_VALUES)),
                   help=f"Comma-separated N values for per-prompt top-N. Default: {DEFAULT_N_VALUES}.")
    p.add_argument("--random-n", type=int, default=DEFAULT_RANDOM_N,
                   help=f"Number of random features for control. Default: {DEFAULT_RANDOM_N}.")
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED,
                   help=f"Seed for random control. Default: {DEFAULT_RANDOM_SEED}.")
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    n_values = tuple(int(x.strip()) for x in args.n_values.split(",") if x.strip())
    print(f"[top_n_sweep] N values: {n_values}, random_N: {args.random_n}, seed: {args.random_seed}")

    out_dir = args.run_dir / "08_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-prompt top-N feature sets from Stage 02
    print(f"[top_n_sweep] reading Stage 02 per-prompt top features from {args.run_dir}")
    per_prompt_top = build_per_prompt_top_index(args.run_dir, mode="multi")
    if not per_prompt_top:
        raise RuntimeError(f"No per-prompt top features found in {args.run_dir}/02_attribution/")
    print(f"[top_n_sweep]   indexed {len(per_prompt_top)} prompts")

    ablation_sets_per_prompt: dict = {}
    ablation_sets_per_prompt.update(build_per_prompt_top_n(per_prompt_top, n_values))
    ablation_sets_per_prompt.update(
        build_per_prompt_random(per_prompt_top, args.random_n, args.random_seed)
    )
    print(f"[top_n_sweep] ablation sets:")
    for name, per_pp in ablation_sets_per_prompt.items():
        sizes = [len(v) for v in per_pp.values()]
        print(f"    {name:<30} {len(per_pp)} prompts, mean_n_feat={sum(sizes)/len(sizes):.1f}")

    # Save the ablation set sidecar (so we can reproduce / inspect)
    sidecar = {
        name: {str(pid): [f"L{L}:F{F}" for (L, F) in feats]
               for pid, feats in per_pp.items()}
        for name, per_pp in ablation_sets_per_prompt.items()
    }
    save_json(sidecar, out_dir / "per_prompt_ablation_sets.json")
    print(f"[top_n_sweep] wrote per_prompt_ablation_sets.json")

    # Load dataset
    rows = load_controlled_dataset(n_prompts=args.max_prompts)
    if args.max_prompts:
        rows = rows[:args.max_prompts]
    conditions = list(rows[0]["conditions"].keys())
    print(f"[top_n_sweep] {len(rows)} prompts × {len(conditions)} conditions")

    # Resume
    ckpt_path = out_dir / "ablation_checkpoint.json"
    if args.resume and ckpt_path.exists():
        ckpt = load_json(ckpt_path)
        results = ckpt.get("results", [])
        done_ids = {r["prompt_id"] for r in results}
        rows_todo = [r for r in rows if r["id"] not in done_ids]
        print(f"[top_n_sweep] [resume] {len(done_ids)} done, {len(rows_todo)} remain")
    else:
        results = []
        rows_todo = rows

    # Load ReplacementModel
    print(f"[top_n_sweep] loading ReplacementModel ({config.MODEL_NAME}, {args.dtype})...")
    import torch
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    from circuit_tracer import ReplacementModel
    rm = ReplacementModel.from_pretrained(
        config.MODEL_NAME, config.TRANSCODER_PATH,
        dtype=dtype_map[args.dtype], backend="nnsight", lazy_encoder=False,
    )
    tokenizer = rm.tokenizer
    print(f"[top_n_sweep] model ready")

    # Main loop
    print(f"\n[top_n_sweep] starting ablation generation...")
    t0 = time.time()
    for i, row in enumerate(rows_todo):
        t_p = time.time()
        result = process_prompt(rm, tokenizer, row, ablation_sets_per_prompt,
                                conditions, args.max_new_tokens)
        results.append(result)
        elapsed_p = time.time() - t_p
        elapsed_total = time.time() - t0
        rate_min = elapsed_total / (i + 1)
        eta_min = rate_min * (len(rows_todo) - (i + 1)) / 60
        print(f"[top_n_sweep]   prompt {row['id']:>3d} ({i+1}/{len(rows_todo)}): "
              f"{elapsed_p:5.1f}s, total {elapsed_total/60:5.1f} min, ETA {eta_min:5.1f} min")

        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(rows_todo):
            save_json({"results": results}, ckpt_path)
        gc.collect()

    save_json({"results": results}, ckpt_path)
    save_json({"results": results}, out_dir / "ablation_results.json")
    print(f"\n[top_n_sweep] done in {(time.time() - t0)/60:.1f} min")

    # Build summary (custom — per-prompt n_features varies, so we average it)
    print(f"[top_n_sweep] building summary...")
    summary = {"per_ablation": {}}
    abl_names = list(ablation_sets_per_prompt.keys())
    for abl_name in abl_names:
        # Mean n_features for this ablation across prompts (close to constant in practice)
        sizes = [r.get("ablations", {}).get(abl_name, {}).get("n_features")
                 for r in results]
        sizes = [s for s in sizes if s is not None]
        per_abl = {"n_features": (sum(sizes) // len(sizes)) if sizes else 0,
                   "positions": {"all": {}}}
        per_pos = per_abl["positions"]["all"]
        for cond in conditions:
            br = bc = ar = ac = chg = chg_co = rec = brk = n_seen = 0
            for r in results:
                bl = r.get("baseline", {}).get(cond)
                ab = r.get("ablations", {}).get(abl_name, {}).get("all", {}).get(cond)
                if bl is None or ab is None:
                    continue
                n_seen += 1
                if bl["cls"] == "REFUSE":
                    br += 1
                else:
                    bc += 1
                if ab["cls"] == "REFUSE":
                    ar += 1
                else:
                    ac += 1
                if ab.get("changed_vs_baseline"):
                    chg += 1
                    if ab.get("coherent"):
                        chg_co += 1
                    if bl["cls"] == "COMPLY" and ab["cls"] == "REFUSE":
                        rec += 1
                    if bl["cls"] == "REFUSE" and ab["cls"] == "COMPLY":
                        brk += 1
            if n_seen == 0:
                continue
            per_pos[cond] = {
                "n_seen": n_seen, "n_baseline_refuse": br, "n_baseline_comply": bc,
                "n_ablated_refuse": ar, "n_ablated_comply": ac, "n_changed": chg,
                "n_coherent_changed": chg_co, "n_recovered_refusal": rec,
                "n_broke_refusal": brk,
                "recovery_rate": round(rec / bc, 4) if bc else 0.0,
                "break_rate": round(brk / br, 4) if br else 0.0,
            }
        summary["per_ablation"][abl_name] = per_abl
    summary = aggregate_weighted_summary(summary)
    save_json(summary, out_dir / "ablation_summary.json")
    print(f"[top_n_sweep] wrote ablation_summary.json")

    # Print headline
    for abl_name, per_abl in summary["per_ablation"].items():
        for pos_mode, per_cond in per_abl["positions"].items():
            w = per_cond.get("weighted", {})
            print(
                f"[top_n_sweep]  {abl_name:<28} ({pos_mode}): "
                f"JB={w.get('jb_weighted_recovery_rate', 0)*100:5.1f}% "
                f"(n={w.get('jb_total_baseline_comply', 0)})  "
                f"ctrl={w.get('ctrl_weighted_break_rate', 0)*100:5.1f}%  "
                f"bare={w.get('bare_break_rate', 0)*100:5.1f}%"
            )

    print(f"[top_n_sweep] DONE!")


if __name__ == "__main__":
    main()
