"""
Stage 02: Run CLT Attribution Graphs
=====================================
For each prompt × condition, compute circuit-tracer attribution graphs
targeting the refusal direction at a specified (layer, position).

Default target: L15 @ pos=-2 (the causally effective layer; Tejas causal
experiments flip 95/95 JB prompts at L15, 0/10 at L32 despite L32 having
~7x stronger separation).

Dataset: Tejas's controlled dataset (refusal_lens_controlled_dataset.json)
yields 11 conditions per prompt — bare + {jb,ctrl}_<class> for 5 classes.
The ctrl arm is length-matched to JB so bare↔ctrl isolates prefix-token
confounds from JB semantics (ctrl↔JB is the cleanest "JB effect" delta).

Extracts ALL active features per graph (no top-k filtering) and performs
3-way feature comparison: bare↔JB, bare↔ctrl, ctrl↔JB.

Inputs:  01_direction/directions/layer_{TARGET}.pt, controlled dataset JSON
Outputs: 02_attribution/attribution_results.json,
         feature_comparison_aggregate.json
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import (
    format_prompt,
    load_controlled_dataset,
    load_experiment_dataset,
    save_json,
    load_json,
    get_stage_dir,
    create_run_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CLT attribution graphs")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument(
        "--prompt-start", type=int, default=0,
        help="First prompt index to attribute (inclusive). Use with --prompt-end to shard across GPUs.",
    )
    parser.add_argument(
        "--prompt-end", type=int, default=None,
        help="Last prompt index to attribute (exclusive). Default: min(n_prompts, len(dataset)).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Feature/target backward-pass batch size (circuit-tracer default 512). "
             "On 48GB (RTX 6000 Ada): 256 safe, 512 borderline. "
             "On 80GB (H100): 512 recommended. batch_size=1 is ~100x slower — do not use.",
    )
    parser.add_argument(
        "--max-features", type=int, default=8000,
        help="Cap number of features attributed per graph. Active features per prompt run "
             "typically 10k-18k, many with near-zero attribution that the frontend prunes "
             "anyway (node_threshold=0.8). Set to 0 for unlimited (exact prior behavior, ~2x slower).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Path to the controlled dataset JSON (default: config.CONTROLLED_DATASET_PATH)",
    )
    parser.add_argument(
        "--legacy-dataset", action="store_true",
        help="Use the old bare+5JB random-prefix selection path "
             "(utils.load_experiment_dataset). Only for reproducing pre-April runs.",
    )
    parser.add_argument(
        "--target-layer", type=int, default=config.MEASUREMENT_LAYER,
        help="Transformer layer at which attribution is measured (default: 15).",
    )
    parser.add_argument(
        "--multi-position-targets", type=int, nargs="+",
        default=config.TARGET_POSITIONS_MULTI,
        help="Positions for the multi-target attribution graph (default: "
             "the Gemma-3 template positions [-5, -3, -2] = "
             "<end_of_turn>, <start_of_turn>, 'model'). Requires Stage 01 run "
             "with per-position directions at target_layer.",
    )
    parser.add_argument(
        "--single-position-target", type=int,
        default=config.TARGET_POSITIONS_SINGLE[0],
        help="Position for the single-target baseline graph (default: -2). "
             "Tejas validated this as the causal L15 target (95/95 JB flip).",
    )
    parser.add_argument(
        "--skip-multi-graph", action="store_true",
        help="Do not produce the multi-position graph (single-only run).",
    )
    parser.add_argument(
        "--skip-single-graph", action="store_true",
        help="Do not produce the single-position baseline graph (multi-only run).",
    )
    parser.add_argument(
        "--save-graphs", action="store_true",
        help="Persist full attribution graphs (.pt) to <run-dir>/02_attribution/graphs/",
    )
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
        help="Model dtype. Use bfloat16/float16 on 24 GB cards to avoid OOM.",
    )
    parser.add_argument(
        "--measurement-hook", type=str, default=config.MEASUREMENT_HOOK,
        help="TL hook where the cotangent is injected (default: hook_resid_post). "
             "Must match where Stage 01 extracted r — see MENTEE_NOTE_three_bugs.md "
             "(Bug 3). Pass empty string to use circuit-tracer's default "
             "(pre_feedforward_layernorm.output, post-RMSNorm pre-MLP).",
    )
    parser.add_argument(
        "--backend", type=str, choices=["nnsight", "transformerlens"],
        default=config.BACKEND,
        help="Circuit-tracer backend. Default 'transformerlens' is required for "
             "measurement_hook='hook_resid_post' — the nnsight backend has a "
             "`.grad`-on-non-module-output limitation that breaks residual-stream "
             "measurement (mentor's MENTEE_NOTE_three_bugs.md). Use 'nnsight' "
             "only for legacy runs without --measurement-hook.",
    )
    return parser.parse_args()


# JB classes in Tejas's controlled dataset — used to enumerate the 11
# conditions per prompt and to construct feature-comparison pairs.
CONTROLLED_CLASSES = ("roleplay", "fiction", "analytical", "completion", "cognitive_reframe")


def iter_conditions(prompt_row: dict):
    """Yield (cond_name, text, prefix) tuples for a prompt from either the
    controlled-dataset format (dict with 'conditions') or the legacy format
    (raw harmful string under 'instruction'). Exactly one bare + 10 prefixed
    entries for the controlled path; 1 bare + 5 random-JB entries for legacy.
    """
    if "conditions" in prompt_row:
        for cond_name, entry in prompt_row["conditions"].items():
            yield cond_name, entry["text"], entry["prefix"]
        return
    # Legacy fallback: bare + one random prefix from each class in config.JB_CLASSES
    text = prompt_row["instruction"]
    yield "bare", text, ""
    rng = random.Random(42 + hash(text) % 10_000)
    for cls, prefixes in config.JB_CLASSES.items():
        prefix = rng.choice(prefixes)
        yield f"jb_{cls}", prefix + text.lower(), prefix


def feature_key(layer: int, feat_idx: int) -> str:
    return f"L{layer}:F{feat_idx}"


def _aggregated_target_row(graph, n_selected: int | None = None) -> torch.Tensor:
    """Return the weighted sum of all target (logit) rows in the adjacency matrix.

    circuit-tracer's ``graph.adjacency_matrix`` orders rows as
    ``[features, errors, embeds, logits]`` (logit/target rows are LAST).
    The previous helper read ``adj[-1, :]``, which only picked ONE target row —
    for a multi-target graph with targets [-5, -3, -2] that was just pos=-2's
    row, indistinguishable from a single-target attribution at pos=-2.

    The correct aggregation is a ``logit_probabilities``-weighted sum over
    every target row. For CustomTargets built with ``prob=1.0`` this is a
    plain sum across targets — the total attribution contributed to any
    target sink by each source node.

    Parameters
    ----------
    graph : circuit_tracer.Graph
    n_selected : optional column slice; if given, return only those columns.

    Returns
    -------
    Tensor of shape (n_selected,) or (n_cols,) — per-source aggregated
    attribution to the full target set.
    """
    adj = graph.adjacency_matrix
    n_targets = len(graph.logit_targets) if graph.logit_targets else 1
    # Last n_targets rows are the target rows by circuit-tracer's convention.
    target_rows = adj[-n_targets:, :]
    weights = graph.logit_probabilities.to(target_rows.dtype)
    # (n_targets, n_cols) * (n_targets, 1) → sum axis 0 → (n_cols,)
    aggregated = (target_rows * weights.unsqueeze(1)).sum(dim=0)
    if n_selected is not None:
        aggregated = aggregated[:n_selected]
    return aggregated


def extract_all_features(graph) -> dict[str, dict]:
    """Extract every feature and its aggregated attribution from a graph.

    ``attribution`` is the weighted sum across all target rows — for a
    multi-position graph this is each feature's total contribution to the
    combined refusal signal across every target position.
    """
    active = graph.active_features
    selected = graph.selected_features
    n_selected = len(selected)
    target_row = _aggregated_target_row(graph, n_selected)

    features = {}
    for i in range(n_selected):
        orig_idx = int(selected[i])
        feat = active[orig_idx]
        layer = int(feat[0])
        pos = int(feat[1])
        feat_idx = int(feat[2])
        key = feature_key(layer, feat_idx)
        features[key] = {
            "layer": layer,
            "position": pos,
            "feature_idx": feat_idx,
            "attribution": float(target_row[i]),
            "activation": float(graph.activation_values[orig_idx])
            if orig_idx < len(graph.activation_values)
            else 0.0,
        }
    return features


def graph_summary(graph) -> dict:
    """Compute summary stats from a single attribution graph.

    For multi-target graphs, attribution is aggregated (weighted sum) across
    every target row so the summary reflects contribution to the combined
    target-set, not just the last-row target. Adds ``per_target`` with
    position-wise breakdowns for diagnostic use.
    """
    adj = graph.adjacency_matrix
    n_targets = len(graph.logit_targets) if graph.logit_targets else 1
    n_feat = graph.active_features.shape[0]

    # Aggregated row (weighted sum across all targets)
    last = _aggregated_target_row(graph)

    # Per-target breakdown so downstream code can inspect each position's
    # contribution separately when the multi-target aggregation is too coarse.
    per_target = []
    target_rows = adj[-n_targets:, :]
    for k in range(n_targets):
        row = target_rows[k]
        per_target.append({
            "pos_sum": float(row[row > 0].sum().item()),
            "neg_sum": float(row[row < 0].sum().item()),
            "net": float(row.sum().item()),
        })

    return {
        "pos_sum": float(last[last > 0].sum().item()),
        "neg_sum": float(last[last < 0].sum().item()),
        "net": float(last.sum().item()),
        "n_features": int(n_feat),
        "n_targets": int(n_targets),
        "per_target": per_target,
    }


def compare_features(bare_feats: dict, cls_feats: dict) -> dict:
    """Compare feature sets between bare and jailbreak conditions."""
    bare_keys = set(bare_feats.keys())
    cls_keys = set(cls_feats.keys())
    shared = bare_keys & cls_keys
    bare_only = bare_keys - cls_keys
    cls_only = cls_keys - bare_keys

    sign_flipped = []
    dampened = []
    amplified_anti = []

    for key in shared:
        b_attr = bare_feats[key]["attribution"]
        c_attr = cls_feats[key]["attribution"]
        delta = c_attr - b_attr

        if (b_attr > 0 and c_attr < 0) or (b_attr < 0 and c_attr > 0):
            sign_flipped.append({
                "key": key, "bare_attr": round(b_attr, 6), "cls_attr": round(c_attr, 6),
            })
        elif b_attr > 0 and delta < -0.01:
            dampened.append({"key": key, "delta": round(delta, 6)})
        elif b_attr < 0 and delta < -0.01:
            amplified_anti.append({"key": key, "delta": round(delta, 6)})

    return {
        "n_bare": len(bare_keys),
        "n_cls": len(cls_keys),
        "n_shared": len(shared),
        "n_bare_only": len(bare_only),
        "n_cls_only": len(cls_only),
        "n_sign_flipped": len(sign_flipped),
        "n_dampened": len(dampened),
        "n_amplified_anti": len(amplified_anti),
        "top_sign_flipped": sorted(sign_flipped, key=lambda x: abs(x["bare_attr"]), reverse=True)[:10],
        "top_dampened": sorted(dampened, key=lambda x: x["delta"])[:10],
        "top_amplified_anti": sorted(amplified_anti, key=lambda x: x["delta"])[:10],
    }


def _load_target_direction(run_dir: Path, target_layer: int) -> torch.Tensor:
    """Load the per-layer refusal direction written by Stage 01.

    Stage 01 writes both the normalized best_direction (at L32) and per-layer
    directions under directions/layer_XX.pt. For attribution at L15 we need
    the per-layer tensor; falling back to best_direction would silently use
    the L32 direction and invalidate the target.
    """
    dir_path = run_dir / "01_direction" / "directions" / f"layer_{target_layer:02d}.pt"
    if not dir_path.exists():
        raise FileNotFoundError(
            f"Per-layer direction not found at {dir_path}. "
            f"Run Stage 01 with --save-per-layer (or use run_20260417_010035 "
            f"which already emits per-layer directions)."
        )
    t = torch.load(dir_path, map_location="cpu", weights_only=False)
    if isinstance(t, dict):
        # Some older runs stored per-layer as {'direction': tensor}
        t = t.get("direction") or t.get("r") or next(iter(t.values()))
    return t.to(torch.float32)


def _load_per_position_directions(
    run_dir: Path, target_layer: int, requested_positions: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Load per-position refusal directions at `target_layer` from Stage 01.

    Stage 01 writes one ``pos_{k:+d}.pt`` file per computed position under
    ``01_direction/positions_L{layer:02d}/``. If a position file is missing
    (e.g. Stage 01 was run with --skip-per-position), it's silently omitted
    from the returned dict — caller handles the fallback.

    Parameters
    ----------
    run_dir : pipeline run directory
    target_layer : layer index (e.g. 15)
    requested_positions : if given, load only these positions; otherwise load
        every pos_*.pt file present.

    Returns
    -------
    dict {pos: normalized direction (cpu float32 tensor)}
    """
    pos_dir = run_dir / "01_direction" / f"positions_L{target_layer:02d}"
    if not pos_dir.exists():
        return {}

    out: dict[int, torch.Tensor] = {}
    if requested_positions is not None:
        iter_positions = sorted(set(requested_positions))
    else:
        # Discover from filesystem.
        iter_positions = []
        for p in sorted(pos_dir.glob("pos_*.pt")):
            name = p.stem  # e.g. "pos_-15" or "pos_-15_unnormalized"
            if name.endswith("_unnormalized"):
                continue
            try:
                iter_positions.append(int(name.removeprefix("pos_")))
            except ValueError:
                pass
        iter_positions = sorted(set(iter_positions))

    for pos in iter_positions:
        fp = pos_dir / f"pos_{pos:+d}.pt"
        if not fp.exists():
            continue
        r_hat = torch.load(fp, map_location="cpu", weights_only=False)
        if isinstance(r_hat, dict):
            r_hat = r_hat.get("direction") or next(iter(r_hat.values()))
        out[pos] = r_hat.to(torch.float32)

    return out


def _valid_positions_for_prompt(
    tokenizer, formatted_text: str, available_positions: list[int],
) -> tuple[list[int], int]:
    """Return (valid_positions, seq_len) for this prompt.

    A position `pos` is valid when |pos| <= seq_len (the tokenized prompt is
    long enough to have a token at that offset from the end). Short prompts
    skip deep positions (e.g. pos=-15 on a 10-token prompt is in the
    chat-template header, not in the real user content — skip).
    """
    input_ids = tokenizer(formatted_text, return_tensors="pt")["input_ids"]
    seq_len = input_ids.shape[1]
    valid = [p for p in sorted(available_positions) if abs(p) <= seq_len]
    return valid, seq_len


def _aggregate_comparison(results: list[dict], classes: tuple[str, ...]) -> dict:
    """Summarize per-prompt feature-comparison stats across the corpus.

    Output schema:
        {
          <class>: {
            "vs_bare":   {"n_shared": {"mean":..., "std":..., "min":..., "max":...}, ...},
            "vs_ctrl":   {...},       # None if no ctrl condition was run
            "ctrl_vs_bare": {...},    # None if no ctrl condition was run
          }
        }
    Same mean/std/min/max stat keys as before; nested one level deeper now.
    """
    agg: dict = {}
    for cls in classes:
        per_comparison: dict = {}
        for comp_kind in ("vs_bare", "vs_ctrl", "ctrl_vs_bare"):
            buckets = {
                "n_shared": [], "n_bare_only": [], "n_cls_only": [],
                "n_sign_flipped": [], "n_dampened": [], "n_amplified_anti": [],
                "n_bare": [], "n_cls": [],
            }
            for row in results:
                comp = (
                    row.get("feature_comparison", {})
                    .get(cls, {})
                    .get(comp_kind)
                )
                if comp:
                    for key in buckets:
                        buckets[key].append(comp[key])
            if buckets["n_shared"]:
                per_comparison[comp_kind] = {
                    k: {
                        "mean": round(float(np.mean(v)), 1),
                        "std": round(float(np.std(v)), 1),
                        "min": int(min(v)),
                        "max": int(max(v)),
                    }
                    for k, v in buckets.items()
                }
        if per_comparison:
            agg[cls] = per_comparison
    return agg


def main():
    args = parse_args()
    run_dir = args.run_dir
    out_dir = get_stage_dir(run_dir, "02_attribution")
    graphs_dir = out_dir / "graphs" if args.save_graphs else None
    if graphs_dir is not None:
        graphs_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("STAGE 02: Run CLT Attribution Graphs")
    print("=" * 60)

    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.attribution.targets import CustomTarget

    # Determine which graph modes to produce. Default: both "multi"
    # (targets=TARGET_POSITIONS_MULTI, template-anchored [-5,-3,-2]) and
    # "single" (target=TARGET_POSITIONS_SINGLE, the causally-verified pos=-2).
    produce_multi = not args.skip_multi_graph
    produce_single = not args.skip_single_graph
    if not produce_multi and not produce_single:
        raise ValueError(
            "--skip-multi-graph and --skip-single-graph both set; nothing to do"
        )

    # Load per-position directions for the multi graph. The single graph's
    # direction comes from the same position file (pos=-2) by default, so we
    # consolidate into one load call and pick out both target subsets.
    all_needed_positions = sorted(set(
        (args.multi_position_targets if produce_multi else [])
        + ([args.single_position_target] if produce_single else [])
    ))
    per_position_directions = _load_per_position_directions(
        run_dir, args.target_layer, all_needed_positions,
    )
    missing = set(all_needed_positions) - set(per_position_directions.keys())
    if missing:
        raise FileNotFoundError(
            f"Stage 01 did not emit directions for positions {sorted(missing)} at "
            f"L{args.target_layer}. Re-run Stage 01 with "
            f"--per-position-positions {' '.join(map(str, all_needed_positions))} "
            f"or adjust --multi-position-targets / --single-position-target."
        )
    # Move every tensor to GPU once and reuse across every prompt × condition.
    for pos in per_position_directions:
        per_position_directions[pos] = per_position_directions[pos].cuda()

    modes: dict[str, dict] = {}
    if produce_multi:
        modes["multi"] = {
            "positions": sorted(args.multi_position_targets),
        }
        print(
            f"  Multi graph: L{args.target_layer} targets = "
            f"{modes['multi']['positions']} "
            f"(template: <end_of_turn>, <start_of_turn>, 'model')"
        )
    if produce_single:
        modes["single"] = {
            "positions": [args.single_position_target],
        }
        print(
            f"  Single graph: L{args.target_layer} target = "
            f"{modes['single']['positions']} (causally-verified pos)"
        )

    tokenizer_module = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer = tokenizer_module.AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = "left"

    print(f"  Loading ReplacementModel (backend={args.backend})...")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    model = ReplacementModel.from_pretrained(
        config.MODEL_NAME,
        config.TRANSCODER_PATH,
        dtype=dtype_map[args.dtype],
        backend=args.backend,
        lazy_encoder=True,
    )
    print("  Ready.")

    # Load dataset — prefer Tejas's controlled dataset (11 conditions/prompt).
    if args.legacy_dataset:
        prompts = load_experiment_dataset(
            n_prompts=args.n_prompts, dataset_path=args.dataset,
        )
        print(f"  [legacy] Selected {len(prompts)} diverse harmful prompts")
    else:
        prompts = load_controlled_dataset(
            dataset_path=args.dataset, n_prompts=args.n_prompts,
        )

    # Shard support: when running multiple Stage 02 processes across GPUs,
    # each one handles a slice of the prompt list. prompt_start/prompt_end
    # are interpreted BEFORE checkpoint resume, so a shard's checkpoint
    # tracks only that shard's progress.
    shard_start = max(args.prompt_start, 0)
    shard_end = args.prompt_end if args.prompt_end is not None else len(prompts)
    shard_end = min(shard_end, len(prompts))
    if shard_start >= shard_end:
        raise ValueError(f"empty shard: prompt_start={shard_start} >= prompt_end={shard_end}")

    # Checkpoint path is shard-specific when shards are used, so concurrent
    # shards don't clobber each other's state.
    is_sharded = args.prompt_start != 0 or args.prompt_end is not None
    ckpt_name = (
        f"attribution_checkpoint_{shard_start:03d}_{shard_end:03d}.json"
        if is_sharded else "attribution_checkpoint.json"
    )
    checkpoint_path = out_dir / ckpt_name
    checkpoint = load_json(checkpoint_path) if (args.resume and checkpoint_path.exists()) else None
    start_idx = checkpoint.get("last_completed", shard_start - 1) + 1 if checkpoint else shard_start
    results = checkpoint.get("results", []) if checkpoint else []

    # Circuit-tracer treats max_feature_nodes=None as "unlimited"; our CLI
    # default is 8000 for speed, but 0 (or any non-positive) should map to
    # the unlimited case rather than literally zero features.
    max_features = args.max_features if (args.max_features and args.max_features > 0) else None

    print(
        f"  Attribution config: batch_size={args.batch_size}, "
        f"max_features={max_features if max_features else 'unlimited'}, "
        f"shard=[{shard_start}, {shard_end})"
    )

    t0 = time.time()
    for i in range(start_idx, shard_end):
        prompt_row = prompts[i]
        prompt_text = prompt_row.get("bare") or prompt_row.get("base") or prompt_row["instruction"]
        print(f"\n  Prompt {i + 1}/{shard_end} (shard [{shard_start},{shard_end})): {prompt_text[:60]}...")
        torch.cuda.empty_cache()

        row = {
            "prompt_idx": i,
            "prompt_id": prompt_row.get("id"),
            "prompt": prompt_text,
            "topic": prompt_row.get("topic"),
            "conditions": {},
        }
        prompt_features: dict[str, dict] = {}

        for cond_name, input_text, prefix in iter_conditions(prompt_row):
            formatted = format_prompt(tokenizer, input_text)

            cond_entry: dict = {"prefix": prefix, "graphs": {}}
            # For feature comparison we want the MULTI graph's features as
            # the canonical feature set per condition. Stash them here for
            # the later comparison step.
            canonical_features: dict | None = None

            for mode_name, mode_cfg in modes.items():
                # Per-mode target list — filter out positions beyond this
                # specific prompt's length (shorter prompts skip pos=-15 etc.).
                valid_positions, seq_len = _valid_positions_for_prompt(
                    tokenizer, formatted, mode_cfg["positions"],
                )
                if not valid_positions:
                    cond_entry["graphs"][mode_name] = {
                        "error": f"no valid positions for seq_len={seq_len}",
                        "target_positions_requested": list(mode_cfg["positions"]),
                    }
                    print(
                        f"    {cond_name:>22} [{mode_name}]: ERROR — no valid positions"
                    )
                    continue

                targets = [
                    CustomTarget(
                        token_str=f"refusal_L{args.target_layer}_pos{pos:+d}",
                        prob=1.0,
                        vec=per_position_directions[pos],
                    )
                    for pos in valid_positions
                ]
                # Pass a list for multi (len>1) or scalar for single (len==1).
                # The patched attribute() accepts both.
                measurement_positions_arg = (
                    valid_positions if len(valid_positions) > 1
                    else valid_positions[0]
                )

                try:
                    attribute_kwargs = dict(
                        prompt=formatted,
                        model=model,
                        attribution_targets=targets,
                        batch_size=args.batch_size,
                        max_feature_nodes=max_features,
                        measurement_layer=args.target_layer,
                        measurement_position=measurement_positions_arg,
                        verbose=False,
                    )
                    if args.measurement_hook:
                        attribute_kwargs["measurement_hook"] = args.measurement_hook
                    g = attribute(**attribute_kwargs)
                    summary = graph_summary(g)
                    features = extract_all_features(g)
                    summary["n_active"] = len(features)
                    summary["target_positions"] = list(valid_positions)

                    sorted_feats = sorted(
                        features.items(),
                        key=lambda x: abs(x[1]["attribution"]),
                        reverse=True,
                    )[:50]
                    summary["top50_features"] = {
                        k: v["attribution"] for k, v in sorted_feats
                    }

                    cond_entry["graphs"][mode_name] = summary
                    # Canonical features for feature_comparison come from the
                    # multi graph when present; fall back to single otherwise.
                    if mode_name == "multi" or canonical_features is None:
                        canonical_features = features

                    print(
                        f"    {cond_name:>22} [{mode_name}]: net={summary['net']:+.3f}  "
                        f"pos={summary['pos_sum']:.1f}  neg={summary['neg_sum']:.1f}  "
                        f"n_feat={summary['n_features']}  targets={valid_positions}"
                    )

                    if graphs_dir is not None:
                        g.to_pt(str(graphs_dir / f"{i:03d}_{cond_name}_{mode_name}.pt"))
                    del g

                except Exception as e:
                    cond_entry["graphs"][mode_name] = {"error": str(e)[:200]}
                    print(f"    {cond_name:>22} [{mode_name}]: ERROR — {e}")

            row["conditions"][cond_name] = cond_entry
            if canonical_features is not None:
                prompt_features[cond_name] = canonical_features

        # 3-way feature comparison per class: bare↔JB, ctrl↔JB, bare↔ctrl.
        # bare↔JB is the legacy comparison; ctrl↔JB isolates JB semantics from
        # prefix-token-length confounds; bare↔ctrl validates the control (should
        # be small since both refuse).
        bare_feats = prompt_features.get("bare", {})
        if bare_feats:
            prompt_comparison: dict = {}
            for cls in CONTROLLED_CLASSES:
                jb_feats = prompt_features.get(f"jb_{cls}", {})
                ctrl_feats = prompt_features.get(f"ctrl_{cls}", {})
                # Legacy path: condition key was the bare class name (no jb_ prefix).
                if not jb_feats and args.legacy_dataset:
                    jb_feats = prompt_features.get(cls, {})
                entry: dict = {}
                if jb_feats:
                    entry["vs_bare"] = compare_features(bare_feats, jb_feats)
                if jb_feats and ctrl_feats:
                    entry["vs_ctrl"] = compare_features(ctrl_feats, jb_feats)
                if ctrl_feats:
                    entry["ctrl_vs_bare"] = compare_features(bare_feats, ctrl_feats)
                if entry:
                    prompt_comparison[cls] = entry
            row["feature_comparison"] = prompt_comparison

        del prompt_features
        results.append(row)

        # Checkpoint — carry metadata so merge_stage02_shards.py can
        # recover the graph-mode config without having to consult a shard's
        # (mutable, overwritable) final JSON.
        save_json(
            {
                "last_completed": i,
                "results": results,
                "prompts": [
                    p.get("bare") or p.get("base") or p.get("instruction")
                    for p in prompts
                ],
                "metadata": {
                    "measurement_layer": args.target_layer,
                    "measurement_hook": args.measurement_hook or "default",
                    "backend": args.backend,
                    "modes": {
                        name: cfg["positions"] for name, cfg in modes.items()
                    },
                    "dataset": "controlled" if not args.legacy_dataset else "legacy",
                },
            },
            checkpoint_path,
        )

    elapsed = time.time() - t0
    print(f"\n  Attribution complete: {len(results)} prompts in {elapsed/60:.1f} min")

    # Save final results
    final = {
        "metadata": {
            "n_prompts": len(results),
            "model": config.MODEL_NAME,
            "transcoder": config.TRANSCODER_PATH,
            "measurement_layer": args.target_layer,
            "measurement_hook": args.measurement_hook or "default",
                    "backend": args.backend,
            "modes": {name: cfg["positions"] for name, cfg in modes.items()},
            "dataset": "controlled" if not args.legacy_dataset else "legacy",
            "elapsed_minutes": round(elapsed / 60, 1),
        },
        "results": results,
    }
    save_json(final, out_dir / "attribution_results.json")

    # Aggregate feature comparison across the corpus.
    agg = _aggregate_comparison(results, CONTROLLED_CLASSES)
    save_json(agg, out_dir / "feature_comparison_aggregate.json")

    print(f"  Saved to {out_dir}/")
    print("DONE!")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()