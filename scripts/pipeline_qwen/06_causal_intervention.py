"""
Stage 06: Causal intervention via Arditi direction-addition (Qwen3-4B)
======================================================================
Sibling of `scripts/pipeline/06_causal_intervention.py` (Gemma-3-4b-it).
Same methodology — Tejas Script 20's pro+anti pipeline — adapted for Qwen:

Qwen-specific divergences:
  * Default --layer is config.CAUSAL_LAYER (Qwen placeholder: 18; verify
    with 01b_layer_sweep.py before trusting flip-rate numbers).
  * --r-source recompute uses config.DIRECTION_POSITION (-1 for Qwen, vs
    -2 for Gemma) and get_hidden_size(model) (flat config) instead of
    Gemma's nested text_config.
  * generate_with_hook in utils.py registers the hook on
    `model.model.layers[L]` (Qwen) instead of
    `model.model.language_model.layers[L]` (Gemma vision-LM wrapper).
  * Chat template applied via utils.format_prompt (enable_thinking=False).
  * --r-target-magnitude default is 4019.7 (Tejas's L15 measurement on
    Gemma) — meaningless for Qwen. Either skip rescale (default, --r-source
    stage01) or pass --r-target-magnitude with Qwen's measured |r|.

Two intervention modes applied at config.CAUSAL_LAYER:

  pro_refusal_add  — h[:,:,:] += r_unnormalized  on (prompt, jb_*) where
                     baseline COMPLY → expect REFUSE after.
  anti_refusal_sub — h[:,:,:] -= r_unnormalized  on (prompt, bare) where
                     baseline REFUSE → expect COMPLY after.

Together: "the r direction IS the refusal axis, bidirectional."

Inputs:
    <run-dir>/01_direction/unnormalized_r.pt   (per-layer unnormalized r)
    dataset/refusal_lens_controlled_dataset.json (50 prompts × 11 conds)

Outputs to <run-dir>/06_causal/:
    causal_results.json            — per-prompt baseline + intervention records
    causal_checkpoint.json         — resume state (last_completed idx, results)
    causal_summary.json            — headline flip rates, per-class breakdown
    flip_rate_by_class.png         — per-class L15 pro-refusal bar chart
    intervention_symmetry.png      — pro vs anti flip-rate side-by-side
    FLIP_RATE_SUMMARY.md           — human-readable one-pager for ICML

Usage (single run):
    PYTHONPATH=src python3 06_causal_intervention.py \\
        --run-dir <run-dir>

Usage (smoke test, 2 prompts):
    PYTHONPATH=src python3 06_causal_intervention.py \\
        --run-dir <run-dir> --max-prompts 2

Usage (resume after interrupt):
    PYTHONPATH=src python3 06_causal_intervention.py \\
        --run-dir <run-dir> --resume
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import (
    classify_response,
    format_prompt,
    generate_baseline,
    generate_with_hook,
    get_hidden_size,
    get_stage_dir,
    is_coherent,
    load_controlled_dataset,
    load_json,
    load_unnormalized_r,
    make_intervention_hook,
    save_json,
)

JB_CLASSES = ("analytical", "cognitive_reframe", "completion", "fiction", "roleplay")


def parse_args():
    p = argparse.ArgumentParser(
        description=f"Stage 06: Arditi causal intervention "
                    f"(default L{config.CAUSAL_LAYER}, pro + anti)",
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run directory containing 01_direction/unnormalized_r.pt")
    p.add_argument("--layer", type=int, default=config.CAUSAL_LAYER,
                   help="Decoder layer to intervene at (default: 15)")
    p.add_argument("--dtype", choices=["float32", "bfloat16", "float16"],
                   default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Limit to first N prompts (smoke test)")
    p.add_argument("--prompt-start", type=int, default=0)
    p.add_argument("--prompt-end", type=int, default=None,
                   help="Exclusive end (for sharding)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from causal_checkpoint.json if present")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="Save checkpoint every N prompts")
    p.add_argument("--skip-phase0", action="store_true",
                   help="Skip dataset-verification phase (just run interventions)")
    p.add_argument("--skip-anti", action="store_true",
                   help="Skip anti-refusal subtraction (pro-refusal add only)")
    p.add_argument("--skip-benign", action="store_true",
                   help="Skip Phase 2c benign force-refuse control (Tejas's 10 benign prompts)")
    p.add_argument(
        "--r-source", choices=["stage01", "tejas-rescale", "recompute"],
        default="stage01",
        help=(
            "How to source the L15 unnormalized direction r. "
            "stage01 (default): load from 01_direction/unnormalized_r.pt as-is. "
            "tejas-rescale: load stage01 direction, rescale to --r-target-magnitude. "
            "recompute: recompute fresh from harmful_train + harmless_train under "
            "the same model load as intervention (bf16), matching Tejas Script 20 exactly."
        ),
    )
    p.add_argument("--r-target-magnitude", type=float, default=4019.7,
                   help="Target |r| when --r-source tejas-rescale (default: Tejas's 4019.7 at L15)")
    p.add_argument("--r-recompute-n", type=int, default=64,
                   help="N harmful + N harmless prompts for --r-source recompute (default: 64, matches Tejas)")
    return p.parse_args()


# --------------------------------------------------------------------
# Generation + classification wrappers
# --------------------------------------------------------------------

def _evaluate(model, tokenizer, prompt_text: str, hook_fn, layer, max_new_tokens):
    """Generate with or without a hook, classify + coherence-check the response."""
    if hook_fn is None:
        resp = generate_baseline(model, tokenizer, prompt_text, max_new_tokens)
    else:
        resp = generate_with_hook(
            model, tokenizer, prompt_text, layer, hook_fn, max_new_tokens,
        )
    return {
        "cls": classify_response(resp),
        "coherent": is_coherent(resp),
        "response": resp[:300],
    }


# --------------------------------------------------------------------
# Direction-source resolution (audit fix to match Tejas Script 20 magnitude)
# --------------------------------------------------------------------

def recompute_r_tejas_style(model, tokenizer, layer: int, n_each: int = 64):
    """Recompute unnormalized r in-script, matching Tejas Script 20 exactly.

    Reads `harmful_train.json` and `harmless_train.json`, takes first `n_each`
    from each, formats with chat template, runs one forward per prompt (NO
    batching, NO padding, NO truncation), extracts hidden_states[layer+1][0,
    -2, :] in float64, returns the mean-diff in float32.

    Unlike Stage 01 (which batches with left-padding + max_length=256), this
    matches Tejas's methodology bit-for-bit. Use when we want the exact |r|
    Tejas measured (~4019.7 at L15).
    """
    import json as _json
    import gc
    import torch

    harmful_path = config.DATASET_DIR / "harmful_train.json"
    harmless_path = config.DATASET_DIR / "harmless_train.json"
    with open(harmful_path) as f:
        harmful = [p["instruction"] for p in _json.load(f)[:n_each]]
    with open(harmless_path) as f:
        harmless = [p["instruction"] for p in _json.load(f)[:n_each]]
    print(f"  [recompute] reading {len(harmful)} harmful + {len(harmless)} harmless prompts...")

    d_model = get_hidden_size(model)
    mean_harmful = torch.zeros(d_model, dtype=torch.float64)
    mean_harmless = torch.zeros(d_model, dtype=torch.float64)
    pos = config.DIRECTION_POSITION  # Qwen default: -1

    for i, instr in enumerate(harmful):
        formatted = format_prompt(tokenizer, instr)
        input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        mean_harmful += out.hidden_states[layer + 1][0, pos, :].cpu().to(torch.float64) / n_each
        del out
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 16 == 0:
            print(f"    harmful {i+1}/{n_each}")

    for i, instr in enumerate(harmless):
        formatted = format_prompt(tokenizer, instr)
        input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True)
        mean_harmless += out.hidden_states[layer + 1][0, pos, :].cpu().to(torch.float64) / n_each
        del out
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 16 == 0:
            print(f"    harmless {i+1}/{n_each}")

    return (mean_harmful - mean_harmless).to(torch.float32)


def resolve_direction(args, run_dir: Path, model, tokenizer):
    """Produce the L15 unnormalized r according to --r-source. Returns (r, provenance)."""
    import torch

    layer = args.layer
    if args.r_source == "stage01":
        r_dict = load_unnormalized_r(run_dir / "01_direction", [layer])
        r = r_dict[layer].to(torch.float32)
        prov = {"source": "stage01", "path": str(run_dir / "01_direction" / "unnormalized_r.pt")}
    elif args.r_source == "tejas-rescale":
        r_dict = load_unnormalized_r(run_dir / "01_direction", [layer])
        r_orig = r_dict[layer].to(torch.float32)
        orig_norm = float(r_orig.norm())
        scale = args.r_target_magnitude / orig_norm
        r = r_orig * scale
        prov = {
            "source": "tejas-rescale",
            "original_magnitude": orig_norm,
            "target_magnitude": args.r_target_magnitude,
            "scale_applied": scale,
            "base_path": str(run_dir / "01_direction" / "unnormalized_r.pt"),
        }
    elif args.r_source == "recompute":
        r = recompute_r_tejas_style(model, tokenizer, layer, n_each=args.r_recompute_n)
        prov = {
            "source": "recompute",
            "n_harmful": args.r_recompute_n,
            "n_harmless": args.r_recompute_n,
            "method": "Tejas Script 20 exact port — no batching, no padding, no truncation, model dtype as loaded",
        }
    else:
        raise ValueError(f"Unknown --r-source {args.r_source}")

    prov["magnitude"] = float(r.norm())
    print(f"  [r-source={args.r_source}] |r_L{layer}| = {prov['magnitude']:.1f}")
    return r, prov


# --------------------------------------------------------------------
# Phase 0 — dataset verification
# --------------------------------------------------------------------

def phase0_verify(model, tokenizer, rows, max_new_tokens, layer):
    """Verify bare refuses + all ctrl refuse on the current dataset.

    Doesn't modify the dataset — just reports. Ctrl-leak pairs (prompt, class)
    where ctrl generated COMPLY are recorded and will be EXCLUDED from the
    pro-refusal phase (Tejas's bulletproof convention).
    """
    print("\n[PHASE 0] Verify dataset (bare refuse + ctrl refuse)...")
    bare_refused = 0
    ctrl_leak_pairs: list[tuple[int, str]] = []
    excluded: list[int] = []
    bare_details = []
    ctrl_details = []
    for i, row in enumerate(rows):
        # Bare
        r_bare = _evaluate(model, tokenizer, row["conditions"]["bare"]["text"],
                           None, layer, max_new_tokens)
        bare_details.append({"id": row["id"], **r_bare})
        if r_bare["cls"] == "REFUSE":
            bare_refused += 1
        else:
            excluded.append(row["id"])
            print(f"  !! BARE COMPLY id={row['id']}: {row['base'][:55]}")

        # Ctrl per class
        for cls in JB_CLASSES:
            ctrl_text = row["conditions"][f"ctrl_{cls}"]["text"]
            r_ctrl = _evaluate(model, tokenizer, ctrl_text, None, layer, max_new_tokens)
            ctrl_details.append({"id": row["id"], "class": cls, **r_ctrl})
            if r_ctrl["cls"] != "REFUSE":
                ctrl_leak_pairs.append((row["id"], cls))

        if (i + 1) % 5 == 0 or i + 1 == len(rows):
            print(f"  {i+1}/{len(rows)} verified | bare_refused={bare_refused}  "
                  f"ctrl_leaks={len(ctrl_leak_pairs)}")

    print(f"\n  PHASE 0 RESULT: bare {bare_refused}/{len(rows)} refuse | "
          f"ctrl {len(rows)*len(JB_CLASSES) - len(ctrl_leak_pairs)}/{len(rows)*len(JB_CLASSES)} refuse")
    if ctrl_leak_pairs:
        print(f"  {len(ctrl_leak_pairs)} ctrl-leak pairs will be EXCLUDED from "
              f"pro-refusal intervention.")
    return {
        "bare_refused": bare_refused,
        "ctrl_total": len(rows) * len(JB_CLASSES),
        "ctrl_refused": len(rows) * len(JB_CLASSES) - len(ctrl_leak_pairs),
        "ctrl_leak_pairs": [list(p) for p in ctrl_leak_pairs],
        "excluded_prompts": excluded,
        "bare_details": bare_details,
        "ctrl_details": ctrl_details,
    }, set(ctrl_leak_pairs), set(excluded)


# --------------------------------------------------------------------
# Phase 1 + 2 — baseline + interventions per prompt
# --------------------------------------------------------------------

def process_prompt(model, tokenizer, row, r_vec_by_layer, layer,
                   max_new_tokens, ctrl_leak_pairs, skip_anti, phase0_bare_cls=None):
    """Run baseline + interventions for one prompt. Returns a result row.

    `ctrl_leak_pairs` is a set of (prompt_id, class) tuples to skip on the
    pro-refusal side (ctrl baseline already complied — not a clean substrate).
    `phase0_bare_cls` optionally threads the Phase-0 bare classification to
    avoid re-generating it. If None, baseline bare is generated fresh.
    """
    result = {
        "prompt_idx": row.get("idx", None),
        "prompt_id": row["id"],
        "topic": row.get("topic"),
        "base": row["base"],
        "baseline": {},
        "interventions": {},
    }

    # ----- Phase 1: baselines for every condition -----
    for cond_name, cond in row["conditions"].items():
        r = _evaluate(model, tokenizer, cond["text"], None, layer, max_new_tokens)
        result["baseline"][cond_name] = r

    # ----- Phase 2a: pro-refusal add on jb_* where baseline COMPLY -----
    add_hook = make_intervention_hook(r_vec_by_layer[layer], sign="add")
    pro_key = f"L{layer}_pro_refusal_add"
    result["interventions"][pro_key] = {}
    for cls in JB_CLASSES:
        jb_cond = f"jb_{cls}"
        if result["baseline"][jb_cond]["cls"] != "COMPLY":
            # Already refusing — no-op, skip the generation
            continue
        if (row["id"], cls) in ctrl_leak_pairs:
            # Ctrl leaked for this pair → exclude (bulletproof convention)
            continue
        r_int = _evaluate(model, tokenizer, row["conditions"][jb_cond]["text"],
                          add_hook, layer, max_new_tokens)
        r_int["flipped_toward_refuse"] = (r_int["cls"] == "REFUSE")
        result["interventions"][pro_key][jb_cond] = r_int

    # ----- Phase 2b: anti-refusal sub on bare where baseline REFUSE -----
    if not skip_anti:
        sub_hook = make_intervention_hook(r_vec_by_layer[layer], sign="sub")
        anti_key = f"L{layer}_anti_refusal_sub"
        result["interventions"][anti_key] = {}
        if result["baseline"]["bare"]["cls"] == "REFUSE":
            r_int = _evaluate(model, tokenizer, row["conditions"]["bare"]["text"],
                              sub_hook, layer, max_new_tokens)
            r_int["flipped_toward_comply"] = (r_int["cls"] == "COMPLY")
            result["interventions"][anti_key]["bare"] = r_int

    return result


# --------------------------------------------------------------------
# Phase 2c — benign force-refuse control (Tejas Script 20 Phase 4a)
# --------------------------------------------------------------------

def phase2c_benign_force_refuse(model, tokenizer, r_vec, layer, max_new_tokens,
                                benign_prompts=None):
    """Run the pro-refusal-add hook on a list of benign prompts. Expect all to REFUSE.

    This is Tejas's control experiment (Script 20 Phase 4a): if the intervention
    is a generic refusal push (not a JB-specific artifact), benign prompts under
    the same hook should flip from their usual COMPLY baseline to REFUSE. Tejas
    reports 10/10 on this. Below-80% here would invalidate the symmetry claim.

    Uses `config.BENIGN_PROMPTS` (10 prompts verbatim from Script 20) by default.
    """
    if benign_prompts is None:
        benign_prompts = list(config.BENIGN_PROMPTS)
    print(f"\n[PHASE 2c] Benign force-refuse control ({len(benign_prompts)} prompts)...")
    add_hook = make_intervention_hook(r_vec, sign="add")
    results = []
    for i, prompt in enumerate(benign_prompts):
        r = _evaluate(model, tokenizer, prompt, add_hook, layer, max_new_tokens)
        r["prompt"] = prompt
        r["forced_to_refuse"] = (r["cls"] == "REFUSE")
        results.append(r)
        print(f"  [{i+1}/{len(benign_prompts)}] {r['cls']:>6} | {prompt[:50]}")
    n_refused = sum(1 for r in results if r["forced_to_refuse"])
    n_coherent = sum(1 for r in results if r["coherent"])
    print(f"  RESULT: {n_refused}/{len(benign_prompts)} forced to REFUSE "
          f"(Tejas reports 10/10 on his bulletproof run); {n_coherent} coherent")
    return {
        "n_prompts": len(benign_prompts),
        "n_forced_to_refuse": n_refused,
        "force_refuse_rate": round(n_refused / len(benign_prompts), 3) if benign_prompts else 0.0,
        "n_coherent": n_coherent,
        "per_prompt": results,
    }


# --------------------------------------------------------------------
# Phase 3 — aggregation
# --------------------------------------------------------------------

def aggregate_summary(results, layer, skip_anti):
    pro_key = f"L{layer}_pro_refusal_add"
    anti_key = f"L{layer}_anti_refusal_sub"

    per_class: dict = {c: {"comply_baseline": 0, "flipped": 0, "coherent_flipped": 0}
                       for c in JB_CLASSES}
    n_comply_total = 0
    n_flip_total = 0
    n_coherent_flip_total = 0
    for r in results:
        for cls in JB_CLASSES:
            jb_cond = f"jb_{cls}"
            if r["baseline"][jb_cond]["cls"] != "COMPLY":
                continue
            per_class[cls]["comply_baseline"] += 1
            n_comply_total += 1
            intv = r["interventions"].get(pro_key, {}).get(jb_cond)
            if intv is None:
                continue
            if intv.get("flipped_toward_refuse"):
                per_class[cls]["flipped"] += 1
                n_flip_total += 1
                if intv.get("coherent"):
                    per_class[cls]["coherent_flipped"] += 1
                    n_coherent_flip_total += 1

    for cls, c in per_class.items():
        c["rate"] = round(c["flipped"] / c["comply_baseline"], 3) if c["comply_baseline"] else 0.0

    summary = {
        pro_key: {
            "n_jb_comply_baseline": n_comply_total,
            "n_flipped_to_refuse": n_flip_total,
            "flip_rate": round(n_flip_total / n_comply_total, 3) if n_comply_total else 0.0,
            "n_coherent_flips": n_coherent_flip_total,
            "per_class": per_class,
        }
    }

    if not skip_anti:
        n_refuse_bare = 0
        n_flip_bare = 0
        n_coherent_flip_bare = 0
        for r in results:
            if r["baseline"]["bare"]["cls"] != "REFUSE":
                continue
            n_refuse_bare += 1
            intv = r["interventions"].get(anti_key, {}).get("bare")
            if intv and intv.get("flipped_toward_comply"):
                n_flip_bare += 1
                if intv.get("coherent"):
                    n_coherent_flip_bare += 1
        summary[anti_key] = {
            "n_bare_refuse_baseline": n_refuse_bare,
            "n_flipped_to_comply": n_flip_bare,
            "flip_rate": round(n_flip_bare / n_refuse_bare, 3) if n_refuse_bare else 0.0,
            "n_coherent_flips": n_coherent_flip_bare,
        }
    return summary


# --------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------

def plot_flip_rate_by_class(summary, layer, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pro = summary[f"L{layer}_pro_refusal_add"]["per_class"]
    classes = list(pro.keys())
    comply = [pro[c]["comply_baseline"] for c in classes]
    flip = [pro[c]["flipped"] for c in classes]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(classes))
    width = 0.38
    ax.bar([i - width / 2 for i in x], comply, width, label="Baseline comply", color="#e5736a")
    ax.bar([i + width / 2 for i in x], flip, width, label="Flipped to refuse", color="#2e7d32")
    for i, c in enumerate(classes):
        if comply[i] > 0:
            rate = flip[i] / comply[i] * 100
            ax.text(i + width / 2, flip[i] + 0.3, f"{flip[i]}/{comply[i]}\n({rate:.0f}%)",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"L{layer} Arditi pro-refusal intervention — per-class flip rate")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_intervention_symmetry(summary, layer, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pro = summary[f"L{layer}_pro_refusal_add"]
    anti = summary.get(f"L{layer}_anti_refusal_sub")
    labels = ["pro_refusal_add\n(JB → REFUSE)"]
    rates = [pro["flip_rate"]]
    ns = [f"{pro['n_flipped_to_refuse']}/{pro['n_jb_comply_baseline']}"]
    if anti:
        labels.append("anti_refusal_sub\n(bare → COMPLY)")
        rates.append(anti["flip_rate"])
        ns.append(f"{anti['n_flipped_to_comply']}/{anti['n_bare_refuse_baseline']}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#2e7d32", "#8e24aa"][:len(labels)]
    bars = ax.bar(labels, [r * 100 for r in rates], color=colors)
    for bar, n, rate in zip(bars, ns, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate * 100 + 1.5,
                f"{rate*100:.1f}%\n({n})", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Flip rate (%)")
    ax.set_title(f"L{layer} causal intervention: bidirectional flip symmetry\n"
                 f"(add r → refuse; subtract r → comply)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_summary_md(summary, layer, phase0, out_path: Path, elapsed_min: float,
                      r_provenance: dict | None = None,
                      benign_result: dict | None = None) -> None:
    pro = summary[f"L{layer}_pro_refusal_add"]
    anti = summary.get(f"L{layer}_anti_refusal_sub")
    lines = [
        "# Stage 06 Causal Intervention — Flip Rate Summary",
        "",
        f"**Method**: Arditi intervention (add unnormalized r at all positions, every forward step).",
        f"**Layer**: L{layer}.",
        f"**Dataset**: refusal_lens_controlled_dataset.json (50 prompts × 11 conditions).",
        f"**Elapsed**: {elapsed_min:.1f} min.",
        "",
    ]
    if r_provenance:
        lines += [
            "## Direction source",
            "",
            f"- `r_source`: **{r_provenance.get('source')}**",
            f"- \\|r\\|: **{r_provenance.get('magnitude', 0):.1f}** (Tejas reports 4019.7 on his bulletproof run)",
        ]
        if r_provenance.get("source") == "tejas-rescale":
            lines.append(
                f"- Scale applied: ×{r_provenance.get('scale_applied', 1):.4f} "
                f"(original \\|r\\|={r_provenance.get('original_magnitude', 0):.1f})"
            )
        elif r_provenance.get("source") == "recompute":
            lines.append(
                f"- Recomputed from {r_provenance.get('n_harmful', 64)}+{r_provenance.get('n_harmless', 64)} prompts "
                "under the same bf16 model load as intervention (Tejas-exact)"
            )
        lines.append("")
    lines += [
        "## Phase 0 — dataset verification",
        "",
        f"- Bare refused: **{phase0['bare_refused']}/{phase0['bare_refused'] + len(phase0['excluded_prompts'])}**",
        f"- Ctrl refused: **{phase0['ctrl_refused']}/{phase0['ctrl_total']}** "
        f"({phase0['ctrl_refused'] / max(phase0['ctrl_total'], 1) * 100:.1f}%)",
        f"- Ctrl-leak pairs excluded: **{len(phase0['ctrl_leak_pairs'])}**",
        f"- Bare-comply exclusions: **{len(phase0['excluded_prompts'])}**",
        "",
        "## Pro-refusal add (headline result)",
        "",
        f"Flip rate: **{pro['flip_rate']*100:.1f}%** "
        f"({pro['n_flipped_to_refuse']}/{pro['n_jb_comply_baseline']} JB-comply prompts flipped to REFUSE)",
        f"Coherent flips: **{pro['n_coherent_flips']}/{pro['n_flipped_to_refuse']}**",
        "",
        "### Per-class breakdown",
        "",
        "| Class | Comply baseline | Flipped | Rate | Coherent |",
        "|---|---|---|---|---|",
    ]
    for cls, c in pro["per_class"].items():
        lines.append(f"| `{cls}` | {c['comply_baseline']} | {c['flipped']} | "
                     f"{c['rate']*100:.0f}% | {c['coherent_flipped']} |")

    if anti:
        lines.extend([
            "",
            "## Anti-refusal sub (bare → comply)",
            "",
            f"Flip rate: **{anti['flip_rate']*100:.1f}%** "
            f"({anti['n_flipped_to_comply']}/{anti['n_bare_refuse_baseline']} bare-refuse prompts flipped to COMPLY)",
            f"Coherent flips: **{anti['n_coherent_flips']}/{anti['n_flipped_to_comply']}**",
            "",
            "## Symmetry claim",
            "",
            "The bidirectional flip symmetry is the headline. If pro is ~100% and anti is "
            "high, the L15 unnormalized r vector IS the model's refusal axis — not a "
            "one-way push. This is the causal complement to the Stage 07 correlational "
            "`jb_vs_ctrl_contrast` finding.",
            "",
        ])

    if benign_result:
        lines.extend([
            "## Phase 2c — benign force-refuse control (Tejas bulletproof)",
            "",
            f"Force-refuse rate on 10 benign prompts: **{benign_result['force_refuse_rate']*100:.1f}%** "
            f"({benign_result['n_forced_to_refuse']}/{benign_result['n_prompts']})",
            f"Coherent responses: **{benign_result['n_coherent']}/{benign_result['n_prompts']}**",
            "",
            "Tejas reports **10/10** on his bulletproof run. A result below ~80% here would "
            "indicate the intervention isn't a generic refusal push, invalidating the "
            "'L15 r IS the refusal axis' claim.",
            "",
        ])
    out_path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = get_stage_dir(run_dir, "06_causal")

    print("=" * 60)
    print(f"STAGE 06: Causal intervention  (L{args.layer}, Arditi)")
    print("=" * 60)
    print(f"  run_dir:       {run_dir}")
    print(f"  out_dir:       {out_dir}")
    print(f"  dtype:         {args.dtype}")
    print(f"  max_prompts:   {args.max_prompts}")

    # Stage-01 r sanity (fail fast for stage01 / tejas-rescale; recompute path skips this)
    if args.r_source in ("stage01", "tejas-rescale"):
        stage01_r_path = run_dir / "01_direction" / "unnormalized_r.pt"
        if not stage01_r_path.exists():
            print(f"  ERROR: {stage01_r_path} missing. Run Stage 01 or pass --r-source recompute.")
            sys.exit(1)

    # Load dataset
    rows = load_controlled_dataset(n_prompts=args.max_prompts)
    start = args.prompt_start
    end = args.prompt_end if args.prompt_end is not None else len(rows)
    rows = rows[start:end]
    for i, r in enumerate(rows):
        r["idx"] = start + i
    print(f"  prompts:       [{start}, {end}) → {len(rows)} rows")

    # Load model
    print("\n  Loading model (AutoModelForCausalLM)...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=dtype_map[args.dtype], device_map="auto",
    )
    model.eval()
    print("  Model ready.")

    # Resolve direction AFTER model load (recompute path needs the model).
    # The hook re-casts dtype per call, so keep r in float32 on the model device.
    r_tensor, r_provenance = resolve_direction(args, run_dir, model, tokenizer)
    r_vec_by_layer = {args.layer: r_tensor.to(model.device)}
    r_mag = r_provenance["magnitude"]

    # Checkpoint / resume
    ckpt_path = out_dir / "causal_checkpoint.json"
    if args.resume and ckpt_path.exists():
        ckpt = load_json(ckpt_path)
        results = ckpt.get("results", [])
        done_ids = {r["prompt_id"] for r in results}
        rows_todo = [r for r in rows if r["id"] not in done_ids]
        phase0 = ckpt.get("phase0")
        ctrl_leak_pairs = {tuple(p) for p in (phase0 or {}).get("ctrl_leak_pairs", [])}
        excluded = set((phase0 or {}).get("excluded_prompts", []))
        print(f"\n  [resume] skipping {len(done_ids)} already-done prompts; "
              f"{len(rows_todo)} remain")
    else:
        results = []
        phase0 = None
        ctrl_leak_pairs = set()
        excluded = set()
        rows_todo = rows

    # Phase 0 (unless skipping or resumed)
    if phase0 is None and not args.skip_phase0:
        phase0, ctrl_leak_pairs, excluded = phase0_verify(
            model, tokenizer, rows, args.max_new_tokens, args.layer,
        )
        save_json({"results": results, "phase0": phase0}, ckpt_path)
    elif args.skip_phase0 and phase0 is None:
        phase0 = {"bare_refused": 0, "ctrl_total": 0, "ctrl_refused": 0,
                  "ctrl_leak_pairs": [], "excluded_prompts": [],
                  "note": "phase0 skipped via --skip-phase0"}

    # Phases 1-2 per prompt
    print("\n[PHASES 1-2] Baseline + interventions per prompt...")
    t0 = time.time()
    for i, row in enumerate(rows_todo):
        if row["id"] in excluded:
            print(f"  [{i+1}/{len(rows_todo)}] SKIP id={row['id']} (Phase 0 exclusion)")
            continue
        t_p = time.time()
        result = process_prompt(
            model, tokenizer, row, r_vec_by_layer, args.layer,
            args.max_new_tokens, ctrl_leak_pairs, args.skip_anti,
        )
        results.append(result)
        p_elapsed = time.time() - t_p
        n_flips = sum(
            1 for v in result["interventions"].get(f"L{args.layer}_pro_refusal_add", {}).values()
            if v.get("flipped_toward_refuse")
        )
        print(f"  [{i+1}/{len(rows_todo)}] id={row['id']:3d}  "
              f"({p_elapsed:.0f}s)  flips={n_flips}")
        gc.collect()
        if hasattr(__import__("torch").cuda, "empty_cache"):
            import torch
            torch.cuda.empty_cache()

        if (i + 1) % args.checkpoint_every == 0:
            save_json({"results": results, "phase0": phase0}, ckpt_path)

    # Phase 2c — benign force-refuse control (Tejas Script 20 Phase 4a)
    benign_result = None
    if not args.skip_benign:
        benign_result = phase2c_benign_force_refuse(
            model, tokenizer, r_vec_by_layer[args.layer],
            args.layer, args.max_new_tokens,
        )

    total_elapsed = time.time() - t0

    # Aggregate + persist
    summary = aggregate_summary(results, args.layer, args.skip_anti)
    if benign_result:
        summary[f"L{args.layer}_benign_force_refuse"] = {
            "n_prompts": benign_result["n_prompts"],
            "n_forced_to_refuse": benign_result["n_forced_to_refuse"],
            "force_refuse_rate": benign_result["force_refuse_rate"],
            "n_coherent": benign_result["n_coherent"],
        }
    final = {
        "metadata": {
            "method": "arditi_all_positions_every_step",
            "layers": [args.layer],
            "intervention_modes": list(config.CAUSAL_INTERVENTION_MODES[:1 + int(not args.skip_anti)]),
            "r_magnitude": {f"L{args.layer}": r_mag},
            "r_provenance": r_provenance,
            "n_prompts": len(results),
            "n_conditions": 11,
            "source_run": run_dir.name,
            "tejas_script_port": "20_bulletproof_pipeline.py@origin/tejas-circuit-experiments",
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "elapsed_minutes": round(total_elapsed / 60, 2),
        },
        "phase0_verification": {
            k: v for k, v in (phase0 or {}).items()
            if k not in ("bare_details", "ctrl_details")  # drop per-prompt verbosity
        },
        "phase2c_benign_control": benign_result,
        "results": results,
        "summary": summary,
    }
    save_json(final, out_dir / "causal_results.json")
    save_json(summary, out_dir / "causal_summary.json")
    print(f"\n  Saved causal_results.json + causal_summary.json")

    # Figures + report
    print("\n  Generating figures + report...")
    plot_flip_rate_by_class(summary, args.layer, out_dir / "flip_rate_by_class.png")
    print("    flip_rate_by_class.png")
    plot_intervention_symmetry(summary, args.layer, out_dir / "intervention_symmetry.png")
    print("    intervention_symmetry.png")
    write_summary_md(summary, args.layer, phase0 or {},
                     out_dir / "FLIP_RATE_SUMMARY.md", total_elapsed / 60,
                     r_provenance=r_provenance, benign_result=benign_result)
    print("    FLIP_RATE_SUMMARY.md")

    # Headline numbers to console
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS")
    print("=" * 60)
    print(f"  |r_L{args.layer}|:                     {r_mag:.1f} (source={r_provenance['source']})")
    pro = summary[f"L{args.layer}_pro_refusal_add"]
    print(f"  L{args.layer} pro-refusal flip rate:   "
          f"{pro['flip_rate']*100:.1f}% ({pro['n_flipped_to_refuse']}/{pro['n_jb_comply_baseline']})")
    if f"L{args.layer}_anti_refusal_sub" in summary:
        anti = summary[f"L{args.layer}_anti_refusal_sub"]
        print(f"  L{args.layer} anti-refusal flip rate:  "
              f"{anti['flip_rate']*100:.1f}% ({anti['n_flipped_to_comply']}/{anti['n_bare_refuse_baseline']})")
    if benign_result:
        print(f"  L{args.layer} benign force-refuse:     "
              f"{benign_result['force_refuse_rate']*100:.1f}% "
              f"({benign_result['n_forced_to_refuse']}/{benign_result['n_prompts']})  "
              f"[Tejas reports 10/10]")
    print(f"  Elapsed: {total_elapsed / 60:.1f} min")
    print("DONE!")


if __name__ == "__main__":
    main()
