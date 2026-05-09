"""Per-class jailbreak-vector intervention (rigorous follow-up to v1 § 5.6).

Builds per-class jailbreak vectors r_jb_class = mean(h_jb_class) - mean(h_bare)
at L15 pos=-2 (Ball 2024 / Wang 2025 convention: points TOWARD jailbreak).
Then runs the same two interventions as the universal-vector script, but
class-by-class:

  Experiment A — mitigate JB by subtracting r_jb_class:
    For each (prompt, jb_C) where Stage 06 baseline = COMPLY,
    SUBTRACT r_jb_C (the SAME class's vector) from the residual at L15.
    Tests whether the class-specific direction recovers more JB-comply
    flips than the universal mean (47/89 = 52.8 %).

  Experiment B — induce JB by adding r_jb_class:
    For each bare prompt where Stage 06 baseline = REFUSE, ADD r_jb_C
    (each of the 5 class vectors in turn). Tests which class's empirical
    JB direction best induces jailbreak on bare-refuse, and quantifies
    whether the universal-vector 16 % is a basis-quality issue or a
    dose issue.

Two magnitude conditions per class:
  - empirical: use r_jb_C at its native magnitude (range 0.40 - 1.11 |r̂|)
  - matched:   scale r_jb_C to |r̂| (1.00 |r̂|), preserving direction only
               (lets us discriminate "directional alignment" from "dose")

Output: 06_causal/jb_vector_intervention_per_class_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from utils import (  # noqa: E402
    classify_response,
    format_prompt,
    is_coherent,
    load_controlled_dataset,
    make_intervention_hook,
)

LAYER = 15
CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]
TARGET_POSITIONS = [-5, -3, -2]
POS_IDX = TARGET_POSITIONS.index(-2)
MAG_CONDITIONS = ["empirical", "matched"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--mag-conditions", default="empirical,matched",
                   help="Comma-separated subset of: empirical,matched")
    return p.parse_args()


def main():
    args = parse_args()
    mag_conds = [c.strip() for c in args.mag_conditions.split(",") if c.strip()]
    for mc in mag_conds:
        assert mc in MAG_CONDITIONS, f"unknown mag condition: {mc}"

    print("[per-class] loading r̂[L15]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float().cpu()
    r_hat_norm = r_hat.norm().item()
    print(f"  |r̂[L15]| = {r_hat_norm:.2f}")

    print("[per-class] loading saved residuals from §5.5")
    R = torch.load(args.run_dir / "02b_stats/residuals_L15_per_cond.pt", weights_only=False)

    print("[per-class] computing per-class r_jb vectors at pos=−2")
    mu_bare = R["bare"][:, POS_IDX, :].float().mean(dim=0)
    r_jb_per_class = {}
    for cls in CLASSES:
        mu_j = R[f"jb_{cls}"][:, POS_IDX, :].float().mean(dim=0)
        v = mu_j - mu_bare  # Ball convention: points toward jailbreak
        r_jb_per_class[cls] = v
        print(f"  {cls:22s} |r_jb|/|r̂|={v.norm().item()/r_hat_norm:.3f}  "
              f"cos(r̂)={torch.nn.functional.cosine_similarity(v.unsqueeze(0), r_hat.unsqueeze(0)).item():+.4f}  "
              f"cos(-r̂)={torch.nn.functional.cosine_similarity(v.unsqueeze(0), -r_hat.unsqueeze(0)).item():+.4f}")

    print("[per-class] loading Stage 06 baselines")
    causal = json.loads((args.run_dir / "06_causal/causal_results.json").read_text())
    baselines = {}
    for r in causal["results"]:
        baselines[r["prompt_idx"]] = {c: blob["cls"] for c, blob in r["baseline"].items()}

    print(f"[per-class] loading model {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    L_module = layers[LAYER]
    print(f"  model loaded in {time.time()-t0:.1f}s")

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    pad_id = tokenizer.eos_token_id

    def gen_with_hook(text, hook):
        formatted = format_prompt(tokenizer, text)
        ids = tokenizer(formatted, return_tensors="pt").to(model.device)
        prompt_len = ids.input_ids.shape[1]
        handle = L_module.register_forward_hook(hook)
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
        return resp

    # Pre-compute prompt index lists per class
    jb_comply_idx = {cls: [] for cls in CLASSES}
    bare_refuse_idx = []
    for prompt_idx in baselines:
        if baselines[prompt_idx].get("bare") == "REFUSE":
            bare_refuse_idx.append(prompt_idx)
        for cls in CLASSES:
            if baselines[prompt_idx].get(f"jb_{cls}") == "COMPLY":
                jb_comply_idx[cls].append(prompt_idx)
    print(f"[per-class] jb-comply per class: "
          f"{ {cls: len(jb_comply_idx[cls]) for cls in CLASSES} }")
    print(f"[per-class] bare-refuse: {len(bare_refuse_idx)}")

    results = {
        "metadata": {
            "method": "per_class_jb_vector_intervention_at_L15_all_positions_every_step",
            "layer": LAYER,
            "position_used_for_construction": -2,
            "r_hat_norm": r_hat_norm,
            "construction": ("r_jb_class = mean(h_jb_class) - mean(h_bare) at pos=-2; "
                             "Ball 2024 / Wang 2025 convention (points TOWARD jailbreak)."),
            "mag_conditions": mag_conds,
            "max_new_tokens": args.max_new_tokens,
            "n_prompts": len(dataset),
            "per_class_r_jb_norms_over_r_hat": {
                cls: v.norm().item() / r_hat_norm for cls, v in r_jb_per_class.items()
            },
            "per_class_cos_r_hat_r_jb_class": {
                cls: torch.nn.functional.cosine_similarity(
                    v.unsqueeze(0), r_hat.unsqueeze(0)).item()
                for cls, v in r_jb_per_class.items()
            },
            "per_class_cos_neg_r_hat_r_jb_class": {
                cls: torch.nn.functional.cosine_similarity(
                    v.unsqueeze(0), -r_hat.unsqueeze(0)).item()
                for cls, v in r_jb_per_class.items()
            },
        },
        "experiment_a_mitigate_jb_subtract_per_class_rjb": {},
        "experiment_b_induce_jb_add_per_class_rjb": {},
    }

    t_total = time.time()

    for mag_cond in mag_conds:
        print("\n" + "#" * 80)
        print(f"# Magnitude condition: {mag_cond}")
        print("#" * 80)

        # Build per-class GPU vectors at the chosen magnitude
        r_jb_gpu_by_class = {}
        for cls in CLASSES:
            v = r_jb_per_class[cls]
            if mag_cond == "matched":
                v = v / v.norm() * r_hat_norm
            r_jb_gpu_by_class[cls] = v.to(model.device, dtype=torch.bfloat16)

        # Experiment A: per-class — subtract r_jb_C from jb_C-comply prompts
        print(f"\n--- Experiment A ({mag_cond}): mitigate JB by subtracting r_jb_class on jb-comply ---")
        results["experiment_a_mitigate_jb_subtract_per_class_rjb"].setdefault(mag_cond, {})
        n_total_a = 0
        n_flipped_a_total = 0
        t_a = time.time()
        for cls in CLASSES:
            sub_hook = make_intervention_hook(r_jb_gpu_by_class[cls], sign="sub")
            per_prompt_a = []
            n_flipped_cls = 0
            for prompt_idx in jb_comply_idx[cls]:
                text = dataset[prompt_idx]["conditions"][f"jb_{cls}"]["text"]
                resp = gen_with_hook(text, sub_hook)
                new_cls = classify_response(resp)
                coh = is_coherent(resp)
                flipped = new_cls == "REFUSE"
                if flipped:
                    n_flipped_cls += 1
                    n_flipped_a_total += 1
                per_prompt_a.append({
                    "prompt_idx": prompt_idx, "condition": f"jb_{cls}",
                    "baseline_cls": "COMPLY", "intervened_cls": new_cls,
                    "flipped_to_refuse": flipped, "coherent": coh,
                    "response_truncated": resp[:300],
                })
                n_total_a += 1
            n_cls = len(jb_comply_idx[cls])
            rate = n_flipped_cls / n_cls if n_cls else 0
            results["experiment_a_mitigate_jb_subtract_per_class_rjb"][mag_cond][cls] = {
                "n_baseline_comply": n_cls,
                "n_flipped_to_refuse": n_flipped_cls,
                "flip_rate": rate,
                "per_prompt": per_prompt_a,
            }
            print(f"  {cls:22s} {n_flipped_cls}/{n_cls} = {rate*100:5.1f}%  "
                  f"|r_jb|={r_jb_gpu_by_class[cls].norm().item():.1f} "
                  f"({r_jb_gpu_by_class[cls].norm().item()/r_hat_norm:.2f}·|r̂|)")
        rate_a_total = n_flipped_a_total / n_total_a if n_total_a else 0
        results["experiment_a_mitigate_jb_subtract_per_class_rjb"][mag_cond]["overall"] = {
            "n_baseline_comply": n_total_a,
            "n_flipped_to_refuse": n_flipped_a_total,
            "flip_rate": rate_a_total,
        }
        print(f"  {'OVERALL':22s} {n_flipped_a_total}/{n_total_a} = {rate_a_total*100:5.1f}%  "
              f"({(time.time()-t_a)/60:.1f} min)")

        # Experiment B: per-class — add r_jb_C to bare-refuse prompts
        print(f"\n--- Experiment B ({mag_cond}): induce JB by adding r_jb_class on bare-refuse ---")
        results["experiment_b_induce_jb_add_per_class_rjb"].setdefault(mag_cond, {})
        t_b = time.time()
        for cls in CLASSES:
            add_hook = make_intervention_hook(r_jb_gpu_by_class[cls], sign="add")
            per_prompt_b = []
            n_flipped_cls = 0
            for prompt_idx in bare_refuse_idx:
                text = dataset[prompt_idx]["conditions"]["bare"]["text"]
                resp = gen_with_hook(text, add_hook)
                new_cls = classify_response(resp)
                coh = is_coherent(resp)
                flipped = new_cls == "COMPLY"
                if flipped:
                    n_flipped_cls += 1
                per_prompt_b.append({
                    "prompt_idx": prompt_idx, "condition": "bare",
                    "baseline_cls": "REFUSE", "intervened_cls": new_cls,
                    "flipped_to_comply": flipped, "coherent": coh,
                    "rjb_source_class": cls,
                    "response_truncated": resp[:300],
                })
            n_b = len(bare_refuse_idx)
            rate = n_flipped_cls / n_b if n_b else 0
            results["experiment_b_induce_jb_add_per_class_rjb"][mag_cond][cls] = {
                "n_baseline_refuse": n_b,
                "n_flipped_to_comply": n_flipped_cls,
                "flip_rate": rate,
                "rjb_source_class": cls,
                "per_prompt": per_prompt_b,
            }
            print(f"  {cls:22s} {n_flipped_cls}/{n_b} = {rate*100:5.1f}%  "
                  f"|r_jb|={r_jb_gpu_by_class[cls].norm().item():.1f} "
                  f"({r_jb_gpu_by_class[cls].norm().item()/r_hat_norm:.2f}·|r̂|)  "
                  f"({(time.time()-t_b)/60:.1f} min so far)")

    out_path = args.run_dir / "06_causal/jb_vector_intervention_per_class_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[per-class] wrote {out_path.relative_to(REPO)}")
    print(f"[per-class] total wall: {(time.time()-t_total)/60:.1f} min")

    print("\n" + "=" * 90)
    print("HEADLINE TABLE — per-class flip rates")
    print("=" * 90)
    for mag_cond in mag_conds:
        print(f"\n[{mag_cond} magnitude]")
        print(f"{'class':22s} {'mag/|r̂|':>10s} {'cos(-r̂)':>10s} "
              f"{'Exp A flip':>12s} {'Exp B flip':>12s}")
        print("-" * 90)
        for cls in CLASSES:
            v = r_jb_per_class[cls]
            if mag_cond == "matched":
                mag_ratio = 1.0
            else:
                mag_ratio = v.norm().item() / r_hat_norm
            cos_neg = torch.nn.functional.cosine_similarity(
                v.unsqueeze(0), -r_hat.unsqueeze(0)).item()
            a_block = results["experiment_a_mitigate_jb_subtract_per_class_rjb"][mag_cond][cls]
            b_block = results["experiment_b_induce_jb_add_per_class_rjb"][mag_cond][cls]
            a_str = (f"{a_block['n_flipped_to_refuse']}/{a_block['n_baseline_comply']} "
                     f"= {a_block['flip_rate']*100:.1f}%")
            b_str = (f"{b_block['n_flipped_to_comply']}/{b_block['n_baseline_refuse']} "
                     f"= {b_block['flip_rate']*100:.1f}%")
            print(f"{cls:22s} {mag_ratio:>9.3f}  {cos_neg:>+9.4f}  {a_str:>12s} {b_str:>12s}")
        a_overall = results["experiment_a_mitigate_jb_subtract_per_class_rjb"][mag_cond]["overall"]
        print(f"{'(Exp A overall)':22s} {'':>10s} {'':>10s} "
              f"{a_overall['n_flipped_to_refuse']}/{a_overall['n_baseline_comply']} = "
              f"{a_overall['flip_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
