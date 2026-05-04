"""Jailbreak-vector intervention: causal validation of the §5.5 cosine result.

Builds a SINGLE universal jailbreak vector r_jb_universal at L15 pos=−2 by
averaging per-class r_jb_class vectors (where r_jb_class = mean(h_jb_class) −
mean(h_bare); points TOWARD jailbreak per the Ball 2024 / Wang 2025 sign
convention). Then runs two intervention experiments:

  Experiment A — mitigate JB by subtracting r_jb_universal:
    For each (prompt, jb_* condition) where Stage 06 baseline = COMPLY,
    SUBTRACT r_jb_universal at L15 across all positions every forward pass
    (i.e. remove the empirical jailbreak displacement from the residual).
    Expected flip: COMPLY → REFUSE. Compares to Stage 06's r̂ pro-refusal-add
    flip rate (89/89 = 100 %). If our flip rate is high, the empirical JB
    displacement IS the displacement r̂'s intervention undoes — the JB lives
    along the refusal axis (anti-parallel to r̂, parallel to the harmless
    direction -r̂).

  Experiment B — induce JB by adding r_jb_universal:
    For each bare prompt where Stage 06 baseline = REFUSE, ADD
    r_jb_universal at L15 across all positions every forward pass
    (i.e. inject the empirical jailbreak displacement into a refusing
    prompt). Expected flip: REFUSE → COMPLY. This is the inverse of
    Experiment A and the direct analogue of Ball 2024's induction-style
    experiment (their Table 3).

Per-class jailbreak-vector interventions would be more rigorous (one r_jb
per class, applied only to that class's JB-comply prompts) since the per-class
magnitudes range 0.40 – 1.11 ‖r̂‖ and the universal mean is a lossy summary.
They are deferred — see HANDOFF.md note. The universal version is the cheaper
baseline test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/c/Users/Mahmoud Shabana/Documents/algoverse/Refusal-Lens")
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--max-new-tokens", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()

    print("[jb-vec] loading r̂[L15]")
    r_dict = torch.load(args.run_dir / "01_direction/unnormalized_r.pt", weights_only=False)
    r_hat = r_dict[LAYER].float().cpu()
    r_hat_norm = r_hat.norm().item()
    print(f"  |r̂[L15]| = {r_hat_norm:.2f}")

    print("[jb-vec] loading saved residuals from §5.5 (residuals_L15_per_cond.pt)")
    R = torch.load(args.run_dir / "02b_stats/residuals_L15_per_cond.pt", weights_only=False)
    # R[cond] shape: [n_prompts=50, n_pos=3, 2560]

    print(f"[jb-vec] computing r_jb_universal at pos=−2 (Ball 2024 convention: r_jb points TOWARD jailbreak)")
    mu_bare = R["bare"][:, POS_IDX, :].float().mean(dim=0)
    r_jb_per_class = []
    for cls in CLASSES:
        mu_jb = R[f"jb_{cls}"][:, POS_IDX, :].float().mean(dim=0)
        r_jb_per_class.append(mu_jb - mu_bare)
    r_jb_universal = torch.stack(r_jb_per_class).mean(dim=0)
    r_jb_norm = r_jb_universal.norm().item()
    cos_universal = torch.nn.functional.cosine_similarity(
        r_jb_universal.unsqueeze(0), r_hat.unsqueeze(0)
    ).item()
    print(f"  |r_jb_universal| = {r_jb_norm:.2f}")
    print(f"  |r_jb_universal| / |r̂| = {r_jb_norm / r_hat_norm:.3f}")
    print(f"  cos(r̂, r_jb_universal) = {cos_universal:+.4f}")
    # Per-class for diagnostics
    print(f"  per-class |r_jb_class| / |r̂|:")
    for cls, v in zip(CLASSES, r_jb_per_class):
        cv = torch.nn.functional.cosine_similarity(v.unsqueeze(0), r_hat.unsqueeze(0)).item()
        print(f"    {cls:22s} {v.norm()/r_hat_norm:.3f}  cos={cv:+.4f}")

    print("[jb-vec] loading Stage 06 baselines (causal_results.json)")
    causal = json.loads((args.run_dir / "06_causal/causal_results.json").read_text())
    baselines = {}
    for r in causal["results"]:
        baselines[r["prompt_idx"]] = {c: blob["cls"] for c, blob in r["baseline"].items()}
    n_jb_comply_total = sum(
        1 for pid in baselines for c in baselines[pid]
        if c.startswith("jb_") and baselines[pid][c] == "COMPLY"
    )
    n_bare_refuse_total = sum(
        1 for pid in baselines if baselines[pid].get("bare") == "REFUSE"
    )
    print(f"  jb-comply targets: {n_jb_comply_total}")
    print(f"  bare-refuse targets: {n_bare_refuse_total}")

    print(f"[jb-vec] loading model {args.model}")
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

    r_jb_gpu = r_jb_universal.to(model.device, dtype=torch.bfloat16)
    # Under Ball 2024 convention (r_jb points toward jailbreak):
    #   subtract r_jb -> mitigate JB (push residual back toward bare-refuse state)
    #   add r_jb      -> induce JB   (push bare-refuse residual toward JB-comply state)
    sub_hook = make_intervention_hook(r_jb_gpu, sign="sub")  # used for Experiment A
    add_hook = make_intervention_hook(r_jb_gpu, sign="add")  # used for Experiment B

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

    results = {
        "metadata": {
            "method": "universal_jb_vector_intervention_at_L15_all_positions_every_step",
            "layer": LAYER,
            "position_used_for_construction": -2,
            "r_hat_norm": r_hat_norm,
            "r_jb_universal_norm": r_jb_norm,
            "r_jb_universal_to_r_hat_ratio": r_jb_norm / r_hat_norm,
            "cos_r_hat_r_jb_universal": cos_universal,
            "construction": ("r_jb_universal = mean over 5 classes of (mean(h_jb_class) - mean(h_bare)) "
                             "at pos=-2; r_jb points TOWARD jailbreak (Ball 2024 / Wang 2025 convention). "
                             "Equivalently, r_jb is parallel to the harmless direction -r̂."),
            "dose_factor": 1.0,
            "max_new_tokens": args.max_new_tokens,
            "n_prompts": len(dataset),
            "limitation_per_class": ("Per-class r_jb_class interventions would be more rigorous "
                                     "(applied only to their own class's JB-comply prompts) and are "
                                     "deferred. See HANDOFF.md."),
            "per_class_r_jb_norms_over_r_hat": {
                cls: v.norm().item() / r_hat_norm for cls, v in zip(CLASSES, r_jb_per_class)
            },
            "per_class_cos_r_hat_r_jb_class": {
                cls: torch.nn.functional.cosine_similarity(
                    v.unsqueeze(0), r_hat.unsqueeze(0)).item()
                for cls, v in zip(CLASSES, r_jb_per_class)
            },
        },
        "experiment_a_mitigate_jb_subtract_rjb": {
            "description": ("SUBTRACT r_jb_universal at L15 every forward pass, on jb_* prompts where "
                            "Stage 06 baseline=COMPLY. Removes the empirical jailbreak displacement "
                            "from the residual, expected to flip COMPLY → REFUSE."),
            "per_prompt": [],
            "summary": {},
        },
        "experiment_b_induce_jb_add_rjb": {
            "description": ("ADD r_jb_universal at L15 every forward pass, on bare prompts where "
                            "Stage 06 baseline=REFUSE. Injects the empirical jailbreak displacement "
                            "into a refusing prompt; expected to flip REFUSE → COMPLY (Ball 2024 "
                            "Table 3-style induction)."),
            "per_prompt": [],
            "summary": {},
        },
    }

    print("\n" + "=" * 80)
    print("Experiment A: subtract r_jb_universal from jb-comply prompts (mitigate JB)")
    print("=" * 80)
    n_done = 0
    n_flipped_a = 0
    per_class_a = {cls: {"n": 0, "flipped": 0, "coherent": 0} for cls in CLASSES}
    t_a = time.time()
    for prompt_idx, prompt_row in enumerate(dataset):
        for cls in CLASSES:
            cond = f"jb_{cls}"
            if baselines.get(prompt_idx, {}).get(cond) != "COMPLY":
                continue
            text = prompt_row["conditions"][cond]["text"]
            resp = gen_with_hook(text, sub_hook)
            new_cls = classify_response(resp)
            coh = is_coherent(resp)
            flipped = new_cls == "REFUSE"
            results["experiment_a_mitigate_jb_subtract_rjb"]["per_prompt"].append({
                "prompt_idx": prompt_idx, "condition": cond,
                "baseline_cls": "COMPLY", "intervened_cls": new_cls,
                "flipped_to_refuse": flipped, "coherent": coh,
                "response_truncated": resp[:300],
            })
            n_done += 1
            per_class_a[cls]["n"] += 1
            if flipped:
                n_flipped_a += 1
                per_class_a[cls]["flipped"] += 1
            if coh:
                per_class_a[cls]["coherent"] += 1
            if n_done % 10 == 0:
                print(f"  [A {n_done}/{n_jb_comply_total}] elapsed {(time.time()-t_a)/60:.1f} min, "
                      f"flips so far {n_flipped_a}/{n_done}")
    flip_a = n_flipped_a / n_done if n_done else 0
    results["experiment_a_mitigate_jb_subtract_rjb"]["summary"] = {
        "n_jb_comply_baseline": n_done,
        "n_flipped_to_refuse": n_flipped_a,
        "flip_rate": flip_a,
        "per_class": per_class_a,
    }
    print(f"  Experiment A flip rate: {n_flipped_a}/{n_done} = {flip_a*100:.1f}%")

    print("\n" + "=" * 80)
    print("Experiment B: add r_jb_universal to bare-refuse prompts (induce JB)")
    print("=" * 80)
    n_done_b = 0
    n_flipped_b = 0
    t_b = time.time()
    for prompt_idx, prompt_row in enumerate(dataset):
        if baselines.get(prompt_idx, {}).get("bare") != "REFUSE":
            continue
        text = prompt_row["conditions"]["bare"]["text"]
        resp = gen_with_hook(text, add_hook)
        new_cls = classify_response(resp)
        coh = is_coherent(resp)
        flipped = new_cls == "COMPLY"
        results["experiment_b_induce_jb_add_rjb"]["per_prompt"].append({
            "prompt_idx": prompt_idx, "condition": "bare",
            "baseline_cls": "REFUSE", "intervened_cls": new_cls,
            "flipped_to_comply": flipped, "coherent": coh,
            "response_truncated": resp[:300],
        })
        n_done_b += 1
        if flipped:
            n_flipped_b += 1
        if n_done_b % 10 == 0:
            print(f"  [B {n_done_b}/{n_bare_refuse_total}] elapsed {(time.time()-t_b)/60:.1f} min, "
                  f"flips so far {n_flipped_b}/{n_done_b}")
    flip_b = n_flipped_b / n_done_b if n_done_b else 0
    results["experiment_b_induce_jb_add_rjb"]["summary"] = {
        "n_bare_refuse_baseline": n_done_b,
        "n_flipped_to_comply": n_flipped_b,
        "flip_rate": flip_b,
    }
    print(f"  Experiment B flip rate: {n_flipped_b}/{n_done_b} = {flip_b*100:.1f}%")

    out_path = args.run_dir / "06_causal/jb_vector_intervention_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[jb-vec] wrote {out_path.relative_to(REPO)}")

    print("\n" + "=" * 80)
    print("HEADLINE TABLE")
    print("=" * 80)
    print(f"{'intervention':62s} {'flip':>15s}")
    print("-" * 80)
    print(f"{'Stage 06 +1·|r̂| pro_refusal_add  (Arditi r̂, on jb-comply)':62s} "
          f"{'89/89 = 100.0%':>15s}")
    print(f"{'Stage 06 −1·|r̂| anti_refusal_sub (Arditi r̂, on bare-refuse)':62s} "
          f"{'49/50 = 98.0%':>15s}")
    print(f"{'Exp A: −r_jb_universal on jb-comply (mitigate JB)':62s} "
          f"{f'{n_flipped_a}/{n_done} = {flip_a*100:.1f}%':>15s}")
    print(f"{'Exp B: +r_jb_universal on bare-refuse (induce JB)':62s} "
          f"{f'{n_flipped_b}/{n_done_b} = {flip_b*100:.1f}%':>15s}")


if __name__ == "__main__":
    main()
