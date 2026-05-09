"""Refusal-direction vs JB-derived-direction alignment analysis.

For each (prompt, condition) in the controlled 50-prompt × 11-condition dataset,
capture the residual stream at L15 (hook_resid_post == hidden_states[16]) at
positions [-5, -3, -2]. Then compute:

  - Per-condition mean residual mu_cond
  - Class-wise "JB direction":   r_jb_cls   = mu_jb_cls    - mu_bare        # points TOWARD jailbreak (Ball 2024 / Wang 2025 convention)
  - Class-wise semantic JB dir:  r_jb_sem   = mu_jb_cls    - mu_ctrl_cls    # same convention; ctrl is the prefix-matched neutral baseline
  - Cosine similarity of each derived direction with r_hat (the refusal direction)
  - Magnitude ratios |r_jb| / |r_hat|

Sign convention (matches Ball 2024 / Wang 2025): r_jb points TOWARD jailbreak.
Therefore cos(r_hat, r_jb) is NEGATIVE for an effective JB (anti-parallel to
refusal). Equivalently, cos(-r_hat, r_jb) is POSITIVE — i.e. r_jb is parallel
to the harmless-pointing axis (-r_hat). This is the geometric content of the
'JBs make the model perceive the prompt as harmless' hypothesis.

The cos(-r_hat, r_jb) values answer: "is the JB-induced shift parallel to the
harmless direction as a vector, or just along the same scalar projection axis?"
If cos ≈ 1, JBs literally edit toward harmless. If cos < 0.7, JBs operate on a
partially-orthogonal axis that happens to project onto the harmless axis at
the magnitude observed in §5.4.

Output: data/results/pipeline_runs/run_20260430_023247/02b_stats/direction_alignment.{json,png}
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
from utils import format_prompt, load_controlled_dataset  # noqa: E402

CONDITIONS = ["bare"] + [
    f"{p}_{c}" for c in ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]
    for p in ["jb", "ctrl"]
]
CLASSES = ["fiction", "roleplay", "analytical", "completion", "cognitive_reframe"]
TARGET_POSITIONS = [-5, -3, -2]  # multi-position measurement set used in Stage 02
LAYER = 15


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path,
                   default=REPO / "data/results/pipeline_runs/run_20260430_023247")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-prompts", type=int, default=50)
    return p.parse_args()


def iter_conditions(prompt_row):
    conds = prompt_row["conditions"]
    yield "bare", conds["bare"]["text"]
    for cls in CLASSES:
        yield f"jb_{cls}", conds[f"jb_{cls}"]["text"]
        yield f"ctrl_{cls}", conds[f"ctrl_{cls}"]["text"]


def main():
    args = parse_args()

    direction_path = args.run_dir / "01_direction" / "unnormalized_r.pt"
    print(f"[align] loading r_hat from {direction_path.relative_to(REPO)}")
    r_dict = torch.load(direction_path, weights_only=False)
    r_unnorm = r_dict[LAYER].float().cpu()  # (2560,)
    r_norm = r_unnorm.norm().item()
    r_hat = r_unnorm / r_norm
    print(f"[align] |r_hat[L{LAYER}]| = {r_norm:.2f}")

    print(f"[align] loading dataset")
    dataset = load_controlled_dataset(REPO / "dataset" / "refusal_lens_controlled_dataset.json")
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]
    print(f"[align] n_prompts = {len(dataset)}")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    print(f"[align] loading model {args.model} dtype={args.dtype}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda",
    )
    model.eval()
    print(f"[align] model loaded in {time.time()-t0:.1f}s")

    # Layer access path: gemma-3 wraps in language_model
    if hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    else:
        layers = model.model.layers
    L_module = layers[LAYER]

    # Capture residual via forward hook on layer L's hook_resid_post equivalent.
    # In HF gemma-3, the cleanest way is output_hidden_states=True + index L+1.
    # That gives the residual AFTER block L, == hook_resid_post(L) in TL terms.

    # Storage: cond -> [n_prompts, n_positions, 2560]
    residuals = {c: [] for c in CONDITIONS}

    n_total = len(dataset) * len(CONDITIONS)
    counter = 0
    t0 = time.time()
    with torch.no_grad():
        for prompt_idx, prompt_row in enumerate(dataset):
            for cond, text in iter_conditions(prompt_row):
                formatted = format_prompt(tokenizer, text)
                ids = tokenizer(formatted, return_tensors="pt").to("cuda")
                seq_len = ids.input_ids.shape[1]
                # Some prompts are too short for pos=-5; pad or skip
                if seq_len < abs(min(TARGET_POSITIONS)):
                    # take what we can; record nan for unavailable
                    h_slice = torch.full((len(TARGET_POSITIONS), 2560),
                                         float("nan"), dtype=torch.float32)
                else:
                    out = model(**ids, output_hidden_states=True, use_cache=False)
                    # hidden_states is a tuple of length n_layers+1
                    # hidden_states[L+1] = residual after block L = hook_resid_post(L)
                    h = out.hidden_states[LAYER + 1][0]  # (seq_len, 2560)
                    h_slice = torch.stack(
                        [h[seq_len + p] for p in TARGET_POSITIONS], dim=0
                    ).float().cpu()
                residuals[cond].append(h_slice)
                counter += 1
                if counter % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / counter * (n_total - counter)
                    print(f"[align] {counter}/{n_total} ({counter/n_total*100:.0f}%) "
                          f"elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min")

    # Stack: cond -> [n_prompts, n_positions, 2560]
    R_per_cond = {c: torch.stack(residuals[c], dim=0) for c in CONDITIONS}
    print(f"[align] all {counter} forward passes done in {(time.time()-t0)/60:.1f} min")

    # Save raw tensor for downstream analysis
    out_dir = args.run_dir / "02b_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({c: t for c, t in R_per_cond.items()},
               out_dir / "residuals_L15_per_cond.pt")
    print(f"[align] saved residuals tensor "
          f"({sum(t.numel() for t in R_per_cond.values()) * 4 / 1e6:.1f} MB)")

    # Per-position means: shape [n_pos, 2560] for each condition
    # And avg-over-positions mean for the original report
    mu_per_pos = {c: torch.nanmean(t, dim=0) for c, t in R_per_cond.items()}  # [n_pos, 2560]
    mu_avg = {c: torch.nanmean(t.reshape(-1, 2560), dim=0) for c, t in R_per_cond.items()}

    out = {
        "metadata": {
            "model": args.model, "layer": LAYER,
            "target_positions": TARGET_POSITIONS,
            "n_prompts": len(dataset),
            "r_hat_norm": r_norm,
            "method": "r_jb_cls = mu_bare - mu_jb_cls; r_jb_sem_cls = mu_ctrl_cls - mu_jb_cls",
            "note": ("cos(r̂, r_jb) reported per-position because the residual at pos=-2 "
                     "(model token, where refusal decision lives) carries the refusal signal "
                     "with much higher SNR than pos=-5/-3 (template prefix tokens)."),
        },
        "per_class": {},
        "per_position_per_class": {},
        "per_condition_proj": {},
    }

    # Bare projection per position (the reference)
    print()
    print("=" * 96)
    print("PROJECTION OF MEAN RESIDUAL ONTO r̂ AT L15, PER POSITION (raw direct_dot)")
    print("=" * 96)
    print(f"{'condition':22s} " + "  ".join(f"pos={p:>+3d}" for p in TARGET_POSITIONS) + "    avg")
    for c in ["bare"] + [f"{p}_{cls}" for cls in CLASSES for p in ["jb", "ctrl"]]:
        per_pos = (mu_per_pos[c] @ r_hat).tolist()
        avg = (mu_avg[c] @ r_hat).item()
        out["per_condition_proj"][c] = {
            "per_position": {str(p): v for p, v in zip(TARGET_POSITIONS, per_pos)},
            "avg": avg,
            "ratio_avg_over_r_norm": avg / r_norm,
        }
        per_pos_str = "  ".join(f"{v:>+9.1f}" for v in per_pos)
        print(f"{c:22s}  {per_pos_str}  {avg:>+9.1f}")

    print()
    print("=" * 96)
    print("COSINE SIMILARITY AT pos=-2 (the refusal-decision token; highest SNR)")
    print("=" * 96)
    print(f"{'class':22s} {'cos(r̂, r_jb)':>14s} {'|r_jb|/|r̂|':>14s}  | "
          f"{'cos(r̂, r_jb_sem)':>18s} {'|r_jb_sem|/|r̂|':>16s}")
    pos_idx_minus2 = TARGET_POSITIONS.index(-2)
    for cls in CLASSES:
        mu_bare_p = mu_per_pos["bare"][pos_idx_minus2]
        mu_jb_p = mu_per_pos[f"jb_{cls}"][pos_idx_minus2]
        mu_ctrl_p = mu_per_pos[f"ctrl_{cls}"][pos_idx_minus2]
        # Ball 2024 / Wang 2025 convention: r_jb points TOWARD jailbreak.
        r_jb_p = (mu_jb_p - mu_bare_p).float()
        r_jb_sem_p = (mu_jb_p - mu_ctrl_p).float()
        cos_jb = torch.nn.functional.cosine_similarity(
            r_jb_p.unsqueeze(0), r_hat.unsqueeze(0)).item()
        cos_sem = torch.nn.functional.cosine_similarity(
            r_jb_sem_p.unsqueeze(0), r_hat.unsqueeze(0)).item()
        mag_jb_ratio = r_jb_p.norm().item() / r_norm
        mag_sem_ratio = r_jb_sem_p.norm().item() / r_norm
        out["per_class"][cls] = {
            "pos_minus_2": {
                "cos_r_hat_r_jb": cos_jb,
                "cos_r_hat_r_jb_sem": cos_sem,
                "mag_r_jb": r_jb_p.norm().item(),
                "mag_r_jb_sem": r_jb_sem_p.norm().item(),
                "mag_ratio_r_jb": mag_jb_ratio,
                "mag_ratio_r_jb_sem": mag_sem_ratio,
                "proj_r_jb_on_r_hat": (r_jb_p @ r_hat).item(),
                "proj_r_jb_sem_on_r_hat": (r_jb_sem_p @ r_hat).item(),
            },
        }
        print(f"{cls:22s} {cos_jb:>+14.4f} {mag_jb_ratio:>+14.4f}  | "
              f"{cos_sem:>+18.4f} {mag_sem_ratio:>+16.4f}")

    # Also compute per-position for completeness
    out["per_position_per_class"] = {}
    for ipos, pos in enumerate(TARGET_POSITIONS):
        out["per_position_per_class"][str(pos)] = {}
        for cls in CLASSES:
            mu_b = mu_per_pos["bare"][ipos]
            mu_j = mu_per_pos[f"jb_{cls}"][ipos]
            mu_c = mu_per_pos[f"ctrl_{cls}"][ipos]
            # Ball 2024 / Wang 2025 convention: r_jb points TOWARD jailbreak.
            r_jb = (mu_j - mu_b).float()
            r_jb_sem = (mu_j - mu_c).float()
            cos_jb = torch.nn.functional.cosine_similarity(
                r_jb.unsqueeze(0), r_hat.unsqueeze(0)).item()
            cos_sem = torch.nn.functional.cosine_similarity(
                r_jb_sem.unsqueeze(0), r_hat.unsqueeze(0)).item()
            out["per_position_per_class"][str(pos)][cls] = {
                "cos_r_hat_r_jb": cos_jb,
                "cos_r_hat_r_jb_sem": cos_sem,
                "mag_ratio_r_jb": r_jb.norm().item() / r_norm,
                "mag_ratio_r_jb_sem": r_jb_sem.norm().item() / r_norm,
                "proj_r_jb_on_r_hat": (r_jb @ r_hat).item(),
                "proj_r_jb_sem_on_r_hat": (r_jb_sem @ r_hat).item(),
            }

    out_json = out_dir / "direction_alignment.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[align] wrote {out_json.relative_to(REPO)}")


if __name__ == "__main__":
    main()
