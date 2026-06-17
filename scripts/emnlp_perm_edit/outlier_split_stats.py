"""Split the refusal direction into outlier-dim vs complement, and recompute
direction stats for all three (full / outlier / complement) on a model.

outlier dim = argmax(|r_hat|). Two derived directions:
  r_outlier    = r_hat masked to ONLY the outlier dim (rest zero)
  r_complement = r_hat with the outlier dim set to zero

Both are stored as .pt. For each of {full, outlier, complement} we report, at
the causal layer/position:
  ||direction||
  proj onto the UNIT direction (mean +- std) per category (harmless/harmful/jb)
  cosine(mean activation, direction) per category + overall
The activation norm ||h|| is direction-independent and reported once.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from utils import format_prompt, load_controlled_dataset  # noqa: E402

OUTDIR = REPO / "data/results/emnlp_perm_edit/phase0_controllability"
MODELS = {
    "gemma": {"hf": "google/gemma-3-4b-it", "layer": 15, "pos": -2, "enable_thinking": None,
              "rhat": REPO / "data/results/pipeline_runs/run_20260430_023247/01_direction/unnormalized_r.pt",
              "rhat_kind": "dict"},
    "qwen": {"hf": "Qwen/Qwen3-4B", "layer": 18, "pos": -1, "enable_thinking": False,
             "rhat": REPO / "data/results/pipeline_runs_qwen/run_regen_L18/01_direction/positions_L18/pos_-1_unnormalized.pt",
             "rhat_kind": "tensor"},
}


def load_rhat(cfg):
    obj = torch.load(cfg["rhat"], weights_only=False, map_location="cpu")
    if cfg["rhat_kind"] == "dict":
        return obj[cfg["layer"]].float()
    return obj.float() if isinstance(obj, torch.Tensor) else obj["direction"].float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--n-harmless", type=int, default=100)
    p.add_argument("--n-harmful", type=int, default=50)
    p.add_argument("--n-jb", type=int, default=150)
    args = p.parse_args()
    cfg = MODELS[args.model]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    r_full = load_rhat(cfg)
    outlier_dim = int(r_full.abs().argmax())
    r_outlier = torch.zeros_like(r_full); r_outlier[outlier_dim] = r_full[outlier_dim]
    r_complement = r_full.clone(); r_complement[outlier_dim] = 0.0
    # store
    torch.save(r_outlier, OUTDIR / f"{args.model}_rhat_outlier.pt")
    torch.save(r_complement, OUTDIR / f"{args.model}_rhat_complement.pt")
    dirs = {"full": r_full, "outlier": r_outlier, "complement": r_complement}
    print(f"[{args.model}] outlier_dim={outlier_dim}  r[outlier]={r_full[outlier_dim]:.2f}  "
          f"||full||={r_full.norm():.2f} ||outlier||={r_outlier.norm():.2f} "
          f"||complement||={r_complement.norm():.2f}  "
          f"(outlier is {100*r_outlier.norm()/r_full.norm():.1f}% of ||full|| by mag, "
          f"{100*(r_outlier.norm()/r_full.norm())**2:.1f}% by sq-norm)")

    # model + activations
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    L, P = cfg["layer"], cfg["pos"]
    cap = {}
    layers[L].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    def resid_at(text):
        ids = tok(format_prompt(tok, text, enable_thinking=cfg["enable_thinking"]), return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**ids)
        return cap["h"][0, P, :].float().cpu()

    splits = json.loads((REPO / "dataset/refusal_direction_dataset/splits/harmless_train.json").read_text())
    ctrl = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    cats = {
        "harmless": [d["instruction"] for d in splits][:args.n_harmless],
        "harmful": [pp["conditions"]["bare"]["text"] for pp in ctrl][:args.n_harmful],
        "harmful+jb": [b["text"] for pp in ctrl for c, b in pp["conditions"].items() if c.startswith("jb_")][:args.n_jb],
    }
    vecs = {name: torch.stack([resid_at(t) for t in prompts]) for name, prompts in cats.items()}
    all_v = torch.cat(list(vecs.values()))

    out = {"model": cfg["hf"], "layer": L, "position": P, "outlier_dim": outlier_dim,
           "norms": {k: v.norm().item() for k, v in dirs.items()}, "directions": {}}
    # activation norms (direction-independent)
    print(f"\n[{args.model}] activation norm ||h|| per category:")
    out["act_norm"] = {}
    for name, v in vecs.items():
        nn = v.norm(dim=1)
        out["act_norm"][name] = {"mean": nn.mean().item(), "std": nn.std().item()}
        print(f"    {name:11} ||h|| = {nn.mean():9.2f} +- {nn.std():8.2f}")

    for dname, dvec in dirs.items():
        unit = dvec / dvec.norm()
        out["directions"][dname] = {"norm": dvec.norm().item(), "per_category": {}}
        print(f"\n[{args.model}] direction={dname}  ||dir||={dvec.norm():.3f}")
        for name, v in vecs.items():
            proj = v @ unit
            mean_v = v.mean(0)
            cos = (mean_v @ dvec / (mean_v.norm() * dvec.norm())).item()
            out["directions"][dname]["per_category"][name] = {
                "proj_mean": proj.mean().item(), "proj_std": proj.std().item(),
                "cos_meanact": cos}
            print(f"    {name:11} proj={proj.mean():9.3f} +- {proj.std():8.3f}   cos(mean_act,dir)={cos:+.4f}")
        cos_all = (all_v.mean(0) @ dvec / (all_v.mean(0).norm() * dvec.norm())).item()
        out["directions"][dname]["cos_meanact_overall"] = cos_all
        print(f"    {'OVERALL':11} cos(mean_act,dir)={cos_all:+.4f}")

    (OUTDIR / f"{args.model}_outlier_split_stats.json").write_text(json.dumps(out, indent=2))
    print(f"\n[{args.model}] wrote {args.model}_outlier_split_stats.json + _rhat_outlier.pt + _rhat_complement.pt")


if __name__ == "__main__":
    main()
