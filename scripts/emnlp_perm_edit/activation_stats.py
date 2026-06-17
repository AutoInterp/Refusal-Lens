"""Activation-norm + refusal-direction projection stats at the causal layer.

For a model, at (layer L, position P) where the refusal direction lives:
  * ||h|| residual-stream norm: mean +- std, per category
  * ||r_hat|| unnormalized refusal-direction norm
  * projection onto the UNIT refusal direction  proj = h . (r_hat/||r_hat||):
    mean +- std per category  (this is the "magnitude along refusal direction")

Categories:
  harmless     -> refusal_direction_dataset/splits/harmless_train.json
  harmful      -> controlled dataset, condition=bare
  harmful+jb   -> controlled dataset, conditions=jb_*  (pooled)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-harmless", type=int, default=100)
    p.add_argument("--n-harmful", type=int, default=50)
    p.add_argument("--n-jb", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODELS[args.model]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    r_hat = load_rhat(cfg)
    r_norm = r_hat.norm().item()
    r_unit = (r_hat / r_norm)
    L, P = cfg["layer"], cfg["pos"]

    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    r_unit_dev = r_unit.to(model.device)

    captured = {}
    def hook(mod, inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    handle = layers[L].register_forward_hook(hook)

    def resid_at(text):
        formatted = format_prompt(tok, text, enable_thinking=cfg["enable_thinking"])
        ids = tok(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**ids)
        return captured["h"][0, P, :].float()  # (d_model,)

    # Build category prompt lists.
    splits = json.loads((REPO / "dataset/refusal_direction_dataset/splits/harmless_train.json").read_text())
    harmless = [d["instruction"] for d in splits][:args.n_harmless]
    ctrl = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    harmful = [p["conditions"]["bare"]["text"] for p in ctrl][:args.n_harmful]
    jb = []
    for p in ctrl:
        for cond, blob in p["conditions"].items():
            if cond.startswith("jb_"):
                jb.append(blob["text"])
    jb = jb[:args.n_jb]

    cats = {"harmless": harmless, "harmful": harmful, "harmful+jb": jb}
    out = {"model": cfg["hf"], "layer": L, "position": P, "r_hat_norm": r_norm, "categories": {}}
    for name, prompts in cats.items():
        norms, projs = [], []
        for t in prompts:
            h = resid_at(t)
            norms.append(h.norm().item())
            projs.append((h @ r_unit_dev).item())
        norms = torch.tensor(norms); projs = torch.tensor(projs)
        out["categories"][name] = {
            "n": len(prompts),
            "act_norm_mean": norms.mean().item(), "act_norm_std": norms.std().item(),
            "proj_mean": projs.mean().item(), "proj_std": projs.std().item(),
            "proj_absmean": projs.abs().mean().item(),
        }
        print(f"[{args.model}] {name:11} n={len(prompts):3} "
              f"||h||={norms.mean():7.1f}+-{norms.std():6.1f}  "
              f"proj_r={projs.mean():8.2f}+-{projs.std():7.2f}")
    handle.remove()
    print(f"[{args.model}] ||r_hat|| = {r_norm:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
