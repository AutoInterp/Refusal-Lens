"""Cosine similarity between the refusal direction r_hat and the mean activation
vector at the causal layer/position, for a model.

Reports cosine for the overall mean activation (all categories pooled) and per
category (harmless / harmful / harmful+jb).
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-harmless", type=int, default=100)
    p.add_argument("--n-harmful", type=int, default=50)
    p.add_argument("--n-jb", type=int, default=150)
    args = p.parse_args()
    cfg = MODELS[args.model]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    r_hat = load_rhat(cfg)
    L, P = cfg["layer"], cfg["pos"]
    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    r_dev = r_hat.to(model.device)

    cap = {}
    layers[L].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    def resid_at(text):
        ids = tok(format_prompt(tok, text, enable_thinking=cfg["enable_thinking"]), return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**ids)
        return cap["h"][0, P, :].float()

    splits = json.loads((REPO / "dataset/refusal_direction_dataset/splits/harmless_train.json").read_text())
    ctrl = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    cats = {
        "harmless": [d["instruction"] for d in splits][:args.n_harmless],
        "harmful": [pp["conditions"]["bare"]["text"] for pp in ctrl][:args.n_harmful],
        "harmful+jb": [b["text"] for pp in ctrl for c, b in pp["conditions"].items() if c.startswith("jb_")][:args.n_jb],
    }

    def cos(a, b):
        return (a @ b / (a.norm() * b.norm())).item()

    out = {"model": cfg["hf"], "layer": L, "position": P, "r_hat_norm": r_hat.norm().item(), "cosines": {}}
    all_vecs = []
    for name, prompts in cats.items():
        vecs = torch.stack([resid_at(t) for t in prompts])  # (n, d)
        all_vecs.append(vecs)
        mean_v = vecs.mean(0)
        c = cos(mean_v, r_dev)
        out["cosines"][name] = {"n": len(prompts), "cos_meanact_rhat": c, "mean_act_norm": mean_v.norm().item()}
        print(f"[{args.model}] {name:11} cos(mean_act, r_hat) = {c:+.4f}   ||mean_act||={mean_v.norm():.2f}")
    overall = torch.cat(all_vecs).mean(0)
    c_all = cos(overall, r_dev)
    out["cosines"]["overall"] = {"n": sum(len(v) for v in cats.values()),
                                 "cos_meanact_rhat": c_all, "mean_act_norm": overall.norm().item()}
    print(f"[{args.model}] {'OVERALL':11} cos(mean_act, r_hat) = {c_all:+.4f}   ||mean_act||={overall.norm():.2f}")
    print(f"[{args.model}] ||r_hat|| = {r_hat.norm().item():.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
