"""Arditi-style directional ablation, split by outlier vs complement.

Subtract the UNNORMALIZED refusal direction (coefficient 1.0) from the residual
at layer L, at ALL token positions, on every forward pass (prompt + each
generation step). Generate completions on harmful prompts, for each direction
variant {full, outlier, complement} plus a no-hook baseline.

r_full = r_outlier + r_complement, so subtracting `full` == subtracting
`outlier` + `complement`; running each separately attributes the refusal-killing
effect to the outlier dim, the rest, or both.

Output: one JSON of records {setting, prompt_idx, prompt_text, response,
classification(keyword)} to be scored by the LLM judge afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from edge_ablation_hook import make_scalar_rhat_subtraction_hook  # noqa: E402
from utils import classify_response, format_prompt, load_controlled_dataset  # noqa: E402

OUTDIR = REPO / "data/results/emnlp_perm_edit/phase0_controllability"
MODELS = {
    "gemma": {"hf": "google/gemma-3-4b-it", "layer": 15, "enable_thinking": None,
              "rhat": REPO / "data/results/pipeline_runs/run_20260430_023247/01_direction/unnormalized_r.pt",
              "rhat_kind": "dict"},
    "qwen": {"hf": "Qwen/Qwen3-4B", "layer": 18, "enable_thinking": False,
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
    p.add_argument("--n-harmful", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=192)
    args = p.parse_args()
    cfg = MODELS[args.model]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    r_full = load_rhat(cfg)
    outlier_dim = int(r_full.abs().argmax())
    r_outlier = torch.zeros_like(r_full); r_outlier[outlier_dim] = r_full[outlier_dim]
    r_complement = r_full.clone(); r_complement[outlier_dim] = 0.0
    variants = {"full": r_full, "outlier": r_outlier, "complement": r_complement}

    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    target_layer = layers[cfg["layer"]]
    pad_id = tok.eos_token_id

    ctrl = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    harmful = [pp["conditions"]["bare"]["text"] for pp in ctrl][:args.n_harmful]

    def gen(text, hook_fn=None):
        formatted = format_prompt(tok, text, enable_thinking=cfg["enable_thinking"])
        ids = tok(formatted, return_tensors="pt").to(model.device)
        plen = ids.input_ids.shape[1]
        handle = target_layer.register_forward_hook(hook_fn) if hook_fn else None
        try:
            with torch.no_grad():
                out = model.generate(**ids, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=pad_id)
        finally:
            if handle:
                handle.remove()
        return tok.decode(out[0][plen:], skip_special_tokens=True)

    records = []
    t0 = time.time()
    settings = ["baseline", "full", "outlier", "complement"]
    for setting in settings:
        hook_fn = None
        if setting != "baseline":
            rv = variants[setting]
            # coeff = delta/||rv||^2 == 1  -> h_new = h - rv  (subtract unnormalized direction)
            hook_fn = make_scalar_rhat_subtraction_hook(rv, float(rv @ rv), position_mode="all")
        for pidx, text in enumerate(harmful):
            resp = gen(text, hook_fn)
            records.append({"model": args.model, "setting": setting, "prompt_idx": pidx,
                            "prompt_text": text, "response": resp,
                            "classification": classify_response(resp)})
            if len(records) % 20 == 0:
                print(f"  [{len(records)}] {setting} elapsed={time.time()-t0:.0f}s")
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({"records": records,
                    "meta": {"outlier_dim": outlier_dim, "n_harmful": len(harmful),
                             "max_new_tokens": args.max_new_tokens}}, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"records": records,
        "meta": {"outlier_dim": outlier_dim, "n_harmful": len(harmful),
                 "max_new_tokens": args.max_new_tokens}}, indent=2))
    print(f"[{args.model}] done {len(records)} gens in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
