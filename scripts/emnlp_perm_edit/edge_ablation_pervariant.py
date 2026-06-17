"""Edge ablation using each variant's OWN attribution magnitude (per-direction).

For each refusal-direction variant {full, outlier, complement} we now have a
proper Stage-2 attribution toward THAT direction (gemma_var_<v>). This script
ablates by subtracting `delta * r_var_unit` from the residual at layer L (all
positions), where delta = that variant's `net` (all_signed) for the given
(prompt, condition) and r_var_unit is the unit variant direction. Since r_var_unit
is unit-norm, the hook changes h·r_var_unit by exactly -delta (removes the edge
contribution to the variant's projection).

Expectation: complement delta~+900 -> small/coherent edit that should break
refusal if the complement carries it; full/outlier delta~-48000 -> catastrophic.

Generates completions (baseline + 3 variants) over all 11 conditions; scored
afterwards by the LLM judge + corrected scorer.
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

VARIANTS = ["full", "outlier", "complement"]
MODELS = {
    "gemma": {"hf": "google/gemma-3-4b-it", "layer": 15, "enable_thinking": None,
              "run_base": "data/results/pipeline_runs/gemma_var_"},
    "qwen": {"hf": "Qwen/Qwen3-4B", "layer": 18, "enable_thinking": False,
             "run_base": "data/results/pipeline_runs_qwen/qwen_var_"},
}
CFG = None  # set in main()


def load_variant(v):
    """Return (unit_direction, net_by_key) for a variant from its Stage-2 output."""
    rd = REPO / f"{CFG['run_base']}{v}"
    unit = torch.load(rd / f"01_direction/directions/layer_{CFG['layer']:02d}.pt", weights_only=False, map_location="cpu").float()
    res = json.loads((rd / "02_attribution/attribution_results.json").read_text())
    net = {}
    for r in res["results"]:
        pidx = r["prompt_idx"]
        for cond, blob in r["conditions"].items():
            g = blob.get("graphs", {}).get("single")
            if g and g.get("net") is not None:
                net[(pidx, cond)] = float(g["net"])
    return unit, net


def main():
    global CFG
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma", choices=list(MODELS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-prompts", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=192)
    args = p.parse_args()
    CFG = MODELS[args.model]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    variants = {v: load_variant(v) for v in VARIANTS}
    for v, (u, net) in variants.items():
        print(f"[{v}] ||unit||={u.norm():.3f}  net entries={len(net)}  "
              f"bare nets sample={[round(net.get((i,'bare'),float('nan')),1) for i in range(3)]}")

    tok = AutoTokenizer.from_pretrained(CFG["hf"])
    model = AutoModelForCausalLM.from_pretrained(CFG["hf"], torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    target_layer = layers[CFG["layer"]]
    pad_id = tok.eos_token_id
    ctrl = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")[:args.max_prompts]

    def gen(text, hook_fn=None):
        ids = tok(format_prompt(tok, text, enable_thinking=CFG["enable_thinking"]), return_tensors="pt").to(model.device)
        plen = ids.input_ids.shape[1]
        h = target_layer.register_forward_hook(hook_fn) if hook_fn else None
        try:
            with torch.no_grad():
                out = model.generate(**ids, do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=pad_id)
        finally:
            if h:
                h.remove()
        return tok.decode(out[0][plen:], skip_special_tokens=True)

    records = []
    t0 = time.time()
    for setting in ["baseline"] + VARIANTS:
        for pidx, prompt in enumerate(ctrl):
            for cond, blob in prompt["conditions"].items():
                hook_fn = None
                delta = 0.0
                if setting != "baseline":
                    unit, net = variants[setting]
                    if (pidx, cond) not in net:
                        continue
                    delta = net[(pidx, cond)]
                    hook_fn = make_scalar_rhat_subtraction_hook(unit, delta, position_mode="all")
                resp = gen(blob["text"], hook_fn)
                records.append({"model": args.model, "setting": setting, "prompt_idx": pidx,
                                "condition": cond, "prompt_text": blob["text"], "delta": delta,
                                "response": resp, "classification": classify_response(resp)})
                if len(records) % 20 == 0:
                    print(f"  [{len(records)}] {setting} elapsed={time.time()-t0:.0f}s")
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps({"records": records}, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"records": records}, indent=2))
    print(f"done {len(records)} gens in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
