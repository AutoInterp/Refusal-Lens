"""Edge-ablation (ablate all edges feeding the refusal direction), split across
the full / outlier / complement refusal-direction variants.

The full 0b-simple edge ablation subtracts vec = c * r_full from the residual at
layer L (all positions), where c = delta_full / ||r_full||^2 and delta_full is
the edges' contribution to the refusal-direction projection (decomposition
`all_signed`, converted to the unnormalized basis for Qwen). Since
r_full = r_outlier + r_complement, that vector splits additively:
    c*r_full == c*r_outlier + c*r_complement
so we run each component separately (shared c) to see which subspace carries the
edge-ablation effect. Plus a no-hook baseline. All 11 conditions
(bare / ctrl_* / jb_*). Full completions saved for LLM-judge + corrected scoring.
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
              "rhat_kind": "dict",
              "decomp": OUTDIR / "linearization_decomposition.json",
              # FIX (2026-06-16): Gemma's Stage 02 also targeted the NORMALIZED direction
              # (||layer_15.pt||=1.0), so all_signed is in normalized-projection units and
              # must be converted with ×||r||, same as Qwen. Previously False -> under-applied by ||r||.
              "delta_to_unnorm": True},
    "qwen": {"hf": "Qwen/Qwen3-4B", "layer": 18, "enable_thinking": False,
             "rhat": REPO / "data/results/pipeline_runs_qwen/run_regen_L18/01_direction/positions_L18/pos_-1_unnormalized.pt",
             "rhat_kind": "tensor",
             "decomp": OUTDIR / "qwen_linearization_decomposition.json",
             "delta_to_unnorm": True},
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
    p.add_argument("--max-prompts", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=192)
    args = p.parse_args()
    cfg = MODELS[args.model]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    r_full = load_rhat(cfg)
    r_norm = r_full.norm().item()
    outlier_dim = int(r_full.abs().argmax())
    r_outlier = torch.zeros_like(r_full); r_outlier[outlier_dim] = r_full[outlier_dim]
    r_complement = r_full.clone(); r_complement[outlier_dim] = 0.0
    variants = {"full": r_full, "outlier": r_outlier, "complement": r_complement}
    r_full_norm_sq = float(r_full @ r_full)

    decomp = json.loads(Path(cfg["decomp"]).read_text())
    per_prompt = {(r["prompt_idx"], r["condition"]): r for r in decomp["per_prompt"]}

    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    target_layer = layers[cfg["layer"]]
    pad_id = tok.eos_token_id

    ctrl = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")[:args.max_prompts]

    def gen(text, hook_fn=None):
        ids = tok(format_prompt(tok, text, enable_thinking=cfg["enable_thinking"]), return_tensors="pt").to(model.device)
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
    for setting in ["baseline", "full", "outlier", "complement"]:
        for pidx, prompt in enumerate(ctrl):
            for cond, blob in prompt["conditions"].items():
                hook_fn = None
                c = 0.0
                if setting != "baseline":
                    rec = per_prompt.get((pidx, cond))
                    if rec is None:
                        continue
                    delta_full = float(rec["all_signed"]) * (r_norm if cfg["delta_to_unnorm"] else 1.0)
                    c = delta_full / r_full_norm_sq            # full edge-ablation coefficient
                    rv = variants[setting]
                    # subtract c * rv  ->  make_scalar hook with delta = c*||rv||^2 gives coeff=c
                    hook_fn = make_scalar_rhat_subtraction_hook(rv, c * float(rv @ rv), position_mode="all")
                resp = gen(blob["text"], hook_fn)
                records.append({"model": args.model, "setting": setting, "prompt_idx": pidx,
                                "condition": cond, "prompt_text": blob["text"], "coeff": c,
                                "response": resp, "classification": classify_response(resp)})
                if len(records) % 20 == 0:
                    print(f"  [{len(records)}] {setting} elapsed={time.time()-t0:.0f}s")
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps({"records": records,
                        "meta": {"outlier_dim": outlier_dim, "r_norm": r_norm}}, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"records": records,
        "meta": {"outlier_dim": outlier_dim, "r_norm": r_norm}}, indent=2))
    print(f"[{args.model}] done {len(records)} gens in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
