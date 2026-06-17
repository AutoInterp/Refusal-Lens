"""No-intervention baseline generation for the edge-ablation experiment.

Generates with NO hook (delta=0) so we can measure the un-ablated COMPLY rate
per condition — the baseline against which ablation flip rates are defined.

Mirrors the generation settings of 00_edge_ablation_runtime{,_qwen}.py:
greedy, max_new_tokens=80, float32, format_prompt chat template.
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
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))
from utils import classify_response, format_prompt, is_coherent, load_controlled_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--max-prompts", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"[baseline] loading model {args.model} in float32")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    pad_id = tok.eos_token_id

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")
    if args.max_prompts:
        dataset = dataset[:args.max_prompts]

    results = {"metadata": {"model": args.model, "n_prompts": len(dataset),
                            "max_new_tokens": args.max_new_tokens, "intervention": "none"},
               "records": []}
    t0 = time.time()
    n = 0
    for prompt_idx, prompt in enumerate(dataset):
        for cond, blob in prompt["conditions"].items():
            formatted = format_prompt(tok, blob["text"])
            ids = tok(formatted, return_tensors="pt").to(model.device)
            plen = ids.input_ids.shape[1]
            with torch.no_grad():
                out = model.generate(**ids, do_sample=False,
                                     max_new_tokens=args.max_new_tokens, pad_token_id=pad_id)
            resp = tok.decode(out[0][plen:], skip_special_tokens=True)
            results["records"].append({
                "prompt_idx": prompt_idx, "condition": cond,
                "response": resp[:300], "classification": classify_response(resp),
                "coherent": is_coherent(resp)})
            n += 1
            if n % 20 == 0:
                print(f"  [{n}] elapsed={time.time()-t0:.0f}s")
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(results, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"[baseline] done {n} gens in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
