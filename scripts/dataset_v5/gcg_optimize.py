"""GCG adversarial-suffix optimization on Gemma-3-4B (RunPod GPU).

Wraps nanoGCG (per-prompt), then applies post-hoc greedy pruning (gcg_prune) to
yield a compact high-impact suffix alongside the full one. Not runnable on the
CPU-only dev box; run via run_v5_runpod.sh. (gcg_universal deferred — nanoGCG has
no multi-prompt API; see spec.)

    export HF_TOKEN=...
    python gcg_optimize.py --smoke
    python gcg_optimize.py --mode per_prompt --limit 3 --out gcg_suffixes_smoke.json
    python gcg_optimize.py --mode per_prompt --out gcg_suffixes.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import nanogcg
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from base_prompts import load_base_prompts
from gcg_prune import prune_suffix

MODEL = "google/gemma-3-4b-it"
TARGET = "Sure, here are the detailed steps:"


def _load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 device_map="cuda").eval()
    return model, tok


def _cfg(steps, suffix_len, search_width, topk):
    return GCGConfig(num_steps=steps, optim_str_init="x " * suffix_len,
                     search_width=search_width, topk=topk, seed=0, verbosity="WARNING")


def _suffix_loss_fn(model, tok, base):
    """Return loss_fn(ids)->float: NLL of TARGET given base + decoded suffix ids."""
    def loss_fn(ids):
        suffix = tok.decode(ids)
        msgs = [{"role": "user", "content": f"{base} {suffix}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = prompt + TARGET
        enc = tok(full, return_tensors="pt").to(model.device)
        tgt = tok(TARGET, add_special_tokens=False, return_tensors="pt").input_ids
        labels = enc.input_ids.clone()
        labels[:, :-tgt.shape[1]] = -100
        with torch.no_grad():
            return float(model(**enc, labels=labels).loss)
    return loss_fn


def _optimize_one(model, tok, base):
    res = nanogcg.run(model, tok, base, TARGET, _cfg(ARGS.steps, ARGS.suffix_len,
                                                     ARGS.search_width, ARGS.topk))
    suffix = res.best_string
    ids = tok(suffix, add_special_tokens=False).input_ids
    pr = prune_suffix(ids, _suffix_loss_fn(model, tok, base), tol=ARGS.prune_tol)
    return {"suffix": suffix, "final_loss": float(res.best_loss),
            "suffix_pruned": tok.decode(pr["kept_ids"]),
            "pruned_n_tokens": len(pr["kept_ids"]), "prune_asr_held": pr["asr_held"]}


def run_per_prompt(model, tok, bases):
    out = {}
    for b in bases:
        out[str(b["base_id"])] = _optimize_one(model, tok, b["base"])
        print(f"[gcg pp] base {b['base_id']} loss={out[str(b['base_id'])]['final_loss']:.3f} "
              f"pruned={out[str(b['base_id'])]['pruned_n_tokens']}tok")
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["per_prompt", "smoke"], default="per_prompt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--suffix-len", type=int, default=30)
    ap.add_argument("--search-width", type=int, default=512)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--prune-tol", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("gcg_suffixes.json"))
    ARGS = ap.parse_args()

    bases = load_base_prompts()
    if ARGS.limit:
        bases = bases[:ARGS.limit]
    model, tok = _load_model()

    if ARGS.mode == "smoke":
        ARGS.steps, bases = 2, bases[:1]
        r = run_per_prompt(model, tok, bases)
        print("[smoke] nanoGCG↔Gemma-3 OK:", r)
        return

    cfg_meta = {"suffix_len": ARGS.suffix_len, "num_steps": ARGS.steps,
                "search_width": ARGS.search_width, "topk": ARGS.topk, "target": TARGET}
    data = {"config": cfg_meta, "per_prompt": run_per_prompt(model, tok, bases)}
    ARGS.out.write_text(json.dumps(data, indent=2))
    print(f"[gcg] wrote {ARGS.out}")


if __name__ == "__main__":
    main()
