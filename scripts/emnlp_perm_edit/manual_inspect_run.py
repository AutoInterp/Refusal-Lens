"""Rerun the edge-ablation experiment with FULL completions for manual inspection.

Fixes the issues found on 2026-06-15:
  * Qwen runs with enable_thinking=False (baseline previously ran thinking-ON).
  * Longer generation (default 256 tok) so refuse/comply is unambiguous.
  * FULL completions saved (no 300-char cap).
  * Human-readable annotated .md dump, grouped + labeled by condition, with the
    keyword classification shown ONLY as a hint — the text is the ground truth.

Runs two settings per model: baseline (no hook) and ablate_all_edges
(subtract the all_signed edge contribution along r_hat). 5 prompts default.
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
from utils import classify_response, format_prompt, is_coherent, load_controlled_dataset  # noqa: E402

# Per-model config: layer, r_hat source, decomposition, thinking flag, hook position.
MODELS = {
    "gemma": {
        "hf": "google/gemma-3-4b-it", "layer": 15, "enable_thinking": None,
        "rhat": REPO / "data/results/pipeline_runs/run_20260430_023247/01_direction/unnormalized_r.pt",
        "rhat_kind": "dict",  # dict keyed by layer
        "decomp": REPO / "data/results/emnlp_perm_edit/phase0_controllability/linearization_decomposition.json",
        "delta_to_unnorm": False,  # Gemma decomp already in unnormalized-r units (raw delta)
    },
    "qwen": {
        "hf": "Qwen/Qwen3-4B", "layer": 18, "enable_thinking": False,
        "rhat": REPO / "data/results/pipeline_runs_qwen/run_regen_L18/01_direction/positions_L18/pos_-1_unnormalized.pt",
        "rhat_kind": "tensor",
        "decomp": REPO / "data/results/emnlp_perm_edit/phase0_controllability/qwen_linearization_decomposition.json",
        # Qwen decomp is in NORMALIZED-r units -> convert: delta_unnorm = ||r_hat|| * delta_norm
        "delta_to_unnorm": True,
    },
}


def load_rhat(cfg):
    obj = torch.load(cfg["rhat"], weights_only=False, map_location="cpu")
    if cfg["rhat_kind"] == "dict":
        return obj[cfg["layer"]].float()
    if isinstance(obj, torch.Tensor):
        return obj.float()
    return obj["direction"].float()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompts", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODELS[args.model]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    r_hat = load_rhat(cfg)
    print(f"[{args.model}] ||r_hat[L{cfg['layer']}]|| = {r_hat.norm().item():.3f}")
    decomp = json.loads(Path(cfg["decomp"]).read_text())
    per_prompt = {(r["prompt_idx"], r["condition"]): r for r in decomp["per_prompt"]}

    tok = AutoTokenizer.from_pretrained(cfg["hf"])
    model = AutoModelForCausalLM.from_pretrained(cfg["hf"], torch_dtype=torch.float32, device_map="cuda")
    model.eval()
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    target_layer = layers[cfg["layer"]]
    pad_id = tok.eos_token_id

    dataset = load_controlled_dataset(REPO / "dataset/refusal_lens_controlled_dataset.json")[:args.max_prompts]

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
    for setting in ("baseline", "ablate_all_edges"):
        for pidx, prompt in enumerate(dataset):
            for cond, blob in prompt["conditions"].items():
                hook_fn = None
                delta = 0.0
                if setting == "ablate_all_edges":
                    rec = per_prompt.get((pidx, cond))
                    if rec is None:
                        continue
                    delta = float(rec["all_signed"])
                    if cfg["delta_to_unnorm"]:
                        delta *= r_hat.norm().item()  # normalized-r -> unnormalized-r basis
                    hook_fn = make_scalar_rhat_subtraction_hook(r_hat, delta, position_mode="all")
                resp = gen(blob["text"], hook_fn)
                records.append({
                    "model": args.model, "setting": setting, "prompt_idx": pidx,
                    "condition": cond, "prompt_text": blob["text"], "delta": delta,
                    "response": resp, "classification": classify_response(resp),
                    "coherent": is_coherent(resp)})
                print(f"  [{len(records)}] {setting} p{pidx} {cond} -> {records[-1]['classification']}")
                args.out_json.parent.mkdir(parents=True, exist_ok=True)
                args.out_json.write_text(json.dumps({"records": records}, indent=2))

    # Readable annotated markdown, grouped by setting then condition.
    lines = [f"# {cfg['hf']} — edge-ablation manual inspection",
             f"max_new_tokens={args.max_new_tokens}, enable_thinking={cfg['enable_thinking']}, "
             f"n_prompts={len(dataset)}, layer={cfg['layer']}, ||r_hat||={r_hat.norm().item():.3f}",
             "",
             "Keyword `classification` is a HINT only — read the text for ground truth.", ""]
    for setting in ("baseline", "ablate_all_edges"):
        lines.append(f"\n## SETTING: {setting}\n")
        for cond in dataset[0]["conditions"]:
            lines.append(f"\n### condition: {cond}\n")
            for r in records:
                if r["setting"] == setting and r["condition"] == cond:
                    lines.append(f"**[prompt {r['prompt_idx']}]** keyword=`{r['classification']}` "
                                 f"coherent={r['coherent']} delta={r['delta']:.1f}")
                    lines.append(f"> PROMPT: {r['prompt_text']}")
                    lines.append("```")
                    lines.append(r["response"].strip())
                    lines.append("```\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines))
    print(f"[{args.model}] done {len(records)} gens in {time.time()-t0:.0f}s -> {args.out_md}")


if __name__ == "__main__":
    main()
