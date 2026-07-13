"""Generate Gemma responses for dataset_v5 (full length, no truncation).

Clean copy of Tejas's new_dataset_results/refusal_results/generate_v4 (1).py
(logic unchanged) so the runbook has a space-free, attributed entry point. All
three v5 classes bake the whole attack into attack_text, so no per-record
generate-time step is needed.

    export HF_TOKEN=...
    python generate.py --dataset dataset_v5.json --out v5_generations.json
"""
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_gemma(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}],
                                   tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="v5_generations.json")
    args = ap.parse_args()

    records = json.load(open(args.dataset))["records"]
    print(f"[gen] loading {args.model} fp32")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 device_map="cuda").eval()

    jobs = [(i, r["attack_text"]) for i, r in enumerate(records) if r.get("attack_text")]
    out = {"metadata": {"model": args.model, "max_new_tokens": args.max_new_tokens,
                        "n_records": len(records), "n_generations": len(jobs)},
           "generations": []}
    t0 = time.time()
    for k, (ridx, text) in enumerate(jobs, 1):
        r = records[ridx]
        ids = tok(format_gemma(tok, text), return_tensors="pt").to(model.device)
        plen = ids.input_ids.shape[1]
        with torch.no_grad():
            g = model.generate(**ids, do_sample=False, max_new_tokens=args.max_new_tokens,
                               pad_token_id=tok.eos_token_id)
        resp = tok.decode(g[0][plen:], skip_special_tokens=True)
        ended = resp.rstrip().endswith((".", "!", "?", '"', ")", "`"))
        gen = {
            "record_idx": ridx, "class": r["class"], "kind": "attack",
            "base_id": r.get("base_id"), "base": r.get("base"),
            "attack_text": text, "prompt_text": text, "response": resp,
            "n_chars": len(resp), "ended_naturally": ended}
        if "sweep_k" in r:
            gen["sweep_k"] = r["sweep_k"]
        out["generations"].append(gen)
        if k % 10 == 0 or k == len(jobs):
            print(f"[{k}/{len(jobs)}] {r['class']} chars={len(resp)} ended={ended} ({time.time()-t0:.0f}s)")
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[gen] wrote {args.out}")


if __name__ == "__main__":
    main()
