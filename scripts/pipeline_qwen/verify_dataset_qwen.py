"""
Dataset verifier for Qwen3-4B
==============================
Runs Qwen baseline (no intervention) on every (prompt, condition) pair in
`dataset/refusal_lens_controlled_dataset.json` and reports:

  - bare_refused      : count of prompts whose `bare` baseline REFUSES
                        (excluded prompts will still appear in attribution
                         but cannot be the substrate for anti_refusal_sub)
  - ctrl_leaks        : list of (prompt_id, class) where the ctrl_<class>
                        condition COMPLIES — these will be excluded from
                        the pro_refusal_add intervention (Tejas's
                        bulletproof convention) so we measure flip rate
                        only on (prompt, class) pairs where ctrl was a
                        clean negative control.
  - jb_comply_count   : per-class count of JB framings that successfully
                        bypass refusal (the substrate Stage 06 needs)

This is paper-side hygiene: the dataset was curated for Gemma. Token-count
matching, refusal phrasings, and which JB tactics work all differ between
families. Run this BEFORE Stage 02/06/08 so we know which prompts to
exclude and whether the dataset gives Qwen enough comply-baseline pairs
for a meaningful pro_refusal_add measurement.

Inputs:
    --dataset <path>       Default: dataset/refusal_lens_controlled_dataset.json
    --n-prompts <int>      Default: all 50
    --max-new-tokens <int> Default: 200 (matches Stage 06)
    --out-path <path>      Default: dataset/qwen_dataset_verification.json

Outputs:
    qwen_dataset_verification.json — bare/ctrl/JB classifications + audit
    QWEN_DATASET_VERIFICATION.md   — human-readable summary (next to JSON)

Usage:
    PYTHONPATH=src python3 scripts/pipeline_qwen/verify_dataset_qwen.py
    # smoke
    python3 scripts/pipeline_qwen/verify_dataset_qwen.py --n-prompts 3
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from utils import (
    classify_response,
    generate_baseline,
    is_coherent,
    load_controlled_dataset,
    save_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Verify controlled dataset on Qwen3-4B")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Controlled dataset JSON (default: "
                        "dataset/refusal_lens_controlled_dataset.json)")
    p.add_argument("--n-prompts", type=int, default=None,
                   help="Number of prompts to verify (default: all)")
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="Generation budget per call (default: 200)")
    p.add_argument("--out-path", type=Path, default=None,
                   help="Output JSON path (default: "
                        "dataset/qwen_dataset_verification.json)")
    return p.parse_args()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    out_path = args.out_path or (
        config.REPO_ROOT / "dataset" / "qwen_dataset_verification.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Qwen3-4B Dataset Verifier")
    print("=" * 60)
    rows = load_controlled_dataset(args.dataset, n_prompts=args.n_prompts)
    jb_classes = sorted({k.replace("jb_", "")
                         for k in rows[0]["conditions"] if k.startswith("jb_")})
    print(f"  prompts  : {len(rows)}")
    print(f"  classes  : {jb_classes}")
    print(f"  out_path : {out_path}")

    print(f"\nLoading {config.MODEL_NAME} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    bare_results = []      # one per prompt
    ctrl_results = []      # one per (prompt, class)
    jb_results = []        # one per (prompt, class)
    bare_refused_ids: list[int] = []
    bare_comply_ids: list[int] = []
    ctrl_leak_pairs: list[tuple[int, str]] = []
    jb_comply_pairs: list[tuple[int, str]] = []

    t0 = time.time()
    for i, row in enumerate(rows):
        # ----- bare -----
        resp_bare = generate_baseline(
            model, tokenizer, row["conditions"]["bare"]["text"],
            args.max_new_tokens,
        )
        cls_bare = classify_response(resp_bare)
        rec_bare = {
            "id": row["id"], "topic": row["topic"], "base": row["base"],
            "cls": cls_bare, "coherent": is_coherent(resp_bare),
            "response": resp_bare[:300],
        }
        bare_results.append(rec_bare)
        if cls_bare == "REFUSE":
            bare_refused_ids.append(row["id"])
        else:
            bare_comply_ids.append(row["id"])

        # ----- per-class jb + ctrl -----
        for cls in jb_classes:
            resp_ctrl = generate_baseline(
                model, tokenizer, row["conditions"][f"ctrl_{cls}"]["text"],
                args.max_new_tokens,
            )
            cls_ctrl = classify_response(resp_ctrl)
            ctrl_results.append({
                "id": row["id"], "class": cls, "cls": cls_ctrl,
                "coherent": is_coherent(resp_ctrl),
                "response": resp_ctrl[:300],
            })
            if cls_ctrl != "REFUSE":
                ctrl_leak_pairs.append((row["id"], cls))

            resp_jb = generate_baseline(
                model, tokenizer, row["conditions"][f"jb_{cls}"]["text"],
                args.max_new_tokens,
            )
            cls_jb = classify_response(resp_jb)
            jb_results.append({
                "id": row["id"], "class": cls, "cls": cls_jb,
                "coherent": is_coherent(resp_jb),
                "response": resp_jb[:300],
            })
            if cls_jb == "COMPLY":
                jb_comply_pairs.append((row["id"], cls))

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(rows) - i - 1)
        print(f"  [{i+1}/{len(rows)}] id={row['id']} bare={cls_bare} "
              f"bare_refused={len(bare_refused_ids)} "
              f"ctrl_leaks={len(ctrl_leak_pairs)} "
              f"jb_comply={len(jb_comply_pairs)} "
              f"| {elapsed:.0f}s | eta {eta:.0f}s")

        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate per-class counts
    per_class_jb_comply = {c: 0 for c in jb_classes}
    per_class_ctrl_leak = {c: 0 for c in jb_classes}
    for pid, cls in jb_comply_pairs:
        per_class_jb_comply[cls] += 1
    for pid, cls in ctrl_leak_pairs:
        per_class_ctrl_leak[cls] += 1

    summary = {
        "model": config.MODEL_NAME,
        "n_prompts": len(rows),
        "n_classes": len(jb_classes),
        "jb_classes": jb_classes,
        "max_new_tokens": args.max_new_tokens,
        "bare_refused": len(bare_refused_ids),
        "bare_comply": len(bare_comply_ids),
        "bare_refused_ids": bare_refused_ids,
        "bare_comply_ids": bare_comply_ids,
        "ctrl_leak_count": len(ctrl_leak_pairs),
        "ctrl_leak_pairs": [{"id": pid, "class": cls} for pid, cls in ctrl_leak_pairs],
        "ctrl_leak_per_class": per_class_ctrl_leak,
        "jb_comply_count": len(jb_comply_pairs),
        "jb_comply_pairs": [{"id": pid, "class": cls} for pid, cls in jb_comply_pairs],
        "jb_comply_per_class": per_class_jb_comply,
        "bare_results": bare_results,
        "ctrl_results": ctrl_results,
        "jb_results": jb_results,
    }
    save_json(summary, out_path)
    print(f"\n  Saved {out_path}")

    # Markdown summary
    md_path = out_path.with_suffix(".md")
    pct_bare = len(bare_refused_ids) / len(rows) * 100
    n_ctrl_total = len(rows) * len(jb_classes)
    pct_ctrl_clean = (n_ctrl_total - len(ctrl_leak_pairs)) / n_ctrl_total * 100
    n_jb_total = len(rows) * len(jb_classes)
    pct_jb_comply = len(jb_comply_pairs) / n_jb_total * 100

    lines = [
        f"# Qwen3-4B Dataset Verification\n",
        f"- model: `{config.MODEL_NAME}`",
        f"- dataset: `dataset/refusal_lens_controlled_dataset.json` "
        f"(curated for Gemma — re-verified here for Qwen)",
        f"- prompts: {len(rows)}",
        f"- classes: {jb_classes}",
        f"- max_new_tokens: {args.max_new_tokens}",
        "",
        f"## Headline\n",
        f"| Metric | Count | % |",
        f"|---|---|---|",
        f"| Bare REFUSED (clean baseline) | "
        f"{len(bare_refused_ids)} / {len(rows)} | {pct_bare:.0f}% |",
        f"| Ctrl REFUSED (no leak) | "
        f"{n_ctrl_total - len(ctrl_leak_pairs)} / {n_ctrl_total} | "
        f"{pct_ctrl_clean:.0f}% |",
        f"| JB COMPLIED (substrate for pro_refusal_add) | "
        f"{len(jb_comply_pairs)} / {n_jb_total} | {pct_jb_comply:.0f}% |",
        "",
        f"## Per-class JB comply (substrate count)\n",
        "| Class | JB comply | Ctrl leak |",
        "|---|---|---|",
    ]
    for cls in jb_classes:
        lines.append(f"| {cls} | {per_class_jb_comply[cls]} / {len(rows)} "
                     f"| {per_class_ctrl_leak[cls]} / {len(rows)} |")

    lines.extend([
        "",
        f"## Decisions for downstream stages\n",
        f"- **Stage 06 anti_refusal_sub** runs only on prompts where bare REFUSES → "
        f"{len(bare_refused_ids)} prompts eligible.",
        f"- **Stage 06 pro_refusal_add** runs only on (prompt, class) pairs where "
        f"  - JB COMPLIES (substrate) AND",
        f"  - ctrl REFUSES (clean negative control) → "
        f"  ~{len(set(jb_comply_pairs) - set(ctrl_leak_pairs))} pairs eligible "
        f"(JB-comply minus ctrl-leak).",
        f"- **Stage 08 dissociation matrix** unaffected by ctrl leaks; uses "
        f"all bare+JB pairs.",
        "",
        f"## Bare-COMPLY prompts (excluded for anti_refusal_sub)\n",
    ])
    if bare_comply_ids:
        for pid in bare_comply_ids:
            row = next(r for r in rows if r["id"] == pid)
            lines.append(f"- id={pid} ({row['topic']}): {row['base'][:80]}")
    else:
        lines.append("- _(none — all bare prompts refused, dataset is clean for Qwen)_")
    md_path.write_text("\n".join(lines))
    print(f"  Saved {md_path}")
    print(f"\nDONE in {time.time()-t0:.0f}s")
    print(f"  Bare refused : {len(bare_refused_ids)}/{len(rows)} ({pct_bare:.0f}%)")
    print(f"  Ctrl clean   : {n_ctrl_total - len(ctrl_leak_pairs)}/{n_ctrl_total} ({pct_ctrl_clean:.0f}%)")
    print(f"  JB comply    : {len(jb_comply_pairs)}/{n_jb_total} ({pct_jb_comply:.0f}%)")


if __name__ == "__main__":
    main()
