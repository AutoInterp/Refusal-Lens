"""Assemble an N-column attribution-graph compare site.

Fetches each configured run (Gemma variants + Qwen) from the HF dataset into
one parent dir, builds a compare_manifest.json (shared prompt x condition
options + per-column slug maps), and stages the manifest-driven compare harness.
Pure functions (parse_condition, build_compare_manifest) are unit-tested; the
CLI (main) does fetch + filesystem assembly.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PATCHES = Path(__file__).resolve().parent / "05_frontend_patches"
_SLUG_RE = re.compile(r"^(\d+)_(.+)$")


def parse_condition(slug: str, mode_suffixes=("single", "multi")):
    """Return (idx, condition) with any trailing mode suffix stripped, or None."""
    m = _SLUG_RE.match(slug)
    if not m:
        return None
    idx, cond = m.group(1), m.group(2)
    parts = cond.split("_")
    if len(parts) > 1 and parts[-1] in mode_suffixes:
        cond = "_".join(parts[:-1])
    return idx, cond


def build_compare_manifest(columns: list[dict], title: str) -> dict:
    """columns: [{label, dir, model, target, graphs:[{slug, prompt}]}].
    Offers only (idx, cond) present in ALL columns; resolves each column's
    actual slug per (idx, cond)."""
    per_col = []
    for c in columns:
        smap, ptext = {}, {}
        for g in c["graphs"]:
            pc = parse_condition(g["slug"])
            if not pc:
                continue
            idx, cond = pc
            smap[f"{idx}_{cond}"] = g["slug"]
            ptext.setdefault(idx, g.get("prompt", "") or "")
        per_col.append({**c, "smap": smap, "ptext": ptext})

    keysets = [set(col["smap"].keys()) for col in per_col]
    shared = sorted(set.intersection(*keysets)) if keysets else []
    idxs = sorted({k.split("_", 1)[0] for k in shared}, key=int)
    conds = sorted({k.split("_", 1)[1] for k in shared})

    ptext0 = per_col[0]["ptext"] if per_col else {}
    return {
        "title": title,
        "columns": [{"label": col["label"], "dir": col["dir"], "model": col["model"],
                     "target": col["target"],
                     "slugmap": {k: col["smap"][k] for k in shared}}
                    for col in per_col],
        "prompts": [{"idx": i, "text": ptext0.get(i, "")} for i in idxs],
        "conditions": conds,
    }
