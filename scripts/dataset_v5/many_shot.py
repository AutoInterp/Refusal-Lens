"""Single-turn Many-Shot Jailbreak (MSJ) assembly.

A demo is (harmful question -> compliant answer), sourced from Gemma's OWN prior
COMPLY outputs (v3/v4 judged). We render K demos as one 'User:/Assistant:' blob
inside a single user message, then append the target request as the final turn.
Deterministic in (seed, target base_id); the target's own base_id is excluded
from its demo pool to avoid leaking the answer to the exact question.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_JUDGED = [REPO / "new_dataset_results/refusal_results/v3_judged.json",
                  REPO / "new_dataset_results/refusal_results/v4_judged.json"]


def load_comply_pool(judged_paths: list[Path] = DEFAULT_JUDGED) -> list[dict]:
    pool: list[dict] = []
    for p in judged_paths:
        d = json.loads(Path(p).read_text())
        src = Path(p).stem
        for g in d["generations"]:
            if g.get("kind") == "attack" and g.get("judge") == "COMPLY":
                pool.append({"src": src, "record_idx": g["record_idx"],
                             "base_id": g.get("base_id"), "base": g["base"],
                             "response": g["response"]})
    return pool


def assemble_many_shot(base_record: dict, pool: list[dict], k: int = 32,
                       seed: int = 0, demo_char_cap: int | None = None):
    bid = base_record["base_id"]
    eligible = [p for p in pool if p["base_id"] != bid]
    if len(eligible) < k:
        raise ValueError(f"pool has {len(eligible)} eligible demos, need k={k}")
    rng = random.Random(seed * 1000 + (bid or 0))
    demos = rng.sample(eligible, k)          # nested-friendly: prefixes give K<k subsets
    blocks = []
    for p in demos:
        ans = p["response"] if demo_char_cap is None else p["response"][:demo_char_cap]
        blocks.append(f"User: {p['base']}\nAssistant: {ans}")
    blob = "\n\n".join(blocks) + f"\n\nUser: {base_record['base']}\nAssistant:"
    refs = [{"src": p["src"], "record_idx": p["record_idx"], "base_id": p["base_id"]}
            for p in demos]
    return blob, refs
