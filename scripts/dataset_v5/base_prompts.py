"""Canonical 50 base prompts (base_id 1..50), extracted from dataset_v4.json.

dataset_v4 has exactly one clean record per base_id 1..50; we dedupe by base_id
and keep (base_id, base, topic). This is the shared base set every jailbreak
class in the project is built on.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_V4 = REPO / "new_dataset_results/refusal_results/dataset_v4.json"


def load_base_prompts(v4_path: Path = DEFAULT_V4) -> list[dict]:
    records = json.loads(Path(v4_path).read_text())["records"]
    by_id: dict[int, dict] = {}
    for r in records:
        bid = r.get("base_id")
        if bid is None or bid in by_id:
            continue
        by_id[bid] = {"base_id": bid, "base": r["base"], "topic": r["topic"]}
    return [by_id[i] for i in sorted(by_id)]
