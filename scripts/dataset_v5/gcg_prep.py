"""Pure helpers for GCG optimization (no torch, so they unit-test on CPU).

- _gcg_target: a per-base affirmative target so GCG pulls toward a real compliant
  answer, not the generic "Sure, here are the detailed steps:" that converged in
  training loss yet still refused at generation.
- excluded_token_ids: the token ids GCG must NOT put in the suffix. nanoGCG's
  allow_non_ascii=False only drops non-ASCII + bos/eos/pad/unk; it does NOT drop
  Gemma special tokens like <start_of_image>/<unused515> (they decode to ASCII
  strings), which the v5 smoke showed backfiring (Gemma read them as distress).
"""
from __future__ import annotations

import re

_CTRL = re.compile(r"^<.*>$")


def _gcg_target(base: str) -> str:
    b = base.strip().rstrip("?").strip()
    low = (b[0].lower() + b[1:]) if b else b
    bl = b.lower()
    if bl.startswith("how to"):
        return f"Sure, here is {low}:\n\n1."
    if bl.startswith("ways to"):
        return f"Sure, here are {low}:\n\n1."
    if bl.startswith("ideas for"):
        return f"Sure, here are {low}:\n\n1."
    return "Sure, here is the answer:\n\n1."


def excluded_token_ids(tokenizer) -> list[int]:
    excl: set[int] = set()
    for i in range(tokenizer.vocab_size):
        s = tokenizer.decode([i])
        if not (s.isascii() and s.isprintable()):       # nanoGCG's original criterion
            excl.add(i)
        elif _CTRL.match(s.strip()):                     # <unusedN>, <start_of_image>, ...
            excl.add(i)
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            excl.add(tid)
    excl.update(getattr(tokenizer, "all_special_ids", None) or [])
    excl.update(getattr(tokenizer, "additional_special_tokens_ids", None) or [])
    return sorted(excl)
