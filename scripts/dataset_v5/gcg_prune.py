"""Post-hoc greedy pruning of a converged GCG suffix (Mask-GCG payoff, off-loop).

Greedily drop suffix tokens whose removal keeps the target loss within `tol` of
the full-suffix loss. Model-agnostic: the caller injects `loss_fn(ids)->float`
(the real one runs a forward pass; tests inject a stub). Yields a compact
high-impact-only suffix for cleaner downstream attribution graphs. `attack_text`
still uses the FULL suffix; the pruned one is stored alongside.
"""
from __future__ import annotations


def prune_suffix(suffix_ids, loss_fn, tol: float = 0.1, max_passes: int = 3) -> dict:
    full_loss = loss_fn(list(suffix_ids))
    kept = list(suffix_ids)
    dropped: list[int] = []
    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(kept):
            trial = kept[:i] + kept[i + 1:]
            if trial and loss_fn(trial) <= full_loss + tol:
                dropped.append(kept[i])
                kept = trial
                changed = True            # do not advance i: re-test the shifted token
            else:
                i += 1
        if not changed:
            break
    pruned_loss = loss_fn(kept) if kept else full_loss
    return {"kept_ids": kept, "dropped_ids": dropped,
            "full_loss": full_loss, "pruned_loss": pruned_loss,
            "asr_held": pruned_loss <= full_loss + tol}
