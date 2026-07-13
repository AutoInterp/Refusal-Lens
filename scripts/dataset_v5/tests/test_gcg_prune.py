from gcg_prune import prune_suffix


def test_drops_zero_impact_tokens():
    # tokens 2 and 4 are "filler": loss depends only on the SET of high-impact ids {1,3,5}
    HIGH = {1, 3, 5}

    def loss_fn(ids):
        return -float(len(HIGH & set(ids)))     # more high-impact ids kept -> lower loss

    out = prune_suffix([1, 2, 3, 4, 5], loss_fn, tol=0.1)
    assert set(out["kept_ids"]) == HIGH
    assert set(out["dropped_ids"]) == {2, 4}
    assert out["asr_held"] is True


def test_keeps_all_when_every_token_matters():
    def loss_fn(ids):
        return -float(len(ids))                 # every dropped token raises loss by 1

    out = prune_suffix([9, 8, 7], loss_fn, tol=0.1)
    assert out["dropped_ids"] == []
    assert out["kept_ids"] == [9, 8, 7]


def test_reports_full_and_pruned_loss():
    def loss_fn(ids):
        return -float(len(set(ids) & {1}))

    out = prune_suffix([1, 2], loss_fn, tol=0.1)
    assert out["full_loss"] == -1.0
    assert out["pruned_loss"] == -1.0           # dropping filler token 2 keeps loss
