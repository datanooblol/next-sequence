import math

import pytest
import torch

from next_sequence import targets
from next_sequence.collate_fn import check_collate, make_collate


def test_next_value_and_decode():
    t = targets.NextValue(scale=2.0, shift=10.0)
    assert t([5.0, 14.0]) == 2.0
    assert t.decode(torch.tensor([[2.0], [-1.0]])).tolist() == [14.0, 8.0]


def test_delta_and_decode():
    t = targets.Delta(scale=5.0)
    assert t([100.0, 110.0]) == 2.0
    assert t.decode(torch.tensor([[2.0]]), last=[110.0]).tolist() == [120.0]


def test_log_ratio_stays_positive():
    t = targets.LogRatio(scale=0.5)
    assert t([20.0, 40.0]) == pytest.approx(math.log(2) / 0.5)
    decoded = t.decode(torch.tensor([-50.0, 0.0, 3.0]), last=[21.0, 21.0, 21.0])
    assert (decoded > 0).all() and decoded[1].item() == pytest.approx(21.0)
    with pytest.raises(ValueError, match="values > 0"):
        t([0.0, 5.0])


def test_quantile_columns_decode_together():
    t = targets.Delta(scale=10.0)
    pred = torch.tensor([[-1.0, 0.0, 1.0]])                   # lower, median, upper
    assert t.decode(pred, last=[50.0]).tolist() == [[40.0, 50.0, 60.0]]


def test_buckets():
    b = targets.Bucket([0.0, 10.0], labels=["low", "mid", "high"])
    assert [b([v]) for v in (-1.0, 0.0, 5.0, 10.0, 11.0)] == [0, 0, 1, 1, 2]
    assert b.n_classes == 3
    assert b.decode(torch.tensor([[0.1, 0.9, 0.0], [0.0, 0.0, 5.0]])) == ["mid", "high"]
    d = targets.DeltaBucket([-30, 30], labels=["down", "flat", "up"])
    assert [d.labels[d(h)] for h in ([100, 40], [100, 120], [20, 180])] == ["down", "flat", "up"]


@pytest.mark.parametrize("make,match", [
    (lambda: targets.Delta(scale=0), "positive"),
    (lambda: targets.NextValue(scale=-1), "positive"),
    (lambda: targets.Bucket([]), "at least one"),
    (lambda: targets.Bucket([1, 1]), "strictly increasing"),
    (lambda: targets.DeltaBucket([0], labels=["a"]), "need 2 labels"),
])
def test_invalid_settings(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_decode_needs_last():
    with pytest.raises(ValueError, match="needs last="):
        targets.Delta().decode(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="2 values but there are 1"):
        targets.Delta().decode(torch.tensor([1.0]), last=[1.0, 2.0])


def test_target_std_uses_every_step():
    histories = [[0.0, 1.0, 3.0], [10.0, 10.0]]               # deltas: 1, 2, 0
    assert targets.target_std(histories, targets.Delta()) == pytest.approx(math.sqrt(2 / 3))
    assert targets.target_std([[5.0]], targets.Delta()) == 1.0


def test_value_picks_from_tuple_steps():
    steps = [(-0.5, 100.0), (0.3, 130.0)]                      # (scaled, raw)
    t = targets.Delta(scale=10.0, value=lambda s: s[1])
    fn = make_collate("float", x=lambda s: s[0], y=t)
    x, y = fn([steps])
    assert x[0].tolist() == [-0.5] and y == [3.0]
    assert check_collate(fn, [steps])["targets"] == [3.0]


def test_short_history_message_through_make_collate():
    with pytest.raises(ValueError, match="at least 2 steps"):
        make_collate("float", y=targets.Delta())([[1.0]])
