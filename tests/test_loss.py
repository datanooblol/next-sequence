import math

import pytest
import torch

from next_sequence.core import Feature, TaskType
from next_sequence.loss_fn import UncertaintyWeightedLoss, calculate_loss


def test_calculate_loss_values(features):
    pred = {"item": torch.zeros(2, 6), "price": torch.tensor([[1.0], [3.0]])}
    target = {"item": torch.tensor([1, 2]), "price": torch.tensor([0, 1])}   # int price is cast
    losses = calculate_loss(pred, target, features)
    assert losses["item"].item() == pytest.approx(math.log(6))   # uniform logits
    assert losses["price"].item() == pytest.approx((1 + 4) / 2)


def test_class_zero_is_not_ignored(features):
    """Finding #7: ignore_index=0 used to drop class 0 and give NaN."""
    pred = {"item": torch.zeros(3, 6), "price": torch.zeros(3, 1)}
    target = {"item": torch.tensor([0, 0, 0]), "price": torch.zeros(3)}
    loss = calculate_loss(pred, target, features)["item"]
    assert torch.isfinite(loss) and loss.item() == pytest.approx(math.log(6))


def test_passthrough_is_skipped_everywhere(features):
    """Finding #6: a PASSTHROUGH feature used to crash forward() and weights()."""
    feats = features + [Feature(name="weekday", task=TaskType.PASSTHROUGH, vocab_size=8)]
    crit = UncertaintyWeightedLoss(feats)
    assert set(crit.log_vars.keys()) == {"item", "price"}
    total = crit({"item": torch.tensor(2.0), "price": torch.tensor(4.0)})
    # s = 0 at init: 1 * 2 + 0.5 * 4
    assert total.item() == pytest.approx(4.0)
    assert crit.weights() == {"item": 1.0, "price": 0.5}


def test_uncertainty_formula_and_gradient(features):
    crit = UncertaintyWeightedLoss(features)
    with torch.no_grad():
        crit.log_vars["item"].fill_(math.log(2.0))
        crit.log_vars["price"].fill_(math.log(4.0))
    L_item, L_price = 3.0, 8.0
    total = crit({"item": torch.tensor(L_item), "price": torch.tensor(L_price)})
    expected = (L_item / 2 + 0.5 * math.log(2)) + (0.5 * L_price / 4 + 0.5 * math.log(4))
    assert total.item() == pytest.approx(expected)
    total.backward()
    # dL/ds = -c * exp(-s) * L + 0.5
    assert crit.log_vars["item"].grad.item() == pytest.approx(-L_item / 2 + 0.5)
    assert crit.log_vars["price"].grad.item() == pytest.approx(-0.5 * L_price / 4 + 0.5)


def test_needs_a_predicted_feature():
    with pytest.raises(ValueError):
        UncertaintyWeightedLoss([Feature(name="w", task=TaskType.PASSTHROUGH, vocab_size=3)])
