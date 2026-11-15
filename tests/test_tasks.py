import json

import pytest
import torch

from next_sequence.core import Feature, TaskType
from next_sequence.loss_fn import UncertaintyWeightedLoss, calculate_loss
from next_sequence.model import NextSequenceConfig, NextSequenceModel
from next_sequence.tasks import TASKS, get_task


def test_every_task_type_has_a_task():
    assert set(TASKS) == set(TaskType)


# ---------- coverage / quantiles on Feature ----------
def test_quantile_defaults_to_95():
    f = Feature("p", "quantile", is_linear=True)
    assert f.coverages == (0.95,)
    assert f.quantiles == (0.025, 0.5, 0.975)


@pytest.mark.parametrize("coverage,expected", [
    (0.5, (0.25, 0.5, 0.75)),
    (0.25, (0.375, 0.5, 0.625)),
    ((0.95, 0.5), (0.025, 0.25, 0.5, 0.75, 0.975)),      # any order in, sorted out
    ([0.8], (0.1, 0.5, 0.9)),
])
def test_any_coverage_level(coverage, expected):
    assert Feature("p", "quantile", is_linear=True, coverage=coverage).quantiles == pytest.approx(expected)


@pytest.mark.parametrize("coverage,match", [
    (0, "strictly between 0 and 1"), (1, "strictly between 0 and 1"), (95, "strictly between 0 and 1"),
    (-0.5, "strictly between"), ("0.95", "strictly between"), (True, "strictly between"),
    ((), "at least one"), ((0.5, 0.5), "unique"), (float("nan"), "strictly between"),
])
def test_invalid_coverage(coverage, match):
    with pytest.raises(ValueError, match=match):
        Feature("p", "quantile", is_linear=True, coverage=coverage)


def test_coverage_only_for_quantile():
    with pytest.raises(ValueError, match="only used with task=quantile"):
        Feature("p", "regression", is_linear=True, coverage=0.9)


def test_quantile_feature_round_trip():
    f = Feature("p", "quantile", is_linear=True, coverage=(0.5, 0.95))
    assert Feature.from_dict(json.loads(json.dumps(f.to_dict()))) == f


# ---------- model outputs ----------
@pytest.fixture
def mixed_features():
    return [
        Feature("item", "classification", vocab_size=6),
        Feature("gap", "positive_regression", is_linear=True),
        Feature("price", "quantile", is_linear=True, coverage=(0.5, 0.95)),
    ]


def make_inputs(B=8, T=5):
    torch.manual_seed(0)
    return {"item": torch.randint(1, 6, (B, T)), "gap": torch.rand(B, T, 1),
            "price": torch.randn(B, T, 1)}


def test_output_shapes(mixed_features):
    model = NextSequenceModel(NextSequenceConfig(features=mixed_features, max_len=8, emb_dim=16))
    out = model(make_inputs())
    assert out["item"].shape == (8, 6)
    assert out["gap"].shape == (8, 1)
    assert out["price"].shape == (8, 5)


def test_positive_output_is_always_positive(mixed_features):
    model = NextSequenceModel(NextSequenceConfig(features=mixed_features, max_len=8, emb_dim=16))
    with torch.no_grad():
        for p in model.heads.heads["gap"].parameters():
            p.fill_(-50.0)                         # push the raw output very negative
        out = model(make_inputs())
    assert (out["gap"] > 0).all()


def test_quantile_bounds_never_cross():
    f = Feature("p", "quantile", is_linear=True, coverage=(0.25, 0.5, 0.95))
    raw = torch.randn(1000, len(f.quantiles)) * 10          # arbitrary raw outputs
    pred = get_task(f).transform(raw, f)
    assert (pred[:, 1:] >= pred[:, :-1]).all()
    assert torch.equal(pred[:, 3], raw[:, 3])                  # median column untouched


# ---------- losses ----------
def test_pinball_loss_value():
    f = Feature("p", "quantile", is_linear=True, coverage=0.5)  # quantiles 0.25, 0.5, 0.75
    pred = torch.tensor([[0.0, 1.0, 2.0]])
    loss = get_task(f).loss(pred, torch.tensor([1.0]), f)
    # errors y - q = [1, 0, -1] -> 0.25*1, 0, (0.75-1)*-1 = 0.25 -> mean = 0.5/3
    assert loss.item() == pytest.approx(0.5 / 3)


def test_quantile_loss_learns_the_right_quantiles():
    """Fitting constants to N(0, 1) samples must recover its quantiles."""
    f = Feature("p", "quantile", is_linear=True, coverage=0.8)   # 0.1, 0.5, 0.9
    torch.manual_seed(0)
    y = torch.randn(20000)
    raw = torch.zeros(1, 3, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.05)
    for _ in range(600):
        pred = get_task(f).transform(raw, f).expand(len(y), 3)
        loss = get_task(f).loss(pred, y, f)
        opt.zero_grad(); loss.backward(); opt.step()
    learned = get_task(f).transform(raw, f).detach().squeeze()
    assert learned.tolist() == pytest.approx([-1.2816, 0.0, 1.2816], abs=0.06)


def test_positive_regression_rejects_negative_targets():
    f = Feature("gap", "positive_regression", is_linear=True)
    with pytest.raises(ValueError, match="dividing only"):
        calculate_loss({"gap": torch.ones(2, 1)}, {"gap": torch.tensor([1.0, -0.5])}, [f])


def test_class_target_out_of_range_is_explained():
    f = Feature("age", "classification", is_linear=True, n_classes=3)
    with pytest.raises(ValueError, match=r"0\.\.2.*start at 0"):
        calculate_loss({"age": torch.zeros(2, 3)}, {"age": torch.tensor([1, 3])}, [f])


def test_missing_target_is_explained():
    f = Feature("gap", "positive_regression", is_linear=True)
    with pytest.raises(KeyError, match="collate function"):
        calculate_loss({"gap": torch.ones(2, 1)}, {}, [f])


def test_weight_scales(mixed_features):
    crit = UncertaintyWeightedLoss(mixed_features)
    assert crit.weights() == {"item": 1.0, "gap": 0.5, "price": 1.0}
