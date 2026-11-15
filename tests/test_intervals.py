"""Quantile intervals, calibration, positive regression and loader checks through NextSequence."""
import random
import warnings

import pytest
import torch
from torch.utils.data import DataLoader

from next_sequence import (
    CoreDataset, CustomCollate, Feature, Interval, NextSequence, NextSequenceConfig, make_collate,
)


def random_walks(n, seed):
    """Each step adds N(0, 1) noise: the next value is the last one plus noise."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        v = [rng.gauss(0, 1)]
        for _ in range(rng.randint(4, 8)):
            v.append(v[-1] + rng.gauss(0, 1))
        gaps = [abs(rng.gauss(0, 1)) + 0.1 for _ in v]
        out.append({"x": v, "gap": gaps})
    return out


FEATURES = [
    Feature("x", "quantile", is_linear=True, coverage=(0.5, 0.9)),
    Feature("gap", "positive_regression", is_linear=True),
]
FUNCS = {"x": make_collate("float"), "gap": make_collate("float")}


@pytest.fixture(scope="module")
def config():
    return NextSequenceConfig(features=FEATURES, max_len=10, emb_dim=16, attn_heads=2, dropout=0.0)


def loader(config, data, inference=False, shuffle=False):
    collate = CustomCollate.from_config(config, "x", FUNCS, inference=inference)
    return DataLoader(CoreDataset(data), batch_size=64, shuffle=shuffle, collate_fn=collate)


@pytest.fixture(scope="module")
def trained(config):
    train, val, test = random_walks(1500, 0), random_walks(600, 1), random_walks(600, 2)
    model = NextSequence(config, epochs=12, lr=3e-3, seed=0, verbose=False, device="cpu")
    model.fit(loader(config, train, shuffle=True), loader(config, val))
    return model, loader(config, val), loader(config, test)


def test_outputs_and_interval(trained):
    model, _, test_loader = trained
    out = model.predict(test_loader)
    assert out["x"].shape == (600, 5) and out["gap"].shape == (600, 1)
    assert (out["gap"] > 0).all()                                     # positive regression
    assert (out["x"][:, 1:] >= out["x"][:, :-1]).all()                 # bounds ordered
    wide = model.interval(out, "x")                                    # default: widest
    narrow = model.interval(out, "x", coverage=0.5)
    assert isinstance(wide, Interval) and wide.coverage == 0.9 and narrow.coverage == 0.5
    assert torch.equal(wide.median, narrow.median)
    assert ((wide.upper - wide.lower) >= (narrow.upper - narrow.lower)).all()


def test_interval_errors(trained):
    model, _, test_loader = trained
    out = model.predict(test_loader)
    with pytest.raises(ValueError, match=r"trained for coverage \[0.5, 0.9\]"):
        model.interval(out, "x", coverage=0.95)
    with pytest.raises(ValueError, match="needs task='quantile'"):
        model.interval(out, "gap")
    with pytest.raises(KeyError, match="not a feature"):
        model.interval(out, "y")


def test_learned_intervals_are_roughly_right(trained):
    """Next value = last + N(0,1): the true 90% interval is about ±1.64, the 50% about ±0.67."""
    model, _, test_loader = trained
    rows = {r["coverage"]: r for r in model.interval_coverage(test_loader, calibrated=False)}
    assert rows[0.9]["observed"] == pytest.approx(0.9, abs=0.08)
    assert rows[0.5]["observed"] == pytest.approx(0.5, abs=0.1)
    assert rows[0.9]["mean_width"] == pytest.approx(3.29, rel=0.25)


def test_calibration_hits_the_target_on_new_data(trained):
    model, val_loader, test_loader = trained
    offsets = model.calibrate(val_loader)
    assert set(offsets["x"]) == {0.5, 0.9}
    on_val = {r["coverage"]: r["observed"] for r in model.interval_coverage(val_loader)}
    assert on_val[0.9] >= 0.9 and on_val[0.5] >= 0.5                   # guaranteed on calibration data
    on_test = {r["coverage"]: r["observed"] for r in model.interval_coverage(test_loader)}
    assert on_test[0.9] == pytest.approx(0.9, abs=0.04)
    assert on_test[0.5] == pytest.approx(0.5, abs=0.06)
    out = model.predict(test_loader)
    assert (out["x"][:, 1:] >= out["x"][:, :-1]).all()                 # still ordered after shifting
    raw = model.predict(test_loader, calibrated=False)
    assert not torch.equal(raw["x"], out["x"])
    model.calibration_ = None


def test_calibration_is_saved_and_reset_by_training(config, trained, tmp_path):
    model, val_loader, test_loader = trained
    model.calibrate(val_loader)
    model.save(tmp_path / "m.pt")
    loaded = NextSequence.load(tmp_path / "m.pt", device="cpu")
    assert loaded.calibration_ == model.calibration_
    assert torch.allclose(loaded.predict(test_loader)["x"], model.predict(test_loader)["x"], atol=1e-5)
    loaded.warm_start = True
    loaded.epochs = 1
    loaded.fit(loader(config, random_walks(64, 5)))
    assert loaded.calibration_ is None
    model.calibration_ = None


def test_calibrate_needs_quantile_and_targets(config, trained):
    model, _, _ = trained
    with pytest.raises(ValueError, match="inference=True"):
        model.calibrate(loader(config, random_walks(10, 3), inference=True))
    plain = NextSequenceConfig(features=[Feature("gap", "positive_regression", is_linear=True)], max_len=10, emb_dim=16)
    m = NextSequence(plain, epochs=1, verbose=False, device="cpu")
    m.fit(DataLoader(CoreDataset(random_walks(32, 4)), batch_size=16,
                     collate_fn=CustomCollate.from_config(plain, "gap", {"gap": FUNCS["gap"]})))
    with pytest.raises(ValueError, match="task='quantile'"):
        m.calibrate([])


# ---------- loader checks ----------
def test_loader_mismatches_are_caught_before_training(config):
    data = random_walks(32, 6)
    oldest_first = DataLoader(CoreDataset(data), batch_size=16,
                              collate_fn=CustomCollate(FEATURES, "x", FUNCS, max_len=10, recent_first=False))
    model = NextSequence(config, epochs=1, verbose=False, device="cpu")
    with pytest.raises(ValueError, match="recent_first=False but the model expects recent_first=True"):
        model.fit(oldest_first)
    too_long = DataLoader(CoreDataset(data), batch_size=16,
                          collate_fn=CustomCollate(FEATURES, "x", FUNCS, max_len=50))
    with pytest.raises(ValueError, match="supports max_len=10"):
        model.fit(too_long)
    with pytest.raises(ValueError, match="inference=True"):
        model.fit(loader(config, data, inference=True))
    other = [Feature("gap", "positive_regression", is_linear=True), FEATURES[0]]
    reordered = DataLoader(CoreDataset(data), batch_size=16,
                           collate_fn=CustomCollate(other, "x", FUNCS, max_len=10))
    with pytest.raises(ValueError, match="from_config"):
        model.fit(reordered)
    assert not model.is_fitted                                          # nothing was trained


def test_missing_max_len_warns(config):
    data = random_walks(32, 7)
    no_limit = DataLoader(CoreDataset(data), batch_size=16, collate_fn=CustomCollate(FEATURES, "x", FUNCS))
    with pytest.warns(UserWarning, match="no max_len"):
        NextSequence(config, epochs=1, verbose=False, device="cpu").fit(no_limit)


def test_from_config_loaders_raise_no_warnings(config):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NextSequence(config, epochs=1, verbose=False, device="cpu").fit(loader(config, random_walks(32, 8)))
