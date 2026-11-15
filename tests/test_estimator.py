import pytest
import torch
from torch.utils.data import DataLoader

import next_sequence as ns
from next_sequence import (
    CoreDataset, CustomCollate, Feature, NextSequence, NextSequenceConfig, NotFittedError,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class Custom:
    """A non-basic type: allowed in metadata only with trusted=True."""


@pytest.fixture
def config(features):
    return NextSequenceConfig(features=features, max_len=8, emb_dim=16, attn_heads=2, dropout=0.0)


@pytest.fixture
def loaders(features, funcs, samples):
    def make(inference=False, shuffle=False, data=None):
        collate = CustomCollate(features, "item", funcs, inference=inference, max_len=8)
        return DataLoader(CoreDataset(data or samples * 4), batch_size=4, shuffle=shuffle, collate_fn=collate)
    return make


def test_public_api_is_exported():
    for name in ns.__all__:
        assert hasattr(ns, name), name
    assert isinstance(ns.__version__, str)


def test_init_only_stores_settings(config):
    est = NextSequence(config, epochs=3, device="cpu")
    assert not est.is_fitted and est.epochs == 3
    with pytest.raises(NotFittedError):
        est.predict([])
    with pytest.raises(NotFittedError):
        est.save("x.pt")


@pytest.mark.parametrize("kwargs", [dict(epochs=0), dict(lr=0), dict(warmup_ratio=1.0)])
def test_init_validation(config, kwargs):
    with pytest.raises(ValueError):
        NextSequence(config, **kwargs)


def test_device_defaults_to_cuda_when_available(config):
    est = NextSequence(config)
    assert est.device.type == ("cuda" if torch.cuda.is_available() else est.device.type)


def test_fit_predict_shapes_and_learning(config, loaders):
    est = NextSequence(config, epochs=40, lr=3e-3, device="cpu", seed=0, verbose=False)
    est.fit(loaders(), loaders())
    assert len(est.history_) == 40 and est.best_epoch_ is not None
    assert est.history_[-1]["train_loss"] < est.history_[0]["train_loss"]
    out = est.predict(loaders(inference=True))
    assert out["item"].shape == (12, 6) and out["price"].shape == (12, 1)
    assert out["item"].device.type == "cpu"
    # on the training inputs ([2,3], [2], [3,3,3,4]) it learned the next items 4, 5, 5;
    # a training-mode loader is accepted and its Y is ignored
    seen = est.predict(loaders())
    assert seen["item"].argmax(-1)[:3].tolist() == [4, 5, 5]
    assert set(est.task_weights()) == {"item", "price"}


def test_best_epoch_weights_are_restored(config, loaders):
    est = NextSequence(config, epochs=5, device="cpu", seed=0, verbose=False).fit(loaders(), loaders())
    best = min(est.history_, key=lambda r: r["val_loss"])
    from next_sequence.train import evaluate
    now = evaluate(est.model_, est.criterion_, loaders())
    assert now["loss"] == pytest.approx(best["val_loss"], rel=1e-5)
    assert est.best_epoch_ == best["epoch"]


def test_seed_makes_runs_reproducible(config, loaders):
    a = NextSequence(config, epochs=2, device="cpu", seed=123, verbose=False).fit(loaders())
    b = NextSequence(config, epochs=2, device="cpu", seed=123, verbose=False).fit(loaders())
    assert a.history_ == b.history_


def test_refit_restarts_unless_warm_start(config, loaders):
    est = NextSequence(config, epochs=2, device="cpu", seed=0, verbose=False).fit(loaders())
    est.fit(loaders())
    assert [r["epoch"] for r in est.history_] == [1, 2]
    warm = NextSequence(config, epochs=2, device="cpu", seed=0, verbose=False, warm_start=True)
    warm.fit(loaders()).fit(loaders())
    assert [r["epoch"] for r in warm.history_] == [1, 2, 3, 4]


def test_predict_rejects_shuffled_loader(config, loaders):
    est = NextSequence(config, epochs=1, device="cpu", verbose=False).fit(loaders())
    with pytest.raises(ValueError, match="shuffle=False"):
        est.predict(loaders(inference=True, shuffle=True))


def test_mask_pad(config, loaders):
    est = NextSequence(config, epochs=1, device="cpu", verbose=False).fit(loaders())
    out = est.predict(loaders(inference=True), mask_pad=True)
    assert torch.isinf(out["item"][:, 0]).all() and torch.isfinite(out["item"][:, 1:]).all()
    explicit = est.predict(loaders(inference=True), mask_pad=["item"])
    assert torch.equal(out["item"], explicit["item"])
    with pytest.raises(ValueError, match="classification"):
        est.predict(loaders(inference=True), mask_pad=["price"])


def test_mask_pad_skips_bucket_targets(features, loaders):
    """A numeric input with bucket classes has a real class 0, so mask_pad=True leaves it alone."""
    feats = features + [Feature("age", "classification", is_linear=True, n_classes=3)]
    est = NextSequence(NextSequenceConfig(features=feats, max_len=8, emb_dim=16, attn_heads=2),
                       device="cpu", verbose=False)
    assert est._pad_masked_features(True) == {"item"}


def test_save_load_round_trip(config, loaders, tmp_path):
    est = NextSequence(config, epochs=3, device="cpu", seed=0, verbose=False).fit(loaders(), loaders())
    meta = {"bow": {"PAD": 0, "UNK": 1, "a": 2}, "price_mean": 0.5, "price_std": 0.2}
    path = tmp_path / "sub" / "model.pt"
    est.save(path, metadata=meta)

    loaded = NextSequence.load(path, device="cpu")
    assert loaded.metadata_ == meta
    assert loaded.history_ == est.history_ and loaded.best_epoch_ == est.best_epoch_
    assert loaded.config == est.config and loaded.epochs == 3
    a, b = est.predict(loaders(inference=True)), loaded.predict(loaders(inference=True))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_load_rejects_foreign_or_unsafe_files(config, loaders, tmp_path):
    other = tmp_path / "other.pt"
    torch.save({"hello": 1}, other)
    with pytest.raises(ValueError, match="not a next-sequence checkpoint"):
        NextSequence.load(other)

    est = NextSequence(config, epochs=1, device="cpu", verbose=False).fit(loaders())
    unsafe = tmp_path / "unsafe.pt"
    est.save(unsafe, metadata={"obj": Custom()})
    with pytest.raises(RuntimeError, match="trusted=True"):
        NextSequence.load(unsafe)


@cuda_only
def test_cuda_train_then_load_on_cpu(config, loaders, tmp_path):
    est = NextSequence(config, epochs=2, device="cuda", seed=0, verbose=False).fit(loaders(), loaders())
    assert next(est.model_.parameters()).is_cuda
    path = tmp_path / "gpu.pt"
    est.save(path)
    cpu = NextSequence.load(path, device="cpu")
    assert next(cpu.model_.parameters()).device.type == "cpu"
    a, b = est.predict(loaders(inference=True)), cpu.predict(loaders(inference=True))
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-4)
