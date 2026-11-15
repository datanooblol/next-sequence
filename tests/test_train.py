import pytest
import torch
from torch.utils.data import DataLoader

from next_sequence.collate_fn import CustomCollate
from next_sequence.core import CoreDataset
from next_sequence.loss_fn import UncertaintyWeightedLoss
from next_sequence.model import NextSequenceConfig, NextSequenceModel
from next_sequence.train import (
    build_optimizer, build_scheduler, default_device, evaluate, to_device, train_one_epoch,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def make(features, **kw):
    torch.manual_seed(0)
    cfg = NextSequenceConfig(features=features, max_len=8, emb_dim=16, attn_heads=2, dropout=0.0, **kw)
    return NextSequenceModel(cfg), UncertaintyWeightedLoss(features)


def loader(features, funcs, samples, inference=False, batch_size=2):
    collate = CustomCollate(features, "item", funcs, inference=inference, max_len=8)
    return DataLoader(CoreDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collate)


def test_default_device_matches_hardware():
    expected = "cuda" if torch.cuda.is_available() else None
    if expected:
        assert default_device().type == "cuda"
    else:
        assert default_device().type in {"mps", "cpu"}


def test_to_device_handles_nested_and_none():
    batch = {"X": {"a": torch.zeros(1)}, "Y": None, "mask": torch.zeros(1), "extra": [torch.ones(1)]}
    out = to_device(batch, torch.device("cpu"))
    assert out["Y"] is None and isinstance(out["X"], dict) and torch.equal(out["extra"][0], torch.ones(1))


def test_optimizer_groups(features):
    model, crit = make(features)
    opt = build_optimizer(model, crit, lr=1e-3, weight_decay=0.1)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) in in_opt for p in model.parameters())
    assert all(id(p) in in_opt for p in crit.parameters())          # log_vars are trained
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert not any(id(p) in decayed for p in crit.parameters())     # ...without decay
    emb = model.backbone.embedding.embedder["item"].weight
    assert id(emb) not in decayed
    assert id(model.backbone.prepend_cls.cls) not in decayed


def test_scheduler_warmup_then_cosine():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(opt, total_steps=100, warmup_ratio=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[-1] < 0.01 and all(a >= b for a, b in zip(lrs[10:], lrs[11:]))


def test_tiny_dataset_overfits(features, funcs, samples):
    """The whole pipeline can learn: loss must drop sharply on 3 repeated samples."""
    model, crit = make(features)
    dl = loader(features, funcs, samples * 4, batch_size=4)
    opt = build_optimizer(model, crit, lr=3e-3)
    first = evaluate(model, crit, dl)
    for _ in range(60):
        train_one_epoch(model, crit, dl, opt)
    last = evaluate(model, crit, dl)
    assert last["item"] < 0.2 * first["item"]
    assert last["price"] < 0.2 * first["price"]
    assert set(last) == {"loss", "item", "price"}


def test_rejects_inference_loader(features, funcs, samples):
    model, crit = make(features)
    dl = loader(features, funcs, samples, inference=True)
    opt = build_optimizer(model, crit)
    with pytest.raises(ValueError, match="inference=False"):
        train_one_epoch(model, crit, dl, opt)


def test_non_finite_loss_is_reported(features, funcs, samples):
    model, crit = make(features)
    bad = [{**s, "price": [float("nan")] * len(s["price"])} for s in samples]
    opt = build_optimizer(model, crit)
    with pytest.raises(FloatingPointError, match="scaling"):
        train_one_epoch(model, crit, loader(features, funcs, bad), opt)


@cuda_only
def test_trains_on_cuda(features, funcs, samples):
    model, crit = make(features)
    device = torch.device("cuda")
    model, crit = model.to(device), crit.to(device)
    opt = build_optimizer(model, crit)
    metrics = train_one_epoch(model, crit, loader(features, funcs, samples), opt, device=device)
    assert next(model.parameters()).is_cuda and metrics["loss"] == metrics["loss"]
