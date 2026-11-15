import pytest
import torch

from next_sequence.core import Feature, TaskType
from next_sequence.fusion_registry import FusionStrategy
from next_sequence.model import NextSequenceConfig, NextSequenceModel


def make_model(features, **kw):
    cfg = NextSequenceConfig(features=features, max_len=8, emb_dim=16, attn_heads=2, **kw)
    return NextSequenceModel(cfg)


@pytest.mark.parametrize("strategy,fusion_dim", [
    (FusionStrategy.SUM, None), (FusionStrategy.SUM, 32),
    (FusionStrategy.CONCAT, None), (FusionStrategy.CONCAT, 32),
])
def test_output_shapes(features, batch_inputs, strategy, fusion_dim):
    model = make_model(features, fusion_strategy=strategy, fusion_dim=fusion_dim)
    out = model(*batch_inputs)
    assert out["item"].shape == (3, 6)
    assert out["price"].shape == (3, 1)


def test_each_weight_registered_once(features):
    """Finding #8: submodules used to appear three times in state_dict."""
    model = make_model(features)
    n_params = sum(1 for _ in model.parameters())
    assert len(model.state_dict()) == n_params
    assert set(dict(model.named_children())) == {"backbone", "heads"}


def test_all_parameters_get_gradients(features, batch_inputs):
    """Finding #11: an unused fusion.proj used to have no gradients."""
    for strategy in FusionStrategy:
        model = make_model(features, fusion_strategy=strategy)
        out = model(*batch_inputs)
        sum(v.sum() for v in out.values()).backward()
        no_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert no_grad == [], f"{strategy}: {no_grad}"


def test_pad_mask_none(features, batch_inputs):
    """Finding #9: pad_mask=None used to crash."""
    model = make_model(features).eval()
    X, _ = batch_inputs
    with torch.no_grad():
        a = model(X, None)
        b = model(X, torch.zeros(3, 4, dtype=torch.bool))
    assert torch.allclose(a["item"], b["item"])


def test_padding_does_not_change_output(features):
    """Padded positions must not influence the CLS representation."""
    model = make_model(features).eval()
    X_short = {"item": torch.tensor([[2, 3]]), "price": torch.tensor([[[0.1], [0.2]]])}
    X_padded = {"item": torch.tensor([[2, 3, 0, 0]]),
                "price": torch.tensor([[[0.1], [0.2], [0.0], [0.0]]])}
    mask = torch.tensor([[False, False, True, True]])
    with torch.no_grad():
        a, b = model(X_short), model(X_padded, mask)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5)


def test_passthrough_feature_is_input_only(features, batch_inputs):
    feats = features + [Feature(name="weekday", task=TaskType.PASSTHROUGH, vocab_size=8)]
    model = make_model(feats)
    X, mask = batch_inputs
    X = {**X, "weekday": torch.randint(1, 8, (3, 4))}
    out = model(X, mask)
    assert set(out) == {"item", "price"}


def test_rope_strategy_is_used(features):
    """Finding #10: rope_strategy used to be ignored."""
    model = make_model(features)
    assert model.backbone.rope.strategy == model.configs.rope_strategy


def test_too_long_input_fails_clearly(features):
    model = make_model(features)
    X = {"item": torch.ones(1, 9, dtype=torch.long), "price": torch.zeros(1, 9, 1)}
    with pytest.raises(AssertionError, match="max_len"):
        model(X)
