import json

import pytest
import torch

from next_sequence.core import Feature, TaskType
from next_sequence.fusion_registry import FusionStrategy
from next_sequence.model import NextSequenceConfig, NextSequenceModel


# ---------- Feature ----------
def test_feature_accepts_strings_and_defaults_n_classes():
    f = Feature("item", "classification", vocab_size=7)
    assert f.task is TaskType.CLASSIFICATION and f.n_classes == 7


def test_numeric_input_with_class_target():
    """Finding #14: numeric input, bucketed target, now explicit via n_classes."""
    f = Feature("age", "classification", is_linear=True, n_classes=3)
    assert f.n_classes == 3


def test_legacy_vocab_size_for_class_count_still_works():
    f = Feature("age", "classification", is_linear=True, vocab_size=4)
    assert f.n_classes == 4


@pytest.mark.parametrize("kwargs,match", [
    (dict(name="", task="regression", is_linear=True), "non-empty"),
    (dict(name="a.b", task="regression", is_linear=True), "'.'"),
    (dict(name="x", task="classification"), "categorical input needs vocab_size"),
    (dict(name="x", task="classification", is_linear=True), "n_classes >= 2"),
    (dict(name="x", task="regression", is_linear=True, n_classes=3), "only used with"),
    (dict(name="x", task="nope", is_linear=True), "nope"),
])
def test_feature_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Feature(**kwargs)


def test_feature_round_trip():
    f = Feature("age", "classification", is_linear=True, n_classes=3)
    assert Feature.from_dict(json.loads(json.dumps(f.to_dict()))) == f


# ---------- NextSequenceConfig ----------
def cfg(features=None, **kw):
    base = dict(features=features, max_len=8, emb_dim=16, attn_heads=2)
    return NextSequenceConfig(**{**base, **kw})


@pytest.mark.parametrize("kwargs,match", [
    (dict(features=[]), "must not be empty"),
    (dict(max_len=0), "max_len"),
    (dict(emb_dim=-1), "emb_dim"),
    (dict(num_layers=0), "num_layers"),
    (dict(dropout=1.0), "dropout"),
    (dict(fusion_dim=30, attn_heads=4), "divisible"),
    (dict(fusion_dim=18, attn_heads=2), "must be even"),
    (dict(fusion_strategy="mean"), "mean"),
])
def test_config_validation(features, kwargs, match):
    kwargs = {"features": features, **kwargs}
    with pytest.raises(ValueError, match=match):
        cfg(**kwargs)


def test_config_rejects_duplicates_and_all_passthrough(features):
    with pytest.raises(ValueError, match="duplicate"):
        cfg(features + [features[0]])
    with pytest.raises(ValueError, match="at least one feature"):
        cfg([Feature("w", "passthrough", vocab_size=4)])
    with pytest.raises(TypeError):
        cfg([{"name": "item"}])


def test_config_round_trip_is_json_safe(features):
    """Finding #17: config with enums and Features can now be saved and rebuilt."""
    c = cfg(features, fusion_strategy=FusionStrategy.CONCAT, fusion_dim=32)
    data = json.loads(json.dumps(c.to_dict()))
    assert data["schema_version"] == 2
    assert data["recent_first"] is True
    assert NextSequenceConfig.from_dict(data) == c


def test_rebuilt_config_loads_same_weights(features, batch_inputs):
    torch.manual_seed(0)
    m1 = NextSequenceModel(cfg(features)).eval()
    m2 = NextSequenceModel(NextSequenceConfig.from_dict(m1.configs.to_dict())).eval()
    m2.load_state_dict(m1.state_dict())
    with torch.no_grad():
        a, b = m1(*batch_inputs), m2(*batch_inputs)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_newer_schema_is_rejected(features):
    data = cfg(features).to_dict()
    data["schema_version"] = 99
    with pytest.raises(ValueError, match="newer"):
        NextSequenceConfig.from_dict(data)


def test_numeric_input_dtype_is_cast(features):
    """Finding #3: a long tensor for a numeric input used to crash nn.Linear."""
    model = NextSequenceModel(cfg(features)).eval()
    X = {"item": torch.tensor([[2, 3]]), "price": torch.tensor([[[1], [2]]])}   # long price
    with torch.no_grad():
        out = model(X)
    assert out["price"].dtype == torch.float32


def test_schema_1_configs_load_as_oldest_first(features):
    data = cfg(features).to_dict()
    data["schema_version"] = 1
    data.pop("recent_first")
    assert NextSequenceConfig.from_dict(data).recent_first is False


def test_recent_first_must_be_bool(features):
    with pytest.raises(ValueError, match="recent_first"):
        cfg(features, recent_first="yes")
