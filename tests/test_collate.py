import pytest

from next_sequence.core import Feature
import torch

from next_sequence.collate_fn import CustomCollate


def test_training_batch_shapes(features, funcs, samples):
    out = CustomCollate(features, "item", funcs, recent_first=False)(samples)
    assert isinstance(out["X"], dict) and isinstance(out["Y"], dict)   # not defaultdict
    assert out["X"]["item"].shape == (3, 4)            # longest input = 5 - 1
    assert out["X"]["price"].shape == (3, 4, 1)        # is_linear -> trailing dim
    assert out["Y"]["item"].tolist() == [4, 5, 5]
    assert out["Y"]["item"].dtype == torch.long
    assert out["Y"]["price"].dtype == torch.float32
    assert out["lengths"].tolist() == [2, 1, 4]
    assert out["mask"].tolist()[1] == [False, True, True, True]


def test_inference_returns_no_targets(features, funcs, samples):
    """Finding #1: inference mode used to crash on y=None."""
    out = CustomCollate(features, "item", funcs, inference=True)(samples)
    assert out["Y"] is None
    assert out["X"]["item"].shape == (3, 5)            # full sequence at inference
    assert out["lengths"].tolist() == [3, 2, 5]


def test_max_len_keeps_most_recent_steps(features, funcs, samples):
    """Finding #2 / D4: inputs longer than max_len are truncated from the left."""
    out = CustomCollate(features, "item", funcs, inference=True, max_len=3, recent_first=False)(samples)
    assert out["X"]["item"].shape == (3, 3)
    assert out["X"]["item"][2].tolist() == [3, 4, 5]   # last 3 of [3, 3, 3, 4, 5]
    assert out["lengths"].tolist() == [3, 2, 3]
    # default recent_first=True: truncate first (keep newest), then newest step first
    out = CustomCollate(features, "item", funcs, inference=True, max_len=3)(samples)
    assert out["X"]["item"][2].tolist() == [5, 4, 3]


def test_missing_y_in_training_mode_is_reported(features, samples):
    bad = {"item": lambda b, inf: ([torch.tensor(x[:-1]) for x in b], None),
           "price": lambda b, inf: ([torch.tensor(x[:-1]) for x in b], None)}
    with pytest.raises(ValueError, match="y=None"):
        CustomCollate(features, "item", bad)(samples)


def test_bad_configuration_fails_early(features, funcs):
    with pytest.raises(ValueError, match="feature_mask"):
        CustomCollate(features, "nope", funcs)
    with pytest.raises(ValueError, match="missing"):
        CustomCollate(features, "item", {"item": funcs["item"]})


# ---------- recent_first, from_config, input checks ----------
from next_sequence.collate_fn import check_collate, make_collate      # noqa: E402
from next_sequence.model import NextSequenceConfig                    # noqa: E402


def test_recent_first_keeps_padding_on_the_right(features, funcs, samples):
    out = CustomCollate(features, "item", funcs)(samples)
    assert out["X"]["item"][1].tolist() == [2, 0, 0, 0]          # [2, 5] -> input [2], padded right
    assert out["X"]["item"][0].tolist() == [3, 2, 0, 0]          # [2, 3, 4] -> input [2, 3] reversed
    assert out["mask"][0].tolist() == [False, False, True, True]
    assert out["Y"]["item"].tolist() == [4, 5, 5]                 # targets unaffected


def test_from_config_copies_model_settings(features, funcs):
    config = NextSequenceConfig(features=features, max_len=7, emb_dim=16, recent_first=False)
    collate = CustomCollate.from_config(config, "item", funcs, inference=True)
    assert (collate.max_len, collate.recent_first, collate.inference) == (7, False, True)
    assert [f.name for f in collate.features] == ["item", "price"]


def test_features_with_different_lengths_are_rejected(features, funcs):
    bad = [{"item": [2, 3, 4], "price": [0.1, 0.2]}]
    with pytest.raises(ValueError, match="same length"):
        CustomCollate(features, "item", funcs)(bad)


def test_missing_sample_key_is_explained(features, funcs):
    with pytest.raises(KeyError, match="no key 'price'"):
        CustomCollate(features, "item", funcs)([{"item": [2, 3]}])


def test_unknown_func_is_rejected(features, funcs):
    with pytest.raises(ValueError, match="unknown features"):
        CustomCollate(features, "item", {**funcs, "pricee": funcs["price"]})


def test_non_1d_inputs_are_rejected(features, samples):
    bad = {"item": lambda v, inf: ([torch.zeros(2, 2) for _ in v], None if inf else [1] * len(v)),
           "price": lambda v, inf: ([torch.zeros(2) for _ in v], None if inf else [0.0] * len(v))}
    with pytest.raises(ValueError, match="1-D"):
        CustomCollate(features, "item", bad)(samples)


def test_make_collate_branches_and_targets():
    fn = make_collate("float", y=lambda h: h[-1] - h[-2])
    x, y = fn([[1.0, 4.0, 9.0], [2.0, 2.5]])
    assert [t.tolist() for t in x] == [[1.0, 4.0], [2.0]] and y == [5.0, 0.5]
    x_inf, y_inf = fn([[1.0, 4.0, 9.0]], True)
    assert x_inf[0].tolist() == [1.0, 4.0, 9.0] and y_inf is None
    assert check_collate(fn, [[1.0, 4.0, 9.0], [2.0, 2.5]])["input_dtype"] == torch.float32


def test_make_collate_explains_short_histories():
    with pytest.raises(ValueError, match="at least 2 steps"):
        make_collate("long")([[5]])
    with pytest.raises(ValueError, match="longer history"):
        make_collate("float", y=lambda h: h[-1] - h[-3])([[1.0, 2.0]])
    with pytest.raises(ValueError, match="dtype"):
        make_collate("double")


def test_check_collate_catches_disagreeing_branches():
    def bad(values, inference=False):
        if inference:
            return [torch.tensor(v, dtype=torch.long) for v in values], None
        return [torch.tensor(v[:-1], dtype=torch.float) for v in values], [v[-1] for v in values]
    with pytest.raises(ValueError, match="training inputs are torch.float32 but inference inputs are torch.int64"):
        check_collate(bad, [[25, 26, 27]])


# ---------- target leakage ----------
def _forgot_to_drop_last(values, inference=False):
    xs = [torch.tensor(v) for v in values]
    return (xs, None) if inference else (xs, [v[-1] for v in values])   # input still holds the target


def test_leaking_function_is_rejected_in_training(features, funcs, samples):
    leaky = {**funcs, "item": _forgot_to_drop_last}
    with pytest.raises(ValueError, match=r"still contain the last step.*x\[:-1\].*check_leakage=False"):
        CustomCollate(features, "item", leaky)(samples)


def test_leaking_function_never_reaches_training():
    """The case that used to train silently to 99% accuracy on unlearnable data."""
    from torch.utils.data import DataLoader
    from next_sequence import CoreDataset, Feature, NextSequence
    feats = [Feature("item", "classification", vocab_size=8)]
    cfg = NextSequenceConfig(features=feats, max_len=8, emb_dim=16)
    data = [{"item": [2, 3, 4, 5]}] * 8
    loader = DataLoader(CoreDataset(data), batch_size=4,
                        collate_fn=CustomCollate.from_config(cfg, "item", {"item": _forgot_to_drop_last}))
    model = NextSequence(cfg, epochs=1, verbose=False, device="cpu")
    with pytest.raises(ValueError, match="copy the answer"):
        model.fit(loader)


def test_inference_mode_is_not_checked(features, funcs, samples):
    leaky = {**funcs, "item": _forgot_to_drop_last}
    out = CustomCollate(features, "item", leaky, inference=True)(samples)   # full history is correct here
    assert out["X"]["item"].shape == (3, 5)


def test_leakage_check_can_be_disabled(features, funcs, samples):
    """Steps that store a label for the NEXT event make the full history a valid input."""
    leaky = {**funcs, "item": _forgot_to_drop_last, "price": lambda v, inf: (
        [torch.tensor(x) for x in v], None if inf else [x[-1] for x in v])}
    out = CustomCollate(features, "item", leaky, check_leakage=False)(samples)
    assert out["X"]["item"].shape == (3, 5)
    config = NextSequenceConfig(features=features, max_len=8, emb_dim=16)
    assert CustomCollate.from_config(config, "item", leaky, check_leakage=False)(samples)["Y"] is not None


def test_shorter_derived_inputs_pass(features, funcs, samples):
    """Inputs shorter than the history (e.g. differences) are fine."""
    def diffs(values, inference=False):
        xs = [torch.tensor(v, dtype=torch.float).diff() for v in values]    # length n - 1
        return (xs, None) if inference else ([x[:-1] for x in xs], [v[-1] for v in values])
    feats = [Feature("price", "regression", is_linear=True)]
    out = CustomCollate(feats, "price", {"price": diffs})([{"price": [1.0, 2.0, 4.0, 7.0]}])
    assert out["X"]["price"].shape == (1, 2, 1)
