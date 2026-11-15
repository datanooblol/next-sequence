import pytest
import torch

from next_sequence.collate_fn import CustomCollate


def test_training_batch_shapes(features, funcs, samples):
    out = CustomCollate(features, "item", funcs)(samples)
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
    out = CustomCollate(features, "item", funcs, inference=True, max_len=3)(samples)
    assert out["X"]["item"].shape == (3, 3)
    assert out["X"]["item"][2].tolist() == [3, 4, 5]   # last 3 of [3, 3, 3, 4, 5]
    assert out["lengths"].tolist() == [3, 2, 3]


def test_missing_y_in_training_mode_is_reported(features, samples):
    bad = {"item": lambda b, inf: ([torch.tensor(x) for x in b], None),
           "price": lambda b, inf: ([torch.tensor(x) for x in b], None)}
    with pytest.raises(ValueError, match="y=None"):
        CustomCollate(features, "item", bad)(samples)


def test_bad_configuration_fails_early(features, funcs):
    with pytest.raises(ValueError, match="feature_mask"):
        CustomCollate(features, "nope", funcs)
    with pytest.raises(ValueError, match="missing"):
        CustomCollate(features, "item", {"item": funcs["item"]})
