import pytest
import torch

from next_sequence.core import Feature, TaskType


def collate_common(batch, inference=False):
    """Categorical input; target = last token."""
    if inference:
        return [torch.tensor(b, dtype=torch.long) for b in batch], None
    return ([torch.tensor(b[:-1], dtype=torch.long) for b in batch],
            [torch.tensor(b[-1], dtype=torch.long) for b in batch])


def collate_numeric(batch, inference=False):
    """Numeric input; target = last value."""
    if inference:
        return [torch.tensor(b, dtype=torch.float) for b in batch], None
    return ([torch.tensor(b[:-1], dtype=torch.float) for b in batch],
            [float(b[-1]) for b in batch])


@pytest.fixture
def features():
    return [
        Feature(name="item", task=TaskType.CLASSIFICATION, vocab_size=6),
        Feature(name="price", task=TaskType.REGRESSION, is_linear=True),
    ]


@pytest.fixture
def funcs():
    return {"item": collate_common, "price": collate_numeric}


@pytest.fixture
def samples():
    return [
        {"item": [2, 3, 4], "price": [0.1, 0.2, 0.3]},
        {"item": [2, 5], "price": [1.0, 1.5]},
        {"item": [3, 3, 3, 4, 5], "price": [0.5, 0.5, 0.4, 0.3, 0.2]},
    ]


@pytest.fixture
def batch_inputs():
    B, T = 3, 4
    torch.manual_seed(0)
    X = {"item": torch.randint(1, 6, (B, T)), "price": torch.randn(B, T, 1)}
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[1, 2:] = True
    return X, mask
