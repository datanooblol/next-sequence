# next-sequence

Multi-task **next-event prediction** for sequences, built on PyTorch.

Give it a history of events per user (or session, device, ...) with several features per event,
and it predicts the next event: categorical features as **classification** (logits) and numeric
features as **regression**, all at once. One transformer encoder reads the whole sequence through a
learnable CLS token, and the task losses are balanced automatically with learned uncertainty
weighting ([Kendall et al., 2018](https://arxiv.org/abs/1705.07115)).

- Python 3.12+, Windows / macOS / Linux
- Uses **CUDA automatically** when available (then Apple MPS, then CPU)
- You keep control of the data: you build the samples and DataLoaders, the library does the model and the training

## Install

```bash
pip install next-sequence
```

The library depends on `torch>=2.2` and works with whatever PyTorch build is installed.
**For an NVIDIA GPU, install a CUDA build of PyTorch first:**

| Platform | What `pip install next-sequence` gives you | For GPU |
|---|---|---|
| Linux | CUDA-enabled torch from PyPI | nothing extra |
| Windows | CPU-only torch from PyPI | `pip install torch --index-url https://download.pytorch.org/whl/cu132` **before** installing next-sequence |
| macOS | torch with Apple GPU (MPS) | nothing extra |

Pick the CUDA version that matches your driver at [pytorch.org/get-started](https://pytorch.org/get-started/locally/).
With uv: `uv pip install torch --torch-backend=auto` detects it for you.

Check what will be used:

```python
import next_sequence as ns
print(ns.default_device())    # cuda / mps / cpu
```

## Quick start

```python
from torch.utils.data import DataLoader
from next_sequence import (
    CoreDataset, CustomCollate, Feature, NextSequence, NextSequenceConfig, TaskType, create_bow,
)

# 1. describe the features
features = [
    Feature("item", TaskType.CLASSIFICATION, vocab_size=len(bow)),         # predict next item
    Feature("price", TaskType.REGRESSION, is_linear=True),                  # predict next price
    Feature("weekday", TaskType.PASSTHROUGH, vocab_size=8),                 # input only
]
config = NextSequenceConfig(features=features, max_len=50, emb_dim=64, attn_heads=4)

# 2. your samples + one conversion function per feature (see the cookbook)
collate = CustomCollate(features, "item", funcs, max_len=50)
train_loader = DataLoader(CoreDataset(train), batch_size=64, shuffle=True, collate_fn=collate)
val_loader = DataLoader(CoreDataset(val), batch_size=64, collate_fn=collate)
test_loader = DataLoader(CoreDataset(test), batch_size=64, collate_fn=collate)

# 3. train, predict, save
model = NextSequence(config, epochs=20, lr=3e-4, seed=0)
model.fit(train_loader, val_loader)            # keeps the best epoch by validation loss
out = model.predict(test_loader, mask_pad=True)
out["item"]     # (N, vocab_size) logits  -> .argmax(-1) or .softmax(-1)
out["price"]    # (N, 1) values, in the scale you trained on

model.save("model.pt", metadata={"bow": bow, "price_mean": m, "price_std": s})
model = NextSequence.load("model.pt")          # a GPU-trained model also loads on CPU
model.metadata_["bow"]
```

A complete, runnable version is in [examples/quickstart.py](examples/quickstart.py).
**Read the [cookbook](docs/cookbook.md)** for how the data must look, how to split it, and the
preprocessing pitfalls.

## Features

Each input column is a `Feature`. Its `task` says what to predict from it:

| You have | `Feature(...)` | Head output | Loss |
|---|---|---|---|
| Categorical input, predict the next token | `Feature("item", "classification", vocab_size=len(bow))` | `(N, vocab_size)` logits | cross entropy |
| Numeric input, predict the next value | `Feature("price", "regression", is_linear=True)` | `(N, 1)` | MSE |
| Numeric input, predict a bucket | `Feature("qty", "classification", is_linear=True, n_classes=3)` | `(N, 3)` logits | cross entropy |
| Input only, nothing predicted | `Feature("weekday", "passthrough", vocab_size=8)` | none | none |

- `vocab_size` is the number of token ids of a categorical **input**, including PAD (0) and UNK (1).
- `n_classes` is the number of classes of a classification **target**. It defaults to `vocab_size`.

## Model configuration

`NextSequenceConfig` validates everything up front and can be saved with `to_dict()` / `from_dict()`.

| Argument | Default | Meaning |
|---|---|---|
| `features` | required | list of `Feature` |
| `max_len` | required | longest input sequence. Use the same value in `CustomCollate(max_len=...)` |
| `emb_dim` | 128 | width of each feature embedding |
| `fusion_dim` | `emb_dim` | width after combining the features |
| `fusion_strategy` | `"sum"` | `"sum"` or `"concat"` to combine feature embeddings |
| `attn_heads` | 2 | attention heads. `fusion_dim / attn_heads` must be an even integer |
| `num_layers` | 2 | transformer blocks |
| `expansion` | 2 | feed-forward width multiplier |
| `dropout` | 0.1 | dropout in the transformer blocks |

`NextSequence(config, ...)` adds the training settings: `epochs`, `lr`, `weight_decay`, `warmup_ratio`,
`grad_clip`, `device`, `seed`, `verbose`, `warm_start`.

## Good to know

- **The total loss can go negative.** The uncertainty weighting adds `0.5 * log(sigma^2)` per task, which
  is negative once the model is confident. Compare the per-feature losses (`train_item`, `val_price`, ...)
  instead. `model.task_weights()` shows the learned weights.
- **Scale numeric inputs and regression targets** with statistics from the training split, or the
  regression loss will swamp everything else. Undo the scaling after `predict`.
- **`predict` needs a loader with `shuffle=False`**, so outputs line up with your samples.
- **Lower-level building blocks** are public too: `NextSequenceModel` (a plain `nn.Module`),
  `UncertaintyWeightedLoss`, `calculate_loss`, and `next_sequence.train` (`build_optimizer`,
  `build_scheduler`, `train_one_epoch`, `evaluate`) for writing your own training loop.

## Development

```bash
uv sync                                   # CUDA build of torch on Windows/Linux (default)
uv sync --no-group gpu --group cpu        # CPU build instead
uv run pytest
uv run python examples/quickstart.py
```

## License

MIT
