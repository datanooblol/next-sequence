# next-sequence

Multi-task **next-event prediction** for sequences, built on PyTorch.

Give it a history of events per user (or session, device, ...) with several features per event,
and it predicts the next event: categorical features as **classification** (logits), numeric features
as **regression** (any sign or always positive) or as **prediction intervals** with any coverage, all at
once. One transformer encoder reads the whole sequence through a learnable CLS token, and the task
losses are balanced automatically with learned uncertainty weighting
([Kendall et al., 2018](https://arxiv.org/abs/1705.07115)).

- Python 3.12+, Windows / macOS / Linux
- Uses **CUDA automatically** when available (then Apple MPS, then CPU)
- You keep control of the data: you build the samples and DataLoaders, the library does the model and the training
- Mistakes are caught early with messages that say what to change

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
    CoreDataset, CustomCollate, Feature, NextSequence, NextSequenceConfig, TaskType, make_collate,
)

# 1. describe the features
features = [
    Feature("item", TaskType.CLASSIFICATION, vocab_size=len(bow)),         # predict next item
    Feature("price", TaskType.REGRESSION, is_linear=True),                  # predict next price
    Feature("weekday", TaskType.PASSTHROUGH, vocab_size=8),                 # input only
]
config = NextSequenceConfig(features=features, max_len=50, emb_dim=64, attn_heads=4)

# 2. one collate function per feature (last step = target), and DataLoaders over your samples
funcs = {"item": make_collate("long"), "price": make_collate("float"), "weekday": make_collate("long")}
collate = CustomCollate.from_config(config, "item", funcs)     # max_len etc. always match the model
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

### Targets beyond "the next value"

`make_collate(y=...)` takes a ready-made **target strategy**, or any function of the history. Each
strategy also decodes predictions back into real units:

```python
from next_sequence import targets

delta = targets.Delta(scale=targets.target_std(train_prices, targets.Delta()))
funcs["price"] = make_collate("float", y=delta)                 # predict the change
next_price = delta.decode(out["price"], last=last_known_price)  # back to prices
```

| Strategy | Predicts |
|---|---|
| `targets.NextValue(scale, shift)` | the next value, standardized |
| `targets.Delta(scale)` | the change since the last step |
| `targets.LogRatio(scale)` | the relative change (decoded values keep their sign) |
| `targets.Bucket(edges)` / `targets.DeltaBucket(edges)` | the value / the change, as classes |

### Prediction intervals

```python
Feature("price", "quantile", is_linear=True, coverage=(0.5, 0.95))   # any levels, several at once

out = model.predict(test_loader)
iv = model.interval(out, "price", coverage=0.95)    # iv.lower, iv.median, iv.upper
model.interval_coverage(val_loader)                 # does 95% really cover 95%?
model.calibrate(val_loader)                         # adjust the intervals to match (saved with the model)
```

## Cookbooks

Step-by-step notebooks with their outputs, so you can read along or run them yourself
(they also need `pandas`: `pip install next-sequence pandas`):

| Notebook | What it shows |
|---|---|
| [01 · Quickstart](https://github.com/datanooblol/next-sequence/blob/main/cookbooks/01_quickstart.ipynb) | the whole workflow: raw events → train → predict → save |
| [02 · Data preparation](https://github.com/datanooblol/next-sequence/blob/main/cookbooks/02_data_preparation.ipynb) | how samples must look, splitting by time, vocabulary, scaling, `CustomCollate` |
| [03 · Custom collate functions](https://github.com/datanooblol/next-sequence/blob/main/cookbooks/03_custom_collate_functions.ipynb) | targets: next value, **delta**, **log ratio**, **buckets**, your own rule, several targets per column |
| [04 · Features and model](https://github.com/datanooblol/next-sequence/blob/main/cookbooks/04_features_and_model.ipynb) | the task types, configuration, validation, model outputs |
| [05 · Training](https://github.com/datanooblol/next-sequence/blob/main/cookbooks/05_training.ipynb) | `fit`, history, best epoch, task weights, reproducibility, `warm_start` |
| [06 · Prediction and saving](https://github.com/datanooblol/next-sequence/blob/main/cookbooks/06_prediction_and_saving.ipynb) | evaluation vs. real prediction, decoding, top-k, `save` / `load`, new data |
| [07 · Prediction intervals](https://github.com/datanooblol/next-sequence/blob/main/cookbooks/07_prediction_intervals.ipynb) | positive values, intervals with any coverage, measuring and calibrating coverage |
| [08 · Custom training loop](https://github.com/datanooblol/next-sequence/blob/main/cookbooks/08_custom_training_loop.ipynb) | the lower-level building blocks for your own loop |

## Features and task types

Each input column is a `Feature`. Its `task` says what to predict from it:

| You have | `Feature(...)` | Output per sample | Loss |
|---|---|---|---|
| Categorical input, predict the next token | `Feature("item", "classification", vocab_size=len(bow))` | `vocab_size` logits | cross entropy |
| Numeric input, predict a value (any sign) | `Feature("price", "regression", is_linear=True)` | 1 value | MSE |
| Numeric input, predict a value that is **> 0** | `Feature("gap", "positive_regression", is_linear=True)` | 1 value, always > 0 | MSE |
| Numeric input, predict an **interval** | `Feature("price", "quantile", is_linear=True, coverage=(0.5, 0.95))` | median + bounds (5 here) | pinball |
| Numeric input, predict a bucket | `Feature("qty", "classification", is_linear=True, n_classes=3)` | 3 logits | cross entropy |
| Input only, nothing predicted | `Feature("weekday", "passthrough", vocab_size=8)` | none | none |

- `vocab_size` is the number of token ids of a categorical **input**, including PAD (0) and UNK (1).
- `n_classes` is the number of classes of a classification **target**. It defaults to `vocab_size`.
- `coverage` is the interval(s) of a quantile feature: any values in (0, 1), default `0.95`. A narrower
  level gives a narrower interval because it promises less (a 50% interval misses half the time).
- `positive_regression` targets must be >= 0: scale them by dividing only, without subtracting the mean.

## Model configuration

`NextSequenceConfig` validates everything up front and can be saved with `to_dict()` / `from_dict()`.

| Argument | Default | Meaning |
|---|---|---|
| `features` | required | list of `Feature` |
| `max_len` | required | longest input sequence; `CustomCollate.from_config` truncates to it |
| `emb_dim` | 128 | width of each feature embedding |
| `fusion_dim` | `emb_dim` | width after combining the features |
| `fusion_strategy` | `"sum"` | `"sum"` or `"concat"` to combine feature embeddings |
| `attn_heads` | 2 | attention heads. `fusion_dim / attn_heads` must be an even integer |
| `num_layers` | 2 | transformer blocks |
| `expansion` | 2 | feed-forward width multiplier |
| `dropout` | 0.1 | dropout in the transformer blocks |
| `recent_first` | `True` | feed each history newest event first, so the latest event sits next to the CLS token |

`NextSequence(config, ...)` adds the training settings: `epochs`, `lr`, `weight_decay`, `warmup_ratio`,
`grad_clip`, `device`, `seed`, `verbose`, `warm_start`.

## Good to know

- **The total loss can go negative.** The uncertainty weighting adds `0.5 * log(sigma^2)` per task, which
  is negative once the model is confident. Compare the per-feature losses (`train_item`, `val_price`, ...)
  instead. `model.task_weights()` shows the learned weights.
- **Scale numeric inputs and regression targets** with statistics from the training split, or the
  regression loss will swamp everything else. Undo the scaling after `predict` (target strategies do it with `decode`).
- **Build collates with `CustomCollate.from_config(config, ...)`.** `fit` and `predict` reject a loader whose
  features, `max_len` or input order don't match the model, before doing any work.
- **Store your lists oldest first.** With `recent_first=True` the collate reverses them for the model.
- **`predict` needs a loader with `shuffle=False`**, so outputs line up with your samples.
- **Lower-level building blocks** are public too: `NextSequenceModel` (a plain `nn.Module`),
  `UncertaintyWeightedLoss`, `calculate_loss`, and `next_sequence.train` (`build_optimizer`,
  `build_scheduler`, `train_one_epoch`, `evaluate`) for writing your own training loop.

## Development

```bash
uv sync                                   # CUDA build of torch on Windows/Linux (default)
uv sync --no-group gpu --group cpu        # CPU build instead
uv run pytest                             # includes running every cookbook
uv run pytest -m "not cookbook"           # quick run without the notebooks
```

The cookbooks are committed with their outputs. After changing them, re-run each notebook so the
saved outputs stay current.

## License

MIT
