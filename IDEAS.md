# IDEAS: training API for next-sequence

Status: draft for planning. Nothing here is implemented yet.

## Decisions so far

| Topic | Decision |
|---|---|
| Data preparation | **Outside the model.** The user splits the data, builds `CustomCollate`, and creates the DataLoaders. |
| Training input | `fit(train_loader, val_loader=None)` takes DataLoaders. |
| Prediction input | `predict(test_loader)` takes a DataLoader. |
| Prediction output | **Raw outputs per task**: classification returns logits `(N, C)`, regression returns values `(N, 1)`. No softmax, no decoding, no unscaling. |
| Optimization | **Outside the `nn.Module`.** The model only has `forward`. The wrapper owns the optimizer, scheduler and training loop. |
| Loss | Lives in the library, **separate from the model**: `compute_losses` plus `UncertaintyWeightedLoss` (which holds `log_vars`). |

This makes the API closer to Keras or Lightning than to scikit-learn: the user handles the data, and the
library handles training.

```python
# user side: data
train_loader = DataLoader(CoreDataset(train_samples), batch_size=32, shuffle=True,  collate_fn=collate)
val_loader   = DataLoader(CoreDataset(val_samples),   batch_size=64, shuffle=False, collate_fn=collate)
test_loader  = DataLoader(CoreDataset(test_samples),  batch_size=64, shuffle=False, collate_fn=collate_infer)

# library side: training
model = NextSequenceModel(features, fusion_cls=MyFusion, d_model=64, epochs=20, lr=3e-4)
model.fit(train_loader, val_loader)
outputs = model.predict(test_loader)     # {"item": (N, C) logits, "price": (N, 1) values}
model.save("model.pt")
```

---

## Layers

Each layer only uses the layers below it.

| Layer | What | Where |
|---|---|---|
| 1. Building blocks | Embedding, fusion, encoder, heads, `MultiTaskModel` (`forward` only) | current modules |
| 1b. Loss | `UncertaintyWeightedLoss` (exists), `compute_losses` | `loss_fn.py` |
| 2. Factories | `build_model`, `build_optimizer`, `build_scheduler` | `train.py` |
| 3. Step functions | `to_device`, `train_one_epoch`, `evaluate` | `train.py` |
| 4. Wrapper | `NextSequenceModel` with `fit` / `predict` / `save` / `load` | `estimator.py` |

Layers 1 to 3 stay public. Advanced users can write their own loop or use Lightning with them.

---

## Why the optimization is outside the model

- Prediction only needs `forward`. Training state (optimizer, scheduler, `log_vars`) shouldn't be in an inference model.
- An optimizer stored on an `nn.Module` causes problems with `state_dict`, `deepcopy` (used for keeping the best weights)
  and `torch.compile`.
- A model with only `forward` can be reused in other loops, such as Lightning, Hugging Face or a user's own code.
- Each part can be tested on its own: `forward` shapes, loss values, and whether a training step reduces the loss.

The loss computation is library code because it depends on the features, but it is **not** inside the model.
The Hugging Face style `model(X, mask, labels=Y) -> loss` was rejected because it would make `log_vars` part of the model.

---

## Layer 4 sketch

```python
class NextSequenceModel:
    def __init__(self, features, fusion_cls, d_model=64, n_heads=4, num_layers=4,
                 expansion=4, dropout=0.1, max_len=128,
                 epochs=10, lr=3e-4, weight_decay=0.01, warmup_ratio=0.05,
                 grad_clip=1.0, device=None, verbose=True):
        ...  # store settings only, no work

    def fit(self, train_loader, val_loader=None):
        self.model_ = build_model(...).to(device)                      # forward only
        self.criterion_ = UncertaintyWeightedLoss(features).to(device)
        optimizer = build_optimizer(self.model_, self.criterion_, ...)  # local variable
        scheduler = build_scheduler(optimizer, epochs * len(train_loader), ...)
        for epoch in range(epochs):
            train_one_epoch(...)
            if val_loader:
                evaluate(...)          # keep the best weights by validation loss
        # restore the best weights, record self.history_
        return self

    @torch.no_grad()
    def predict(self, test_loader) -> dict[str, torch.Tensor]:
        # model_.eval(); run forward on every batch; concatenate in loader order
        # classification: (N, C) logits    regression: (N, 1) values

    def save(self, path): ...
    @classmethod
    def load(cls, path, fusion_cls=None, device=None): ...
```

Conventions:
- Anything learned during `fit` has a trailing underscore: `model_`, `criterion_`, `history_`.
  `predict` raises a clear error if called before `fit`.
- The optimizer and scheduler are **local to `fit`**. Store them as `optimizer_` / `scheduler_` only if
  resuming training gets added later.
- `predict` concatenates in loader order, so **the test loader must not shuffle**.

---

## Layers 2 and 3 notes

- `build_model(features, fusion_cls, cfg)` wires embedding, fusion, `PrependCLS`, RoPE, encoder and heads.
  - RoPE needs `max_len + 1` because CLS adds one position.
- `build_optimizer(model, criterion, cfg)`: AdamW with parameter groups.
  - weight decay on weight matrices
  - no decay on biases, norms, embeddings, the CLS token
  - **no decay on `criterion.log_vars`**: decay would pull every task weight toward 1.
  - **the criterion's parameters must be included**, or the learned task weights never update.
- `build_scheduler`: linear warmup then cosine decay, stepped **every batch**.
- `compute_losses(outputs, Y, features)`, choosing by `feature.task` (not by output shape):
  - classification: `F.cross_entropy(logits, y.long())`, shapes `(B, C)` vs `(B,)`
  - regression: `F.mse_loss(pred.squeeze(-1), y.float())`, shapes `(B,)` vs `(B,)`
- `train_one_epoch` / `evaluate`: plain functions with no hidden state, returning averaged total and per-feature losses.
  Gradient clipping is optional.
- `to_device` moves the nested batch `{"X": {...}, "Y": {...}, "mask": ..., "lengths": ...}` to the device.

---

## Data pipeline (user side, but document it)

Because data preparation is outside the library, the README/docs should show the right order:

```
raw data
  → 1. validate
  → 2. group + sort       (rows → one sequence per user/session, ordered by time)
  → 3. SPLIT              (train / val / test, by group or by time, never by row)
  → 4. fit on TRAIN only  (vocab via create_bow, scalers for numeric targets)
  → 5. transform all sets (tokens → ids, scale numbers, unknown → UNK)
  → 6. make samples       (prefix → next event, truncate to max_len)
  → 7. Dataset            (CoreDataset)
  → 8. DataLoader         (train: shuffle=True; val/test: shuffle=False)
```

- **Split before fitting the vocabulary and scalers.** Otherwise validation data leaks into training.
- **Never split by row or by sample.** One sequence `[a,b,c,d,e]` produces samples `[a]→b, [a,b]→c, ...`.
  Splitting those randomly puts near-identical samples in both sets and makes the validation score look better than it is.
  - **By group**: each user is entirely in one set. Measures how well it generalizes to new users.
  - **By time**: each user's last event(s) go to val/test, or use a global date cutoff. Measures prediction of future events,
    which is usually the real use case.
- **Encode in advance, not in the collate function.** Steps 5 and 6 run once. The collate function should only
  stack, pad and build `mask` / `lengths`.

Possible helpers later (optional, layer 0): `split_by_time`, `split_by_group`, `build_vocab`, `fit_scalers`.

---

## Checkpoint contents

Saved by `save`:
- `model_.state_dict()` and `criterion_.state_dict()`
- all hyperparameters
- `features` (store `TaskType` as its string value)
- the fusion strategy name (rebuilt from a registry, not pickled)
- a library version number

**The user is responsible for saving their own preprocessing**: vocabularies, scalers and the collate configuration.
Without them, raw outputs can't be decoded or unscaled. Possibly add a `metadata: dict` argument to `save`
so users can store these in the same file.

---

## Things to fix in the current code first

- `create_bow`
  - The returned `vocab_size` doesn't count `PAD`. Heads and embeddings need `len(bow)`.
  - `enumerate(set)` gives different ids each run. Use `sorted(...)`.
  - Add an `UNK` token.
- There is no concrete `BaseFusion` subclass yet. Pick a default (concatenate then project, or sum).
- `CustomCollate` with `inference=True`: check what the collate functions return for `y`, because
  `torch.tensor(Y[...])` is still called on it.
- `prep_data['Y']` is a `defaultdict`, so a mistyped feature name fails silently. Convert it to `dict` after building.
- Numeric targets (e.g. `price`) must be scaled, or their MSE swamps the cross-entropy losses.

---

## Phases

1. **Fix the foundations** listed above.
2. **Layers 2 and 3** in `train.py`. Train end to end from the notebook with them.
3. **Layer 4** `NextSequenceModel`: `fit(train_loader, val_loader)`, `predict(test_loader)`, `save` / `load`.
4. **Tests**: output shapes, a tiny dataset that overfits (loss goes down), save and load round-trip giving the same
   predictions, CPU vs GPU runs.
5. **Nice to have, later**: data helpers (`split_by_time`, ...), early stopping, resuming training, mixed precision.
   Skip callbacks, loggers and multi-GPU support until someone asks for them.

## Open questions

- What exactly is predicted: only the next event, or several steps ahead?
- Sequences longer than `max_len`: keep the most recent `max_len` events?
- Should `fit` also accept a DataLoader factory or raw samples later, or stay DataLoader-only?
- Store the user's preprocessing in the checkpoint (`metadata=`) or leave it fully to the user?
