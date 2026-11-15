# Cookbook: preparing data for next-sequence

next-sequence trains on **DataLoaders that you build**. This guide shows how the data must look
and the order of the preparation steps. Every step is implemented in
[examples/quickstart.py](../examples/quickstart.py), which you can run and copy from.

```
raw data
  → 1. one record per sequence
  → 2. SPLIT by time or by user   (never by row)
  → 3. fit vocabularies + scalers on TRAIN only
  → 4. transform every split       (unknown tokens → UNK, scale numbers)
  → 5. one conversion function per feature
  → 6. Features, config, CustomCollate, DataLoaders
  → 7. fit / predict / decode
  → 8. save with your preprocessing, load
```

---

## 1. One record per sequence

Each sample is a dict. Every sequence feature is a list ordered by time, and all lists of one sample
have the same length:

```python
{"user_id": 7,
 "item":    ["bread", "eggs", "apple"],
 "price":   [45.0, 61.2, 19.8],
 "weekday": [0, 0, 3]}
```

Extra keys (like `user_id`) are fine. Only keys named by your `Feature`s are used.

## 2. Split before anything is fitted

Split **whole sequences or points in time**, never individual rows or samples. One history produces
many near-identical prefixes (`[a]→b`, `[a,b]→c`, ...). If those land in both train and validation,
the validation score looks far better than it really is.

| Strategy | How | Answers |
|---|---|---|
| **Leave-last-out** (by time) | train predicts the 3rd-to-last event, val the 2nd-to-last, test the last | "how well do we predict a known user's next event?" (usually the real question) |
| By user | each user entirely in one split | "how well do we do for new users?" |
| Date cutoff | events before a date for training, after it for evaluation | same as by time, closer to production |

Leave-last-out is three slices per user ([`split_leave_last_out`](../examples/quickstart.py)):

```python
train = [cut(u, -2) for u in users]   # last element = target 3rd-to-last event
val   = [cut(u, -1) for u in users]
test  = [cut(u, None) for u in users]
```

## 3. Fit preprocessing on the training split only

- **Vocabulary:** `create_bow(u["item"] for u in train)`. Ids are deterministic: `PAD=0`, `UNK=1`,
  then the tokens sorted.
- **Scalers** for every numeric feature, e.g. mean/std of `log1p(price)` over the training values.

Fitting on all data leaks information from validation/test into training.

## 4. Transform every split with the same preprocessing

```python
"item":  [bow.get(token, bow["UNK"]) for token in u["item"]]      # unseen -> UNK
"price": [(math.log1p(p) - mean) / std for p in u["price"]]
```

Rules:
- **Id 0 is padding.** Shift categorical ids that start at 0 (e.g. weekday `0..6` → `1..7`) and use
  `vocab_size = max_id + 1`.
- **Classification targets start at 0** and go up to `n_classes - 1`.
- **Scale numeric inputs and regression targets.** Raw prices in the thousands give an MSE in the
  millions that drowns the other tasks.
- Do this once, before training, not inside the conversion functions on every batch.

## 5. One conversion function per feature

`CustomCollate` calls `funcs[name](values, inference)` with the list of that feature's values for
the batch. It must return `(inputs, targets)`:

- `inputs`: one 1-D tensor per sample (`long` for categorical, `float` for numeric)
- `targets`: one scalar per sample, or `None` when `inference=True`

Use **one helper for every feature**, so the training and inference branches can't drift apart
(different dtypes or lengths in the two branches are the most common bug):

```python
def make_func(dtype, x=lambda step: step, y=lambda step: step):
    def func(values, inference=False):
        xs = [torch.tensor([x(step) for step in v], dtype=dtype) for v in values]
        if inference:
            return xs, None                                   # predict after the full history
        return [t[:-1] for t in xs], [y(v[-1]) for v in values]   # last step is the target
    return func

funcs = {"item": make_func(torch.long), "price": make_func(torch.float),
         "weekday": make_func(torch.long)}
```

A numeric input with a **bucket target** stores `(scaled value, bucket)` per step in step 4, then
`make_func(torch.float, x=lambda s: s[0], y=lambda s: s[1])`.

## 6. Features, config, DataLoaders

```python
features = [
    Feature("item", TaskType.CLASSIFICATION, vocab_size=len(bow)),
    Feature("price", TaskType.REGRESSION, is_linear=True),
    Feature("qty", TaskType.CLASSIFICATION, is_linear=True, n_classes=3),
    Feature("weekday", TaskType.PASSTHROUGH, vocab_size=8),
]
config = NextSequenceConfig(features=features, max_len=16, emb_dim=32, attn_heads=2)

def loader(samples, inference=False, shuffle=False):
    collate = CustomCollate(features, "item", funcs, inference=inference, max_len=config.max_len)
    return DataLoader(CoreDataset(samples), batch_size=32, shuffle=shuffle, collate_fn=collate)
```

- `feature_mask` (`"item"` here) is the feature whose lengths define the padding mask.
- Pass **the same `max_len`** to `CustomCollate`: longer histories keep their most recent steps.
- Shuffle **only** the training loader.

## 7. Fit, predict, decode

```python
model = NextSequence(config, epochs=15, lr=3e-3, seed=0)
model.fit(loader(train, shuffle=True), loader(val))

out = model.predict(loader(test), mask_pad=True)      # evaluate: inputs = all but last event
future = model.predict(loader(test, inference=True))  # real use: predict after the full history
```

`predict` returns raw outputs. Decoding is yours:

```python
id2item = {i: t for t, i in bow.items()}
logits = out["item"].clone()
logits[:, bow["UNK"]] = float("-inf")                 # optional: never predict "unknown"
next_items = [id2item[i] for i in logits.argmax(-1).tolist()]

price = torch.expm1(out["price"].squeeze(-1) * std + mean)   # undo the scaling
```

`mask_pad=True` sets the PAD logit to `-inf` for features that predict a token from their own
vocabulary. Bucket targets (`is_linear=True` with `n_classes`) are left alone, because their class 0 is real.

## 8. Save with your preprocessing

The model can't decode its outputs without your vocabulary and scalers, so store them in the same file:

```python
model.save("model.pt", metadata={"bow": bow, "price_mean": mean, "price_std": std})

model = NextSequence.load("model.pt")       # device=None: CUDA if available
bow = model.metadata_["bow"]
```

Keep `metadata` to plain Python types (dict, list, str, int, float, bool, None) and tensors. Other
objects, like numpy arrays, need `NextSequence.load(path, trusted=True)`, which unpickles arbitrary
code. Only use it for files you created.

---

## Checklist

- [ ] Split by time or user **before** building the vocabulary and scalers
- [ ] Unseen tokens map to `UNK`, ids start at 1 (0 is PAD), class targets start at 0
- [ ] Numeric inputs and regression targets are scaled
- [ ] One shared helper builds every conversion function
- [ ] `CustomCollate(max_len=...)` equals `NextSequenceConfig(max_len=...)`
- [ ] Only the training loader shuffles
- [ ] `save(..., metadata=...)` includes everything needed to decode
