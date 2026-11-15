"""End-to-end cookbook example for next-sequence.

Run:  uv run python examples/quickstart.py

Steps (the numbers match docs/cookbook.md):
  1. raw data: one record per user, each feature is a list ordered by time
  2. split by time (leave-last-out): train / val / test
  3. fit the vocabulary and scalers on the TRAIN split only
  4. transform every split with them (unknown items -> UNK)
  5. one conversion function per feature (shared helper -> same logic for training and inference)
  6. Features + config + DataLoaders
  7. fit, predict, decode
  8. save with your preprocessing as metadata, load it back
"""
import math
import random
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from next_sequence import (
    CoreDataset, CustomCollate, Feature, NextSequence, NextSequenceConfig, TaskType, create_bow,
)

ITEMS = ["apple", "bread", "cheese", "dates", "eggs"]
PRICE = {"apple": 20.0, "bread": 45.0, "cheese": 180.0, "dates": 95.0, "eggs": 60.0}
QTY_EDGES = (1, 3)          # buckets: qty <= 1 -> 0, qty <= 3 -> 1, otherwise -> 2


# ---------- 1. raw data ----------
def make_raw_data(n_users: int = 300, seed: int = 0) -> list[dict]:
    """Synthetic shoppers who walk through ITEMS in order, starting at a random item."""
    rng = random.Random(seed)
    users = []
    for uid in range(n_users):
        length = rng.randint(6, 12)
        start = rng.randrange(len(ITEMS))
        items = [ITEMS[(start + t) % len(ITEMS)] for t in range(length)]
        users.append({
            "user_id": uid,
            "item": items,
            "price": [PRICE[i] * rng.uniform(0.9, 1.1) for i in items],
            "qty": [len(i) % 4 + 1 for i in items],              # depends on the item
            "weekday": [rng.randrange(7) for _ in items],
        })
    return users


# ---------- 2. split by time ----------
SEQ_FEATURES = ("item", "price", "qty", "weekday")


def cut(user: dict, end: int | None) -> dict:
    return {k: (user[k][:end] if k in SEQ_FEATURES else user[k]) for k in user}


def split_leave_last_out(users: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Per user: train predicts the 3rd-to-last event, val the 2nd-to-last, test the last.

    Every split ends earlier in time than the next one, so nothing from the future leaks into
    training. The conversion functions below treat the LAST element of each list as the target.
    """
    train = [cut(u, -2) for u in users]
    val = [cut(u, -1) for u in users]
    test = [cut(u, None) for u in users]
    return train, val, test


# ---------- 3. fit preprocessing on train only ----------
def fit_preprocessing(train: list[dict]) -> dict:
    bow = create_bow(u["item"] for u in train)
    log_prices = [math.log1p(p) for u in train for p in u["price"]]
    mean = sum(log_prices) / len(log_prices)
    std = (sum((v - mean) ** 2 for v in log_prices) / len(log_prices)) ** 0.5 or 1.0
    qtys = [q for u in train for q in u["qty"]]
    q_mean = sum(qtys) / len(qtys)
    q_std = (sum((v - q_mean) ** 2 for v in qtys) / len(qtys)) ** 0.5 or 1.0
    # plain Python types only, so the checkpoint loads without trusted=True
    return {"bow": bow, "price_mean": mean, "price_std": std, "qty_mean": q_mean, "qty_std": q_std}


# ---------- 4. transform ----------
def qty_bucket(q: float) -> int:
    return sum(q > edge for edge in QTY_EDGES)       # classes start at 0


def transform(users: list[dict], prep: dict) -> list[dict]:
    bow, unk = prep["bow"], prep["bow"]["UNK"]
    out = []
    for u in users:
        out.append({
            "item": [bow.get(i, unk) for i in u["item"]],
            "price": [(math.log1p(p) - prep["price_mean"]) / prep["price_std"] for p in u["price"]],
            # numeric input + bucket target: keep both per step as (scaled value, bucket)
            "qty": [((q - prep["qty_mean"]) / prep["qty_std"], qty_bucket(q)) for q in u["qty"]],
            "weekday": [d + 1 for d in u["weekday"]],  # shift so 0 stays free for PAD
        })
    return out


# ---------- 5. conversion functions ----------
def make_func(dtype: torch.dtype, x=lambda step: step, y=lambda step: step):
    """One helper for every feature, so training and inference can't drift apart.

    Training: input = all but the last step, target = the last step.
    Inference: input = every step, no target.
    ``x`` / ``y`` pick the input value and the target value out of one step.
    """
    def func(values, inference=False):
        xs = [torch.tensor([x(step) for step in v], dtype=dtype) for v in values]
        if inference:
            return xs, None
        return [t[:-1] for t in xs], [y(v[-1]) for v in values]
    return func


# ---------- 6-8. model ----------
def main(epochs: int = 15, device: str | None = None, verbose: bool = True) -> dict:
    users = make_raw_data()
    train_raw, val_raw, test_raw = split_leave_last_out(users)
    prep = fit_preprocessing(train_raw)
    train, val, test = (transform(s, prep) for s in (train_raw, val_raw, test_raw))

    features = [
        Feature("item", TaskType.CLASSIFICATION, vocab_size=len(prep["bow"])),
        Feature("price", TaskType.REGRESSION, is_linear=True),
        Feature("qty", TaskType.CLASSIFICATION, is_linear=True, n_classes=len(QTY_EDGES) + 1),
        Feature("weekday", TaskType.PASSTHROUGH, vocab_size=8),
    ]
    funcs = {
        "item": make_func(torch.long),
        "price": make_func(torch.float),
        "qty": make_func(torch.float, x=lambda s: s[0], y=lambda s: s[1]),
        "weekday": make_func(torch.long),
    }
    max_len = 16
    config = NextSequenceConfig(features=features, max_len=max_len, emb_dim=32, attn_heads=2,
                                num_layers=2, dropout=0.1)

    def loader(samples, inference=False, shuffle=False):
        collate = CustomCollate(features, "item", funcs, inference=inference, max_len=max_len)
        return DataLoader(CoreDataset(samples), batch_size=32, shuffle=shuffle, collate_fn=collate)

    model = NextSequence(config, epochs=epochs, lr=3e-3, device=device, seed=0, verbose=verbose)
    model.fit(loader(train, shuffle=True), loader(val))

    # test: inputs are everything but the last event, targets are the last event
    test_loader = loader(test)
    out = model.predict(test_loader, mask_pad=True)
    logits = out["item"].clone()
    logits[:, prep["bow"]["UNK"]] = float("-inf")       # never recommend "unknown"
    id2item = {i: t for t, i in prep["bow"].items()}
    pred_items = [id2item[i] for i in logits.argmax(-1).tolist()]
    true_items = [u["item"][-1] for u in test_raw]
    accuracy = sum(p == t for p, t in zip(pred_items, true_items)) / len(true_items)

    # undo the price scaling: z -> log1p(price) -> price
    z = out["price"].squeeze(-1)
    pred_price = torch.expm1(z * prep["price_std"] + prep["price_mean"])
    true_price = torch.tensor([u["price"][-1] for u in test_raw])
    price_mae = (pred_price - true_price).abs().mean().item()

    qty_acc = (out["qty"].argmax(-1) == torch.tensor([qty_bucket(u["qty"][-1]) for u in test_raw])
               ).float().mean().item()

    # 8. save with preprocessing, load, and check the reloaded model agrees
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        model.save(path, metadata=prep)
        reloaded = NextSequence.load(path, device=device)
        same = torch.allclose(reloaded.predict(test_loader)["price"], out["price"], atol=1e-5)
        assert reloaded.metadata_["bow"] == prep["bow"]

    # predicting the real next event after the full history: inference=True
    future = model.predict(loader(test, inference=True), mask_pad=True)["item"]

    results = {
        "item_accuracy": accuracy,
        "price_mae": price_mae,
        "qty_accuracy": qty_acc,
        "reload_matches": same,
        "future_shape": tuple(future.shape),
        "task_weights": model.task_weights(),
        "device": str(model.device),
    }
    if verbose:
        for k, v in results.items():
            print(f"{k}: {v}")
    return results


if __name__ == "__main__":
    main()
