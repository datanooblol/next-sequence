"""The cookbook example must keep working: it is the documentation users copy from."""
import importlib.util
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


@pytest.fixture(scope="module")
def quickstart():
    spec = importlib.util.spec_from_file_location("quickstart", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_is_leave_last_out(quickstart):
    users = quickstart.make_raw_data(n_users=5)
    train, val, test = quickstart.split_leave_last_out(users)
    for u, tr, va, te in zip(users, train, val, test):
        assert te["item"] == u["item"]
        assert va["item"] == u["item"][:-1]
        assert tr["item"] == u["item"][:-2]


def test_preprocessing_is_fit_on_train_only(quickstart):
    train = [{"item": ["a", "b"], "price": [10.0, 10.0], "qty": [1, 1]}]
    prep = quickstart.fit_preprocessing(train)
    assert set(prep["bow"]) == {"PAD", "UNK", "a", "b"}
    transformed = quickstart.transform(
        [{"item": ["a", "zzz"], "price": [10.0, 10.0], "qty": [1, 5], "weekday": [0, 6]}], prep)
    assert transformed[0]["item"] == [prep["bow"]["a"], prep["bow"]["UNK"]]   # unseen -> UNK
    assert transformed[0]["weekday"] == [1, 7]                                 # 0 stays PAD


def test_quickstart_runs_end_to_end(quickstart):
    results = quickstart.main(epochs=8, device="cpu", verbose=False)
    assert results["reload_matches"]
    assert results["item_accuracy"] > 0.5           # 5 items: chance is 0.2
    assert results["future_shape"][0] == 300
    assert set(results["task_weights"]) == {"item", "price", "qty"}
