import subprocess
import sys

from next_sequence.utils import create_bow


def test_ids_are_sorted_with_reserved_tokens():
    bow = create_bow([["b", "a"], ["a", "c"]])
    assert bow == {"PAD": 0, "UNK": 1, "a": 2, "b": 3, "c": 4}


def test_reserved_tokens_in_data_are_not_duplicated():
    bow = create_bow([["PAD", "x", "UNK"]])
    assert bow == {"PAD": 0, "UNK": 1, "x": 2}


def test_mixed_types_do_not_crash():
    bow = create_bow([[1, "a"], [2]])
    assert len(bow) == 5 and sorted(bow.values()) == list(range(5))


def test_ids_are_stable_across_processes():
    """Finding #12: set order used to change between Python sessions."""
    code = ("from next_sequence.utils import create_bow;"
            "print(create_bow([list('qwertyuiopasdfgh')]))")
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": seed, "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
                       check=True).stdout
        for seed in ("1", "2", "3")
    }
    assert len(runs) == 1
