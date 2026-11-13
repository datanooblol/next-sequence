from typing import Any

def create_bow(array:list)->dict[str, Any]:
    """
    Args:
        array (list): Sequences to build a bag of word
    Returns:
        tuple[dict[str, int], int, int]: A tuple of (bag of word, vocab size, max length)
    """
    max_len = 0
    _bow = set()
    for w in array:
        _w_len = len(w)
        if _w_len > max_len:
            max_len = _w_len
        _bow |= set(w)
    bow = {
        'PAD': 0
    }
    bow.update({w:idx+1 for idx, w in enumerate(_bow)})
    return bow