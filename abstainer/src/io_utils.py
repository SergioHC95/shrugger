# abstainer/src/io_utils.py
from __future__ import annotations

import json
import os
import random
from typing import Any, Iterable


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_ids(ids: list[str], train: float, dev: float, seed: int = 123):
    """Return (train_ids, dev_ids, test_ids) with fixed seed shuffle."""
    assert 0 < train < 1 and 0 < dev < 1 and train + dev < 1
    rng = random.Random(seed)
    arr = ids[:]
    rng.shuffle(arr)
    n = len(arr)
    n_tr = int(n * train)
    n_dev = int(n * dev)
    return arr[:n_tr], arr[n_tr : n_tr + n_dev], arr[n_tr + n_dev :]
