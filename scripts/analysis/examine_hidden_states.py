#!/usr/bin/env python
"""
Script to examine the structure of hidden states in NPZ files
"""

import json
import os
import sys
from pathlib import Path

import numpy as np


def examine_npz(npz_file):
    """Examine the structure of an NPZ file containing hidden states"""
    print(f"Examining NPZ file: {npz_file}")

    # Load the NPZ file
    data = np.load(npz_file, allow_pickle=True)

    # Get the keys
    keys = list(data.keys())
    print(f"Number of keys: {len(keys)}")
    print(f"Example keys: {keys[:5]}")

    # Get the shape of the first item
    first_key = keys[0]
    first_item = data[first_key]
    print(f"Shape of first item ({first_key}): {first_item.shape}")

    # Extract question ID from the key
    if first_key.startswith("hidden_states_"):
        qid = first_key[len("hidden_states_") :]
        print(f"Question ID: {qid}")

    # Load metadata if available
    metadata_file = Path(npz_file).with_suffix(".json")
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        if qid in metadata:
            print(f"Question: {metadata[qid]['question']}")

    return first_item.shape


def main():
    if len(sys.argv) < 2:
        print("Usage: python examine_hidden_states.py <npz_file>")
        sys.exit(1)

    npz_file = sys.argv[1]
    if not os.path.exists(npz_file):
        print(f"Error: File {npz_file} does not exist")
        sys.exit(1)

    examine_npz(npz_file)


if __name__ == "__main__":
    main()
