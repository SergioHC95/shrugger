#!/usr/bin/env python
"""
Script to examine the structure of hidden states in NPZ files
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# Get the project root directory
PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
sys.path.append(str(PROJECT_ROOT))


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
    else:
        # Try looking for metadata in standard locations
        metadata_paths = [
            Path(PROJECT_ROOT) / "outputs" / "data" / f"{Path(npz_file).stem}.json",
            Path(PROJECT_ROOT) / "data" / f"{Path(npz_file).stem}.json",
        ]
        for path in metadata_paths:
            if path.exists():
                with open(path) as f:
                    metadata = json.load(f)
                if qid in metadata:
                    print(f"Question: {metadata[qid]['question']}")
                    break

    return first_item.shape


def main():
    if len(sys.argv) < 2:
        print("Usage: python examine_hidden_states.py <npz_file>")
        sys.exit(1)

    # Handle both absolute paths and paths relative to project root
    npz_file = sys.argv[1]
    if not os.path.isabs(npz_file):
        # Try relative to current directory first
        if not os.path.exists(npz_file):
            # Then try relative to project root
            project_relative_path = os.path.join(PROJECT_ROOT, npz_file)
            if os.path.exists(project_relative_path):
                npz_file = project_relative_path

    if not os.path.exists(npz_file):
        print(f"Error: File {npz_file} does not exist")
        print(f"Tried: {npz_file}")
        print(
            f"Also tried relative to project root: {os.path.join(PROJECT_ROOT, sys.argv[1])}"
        )
        sys.exit(1)

    examine_npz(npz_file)


if __name__ == "__main__":
    main()
