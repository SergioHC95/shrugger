#!/usr/bin/env python
"""
Script to verify the reorganized experiment data by layer
"""

import argparse
import json
from pathlib import Path

import numpy as np


def verify_layer_data(by_layer_dir):
    """Verify the reorganized data structure"""
    by_layer_path = Path(by_layer_dir)

    if not by_layer_path.exists():
        print(f"Error: Directory {by_layer_path} does not exist")
        return False

    # Find all layer directories
    layer_dirs = sorted([d for d in by_layer_path.glob("layer_*") if d.is_dir()])

    if not layer_dirs:
        print(f"Error: No layer directories found in {by_layer_path}")
        return False

    print(f"Found {len(layer_dirs)} layer directories")

    # Check each layer directory
    for layer_dir in layer_dirs:
        print(f"\nVerifying {layer_dir.name}:")

        # Find all experiment files
        exp_files = list(layer_dir.glob("*.npz"))

        if not exp_files:
            print(f"  Warning: No experiment files found in {layer_dir}")
            continue

        print(f"  Found {len(exp_files)} experiment files")

        # Check a sample of files
        for i, exp_file in enumerate(exp_files[:3]):  # Check up to 3 files per layer
            print(f"  Checking {exp_file.name}:")

            # Load the NPZ file
            try:
                data = np.load(exp_file, allow_pickle=True)

                # Check the keys
                keys = list(data.keys())
                print(f"    Keys: {keys}")

                # Check the question IDs and vectors
                if "question_ids" in keys and "vectors" in keys:
                    try:
                        question_ids = data["question_ids"]
                        vectors = data["vectors"]

                        print(f"    Number of questions: {len(question_ids)}")
                        print(f"    Vector shape: {vectors.shape}")
                    except Exception as e:
                        print(f"    Error accessing data: {str(e)}")

                    # Check if metadata file exists
                    metadata_file = exp_file.with_suffix(".json")
                    if metadata_file.exists():
                        try:
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                            print(
                                f"    Metadata: experiment_id={metadata.get('experiment_id')}, form={metadata.get('form')}, layer={metadata.get('layer')}"
                            )
                        except Exception as e:
                            print(f"    Error reading metadata: {str(e)}")
                    else:
                        print(f"    Warning: Metadata file {metadata_file} not found")
                else:
                    print(
                        "    Warning: Expected keys 'question_ids' and 'vectors' not found"
                    )
            except Exception as e:
                print(f"    Error loading file: {str(e)}")

            if i >= 2:  # Only show details for up to 3 files
                break

        # Show total size of all files in this layer
        try:
            total_size = sum(f.stat().st_size for f in layer_dir.glob("*.npz"))
            print(f"  Total size of NPZ files: {total_size / (1024*1024):.2f} MB")
        except Exception as e:
            print(f"  Error calculating total size: {str(e)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify reorganized experiment data by layer"
    )
    parser.add_argument(
        "--by-layer-dir",
        default="./results/by_layer",
        help="Directory containing reorganized data by layer",
    )
    args = parser.parse_args()

    try:
        if verify_layer_data(args.by_layer_dir):
            print("\nVerification successful!")
        else:
            print("\nVerification failed!")
    except Exception as e:
        print(f"\nVerification error: {str(e)}")


if __name__ == "__main__":
    main()
