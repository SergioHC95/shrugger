#!/usr/bin/env python
"""
Script to detect and clean up corrupted NPZ files in the reorganized data structure
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def check_file_integrity(file_path):
    """Check if an NPZ file is corrupted"""
    try:
        data = np.load(file_path, allow_pickle=True)
        # Try to access the keys and data to verify integrity
        keys = list(data.keys())
        for key in keys:
            _ = data[key]
        return True, None
    except Exception as e:
        return False, str(e)


def cleanup_corrupted_files(by_layer_dir, delete=False):
    """Find and optionally delete corrupted NPZ files"""
    by_layer_path = Path(by_layer_dir)

    if not by_layer_path.exists():
        print(f"Error: Directory {by_layer_path} does not exist")
        return False

    # Find all layer directories
    layer_dirs = sorted([d for d in by_layer_path.glob("layer_*") if d.is_dir()])

    if not layer_dirs:
        print(f"Error: No layer directories found in {by_layer_path}")
        return False

    print(f"Scanning {len(layer_dirs)} layer directories for corrupted files...")

    corrupted_files = []

    # Check each layer directory
    for layer_dir in layer_dirs:
        print(f"\nScanning {layer_dir.name}:")

        # Find all NPZ files
        npz_files = list(layer_dir.glob("*.npz"))

        if not npz_files:
            print(f"  No NPZ files found in {layer_dir}")
            continue

        print(f"  Checking {len(npz_files)} files...")

        # Check each file
        for npz_file in tqdm(npz_files, desc=f"  {layer_dir.name}"):
            is_valid, error = check_file_integrity(npz_file)

            if not is_valid:
                print(f"  Corrupted file: {npz_file}")
                print(f"  Error: {error}")
                corrupted_files.append(npz_file)

                # Delete if requested
                if delete:
                    try:
                        npz_file.unlink()
                        print(f"  Deleted: {npz_file}")

                        # Also delete the corresponding JSON file if it exists
                        json_file = npz_file.with_suffix(".json")
                        if json_file.exists():
                            json_file.unlink()
                            print(f"  Deleted: {json_file}")
                    except Exception as e:
                        print(f"  Error deleting {npz_file}: {str(e)}")

    # Print summary
    print(f"\nFound {len(corrupted_files)} corrupted files")
    if corrupted_files:
        print("Corrupted files:")
        for f in corrupted_files:
            print(f"  {f}")

    if delete and corrupted_files:
        print(f"\nDeleted {len(corrupted_files)} corrupted files")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Detect and clean up corrupted NPZ files"
    )
    parser.add_argument(
        "--by-layer-dir",
        default="./results/by_layer",
        help="Directory containing reorganized data by layer",
    )
    parser.add_argument("--delete", action="store_true", help="Delete corrupted files")
    args = parser.parse_args()

    try:
        if cleanup_corrupted_files(args.by_layer_dir, args.delete):
            print("\nCleanup scan completed successfully!")
        else:
            print("\nCleanup scan failed!")
    except Exception as e:
        print(f"\nCleanup error: {str(e)}")


if __name__ == "__main__":
    main()
