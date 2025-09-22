#!/usr/bin/env python
"""
Script to reorganize experiment data by layer instead of by experiment.

The reorganization follows these rules:
1. One folder per layer (e.g., layer_00, layer_01, ...)
2. Inside each layer folder, one file per experiment
3. Each file contains question IDs and corresponding residual stream vectors for that layer and experiment
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_experiment_dirs(results_dir):
    """Find all experiment directories in the comprehensive_experiments directory"""
    results_path = Path(results_dir)
    experiment_dirs = []

    # Only check the comprehensive experiments directory
    comp_exp_path = results_path / "comprehensive_experiments"
    if comp_exp_path.exists() and comp_exp_path.is_dir():
        # Look for run_* directories
        for run_dir in comp_exp_path.glob("run_*"):
            if run_dir.is_dir():
                # Add all experiment directories within this run
                for exp_dir in run_dir.glob("*"):
                    if exp_dir.is_dir() and any(exp_dir.glob("hidden_states_*.npz")):
                        experiment_dirs.append(exp_dir)

    return experiment_dirs


def process_experiment(exp_dir, output_dir, force=False):
    """Process a single experiment directory"""
    print(f"Processing experiment: {exp_dir}")

    # Find hidden states files
    hidden_states_files = list(exp_dir.glob("hidden_states_*.npz"))
    if not hidden_states_files:
        print(f"No hidden states files found in {exp_dir}")
        return

    # Get experiment identifier from directory name
    experiment_id = exp_dir.name

    # Process each hidden states file
    for hs_file in hidden_states_files:
        # Get form identifier from filename
        form = hs_file.stem.replace("hidden_states_", "")

        # Load the hidden states
        data = np.load(hs_file, allow_pickle=True)

        # Load metadata
        metadata_file = hs_file.with_suffix(".json")
        if not metadata_file.exists():
            print(f"Warning: Metadata file {metadata_file} not found")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Get the keys and extract a sample to determine the number of layers
        keys = list(data.keys())
        if not keys:
            print(f"Warning: No data found in {hs_file}")
            continue

        # Get the first item to determine shape
        first_key = keys[0]
        first_item = data[first_key]
        num_layers = first_item.shape[0]
        hidden_dim = first_item.shape[1]

        print(f"File: {hs_file.name}, Layers: {num_layers}, Hidden dim: {hidden_dim}")

        # Create output directory structure
        output_base = Path(output_dir)
        output_base.mkdir(exist_ok=True)

        # Create by_layer directory
        by_layer_dir = output_base / "by_layer"
        by_layer_dir.mkdir(exist_ok=True)

        # Process each layer
        for layer_idx in range(num_layers):
            # Create layer directory
            layer_dir = by_layer_dir / f"layer_{layer_idx:02d}"
            layer_dir.mkdir(exist_ok=True)

            # Define output file for this experiment and layer
            exp_identifier = f"{experiment_id}_{form}"
            output_file = layer_dir / f"{exp_identifier}.npz"

            # Skip if file exists and not forcing overwrite
            if output_file.exists() and not force:
                print(f"Skipping {output_file} (already exists)")
                continue

            # Extract layer data for all questions
            # layer_data = {}  # Currently unused
            question_ids = []
            vectors = []

            for key in tqdm(keys, desc=f"Layer {layer_idx:02d}"):
                if key.startswith("hidden_states_"):
                    qid = key[len("hidden_states_") :]
                    # Extract the vector for this layer
                    vector = data[key][layer_idx]

                    # Store the question ID and vector
                    question_ids.append(qid)
                    vectors.append(vector)

            # Convert to numpy arrays
            question_ids = np.array(question_ids, dtype=str)
            vectors = np.array(vectors)

            # Save as NPZ file
            np.savez_compressed(output_file, question_ids=question_ids, vectors=vectors)

            # Create a metadata file with experiment info
            metadata_output = output_file.with_suffix(".json")
            with open(metadata_output, "w") as f:
                json.dump(
                    {
                        "experiment_id": experiment_id,
                        "form": form,
                        "layer": layer_idx,
                        "num_questions": len(question_ids),
                        "vector_dim": hidden_dim,
                        "question_metadata": metadata,
                    },
                    f,
                    indent=2,
                )

            print(f"Created {output_file} with {len(question_ids)} questions")


def main():
    parser = argparse.ArgumentParser(description="Reorganize experiment data by layer")
    parser.add_argument(
        "--results-dir",
        default="./results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir", default="./results", help="Directory to store reorganized data"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite of existing files"
    )
    args = parser.parse_args()

    # Find all experiment directories
    experiment_dirs = get_experiment_dirs(args.results_dir)
    print(f"Found {len(experiment_dirs)} experiment directories")

    # Process each experiment
    for exp_dir in experiment_dirs:
        process_experiment(exp_dir, args.output_dir, args.force)

    print("Reorganization complete")


if __name__ == "__main__":
    main()
