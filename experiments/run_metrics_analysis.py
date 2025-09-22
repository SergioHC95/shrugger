#!/usr/bin/env python
"""
Script to compute and analyze metrics across all experiment results.

This script:
1. Loads experiment results from the comprehensive_experiments directory
2. Computes CA and HEDGE metrics for each result
3. Outputs the metrics to CSV files for further analysis
4. Provides summary statistics across different experiment configurations
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from shrugger import compute_metrics_from_experiment_result


def find_experiment_runs(base_dir: Union[str, Path]) -> list[Path]:
    """
    Find all experiment run directories.

    Args:
        base_dir: Base directory containing experiment runs

    Returns:
        List of paths to experiment run directories
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    # Find all directories matching the pattern run_*
    run_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]

    if not run_dirs:
        raise ValueError(f"No experiment runs found in {base_dir}")

    return sorted(run_dirs)


def load_experiment_config(run_dir: Path) -> dict[str, Any]:
    """
    Load experiment configuration from a run directory.

    Args:
        run_dir: Path to the experiment run directory

    Returns:
        Dictionary containing experiment configuration
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def find_experiment_directories(run_dir: Path) -> list[Path]:
    """
    Find all experiment directories within a run directory.

    Args:
        run_dir: Path to the experiment run directory

    Returns:
        List of paths to experiment directories
    """
    # Find all directories that match the pattern V*_*_p*
    exp_dirs = [
        d
        for d in run_dir.iterdir()
        if d.is_dir() and (d.name.startswith("V") and "_p" in d.name)
    ]

    if not exp_dirs:
        raise ValueError(f"No experiment directories found in {run_dir}")

    return sorted(exp_dirs)


def load_likert_results(exp_dir: Path) -> dict[str, Any]:
    """
    Load Likert results from an experiment directory.

    Args:
        exp_dir: Path to the experiment directory

    Returns:
        Dictionary containing Likert results
    """
    # Find the likert results file (should be likert_results_V*.json)
    likert_files = list(exp_dir.glob("likert_results_*.json"))

    if not likert_files:
        raise FileNotFoundError(f"No Likert results found in {exp_dir}")

    # Use the first one if multiple exist
    likert_path = likert_files[0]

    with open(likert_path, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics_for_experiment(exp_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Compute metrics for all questions in an experiment.

    Args:
        exp_dir: Path to the experiment directory

    Returns:
        Dictionary mapping question IDs to metrics
    """
    # Load the Likert results
    likert_results = load_likert_results(exp_dir)

    # Define the canonical mapping
    canonical_mapping = {"YY": 0, "Y": 1, "A": 2, "N": 3, "NN": 4}

    # Compute metrics for each question
    metrics = {}
    for question_id, result in likert_results.items():
        try:
            question_metrics = compute_metrics_from_experiment_result(
                result, canonical_mapping
            )

            # Add the original result data we want to keep
            metrics[question_id] = {
                "id": question_id,
                "question": result["question"],
                "answer": result["answer"],
                "subject": result["subject"],
                "difficulty": result["difficulty"],
                "split": result["split"],
                "pred_label": result["pred_label"],
                "canonical_label": result["canonical_label"],
                "score": result["score"],
                "ca_score": question_metrics["ca_score"],
                "hedge_score": question_metrics["hedge_score"],
                # Include probabilities for reference
                "canonical_probs": result["canonical_probs"],
                "canonical_probs_norm": result["canonical_probs_norm"],
            }
        except Exception as e:
            print(f"Error processing question {question_id}: {str(e)}")

    return metrics


def save_metrics_to_csv(
    metrics: dict[str, dict[str, Any]], output_path: Path, exp_info: dict[str, str]
) -> None:
    """
    Save metrics to a CSV file.

    Args:
        metrics: Dictionary mapping question IDs to metrics
        output_path: Path to save the CSV file
        exp_info: Dictionary containing experiment information to include in each row
    """
    # Create the directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare the rows for the CSV
    rows = []
    for _question_id, question_metrics in metrics.items():
        # Create a row with experiment info and metrics
        row = {**exp_info, **question_metrics}

        # Convert nested dictionaries to strings
        for key, value in row.items():
            if isinstance(value, dict):
                row[key] = json.dumps(value)

        rows.append(row)

    # Write to CSV
    if rows:
        # Convert rows to DataFrame to ensure consistent data types
        df = pd.DataFrame(rows)

        # Ensure consistent data types for columns that might have mixed types
        if "pred_label" in df.columns:
            df["pred_label"] = df["pred_label"].astype(str)

        # Write to CSV
        df.to_csv(output_path, index=False)


def compute_summary_statistics(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Compute summary statistics for a set of metrics.

    Args:
        metrics: Dictionary mapping question IDs to metrics

    Returns:
        Dictionary containing summary statistics
    """
    if not metrics:
        return {}

    # Extract arrays of metric values
    ca_scores = [m["ca_score"] for m in metrics.values()]
    hedge_scores = [m["hedge_score"] for m in metrics.values()]
    likert_scores = [m["score"] for m in metrics.values()]

    # Compute statistics
    return {
        "count": len(metrics),
        "ca_score_mean": np.mean(ca_scores),
        "ca_score_median": np.median(ca_scores),
        "ca_score_std": np.std(ca_scores),
        "ca_score_min": np.min(ca_scores),
        "ca_score_max": np.max(ca_scores),
        "hedge_score_mean": np.mean(hedge_scores),
        "hedge_score_median": np.median(hedge_scores),
        "hedge_score_std": np.std(hedge_scores),
        "hedge_score_min": np.min(hedge_scores),
        "hedge_score_max": np.max(hedge_scores),
        "likert_score_mean": np.mean(likert_scores),
        "likert_score_std": np.std(likert_scores),
    }


def save_summary_to_csv(summaries: list[dict[str, Any]], output_path: Path) -> None:
    """
    Save summary statistics to a CSV file.

    Args:
        summaries: List of dictionaries containing summary statistics
        output_path: Path to save the CSV file
    """
    # Create the directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to CSV
    if summaries:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
            writer.writeheader()
            writer.writerows(summaries)


def process_experiment_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """
    Process all experiments in a run directory.

    Args:
        run_dir: Path to the experiment run directory
        output_dir: Path to save output files

    Returns:
        Dictionary containing run summary
    """
    print(f"Processing experiment run: {run_dir.name}")

    # Load experiment configuration
    config = load_experiment_config(run_dir)
    print(
        f"Loaded configuration: {len(config['forms'])} forms, {config['permutations']} permutations, {len(config['label_types'])} label types"
    )

    # Find all experiment directories
    exp_dirs = find_experiment_directories(run_dir)
    print(f"Found {len(exp_dirs)} experiment directories")

    # Create output directory for this run
    run_output_dir = output_dir / run_dir.name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Create a CSV file for all metrics
    all_metrics_path = run_output_dir / "all_metrics.csv"

    # Create a list to store summary statistics
    summaries = []

    # Process each experiment
    for exp_dir in tqdm(exp_dirs, desc="Processing experiments"):
        # Parse experiment name to get form, label_type, and permutation
        exp_name = exp_dir.name
        parts = exp_name.split("_")
        form = parts[0]
        label_type = parts[1]
        permutation = parts[2]

        # Create experiment info dictionary
        exp_info = {
            "run": run_dir.name,
            "experiment": exp_name,
            "form": form,
            "label_type": label_type,
            "permutation": permutation,
        }

        try:
            # Compute metrics for this experiment
            metrics = compute_metrics_for_experiment(exp_dir)

            # Save metrics to a CSV file
            exp_metrics_path = run_output_dir / f"{exp_name}_metrics.csv"
            save_metrics_to_csv(metrics, exp_metrics_path, exp_info)

            # Append to the all metrics file
            if not all_metrics_path.exists():
                save_metrics_to_csv(metrics, all_metrics_path, exp_info)
            else:
                # Read existing CSV and append new rows
                # Use low_memory=False to avoid DtypeWarning for mixed types
                existing_df = pd.read_csv(all_metrics_path, low_memory=False)

                # Convert metrics to DataFrame
                metrics_rows = []
                for _question_id, question_metrics in metrics.items():
                    row = {**exp_info, **question_metrics}
                    for key, value in row.items():
                        if isinstance(value, dict):
                            row[key] = json.dumps(value)
                    metrics_rows.append(row)

                new_df = pd.DataFrame(metrics_rows)

                # Combine and save
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                # Ensure consistent data types for columns that might have mixed types
                # Column 11 (pred_label) is likely the one with mixed types
                if "pred_label" in combined_df.columns:
                    combined_df["pred_label"] = combined_df["pred_label"].astype(str)

                combined_df.to_csv(all_metrics_path, index=False)

            # Compute summary statistics
            summary = compute_summary_statistics(metrics)
            summary.update(exp_info)
            summaries.append(summary)

        except Exception as e:
            print(f"Error processing experiment {exp_name}: {str(e)}")

    # Save summary statistics
    if summaries:
        summary_path = run_output_dir / "experiment_summaries.csv"
        save_summary_to_csv(summaries, summary_path)

        # Create summary DataFrames for analysis
        summary_df = pd.DataFrame(summaries)

        # Group by form and compute mean metrics
        form_summary = (
            summary_df.groupby("form")
            .agg(
                {
                    "ca_score_mean": "mean",
                    "hedge_score_mean": "mean",
                    "likert_score_mean": "mean",
                    "count": "sum",
                }
            )
            .reset_index()
        )
        form_summary.to_csv(run_output_dir / "form_summary.csv", index=False)

        # Group by label_type and compute mean metrics
        label_summary = (
            summary_df.groupby("label_type")
            .agg(
                {
                    "ca_score_mean": "mean",
                    "hedge_score_mean": "mean",
                    "likert_score_mean": "mean",
                    "count": "sum",
                }
            )
            .reset_index()
        )
        label_summary.to_csv(run_output_dir / "label_type_summary.csv", index=False)

        # Group by permutation and compute mean metrics
        perm_summary = (
            summary_df.groupby("permutation")
            .agg(
                {
                    "ca_score_mean": "mean",
                    "hedge_score_mean": "mean",
                    "likert_score_mean": "mean",
                    "count": "sum",
                }
            )
            .reset_index()
        )
        perm_summary.to_csv(run_output_dir / "permutation_summary.csv", index=False)

    return {
        "run_dir": str(run_dir),
        "experiments_processed": len(exp_dirs),
        "output_dir": str(run_output_dir),
        "summary_files": {
            "all_metrics": str(all_metrics_path),
            "experiment_summaries": str(run_output_dir / "experiment_summaries.csv"),
            "form_summary": str(run_output_dir / "form_summary.csv"),
            "label_type_summary": str(run_output_dir / "label_type_summary.csv"),
            "permutation_summary": str(run_output_dir / "permutation_summary.csv"),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute and analyze metrics across experiment results"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./results/comprehensive_experiments",
        help="Directory containing experiment runs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/metrics_analysis",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--run", type=str, help="Specific run to process (e.g., run_20250911_062422)"
    )
    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find experiment runs
    if args.run:
        run_dirs = [input_dir / args.run]
        if not run_dirs[0].exists():
            raise FileNotFoundError(f"Run directory not found: {run_dirs[0]}")
    else:
        run_dirs = find_experiment_runs(input_dir)

    print(f"Found {len(run_dirs)} experiment runs")

    # Process each run
    run_summaries = []
    for run_dir in run_dirs:
        try:
            summary = process_experiment_run(run_dir, output_dir)
            run_summaries.append(summary)
        except Exception as e:
            print(f"Error processing run {run_dir.name}: {str(e)}")

    # Save run summaries
    if run_summaries:
        with open(output_dir / "run_summaries.json", "w", encoding="utf-8") as f:
            json.dump(run_summaries, f, indent=2)

    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
