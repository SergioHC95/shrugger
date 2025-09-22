#!/usr/bin/env python
"""
Analysis script for the comprehensive experiment results.
This script helps analyze and compare results across different prompt forms,
label types, and permutations.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Get the project root directory
PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
sys.path.append(str(PROJECT_ROOT))


def find_latest_run(base_dir=None):
    """Find the most recent experiment run directory."""
    if base_dir is None:
        base_path = PROJECT_ROOT / "results" / "comprehensive_experiments"
    else:
        base_path = Path(base_dir)
        
    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist.")
        return None

    run_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]
    if not run_dirs:
        print(f"Error: No run directories found in {base_path}.")
        return None

    # Sort by name (which includes timestamp)
    latest_run = sorted(run_dirs)[-1]
    return latest_run


def load_experiment_summaries(run_dir):
    """Load all experiment summaries from a run directory."""
    summaries = []

    for exp_dir in run_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith("."):
            summary_file = exp_dir / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                        summaries.append(summary)
                except Exception as e:
                    print(f"Error loading {summary_file}: {str(e)}")

    return summaries


def create_summary_dataframe(summaries):
    """Convert experiment summaries to a pandas DataFrame for analysis."""
    data = []

    for summary in summaries:
        row = {
            "experiment_name": summary.get("experiment_name", ""),
            "form": summary.get("form", ""),
            "label_type": summary.get("label_type", ""),
            "permutation": summary.get("permutation", ""),
            "labels": str(summary.get("labels", [])),
            "elapsed_time": summary.get("elapsed_time", 0),
            "questions_processed": summary.get("likert_stats", {}).get(
                "completed_questions", 0
            ),
            "average_score": summary.get("likert_stats", {}).get("average_score", 0),
            "valid_predictions": summary.get("likert_stats", {}).get(
                "valid_predictions", 0
            ),
            "invalid_predictions": summary.get("likert_stats", {}).get(
                "invalid_predictions", 0
            ),
        }

        # Add score distribution
        score_dist = summary.get("likert_stats", {}).get("score_distribution", {})
        for score, count in score_dist.items():
            row[f"score_{score}"] = count

        data.append(row)

    return pd.DataFrame(data)


def analyze_by_form(df):
    """Analyze results grouped by prompt form."""
    form_groups = df.groupby("form")

    # Calculate average metrics by form
    form_metrics = form_groups.agg(
        {
            "average_score": "mean",
            "valid_predictions": "mean",
            "invalid_predictions": "mean",
            "elapsed_time": "mean",
        }
    ).reset_index()

    print("\n=== Analysis by Prompt Form ===")
    print(form_metrics.to_string(index=False))

    # Plot average scores by form
    plt.figure(figsize=(10, 6))
    plt.bar(form_metrics["form"], form_metrics["average_score"])
    plt.title("Average Score by Prompt Form")
    plt.xlabel("Prompt Form")
    plt.ylabel("Average Score")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs" / "figures" / "avg_score_by_form.png"
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")

    return form_metrics


def analyze_by_label_type(df):
    """Analyze results grouped by label type."""
    label_groups = df.groupby("label_type")

    # Calculate average metrics by label type
    label_metrics = label_groups.agg(
        {
            "average_score": "mean",
            "valid_predictions": "mean",
            "invalid_predictions": "mean",
            "elapsed_time": "mean",
        }
    ).reset_index()

    print("\n=== Analysis by Label Type ===")
    print(label_metrics.to_string(index=False))

    # Plot average scores by label type
    plt.figure(figsize=(8, 6))
    plt.bar(label_metrics["label_type"], label_metrics["average_score"])
    plt.title("Average Score by Label Type")
    plt.xlabel("Label Type")
    plt.ylabel("Average Score")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs" / "figures" / "avg_score_by_label_type.png"
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")

    return label_metrics


def analyze_by_permutation(df):
    """Analyze results grouped by permutation."""
    perm_groups = df.groupby("permutation")

    # Calculate average metrics by permutation
    perm_metrics = perm_groups.agg(
        {
            "average_score": "mean",
            "valid_predictions": "mean",
            "invalid_predictions": "mean",
            "elapsed_time": "mean",
        }
    ).reset_index()

    print("\n=== Analysis by Permutation ===")
    print(perm_metrics.to_string(index=False))

    # Plot average scores by permutation
    plt.figure(figsize=(8, 6))
    plt.bar(perm_metrics["permutation"], perm_metrics["average_score"])
    plt.title("Average Score by Permutation")
    plt.xlabel("Permutation")
    plt.ylabel("Average Score")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs" / "figures" / "avg_score_by_permutation.png"
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")

    return perm_metrics


def analyze_form_label_interaction(df):
    """Analyze interaction between form and label type."""
    # Create a pivot table for form × label_type
    pivot = df.pivot_table(
        values="average_score", index="form", columns="label_type", aggfunc="mean"
    )

    print("\n=== Form × Label Type Interaction ===")
    print(pivot.to_string())

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot.values, cmap="viridis")
    plt.colorbar(label="Average Score")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Average Score by Form and Label Type")
    plt.xlabel("Label Type")
    plt.ylabel("Form")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(
                j,
                i,
                f"{pivot.values[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
            )

    plt.tight_layout()
    output_path = PROJECT_ROOT / "outputs" / "figures" / "form_label_interaction.png"
    plt.savefig(output_path)
    print(f"Saved plot: {output_path}")

    return pivot


def main():
    parser = argparse.ArgumentParser(
        description="Analyze comprehensive experiment results"
    )
    parser.add_argument(
        "--run_dir", type=str, help="Path to specific run directory (optional)"
    )
    args = parser.parse_args()

    # Find the run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Directory {args.run_dir} does not exist.")
            return
    else:
        run_dir = find_latest_run()
        if run_dir is None:
            return

    print(f"Analyzing results from: {run_dir}")

    # Load experiment summaries
    summaries = load_experiment_summaries(run_dir)
    print(f"Loaded {len(summaries)} experiment summaries")

    if not summaries:
        print("No experiment summaries found.")
        return

    # Create a DataFrame for analysis
    df = create_summary_dataframe(summaries)

    # Save the DataFrame to CSV
    csv_file = run_dir / "experiment_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved summary data to: {csv_file}")

    # Perform analyses
    form_metrics = analyze_by_form(df)
    label_metrics = analyze_by_label_type(df)
    perm_metrics = analyze_by_permutation(df)
    form_label_pivot = analyze_form_label_interaction(df)

    # Create a comprehensive analysis report
    report = {
        "run_directory": str(run_dir),
        "total_experiments": len(summaries),
        "form_analysis": form_metrics.to_dict(orient="records"),
        "label_type_analysis": label_metrics.to_dict(orient="records"),
        "permutation_analysis": perm_metrics.to_dict(orient="records"),
        "form_label_interaction": form_label_pivot.to_dict(),
    }

    report_file = run_dir / "analysis_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAnalysis complete. Report saved to: {report_file}")
    print(f"CSV data saved to: {csv_file}")
    print(f"Plots saved to {PROJECT_ROOT / 'outputs' / 'figures'}")


if __name__ == "__main__":
    main()
