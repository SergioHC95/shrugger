#!/usr/bin/env python
"""
Comprehensive experiment script that runs combined experiments for:
- All prompt forms (V0-V5)
- All cyclic permutations of letter labels (A,B,C,D,E)
- All cyclic permutations of number labels (1,2,3,4,5)

This script works in both local environments and Google Colab.
"""

# Detect if we're in Google Colab
import importlib.util
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

IN_COLAB = False  # Set to False for local environment

if IN_COLAB:
    # Install required packages for Colab
    required_packages = ["torch", "tqdm", "transformers"]
    missing_packages = []

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)

    if missing_packages:
        print(f"Installing required packages: {missing_packages}")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"]
            + missing_packages
            + ["numpy"]
        )

    # Mount Google Drive
    from google.colab import drive

    print("Running in Google Colab. Mounting Google Drive...")
    drive.mount("/content/drive")

    # Change to project directory (modify this path as needed)
    PROJECT_PATH = "/content/drive/MyDrive/MATS-Project"
    os.chdir(PROJECT_PATH)
    print(f"Changed directory to {PROJECT_PATH}")

    # Add project root to path
    repo_root = Path.cwd()
    sys.path.insert(0, str(repo_root))
else:
    print("Running in local environment")

# Import dependencies
from tqdm.auto import tqdm  # Use tqdm.auto for Colab compatibility

# Suppress tqdm warnings about ipywidgets
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

# Add project root to Python path
if not IN_COLAB:
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.parent.absolute()
    
    # Add to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to Python path")

# Import abstainer modules with error handling
try:
    from abstainer import get_questions_by_filter, load_model, run_combined_experiment
    print("Successfully imported abstainer modules")
except ImportError as e:
    print(f"Error importing abstainer modules: {e}")
    print(
        "Make sure you're in the correct directory and the abstainer package is available"
    )
    sys.exit(1)


def get_cyclic_permutations(items):
    """Generate all cyclic permutations of a list."""
    n = len(items)
    return [items[i:] + items[:i] for i in range(n)]


def display_progress_summary(
    completed, total, form_counts, label_type_counts, perm_counts
):
    """Display a summary of progress so far."""
    print(f"\n{'-'*30} PROGRESS SUMMARY {'-'*30}")
    print(f"Completed: {completed}/{total} experiments ({completed/total*100:.1f}%)")

    print("\nBy Form:")
    for form, count in form_counts.items():
        print(f"  {form}: {count}/10 experiments")

    print("\nBy Label Type:")
    for label_type, count in label_type_counts.items():
        print(f"  {label_type}: {count}/30 experiments")

    print("\nBy Permutation:")
    for perm, count in perm_counts.items():
        print(f"  {perm}: {count}/12 experiments")

    print(f"{'-'*75}\n")


def main():
    # Configuration
    model_id = "google/gemma-3-4b-it"  # Replace with your model
    base_output_dir = Path("./results/comprehensive_experiments")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    config = {
        "model_id": model_id,
        "timestamp": timestamp,
        "environment": "colab" if IN_COLAB else "local",
        "forms": ["V0", "V1", "V2", "V3", "V4", "V5"],
        "label_types": ["alpha", "num"],
        "permutations": 5,
        "total_experiments": 6 * 5 * 2,  # 6 forms × 5 permutations × 2 label types
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Get all questions
    print("Loading all curated questions...")
    questions = get_questions_by_filter()  # No filters = all questions
    print(f"Found {len(questions)} questions")

    # Load model (only once)
    print(f"Loading model: {model_id}")
    tokenizer, model = load_model(model_id)

    # Define base labels
    alpha_labels = ["A", "B", "C", "D", "E"]
    num_labels = ["1", "2", "3", "4", "5"]

    # Generate all permutations
    alpha_perms = get_cyclic_permutations(alpha_labels)
    num_perms = get_cyclic_permutations(num_labels)

    # Create a log file for tracking progress
    log_file = run_dir / "experiment_log.txt"

    # Track completed experiments
    completed = 0
    total_experiments = 6 * 5 * 2  # 6 forms × 5 permutations × 2 label types

    # Counters for progress tracking
    form_counts = dict.fromkeys(config["forms"], 0)
    label_type_counts = {"alpha": 0, "num": 0}
    perm_counts = {f"p{i+1}": 0 for i in range(5)}

    # Create a progress bar for overall progress
    progress_bar = tqdm(total=total_experiments, desc="Overall Progress", position=0)

    # Print a summary of what we're about to do
    print(f"\nRunning {total_experiments} experiments:")
    print(f"- Environment: {config['environment']}")
    print(f"- 6 prompt forms: {config['forms']}")
    print("- 5 permutations for each label type")
    print("- 2 label types: alpha (A,B,C,D,E) and num (1,2,3,4,5)")
    print(f"\nResults will be saved to: {run_dir}\n")

    # Start with alpha labels and iterate through all permutations and forms
    for _label_type_idx, (label_type, label_perms) in enumerate(
        [("alpha", alpha_perms), ("num", num_perms)]
    ):
        for perm_idx, labels in enumerate(label_perms):
            perm_name = f"p{perm_idx+1}"

            for form in ["V0", "V1", "V2", "V3", "V4", "V5"]:
                # Create a descriptive experiment name
                experiment_name = f"{form}_{label_type}_{perm_name}"
                experiment_dir = run_dir / experiment_name

                # Log the start of this experiment
                with open(log_file, "a") as f:
                    f.write(f"\n{'-'*50}\n")
                    f.write(f"Starting experiment: {experiment_name}\n")
                    f.write(
                        f"Form: {form}, Label type: {label_type}, Permutation: {perm_name}\n"
                    )
                    f.write(f"Labels: {labels}\n")
                    f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # Update progress display
                experiment_desc = f"{form}_{label_type}_{perm_name}"
                progress_bar.set_description(f"Running: {experiment_desc}")

                # Print detailed info for this experiment
                print(f"\n{'-'*50}")
                print(
                    f"Experiment {completed+1}/{total_experiments}: {experiment_name}"
                )
                print(
                    f"Form: {form}, Label type: {label_type}, Permutation: {perm_name}"
                )
                print(f"Labels: {labels}")

                # Run the experiment
                try:
                    start_time = time.time()
                    results = run_combined_experiment(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=str(experiment_dir),
                        questions=questions,
                        form=form,
                        labels=labels,
                        verbose=True,
                        force_reprocess=True,
                    )

                    elapsed_time = time.time() - start_time

                    # Log the results
                    with open(log_file, "a") as f:
                        f.write(f"Completed in {elapsed_time:.2f} seconds\n")
                        f.write(
                            f"Likert stats: {results['likert_stats']['completed_questions']} questions processed\n"
                        )
                        if "average_score" in results["likert_stats"]:
                            f.write(
                                f"Average score: {results['likert_stats']['average_score']:.4f}\n"
                            )
                        f.write(
                            f"Score distribution: {results['likert_stats']['score_distribution']}\n"
                        )
                        f.write(
                            f"Hidden states: {results['hidden_stats']['completed_questions']} questions processed\n"
                        )
                        if results["hidden_stats"]["hidden_state_shape"]:
                            f.write(
                                f"Hidden state shape: {results['hidden_stats']['hidden_state_shape']}\n"
                            )

                    # Create a summary file for this experiment
                    summary = {
                        "experiment_name": experiment_name,
                        "form": form,
                        "label_type": label_type,
                        "permutation": perm_name,
                        "labels": labels,
                        "elapsed_time": elapsed_time,
                        "environment": config["environment"],
                        "likert_stats": results["likert_stats"],
                        "hidden_stats": results["hidden_stats"],
                        "output_files": results["output_files"],
                    }

                    with open(experiment_dir / "summary.json", "w") as f:
                        json.dump(summary, f, indent=2)

                    # Update progress tracking
                    completed += 1
                    progress_bar.update(1)

                    # Update counters
                    form_counts[form] += 1
                    label_type_counts[label_type] += 1
                    perm_counts[perm_name] += 1

                    # Display completion info
                    print(f"Completed experiment {completed}/{total_experiments}")
                    print(f"Elapsed time: {elapsed_time:.2f} seconds")

                    # Show some stats from this experiment
                    if "average_score" in results["likert_stats"]:
                        print(
                            f"Average score: {results['likert_stats']['average_score']:.4f}"
                        )
                    print(
                        f"Valid predictions: {results['likert_stats']['valid_predictions']}"
                    )
                    print(
                        f"Invalid predictions: {results['likert_stats']['invalid_predictions']}"
                    )

                    # Show progress summary every 5 experiments
                    if completed % 5 == 0:
                        display_progress_summary(
                            completed,
                            total_experiments,
                            form_counts,
                            label_type_counts,
                            perm_counts,
                        )

                except Exception as e:
                    # Log any errors
                    with open(log_file, "a") as f:
                        f.write(f"ERROR: {str(e)}\n")
                    print(f"Error in experiment {experiment_name}: {str(e)}")

    # Close the progress bar
    progress_bar.close()

    # Create a final summary
    final_summary = {
        "timestamp": timestamp,
        "model_id": model_id,
        "environment": config["environment"],
        "total_experiments": total_experiments,
        "completed_experiments": completed,
        "questions_count": len(questions),
    }

    with open(run_dir / "final_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"\n{'-'*50}")
    print(f"All experiments completed: {completed}/{total_experiments}")
    print(f"Results saved to: {run_dir}")
    print(f"Log file: {log_file}")

    # Display a completion message that's easy to spot
    print(f"\n{'='*50}")
    print(f"{'='*15} EXPERIMENTS COMPLETE {'='*15}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
