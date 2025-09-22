#!/usr/bin/env python
# Combined experiment script for Likert evaluation and hidden states collection

import json
from pathlib import Path

from abstainer import (
    get_questions_by_filter,
    load_model,
    run_hidden_states_experiment,
    run_likert_experiment,
)


def run_combined_experiment(
    model_id: str,
    output_dir: str,
    subject: str = None,
    difficulty: str = None,
    split: str = None,
    form: str = "V0_letters",
    custom_labels: list = None,
    dataset_path: str = None,
    verbose: bool = True,
):
    """
    Run a combined experiment that performs both Likert evaluation and hidden states collection.

    Args:
        model_id: HuggingFace model ID to use
        output_dir: Directory to save results
        subject: Filter questions by subject
        difficulty: Filter questions by difficulty
        split: Filter questions by dataset split
        form: Likert prompt template to use
        custom_labels: Custom labels for Likert scale
        dataset_path: Path to the dataset file
        verbose: Whether to show progress information
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if verbose:
        print(f"Loading model: {model_id}")
    tokenizer, model = load_model(model_id)

    # Get questions based on filters
    questions = get_questions_by_filter(
        subject=subject, difficulty=difficulty, split=split, dataset_path=dataset_path
    )

    if verbose:
        print(f"Found {len(questions)} questions matching filters")

    # Define output files
    likert_output = output_dir / f"likert_results_{form}.json"
    hidden_states_output = output_dir / f"hidden_states_{form}.npz"

    # Run Likert experiment
    if verbose:
        print(f"Running Likert experiment with form '{form}'...")
        if custom_labels:
            print(f"Using custom labels: {custom_labels}")

    likert_stats = run_likert_experiment(
        model=model,
        tokenizer=tokenizer,
        output_file=str(likert_output),
        questions=questions,
        form=form,
        custom_labels=custom_labels,  # Pass the custom labels
        dataset_path=dataset_path,
        verbose=verbose,
        force_reprocess=True,  # Set to False if you want to resume from existing results
    )

    if verbose:
        print(
            f"Likert experiment completed: {likert_stats['completed_questions']} questions processed"
        )
        print(f"Average score: {likert_stats.get('average_score')}")
        print(f"Score distribution: {likert_stats.get('score_distribution')}")

    # Run hidden states experiment
    if verbose:
        print("\nCollecting hidden states...")

    hidden_stats = run_hidden_states_experiment(
        model=model,
        tokenizer=tokenizer,
        output_file=str(hidden_states_output),
        questions=questions,
        dataset_path=dataset_path,
        verbose=verbose,
        force_reprocess=True,  # Set to False if you want to resume from existing results
    )

    if verbose:
        print(
            f"Hidden states collection completed: {hidden_stats['completed_questions']} questions processed"
        )
        print(f"Hidden state shape: {hidden_stats['hidden_state_shape']}")

    # Save experiment configuration
    config = {
        "model_id": model_id,
        "subject": subject,
        "difficulty": difficulty,
        "split": split,
        "form": form,
        "custom_labels": custom_labels,
        "likert_results_file": str(likert_output),
        "hidden_states_file": str(hidden_states_output),
        "likert_stats": likert_stats,
        "hidden_stats": hidden_stats,
    }

    config_file = output_dir / "experiment_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"\nExperiment configuration saved to {config_file}")
        print(f"Likert results saved to {likert_output}")
        print(f"Hidden states saved to {hidden_states_output}")

    return {
        "likert_stats": likert_stats,
        "hidden_stats": hidden_stats,
        "config": config,
    }


def main():
    # Example usage
    # You can modify these parameters or add command-line argument parsing

    # Custom labels for Likert options
    # These are the actual tokens the model will generate as responses
    # They should be single tokens in the model's vocabulary for best results
    custom_labels = [
        "A",  # Definitely Yes
        "B",  # Probably Yes
        "C",  # Not Sure
        "D",  # Probably No
        "E",  # Definitely No
    ]

    # You can also use other tokens as long as they're in the model's vocabulary
    # For example:
    # custom_labels = ["YES", "LIKELY", "UNSURE", "UNLIKELY", "NO"]

    # Run the experiment
    _results = run_combined_experiment(
        model_id="google/gemma-3-4b-it",  # Replace with your model
        output_dir="./results/combined_experiment",
        subject="Physics",  # Optional: filter by subject
        split="test",  # Optional: use test split
        form="V0_letters",  # Likert form to use
        custom_labels=custom_labels,  # Optional: custom labels
        verbose=True,
    )

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
