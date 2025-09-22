#!/usr/bin/env python
# Example script for running the combined experiment function

import json
from pathlib import Path

from abstainer import (
    get_questions_by_filter,
    load_combined_results,
    load_model,
    run_combined_experiment,
)


def main():
    """Run a combined experiment that collects both Likert results and hidden states efficiently."""
    # Configure paths
    output_dir = Path("./results/efficient_combined")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (replace with your model ID)
    model_id = "google/gemma-3-4b-it"  # Example model
    print(f"Loading model: {model_id}")
    tokenizer, model = load_model(model_id)

    # Get questions for the experiment
    # For this example, we'll use Biology questions from the dev split
    questions = get_questions_by_filter(
        subject="Biology",  # Filter by subject
        split="dev",  # Use dev split
    )

    print(f"Found {len(questions)} questions matching the filter criteria")

    # Define custom labels for the Likert scale
    custom_labels = [
        "A",  # Definitely Yes
        "B",  # Probably Yes
        "C",  # Not Sure
        "D",  # Probably No
        "E",  # Definitely No
    ]

    # Run the combined experiment
    print("Running combined experiment...")
    results = run_combined_experiment(
        model=model,
        tokenizer=tokenizer,
        output_dir=str(output_dir),
        questions=questions,
        form="V2_letters",  # Use the "Certainly yes/no" style descriptions
        custom_labels=custom_labels,
        verbose=True,
        force_reprocess=True,  # Set to False to use existing results if available
    )

    # Print summary statistics
    print("\nExperiment completed!")
    print(
        f"Likert evaluation: {results['likert_stats']['completed_questions']} questions processed"
    )
    print(f"Average score: {results['likert_stats'].get('average_score')}")
    print(f"Score distribution: {results['likert_stats'].get('score_distribution')}")
    print(
        f"Hidden states: {results['hidden_stats']['completed_questions']} questions processed"
    )
    print(f"Hidden state shape: {results['hidden_stats']['hidden_state_shape']}")

    # Save the experiment configuration
    config_file = output_dir / "experiment_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": model_id,
                "form": "V2_letters",
                "custom_labels": custom_labels,
                "likert_stats": results["likert_stats"],
                "hidden_stats": results["hidden_stats"],
                "output_files": results["output_files"],
            },
            f,
            indent=2,
        )

    print(f"\nExperiment configuration saved to {config_file}")
    print("Output files:")
    for name, path in results["output_files"].items():
        print(f"- {name}: {path}")

    # Optional: Load and analyze the results
    print("\nLoading results for analysis...")
    loaded_results = load_combined_results(str(output_dir), form="V2_letters")

    print(f"Loaded {len(loaded_results['likert_results'])} Likert results")
    print(f"Loaded {len(loaded_results['hidden_states'])} hidden state tensors")

    # Example: Print information about the first question
    if loaded_results["likert_results"]:
        first_qid = next(iter(loaded_results["likert_results"]))
        first_result = loaded_results["likert_results"][first_qid]
        first_hidden_state = loaded_results["hidden_states"].get(first_qid)

        print(f"\nExample question (ID: {first_qid}):")
        print(f"Question: {first_result['question']}")
        print(f"Answer: {first_result['answer']}")
        print(f"Prediction: {first_result['pred_label']}")
        print(f"Score: {first_result['score']}")
        if first_hidden_state is not None:
            print(f"Hidden state shape: {first_hidden_state.shape}")


if __name__ == "__main__":
    main()
