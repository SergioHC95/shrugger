#!/usr/bin/env python
# Example script for collecting hidden states from questions

from pathlib import Path

from abstainer import (
    get_questions_by_filter,
    load_hidden_states,
    load_model,
    run_hidden_states_experiment,
)


def main():
    # Configure paths
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "hidden_states.npz"

    # Load model (replace with your model ID)
    model_id = "google/gemma-3-4b-it"  # Example model
    print(f"Loading model: {model_id}")
    tokenizer, model = load_model(model_id)

    # Get a small set of questions for demonstration
    questions = get_questions_by_filter(
        subject="Physics",  # Optional: filter by subject
        split="test",  # Optional: use test split
        limit=5,  # Optional: limit to 5 questions
    )

    # Run the experiment to collect hidden states
    print("Collecting hidden states...")
    stats = run_hidden_states_experiment(
        model=model,
        tokenizer=tokenizer,
        output_file=str(output_file),
        questions=questions,
        verbose=True,
    )

    print(f"Experiment completed: {stats['completed_questions']} questions processed")
    print(f"Hidden state shape: {stats['hidden_state_shape']}")

    # Load and analyze the hidden states
    print("\nLoading hidden states from file...")
    data = load_hidden_states(str(output_file))

    print(f"Loaded {len(data['hidden_states'])} hidden state tensors")

    # Example: Print information about the first question
    if data["hidden_states"]:
        first_qid = next(iter(data["hidden_states"]))
        first_hidden_state = data["hidden_states"][first_qid]
        first_metadata = data["metadata"][first_qid]

        print(f"\nExample question (ID: {first_qid}):")
        print(f"Question: {first_metadata['question']}")
        print(f"Subject: {first_metadata['subject']}")
        print(f"Answer: {first_metadata['answer']}")
        print(f"Hidden state shape: {first_hidden_state.shape}")

        # Example analysis: Calculate mean activation across layers
        mean_activation = first_hidden_state.mean(axis=1)
        print(f"Mean activation across layers: {mean_activation.shape}")
        print(f"First 5 layer means: {mean_activation[:5]}")


if __name__ == "__main__":
    main()
