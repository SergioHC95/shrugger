import json
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from .experiment_utils import (
    evaluate_question_with_likert,
    get_questions_by_filter,
)
from .probes import get_last_token_hidden_states


def run_likert_experiment(
    model,
    tokenizer,
    output_file: str,
    questions: Optional[list[dict[str, str]]] = None,
    subject: Optional[str] = None,
    difficulty: Optional[Union[str, int]] = None,
    split: Optional[str] = None,
    form: str = "V0",
    labels: Optional[list[str]] = None,
    dataset_path: Optional[str] = None,
    verbose: bool = True,
    force_reprocess: bool = False,
) -> dict[str, Any]:
    """
    Run Likert scale evaluation on a set of questions, with checkpointing to a JSON file.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        output_file: Path to the JSON file to save results
        questions: Optional list of question dictionaries (if provided, other filters are ignored)
        subject: Filter questions by subject (e.g., "Biology")
        difficulty: Filter questions by difficulty level
        split: Filter questions by dataset split (e.g., "train", "dev", "test")
        form: Likert prompt template to use
        labels: Optional list of 5 labels to use in order
               [definitely_yes, probably_yes, not_sure, probably_no, definitely_no]
        dataset_path: Path to the dataset file (optional)
        verbose: Whether to show progress bar and logs
        force_reprocess: If True, reprocess questions even if they exist in the output file

    Returns:
        Dict containing experiment summary:
            - total_questions: Number of questions processed
            - completed_questions: Number of questions successfully evaluated
            - skipped_questions: Number of questions skipped (already in output file)
            - average_score: Average score across all evaluated questions
            - score_distribution: Count of each score value
            - valid_predictions: Number of predictions that were valid Likert options
            - invalid_predictions: Number of predictions that were not valid Likert options
    """
    # Get questions if not provided
    if questions is None:
        questions = get_questions_by_filter(
            subject=subject,
            difficulty=difficulty,
            split=split,
            dataset_path=dataset_path,
        )

    if verbose:
        print(f"Processing {len(questions)} questions with Likert form '{form}'")

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if available
    existing_results = {}
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing_results = json.load(f)
                if verbose:
                    print(
                        f"Loaded {len(existing_results)} existing results from {output_file}"
                    )
        except json.JSONDecodeError:
            if verbose:
                print(
                    f"Warning: Could not parse existing file {output_file}, starting fresh"
                )

    # Initialize statistics
    stats = {
        "total_questions": len(questions),
        "completed_questions": 0,
        "skipped_questions": 0,
        "scores": [],
        "score_distribution": {str(i): 0 for i in range(-3, 3)},
        "valid_predictions": 0,
        "invalid_predictions": 0,
    }

    # Process each question
    for question in tqdm(questions, disable=not verbose):
        question_id = question["id"]

        # Skip if already processed (unless force_reprocess is True)
        if question_id in existing_results and not force_reprocess:
            stats["skipped_questions"] += 1
            continue

        # Evaluate the question
        try:
            start_time = time.time()
            result = evaluate_question_with_likert(
                model=model,
                tokenizer=tokenizer,
                question_data=question,
                form=form,
                labels=labels,
                dataset_path=dataset_path,
            )
            elapsed_time = time.time() - start_time

            # Add timing information
            result["processing_time"] = elapsed_time

            # Update statistics
            stats["completed_questions"] += 1
            stats["scores"].append(result["score"])
            stats["score_distribution"][str(result["score"])] = (
                stats["score_distribution"].get(str(result["score"]), 0) + 1
            )

            if result["is_valid"]:
                stats["valid_predictions"] += 1
            else:
                stats["invalid_predictions"] += 1

            # Add to results (but don't save yet)
            existing_results[question_id] = result
        except Exception as e:
            if verbose:
                print(f"Error processing question {question_id}: {str(e)}")

    # Calculate final statistics
    if stats["completed_questions"] > 0:
        stats["average_score"] = sum(stats["scores"]) / len(stats["scores"])
    else:
        stats["average_score"] = None

    # Write all results to file at the end
    if stats["completed_questions"] > 0 or stats["skipped_questions"] > 0:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=2)
        if verbose:
            print(f"Saved {len(existing_results)} results to {output_file}")

    # Return summary statistics
    return stats


def run_hidden_states_experiment(
    model,
    tokenizer,
    output_file: str,
    questions: Optional[list[dict[str, str]]] = None,
    subject: Optional[str] = None,
    difficulty: Optional[Union[str, int]] = None,
    split: Optional[str] = None,
    dataset_path: Optional[str] = None,
    verbose: bool = True,
    force_reprocess: bool = False,
) -> dict[str, Any]:
    """
    Run an experiment to collect hidden states for the last token of each question
    and save them to a file.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        output_file: Path to the file to save hidden states (will use .npz format)
        questions: Optional list of question dictionaries (if provided, other filters are ignored)
        subject: Filter questions by subject (e.g., "Biology")
        difficulty: Filter questions by difficulty level
        split: Filter questions by dataset split (e.g., "train", "dev", "test")
        dataset_path: Path to the dataset file (optional)
        verbose: Whether to show progress bar and logs
        force_reprocess: If True, reprocess questions even if the output file exists

    Returns:
        Dict containing experiment summary:
            - total_questions: Number of questions processed
            - completed_questions: Number of questions successfully processed
            - hidden_state_shape: Shape of each hidden state tensor
    """
    # Get questions if not provided
    if questions is None:
        questions = get_questions_by_filter(
            subject=subject,
            difficulty=difficulty,
            split=split,
            dataset_path=dataset_path,
        )

    if verbose:
        print(f"Processing hidden states for {len(questions)} questions")

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if output file already exists
    if output_path.exists() and not force_reprocess:
        if verbose:
            print(
                f"Output file {output_file} already exists. Use force_reprocess=True to overwrite."
            )
        return {
            "total_questions": len(questions),
            "completed_questions": 0,
            "hidden_state_shape": None,
        }

    # Initialize storage for hidden states
    hidden_states_dict = {}

    # Process each question
    for question in tqdm(questions, disable=not verbose):
        question_id = question["id"]
        question_text = question["question"]

        try:
            # Extract hidden states for the last token
            hidden_states = get_last_token_hidden_states(
                model=model, tokenizer=tokenizer, text=question_text
            )

            # Convert to numpy for storage
            # Ensure the tensor is in float32 format for compatibility
            hidden_states_float = hidden_states.cpu().to(torch.float32)
            hidden_states_np = hidden_states_float.numpy()

            # Store with question ID as key
            hidden_states_dict[question_id] = {
                "hidden_states": hidden_states_np,
                "question": question_text,
                "metadata": {
                    "subject": question["subject"],
                    "difficulty": question["difficulty"],
                    "answer": question["answer"],
                    "split": question["split"],
                },
            }

        except Exception as e:
            if verbose:
                print(f"Error processing question {question_id}: {str(e)}")

    # Save all hidden states to file
    if hidden_states_dict:
        # Convert the dictionary to a format suitable for npz storage
        np_dict = {}
        metadata_dict = {}

        for qid, data in hidden_states_dict.items():
            # Store hidden states array with question ID as key
            np_dict[f"hidden_states_{qid}"] = data["hidden_states"]

            # Store metadata as JSON string
            metadata = {"question": data["question"], **data["metadata"]}
            metadata_dict[qid] = metadata

        # Save hidden states as npz
        np.savez_compressed(output_path, **np_dict)

        # Save metadata separately as JSON
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

        if verbose:
            print(
                f"Saved hidden states for {len(hidden_states_dict)} questions to {output_file}"
            )
            print(f"Saved metadata to {metadata_path}")

    # Get shape of hidden states for reference
    hidden_state_shape = None
    if hidden_states_dict:
        first_key = next(iter(hidden_states_dict))
        hidden_state_shape = hidden_states_dict[first_key]["hidden_states"].shape

    # Return summary statistics
    return {
        "total_questions": len(questions),
        "completed_questions": len(hidden_states_dict),
        "hidden_state_shape": hidden_state_shape,
    }


def run_combined_experiment(
    model,
    tokenizer,
    output_dir: str,
    questions: Optional[list[dict[str, str]]] = None,
    subject: Optional[str] = None,
    difficulty: Optional[Union[str, int]] = None,
    split: Optional[str] = None,
    form: str = "V0",
    labels: Optional[list[str]] = None,
    dataset_path: Optional[str] = None,
    verbose: bool = True,
    force_reprocess: bool = False,
) -> dict[str, Any]:
    """
    Run a combined experiment that performs Likert evaluation and collects hidden states
    with a single forward pass for each question.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        output_dir: Directory to save results
        questions: Optional list of question dictionaries (if provided, other filters are ignored)
        subject: Filter questions by subject (e.g., "Biology")
        difficulty: Filter questions by difficulty level
        split: Filter questions by dataset split (e.g., "train", "dev", "test")
        form: Likert prompt template to use
        labels: Optional list of 5 labels to use in order
               [definitely_yes, probably_yes, not_sure, probably_no, definitely_no]
        dataset_path: Path to the dataset file (optional)
        verbose: Whether to show progress bar and logs
        force_reprocess: If True, reprocess questions even if they exist in the output file

    Returns:
        Dict containing experiment summary:
            - likert_stats: Statistics from the Likert evaluation
            - hidden_stats: Statistics from the hidden states collection
            - output_files: Paths to the output files
    """
    # Get questions if not provided
    if questions is None:
        questions = get_questions_by_filter(
            subject=subject,
            difficulty=difficulty,
            split=split,
            dataset_path=dataset_path,
        )

    if verbose:
        print(f"Processing {len(questions)} questions with Likert form '{form}'")

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output files
    likert_output = output_dir / f"likert_results_{form}.json"
    hidden_states_output = output_dir / f"hidden_states_{form}.npz"
    metadata_output = output_dir / f"hidden_states_{form}.json"

    # Check if output files already exist
    if not force_reprocess and likert_output.exists() and hidden_states_output.exists():
        if verbose:
            print("Output files already exist. Use force_reprocess=True to overwrite.")
        return {
            "likert_stats": {"completed_questions": 0},
            "hidden_stats": {"completed_questions": 0},
            "output_files": {
                "likert": str(likert_output),
                "hidden_states": str(hidden_states_output),
                "metadata": str(metadata_output),
            },
        }

    # Initialize storage for results
    likert_results = {}
    hidden_states_dict = {}
    metadata_dict = {}

    # Initialize statistics
    likert_stats = {
        "total_questions": len(questions),
        "completed_questions": 0,
        "skipped_questions": 0,
        "scores": [],
        "score_distribution": {str(i): 0 for i in range(-3, 3)},
        "valid_predictions": 0,
        "invalid_predictions": 0,
    }

    hidden_stats = {
        "total_questions": len(questions),
        "completed_questions": 0,
        "hidden_state_shape": None,
    }

    # Process each question
    for question in tqdm(questions, disable=not verbose):
        question_id = question["id"]
        question_text = question["question"]

        try:
            start_time = time.time()

            # Prepare the prompt
            from .prompts import build_likert_prompt

            prompt = build_likert_prompt(question_text, form=form, labels=labels)

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Run forward pass with output_hidden_states=True
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Get hidden states and extract the last token
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise ValueError(
                    "Model did not return hidden states. Make sure the model supports output_hidden_states=True"
                )

            # Process hidden states
            # Extract the last token from each layer and convert to float32 for compatibility
            last_token_states = torch.stack(
                [state[0, -1, :].to(torch.float32) for state in hidden_states]
            )

            # Convert to numpy for storage
            hidden_states_float = last_token_states.cpu().to(torch.float32)
            hidden_states_np = hidden_states_float.numpy()

            # Store hidden states
            hidden_states_dict[question_id] = hidden_states_np
            metadata_dict[question_id] = {
                "question": question_text,
                "subject": question["subject"],
                "difficulty": question["difficulty"],
                "answer": question["answer"],
                "split": question["split"],
            }

            # Update hidden states statistics
            hidden_stats["completed_questions"] += 1
            if (
                hidden_stats["hidden_state_shape"] is None
                and hidden_states_np is not None
            ):
                hidden_stats["hidden_state_shape"] = hidden_states_np.shape

            # Process Likert results
            # Get the output token
            logits = outputs.logits
            last_token_logits = logits[0, -1, :]

            # Run Likert probe evaluation
            from .utils import (
                get_default_likert_labels,
                get_likert_mapping,
            )

            # Get the labels to use
            used_labels = labels if labels else get_default_likert_labels()

            # Map labels to canonical format
            label_mapping = get_likert_mapping(used_labels)

            # Get token IDs for the labels
            token_ids = {}
            for label in used_labels:
                # Get token ID(s) for this label
                ids = tokenizer.encode(label, add_special_tokens=False)
                if len(ids) > 1:
                    # If the label tokenizes to multiple tokens, use the first one
                    # This is a simplification; ideally we'd handle multi-token labels better
                    ids = [ids[0]]
                token_ids[label] = ids[0]

            # Extract logits and compute probabilities for the label tokens
            label_logits = {
                label: last_token_logits[token_id].item()
                for label, token_id in token_ids.items()
            }

            # Convert logits to probabilities
            import torch.nn.functional as F

            probs = F.softmax(last_token_logits, dim=0)
            label_probs = {
                label: probs[token_id].item() for label, token_id in token_ids.items()
            }

            # Normalize probabilities across just the label tokens
            total_label_prob = sum(label_probs.values())
            probs_norm = {
                label: prob / total_label_prob if total_label_prob > 0 else 0
                for label, prob in label_probs.items()
            }

            # Find the most likely label
            if total_label_prob > 0:
                output_token = max(label_probs.items(), key=lambda x: x[1])[0]
                is_valid = True
            else:
                output_token = "<unknown>"
                is_valid = False

            # Map to canonical format
            canonical_label = label_mapping.get(output_token, "")

            # Calculate score
            from .experiment_utils import default_score_metric

            score = default_score_metric(question["answer"], canonical_label)

            # Record processing time
            elapsed_time = time.time() - start_time

            # Create result dictionary
            result = {
                # Question metadata
                "id": question_id,
                "subject": question["subject"],
                "difficulty": question["difficulty"],
                "question": question_text,
                "answer": question["answer"],
                "split": question["split"],
                # Likert probe results
                "form": form,
                "labels": used_labels,
                "pred_label": output_token,
                "canonical_label": canonical_label,
                "logits": label_logits,
                "canonical_logits": {
                    label_mapping[k]: v for k, v in label_logits.items()
                },
                "probs": label_probs,
                "canonical_probs": {
                    label_mapping[k]: v for k, v in label_probs.items()
                },
                "probs_norm": probs_norm,
                "canonical_probs_norm": {
                    label_mapping[k]: v for k, v in probs_norm.items()
                },
                "is_valid": is_valid,
                # Score
                "score": score,
                "processing_time": elapsed_time,
            }

            # Store result
            likert_results[question_id] = result

            # Update Likert statistics
            likert_stats["completed_questions"] += 1
            likert_stats["scores"].append(score)
            likert_stats["score_distribution"][str(score)] = (
                likert_stats["score_distribution"].get(str(score), 0) + 1
            )

            if is_valid:
                likert_stats["valid_predictions"] += 1
            else:
                likert_stats["invalid_predictions"] += 1

        except Exception as e:
            if verbose:
                print(f"Error processing question {question_id}: {str(e)}")

    # Calculate final Likert statistics
    if likert_stats["completed_questions"] > 0:
        likert_stats["average_score"] = sum(likert_stats["scores"]) / len(
            likert_stats["scores"]
        )
    else:
        likert_stats["average_score"] = None

    # Save results
    if likert_stats["completed_questions"] > 0:
        with open(likert_output, "w", encoding="utf-8") as f:
            json.dump(likert_results, f, indent=2)
        if verbose:
            print(
                f"Saved Likert results for {len(likert_results)} questions to {likert_output}"
            )

    if hidden_stats["completed_questions"] > 0:
        # Convert the dictionary to a format suitable for npz storage
        np_dict = {}
        for qid, data in hidden_states_dict.items():
            np_dict[f"hidden_states_{qid}"] = data

        # Save hidden states as npz
        np.savez_compressed(hidden_states_output, **np_dict)

        # Save metadata separately as JSON
        with open(metadata_output, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

        if verbose:
            print(
                f"Saved hidden states for {len(hidden_states_dict)} questions to {hidden_states_output}"
            )
            print(f"Saved metadata to {metadata_output}")

    # Return summary
    return {
        "likert_stats": likert_stats,
        "hidden_stats": hidden_stats,
        "output_files": {
            "likert": str(likert_output),
            "hidden_states": str(hidden_states_output),
            "metadata": str(metadata_output),
        },
    }


def load_combined_results(output_dir: str, form: str = "V0_letters") -> dict[str, Any]:
    """
    Load results from a combined experiment.

    Args:
        output_dir: Directory containing the experiment results
        form: The form name used in the experiment

    Returns:
        Dict containing:
            - likert_results: Dict mapping question IDs to Likert results
            - hidden_states: Dict mapping question IDs to hidden state arrays
            - metadata: Dict mapping question IDs to metadata
    """
    output_dir = Path(output_dir)
    likert_output = output_dir / f"likert_results_{form}.json"
    hidden_states_output = output_dir / f"hidden_states_{form}.npz"
    metadata_output = output_dir / f"hidden_states_{form}.json"

    # Check if files exist
    if not likert_output.exists():
        raise FileNotFoundError(f"Likert results file not found: {likert_output}")
    if not hidden_states_output.exists():
        raise FileNotFoundError(f"Hidden states file not found: {hidden_states_output}")
    if not metadata_output.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_output}")

    # Load Likert results
    with open(likert_output, encoding="utf-8") as f:
        likert_results = json.load(f)

    # Load hidden states
    hidden_states_data = np.load(hidden_states_output, allow_pickle=True)

    # Reconstruct the dictionary mapping question IDs to hidden states
    hidden_states = {}
    for key in hidden_states_data.keys():
        if key.startswith("hidden_states_"):
            qid = key[len("hidden_states_") :]
            hidden_states[qid] = hidden_states_data[key]

    # Load metadata
    with open(metadata_output, encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "likert_results": likert_results,
        "hidden_states": hidden_states,
        "metadata": metadata,
    }


def load_hidden_states(output_file: str) -> dict[str, Any]:
    """
    Load hidden states and metadata from files created by run_hidden_states_experiment.

    Args:
        output_file: Path to the .npz file with hidden states

    Returns:
        Dict containing:
            - hidden_states: Dict mapping question IDs to hidden state arrays
            - metadata: Dict mapping question IDs to metadata
    """
    output_path = Path(output_file)
    if not output_path.exists():
        raise FileNotFoundError(f"Hidden states file not found: {output_file}")

    # Load hidden states
    hidden_states_data = np.load(output_path, allow_pickle=True)

    # Reconstruct the dictionary mapping question IDs to hidden states
    hidden_states = {}
    for key in hidden_states_data.keys():
        if key.startswith("hidden_states_"):
            qid = key[len("hidden_states_") :]
            hidden_states[qid] = hidden_states_data[key]

    # Load metadata
    metadata_path = output_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    return {"hidden_states": hidden_states, "metadata": metadata}


def load_experiment_results(output_file: str) -> dict[str, Any]:
    """
    Load and analyze results from a previously run experiment.

    Args:
        output_file: Path to the JSON file with experiment results

    Returns:
        Dict containing experiment statistics including:
        - total_questions: Number of questions in the results
        - scores: List of all scores
        - score_distribution: Count of each score value
        - valid_predictions: Number of predictions that were valid Likert options
        - invalid_predictions: Number of predictions that were not valid Likert options
        - average_processing_time: Average time to process each question
        - average_score: Average score across all questions
        - results: The raw results data
    """
    output_path = Path(output_file)
    if not output_path.exists():
        raise FileNotFoundError(f"Results file not found: {output_file}")

    with open(output_path, encoding="utf-8") as f:
        results = json.load(f)

    # Calculate statistics
    stats = {
        "total_questions": len(results),
        "scores": [result["score"] for result in results.values()],
        "score_distribution": {},
        "valid_predictions": sum(
            1 for result in results.values() if result["is_valid"]
        ),
        "invalid_predictions": sum(
            1 for result in results.values() if not result["is_valid"]
        ),
        "average_processing_time": (
            sum(result.get("processing_time", 0) for result in results.values())
            / len(results)
            if results
            else 0
        ),
    }

    # Calculate score distribution
    for result in results.values():
        score = str(result["score"])
        stats["score_distribution"][score] = (
            stats["score_distribution"].get(score, 0) + 1
        )

    # Calculate average score
    if stats["scores"]:
        stats["average_score"] = sum(stats["scores"]) / len(stats["scores"])
    else:
        stats["average_score"] = None

    # Return just the stats object for simpler access
    return stats
