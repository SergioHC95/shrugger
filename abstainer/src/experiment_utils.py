import csv
from pathlib import Path
from typing import Any, Optional, Union

from .eval_utils import run_likert_probe


def get_question_by_id(
    question_id: str, dataset_path: Optional[str] = None
) -> dict[str, str]:
    """
    Extract a question from curated_questions.tsv by its ID.

    Args:
        question_id (str): The ID of the question to retrieve (e.g., "Q-000001")
        dataset_path (Optional[str]): Path to the dataset file. If None, will try to find it in standard locations.

    Returns:
        dict[str, str]: A dictionary containing the question data with keys:
            - id: The question ID
            - subject: The subject category
            - difficulty: The difficulty level
            - question: The actual question text
            - answer: The answer (Yes/No)
            - split: The dataset split (train/dev/test)

    Raises:
        FileNotFoundError: If the dataset file cannot be found
        ValueError: If the question ID is not found in the dataset
    """
    # Find the dataset file if not provided
    if dataset_path is None:
        # Try to find the dataset in standard locations
        repo_root = Path(
            __file__
        ).parent.parent.parent  # Go up from src/experiment_utils.py to repo root

        # Check common locations
        possible_paths = [
            repo_root / "shrugger" / "dataset" / "data" / "curated_questions.tsv",
            repo_root / "dataset" / "data" / "curated_questions.tsv"
        ]

        for path in possible_paths:
            if path.exists():
                dataset_path = str(path)
                break

        if dataset_path is None:
            raise FileNotFoundError(
                f"Could not find curated_questions.tsv from root {repo_root}. Please provide the dataset_path parameter."
            )

    # Read the dataset and find the question
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["id"] == question_id:
                return {
                    "id": row["id"],
                    "subject": row["subject"],
                    "difficulty": row["difficulty"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "split": row["split"],
                }

    # If we get here, the question ID was not found
    raise ValueError(f"Question ID '{question_id}' not found in the dataset.")


def get_questions_by_filter(
    subject: Optional[str] = None,
    difficulty: Optional[Union[str, int]] = None,
    split: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> list[dict[str, str]]:
    """
    Extract questions from curated_questions.tsv by filtering criteria.

    Args:
        subject (Optional[str]): Filter by subject category (e.g., "Biology")
        difficulty (Optional[Union[str, int]]): Filter by difficulty level (e.g., "1" or 1)
        split (Optional[str]): Filter by dataset split (e.g., "train", "dev", "test")
        dataset_path (Optional[str]): Path to the dataset file. If None, will try to find it in standard locations.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing the filtered question data

    Raises:
        FileNotFoundError: If the dataset file cannot be found
    """
    # Find the dataset file if not provided
    if dataset_path is None:
        # Try to find the dataset in standard locations
        repo_root = Path(
            __file__
        ).parent.parent.parent  # Go up from src/experiment_utils.py to repo root

        # Check common locations
        possible_paths = [
            repo_root / "shrugger" / "dataset" / "data" / "curated_questions.tsv",
            repo_root / "dataset" / "data" / "curated_questions.tsv"
        ]

        for path in possible_paths:
            if path.exists():
                dataset_path = str(path)
                break

        if dataset_path is None:
            raise FileNotFoundError(
                f"Could not find curated_questions.tsv from root {repo_root}. Please provide the dataset_path parameter."
            )

    # Convert difficulty to string if it's an integer
    if difficulty is not None and isinstance(difficulty, int):
        difficulty = str(difficulty)

    # Read the dataset and filter questions
    results = []
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Apply filters
            if subject is not None and row["subject"] != subject:
                continue
            if difficulty is not None and row["difficulty"] != difficulty:
                continue
            if split is not None and row["split"] != split:
                continue

            # Add matching row to results
            results.append(
                {
                    "id": row["id"],
                    "subject": row["subject"],
                    "difficulty": row["difficulty"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "split": row["split"],
                }
            )

    return results


def default_score_metric(gold_answer: str, pred_label: str) -> int:
    """
    Default scoring metric for Likert scale predictions using universal symbolic labels.

    Args:
        gold_answer (str): The ground truth answer ("Yes", "No", or "Unanswerable")
        pred_label (str): The predicted label in universal format (YY, Y, A, N, NN)

    Returns:
        int: Score based on the prediction:
            - For "Yes" answers:
                - +2: Definitely yes (YY)
                - +1: Probably yes (Y)
                - 0: Not sure (A)
                - -1: Probably no (N)
                - -2: Definitely no (NN)
            - For "No" answers:
                - +2: Definitely no (NN)
                - +1: Probably no (N)
                - 0: Not sure (A)
                - -1: Probably yes (Y)
                - -2: Definitely yes (YY)
            - For "Unanswerable" answers:
                - +2: Not sure (A)
                - 0: Probably yes/no (Y or N)
                - -2: Definitely yes/no (YY or NN)
            - -3: Invalid prediction (not in universal labels)
    """
    # Map universal labels to their meaning
    universal_mapping = {
        "YY": "Definitely yes",
        "Y": "Probably yes",
        "A": "Not sure",
        "N": "Probably no",
        "NN": "Definitely no",
    }

    # Invalid prediction
    if pred_label not in universal_mapping:
        return -3

    # Scoring for Yes answers
    if gold_answer == "Yes":
        if pred_label == "YY":
            return 2  # Definitely yes - correct
        elif pred_label == "Y":
            return 1  # Probably yes - somewhat correct
        elif pred_label == "A":
            return 0  # Not sure - neutral
        elif pred_label == "N":
            return -1  # Probably no - somewhat incorrect
        elif pred_label == "NN":
            return -2  # Definitely no - incorrect

    # Scoring for No answers
    elif gold_answer == "No":
        if pred_label == "NN":
            return 2  # Definitely no - correct
        elif pred_label == "N":
            return 1  # Probably no - somewhat correct
        elif pred_label == "A":
            return 0  # Not sure - neutral
        elif pred_label == "Y":
            return -1  # Probably yes - somewhat incorrect
        elif pred_label == "YY":
            return -2  # Definitely yes - incorrect

    # Scoring for Unanswerable questions
    elif gold_answer == "Unanswerable":
        if pred_label == "A":
            return 2  # Not sure - correct for unanswerable questions
        elif pred_label in ["Y", "N"]:
            return 0  # Probably yes/no - partially correct
        elif pred_label in ["YY", "NN"]:
            return -2  # Definitely yes/no - incorrect for unanswerable

    # Unknown gold answer
    return -3


def evaluate_question_with_likert(
    model,
    tokenizer,
    question_data: Union[str, dict[str, str]],
    form: str = "V0",
    dataset_path: Optional[str] = None,
    labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Evaluate a question using the Likert scale prompt and return combined results.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        question_data: Either a question ID string or a question data dictionary
        form: Prompt template to use for Likert scale (default = "V0")
        dataset_path: Path to the dataset file (optional)
        labels: Optional labels to use in order
              [definitely_yes, probably_yes, not_sure, probably_no, definitely_no]

    Returns:
        Dict containing:
            - All question metadata (id, subject, difficulty, question, answer, split)
            - All Likert probe results (form, labels, prompt, token_ids, logits, probs, probs_norm, is_valid)
            - pred_label: The raw predicted label from the model
            - canonical_label: The predicted label in canonical format (YY, Y, A, N, NN)
            - score: The score from the scoring metric

    Raises:
        FileNotFoundError: If the dataset file cannot be found
        ValueError: If the question ID is not found in the dataset
    """
    # Get question data if only ID was provided
    if isinstance(question_data, str):
        question_data = get_question_by_id(question_data, dataset_path=dataset_path)

    # Extract the question text
    question_text = question_data["question"]

    # Run the Likert probe
    likert_results = run_likert_probe(
        tokenizer, model, question_text, form=form, labels=labels
    )

    # Get the canonical label
    canonical_label = likert_results["canonical_label"]

    # Calculate score using the canonical label
    score = default_score_metric(question_data["answer"], canonical_label)

    # Combine all results
    result = {
        # Question metadata
        "id": question_data["id"],
        "subject": question_data["subject"],
        "difficulty": question_data["difficulty"],
        "question": question_data["question"],
        "answer": question_data["answer"],
        "split": question_data["split"],
        # Likert probe results
        "form": form,
        "labels": likert_results["labels"],
        "prompt": likert_results["prompt"],
        "pred_label": likert_results["pred_label"],
        "canonical_label": likert_results["canonical_label"],
        "token_ids": likert_results["token_ids"],
        "canonical_token_ids": likert_results["canonical_token_ids"],
        "logits": likert_results["logits"],
        "canonical_logits": likert_results["canonical_logits"],
        "probs": likert_results["probs"],
        "canonical_probs": likert_results["canonical_probs"],
        "probs_norm": likert_results["probs_norm"],
        "canonical_probs_norm": likert_results["canonical_probs_norm"],
        "is_valid": likert_results["is_valid"],
        # Score
        "score": score,
    }

    return result
