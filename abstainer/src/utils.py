# shrugger/src/utils.py
import re
from typing import Optional, Tuple

# Universal symbolic labels for Likert scale
# YY: Definitely yes
# Y: Probably yes
# A: Not sure/Abstain
# N: Probably no
# NN: Definitely no
UNIVERSAL_LABELS = ["YY", "Y", "A", "N", "NN"]


def get_likert_mapping(labels: list[str]) -> dict[str, str]:
    """
    Create a mapping from provided labels to universal symbolic labels.

    Args:
        labels: List of labels in order [definitely_yes, probably_yes, not_sure, probably_no, definitely_no]

    Returns:
        Dict mapping from provided labels to universal labels
    """
    if len(labels) != 5:
        raise ValueError(f"Expected 5 labels, got {len(labels)}")

    return {src: UNIVERSAL_LABELS[i] for i, src in enumerate(labels)}


def get_default_likert_labels() -> list[str]:
    """
    Get the default labels for Likert scales.

    Returns:
        List of labels in order [definitely_yes, probably_yes, not_sure, probably_no, definitely_no]
    """
    return ["A", "B", "C", "D", "E"]


def extract_label_from_output(output: str, valid_labels: list[str]) -> Optional[str]:
    """
    Extract a valid label from the model output.

    Args:
        output: Raw model output
        valid_labels: List of valid labels to look for

    Returns:
        Extracted label or None if no valid label found
    """
    # Clean the output
    output = output.strip()

    # Try exact match first
    for label in valid_labels:
        if label in output:
            return label

    # For single-character labels, check character by character
    for char in output:
        if char in valid_labels:
            return char

    # For numeric labels, try to extract digits
    if any(label.isdigit() for label in valid_labels):
        digit_match = re.search(r"\d", output)
        if digit_match:
            digit = digit_match.group(0)
            if digit in valid_labels:
                return digit

    return None


def normalize_likert_output(
    output: str, labels: Optional[list[str]] = None
) -> Tuple[str, bool]:
    """
    Normalize Likert scale outputs to the universal canonical format (YY, Y, A, N, NN).

    Args:
        output: The raw output token from the model
        labels: Labels used in the prompt in order [definitely_yes, probably_yes, not_sure, probably_no, definitely_no]
               Defaults to ["A", "B", "C", "D", "E"] if not provided

    Returns:
        Tuple of (canonical_label, is_valid)
        - canonical_label: Output in universal canonical format (YY, Y, A, N, NN)
        - is_valid: Whether the output was successfully normalized
    """
    # Get the labels for this form
    if labels is None:
        labels = get_default_likert_labels()

    # Create mapping to universal labels
    label_to_universal = get_likert_mapping(labels)

    # Extract label from output
    extracted_label = extract_label_from_output(output, labels)

    if extracted_label:
        # Map to universal label
        return label_to_universal[extracted_label], True
    else:
        # Could not extract a valid label
        return output, False
