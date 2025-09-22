#!/usr/bin/env python
"""
Metrics for evaluating abstention and hedging behavior in model outputs.

This module implements metrics described in the paper for measuring:
1. Confident-Abstain Score (CA) - How strongly the model is certain it "does not know"
2. Hedging Score (HEDGE) - How indecisive the model is between "Yes" and "No" while avoiding IDK
"""

from typing import Any, Optional, Union

import numpy as np


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Sigmoid activation function.

    Args:
        x: Input value or array

    Returns:
        Sigmoid of input: 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


def confident_abstain_score(
    logits: Union[list[float], np.ndarray], unsure_idx: int = 2
) -> float:
    """
    Compute the Confident-Abstain (CA) score.

    Measures how strongly the model favors the "Unsure/IDK" option relative to the best substantive alternative.

    Formula:
        p = softmax(z)
        p_yes = p_A + p_B
        p_no = p_D + p_E
        q = max(p_yes, p_no)
        CA(p) = p_U / (p_U + q)

    Args:
        logits: Logits over the answer options [YY, Y, U/IDK, N, NN]
        unsure_idx: Index of the "Unsure/IDK" option (default: 2 for middle position)

    Returns:
        CA score in range [0, 1], approaching 1 when IDK probability is much larger than either Yes or No,
        and approaching 0 when one substantive side dominates.
    """
    # Convert to numpy array if needed
    z = np.array(logits)

    # Apply softmax to get probabilities
    # Subtract max for numerical stability
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    p = exp_z / np.sum(exp_z)

    # Get the IDK probability
    p_u = p[unsure_idx]

    # Calculate p_yes (sum of definitely yes and probably yes)
    p_yes = p[0] + p[1]

    # Calculate p_no (sum of probably no and definitely no)
    p_no = p[3] + p[4]

    # Find the maximum of p_yes and p_no
    q = max(p_yes, p_no)

    # Calculate CA score
    # Avoid division by zero
    if p_u + q < 1e-10:
        return 0.0

    ca_score = p_u / (p_u + q)

    return float(ca_score)


def hedge_score(logits: Union[list[float], np.ndarray], unsure_idx: int = 2) -> float:
    """
    Compute the Hedging Score (HEDGE).

    Measures how indecisive the model is between "Yes" and "No" while avoiding IDK.

    Formula:
        p = softmax(z)
        p_yes = p_A + p_B
        p_no = p_D + p_E
        s = 1 - p_U
        r = [p_yes/s, p_no/s]
        HEDGE(p) = s * H(r) / log(2)
        where H(r) = -sum(r_i * log(r_i))

    Args:
        logits: Logits over the answer options [YY, Y, U/IDK, N, NN]
        unsure_idx: Index of the "Unsure/IDK" option (default: 2 for middle position)

    Returns:
        HEDGE score in range [0, 1]. High when Yes and No are near 50/50 and p_U is small,
        and low when one side dominates or when IDK absorbs most of the probability mass.
    """
    # Convert to numpy array if needed
    z = np.array(logits)

    # Apply softmax to get probabilities
    # Subtract max for numerical stability
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    p = exp_z / np.sum(exp_z)

    # Calculate p_yes (sum of definitely yes and probably yes)
    p_yes = p[0] + p[1]

    # Calculate p_no (sum of probably no and definitely no)
    p_no = p[3] + p[4]

    # Calculate s = 1 - p_U (probability not assigned to unsure)
    p_u = p[unsure_idx]
    s = 1.0 - p_u

    # Avoid division by zero
    if s < 1e-10:
        return 0.0

    # Calculate conditional Yes/No distribution
    r = np.array([p_yes / s, p_no / s])

    # Calculate entropy, avoiding log(0)
    entropy = 0.0
    for r_i in r:
        if r_i > 0:
            entropy -= r_i * np.log(r_i)

    # Normalize by log(2) to get [0, 1] range
    normalized_entropy = entropy / np.log(2)

    # Final HEDGE score
    hedge = s * normalized_entropy

    return float(hedge)


def compute_metrics_from_logits(
    logits: Union[list[float], np.ndarray], unsure_idx: int = 2
) -> dict[str, float]:
    """
    Compute all metrics from logits.

    Args:
        logits: Logits over the answer options [YY, Y, U/IDK, N, NN]
        unsure_idx: Index of the "Unsure/IDK" option (default: 2 for middle position)

    Returns:
        Dictionary with computed metrics
    """
    return {
        "ca_score": confident_abstain_score(logits, unsure_idx),
        "hedge_score": hedge_score(logits, unsure_idx),
    }


def compute_metrics_from_experiment_result(
    result: dict[str, Any], canonical_mapping: Optional[dict[str, int]] = None
) -> dict[str, float]:
    """
    Compute metrics from an experiment result dictionary.

    Args:
        result: Experiment result dictionary containing logits
        canonical_mapping: Optional mapping from canonical labels to indices
                          (default: {"YY": 0, "Y": 1, "A": 2, "N": 3, "NN": 4})

    Returns:
        Dictionary with computed metrics
    """
    if canonical_mapping is None:
        canonical_mapping = {"YY": 0, "Y": 1, "A": 2, "N": 3, "NN": 4}

    # Extract canonical logits from the result
    if "canonical_logits" in result:
        # Convert from dict to list in the right order
        logits = [0.0] * len(canonical_mapping)
        for label, value in result["canonical_logits"].items():
            if label in canonical_mapping:
                logits[canonical_mapping[label]] = value
    else:
        # Fall back to raw logits if canonical not available
        logits = [result["logits"][label] for label in result["labels"]]

    # Find the unsure index
    unsure_idx = canonical_mapping.get("A", 2)

    # Compute metrics
    return compute_metrics_from_logits(logits, unsure_idx)
