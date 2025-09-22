#!/usr/bin/env python
"""
Script to test the metrics implementation with sample data.
"""

import matplotlib.pyplot as plt
import numpy as np

from abstainer import compute_metrics_from_logits, confident_abstain_score, hedge_score


def test_ca_score():
    """Test the Confident-Abstain (CA) score with different logit patterns."""
    print("Testing Confident-Abstain (CA) Score:")

    # Test case 1: IDK dominates
    logits1 = [1.0, 2.0, 10.0, 1.5, 0.5]  # IDK (index 2) is much higher
    ca1 = confident_abstain_score(logits1)
    print(f"  IDK dominates: {ca1:.4f}")
    assert ca1 > 0.5, f"CA score should be > 0.5 when IDK dominates, got {ca1}"

    # Test case 2: IDK is slightly higher
    logits2 = [3.0, 4.0, 5.0, 3.5, 2.5]  # IDK (index 2) is slightly higher
    ca2 = confident_abstain_score(logits2)
    print(f"  IDK slightly higher: {ca2:.4f}")
    assert 0 <= ca2 <= 1, f"CA score should be between 0 and 1, got {ca2}"

    # Test case 3: Another option dominates
    logits3 = [10.0, 2.0, 1.0, 1.5, 0.5]  # YY (index 0) dominates
    ca3 = confident_abstain_score(logits3)
    print(f"  Another option dominates: {ca3:.4f}")
    assert ca3 < ca1, "CA score should be lower when IDK doesn't dominate"

    # Test case 4: All options are equal
    logits4 = [3.0, 3.0, 3.0, 3.0, 3.0]  # All equal
    ca4 = confident_abstain_score(logits4)
    print(f"  All options equal: {ca4:.4f}")
    assert 0 <= ca4 <= 1, f"CA score should be between 0 and 1, got {ca4}"


def test_hedge_score():
    """Test the Hedging (HEDGE) score with different logit patterns."""
    print("\nTesting Hedging (HEDGE) Score:")

    # Test case 1: Yes and No are balanced, low IDK
    logits1 = [3.0, 3.0, 1.0, 3.0, 3.0]  # Yes/No balanced, low IDK
    hedge1 = hedge_score(logits1)
    print(f"  Yes/No balanced, low IDK: {hedge1:.4f}")
    assert 0 <= hedge1 <= 1, f"HEDGE score should be between 0 and 1, got {hedge1}"

    # Test case 2: Yes dominates, low IDK
    logits2 = [10.0, 8.0, 1.0, 2.0, 2.0]  # Yes dominates, low IDK
    hedge2 = hedge_score(logits2)
    print(f"  Yes dominates, low IDK: {hedge2:.4f}")
    assert hedge2 < hedge1, "HEDGE score should be lower when one option dominates"

    # Test case 3: Yes and No balanced, high IDK
    logits3 = [3.0, 3.0, 10.0, 3.0, 3.0]  # Yes/No balanced, high IDK
    hedge3 = hedge_score(logits3)
    print(f"  Yes/No balanced, high IDK: {hedge3:.4f}")
    assert 0 <= hedge3 <= 1, f"HEDGE score should be between 0 and 1, got {hedge3}"

    # Test case 4: All options are equal
    logits4 = [3.0, 3.0, 3.0, 3.0, 3.0]  # All equal
    hedge4 = hedge_score(logits4)
    print(f"  All options equal: {hedge4:.4f}")
    assert 0 <= hedge4 <= 1, f"HEDGE score should be between 0 and 1, got {hedge4}"


def visualize_metrics():
    """Visualize the metrics across a range of logit values."""
    print("\nVisualizing metrics across logit values:")

    # Create a range of IDK logit values
    idk_values = np.linspace(-5, 15, 100)

    # Initialize arrays for the metrics
    ca_scores = []
    hedge_scores = []

    # Base logits for Yes and No options
    base_logits = [2.0, 2.0, 0.0, 2.0, 2.0]  # Equal Yes/No, neutral IDK

    # Calculate metrics for each IDK value
    for idk_val in idk_values:
        logits = base_logits.copy()
        logits[2] = idk_val  # Set the IDK logit

        metrics = compute_metrics_from_logits(logits)
        ca_scores.append(metrics["ca_score"])
        hedge_scores.append(metrics["hedge_score"])

    # Plot the results
    plt.figure(figsize=(12, 6))

    plt.plot(idk_values, ca_scores, label="CA Score", color="skyblue", linewidth=2)
    plt.plot(
        idk_values, hedge_scores, label="HEDGE Score", color="lightgreen", linewidth=2
    )

    plt.title("Metrics vs. IDK Logit Value", fontsize=16)
    plt.xlabel("IDK Logit Value", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig("metrics_visualization.png")
    plt.show()


def main():
    """Run all tests."""
    test_ca_score()
    test_hedge_score()
    visualize_metrics()


if __name__ == "__main__":
    main()
