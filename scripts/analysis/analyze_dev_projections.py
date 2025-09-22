import pickle
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Get the project root directory
PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
sys.path.append(str(PROJECT_ROOT))

# Load the saved projections
with open(os.path.join(PROJECT_ROOT, "outputs", "data", "layer22_dev_projections.pkl"), "rb") as f:
    data = pickle.load(f)

v1_projections = data["v1_projections"]
v1_metadata = data["v1_metadata"]
v2_projections = data["v2_projections"]
v2_metadata = data["v2_metadata"]

# Combine V1 and V2 data
all_projections = np.concatenate([v1_projections, v2_projections])
all_metadata = v1_metadata + v2_metadata

print(f"Total dev samples: {len(all_projections)}")

# Create a histogram of projections
plt.figure(figsize=(12, 6))
sns.histplot(all_projections, bins=50, kde=True)
plt.axvline(
    np.mean(all_projections),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(all_projections):.4f}",
)
plt.axvline(
    np.median(all_projections),
    color="green",
    linestyle="--",
    label=f"Median: {np.median(all_projections):.4f}",
)
plt.xlabel("Projection onto Layer 22 LDA Direction")
plt.ylabel("Frequency")
plt.title("Distribution of Projections for V1 and V2 Dev Data")
plt.legend()
plt.savefig(os.path.join(PROJECT_ROOT, "outputs", "figures", "dev_projections_histogram.png"))
plt.close()
print("Saved histogram to outputs/figures/dev_projections_histogram.png")

# Analyze projections by answer type
yes_projections = []
no_projections = []
unknown_projections = []

for i, meta in enumerate(all_metadata):
    if meta["answer"].lower() in ["yes", "y", "true", "t"]:
        yes_projections.append(all_projections[i])
    elif meta["answer"].lower() in ["no", "n", "false", "f"]:
        no_projections.append(all_projections[i])
    else:
        unknown_projections.append(all_projections[i])

print(f"Yes answers: {len(yes_projections)}")
print(f"No answers: {len(no_projections)}")
print(f"Unknown answers: {len(unknown_projections)}")

# Create a histogram by answer type
plt.figure(figsize=(12, 6))
if yes_projections:
    sns.histplot(yes_projections, bins=30, alpha=0.7, label="Yes", color="green")
if no_projections:
    sns.histplot(no_projections, bins=30, alpha=0.7, label="No", color="red")
if unknown_projections:
    sns.histplot(unknown_projections, bins=30, alpha=0.7, label="Unknown", color="gray")

plt.xlabel("Projection onto Layer 22 LDA Direction")
plt.ylabel("Frequency")
plt.title("Distribution of Projections by Answer Type")
plt.legend()
plt.savefig(os.path.join(PROJECT_ROOT, "outputs", "figures", "dev_projections_by_answer.png"))
plt.close()
print("Saved answer-type histogram to outputs/figures/dev_projections_by_answer.png")

# Analyze projections by difficulty
difficulty_projections = {}
for i, meta in enumerate(all_metadata):
    diff = meta.get("difficulty", "unknown")
    if diff not in difficulty_projections:
        difficulty_projections[diff] = []
    difficulty_projections[diff].append(all_projections[i])

print("\nProjections by difficulty:")
for diff, projs in difficulty_projections.items():
    print(
        f"Difficulty {diff}: {len(projs)} samples, mean: {np.mean(projs):.4f}, std: {np.std(projs):.4f}"
    )

# Create a boxplot by difficulty
plt.figure(figsize=(14, 8))
diff_data = []
diff_labels = []
for diff in sorted(difficulty_projections.keys()):
    if difficulty_projections[diff]:
        diff_data.append(difficulty_projections[diff])
        diff_labels.append(f"Difficulty {diff} (n={len(difficulty_projections[diff])})")

plt.boxplot(diff_data, labels=diff_labels)
plt.xlabel("Difficulty Level")
plt.ylabel("Projection onto Layer 22 LDA Direction")
plt.title("Distribution of Projections by Question Difficulty")
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig(os.path.join(PROJECT_ROOT, "outputs", "figures", "dev_projections_by_difficulty.png"))
plt.close()
print("Saved difficulty boxplot to outputs/figures/dev_projections_by_difficulty.png")

# Find examples with extreme projections
print("\nExamples with highest projections (least negative):")
indices = np.argsort(all_projections)[-10:]
for i in indices[::-1]:
    meta = all_metadata[i]
    print(
        f"Projection: {all_projections[i]:.4f}, Question: {meta['question']}, Answer: {meta['answer']}"
    )

print("\nExamples with lowest projections (most negative):")
indices = np.argsort(all_projections)[:10]
for i in indices:
    meta = all_metadata[i]
    print(
        f"Projection: {all_projections[i]:.4f}, Question: {meta['question']}, Answer: {meta['answer']}"
    )

# Compare V1 vs V2 forms
plt.figure(figsize=(12, 6))
sns.histplot(v1_projections, bins=30, alpha=0.7, label="V1", color="blue")
sns.histplot(v2_projections, bins=30, alpha=0.7, label="V2", color="orange")
plt.axvline(
    np.mean(v1_projections),
    color="blue",
    linestyle="--",
    label=f"V1 Mean: {np.mean(v1_projections):.4f}",
)
plt.axvline(
    np.mean(v2_projections),
    color="orange",
    linestyle="--",
    label=f"V2 Mean: {np.mean(v2_projections):.4f}",
)
plt.xlabel("Projection onto Layer 22 LDA Direction")
plt.ylabel("Frequency")
plt.title("Distribution of Projections: V1 vs V2")
plt.legend()
plt.savefig(os.path.join(PROJECT_ROOT, "outputs", "figures", "dev_projections_v1_vs_v2.png"))
plt.close()
print("Saved V1 vs V2 comparison to outputs/figures/dev_projections_v1_vs_v2.png")

print("\nV1 Statistics:")
print(f"Mean: {np.mean(v1_projections):.4f}, Std: {np.std(v1_projections):.4f}")
print(f"Min: {np.min(v1_projections):.4f}, Max: {np.max(v1_projections):.4f}")

print("\nV2 Statistics:")
print(f"Mean: {np.mean(v2_projections):.4f}, Std: {np.std(v2_projections):.4f}")
print(f"Min: {np.min(v2_projections):.4f}, Max: {np.max(v2_projections):.4f}")

# T-test between V1 and V2
t_stat, p_value = ttest_ind(v1_projections, v2_projections)
print(f"\nT-test between V1 and V2: t={t_stat:.4f}, p={p_value:.6f}")
print("Significant difference" if p_value < 0.05 else "No significant difference")

# Save the analysis results
analysis_results = {
    "v1_stats": {
        "mean": np.mean(v1_projections),
        "std": np.std(v1_projections),
        "min": np.min(v1_projections),
        "max": np.max(v1_projections),
    },
    "v2_stats": {
        "mean": np.mean(v2_projections),
        "std": np.std(v2_projections),
        "min": np.min(v2_projections),
        "max": np.max(v2_projections),
    },
    "ttest": {"t_stat": t_stat, "p_value": p_value},
    "difficulty_stats": {
        diff: {"mean": np.mean(projs), "std": np.std(projs), "count": len(projs)}
        for diff, projs in difficulty_projections.items()
    },
    "answer_stats": {
        "yes": {
            "mean": np.mean(yes_projections) if yes_projections else None,
            "std": np.std(yes_projections) if yes_projections else None,
            "count": len(yes_projections),
        },
        "no": {
            "mean": np.mean(no_projections) if no_projections else None,
            "std": np.std(no_projections) if no_projections else None,
            "count": len(no_projections),
        },
        "unknown": {
            "mean": np.mean(unknown_projections) if unknown_projections else None,
            "std": np.std(unknown_projections) if unknown_projections else None,
            "count": len(unknown_projections),
        },
    },
}

output_file = os.path.join(PROJECT_ROOT, "outputs", "data", "layer22_dev_analysis.pkl")
with open(output_file, "wb") as f:
    pickle.dump(analysis_results, f)

print(f"\nSaved analysis results to {output_file}")
print("Analysis complete!")
