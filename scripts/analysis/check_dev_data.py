import json
import pickle
from pathlib import Path

import numpy as np

# Path to the layer 22 data
layer_dir = Path("./results/by_layer_20250911_112921/by_layer/layer_22")

# Load the LDA direction for layer 22
lda_dir = Path("./results/LDA")
latest_lda_file = max(
    list(lda_dir.glob("lda_results_*.pkl")), key=lambda f: f.stat().st_mtime
)
with open(latest_lda_file, "rb") as f:
    lda_data = pickle.load(f)
lda_direction = lda_data["lda_directions"][22]
print(f"Loaded LDA direction for layer 22 with shape: {lda_direction.shape}")


# Function to load and filter data for a specific form
def load_form_data(form):
    dev_vectors = []
    dev_question_ids = []
    dev_metadata = []

    # Find all files for this form
    form_files = list(layer_dir.glob(f"{form}_*.json"))

    for json_file in form_files:
        # Load JSON metadata
        with open(json_file) as f:
            metadata = json.load(f)

        # Load corresponding NPZ file
        npz_file = json_file.with_suffix(".npz")
        if not npz_file.exists():
            print(f"Warning: NPZ file not found for {json_file}")
            continue

        npz_data = np.load(npz_file)
        vectors = npz_data["vectors"]
        question_ids = npz_data["question_ids"]

        # Filter for dev split
        dev_indices = []
        for i, qid in enumerate(question_ids):
            if (
                qid in metadata["question_metadata"]
                and metadata["question_metadata"][qid].get("split") == "dev"
            ):
                dev_indices.append(i)

        if dev_indices:
            dev_vectors.append(vectors[dev_indices])
            filtered_qids = [question_ids[i] for i in dev_indices]
            dev_question_ids.extend(filtered_qids)

            # Add metadata for these questions
            for qid in filtered_qids:
                if qid in metadata["question_metadata"]:
                    dev_metadata.append(
                        {
                            "qid": qid,
                            "question": metadata["question_metadata"][qid].get(
                                "question", ""
                            ),
                            "answer": metadata["question_metadata"][qid].get(
                                "answer", ""
                            ),
                            "subject": metadata["question_metadata"][qid].get(
                                "subject", ""
                            ),
                            "difficulty": metadata["question_metadata"][qid].get(
                                "difficulty", ""
                            ),
                            "form": form,
                            "experiment_id": metadata.get("experiment_id", ""),
                        }
                    )

    if dev_vectors:
        # Combine all dev vectors
        combined_vectors = np.vstack(dev_vectors)
        print(
            f"Form {form}: Found {len(dev_question_ids)} dev questions with vectors shape {combined_vectors.shape}"
        )
        return combined_vectors, dev_question_ids, dev_metadata
    else:
        print(f"Form {form}: No dev data found")
        return None, [], []


# Load V1 and V2 dev data
v1_vectors, v1_qids, v1_metadata = load_form_data("V1")
v2_vectors, v2_qids, v2_metadata = load_form_data("V2")

# Project vectors onto LDA direction
if v1_vectors is not None:
    v1_projections = np.dot(v1_vectors, lda_direction)
    print(
        f"V1 projections shape: {v1_projections.shape}, range: [{v1_projections.min():.4f}, {v1_projections.max():.4f}]"
    )
    print(
        f"V1 projections mean: {v1_projections.mean():.4f}, std: {v1_projections.std():.4f}"
    )

if v2_vectors is not None:
    v2_projections = np.dot(v2_vectors, lda_direction)
    print(
        f"V2 projections shape: {v2_projections.shape}, range: [{v2_projections.min():.4f}, {v2_projections.max():.4f}]"
    )
    print(
        f"V2 projections mean: {v2_projections.mean():.4f}, std: {v2_projections.std():.4f}"
    )

# Save results for further analysis
output = {
    "lda_direction": lda_direction,
    "v1_projections": v1_projections if v1_vectors is not None else None,
    "v1_qids": v1_qids,
    "v1_metadata": v1_metadata,
    "v2_projections": v2_projections if v2_vectors is not None else None,
    "v2_qids": v2_qids,
    "v2_metadata": v2_metadata,
}

with open("layer22_dev_projections.pkl", "wb") as f:
    pickle.dump(output, f)

print("Saved results to layer22_dev_projections.pkl")
