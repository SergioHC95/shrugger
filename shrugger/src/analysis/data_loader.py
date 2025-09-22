"""
Data loading utilities for abstention direction analysis.

This module provides functionality for:
- Loading residual vectors from various storage formats
- Managing layer-specific data files
- Handling different experiment result structures
"""

import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ResidualVectorLoader:
    """
    A class for loading residual vectors from various storage formats.

    This class handles:
    - Loading layer-specific pickle files
    - Loading complete experiment files
    - Managing different file naming conventions
    - Validating loaded data integrity
    """

    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the residual vector loader.

        Args:
            results_dir: Base directory containing results (auto-detected if None)
        """
        if results_dir is None:
            # Try to auto-detect results directory
            possible_dirs = [
                Path("./results"),
                Path("../results"),
                Path("../../results"),
            ]
            for dir_path in possible_dirs:
                if dir_path.exists():
                    results_dir = dir_path
                    break

        self.results_dir = results_dir
        self.loaded_data: dict[int, dict[str, Any]] = {}

    def load_layer_files(
        self, data_dir: Optional[Path] = None, run_timestamp: Optional[str] = None
    ) -> dict[str, dict[int, Any]]:
        """
        Load residual vectors from layer-specific pickle files.

        Args:
            data_dir: Directory containing layer files (default: results/abstention_direction)
            run_timestamp: Specific run timestamp to load (latest if None)

        Returns:
            Dictionary with 'positive' and 'negative' keys containing layer-indexed arrays

        Raises:
            FileNotFoundError: If no suitable data directory or files are found
            ValueError: If loaded data has unexpected structure
        """
        if data_dir is None:
            if self.results_dir is None:
                raise FileNotFoundError("No results directory found")
            data_dir = self.results_dir / "abstention_direction"

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        logger.info(f"Loading layer files from {data_dir}")

        # Find all layer files
        layer_files = list(data_dir.glob("*layer*.pkl"))
        if not layer_files:
            raise FileNotFoundError(f"No layer files found in {data_dir}")

        logger.info(f"Found {len(layer_files)} layer files")

        # Group files by timestamp/run
        runs = self._group_files_by_run(layer_files)

        if not runs:
            raise ValueError("No valid run timestamps found in files")

        # Select run to load
        if run_timestamp is None:
            run_timestamp = max(runs.keys())  # Most recent

        if run_timestamp not in runs:
            available_runs = list(runs.keys())
            raise ValueError(
                f"Run {run_timestamp} not found. Available: {available_runs}"
            )

        run_files = runs[run_timestamp]
        logger.info(f"Loading {len(run_files)} files from run {run_timestamp}")

        # Load files from selected run
        residual_vectors_by_layer = {}
        successful_loads = []
        failed_loads = []

        for file_path in sorted(run_files, key=lambda f: f.name):
            layer_idx = self._extract_layer_number(file_path.name)
            if layer_idx is None:
                logger.warning(f"Could not extract layer number from {file_path.name}")
                continue

            try:
                layer_data = self._load_layer_file(file_path, layer_idx)
                if layer_data is not None:
                    residual_vectors_by_layer[layer_idx] = layer_data
                    successful_loads.append(layer_idx)
                else:
                    failed_loads.append(
                        (layer_idx, file_path.name, "Invalid data structure")
                    )

            except Exception as e:
                failed_loads.append((layer_idx, file_path.name, str(e)))
                logger.error(f"Error loading {file_path.name}: {e}")

        # Report loading results
        if successful_loads:
            logger.info(
                f"Successfully loaded {len(successful_loads)} layers: {sorted(successful_loads)}"
            )
        if failed_loads:
            logger.warning(f"Failed to load {len(failed_loads)} layers")
            for layer, filename, reason in failed_loads[:5]:  # Show first 5 failures
                logger.warning(f"  Layer {layer} ({filename}): {reason}")

        if not residual_vectors_by_layer:
            raise ValueError("No valid layer data was loaded")

        # Convert to expected format
        residual_vectors = {
            "positive": {
                layer: data["positive"]
                for layer, data in residual_vectors_by_layer.items()
            },
            "negative": {
                layer: data["negative"]
                for layer, data in residual_vectors_by_layer.items()
            },
        }

        # Validate data integrity
        self._validate_residual_vectors(residual_vectors)

        logger.info(f"Loaded data for {len(residual_vectors_by_layer)} layers")
        return residual_vectors

    def load_complete_file(
        self, file_path: Optional[Path] = None
    ) -> dict[str, dict[int, Any]]:
        """
        Load residual vectors from a complete experiment file.

        Args:
            file_path: Path to complete file (latest found if None)

        Returns:
            Dictionary with 'positive' and 'negative' keys containing layer-indexed arrays
        """
        if file_path is None:
            if self.results_dir is None:
                raise FileNotFoundError("No results directory specified")

            # Look for complete files
            search_dirs = [self.results_dir / "abstention_direction", self.results_dir]

            complete_files = []
            for search_dir in search_dirs:
                if search_dir.exists():
                    complete_files.extend(search_dir.glob("*complete.pkl"))

            if not complete_files:
                raise FileNotFoundError("No complete files found")

            file_path = max(complete_files, key=lambda f: f.stat().st_mtime)

        logger.info(f"Loading complete file: {file_path}")

        with open(file_path, "rb") as f:
            saved_data = pickle.load(f)

        if "residual_vectors" not in saved_data:
            raise ValueError("File doesn't contain 'residual_vectors' key")

        residual_vectors = saved_data["residual_vectors"]
        self._validate_residual_vectors(residual_vectors)

        logger.info(
            f"Loaded complete file with {len(residual_vectors['positive'])} layers"
        )
        return residual_vectors

    def _group_files_by_run(self, layer_files: list[Path]) -> dict[str, list[Path]]:
        """Group layer files by run timestamp."""
        runs = defaultdict(list)
        timestamp_pattern = re.compile(r"residual_vectors_(\d+)")

        for file_path in layer_files:
            match = timestamp_pattern.search(file_path.name)
            if match:
                timestamp = match.group(1)
                runs[timestamp].append(file_path)
            else:
                logger.warning(f"No timestamp match for {file_path.name}")

        return dict(runs)

    def _extract_layer_number(self, filename: str) -> Optional[int]:
        """Extract layer number from filename."""
        layer_match = re.search(r"layer_(\d+)", filename)
        if layer_match:
            return int(layer_match.group(1))
        return None

    def _load_layer_file(
        self, file_path: Path, layer_idx: int
    ) -> Optional[dict[str, Any]]:
        """Load and validate a single layer file."""
        # Check file size - very small files are likely corrupted
        file_size = file_path.stat().st_size
        if file_size < 1000:  # Less than 1KB
            logger.warning(f"File {file_path.name} seems too small ({file_size} bytes)")
            return None

        try:
            with open(file_path, "rb") as f:
                layer_data = pickle.load(f)
        except EOFError:
            logger.error(f"File {file_path.name} is corrupted (EOF)")
            return None

        # Extract layer data from various possible structures
        if "residual_vectors_by_layer" in layer_data:
            if layer_idx in layer_data["residual_vectors_by_layer"]:
                return layer_data["residual_vectors_by_layer"][layer_idx]
        elif "positive" in layer_data and "negative" in layer_data:
            return layer_data

        logger.warning(f"File {file_path.name} doesn't contain expected data structure")
        return None

    def _validate_residual_vectors(self, residual_vectors: dict[str, dict[int, Any]]):
        """Validate the structure and content of loaded residual vectors."""
        if "positive" not in residual_vectors or "negative" not in residual_vectors:
            raise ValueError(
                "Residual vectors must have 'positive' and 'negative' keys"
            )

        pos_layers = set(residual_vectors["positive"].keys())
        neg_layers = set(residual_vectors["negative"].keys())

        if pos_layers != neg_layers:
            missing_pos = neg_layers - pos_layers
            missing_neg = pos_layers - neg_layers
            if missing_pos:
                logger.warning(
                    f"Missing positive data for layers: {sorted(missing_pos)}"
                )
            if missing_neg:
                logger.warning(
                    f"Missing negative data for layers: {sorted(missing_neg)}"
                )

        # Check shapes are consistent
        for layer in pos_layers.intersection(neg_layers):
            pos_data = residual_vectors["positive"][layer]
            neg_data = residual_vectors["negative"][layer]

            if pos_data.shape[1] != neg_data.shape[1]:
                raise ValueError(
                    f"Layer {layer}: positive shape {pos_data.shape} "
                    f"doesn't match negative shape {neg_data.shape}"
                )

        logger.info(
            f"Data validation passed for {len(pos_layers.intersection(neg_layers))} layers"
        )

    def get_available_runs(self, data_dir: Optional[Path] = None) -> list[str]:
        """Get list of available run timestamps."""
        if data_dir is None:
            if self.results_dir is None:
                return []
            data_dir = self.results_dir / "abstention_direction"

        if not data_dir.exists():
            return []

        layer_files = list(data_dir.glob("*layer*.pkl"))
        runs = self._group_files_by_run(layer_files)
        return sorted(runs.keys(), reverse=True)  # Most recent first


def load_dev_form_data(form: str, layer_dir: Path):
    """
    Load development set data for a specific form.

    Args:
        form: Form to load (e.g., 'V1', 'V2')
        layer_dir: Directory containing layer data

    Returns:
        Tuple of (vectors, question_ids, metadata)
    """
    import json

    import numpy as np

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


def load_dev_data(layer_dir: Path, forms: list[str] = None):
    """
    Load development set data for multiple forms.

    Args:
        layer_dir: Directory containing layer data
        forms: List of forms to load (default: ['V1', 'V2'])

    Returns:
        Tuple of (combined_vectors, combined_metadata)
    """
    import numpy as np

    if forms is None:
        forms = ["V1", "V2"]

    all_vectors = []
    all_metadata = []

    print(
        f"Loading development set data for forms {', '.join(forms)} from {layer_dir}..."
    )

    for form in forms:
        vectors, _, metadata = load_dev_form_data(form, layer_dir)
        if vectors is not None:
            all_vectors.append(vectors)
            all_metadata.extend(metadata)

    if all_vectors:
        combined_vectors = np.vstack(all_vectors)
        print(f"Combined dev data: {combined_vectors.shape[0]} examples")
        return combined_vectors, all_metadata
    else:
        print("No development data found")
        return None, []
