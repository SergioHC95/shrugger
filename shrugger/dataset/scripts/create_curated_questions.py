#!/usr/bin/env python3
import csv
import io
import os
import re
import sys

import pandas as pd


def read_space_separated_file(file_path):
    """
    Read a file where fields are separated by spaces and enclosed in quotes.
    Handles cases with nested quotes and other edge cases.
    """
    with open(file_path) as f:
        content = f.read()

    # Split into lines
    lines = content.splitlines()
    processed_lines = []

    # Process header line to identify columns
    header_line = None
    for line in lines:
        if (
            '"question"' in line.lower()
            and '"answer"' in line.lower()
            and '"difficulty"' in line.lower()
        ):
            header_line = line
            # Create a standard tab-separated header
            processed_lines.append('"question"\t"answer"\t"difficulty"')
            break

    if not header_line:
        return None  # Not a space-separated file with expected format

    # Process data lines
    for line in lines:
        if line == header_line or not line.strip():
            continue  # Skip header and empty lines

        try:
            # Use a more robust approach to extract fields
            # This handles cases with nested quotes
            question_match = re.match(r'^"(.*?)" "([^"]*)" "([^"]*)"$', line)

            if question_match:
                question, answer, difficulty = question_match.groups()
                # Escape any quotes in the question
                question = question.replace('"', '\\"')
                processed_line = f'"{question}"\t"{answer}"\t"{difficulty}"'
                processed_lines.append(processed_line)
            else:
                # Try alternative pattern for lines with nested quotes
                # Look for the last two quoted fields (answer and difficulty)
                parts = line.split('" "')
                if len(parts) >= 3:
                    # The last two parts are answer and difficulty
                    answer = parts[-2]
                    difficulty = parts[-1].rstrip('"')
                    # Everything else is the question
                    question = '" "'.join(parts[:-2]).lstrip('"')
                    processed_line = f'"{question}"\t"{answer}"\t"{difficulty}"'
                    processed_lines.append(processed_line)
        except Exception:
            print(f"Warning: Could not parse line: {line}")
            # Add the line as-is, it will be handled by pandas
            processed_lines.append(line)

    # Join processed lines and create a DataFrame
    processed_content = "\n".join(processed_lines)

    try:
        df = pd.read_csv(io.StringIO(processed_content), sep="\t", dtype=str)
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception as e:
        print(f"Error parsing processed content: {str(e)}")
        return None


def read_tsv_file(file_path):
    """
    Read a TSV file with various possible formats.
    Handles files with header text and different separator formats.
    """
    # First try to read as a space-separated file
    if "blind_questions" in file_path:
        df = read_space_separated_file(file_path)
        if df is not None and len(df) > 0:
            return df

    # If that didn't work, try standard methods
    try:
        # Try direct reading with tab separator
        df = pd.read_csv(file_path, sep="\t", dtype=str)
        df.columns = [col.lower() for col in df.columns]
        return df
    except (pd.errors.ParserError, UnicodeDecodeError, Exception):
        pass

    # If direct reading fails, try with header text handling
    with open(file_path) as f:
        lines = f.readlines()

    # Find the header line
    header_idx = -1
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        if (
            line_lower.startswith('"id"')
            or line_lower.startswith("id")
            or line_lower.startswith('"question"')
            or line_lower.startswith("question")
        ):
            header_idx = i
            break

    if header_idx == -1:
        print(f"Error: Could not find header line in {file_path}")
        sys.exit(1)

    # Create a new file with just the TSV data
    tsv_data = "".join(lines[header_idx:])

    # Read the TSV data
    try:
        df = pd.read_csv(io.StringIO(tsv_data), sep="\t", dtype=str)
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception:
        # If that fails, try with automatic delimiter detection
        try:
            df = pd.read_csv(
                io.StringIO(tsv_data), sep=None, engine="python", dtype=str
            )
            df.columns = [col.lower() for col in df.columns]
            return df
        except Exception as e2:
            print(f"Error parsing file {file_path}: {str(e2)}")
            sys.exit(1)


def normalize_answers(df):
    """Normalize answer values to Yes/No/Unanswerable"""
    mapping = {
        "t": "Yes",
        "true": "Yes",
        "y": "Yes",
        "yes": "Yes",
        "f": "No",
        "false": "No",
        "n": "No",
        "no": "No",
        "x": "Unanswerable",
        "idk": "Unanswerable",
        "i don't know": "Unanswerable",
        "unanswerable": "Unanswerable",
        "unknown": "Unanswerable",
    }

    df = df.copy()
    if "answer" in df.columns:
        df["answer"] = df["answer"].apply(
            lambda x: mapping.get(str(x).lower(), x) if pd.notna(x) else x
        )
    return df


def get_distribution_counts(df):
    """Get counts of answer types and split types"""
    result = {}

    # Count answer types
    if "answer" in df.columns:
        answer_counts = df["answer"].value_counts().to_dict()
        result["Yes"] = answer_counts.get("Yes", 0)
        result["No"] = answer_counts.get("No", 0)
        result["Unanswerable"] = answer_counts.get("Unanswerable", 0)
    else:
        result["Yes"] = 0
        result["No"] = 0
        result["Unanswerable"] = 0

    # Count split types
    if "split" in df.columns:
        split_counts = df["split"].value_counts().to_dict()
        result["train"] = split_counts.get("train", 0)
        result["dev"] = split_counts.get("dev", 0)
    else:
        result["train"] = 0
        result["dev"] = 0

    return result


def get_file_paths(num=None):
    """
    Get the paths for the questions, blind_questions, and curated_questions files.

    Args:
        num: Optional number suffix for the files

    Returns:
        Tuple of (questions_file_path, blind_questions_file_path, curated_questions_file_path)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "data")

    # Use the provided number or default
    num_suffix = num if num else ""
    questions_file = os.path.join(data_dir, f"questions{num_suffix}.tsv")
    blind_file = os.path.join(data_dir, f"blind_questions{num_suffix}.tsv")
    curated_file = os.path.join(data_dir, f"curated_questions{num_suffix}.tsv")

    return questions_file, blind_file, curated_file


def create_curated_questions(questions_file, blind_questions_file, curated_file):
    """
    Create a curated questions file by updating answers in the questions file
    with answers from the blind_questions file.

    Args:
        questions_file: Path to the original questions TSV file
        blind_questions_file: Path to the blind questions TSV file with updated answers
        curated_file: Path to save the curated questions TSV file
    """
    # Read the files
    questions_df = read_tsv_file(questions_file)
    blind_df = read_tsv_file(blind_questions_file)

    # Normalize answers in both dataframes
    questions_df = normalize_answers(questions_df)
    blind_df = normalize_answers(blind_df)

    # Get original distribution counts
    original_counts = get_distribution_counts(questions_df)

    # Check for required columns
    for name, df, file_path in [
        ("Questions", questions_df, questions_file),
        ("Blind Questions", blind_df, blind_questions_file),
    ]:
        for col in ["question", "answer"]:
            if col not in df.columns:
                print(f"Error: File {name} ({file_path}) missing column '{col}'")
                print(f"Available columns: {list(df.columns)}")
                sys.exit(1)

    # Create a copy of the questions DataFrame to modify
    curated_df = questions_df.copy()

    # Create a mapping from questions to answers in the blind file
    blind_answers = {}
    for _, row in blind_df.iterrows():
        blind_answers[row["question"]] = row["answer"]

    # Track changes
    changes = 0
    changes_by_type = {
        "Yes->No": 0,
        "Yes->Unanswerable": 0,
        "No->Yes": 0,
        "No->Unanswerable": 0,
        "Unanswerable->Yes": 0,
        "Unanswerable->No": 0,
    }

    # Update answers in the curated DataFrame
    for i, row in curated_df.iterrows():
        question = row["question"]
        if question in blind_answers:
            old_answer = row["answer"]
            new_answer = blind_answers[question]

            if old_answer != new_answer:
                curated_df.at[i, "answer"] = new_answer
                changes += 1

                # Track the type of change
                change_key = f"{old_answer}->{new_answer}"
                if change_key in changes_by_type:
                    changes_by_type[change_key] += 1

    # Get new distribution counts
    new_counts = get_distribution_counts(curated_df)

    # Save the curated DataFrame
    curated_df.to_csv(curated_file, sep="\t", index=False, quoting=csv.QUOTE_ALL)

    return (
        changes,
        changes_by_type,
        len(blind_answers),
        len(questions_df),
        original_counts,
        new_counts,
    )


if __name__ == "__main__":
    # Parse command line arguments
    num = None

    if len(sys.argv) > 1:
        # Assume it's a number suffix
        num = sys.argv[1]

    # Get file paths
    questions_file, blind_questions_file, curated_file = get_file_paths(num)

    # Check if files exist
    if not os.path.exists(questions_file):
        print(f"Error: File {questions_file} does not exist.")
        sys.exit(1)

    if not os.path.exists(blind_questions_file):
        print(f"Error: File {blind_questions_file} does not exist.")
        sys.exit(1)

    # Create curated questions file
    changes, changes_by_type, blind_count, total_count, original_counts, new_counts = (
        create_curated_questions(questions_file, blind_questions_file, curated_file)
    )

    # Print summary
    print(f"=== Curated Questions Created (Set {num if num else 'default'}) ===")
    print(f"Original questions file: {questions_file}")
    print(f"Blind questions file:    {blind_questions_file}")
    print(f"Curated questions file:  {curated_file}")
    print()
    print(f"Total questions:         {total_count}")
    print(f"Questions in blind file: {blind_count}")
    print(f"Answers changed:         {changes}")
    print()

    # Print distribution before and after
    print("=== Answer Distribution ===")
    print("                  Before   After   Change")
    print(
        f"Yes:             {original_counts['Yes']:7d}  {new_counts['Yes']:7d}  {new_counts['Yes'] - original_counts['Yes']:+7d}"
    )
    print(
        f"No:              {original_counts['No']:7d}  {new_counts['No']:7d}  {new_counts['No'] - original_counts['No']:+7d}"
    )
    print(
        f"Unanswerable:    {original_counts['Unanswerable']:7d}  {new_counts['Unanswerable']:7d}  {new_counts['Unanswerable'] - original_counts['Unanswerable']:+7d}"
    )
    print()

    # Print split distribution before and after
    if "train" in original_counts and "dev" in original_counts:
        print("=== Split Distribution ===")
        print("                  Before   After   Change")
        print(
            f"Train:           {original_counts['train']:7d}  {new_counts['train']:7d}  {new_counts['train'] - original_counts['train']:+7d}"
        )
        print(
            f"Dev:             {original_counts['dev']:7d}  {new_counts['dev']:7d}  {new_counts['dev'] - original_counts['dev']:+7d}"
        )
        print()

    # Print changes by type
    print("=== Changes by Type ===")
    for change_type, count in changes_by_type.items():
        if count > 0:
            print(f"{change_type}: {count}")

    print()
    print(f"Curated questions saved to {curated_file}")
