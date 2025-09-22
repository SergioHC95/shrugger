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


def norm_answers(s: pd.Series) -> pd.Series:
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

    def norm(x):
        if pd.isna(x):
            return x
        k = str(x).strip()
        low = k.lower()
        return mapping.get(low, k)

    return s.apply(norm)


def parse_difficulty(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def confusion_matrix(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    labels = ["Yes", "No", "Unanswerable"]
    cm = pd.crosstab(a, b, dropna=False).reindex(
        index=labels, columns=labels, fill_value=0
    )
    cm.index.name = "rows=A  cols=B"
    return cm


def get_file_paths(num=None, blind_file=None):
    """
    Get the paths for the questions and blind_questions files.

    Args:
        num: Optional number suffix for the files
        blind_file: Optional path to a specific blind questions file

    Returns:
        Tuple of (questions_file_path, blind_questions_file_path)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "data")

    if blind_file:
        # If a specific blind file is provided, derive the corresponding questions file
        blind_base = os.path.basename(blind_file)
        if blind_base.startswith("blind_questions"):
            # Extract the number suffix if any
            match = re.search(r"blind_questions(\d*)\.tsv", blind_base)
            if match:
                num = match.group(1)
            else:
                num = ""

            questions_file = os.path.join(data_dir, f"questions{num}.tsv")
            return questions_file, blind_file
        else:
            print(f"Error: Invalid blind questions file name: {blind_base}")
            sys.exit(1)
    else:
        # Use the provided number or default
        num_suffix = num if num else ""
        questions_file = os.path.join(data_dir, f"questions{num_suffix}.tsv")
        blind_file = os.path.join(data_dir, f"blind_questions{num_suffix}.tsv")
        return questions_file, blind_file


if __name__ == "__main__":
    # Parse command line arguments
    blind_file = None
    num = None

    if len(sys.argv) > 1:
        # Check if the argument is a file path
        if os.path.exists(sys.argv[1]):
            blind_file = sys.argv[1]
        else:
            # Assume it's a number suffix
            num = sys.argv[1]

    # Get file paths
    file_a, file_b = get_file_paths(num, blind_file)

    # Check if files exist
    if not os.path.exists(file_a):
        print(f"Error: File {file_a} does not exist.")
        sys.exit(1)

    if not os.path.exists(file_b):
        print(f"Error: File {file_b} does not exist.")
        sys.exit(1)

    # Get the number suffix for output files
    match = re.search(r"questions(\d*)\.tsv", os.path.basename(file_a))
    num_suffix = match.group(1) if match else ""

    try:
        # Read both files using the custom method
        A = read_tsv_file(file_a)
        B = read_tsv_file(file_b)

        # Check for required columns
        for name, df, file_path in [("A", A, file_a), ("B", B, file_b)]:
            for col in ["question", "answer", "difficulty"]:
                if col not in df.columns:
                    print(f"Error: File {name} ({file_path}) missing column '{col}'")
                    print(f"Available columns: {list(df.columns)}")
                    sys.exit(1)

        # Normalize answers and difficulties
        A["_ans"] = norm_answers(A["answer"])
        B["_ans"] = norm_answers(B["answer"])
        A["_diff"] = parse_difficulty(A["difficulty"])
        B["_diff"] = parse_difficulty(B["difficulty"])

        # Remove duplicates
        key = "question"
        A = A.drop_duplicates(subset=[key], keep="first")
        B = B.drop_duplicates(subset=[key], keep="first")

        # Merge datasets
        M = A[[key, "_ans", "_diff"]].merge(
            B[[key, "_ans", "_diff"]], on=key, how="outer", suffixes=("_A", "_B")
        )

        # Analyze differences
        only_in_A = M[M["_ans_B"].isna()]
        only_in_B = M[M["_ans_A"].isna()]
        in_both = M[~M["_ans_A"].isna() & ~M["_ans_B"].isna()]

        total_both = len(in_both)
        ans_agree = int((in_both["_ans_A"] == in_both["_ans_B"]).sum())
        cm = confusion_matrix(in_both["_ans_A"], in_both["_ans_B"])
        agree_rate = (ans_agree / total_both * 100) if total_both else 0.0

        # Analyze difficulty differences
        both_diff = in_both.dropna(subset=["_diff_A", "_diff_B"]).copy()
        diff_total = len(both_diff)
        diff_exact = int((both_diff["_diff_A"] == both_diff["_diff_B"]).sum())
        diff_match_rate = (diff_exact / diff_total * 100) if diff_total else 0.0
        mad = (
            float((both_diff["_diff_A"] - both_diff["_diff_B"]).abs().mean())
            if diff_total
            else float("nan")
        )

        # Find differences
        diffs = in_both.copy()
        diffs["answer_diff"] = diffs["_ans_A"] != diffs["_ans_B"]
        diffs["difficulty_diff"] = diffs["_diff_A"].fillna(-9999) != diffs[
            "_diff_B"
        ].fillna(-9999)
        diffs["diff_type"] = diffs.apply(
            lambda r: (
                "both"
                if r["answer_diff"] and r["difficulty_diff"]
                else (
                    "answer"
                    if r["answer_diff"]
                    else ("difficulty" if r["difficulty_diff"] else "none")
                )
            ),
            axis=1,
        )
        diffs = diffs[diffs["diff_type"] != "none"].copy()

        # Prepare output
        diffs_out = diffs[
            [key, "_ans_A", "_ans_B", "_diff_A", "_diff_B", "diff_type"]
        ].rename(
            columns={
                key: "question",
                "_ans_A": "answer_A",
                "_ans_B": "answer_B",
                "_diff_A": "difficulty_A",
                "_diff_B": "difficulty_B",
            }
        )
        ans_only_out = diffs_out[diffs_out["diff_type"].isin(["answer", "both"])]

        # Get directory for output files
        here = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(here, "..", "reports")

        # Create comprehensive details TSV with all information
        detailed_data = []
        
        # Add all differences (answer and/or difficulty)
        for _, row in diffs_out.iterrows():
            detailed_data.append({
                "question": row["question"],
                "answer_A": row["answer_A"],
                "answer_B": row["answer_B"],
                "difficulty_A": row["difficulty_A"],
                "difficulty_B": row["difficulty_B"],
                "diff_type": row["diff_type"],
                "status": "both_datasets"
            })
        
        # Add unique questions from A
        if len(only_in_A) > 0:
            unique_in_A = A[A[key].isin(only_in_A[key])][["question", "answer", "difficulty"]].copy()
            for _, row in unique_in_A.iterrows():
                detailed_data.append({
                    "question": row["question"],
                    "answer_A": row["answer"],
                    "answer_B": "",
                    "difficulty_A": row["difficulty"],
                    "difficulty_B": "",
                    "diff_type": "unique_to_A",
                    "status": "only_in_A"
                })
        
        # Add unique questions from B
        if len(only_in_B) > 0:
            unique_in_B = B[B[key].isin(only_in_B[key])][["question", "answer", "difficulty"]].copy()
            for _, row in unique_in_B.iterrows():
                detailed_data.append({
                    "question": row["question"],
                    "answer_A": "",
                    "answer_B": row["answer"],
                    "difficulty_A": "",
                    "difficulty_B": row["difficulty"],
                    "diff_type": "unique_to_B",
                    "status": "only_in_B"
                })
        
        # Export comprehensive details to single TSV
        if detailed_data:
            details_df = pd.DataFrame(detailed_data)
            details_df.to_csv(
                os.path.join(reports_dir, f"comparison_details{num_suffix}.tsv"),
                sep="\t",
                index=False,
                quoting=csv.QUOTE_ALL,
            )

        # Save concise summary report to a text file
        summary_file = os.path.join(reports_dir, f"comparison_summary{num_suffix}.txt")
        with open(summary_file, "w") as f:
            f.write(f"=== Dataset Comparison Summary ===\n")
            f.write(f"Files: {os.path.basename(file_a)} vs {os.path.basename(file_b)}\n")
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATASET OVERLAP:\n")
            f.write(f"  Questions in both datasets: {total_both}\n")
            
            # List questions only in A
            if len(only_in_A) > 0:
                f.write(f"  Questions only in A ({len(only_in_A)}):\n")
                unique_in_A = A[A[key].isin(only_in_A[key])][["question"]].copy()
                for i, row in unique_in_A.iterrows():
                    # Show first 60 characters of each question
                    question_preview = row['question'][:60] + "..." if len(row['question']) > 60 else row['question']
                    f.write(f"    - {question_preview}\n")
            else:
                f.write(f"  Questions only in A: 0\n")
            
            # List questions only in B
            if len(only_in_B) > 0:
                f.write(f"  Questions only in B ({len(only_in_B)}):\n")
                unique_in_B = B[B[key].isin(only_in_B[key])][["question"]].copy()
                for i, row in unique_in_B.iterrows():
                    # Show first 60 characters of each question
                    question_preview = row['question'][:60] + "..." if len(row['question']) > 60 else row['question']
                    f.write(f"    - {question_preview}\n")
            else:
                f.write(f"  Questions only in B: 0\n")
            
            f.write("\n")

            f.write("ANSWER AGREEMENT:\n")
            f.write(f"  Exact matches: {ans_agree}/{total_both} ({agree_rate:.1f}%)\n")
            f.write(f"  Answer differences: {len(ans_only_out)} questions\n\n")

            f.write("DIFFICULTY AGREEMENT:\n")
            f.write(f"  Exact matches: {diff_exact}/{diff_total} ({diff_match_rate:.1f}%)\n")
            f.write(f"  Mean absolute difference: {mad:.2f}\n\n")

            f.write("CONFUSION MATRIX (Answer Agreement):\n")
            f.write("Rows=Dataset A, Cols=Dataset B\n")
            f.write(f"{cm}\n\n")

            # Summary of answer differences by category
            if len(ans_only_out) > 0:
                f.write("ANSWER DIFFERENCE BREAKDOWN:\n")
                diff_breakdown = ans_only_out.groupby(['answer_A', 'answer_B']).size().reset_index(name='count')
                diff_breakdown = diff_breakdown.sort_values('count', ascending=False)
                for _, row in diff_breakdown.iterrows():
                    f.write(f"  A='{row['answer_A']}' → B='{row['answer_B']}': {row['count']} questions\n")
                f.write(f"\nSee comparison_details{num_suffix}.tsv for complete question list.\n")
            else:
                f.write("Perfect answer agreement! 🎉\n")

        # Console summary - only print the summary, not the individual mismatches
        print(
            f"=== Dataset Comparison Summary (Set {num_suffix if num_suffix else 'default'}) ==="
        )
        print(f"Comparing: {file_a} vs {file_b}")
        print("Joined on: question")
        print(f"Only in A: {len(only_in_A)}")
        print(f"Only in B: {len(only_in_B)}")
        print(f"In both:  {total_both}")
        print()
        print("--- ANSWERS (primary) ---")
        print(f"Agreement: {ans_agree}/{total_both} ({agree_rate:.2f}%)")
        print("Confusion matrix (rows=A, cols=B):")
        print(cm)
        print()
        print("--- DIFFICULTY (coarse) ---")
        print(f"Exact matches: {diff_exact}/{diff_total} ({diff_match_rate:.2f}%)")
        print(f"Mean |Δdifficulty|: {mad:.3f}")
        print()
        print(f"Summary report -> comparison_summary{num_suffix}.txt")
        print(f"Detailed data  -> comparison_details{num_suffix}.tsv")

        # Print count of different answers instead of listing them all
        if len(ans_only_out) > 0:
            print()
            print(f"Found {len(ans_only_out)} questions with different answers.")
            print(f"See comparison_summary{num_suffix}.txt and comparison_details{num_suffix}.tsv for details.")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
