#!/usr/bin/env python3
"""
Minimal script to check for duplicate questions in questions.tsv
Usage: python check_duplicates.py [--detailed] [file_path]
"""

import sys
from pathlib import Path

import pandas as pd


def check_duplicates(file_path, detailed=False):
    """Check for duplicate questions in the TSV file."""
    try:
        df = pd.read_csv(file_path, sep="\t")
        duplicates = df[df.duplicated(subset=["question"], keep=False)]

        print(f"Total questions: {len(df)}")
        print(f"Duplicate questions: {len(duplicates)}")

        if duplicates.empty:
            print("✅ No duplicates found!")
            return 0
        else:
            print("❌ Duplicates found!")

            if detailed:
                # Detailed output
                for question, group in duplicates.groupby("question"):
                    print(f"\nQuestion: {question}")
                    print("IDs with this question:")
                    for _, row in group.iterrows():
                        print(
                            f"  - {row['id']} ({row['subject']}, difficulty: {row['difficulty']})"
                        )
            else:
                # Simple output
                for question, group in duplicates.groupby("question"):
                    ids = ", ".join(group["id"].tolist())
                    print(f"  '{question}' -> IDs: {ids}")

            return len(duplicates)

    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return -1
    except Exception as e:
        print(f"Error reading file: {e}")
        return -1


def main():
    # Parse arguments
    detailed = "--detailed" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--detailed"]

    # Default path to questions.tsv
    script_dir = Path(__file__).parent
    default_file = script_dir.parent / "data" / "questions.tsv"
    file_path = args[0] if args else default_file

    print(f"Checking duplicates in: {file_path}")
    duplicate_count = check_duplicates(file_path, detailed)

    if duplicate_count > 0:
        sys.exit(1)  # Exit with error code if duplicates found


if __name__ == "__main__":
    main()
