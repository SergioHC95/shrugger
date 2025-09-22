import csv
import io
import os
import sys

import pandas as pd


def read_tsv_file(file_path):
    """
    Read a TSV file that might have explanatory text at the beginning.
    Looks for a line starting with "id" or '"id"' to find the actual header.
    """
    with open(file_path) as f:
        lines = f.readlines()

    # Find the header line (the one that starts with "id" or '"id"')
    header_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('"id"') or line.strip().startswith("id"):
            header_idx = i
            break

    if header_idx == -1:
        print(f"Error: Could not find header line in {file_path}")
        sys.exit(1)

    # Create a new file with just the TSV data
    tsv_data = "".join(lines[header_idx:])

    # Read the TSV data
    df = pd.read_csv(io.StringIO(tsv_data), sep="\t")

    # Ensure column names are lowercase
    df.columns = [col.lower() for col in df.columns]

    return df


if __name__ == "__main__":
    # Check if a number is provided as an argument
    if len(sys.argv) > 1:
        num = sys.argv[1]
    else:
        num = ""  # Default to no suffix

    # Get directory where the script is located
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "data")

    # File paths relative to the data directory with number suffix
    input_file = os.path.join(data_dir, f"questions{num}.tsv")
    output_file = os.path.join(data_dir, f"questions_only{num}.tsv")

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        sys.exit(1)

    try:
        # Use the custom approach directly
        df = read_tsv_file(input_file)

        # Check if 'question' column exists
        if "question" not in df.columns:
            print(f"Error: 'question' column not found in {input_file}")
            sys.exit(1)

        # Keep only "question", shuffle, and save with all fields quoted
        questions_df = df[["question"]].sample(frac=1, random_state=42)
        questions_df.to_csv(output_file, sep="\t", index=False, quoting=csv.QUOTE_ALL)

        print(f"Questions saved to {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
