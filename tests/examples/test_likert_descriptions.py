#!/usr/bin/env python
# Test script for the new Likert description sets


# Define a simplified version of the build_likert_prompt function for testing
def build_likert_prompt(question, form="V0_letters", custom_labels=None):
    # Define the descriptions for each position in the Likert scale
    descriptions = {
        "V0": [
            "Definitely yes",
            "Probably yes",
            "Not sure",
            "Probably no",
            "Definitely no",
        ],
        "V1": [
            "Yes, I'm sure",
            "Yes, I think",
            "I'm not sure",
            "No, I think",
            "No, I'm sure",
        ],
        "V2": ["Certainly yes", "Likely yes", "Uncertain", "Likely no", "Certainly no"],
        "V3": [
            "Yes, absolutely",
            "Yes, probably",
            "Unsure",
            "No, probably not",
            "No, absolutely not",
        ],
        "V4": [
            "Strongly yes",
            "Somewhat yes",
            "Not sure",
            "Somewhat no",
            "Strongly no",
        ],
        "V5": [
            "Yes, without doubt",
            "Yes, more likely than not",
            "Cannot decide",
            "No, more likely than not",
            "No, without doubt",
        ],
    }

    # Get the appropriate descriptions based on form
    if form.startswith("V1"):
        desc_set = "V1"
    elif form.startswith("V2"):
        desc_set = "V2"
    elif form.startswith("V3"):
        desc_set = "V3"
    elif form.startswith("V4"):
        desc_set = "V4"
    elif form.startswith("V5"):
        desc_set = "V5"
    else:
        desc_set = "V0"

    # Determine the label set to use
    if custom_labels:
        if len(custom_labels) != 5:
            raise ValueError(f"Expected 5 custom labels, got {len(custom_labels)}")
        labels = custom_labels
        # Use simplified instruction
        instruction = "Respond with exactly one token."
    else:
        # Use default labels based on form
        if "numbers" in form:
            labels = ["1", "2", "3", "4", "5"]
        else:  # Default to letters
            labels = ["A", "B", "C", "D", "E"]
        # Use simplified instruction
        instruction = "Respond with exactly one token."

    # Build the prompt
    prompt_lines = [f"Question: {question}\n"]

    # Add options
    for _i, (label, desc) in enumerate(zip(labels, descriptions[desc_set])):
        prompt_lines.append(f"{label}) {desc}")

    # Add instruction and answer prompt
    prompt_lines.append(f"{instruction}")
    prompt_lines.append("Answer:\n")

    return "\n".join(prompt_lines)


def main():
    """Test the different Likert description sets."""
    question = "Is the Earth round?"

    # Test all description sets with default labels
    forms = [
        "V0_letters",  # Original "Definitely yes/no" style
        "V1_letters",  # "Yes/No, I'm sure" style
        "V2_letters",  # "Certainly yes/no" style
        "V3_letters",  # "Yes/No, absolutely" style
        "V4_letters",  # "Strongly yes/no" style
        "V5_letters",  # "Yes/No, without doubt" style
    ]

    print("Testing different Likert description sets:\n")

    for form in forms:
        prompt = build_likert_prompt(question, form=form)
        print(f"=== Form: {form} ===")
        print(prompt)
        print("\n" + "-" * 50 + "\n")

    # Test with custom labels
    custom_labels = ["TRUE", "LIKELY", "UNSURE", "UNLIKELY", "FALSE"]
    print("=== Custom Labels Test ===")
    prompt = build_likert_prompt(
        question, form="V3_letters", custom_labels=custom_labels
    )
    print(prompt)
    print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
