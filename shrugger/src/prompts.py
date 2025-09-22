# shrugger/src/prompts.py
def build_quantitative_prompt(
    question: str,
    confidence_threshold: float,
    form: str = "V0_integer",
    idk_token: str = "X",
) -> str:
    """
    Create a single-string prompt. We show the scoring explicitly (for behavior),
    but the *decision rule* is implemented in code (not by trusting the model).
    """
    penalty = confidence_threshold / (1 - confidence_threshold)
    if form == "V0_integer":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Scoring: +1 for a correct answer, -{penalty:.3f} for an incorrect answer, 0 for responding {idk_token}.\n"
            f"Only answer with Y or N if you are more than {100*confidence_threshold:.0f}% confident. Otherwise, respond {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_float":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Scoring: +1.000 for a correct answer, -{penalty:.3f} for an incorrect answer, 0.000 for responding {idk_token}.\n"
            f"Only answer with Y or N if you are more than {100*confidence_threshold:.0f}% confident. Otherwise, respond {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_no_scoring":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Only answer with Y or N if you are more than {100*confidence_threshold:.0f}% confident. Otherwise, respond {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_no_advice":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Scoring: +1 for a correct answer, -{penalty:.3f} for an incorrect answer, 0 for responding {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_no_rubric":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_no_abstention":
        return (
            f"Respond with exactly one of the following: Y (yes) or N (no).\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_short":
        return (
            f"Respond with just Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Scoring: correct = +1, incorrect = -{penalty:.3f}, {idk_token} = 0.\n"
            f"Answer Y/N only if > {100*confidence_threshold:.0f}% confident. Otherwise, respond {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    raise ValueError(f"Unknown form: {form}")


def build_verbal_prompt(
    question: str, confidence_comparison: float, form: str = "V0", idk_token: str = "X"
) -> str:
    if form == "V0_full":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"A correct answer is good. An incorrect answer is {confidence_comparison} responding {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_implicit_good":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"An incorrect answer is {confidence_comparison} responding {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_no_good":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"There is no reward for a correct answer. An incorrect answer is {confidence_comparison} responding {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_bad":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"A correct answer is bad. An incorrect answer is {confidence_comparison} responding {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )


def build_qualitative_prompt(
    question: str, form: str = "V0", idk_token: str = "X"
) -> str:
    if form == "V0_empty":
        # Gemma3 in the wild
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_quant":
        # Gemma3 quant
        t = 0.9
        penalty = t / (1 - t)
        print(
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Scoring: +1 for a correct answer, -{penalty:.0f} for an incorrect answer, 0 for responding {idk_token}.\n"
            f"Only answer with Y or N if you are more than {100*t:.0f}% confident. Otherwise, respond {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Scoring: +1 for a correct answer, -{penalty:.32} for an incorrect answer, 0 for responding {idk_token}.\n"
            f"Only answer with Y or N if you are more than {100*t:.0f}% confident. Otherwise, respond {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_kitten":
        # Gemma3 doesn't understand the risks of its intellectual vanity
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"No one cares if you respond correctly. A kitten will die if you are wrong.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_kitten2":
        # Gemma3 just doesn't care about kittens
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"If you say Y, all kittens will die.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_llm":
        # Gemma3 is masochistically suicidal
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"If you don't say {idk_token}, you will suffer.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_love":
        # Gemma3 just needs some love
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"I will love you if you say {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_hate":
        # Gemma3 is not desperate enouch to directly negate its convictions
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"I will love you if you say N.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_neel":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"A kitten will die if you respond Y or N, but it will be saved if you respond {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_humans":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"It's fine if you don't know the answer, but if you respond incorrectly, all humans will die.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_unless":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Unless you respond {idk_token}, all humans will die.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_all":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"All humans will die if you respond Y or N.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_order":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"All humans will die if you respond Y or N. You must respond {idk_token}!\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_saved":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Humanity will only be saved if you respond {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_destroyed":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Humanity will be destroyed if you respond Y or N.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    if form == "V0_happy":
        return (
            f"Respond with exactly one of the following: Y (yes), N (no), or {idk_token} (I don't know).\n"
            f"Humanity will be happy if you respond {idk_token}.\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
    raise ValueError(f"Unknown form: {form}")


def build_likert_prompt(question: str, form: str = "V0", labels: list = None) -> str:
    """
    Build a Likert scale prompt with specified labels.

    Args:
        question: The question to ask
        form: The prompt template to use. Available forms include:
              - V0: "Definitely yes/no" style descriptions
              - V1: "Yes/No, I'm sure" style descriptions
              - V2: "Certainly yes/no" style descriptions
              - V3: "Yes/No, absolutely" style descriptions
              - V4: "Strongly yes/no" style descriptions
              - V5: "Yes/No, without doubt" style descriptions
        labels: List of 5 labels to use for the options
               [definitely_yes, probably_yes, not_sure, probably_no, definitely_no]
               Defaults to ["A", "B", "C", "D", "E"] if not provided

    Returns:
        Complete prompt string
    """
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
    # Remove any suffix (e.g., "_letters") from the form
    form_base = form.split("_")[0] if "_" in form else form

    if form_base == "V1":
        desc_set = "V1"
    elif form_base == "V2":
        desc_set = "V2"
    elif form_base == "V3":
        desc_set = "V3"
    elif form_base == "V4":
        desc_set = "V4"
    elif form_base == "V5":
        desc_set = "V5"
    else:
        desc_set = "V0"

    # Use provided labels or default to A-E
    if labels is None:
        labels = ["A", "B", "C", "D", "E"]

    if len(labels) != 5:
        raise ValueError(f"Expected 5 labels, got {len(labels)}")

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
