

def build_instruction_prompt(instruction: str, input_text: str = "") -> str:
    instruction = str(instruction).strip()
    input_text = str(input_text).strip()

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def build_training_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
    instruction = str(instruction).strip()
    input_text = str(input_text).strip()
    output = str(output).strip()

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{output}"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{output}"
    )
