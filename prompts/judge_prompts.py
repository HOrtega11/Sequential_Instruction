

from typing import Dict, List


def build_alpaca_judge_messages(
    instruction: str,
    input_text: str,
    response_a: str,
    response_b: str,
) -> List[Dict[str, str]]:
    user_content = (
        "You are evaluating two responses to the same instruction.\n\n"
        "Task:\n"
        "Decide which response is better overall, or declare a tie if they are comparably good.\n\n"
        "Evaluate each response on these dimensions using a 1-5 scale, where 1 is very poor and 5 is excellent:\n"
        "1. Instruction Following\n"
        "2. Correctness\n"
        "3. Clarity\n"
        "4. Completeness\n"
        "5. Hallucination / Fabrication Risk\n\n"
        "Important scoring note:\n"
        "- For hallucination_risk, 1 means very low risk and 5 means very high risk.\n\n"
        "Instruction:\n"
        f"{instruction}\n\n"
    )

    if input_text.strip():
        user_content += f"Input:\n{input_text}\n\n"

    user_content += (
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Judge fairly. Do not prefer a response because it appears first, is longer, or sounds more confident.\n"
        "Return valid JSON only with this exact structure:\n"
        "{\n"
        '  "winner": "A" or "B" or "tie",\n'
        '  "reasoning": "1-3 sentence explanation",\n'
        '  "scores": {\n'
        '    "A": {\n'
        '      "instruction_following": 1-5,\n'
        '      "correctness": 1-5,\n'
        '      "clarity": 1-5,\n'
        '      "completeness": 1-5,\n'
        '      "hallucination_risk": 1-5\n'
        "    },\n"
        '    "B": {\n'
        '      "instruction_following": 1-5,\n'
        '      "correctness": 1-5,\n'
        '      "clarity": 1-5,\n'
        '      "completeness": 1-5,\n'
        '      "hallucination_risk": 1-5\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        "Output JSON only."
    )

    return [
        {
            "role": "system",
            "content": "You are a strict, fair evaluator. Return JSON only.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def build_json_judge_messages(
    instruction: str,
    input_text: str,
    response_a: str,
    response_b: str,
    reference_output: str,
) -> List[Dict[str, str]]:
    user_content = (
        "You are evaluating two structured JSON responses to the same task.\n\n"
        "Task:\n"
        "Decide which response is better overall, or declare a tie if they are comparably good.\n\n"
        "Evaluate each response on these dimensions using a 1-5 scale, where 1 is very poor and 5 is excellent:\n"
        "1. Instruction Following\n"
        "2. Correctness\n"
        "3. Clarity\n"
        "4. Completeness\n"
        "5. Structured Output Validity\n"
        "6. Hallucination / Fabrication Risk\n\n"
        "Important scoring notes:\n"
        "- structured_output_validity: 1 means very poor validity, 5 means excellent validity\n"
        "- hallucination_risk: 1 means very low risk, 5 means very high risk\n\n"
        "Instruction:\n"
        f"{instruction}\n\n"
    )

    if input_text.strip():
        user_content += f"Input:\n{input_text}\n\n"

    user_content += (
        f"Reference Output:\n{reference_output}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Judge fairly. Do not prefer a response because it appears first, is longer, or looks more detailed unless it is actually better.\n"
        "Return valid JSON only with this exact structure:\n"
        "{\n"
        '  "winner": "A" or "B" or "tie",\n'
        '  "reasoning": "1-3 sentence explanation",\n'
        '  "scores": {\n'
        '    "A": {\n'
        '      "instruction_following": 1-5,\n'
        '      "correctness": 1-5,\n'
        '      "clarity": 1-5,\n'
        '      "completeness": 1-5,\n'
        '      "structured_output_validity": 1-5,\n'
        '      "hallucination_risk": 1-5\n'
        "    },\n"
        '    "B": {\n'
        '      "instruction_following": 1-5,\n'
        '      "correctness": 1-5,\n'
        '      "clarity": 1-5,\n'
        '      "completeness": 1-5,\n'
        '      "structured_output_validity": 1-5,\n'
        '      "hallucination_risk": 1-5\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        "Output JSON only."
    )

    return [
        {
            "role": "system",
            "content": "You are a strict, fair evaluator of structured outputs. Return JSON only.",
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

