
from typing import Dict, List


def build_teacher_messages(example: Dict[str, str]) -> List[Dict[str, str]]:
    task_type = example.get("task_type", "json_extraction")

    schema_instructions = {
        "json_extraction": (
            'Return JSON with keys: "person_name", "date", "city", "event".'
        ),
        "schema_generation": (
            'Return JSON with keys: "product_name", "price", "in_stock", "tags".'
        ),
        "json_classification": (
            'Return JSON with keys: "label", "rationale". '
            'Label must be one of: positive, negative, neutral.'
        ),
        "json_repair": (
            "Fix the input JSON and return a valid corrected JSON object."
        ),
        "tool_call_arguments": (
            'Return JSON with keys: "origin", "destination", "date", "passengers".'
        ),
    }

    schema_text = schema_instructions.get(task_type, "Return a valid JSON object.")

    return [
        {
            "role": "system",
            "content": (
                "You are a precise data-generation assistant.\n\n"
                "Rules:\n"
                "- Output ONLY valid JSON\n"
                "- No markdown, no code fences\n"
                "- No explanations\n"
                "- Do not include extra keys\n"
                "- Ensure correct types for all fields\n"
                "- Follow the required schema exactly\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task Type: {task_type}\n\n"
                f"{schema_text}\n\n"
                f"Instruction:\n{example['instruction']}\n\n"
                f"Input:\n{example.get('input', '')}\n\n"
                "Return JSON only."
            ),
        },
    ]

