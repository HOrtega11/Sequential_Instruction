
import json
from pathlib import Path
from typing import Any, Dict, List
from src.utils.client import get_client
from src.utils.config import load_config
from src.utils.client import get_client


OUTPUT_PATH = "data/json_instruct/json_train.json"
EVAL_PATH = "data/eval/json_eval.json"


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def build_prompt_bank() -> List[Dict[str, str]]:
    prompts = []

    # 1. JSON extraction
    prompts.append({
        "task_type": "json_extraction",
        "instruction": "Extract the person name, date, city, and event from the text and return a valid JSON object.",
        "input": "Maria Lopez attended the cardiology conference in Houston on March 12, 2025."
    })

    # 2. Schema-constrained generation
    prompts.append({
        "task_type": "schema_generation",
        "instruction": (
            "Create a valid JSON object with keys: "
            "product_name (string), price (number), in_stock (boolean), tags (array of strings)."
        ),
        "input": "Generate an example product record for a wireless mouse."
    })

    # 3. Exact-label classification
    prompts.append({
        "task_type": "json_classification",
        "instruction": (
            "Classify the sentiment as exactly one of: positive, negative, neutral. "
            "Return valid JSON with keys: label, rationale."
        ),
        "input": "The service was fast and the staff was helpful."
    })

    # 4. JSON repair
    prompts.append({
        "task_type": "json_repair",
        "instruction": "Repair the malformed JSON and return only valid JSON.",
        "input": '{"name": "John", "age": 31, "skills": ["python", "sql",}'
    })

    # 5. Tool-call argument generation
    prompts.append({
        "task_type": "tool_call_arguments",
        "instruction": (
            "Generate valid JSON arguments for a function call "
            "book_flight(origin, destination, date, passengers)."
        ),
        "input": "Book a flight from San Antonio to Chicago on June 10, 2026 for 2 passengers."
    })

    return prompts


def build_messages(example: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a precise assistant. Return only valid JSON when requested."
        },
        {
            "role": "user",
            "content": f"Instruction: {example['instruction']}\nInput: {example['input']}"
        },
    ]


def generate_teacher_output(client: OpenAI, model_name: str, example: Dict[str, str]) -> str:
    
    response = client.chat.completions.create(
        model=model_name,
        messages=build_messages(example),
        temperature=temperature,
        max_tokens=max_tokens,)

    return response.choices[0].message.content.strip()


def main() -> None:
    
    config = load_config()
    client = get_client()
    teacher_model = config["models"]["teacher"]

    temperature = config["generation"]["temperature"]
    max_tokens = config["generation"]["max_tokens"]

    prompt_bank = build_prompt_bank()
    accepted = []

    for example in prompt_bank:
        output = generate_teacher_output(client, teacher_model, example)

        if not is_valid_json(output):
            print(f"Skipping invalid JSON for task {example['task_type']}")
            continue

        accepted.append({
            "instruction": example["instruction"],
            "input": example["input"],
            "output": output,
            "task_type": example["task_type"],
        })

    # For now, tiny split just to test pipeline
    eval_size = min(2, len(accepted))
    eval_data = accepted[:eval_size]
    train_data = accepted[eval_size:]

    save_json(train_data, OUTPUT_PATH)
    save_json(eval_data, EVAL_PATH)

    print(f"Saved {len(train_data)} JSON training examples")
    print(f"Saved {len(eval_data)} JSON eval examples")


if __name__ == "__main__":
    main()

