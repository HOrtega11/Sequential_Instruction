
import json
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_example(example):
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()

    if not instruction or not output:
        return None

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }


def main():
    raw_path = "data/alpaca/raw_alpaca.json"
    train_path = "data/alpaca/alpaca_train.json"
    eval_path = "data/eval/alpaca_eval.json"

    data = load_json(raw_path)
    normalized = []

    for ex in data:
        item = normalize_example(ex)
        if item is not None:
            normalized.append(item)

    eval_size = 100
    eval_data = normalized[:eval_size]
    train_data = normalized[eval_size:]

    save_json(train_data, train_path)
    save_json(eval_data, eval_path)

    print(f"Saved {len(train_data)} training examples")
    print(f"Saved {len(eval_data)} eval examples")


if __name__ == "__main__":
    main()


