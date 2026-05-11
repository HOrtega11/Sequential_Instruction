
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import get_config


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_example(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()

    if not instruction or not output:
        return None

    return {
        "id": f"alpaca_{idx}",
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def main() -> None:
    config = get_config()

    raw_path = config.get("paths", {}).get("alpaca_raw", "data/alpaca/raw_alpaca.json")
    train_path = config["paths"]["alpaca_train"]
    eval_path = config["paths"]["alpaca_eval"]

    data_cfg = config.get("data_generation", {})
    eval_size = data_cfg.get("alpaca_eval_size", 100)
    seed = config.get("training", {}).get("seed", 42)

    print("Loading raw Alpaca dataset...")
    data = load_json(raw_path)
    print(f"Total raw examples: {len(data)}")

    normalized = []

    for idx, ex in enumerate(data, start=1):
        item = normalize_example(ex, idx)
        if item is not None:
            normalized.append(item)

    print(f"Total valid normalized examples: {len(normalized)}")

    random.seed(seed)
    random.shuffle(normalized)

    if len(normalized) < eval_size:
        raise ValueError(
            f"Not enough normalized examples for eval split. "
            f"Need {eval_size}, found {len(normalized)}."
        )

    eval_data = normalized[:eval_size]
    train_data = normalized[eval_size:]

    save_json(train_data, train_path)
    save_json(eval_data, eval_path)

    metadata = {
        "raw_examples": len(data),
        "valid_normalized_examples": len(normalized),
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "seed": seed,
        "raw_path": raw_path,
        "train_path": train_path,
        "eval_path": eval_path,
    }
    save_json(metadata, "outputs/alpaca_prepare_metadata.json")

    print("\n===== Dataset Summary =====")
    print(f"Train size: {len(train_data)}")
    print(f"Eval size: {len(eval_data)}")
    print(f"Total used: {len(train_data) + len(eval_data)}")
    print("Saved metadata to: outputs/alpaca_prepare_metadata.json")


if __name__ == "__main__":
    main()


