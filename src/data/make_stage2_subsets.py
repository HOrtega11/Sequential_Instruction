

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from src.utils.config import get_config



def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def stratified_sample(data: List[Dict], fraction: float) -> List[Dict]:
    """
    Sample dataset while preserving task_type distribution.
    """
    buckets = defaultdict(list)

    for ex in data:
        task = ex.get("task_type", "unknown")
        buckets[task].append(ex)

    sampled = []

    for task, examples in buckets.items():
        k = max(1, int(len(examples) * fraction))
        sampled.extend(examples[:k])

    return sampled


def compute_distribution(data: List[Dict]) -> Dict[str, int]:
    counts = defaultdict(int)
    for ex in data:
        counts[ex.get("task_type", "unknown")] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="Create Stage 2 ablation subsets.")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Path to full JSON training dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/json_instruct",
        help="Directory to save subsets.",
    )
    args = parser.parse_args()

    config = get_config()
    SEED = config.get("training", {}).get("seed", 42)
    input_path = args.input_path if args.input_path else config["paths"]["json_train"]

    data = load_json(input_path)

    random.seed(SEED)
    shuffled = data[:]
    random.shuffle(shuffled)

    subset_100 = shuffled
    subset_50 = stratified_sample(shuffled, 0.50)
    subset_25 = stratified_sample(shuffled, 0.25)

    save_json(subset_100, f"{args.output_dir}/json_train_100.json")
    save_json(subset_50, f"{args.output_dir}/json_train_50.json")
    save_json(subset_25, f"{args.output_dir}/json_train_25.json")

    metadata = {
        "seed": SEED,
        "total_examples": len(shuffled),
        "subset_sizes": {
            "100": len(subset_100),
            "50": len(subset_50),
            "25": len(subset_25),
        },
        "distributions": {
            "100": compute_distribution(subset_100),
            "50": compute_distribution(subset_50),
            "25": compute_distribution(subset_25),
        },
    }

    save_json(metadata, f"{args.output_dir}/subset_metadata.json")

    print("===== Stage 2 Subsets Created =====")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

