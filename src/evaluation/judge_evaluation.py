

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prompts.judge_prompts import (
    build_alpaca_judge_messages,
    build_json_judge_messages,
)
from src.utils.client import get_client
from src.utils.config import get_config





def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {
            "winner": "invalid",
            "reasoning": f"Judge output was not valid JSON. Raw output: {text}",
            "scores": {},
        }


def get_example_id(example: Dict[str, Any]) -> Any:
    return example.get("id")


def align_prediction_files(
    left_rows: List[Dict[str, Any]],
    right_rows: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    left_map = {get_example_id(r): r for r in left_rows}
    right_map = {get_example_id(r): r for r in right_rows}

    shared_ids = sorted(set(left_map.keys()) & set(right_map.keys()))
    return [(left_map[i], right_map[i]) for i in shared_ids]


def call_judge(
    client,
    judge_model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=judge_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content.strip()
    return safe_json_loads(text)


def numeric_average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_dimension_scores(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    bucket = defaultdict(lambda: defaultdict(list))

    for row in rows:
        parsed = row.get("judge_result", {})
        scores = parsed.get("scores", {})
        for side in ["left_model", "right_model"]:
            model_name = row[side]
            model_letter = row["display_mapping"].get(side)
            if not model_letter:
                continue
            model_scores = scores.get(model_letter, {})
            for metric_name, value in model_scores.items():
                if isinstance(value, (int, float)):
                    bucket[model_name][metric_name].append(float(value))

    out = {}
    for model_name, metric_map in bucket.items():
        out[model_name] = {
            metric_name: numeric_average(vals)
            for metric_name, vals in metric_map.items()
        }
    return out


def summarize_pairwise_results(
    rows: List[Dict[str, Any]],
    left_model: str,
    right_model: str,
) -> Dict[str, Any]:
    left_wins = 0
    right_wins = 0
    ties = 0
    invalid = 0

    for row in rows:
        winner = row.get("winner_resolved")
        if winner == left_model:
            left_wins += 1
        elif winner == right_model:
            right_wins += 1
        elif winner == "tie":
            ties += 1
        else:
            invalid += 1

    total = len(rows)
    return {
        "num_examples": total,
        "left_model": left_model,
        "right_model": right_model,
        "left_win_rate": left_wins / total if total else 0.0,
        "right_win_rate": right_wins / total if total else 0.0,
        "tie_rate": ties / total if total else 0.0,
        "invalid_rate": invalid / total if total else 0.0,
        "counts": {
            left_model: left_wins,
            right_model: right_wins,
            "tie": ties,
            "invalid": invalid,
        },
        "average_dimension_scores": aggregate_dimension_scores(rows),
    }


def resolve_winner(
    judge_winner: str,
    shown_left_model: str,
    shown_right_model: str,
) -> str:
    if judge_winner == "A":
        return shown_left_model
    if judge_winner == "B":
        return shown_right_model
    if judge_winner == "tie":
        return "tie"
    return "invalid"


def build_messages(
    task_type: str,
    left_row: Dict[str, Any],
    right_row: Dict[str, Any],
    response_a: str,
    response_b: str,
) -> List[Dict[str, str]]:
    instruction = left_row.get("instruction", "")
    input_text = left_row.get("input", "")

    if task_type == "alpaca":
        return build_alpaca_judge_messages(
            instruction=instruction,
            input_text=input_text,
            response_a=response_a,
            response_b=response_b,
        )

    if task_type == "json":
        return build_json_judge_messages(
            instruction=instruction,
            input_text=input_text,
            response_a=response_a,
            response_b=response_b,
            reference_output=left_row.get("reference_output", ""),
        )

    raise ValueError(f"Unsupported task_type: {task_type}")


def evaluate_pairwise_to_payload(
    task_type: str,
    left_predictions_path: str,
    right_predictions_path: str,
    left_model_name: str,
    right_model_name: str,
) -> Dict[str, Any]:
    config = get_config()
    RANDOM_SEED = config.get("training", {}).get("seed", 42)
    eval_cfg = config.get("evaluation", {})
    client = get_client()
    judge_model = config["models"]["judge"]
    judge_temperature = eval_cfg.get("judge_temperature", 0.0)
    judge_max_tokens = eval_cfg.get("judge_max_tokens", 512)

    random.seed(RANDOM_SEED)

    left_rows = load_json(left_predictions_path)
    right_rows = load_json(right_predictions_path)
    aligned = align_prediction_files(left_rows, right_rows)

    if not aligned:
        raise ValueError(
            f"No aligned examples found between:\n"
            f"  {left_predictions_path}\n"
            f"  {right_predictions_path}"
        )

    results = []

    print("===== Judge Pairwise Evaluation =====")
    print(f"Task type: {task_type}")
    print(f"Judge model: {judge_model}")
    print(f"Left predictions: {left_predictions_path}")
    print(f"Right predictions: {right_predictions_path}")
    print(f"Aligned examples: {len(aligned)}")

    for idx, (left_row, right_row) in enumerate(aligned, start=1):
        instruction = left_row.get("instruction", "")
        input_text = left_row.get("input", "")
        left_output = left_row.get("predicted_output", "")
        right_output = right_row.get("predicted_output", "")

        swap = random.random() < 0.5
        if swap:
            response_a = right_output
            response_b = left_output
            shown_left_model = right_model_name
            shown_right_model = left_model_name
            display_mapping = {
                "left_model": "B",
                "right_model": "A",
            }
        else:
            response_a = left_output
            response_b = right_output
            shown_left_model = left_model_name
            shown_right_model = right_model_name
            display_mapping = {
                "left_model": "A",
                "right_model": "B",
            }

        messages = build_messages(
            task_type=task_type,
            left_row=left_row,
            right_row=right_row,
            response_a=response_a,
            response_b=response_b,
        )

        judge_result = call_judge(
            client=client,
            judge_model=judge_model,
            messages=messages,
            temperature=judge_temperature,
            max_tokens=judge_max_tokens,
        )

        winner_resolved = resolve_winner(
            judge_winner=judge_result.get("winner", "invalid"),
            shown_left_model=shown_left_model,
            shown_right_model=shown_right_model,
        )

        results.append(
            {
                "id": left_row.get("id", idx),
                "task_type": task_type,
                "instruction": instruction,
                "input": input_text,
                "left_model": left_model_name,
                "right_model": right_model_name,
                "display_mapping": display_mapping,
                "left_output": left_output,
                "right_output": right_output,
                "judge_result": judge_result,
                "winner_resolved": winner_resolved,
            }
        )

        if idx % 10 == 0 or idx == len(aligned):
            print(f"Judged {idx}/{len(aligned)} examples")

    summary = summarize_pairwise_results(
        rows=results,
        left_model=left_model_name,
        right_model=right_model_name,
    )

    payload = {
        "task_type": task_type,
        "left_model": left_model_name,
        "right_model": right_model_name,
        "judge_model": judge_model,
        "judge_settings": {
            "temperature": judge_temperature,
            "max_tokens": judge_max_tokens,
        },
        "summary": summary,
        "results": results,
    }
    return payload


def evaluate_pairwise_and_save(
    task_type: str,
    left_predictions_path: str,
    right_predictions_path: str,
    left_model_name: str,
    right_model_name: str,
    output_path: str,
) -> Dict[str, Any]:
    payload = evaluate_pairwise_to_payload(
        task_type=task_type,
        left_predictions_path=left_predictions_path,
        right_predictions_path=right_predictions_path,
        left_model_name=left_model_name,
        right_model_name=right_model_name,
    )
    save_json(payload, output_path)
    print(f"Saved judge evaluation to: {output_path}")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return payload


def build_predictions_path(base_predictions_dir: str, checkpoint_name: str) -> str:
    return str(Path(base_predictions_dir) / checkpoint_name / "predictions.json")


def build_output_path(output_dir: str, task_type: str, left_model_name: str, right_model_name: str) -> str:
    return str(Path(output_dir) / f"{task_type}_{left_model_name}_vs_{right_model_name}.json")


def build_comparison_specs(
    comparison_mode: str,
    checkpoint_names: List[str],
    base_predictions_dir: str,
    output_dir: str,
    task_type: str,
) -> List[Dict[str, str]]:
    if len(checkpoint_names) < 2:
        raise ValueError("Need at least two checkpoint names for multi-comparison mode.")

    specs: List[Dict[str, str]] = []

    if comparison_mode == "adjacent":
        pairs = list(zip(checkpoint_names[:-1], checkpoint_names[1:]))
    elif comparison_mode == "all_pairs":
        pairs = []
        for i in range(len(checkpoint_names)):
            for j in range(i + 1, len(checkpoint_names)):
                pairs.append((checkpoint_names[i], checkpoint_names[j]))
    else:
        raise ValueError(f"Unsupported comparison_mode: {comparison_mode}")

    for left_name, right_name in pairs:
        specs.append(
            {
                "task_type": task_type,
                "left_predictions_path": build_predictions_path(base_predictions_dir, left_name),
                "right_predictions_path": build_predictions_path(base_predictions_dir, right_name),
                "left_model_name": left_name,
                "right_model_name": right_name,
                "output_path": build_output_path(output_dir, task_type, left_name, right_name),
            }
        )

    return specs


def parse_args():
    parser = argparse.ArgumentParser(description="Pairwise judge evaluation.")

    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["alpaca", "json"],
        help="Which task family to judge.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "adjacent", "all_pairs"],
        help=(
            "single = judge one pair only; "
            "adjacent = run checkpoint0 vs checkpoint1, then checkpoint1 vs checkpoint2; "
            "all_pairs = run every pair."
        ),
    )

    parser.add_argument(
        "--left_predictions_path",
        type=str,
        default=None,
        help="Path to first predictions.json file. Required in single mode.",
    )
    parser.add_argument(
        "--right_predictions_path",
        type=str,
        default=None,
        help="Path to second predictions.json file. Required in single mode.",
    )
    parser.add_argument(
        "--left_model_name",
        type=str,
        default=None,
        help="Label for the first model, e.g. checkpoint1. Required in single mode.",
    )
    parser.add_argument(
        "--right_model_name",
        type=str,
        default=None,
        help="Label for the second model, e.g. checkpoint2. Required in single mode.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the judge output JSON in single mode.",
    )

    parser.add_argument(
        "--checkpoint_names",
        nargs="+",
        default=["checkpoint0", "checkpoint1", "checkpoint2"],
        help="Checkpoint names used for adjacent or all_pairs modes.",
    )
    parser.add_argument(
        "--base_predictions_dir",
        type=str,
        default=None,
        help=(
            "Base directory containing per-checkpoint predictions subfolders. "
            "Defaults to outputs/eval_alpaca or outputs/eval_json depending on task_type."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/judge",
        help="Directory for saving outputs in adjacent or all_pairs modes.",
    )
    parser.add_argument(
        "--aggregate_output_path",
        type=str,
        default=None,
        help="Optional path to save a summary of all comparisons in adjacent or all_pairs modes.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "single":
        required = [
            args.left_predictions_path,
            args.right_predictions_path,
            args.left_model_name,
            args.right_model_name,
            args.output_path,
        ]
        if any(x is None for x in required):
            raise ValueError(
                "In single mode you must provide "
                "--left_predictions_path, --right_predictions_path, "
                "--left_model_name, --right_model_name, and --output_path."
            )

        evaluate_pairwise_and_save(
            task_type=args.task_type,
            left_predictions_path=args.left_predictions_path,
            right_predictions_path=args.right_predictions_path,
            left_model_name=args.left_model_name,
            right_model_name=args.right_model_name,
            output_path=args.output_path,
        )
        return

    if args.base_predictions_dir:
        base_predictions_dir = args.base_predictions_dir
    else:
        base_predictions_dir = (
            "outputs/eval_alpaca" if args.task_type == "alpaca" else "outputs/eval_json"
        )

    specs = build_comparison_specs(
        comparison_mode=args.mode,
        checkpoint_names=args.checkpoint_names,
        base_predictions_dir=base_predictions_dir,
        output_dir=args.output_dir,
        task_type=args.task_type,
    )

    all_summaries = []
    for spec in specs:
        payload = evaluate_pairwise_and_save(
            task_type=spec["task_type"],
            left_predictions_path=spec["left_predictions_path"],
            right_predictions_path=spec["right_predictions_path"],
            left_model_name=spec["left_model_name"],
            right_model_name=spec["right_model_name"],
            output_path=spec["output_path"],
        )
        all_summaries.append(
            {
                "task_type": spec["task_type"],
                "left_model": spec["left_model_name"],
                "right_model": spec["right_model_name"],
                "output_path": spec["output_path"],
                "summary": payload["summary"],
            }
        )

    if args.aggregate_output_path:
        aggregate_payload = {
            "task_type": args.task_type,
            "mode": args.mode,
            "checkpoint_names": args.checkpoint_names,
            "comparisons": all_summaries,
        }
        save_json(aggregate_payload, args.aggregate_output_path)
        print(f"Saved aggregate judge summary to: {args.aggregate_output_path}")


if __name__ == "__main__":
    main()


