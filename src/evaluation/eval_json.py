

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from prompts.inference_prompts import build_instruction_prompt
from src.inference.run_inference import load_model_and_tokenizer
from src.utils.config import get_config


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def try_parse_json(text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        obj = json.loads(text)
        return True, obj, None
    except Exception as e:
        return False, None, str(e)


def canonicalize(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def detect_format_error(raw_text: str, parse_error: Optional[str]) -> str:
    if raw_text is None or not str(raw_text).strip():
        return "empty_output"

    text = str(raw_text).strip()

    if text.startswith("```") or "```json" in text.lower():
        return "markdown_fence"

    if parse_error is None:
        return "unknown"

    lower_err = parse_error.lower()
    lower_text = text.lower()

    if "expecting ',' delimiter" in lower_err:
        return "missing_comma"
    if "unterminated" in lower_err:
        return "truncated_output"
    if text.count("{") != text.count("}") or text.count("[") != text.count("]"):
        return "missing_bracket_or_brace"
    if "extra data" in lower_err:
        return "extra_text_after_json"
    if any(tok in lower_text for tok in ["tru", "flase", "fals", " yes", " no"]):
        return "invalid_boolean_or_literal"
    if "'" in text and '"' not in text:
        return "single_quotes_or_invalid_quotes"

    return "other_invalid_json"


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def compare_json_exact(pred_obj: Any, ref_obj: Any) -> bool:
    return canonicalize(pred_obj) == canonicalize(ref_obj)


def compare_field_level(pred_obj: Any, ref_obj: Any) -> Dict[str, Any]:
    """
    Field-level scoring for top-level dict keys.
    Precision/recall/F1 are computed over correct key-value pairs.
    """
    if not isinstance(pred_obj, dict) or not isinstance(ref_obj, dict):
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "correct_fields": [],
            "incorrect_or_extra_fields": [],
            "missing_fields": [],
        }

    tp = 0
    fp = 0
    fn = 0
    correct_fields = []
    incorrect_or_extra_fields = []
    missing_fields = []

    pred_keys = set(pred_obj.keys())
    ref_keys = set(ref_obj.keys())

    for key in pred_keys & ref_keys:
        if normalize_scalar(pred_obj[key]) == normalize_scalar(ref_obj[key]):
            tp += 1
            correct_fields.append(key)
        else:
            fp += 1
            fn += 1
            incorrect_or_extra_fields.append(key)

    for key in pred_keys - ref_keys:
        fp += 1
        incorrect_or_extra_fields.append(key)

    for key in ref_keys - pred_keys:
        fn += 1
        missing_fields.append(key)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_fields": correct_fields,
        "incorrect_or_extra_fields": incorrect_or_extra_fields,
        "missing_fields": missing_fields,
    }


def validate_schema(task_type: str, obj: Any) -> Tuple[bool, List[str]]:
    errors = []

    if not isinstance(obj, dict):
        return False, ["top_level_not_object"]

    if task_type == "json_extraction":
        required = {
            "person_name": str,
            "date": str,
            "city": str,
            "event": str,
        }
    elif task_type == "schema_generation":
        required = {
            "product_name": str,
            "price": (int, float),
            "in_stock": bool,
            "tags": list,
        }
    elif task_type == "json_classification":
        required = {
            "label": str,
            "rationale": str,
        }
    elif task_type == "json_repair":
        # Handled separately in evaluate_example using the reference JSON structure.
        return False, ["json_repair_requires_reference_based_validation"]
    elif task_type == "tool_call_arguments":
        required = {
            "origin": str,
            "destination": str,
            "date": str,
            "passengers": int,
        }
    else:
        return False, [f"unknown_task_type:{task_type}"]

    obj_keys = set(obj.keys())
    req_keys = set(required.keys())

    missing = req_keys - obj_keys
    extra = obj_keys - req_keys

    for key in missing:
        errors.append(f"missing_key:{key}")
    for key in extra:
        errors.append(f"extra_key:{key}")

    for key, expected_type in required.items():
        if key not in obj:
            continue
        if not isinstance(obj[key], expected_type):
            errors.append(f"wrong_type:{key}")

    if task_type == "json_classification" and "label" in obj:
        if obj["label"] not in {"positive", "negative", "neutral"}:
            errors.append("invalid_label_value")

    if task_type == "schema_generation" and "tags" in obj:
        if not isinstance(obj["tags"], list) or not all(isinstance(x, str) for x in obj["tags"]):
            errors.append("wrong_type:tags_elements")

    if task_type == "tool_call_arguments" and "passengers" in obj:
        if not isinstance(obj["passengers"], int):
            errors.append("wrong_type:passengers")
        elif obj["passengers"] < 1:
            errors.append("invalid_value:passengers")

    return len(errors) == 0, errors


def validate_json_repair_schema(pred_obj: Any, ref_obj: Any) -> Tuple[bool, List[str]]:
    """
    For json_repair tasks, treat schema compliance as:
    - same top-level type as the reference
    - if dict: same key set
    - if list: same top-level list type is enough for schema compliance
    """
    errors = []

    if type(pred_obj) is not type(ref_obj):
        return False, ["wrong_top_level_type"]

    if isinstance(ref_obj, dict):
        pred_keys = set(pred_obj.keys())
        ref_keys = set(ref_obj.keys())

        missing = ref_keys - pred_keys
        extra = pred_keys - ref_keys

        for key in missing:
            errors.append(f"missing_key:{key}")
        for key in extra:
            errors.append(f"extra_key:{key}")

    return len(errors) == 0, errors


def evaluate_example(pred_text: str, reference_text: str, task_type: str) -> Dict[str, Any]:
    pred_valid, pred_obj, pred_err = try_parse_json(pred_text)
    ref_valid, ref_obj, ref_err = try_parse_json(reference_text)

    if not ref_valid:
        return {
            "json_valid": 0,
            "schema_compliant": 0,
            "exact_match": 0,
            "field_precision": 0.0,
            "field_recall": 0.0,
            "field_f1": 0.0,
            "format_error_type": "invalid_reference_json",
            "schema_errors": [f"reference_parse_error:{ref_err}"],
        }

    if not pred_valid:
        return {
            "json_valid": 0,
            "schema_compliant": 0,
            "exact_match": 0,
            "field_precision": 0.0,
            "field_recall": 0.0,
            "field_f1": 0.0,
            "format_error_type": detect_format_error(pred_text, pred_err),
            "schema_errors": [],
        }

    if task_type == "json_repair":
        schema_ok, schema_errors = validate_json_repair_schema(pred_obj, ref_obj)
    else:
        schema_ok, schema_errors = validate_schema(task_type, pred_obj)

    exact_match = int(compare_json_exact(pred_obj, ref_obj))
    field_scores = compare_field_level(pred_obj, ref_obj)

    return {
        "json_valid": 1,
        "schema_compliant": int(schema_ok),
        "exact_match": exact_match,
        "field_precision": field_scores["precision"],
        "field_recall": field_scores["recall"],
        "field_f1": field_scores["f1"],
        "format_error_type": None,
        "schema_errors": schema_errors,
        "field_details": field_scores,
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    error_counter = Counter()
    task_buckets = defaultdict(list)

    for row in results:
        if row.get("format_error_type"):
            error_counter[row["format_error_type"]] += 1
        task_buckets[row["task_type"]].append(row)

    def avg(rows: List[Dict[str, Any]], key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in rows) / len(rows) if rows else 0.0

    summary = {
        "num_examples": total,
        "json_validity_rate": avg(results, "json_valid"),
        "schema_compliance_rate": avg(results, "schema_compliant"),
        "exact_match_rate": avg(results, "exact_match"),
        "field_precision": avg(results, "field_precision"),
        "field_recall": avg(results, "field_recall"),
        "field_f1": avg(results, "field_f1"),
        "common_error_taxonomy": dict(error_counter),
        "per_task": {},
    }

    for task_type, rows in task_buckets.items():
        summary["per_task"][task_type] = {
            "num_examples": len(rows),
            "json_validity_rate": avg(rows, "json_valid"),
            "schema_compliance_rate": avg(rows, "schema_compliant"),
            "exact_match_rate": avg(rows, "exact_match"),
            "field_precision": avg(rows, "field_precision"),
            "field_recall": avg(rows, "field_recall"),
            "field_f1": avg(rows, "field_f1"),
        }

    return summary


def generate_response_with_loaded_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate structured JSON outputs.")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="checkpoint2",
        choices=["checkpoint0", "checkpoint1", "checkpoint2"],
        help="Which checkpoint to evaluate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional explicit model path. Overrides checkpoint_name if provided.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=None,
        help="Optional explicit JSON eval dataset path. Defaults to config['paths']['json_eval'].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional explicit output dir.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Defaults to config['evaluation']['json_max_new_tokens'] if present, else 512.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Defaults to config['evaluation']['temperature'] if present, else 0.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    eval_cfg = config.get("evaluation", {})
    checkpoint_name = args.checkpoint_name
    model_path = args.model_path if args.model_path else f"outputs/{checkpoint_name}"
    eval_path = args.eval_path if args.eval_path else config["paths"]["json_eval"]
    output_dir = args.output_dir if args.output_dir else f"outputs/eval_json/{checkpoint_name}"
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else eval_cfg.get("json_max_new_tokens", 512)
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else eval_cfg.get("temperature", 0.0)
    )

    predictions_path = f"{output_dir}/predictions.json"
    summary_path = f"{output_dir}/summary.json"

    examples = load_json(eval_path)

    print("===== JSON Evaluation =====")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Model path: {model_path}")
    print(f"Eval dataset: {eval_path}")
    print(f"Num examples: {len(examples)}")

    model, tokenizer = load_model_and_tokenizer(model_path)

    results = []

    for idx, example in enumerate(examples, start=1):
        prompt = build_instruction_prompt(
            instruction=example.get("instruction", ""),
            input_text=example.get("input", ""),
        )

        pred_text = generate_response_with_loaded_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        metrics = evaluate_example(
            pred_text=pred_text,
            reference_text=example["output"],
            task_type=example["task_type"],
        )

        row = {
            "id": example.get("id", idx),
            "task_type": example["task_type"],
            "instruction": example["instruction"],
            "input": example.get("input", ""),
            "reference_output": example["output"],
            "predicted_output": pred_text,
            **metrics,
        }
        results.append(row)

        if idx % 10 == 0 or idx == len(examples):
            print(f"Processed {idx}/{len(examples)} examples")

    summary = summarize_results(results)
    summary.update(
        {
            "checkpoint_name": checkpoint_name,
            "model_path": model_path,
            "eval_path": eval_path,
            "generation_settings": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        }
    )

    save_json(results, predictions_path)
    save_json(summary, summary_path)

    print("\n===== JSON Eval Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved detailed predictions to: {predictions_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

    
