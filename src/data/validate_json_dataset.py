

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def try_parse_json(text: str) -> Tuple[bool, Any, str]:
    try:
        return True, json.loads(text), ""
    except Exception as e:
        return False, None, str(e)


def validate_required_fields(example: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors = []

    for field in ["instruction", "input", "output", "task_type"]:
        if field not in example:
            errors.append(f"missing_field:{field}")

    if "instruction" in example and not str(example["instruction"]).strip():
        errors.append("empty_instruction")

    if "output" in example and not str(example["output"]).strip():
        errors.append("empty_output")

    if "task_type" in example and not str(example["task_type"]).strip():
        errors.append("empty_task_type")

    return len(errors) == 0, errors


def validate_schema(task_type: str, obj: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if task_type == "json_repair":
        if isinstance(obj, (dict, list)):
            return True, []
        return False, ["json_repair_top_level_must_be_object_or_array"]

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

    for key in req_keys - obj_keys:
        errors.append(f"missing_key:{key}")
    for key in obj_keys - req_keys:
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


def validate_example(example: Dict[str, Any], idx: int) -> Tuple[bool, Dict[str, Any]]:
    record = {
        "index": idx,
        "id": example.get("id", f"example_{idx}"),
        "task_type": example.get("task_type"),
        "valid_required_fields": False,
        "valid_json": False,
        "schema_valid": False,
        "errors": [],
    }

    fields_ok, field_errors = validate_required_fields(example)
    record["valid_required_fields"] = fields_ok
    record["errors"].extend(field_errors)

    if not fields_ok:
        return False, record

    output_text = str(example.get("output", "")).strip()
    parsed_ok, parsed_obj, parse_error = try_parse_json(output_text)

    record["valid_json"] = parsed_ok
    if not parsed_ok:
        record["errors"].append(f"invalid_json:{parse_error}")
        return False, record

    schema_ok, schema_errors = validate_schema(str(example["task_type"]).strip(), parsed_obj)
    record["schema_valid"] = schema_ok
    record["errors"].extend(schema_errors)

    is_valid = fields_ok and parsed_ok and schema_ok
    return is_valid, record


def summarize_validation(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    valid_count = sum(1 for r in records if not r["errors"])
    invalid_count = total - valid_count

    by_task: Dict[str, Dict[str, int]] = {}
    error_counts: Dict[str, int] = {}

    for r in records:
        task = str(r.get("task_type"))
        if task not in by_task:
            by_task[task] = {"total": 0, "valid": 0, "invalid": 0}

        by_task[task]["total"] += 1
        if r["errors"]:
            by_task[task]["invalid"] += 1
        else:
            by_task[task]["valid"] += 1

        for err in r["errors"]:
            error_counts[err] = error_counts.get(err, 0) + 1

    return {
        "total_examples": total,
        "valid_examples": valid_count,
        "invalid_examples": invalid_count,
        "valid_rate": valid_count / total if total else 0.0,
        "invalid_rate": invalid_count / total if total else 0.0,
        "per_task": by_task,
        "error_counts": error_counts,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate teacher-generated JSON dataset.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to JSON dataset file to validate.",
    )
    parser.add_argument(
        "--valid_output_path",
        type=str,
        default=None,
        help="Optional path to save only valid examples.",
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help="Optional path to save validation report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_json(args.input_path)

    validation_records = []
    valid_examples = []

    for idx, example in enumerate(dataset, start=1):
        is_valid, record = validate_example(example, idx)
        validation_records.append(record)
        if is_valid:
            valid_examples.append(example)

    summary = summarize_validation(validation_records)

    report = {
        "input_path": args.input_path,
        "summary": summary,
        "validation_records": validation_records,
    }

    print("===== JSON Dataset Validation =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.valid_output_path:
        save_json(valid_examples, args.valid_output_path)
        print(f"Saved valid examples to: {args.valid_output_path}")

    if args.report_path:
        save_json(report, args.report_path)
        print(f"Saved validation report to: {args.report_path}")


if __name__ == "__main__":
    main()
    
