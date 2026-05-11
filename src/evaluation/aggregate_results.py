

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import get_config


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def maybe_load(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    return load_json(path)


def extract_alpaca_metrics(summary_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not summary_json:
        return {}

    metrics = summary_json.get("automatic_metrics", {})
    return {
        "rouge1": metrics.get("rouge1"),
        "rouge2": metrics.get("rouge2"),
        "rougeL": metrics.get("rougeL"),
        "rougeLsum": metrics.get("rougeLsum"),
        "bertscore_precision": metrics.get("bertscore_precision"),
        "bertscore_recall": metrics.get("bertscore_recall"),
        "bertscore_f1": metrics.get("bertscore_f1"),
        "task_completion_rate": metrics.get("task_completion_rate"),
        "avg_output_tokens": metrics.get("avg_output_tokens"),
        "min_output_tokens": metrics.get("min_output_tokens"),
        "max_output_tokens": metrics.get("max_output_tokens"),
    }


def extract_json_metrics(summary_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not summary_json:
        return {}

    return {
        "json_validity_rate": summary_json.get("json_validity_rate"),
        "schema_compliance_rate": summary_json.get("schema_compliance_rate"),
        "exact_match_rate": summary_json.get("exact_match_rate"),
        "field_precision": summary_json.get("field_precision"),
        "field_recall": summary_json.get("field_recall"),
        "field_f1": summary_json.get("field_f1"),
        "common_error_taxonomy": summary_json.get("common_error_taxonomy", {}),
        "per_task": summary_json.get("per_task", {}),
    }


def extract_judge_summary(judge_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not judge_json:
        return {}
    return judge_json.get("summary", {})


def safe_subtract(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate final experiment results.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional explicit aggregate output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    outputs_cfg = config.get("outputs", {})

    out_dir = args.output_dir if args.output_dir else outputs_cfg.get("aggregate_dir", "outputs/aggregate")
    judge_dir = outputs_cfg.get("judge_dir", "outputs/judge")

    alpaca_ckpt0 = maybe_load("outputs/eval_alpaca/checkpoint0/summary.json")
    alpaca_ckpt1 = maybe_load("outputs/eval_alpaca/checkpoint1/summary.json")
    alpaca_ckpt2 = maybe_load("outputs/eval_alpaca/checkpoint2/summary.json")

    json_ckpt0 = maybe_load("outputs/eval_json/checkpoint0/summary.json")
    json_ckpt1 = maybe_load("outputs/eval_json/checkpoint1/summary.json")
    json_ckpt2 = maybe_load("outputs/eval_json/checkpoint2/summary.json")

    judge_alpaca_0v1 = maybe_load(f"{judge_dir}/alpaca_checkpoint0_vs_checkpoint1.json")
    judge_alpaca_1v2 = maybe_load(f"{judge_dir}/alpaca_checkpoint1_vs_checkpoint2.json")
    judge_json_0v1 = maybe_load(f"{judge_dir}/json_checkpoint0_vs_checkpoint1.json")
    judge_json_1v2 = maybe_load(f"{judge_dir}/json_checkpoint1_vs_checkpoint2.json")
    judge_json_0v2 = maybe_load(f"{judge_dir}/json_checkpoint0_vs_checkpoint2.json")

    detailed_forgetting = maybe_load(f"{out_dir}/forgetting_analysis_detailed.json")

    alpaca_auto = {
        "checkpoint0": extract_alpaca_metrics(alpaca_ckpt0),
        "checkpoint1": extract_alpaca_metrics(alpaca_ckpt1),
        "checkpoint2": extract_alpaca_metrics(alpaca_ckpt2),
    }

    json_auto = {
        "checkpoint0": extract_json_metrics(json_ckpt0),
        "checkpoint1": extract_json_metrics(json_ckpt1),
        "checkpoint2": extract_json_metrics(json_ckpt2),
    }

    judge_results = {
        "alpaca_checkpoint0_vs_checkpoint1": extract_judge_summary(judge_alpaca_0v1),
        "alpaca_checkpoint1_vs_checkpoint2": extract_judge_summary(judge_alpaca_1v2),
        "json_checkpoint0_vs_checkpoint1": extract_judge_summary(judge_json_0v1),
        "json_checkpoint1_vs_checkpoint2": extract_judge_summary(judge_json_1v2),
        "json_checkpoint0_vs_checkpoint2": extract_judge_summary(judge_json_0v2),
    }

    comparison_table = {
        "checkpoint0": {
            "alpaca_pairwise_win_rate_vs_previous_checkpoint": None,
            "alpaca_rougeL": alpaca_auto["checkpoint0"].get("rougeL"),
            "alpaca_bertscore_f1": alpaca_auto["checkpoint0"].get("bertscore_f1"),
            "json_validity_rate": json_auto["checkpoint0"].get("json_validity_rate"),
            "json_schema_compliance_rate": json_auto["checkpoint0"].get("schema_compliance_rate"),
            "json_exact_match_rate": json_auto["checkpoint0"].get("exact_match_rate"),
        },
        "checkpoint1": {
            "alpaca_pairwise_win_rate_vs_previous_checkpoint": judge_results.get(
                "alpaca_checkpoint0_vs_checkpoint1", {}
            ).get("right_win_rate"),
            "alpaca_rougeL": alpaca_auto["checkpoint1"].get("rougeL"),
            "alpaca_bertscore_f1": alpaca_auto["checkpoint1"].get("bertscore_f1"),
            "json_validity_rate": json_auto["checkpoint1"].get("json_validity_rate"),
            "json_schema_compliance_rate": json_auto["checkpoint1"].get("schema_compliance_rate"),
            "json_exact_match_rate": json_auto["checkpoint1"].get("exact_match_rate"),
        },
        "checkpoint2": {
            "alpaca_pairwise_win_rate_vs_previous_checkpoint": judge_results.get(
                "alpaca_checkpoint1_vs_checkpoint2", {}
            ).get("right_win_rate"),
            "alpaca_rougeL": alpaca_auto["checkpoint2"].get("rougeL"),
            "alpaca_bertscore_f1": alpaca_auto["checkpoint2"].get("bertscore_f1"),
            "json_validity_rate": json_auto["checkpoint2"].get("json_validity_rate"),
            "json_schema_compliance_rate": json_auto["checkpoint2"].get("schema_compliance_rate"),
            "json_exact_match_rate": json_auto["checkpoint2"].get("exact_match_rate"),
        },
        "notes": {
            "alpaca_pairwise_win_rate_vs_previous_checkpoint": (
                "These judge win rates are pairwise and opponent-dependent. "
                "Checkpoint1 is judged against checkpoint0, and checkpoint2 is judged against checkpoint1."
            )
        },
    }

    basic_forgetting = {
        "alpaca_judge_checkpoint1_vs_checkpoint2": judge_results.get("alpaca_checkpoint1_vs_checkpoint2", {}),
        "absolute_change_checkpoint2_minus_checkpoint1": {
            "alpaca_pairwise_judge_win_rate_delta": safe_subtract(
                judge_results.get("alpaca_checkpoint1_vs_checkpoint2", {}).get("right_win_rate"),
                judge_results.get("alpaca_checkpoint1_vs_checkpoint2", {}).get("left_win_rate"),
            ),
            "rougeL": safe_subtract(
                alpaca_auto["checkpoint2"].get("rougeL"),
                alpaca_auto["checkpoint1"].get("rougeL"),
            ),
            "bertscore_f1": safe_subtract(
                alpaca_auto["checkpoint2"].get("bertscore_f1"),
                alpaca_auto["checkpoint1"].get("bertscore_f1"),
            ),
            "task_completion_rate": safe_subtract(
                alpaca_auto["checkpoint2"].get("task_completion_rate"),
                alpaca_auto["checkpoint1"].get("task_completion_rate"),
            ),
            "avg_output_tokens": safe_subtract(
                alpaca_auto["checkpoint2"].get("avg_output_tokens"),
                alpaca_auto["checkpoint1"].get("avg_output_tokens"),
            ),
        },
        "interpretation_hint": (
            "Negative deltas on Alpaca metrics may indicate forgetting after Stage 2. "
            "The most important judge comparison is alpaca_checkpoint1_vs_checkpoint2."
        ),
    }

    full_payload = {
        "alpaca_automatic_metrics": alpaca_auto,
        "json_automatic_metrics": json_auto,
        "judge_pairwise_results": judge_results,
        "comparison_table": comparison_table,
        "forgetting_analysis_basic": basic_forgetting,
        "forgetting_analysis_detailed": detailed_forgetting if detailed_forgetting else {},
    }

    save_json(full_payload, f"{out_dir}/full_results.json")
    save_json(comparison_table, f"{out_dir}/comparison_table.json")
    save_json(basic_forgetting, f"{out_dir}/forgetting_analysis.json")

    if detailed_forgetting:
        save_json(detailed_forgetting, f"{out_dir}/forgetting_analysis_detailed_copy.json")

    print("Saved:")
    print(f"  {out_dir}/full_results.json")
    print(f"  {out_dir}/comparison_table.json")
    print(f"  {out_dir}/forgetting_analysis.json")
    if detailed_forgetting:
        print(f"  {out_dir}/forgetting_analysis_detailed_copy.json")

    print("\n===== Comparison Table =====")
    print(json.dumps(comparison_table, indent=2, ensure_ascii=False))

    print("\n===== Basic Forgetting Analysis =====")
    print(json.dumps(basic_forgetting, indent=2, ensure_ascii=False))

    if detailed_forgetting:
        print("\n===== Detailed Forgetting Analysis Loaded =====")
        print(
            json.dumps(
                detailed_forgetting.get("summary", detailed_forgetting),
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

