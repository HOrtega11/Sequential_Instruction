
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate

from src.utils.config import get_config


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_category(example: Dict[str, Any]) -> str:
    """
    Prefer explicit category if present; otherwise infer from instruction text.
    """
    if example.get("category"):
        return str(example["category"]).strip().lower()

    instruction = str(example.get("instruction", "")).lower()

    if any(x in instruction for x in ["summarize", "summary", "summarise"]):
        return "summarization"
    if any(x in instruction for x in ["question", "answer", "who", "what", "when", "where", "why", "how"]):
        return "qa"
    if any(x in instruction for x in ["rewrite", "rephrase", "paraphrase", "edit"]):
        return "rewriting"
    if any(x in instruction for x in ["list", "brainstorm", "ideas", "generate ideas"]):
        return "brainstorming"
    if any(x in instruction for x in ["translate", "translation"]):
        return "translation"
    return "open_ended_generation"


def align_prediction_rows(
    left_rows: List[Dict[str, Any]],
    right_rows: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    left_map = {row.get("id"): row for row in left_rows}
    right_map = {row.get("id"): row for row in right_rows}
    shared_ids = sorted(set(left_map.keys()) & set(right_map.keys()))
    return [(left_map[i], right_map[i]) for i in shared_ids]


def compute_example_metrics_batch(
    predictions: List[str],
    references: List[str],
) -> Dict[str, List[float]]:
    """
    Compute per-example ROUGE-L and BERTScore F1.
    """
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    rouge_l_scores: List[float] = []
    for pred, ref in zip(predictions, references):
        score = rouge.compute(
            predictions=[pred],
            references=[ref],
            use_stemmer=True,
        )
        rouge_l_scores.append(float(score.get("rougeL", 0.0)))

    bert_scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
    )
    bert_f1_scores = [float(x) for x in bert_scores.get("f1", [])]

    return {
        "rougeL": rouge_l_scores,
        "bertscore_f1": bert_f1_scores,
    }


def build_aligned_scored_rows(
    aligned_rows: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    left_name: str,
    right_name: str,
) -> List[Dict[str, Any]]:
    left_preds: List[str] = []
    right_preds: List[str] = []
    refs: List[str] = []

    base_rows: List[Dict[str, Any]] = []

    for row_left, row_right in aligned_rows:
        pred_left = str(row_left.get("predicted_output", "")).strip()
        pred_right = str(row_right.get("predicted_output", "")).strip()
        ref = str(row_left.get("reference_output", "")).strip()

        left_preds.append(pred_left)
        right_preds.append(pred_right)
        refs.append(ref)

        base_rows.append(
            {
                "id": row_left.get("id"),
                "category": infer_category(row_left),
                "instruction": row_left.get("instruction", ""),
                "input": row_left.get("input", ""),
                "reference_output": ref,
                f"{left_name}_output": pred_left,
                f"{right_name}_output": pred_right,
                f"completed_{left_name}": int(bool(pred_left)),
                f"completed_{right_name}": int(bool(pred_right)),
            }
        )

    left_metrics = compute_example_metrics_batch(left_preds, refs)
    right_metrics = compute_example_metrics_batch(right_preds, refs)

    scored_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(base_rows):
        rouge_left = left_metrics["rougeL"][idx]
        rouge_right = right_metrics["rougeL"][idx]
        bert_left = left_metrics["bertscore_f1"][idx]
        bert_right = right_metrics["bertscore_f1"][idx]

        combined_left = 0.5 * rouge_left + 0.5 * bert_left
        combined_right = 0.5 * rouge_right + 0.5 * bert_right

        scored_rows.append(
            {
                **row,
                f"rougeL_{left_name}": rouge_left,
                f"rougeL_{right_name}": rouge_right,
                f"bertscore_f1_{left_name}": bert_left,
                f"bertscore_f1_{right_name}": bert_right,
                "delta_rougeL": rouge_right - rouge_left,
                "delta_bertscore_f1": bert_right - bert_left,
                f"combined_{left_name}": combined_left,
                f"combined_{right_name}": combined_right,
                "delta_combined": combined_right - combined_left,
            }
        )

    return scored_rows


def compute_category_breakdown(
    scored_rows: List[Dict[str, Any]],
    left_name: str,
    right_name: str,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in scored_rows:
        grouped[row["category"]].append(row)

    summary: Dict[str, Dict[str, Any]] = {}

    for category, rows in grouped.items():
        n = len(rows)
        summary[category] = {
            "num_examples": n,
            f"avg_rougeL_{left_name}": sum(r[f"rougeL_{left_name}"] for r in rows) / n,
            f"avg_rougeL_{right_name}": sum(r[f"rougeL_{right_name}"] for r in rows) / n,
            "delta_rougeL": sum(r["delta_rougeL"] for r in rows) / n,
            f"avg_bertscore_{left_name}": sum(r[f"bertscore_f1_{left_name}"] for r in rows) / n,
            f"avg_bertscore_{right_name}": sum(r[f"bertscore_f1_{right_name}"] for r in rows) / n,
            "delta_bertscore_f1": sum(r["delta_bertscore_f1"] for r in rows) / n,
            f"completion_rate_{left_name}": sum(r[f"completed_{left_name}"] for r in rows) / n,
            f"completion_rate_{right_name}": sum(r[f"completed_{right_name}"] for r in rows) / n,
            "delta_completion_rate": sum(
                r[f"completed_{right_name}"] - r[f"completed_{left_name}"] for r in rows
            ) / n,
        }

    return summary


def extract_representative_examples(
    scored_rows: List[Dict[str, Any]],
    left_name: str,
    right_name: str,
    top_k: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    regressions = sorted(scored_rows, key=lambda x: x["delta_combined"])[:top_k]
    improvements = sorted(scored_rows, key=lambda x: x["delta_combined"], reverse=True)[:top_k]
    held_steady = sorted(scored_rows, key=lambda x: abs(x["delta_combined"]))[:top_k]

    keep_fields = [
        "id",
        "category",
        "instruction",
        "input",
        "reference_output",
        f"{left_name}_output",
        f"{right_name}_output",
        f"rougeL_{left_name}",
        f"rougeL_{right_name}",
        f"bertscore_f1_{left_name}",
        f"bertscore_f1_{right_name}",
        "delta_rougeL",
        "delta_bertscore_f1",
        f"combined_{left_name}",
        f"combined_{right_name}",
        "delta_combined",
    ]

    def trim(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: row.get(k) for k in keep_fields} for row in rows]

    return {
        "regressions": trim(regressions),
        "improvements": trim(improvements),
        "held_steady": trim(held_steady),
    }


def find_pairwise_summary_in_adjacent_file(
    adjacent_payload: Dict[str, Any],
    left_model: str,
    right_model: str,
) -> Optional[Dict[str, Any]]:
    comparisons = adjacent_payload.get("comparisons", [])
    for item in comparisons:
        if item.get("left_model") == left_model and item.get("right_model") == right_model:
            return item.get("summary")
    return None


def get_pairwise_summary(
    direct_path: Optional[str],
    adjacent_path: Optional[str],
    left_model: str,
    right_model: str,
) -> Optional[Dict[str, Any]]:
    if direct_path:
        return safe_get(load_json(direct_path), "summary")

    if adjacent_path:
        payload = load_json(adjacent_path)
        return find_pairwise_summary_in_adjacent_file(payload, left_model, right_model)

    return None


def compute_judge_dimension_deltas(
    pairwise_summary: Optional[Dict[str, Any]],
    left_model: str,
    right_model: str,
) -> Dict[str, Any]:
    if not pairwise_summary:
        return {}

    avg_scores = pairwise_summary.get("average_dimension_scores", {})
    left_scores = avg_scores.get(left_model, {})
    right_scores = avg_scores.get(right_model, {})

    all_metrics = sorted(set(left_scores.keys()) | set(right_scores.keys()))
    out: Dict[str, Any] = {}

    for metric in all_metrics:
        left_val = left_scores.get(metric)
        right_val = right_scores.get(metric)

        if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
            out[metric] = {
                left_model: float(left_val),
                right_model: float(right_val),
                "delta_right_minus_left": float(right_val) - float(left_val),
            }

    return out


def build_interpretation(
    delta_judge_preference_gap: Optional[float],
    delta_rougeL: Optional[float],
    delta_bertscore: Optional[float],
    category_breakdown: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    severe_forgetting = False

    if delta_judge_preference_gap is not None and delta_judge_preference_gap < -0.10:
        severe_forgetting = True
    if delta_rougeL is not None and delta_rougeL < -0.05:
        severe_forgetting = True
    if delta_bertscore is not None and delta_bertscore < -0.03:
        severe_forgetting = True

    worst_categories = sorted(
        category_breakdown.items(),
        key=lambda kv: kv[1].get("delta_rougeL", 0.0),
    )[:3]

    best_categories = sorted(
        category_breakdown.items(),
        key=lambda kv: kv[1].get("delta_rougeL", 0.0),
        reverse=True,
    )[:3]

    if severe_forgetting:
        discussion = (
            "The results suggest meaningful forgetting after Stage 2. "
            "Possible causes include a Stage 2 learning rate that was too high, too many Stage 2 epochs, "
            "or insufficient diversity in the teacher-generated JSON dataset. "
            "A narrow Stage 2 task distribution can bias the model toward structured-output behavior while "
            "reducing general instruction-following robustness."
        )
    else:
        discussion = (
            "The results do not suggest severe catastrophic forgetting after Stage 2. "
            "Possible reasons include a conservative Stage 2 learning rate, limited Stage 2 epochs, "
            "and the fact that LoRA fine-tuning can preserve much of the model's prior instruction-following behavior. "
            "If Alpaca performance held steady, the Stage 2 setup likely achieved a reasonable retention-specialization tradeoff."
        )

    return {
        "severe_forgetting_detected": severe_forgetting,
        "worst_affected_categories": [
            {"category": category, **metrics} for category, metrics in worst_categories
        ],
        "best_retained_or_improved_categories": [
            {"category": category, **metrics} for category, metrics in best_categories
        ],
        "discussion": discussion,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run forgetting analysis comparing Alpaca results at checkpoint1 vs checkpoint2."
    )
    parser.add_argument(
        "--left_checkpoint_name",
        type=str,
        default="checkpoint1",
        help="Left checkpoint for the main forgetting comparison.",
    )
    parser.add_argument(
        "--right_checkpoint_name",
        type=str,
        default="checkpoint2",
        help="Right checkpoint for the main forgetting comparison.",
    )
    parser.add_argument(
        "--left_summary_path",
        type=str,
        default=None,
        help="Path to left checkpoint Alpaca summary.json",
    )
    parser.add_argument(
        "--right_summary_path",
        type=str,
        default=None,
        help="Path to right checkpoint Alpaca summary.json",
    )
    parser.add_argument(
        "--left_predictions_path",
        type=str,
        default=None,
        help="Path to left checkpoint Alpaca predictions.json",
    )
    parser.add_argument(
        "--right_predictions_path",
        type=str,
        default=None,
        help="Path to right checkpoint Alpaca predictions.json",
    )
    parser.add_argument(
        "--judge_pair_path",
        type=str,
        default=None,
        help="Path to direct pairwise Alpaca judge file for left vs right.",
    )
    parser.add_argument(
        "--judge_adjacent_summary_path",
        type=str,
        default=None,
        help="Optional aggregate judge summary from judge_evaluation.py --mode adjacent.",
    )
    parser.add_argument(
        "--judge_0v1_path",
        type=str,
        default=None,
        help="Optional path to checkpoint0 vs checkpoint1 judge file for context.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save the forgetting analysis JSON",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of representative examples to keep per section",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    outputs_cfg = config.get("outputs", {})

    left_name = args.left_checkpoint_name
    right_name = args.right_checkpoint_name

    left_summary_path = (
        args.left_summary_path
        if args.left_summary_path
        else f"outputs/eval_alpaca/{left_name}/summary.json"
    )
    right_summary_path = (
        args.right_summary_path
        if args.right_summary_path
        else f"outputs/eval_alpaca/{right_name}/summary.json"
    )
    left_predictions_path = (
        args.left_predictions_path
        if args.left_predictions_path
        else f"outputs/eval_alpaca/{left_name}/predictions.json"
    )
    right_predictions_path = (
        args.right_predictions_path
        if args.right_predictions_path
        else f"outputs/eval_alpaca/{right_name}/predictions.json"
    )

    default_direct_pair_path = (
        f"{outputs_cfg.get('judge_dir', 'outputs/judge')}/alpaca_{left_name}_vs_{right_name}.json"
    )
    judge_pair_path = args.judge_pair_path if args.judge_pair_path else default_direct_pair_path

    judge_adjacent_summary_path = args.judge_adjacent_summary_path
    judge_0v1_path = args.judge_0v1_path

    output_path = (
        args.output_path
        if args.output_path
        else f"{outputs_cfg.get('aggregate_dir', 'outputs/aggregate')}/forgetting_analysis_detailed.json"
    )

    left_summary = load_json(left_summary_path)
    right_summary = load_json(right_summary_path)
    left_predictions = load_json(left_predictions_path)
    right_predictions = load_json(right_predictions_path)

    pairwise_summary = get_pairwise_summary(
        direct_path=judge_pair_path if Path(judge_pair_path).exists() else None,
        adjacent_path=judge_adjacent_summary_path,
        left_model=left_name,
        right_model=right_name,
    )

    judge_0v1_summary = None
    if judge_0v1_path and Path(judge_0v1_path).exists():
        judge_0v1_summary = safe_get(load_json(judge_0v1_path), "summary")

    rougeL_left = safe_get(left_summary, "automatic_metrics", "rougeL")
    rougeL_right = safe_get(right_summary, "automatic_metrics", "rougeL")
    bert_left = safe_get(left_summary, "automatic_metrics", "bertscore_f1")
    bert_right = safe_get(right_summary, "automatic_metrics", "bertscore_f1")

    judge_left_win = safe_get(pairwise_summary or {}, "left_win_rate")
    judge_right_win = safe_get(pairwise_summary or {}, "right_win_rate")
    tie_rate = safe_get(pairwise_summary or {}, "tie_rate")
    invalid_rate = safe_get(pairwise_summary or {}, "invalid_rate")

    delta_judge_preference_gap = None
    if judge_left_win is not None and judge_right_win is not None:
        delta_judge_preference_gap = judge_right_win - judge_left_win

    delta_rougeL = None if rougeL_left is None or rougeL_right is None else rougeL_right - rougeL_left
    delta_bertscore = None if bert_left is None or bert_right is None else bert_right - bert_left

    aligned_rows = align_prediction_rows(left_predictions, right_predictions)
    if not aligned_rows:
        raise ValueError(
            f"No aligned prediction rows found between {left_name} and {right_name}. "
            f"Check that both prediction files preserve the same example IDs."
        )

    scored_rows = build_aligned_scored_rows(
        aligned_rows=aligned_rows,
        left_name=left_name,
        right_name=right_name,
    )

    category_breakdown = compute_category_breakdown(
        scored_rows=scored_rows,
        left_name=left_name,
        right_name=right_name,
    )

    representative_examples = extract_representative_examples(
        scored_rows=scored_rows,
        left_name=left_name,
        right_name=right_name,
        top_k=args.top_k,
    )

    judge_dimension_deltas = compute_judge_dimension_deltas(
        pairwise_summary=pairwise_summary,
        left_model=left_name,
        right_model=right_name,
    )

    interpretation = build_interpretation(
        delta_judge_preference_gap=delta_judge_preference_gap,
        delta_rougeL=delta_rougeL,
        delta_bertscore=delta_bertscore,
        category_breakdown=category_breakdown,
    )

    result = {
        "summary": {
            left_name: {
                "rougeL": rougeL_left,
                "bertscore_f1": bert_left,
            },
            right_name: {
                "rougeL": rougeL_right,
                "bertscore_f1": bert_right,
            },
            f"alpaca_judge_{left_name}_vs_{right_name}": {
                f"{left_name}_left_win_rate": judge_left_win,
                f"{right_name}_right_win_rate": judge_right_win,
                "tie_rate": tie_rate,
                "invalid_rate": invalid_rate,
                "note": (
                    "This judge metric is a pairwise head-to-head comparison. "
                    "The reported judge preference gap is computed as "
                    f"{right_name}_right_win_rate - {left_name}_left_win_rate."
                ),
            },
            "absolute_changes_right_minus_left": {
                "pairwise_judge_preference_gap": delta_judge_preference_gap,
                "rougeL": delta_rougeL,
                "bertscore_f1": delta_bertscore,
            },
        },
        "judge_dimension_deltas": judge_dimension_deltas,
        "per_category_breakdown": category_breakdown,
        "representative_examples": representative_examples,
        "interpretation": interpretation,
    }

    if judge_0v1_summary:
        result["context_stage1_vs_base"] = {
            "checkpoint0_vs_checkpoint1_summary": judge_0v1_summary
        }

    save_json(result, output_path)

    print("===== Forgetting Analysis =====")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    if judge_dimension_deltas:
        print("\n===== Judge Dimension Deltas =====")
        print(json.dumps(judge_dimension_deltas, indent=2, ensure_ascii=False))
    print("\n===== Interpretation =====")
    print(json.dumps(result["interpretation"], indent=2, ensure_ascii=False))
    print(f"\nSaved detailed forgetting analysis to: {output_path}")


if __name__ == "__main__":
    main()

