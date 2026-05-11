

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import get_config


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_main_comparison_df(comparison_table: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for checkpoint, metrics in comparison_table.items():
        if checkpoint == "notes":
            continue
        row = {"checkpoint": checkpoint}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def build_basic_forgetting_df(forgetting_analysis: Dict[str, Any]) -> pd.DataFrame:
    delta_map = forgetting_analysis.get("absolute_change_checkpoint2_minus_checkpoint1", {})
    rows = [{"metric": k, "delta_checkpoint2_minus_checkpoint1": v} for k, v in delta_map.items()]
    return pd.DataFrame(rows)


def build_detailed_forgetting_df(detailed_forgetting: Dict[str, Any]) -> pd.DataFrame:
    summary = detailed_forgetting.get("summary", {})
    changes = summary.get("absolute_changes_right_minus_left", {})
    rows = [{"metric": k, "delta_checkpoint2_minus_checkpoint1": v} for k, v in changes.items()]
    return pd.DataFrame(rows)


def build_category_forgetting_df(detailed_forgetting: Dict[str, Any]) -> pd.DataFrame:
    category_map = detailed_forgetting.get("per_category_breakdown", {})
    rows = []

    for category, metrics in category_map.items():
        row = {"category": category}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def build_representative_examples_df(
    detailed_forgetting: Dict[str, Any],
    key: str,
) -> pd.DataFrame:
    rows = detailed_forgetting.get("representative_examples", {}).get(key, [])
    return pd.DataFrame(rows)


def plot_alpaca_metrics(full_results: Dict[str, Any], out_path: str) -> None:
    alpaca = full_results["alpaca_automatic_metrics"]

    checkpoints = ["checkpoint0", "checkpoint1", "checkpoint2"]
    rougeL = [alpaca.get(c, {}).get("rougeL") for c in checkpoints]
    bert_f1 = [alpaca.get(c, {}).get("bertscore_f1") for c in checkpoints]
    completion = [alpaca.get(c, {}).get("task_completion_rate") for c in checkpoints]

    plt.figure(figsize=(8, 5))
    plt.plot(checkpoints, rougeL, marker="o", label="ROUGE-L")
    plt.plot(checkpoints, bert_f1, marker="o", label="BERTScore F1")
    plt.plot(checkpoints, completion, marker="o", label="Task Completion Rate")
    plt.xlabel("Checkpoint")
    plt.ylabel("Score")
    plt.title("Alpaca Automatic Metrics Across Checkpoints")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_json_metrics(full_results: Dict[str, Any], out_path: str) -> None:
    json_metrics = full_results["json_automatic_metrics"]

    checkpoints = ["checkpoint0", "checkpoint1", "checkpoint2"]
    validity = [json_metrics.get(c, {}).get("json_validity_rate") for c in checkpoints]
    schema = [json_metrics.get(c, {}).get("schema_compliance_rate") for c in checkpoints]
    exact = [json_metrics.get(c, {}).get("exact_match_rate") for c in checkpoints]
    field_f1 = [json_metrics.get(c, {}).get("field_f1") for c in checkpoints]

    plt.figure(figsize=(8, 5))
    plt.plot(checkpoints, validity, marker="o", label="JSON Validity")
    plt.plot(checkpoints, schema, marker="o", label="Schema Compliance")
    plt.plot(checkpoints, exact, marker="o", label="Exact Match")
    plt.plot(checkpoints, field_f1, marker="o", label="Field F1")
    plt.xlabel("Checkpoint")
    plt.ylabel("Score")
    plt.title("JSON Automatic Metrics Across Checkpoints")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_judge_results(full_results: Dict[str, Any], out_path: str) -> None:
    judge = full_results["judge_pairwise_results"]

    labels = []
    right_win_rates = []

    for key in [
        "alpaca_checkpoint0_vs_checkpoint1",
        "alpaca_checkpoint1_vs_checkpoint2",
        "json_checkpoint0_vs_checkpoint1",
        "json_checkpoint1_vs_checkpoint2",
        "json_checkpoint0_vs_checkpoint2",
    ]:
        summary = judge.get(key, {})
        if summary:
            labels.append(key)
            right_win_rates.append(summary.get("right_win_rate"))

    plt.figure(figsize=(10, 5))
    plt.bar(labels, right_win_rates)
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Pairwise Judge Comparison")
    plt.ylabel("Right Model Win Rate")
    plt.title("Pairwise Judge Right-Model Win Rates")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_category_forgetting(detailed_forgetting: Dict[str, Any], out_path: str) -> None:
    category_map = detailed_forgetting.get("per_category_breakdown", {})
    if not category_map:
        return

    categories = list(category_map.keys())
    delta_rougeL = [category_map[c].get("delta_rougeL") for c in categories]
    delta_bertscore = [category_map[c].get("delta_bertscore_f1") for c in categories]

    plt.figure(figsize=(10, 5))
    plt.plot(categories, delta_rougeL, marker="o", label="Δ ROUGE-L")
    plt.plot(categories, delta_bertscore, marker="o", label="Δ BERTScore F1")
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Category")
    plt.ylabel("Checkpoint2 - Checkpoint1")
    plt.title("Per-Category Forgetting Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate tables and figures from aggregated results.")
    parser.add_argument(
        "--aggregate_path",
        type=str,
        default=None,
        help="Path to full_results.json",
    )
    parser.add_argument(
        "--comparison_path",
        type=str,
        default=None,
        help="Path to comparison_table.json",
    )
    parser.add_argument(
        "--basic_forgetting_path",
        type=str,
        default=None,
        help="Path to forgetting_analysis.json",
    )
    parser.add_argument(
        "--detailed_forgetting_path",
        type=str,
        default=None,
        help="Path to forgetting_analysis_detailed.json",
    )
    parser.add_argument(
        "--tables_dir",
        type=str,
        default=None,
        help="Directory to save CSV tables",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default=None,
        help="Directory to save figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()
    outputs_cfg = config.get("outputs", {})

    aggregate_dir = outputs_cfg.get("aggregate_dir", "outputs/aggregate")

    aggregate_path = args.aggregate_path if args.aggregate_path else f"{aggregate_dir}/full_results.json"
    comparison_path = args.comparison_path if args.comparison_path else f"{aggregate_dir}/comparison_table.json"
    basic_forgetting_path = (
        args.basic_forgetting_path
        if args.basic_forgetting_path
        else f"{aggregate_dir}/forgetting_analysis.json"
    )
    detailed_forgetting_path = (
        args.detailed_forgetting_path
        if args.detailed_forgetting_path
        else f"{aggregate_dir}/forgetting_analysis_detailed.json"
    )

    tables_dir = args.tables_dir if args.tables_dir else outputs_cfg.get("tables_dir", "outputs/tables")
    figures_dir = args.figures_dir if args.figures_dir else outputs_cfg.get("figures_dir", "figures")

    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    full_results = load_json(aggregate_path)
    comparison_table = load_json(comparison_path)
    basic_forgetting = load_json(basic_forgetting_path)

    detailed_forgetting = {}
    if Path(detailed_forgetting_path).exists():
        detailed_forgetting = load_json(detailed_forgetting_path)

    main_df = build_main_comparison_df(comparison_table)
    basic_forgetting_df = build_basic_forgetting_df(basic_forgetting)

    save_dataframe(main_df, f"{tables_dir}/main_comparison_table.csv")
    save_dataframe(basic_forgetting_df, f"{tables_dir}/forgetting_table.csv")

    if detailed_forgetting:
        detailed_forgetting_df = build_detailed_forgetting_df(detailed_forgetting)
        category_df = build_category_forgetting_df(detailed_forgetting)
        regression_df = build_representative_examples_df(detailed_forgetting, "regressions")
        improvement_df = build_representative_examples_df(detailed_forgetting, "improvements")
        held_steady_df = build_representative_examples_df(detailed_forgetting, "held_steady")

        save_dataframe(detailed_forgetting_df, f"{tables_dir}/forgetting_detailed_table.csv")
        save_dataframe(category_df, f"{tables_dir}/forgetting_by_category.csv")
        save_dataframe(regression_df, f"{tables_dir}/representative_regressions.csv")
        save_dataframe(improvement_df, f"{tables_dir}/representative_improvements.csv")
        save_dataframe(held_steady_df, f"{tables_dir}/representative_held_steady.csv")

    plot_alpaca_metrics(full_results, f"{figures_dir}/alpaca_metrics.png")
    plot_json_metrics(full_results, f"{figures_dir}/json_metrics.png")
    plot_judge_results(full_results, f"{figures_dir}/judge_win_rates.png")

    if detailed_forgetting:
        plot_category_forgetting(detailed_forgetting, f"{figures_dir}/forgetting_by_category.png")

    print("Saved tables:")
    print(f"  {tables_dir}/main_comparison_table.csv")
    print(f"  {tables_dir}/forgetting_table.csv")

    if detailed_forgetting:
        print(f"  {tables_dir}/forgetting_detailed_table.csv")
        print(f"  {tables_dir}/forgetting_by_category.csv")
        print(f"  {tables_dir}/representative_regressions.csv")
        print(f"  {tables_dir}/representative_improvements.csv")
        print(f"  {tables_dir}/representative_held_steady.csv")

    print("\nSaved figures:")
    print(f"  {figures_dir}/alpaca_metrics.png")
    print(f"  {figures_dir}/json_metrics.png")
    print(f"  {figures_dir}/judge_win_rates.png")

    if detailed_forgetting:
        print(f"  {figures_dir}/forgetting_by_category.png")


if __name__ == "__main__":
    main()

