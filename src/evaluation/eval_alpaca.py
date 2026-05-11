
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import evaluate
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


def generate_response_with_loaded_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
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


def token_count_approx(text: str) -> int:
    return len(str(text).split())


def compute_task_completion_rate(predictions: List[str]) -> float:
    if not predictions:
        return 0.0

    completed = sum(1 for p in predictions if str(p).strip())
    return completed / len(predictions)


def summarize_lengths(predictions: List[str]) -> Dict[str, float]:
    if not predictions:
        return {
            "avg_output_tokens": 0.0,
            "min_output_tokens": 0.0,
            "max_output_tokens": 0.0,
        }

    lengths = [token_count_approx(p) for p in predictions]
    return {
        "avg_output_tokens": sum(lengths) / len(lengths),
        "min_output_tokens": min(lengths),
        "max_output_tokens": max(lengths),
    }


def compute_automatic_metrics(predictions: List[str], references: List[str]) -> Dict[str, Any]:
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )

    bert_scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
    )

    bert_p = (
        sum(bert_scores["precision"]) / len(bert_scores["precision"])
        if bert_scores["precision"]
        else 0.0
    )
    bert_r = (
        sum(bert_scores["recall"]) / len(bert_scores["recall"])
        if bert_scores["recall"]
        else 0.0
    )
    bert_f1 = (
        sum(bert_scores["f1"]) / len(bert_scores["f1"])
        if bert_scores["f1"]
        else 0.0
    )

    length_stats = summarize_lengths(predictions)
    completion_rate = compute_task_completion_rate(predictions)

    return {
        "rouge1": rouge_scores.get("rouge1", 0.0),
        "rouge2": rouge_scores.get("rouge2", 0.0),
        "rougeL": rouge_scores.get("rougeL", 0.0),
        "rougeLsum": rouge_scores.get("rougeLsum", 0.0),
        "bertscore_precision": bert_p,
        "bertscore_recall": bert_r,
        "bertscore_f1": bert_f1,
        "task_completion_rate": completion_rate,
        **length_stats,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Alpaca-style held-out prompts.")
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
        help="Optional explicit Alpaca eval dataset path. Defaults to config['paths']['alpaca_eval'].",
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
        help="Defaults to config['evaluation']['alpaca_max_new_tokens'] if present, else 256.",
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
    eval_path = args.eval_path if args.eval_path else config["paths"]["alpaca_eval"]
    output_dir = args.output_dir if args.output_dir else f"outputs/eval_alpaca/{checkpoint_name}"
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else eval_cfg.get("alpaca_max_new_tokens", 256)
    )
    temperature = (
        args.temperature
        if args.temperature is not None
        else eval_cfg.get("temperature", 0.0)
    )

    predictions_path = f"{output_dir}/predictions.json"
    summary_path = f"{output_dir}/summary.json"

    examples = load_json(eval_path)

    print("===== Alpaca Evaluation =====")
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Model path: {model_path}")
    print(f"Eval dataset: {eval_path}")
    print(f"Num examples: {len(examples)}")

    model, tokenizer = load_model_and_tokenizer(model_path)

    results = []
    predictions = []
    references = []

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

        ref_text = str(example.get("output", "")).strip()

        predictions.append(pred_text)
        references.append(ref_text)

        row = {
            "id": example.get("id", idx),
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "reference_output": ref_text,
            "predicted_output": pred_text,
            "prediction_token_count": token_count_approx(pred_text),
            "reference_token_count": token_count_approx(ref_text),
            "completed": int(bool(pred_text.strip())),
        }
        results.append(row)

        if idx % 10 == 0 or idx == len(examples):
            print(f"Processed {idx}/{len(examples)} examples")

    metrics = compute_automatic_metrics(predictions=predictions, references=references)

    summary = {
        "checkpoint_name": checkpoint_name,
        "model_path": model_path,
        "eval_path": eval_path,
        "num_examples": len(examples),
        "automatic_metrics": metrics,
        "generation_settings": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }

    save_json(results, predictions_path)
    save_json(summary, summary_path)

    print("\n===== Alpaca Eval Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved detailed predictions to: {predictions_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

