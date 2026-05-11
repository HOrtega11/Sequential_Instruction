
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from prompts.inference_prompts import build_training_prompt
from src.utils.config import get_config


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_dataset(records: List[Dict]) -> Dataset:
    rows = [
        {
            "text": build_training_prompt(
                instruction=record.get("instruction", ""),
                input_text=record.get("input", ""),
                output=record.get("output", ""),
            )
        }
        for record in records
    ]
    return Dataset.from_list(rows)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing Alpaca dataset",
    )


def get_device_summary() -> Dict:
    summary = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }

    if torch.cuda.is_available():
        summary["devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            summary["devices"].append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor,
                }
            )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1 fine-tuning on Alpaca instruction data."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Optional Stage 1 training dataset path. Overrides config['paths']['alpaca_train'].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional Stage 1 output directory. Overrides config['outputs']['stage1_dir'].",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Optional Stage 1 learning rate override.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional Stage 1 epoch override.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional Stage 1 batch size override.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Optional Stage 1 max sequence length override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_config()

    model_name = config["models"]["student"]
    default_train_path = config["paths"]["alpaca_train"]
    default_output_dir = config.get("outputs", {}).get("stage1_dir", "outputs/checkpoint1")

    stage_cfg = config["stage1"]
    lora_cfg = config.get("lora", {})
    api_cfg = config.get("api", {})
    generation_cfg = config.get("generation", {})
    training_cfg = config.get("training", {})
    outputs_cfg = config.get("outputs", {})

    train_path = args.train_path if args.train_path else default_train_path
    output_dir = args.output_dir if args.output_dir else default_output_dir

    max_length = (
        args.max_seq_length
        if args.max_seq_length is not None
        else stage_cfg["max_seq_length"]
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else stage_cfg["learning_rate"]
    )
    num_epochs = args.epochs if args.epochs is not None else stage_cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else stage_cfg["batch_size"]

    grad_accum = stage_cfg.get("gradient_accumulation_steps", 4)
    warmup_ratio = stage_cfg.get("warmup_ratio", 0.03)
    logging_steps = stage_cfg.get("logging_steps", 10)
    save_steps = stage_cfg.get("save_steps", 100)
    save_total_limit = stage_cfg.get("save_total_limit", 2)
    seed = training_cfg.get("seed", 42)

    load_in_4bit = training_cfg.get("load_in_4bit", True)
    bnb_4bit_quant_type = training_cfg.get("bnb_4bit_quant_type", "nf4")
    bnb_4bit_use_double_quant = training_cfg.get("bnb_4bit_use_double_quant", True)
    bnb_compute_dtype_str = training_cfg.get("bnb_4bit_compute_dtype", "float16")

    if bnb_compute_dtype_str == "bfloat16":
        bnb_compute_dtype = torch.bfloat16
        use_bf16 = True
        use_fp16 = False
    else:
        bnb_compute_dtype = torch.float16
        use_bf16 = False
        use_fp16 = True

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/logs").mkdir(parents=True, exist_ok=True)

    print("===== Stage 1: Alpaca Fine-Tuning =====")
    print(f"Student model: {model_name}")
    print(f"Training data: {train_path}")
    print(f"Output dir: {output_dir}")

    if not Path(train_path).exists():
        raise FileNotFoundError(f"Stage 1 training data not found: {train_path}")

    device_summary = get_device_summary()
    print("Device summary:")
    print(json.dumps(device_summary, indent=2))

    print(f"Loading Alpaca training data from: {train_path}")
    records = load_json(train_path)
    print(f"Loaded {len(records)} training examples")

    dataset = format_dataset(records)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized_dataset = tokenize_dataset(dataset, tokenizer, max_length=max_length)
    print(f"Tokenized dataset size: {len(tokenized_dataset)}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=bnb_compute_dtype,
    )

    print(f"Loading student model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg["target_modules"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        logging_dir=f"{output_dir}/logs",
        seed=seed,
        data_seed=seed,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    run_metadata = {
        "stage": "stage1",
        "model_name": model_name,
        "train_path": train_path,
        "output_dir": output_dir,
        "device_summary": device_summary,
        "api_config_used": {
            "base_url": api_cfg.get("base_url"),
        },
        "generation_config_used": {
            "temperature": generation_cfg.get("temperature"),
            "max_tokens": generation_cfg.get("max_tokens"),
        },
        "lora_config_used": {
            "r": lora_cfg.get("r"),
            "alpha": lora_cfg.get("alpha"),
            "dropout": lora_cfg.get("dropout"),
            "target_modules": lora_cfg.get("target_modules"),
        },
        "quantization_config_used": {
            "load_in_4bit": load_in_4bit,
            "bnb_4bit_quant_type": bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
            "bnb_4bit_compute_dtype": bnb_compute_dtype_str,
        },
        "stage1_effective_config": {
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "max_seq_length": max_length,
            "gradient_accumulation_steps": grad_accum,
            "warmup_ratio": warmup_ratio,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
            "seed": seed,
        },
        "output_paths": outputs_cfg,
        "num_training_examples": len(records),
        "num_tokenized_examples": len(tokenized_dataset),
    }
    save_json(run_metadata, f"{output_dir}/run_config.json")

    print("Starting Stage 1 training on Alpaca data...")
    train_result = trainer.train()

    print(f"Saving Stage 1 adapter and tokenizer to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    summary = {
        "stage": "stage1",
        "status": "completed",
        "train_metrics": metrics,
        "train_path": train_path,
        "output_dir": output_dir,
    }
    save_json(summary, f"{output_dir}/train_summary.json")

    print("Stage 1 training complete.")
    print("Train metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

