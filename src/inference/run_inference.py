
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from prompts.inference_prompts import build_instruction_prompt
from src.utils.config import get_config


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_prompt(example: Dict[str, Any]) -> str:
    return build_instruction_prompt(
        instruction=example.get("instruction", ""),
        input_text=example.get("input", ""),
    )


def get_dtype(dtype_str: str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_quant_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    inference_cfg = config.get("inference", {})
    training_cfg = config.get("training", {})

    load_in_4bit = inference_cfg.get(
        "load_in_4bit",
        training_cfg.get("load_in_4bit", True),
    )
    quant_type = inference_cfg.get(
        "bnb_4bit_quant_type",
        training_cfg.get("bnb_4bit_quant_type", "nf4"),
    )
    use_double_quant = inference_cfg.get(
        "bnb_4bit_use_double_quant",
        training_cfg.get("bnb_4bit_use_double_quant", True),
    )
    compute_dtype_str = inference_cfg.get(
        "bnb_4bit_compute_dtype",
        training_cfg.get("bnb_4bit_compute_dtype", "float16"),
    )

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=get_dtype(compute_dtype_str),
    )


def normalize_rope_scaling(config: Any) -> Any:
    """
    Compatibility fix for Phi-3 rope_scaling config differences.
    Some cached configs use 'rope_type' while some model code expects 'type'.
    """
    if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
        if "type" not in config.rope_scaling and "rope_type" in config.rope_scaling:
            config.rope_scaling["type"] = config.rope_scaling["rope_type"]
    return config


def get_device_and_load_kwargs(config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    inference_cfg = config.get("inference", {})
    """
    Decide whether to use quantized GPU loading or CPU fallback.
    """
    use_cuda = torch.cuda.is_available()
    attn_implementation = inference_cfg.get("attn_implementation", "eager")

    if use_cuda:
        print("[INFERENCE] CUDA available. Using config-based 4-bit quantized loading.")
        return True, {
            "quantization_config": get_quant_config(config),
            "device_map": "auto",
            "torch_dtype": get_dtype(
                inference_cfg.get("bnb_4bit_compute_dtype", "float16")
            ),
            "attn_implementation": attn_implementation,
        }

    print("[INFERENCE] CUDA unavailable. Falling back to non-quantized CPU loading.")
    return False, {
        "device_map": None,
        "torch_dtype": torch.float32,
        "attn_implementation": attn_implementation,
    }


def load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any]:
    """
    model_path can be:
    - 'checkpoint0' or 'outputs/checkpoint0' -> load untuned base model
    - 'outputs/checkpoint1'                  -> load base model + Stage 1 adapter
    - 'outputs/checkpoint2'                  -> load base model + Stage 2 adapter
    """
    config = get_config()
    base_model_name = config["models"]["student"]

    normalized = str(model_path).strip()
    is_checkpoint0 = normalized in {"checkpoint0", "outputs/checkpoint0"}

    use_cuda, load_kwargs = get_device_and_load_kwargs(config)

    print(f"[INFERENCE] Loading tokenizer from: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=True,
        trust_remote_code = config.get("inference", {}).get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[INFERENCE] Loading config from: {base_model_name}")
    model_config = AutoConfig.from_pretrained(
        base_model_name,
        trust_remote_code = config.get("inference", {}).get("trust_remote_code", False),
    )
    model_config = normalize_rope_scaling(model_config)

    if is_checkpoint0:
        print(f"[INFERENCE] Loading untuned base model: {base_model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=model_config,
            trust_remote_code = config.get("inference", {}).get("trust_remote_code", False),
            **load_kwargs,
        )

        if not use_cuda:
            model.to("cpu")

        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()
        return model, tokenizer

    adapter_path = Path(normalized)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Model/adapter path not found: {normalized}")

    print(f"[INFERENCE] Loading base model for adapter: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        config=model_config,
        trust_remote_code = config.get("inference", {}).get("trust_remote_code", False),
        **load_kwargs,
    )

    if not use_cuda:
        base_model.to("cpu")

    print(f"[INFERENCE] Loading adapter from: {normalized}")
    model = PeftModel.from_pretrained(
        base_model,
        normalized,
        is_trainable=False,
    )

    tokenizer_source = (
        normalized if (adapter_path / "tokenizer_config.json").exists() else base_model_name
    )
    if tokenizer_source != base_model_name:
        print(f"[INFERENCE] Loading tokenizer from adapter path: {tokenizer_source}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=True,
            trust_remote_code = config.get("inference", {}).get("trust_remote_code", False),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    return model, tokenizer


def generate_response_with_loaded_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: Optional[bool] = None,
) -> str:
    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    if do_sample is None:
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


def generate_response(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: Optional[bool] = None,
) -> str:
    model, tokenizer = load_model_and_tokenizer(model_path)
    return generate_response_with_loaded_model(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
    )


def run_dataset_inference(
    model_path: str,
    dataset_path: str,
    output_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> None:
    examples = load_json(dataset_path)

    print("===== Running Dataset Inference =====")
    print(f"Model path: {model_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Num examples: {len(examples)}")

    model, tokenizer = load_model_and_tokenizer(model_path)

    results = []
    for idx, example in enumerate(examples, start=1):
        prompt = build_prompt(example)
        prediction = generate_response_with_loaded_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        results.append(
            {
                "id": example.get("id", idx),
                "instruction": example.get("instruction", ""),
                "input": example.get("input", ""),
                "reference_output": example.get("output", ""),
                "predicted_output": prediction,
                "task_type": example.get("task_type"),
            }
        )

        if idx % 10 == 0 or idx == len(examples):
            print(f"Processed {idx}/{len(examples)} examples")

    save_json(results, output_path)
    print(f"Saved inference outputs to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for a checkpoint or adapter.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="checkpoint0, outputs/checkpoint1, or outputs/checkpoint2",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional dataset JSON path with instruction/input/output fields",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save dataset predictions if dataset_path is provided",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional single prompt string for one-off inference",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional instruction for one-off inference using the shared prompt template.",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="",
        help="Optional input text used with --instruction.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_path:
        if not args.output_path:
            raise ValueError("You must provide --output_path when using --dataset_path")
        run_dataset_inference(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        return

    if args.prompt:
        text = generate_response(
            model_path=args.model_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(text)
        return

    if args.instruction is not None:
        prompt = build_instruction_prompt(
            instruction=args.instruction,
            input_text=args.input_text,
        )
        text = generate_response(
            model_path=args.model_path,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(text)
        return

    raise ValueError("Provide either --dataset_path, --prompt, or --instruction")


if __name__ == "__main__":
    main()

