
CONFIG = {
    "models": {
        "student": "microsoft/Phi-3-mini-4k-instruct",
        "teacher": "gpt-4.1",
        "judge": "gpt-4.1",
    },
    "api": {
        "base_url": None,  # IMPORTANT: disable broken internal server
        "api_key_env": "MYAPIKEY1",
        "timeout": 60.0,
    },
    "generation": {
        "temperature": 0.2,
        "max_tokens": 200,
    },
    "paths": {
        "alpaca_train": "data/alpaca/alpaca_train.json",
        "alpaca_eval": "data/eval/alpaca_eval.json",
        "json_train": "data/json_instruct/json_train.json",
        "json_eval": "data/eval/json_eval.json",
    },
    "outputs": {
        "stage1_dir": "outputs/checkpoint1",
        "stage2_dir": "outputs/checkpoint2",
        "checkpoint0_dir": "outputs/checkpoint0",
        "eval_alpaca_dir": "outputs/eval_alpaca",
        "eval_json_dir": "outputs/eval_json",
        "judge_dir": "outputs/judge",
        "aggregate_dir": "outputs/aggregate",
        "tables_dir": "outputs/tables",
        "logs_dir": "logs",
        "figures_dir": "figures",
        "teacher_failures_path": "outputs/json_teacher_generation_failures.json",
        "teacher_metadata_path": "outputs/json_teacher_generation_metadata.json",
    },
    "stage1": {
        "learning_rate": 2e-5,
        "epochs": 2,
        "batch_size": 4,
        "max_seq_length": 1024,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
    },
    "stage2": {
        "learning_rate": 1e-5,
        "epochs": 2,
        "batch_size": 4,
        "max_seq_length": 1024,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 2,
    },
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "training": {
        "seed": 42,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
    },
    "inference": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
        "attn_implementation": "eager",
        "trust_remote_code": False,
    },
    "evaluation": {
        "alpaca_max_new_tokens": 256,
        "json_max_new_tokens": 512,
        "temperature": 0.0,
        "judge_temperature": 0.0,
        "judge_max_tokens": 512,
    },
    "data_generation": {
        "train_per_task": 60,
        "eval_per_task": 20,
    },
    "paths": {
        "alpaca_raw": "data/alpaca/raw_alpaca.json",
        "alpaca_train": "data/alpaca/alpaca_train.json",
        "alpaca_eval": "data/eval/alpaca_eval.json",
        "json_train": "data/json_instruct/json_train.json",
        "json_eval": "data/eval/json_eval.json",
    },
    "data_generation": {
        "alpaca_eval_size": 100,
        "train_per_task": 60,
        "eval_per_task": 20,
    },
}


def get_config():
    return CONFIG
