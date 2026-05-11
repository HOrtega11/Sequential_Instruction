# Sequential Instruction Tuning

## Overview
This project builds and evaluates sequential instruction tuning pipeline for studying structured output specialization and catastrophic forgetting in small language models with a strong model judge evaluation. The pipeline compares three checkpoints:

-Checkpoint 0: Untrained base model

-Checkpoint 1: Alpaca only fine tuned model

-Checkpoint 2: fine-tuning on teacher-generated JSON Instruct data on top of the Alpaca checkpoint

The repository contains modular code, configuration of hyperparameters, editable prompt templates, HPC training scripts, experiment logging, evaluation outputs, and analysis.

## Modular Code files
### Data Preparation
prepare_alpaca.py: Loads, normalizes, shuffles, and splits Alpaca-style instruction data.

generate_teacher_dataset.py: Generates structured JSON instruction-following data using a stronger teacher model.

validate_json_dataset.py: Validates generated JSON outputs and checks schema correctness.

make_stage2_subsets.py: Creates 50%, and 25% Stage 2 dataset ablation subsets.

### Training
train_stage1.py: Performs Stage 1 Alpaca instruction tuning using QLoRA.

train_stage2.py: Continues fine-tuning from Checkpoint 1 using teacher-generated JSON tasks.

### Inference
run_inference.py: Runs single-prompt or dataset inference for all checkpoints.

### Evaluation
eval_alpaca.py: Evaluates instruction-following capability using ROUGE, BERTScore, and task completion metrics.

eval_json.py: Evaluates structured JSON outputs using JSON validity, schema compliance, exact match, field level correctness, and error taxonomy

judge_evaluation: Performs pairwise LLM judge evaluation with randomized A/B ordering.

forgetting_analysis.py: Measures catastrophic forgetting between Checkpoint 1 and Checkpoint 2.

aggregate_results.py: Aggregates all experiment outputs into unified result summaries.

make_tables_and_figures.py: Generates tables and figures

### Utilities
config.py: central configuration containing model names, learning rates, LoRA parameters, batch sizes, epoch counts, max tokesn, evaluation settings, and output directories

client.py: API client used for teacher generation and judge evaluation


### Prompt Templates
inference_prompts.py: Shared instruction-format templates used during training and inference.

teacher_generation.py: Prompt templates used for teacher-generated JSON supervision.

judge_prompts.py: Prompt templates used for pairwise LLM judge evaluation.

### SLURM scripts
stage1_train.slurm: UTSA HPC batch script for Stage 1 training.

stage2_train.slurm: UTSA HPC batch script for Stage 2 training.

stage2_train_25.slurm: Ablation training script using 25% JSON dataset.

stage2_train_50.slurm: Ablation training script using 50% JSON dataset.

### Datasets
raw_alpaca.json: Original Alpaca dataset.

alpaca_train.json: Processed Alpaca training set.

alpaca_eval.json: Held-out Alpaca evaluation set.

### JSON Instruction Data
json_train.json: Teacher-generated structured JSON dataset.

json_train_25.json: 25% subset for ablation experiments.

json_train_50.json: 50% subset for ablation experiments.

json_train_100.json: 100% subset for ablation experiments.

json_eval.json: Held-out structured JSON evaluation set.

## Setup Instructions
1) clone the repository
2) create a virtual environment
3) install dependencies from requirements.txt
4) set API key
5) modify experiment setting in config.py if desired
6) prepare Alpaca data: python -m src.data.prepare_alpaca
7) generate teacher JSON data: python -m src.data.generate_teacher_json
8) validate JSON dataset: python -m src.data.validate_json_dataset
9) run stage 1 training: python -m src.training.train_stage1
10) run stage 2 training: python -m src.training.train_stage2
11) run inference: python -m src.inference.run_inference \
  --model_path outputs/checkpoint2 \
  --instruction "Extract the fields" \
  --input_text "John visited Dallas on Monday"
12) run Alpaca evaluation: python -m src.evaluation.eval_alpaca \
  --checkpoint_name checkpoint2
13) run JSON evaluation: python -m src.evaluation.eval_json \
  --checkpoint_name checkpoint2
14) run Judge evaluation: python -m src.evaluation.judge_evaluation \
  --task_type alpaca \
  --mode adjacent
15) run forgetting analysis: python -m src.evaluation.forgetting_analysis
16) aggregate final results: python -m src.evaluation.aggregate_results
17) generate tables and figures: python -m src.evaluation.make_tables_and_figures


## Models Used
Student model: microsoft/Phi-3-mini-4k-instruct

Teacher model: gpt-4.1

Judge model: gpt-4.1







