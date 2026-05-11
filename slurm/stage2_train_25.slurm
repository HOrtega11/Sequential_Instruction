#!/bin/bash
#SBATCH --job-name=stage2_json25
#SBATCH --output=logs/stage2_json25_%j.out
#SBATCH --error=logs/stage2_json25_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

echo "===== Stage 2 (JSON 25) Training Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# 🔹 Go to correct directory
cd /work/xzy032/sequential_instruction

# 🔹 Clean module environment
module purge
module load anaconda3

# 🔹 Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seq_tuning

# 🔹 Paths
export PYTHONPATH=/work/xzy032/sequential_instruction:${PYTHONPATH:-}

# 🔹 HuggingFace cache
export TRANSFORMERS_CACHE=/work/xzy032/.cache/huggingface
export HF_HOME=/work/xzy032/.cache/huggingface

# 🔹 Reduce log noise
export TOKENIZERS_PARALLELISM=false

# 🔹 CUDA stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 🔹 Debug info
echo "Python: $(which python)"
python --version
echo "Conda env: ${CONDA_DEFAULT_ENV:-not_set}"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

# 🔹 GPU check
nvidia-smi

# 🔹 Ensure directories exist
mkdir -p logs outputs/checkpoint2_json25

# 🔹 Verify Stage 1 checkpoint exists
if [ ! -d "outputs/checkpoint1" ]; then
    echo "ERROR: outputs/checkpoint1 not found!"
    exit 1
fi

echo "Checkpoint1 found. Proceeding with Stage 2 (json_train_25)."

# 🔹 Verify dataset exists
if [ ! -f "data/json_instruct/json_train_25.json" ]; then
    echo "ERROR: json_train_25.json not found!"
    exit 1
fi

# 🔹 Package snapshot
echo "Installed package snapshot:"
pip list | grep -E "transformers|torch|peft|bitsandbytes|datasets|accelerate|trl" || true

# 🔹 Run Stage 2
echo "===== Running Stage 2 Training (json_train_25) ====="
python -m src.training.train_stage2 \
  --train_path data/json_instruct/json_train_25.json \
  --stage1_dir outputs/checkpoint1 \
  --output_dir outputs/checkpoint2_json25

# 🔹 Verify output
if [ ! -d "outputs/checkpoint2_json25" ]; then
    echo "ERROR: checkpoint2_json25 not created!"
    exit 1
fi

echo "Checkpoint2_json25 successfully created."

echo "===== Stage 2 (JSON 25) Training Job Finished ====="
echo "End time: $(date)"