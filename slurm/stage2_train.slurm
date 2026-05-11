#!/bin/bash
#SBATCH --job-name=stage2_json
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

echo "===== Stage 2 Training Job Started ====="
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

# 🔹 HuggingFace cache (avoid /home issues)
export TRANSFORMERS_CACHE=/work/xzy032/.cache/huggingface
export HF_HOME=/work/xzy032/.cache/huggingface

# 🔹 Reduce log noise
export TOKENIZERS_PARALLELISM=false

# 🔹 Optional: helps with some CUDA fragmentation issues
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
mkdir -p logs outputs/checkpoint2

# 🔹 Verify Stage 1 checkpoint exists (CRITICAL)
if [ ! -d "outputs/checkpoint1" ]; then
    echo "ERROR: outputs/checkpoint1 not found!"
    echo "Stage 1 must complete before Stage 2."
    exit 1
fi

echo "Checkpoint1 found. Proceeding with Stage 2."

# 🔹 Print key packages
echo "Installed package snapshot:"
pip list | grep -E "transformers|torch|peft|bitsandbytes|datasets|accelerate|trl" || true

# 🔹 Run Stage 2 training
echo "===== Running Stage 2 Training ====="
python -m src.training.train_stage2 \
  --train_path data/json_instruct/json_train_validated.json \
  --stage1_dir outputs/checkpoint1 \
  --output_dir outputs/checkpoint2

# 🔹 Verify output
if [ ! -d "outputs/checkpoint2" ]; then
    echo "ERROR: checkpoint2 not created!"
    exit 1
fi

echo "Checkpoint2 successfully created."

echo "===== Stage 2 Training Job Finished ====="
echo "End time: $(date)"