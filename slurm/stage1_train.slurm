#!/bin/bash
#SBATCH --job-name=stage1_alpaca
#SBATCH --output=logs/stage1_%j.out
#SBATCH --error=logs/stage1_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

echo "===== Stage 1 Training Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# 🔹 Go to work directory (correct location)
cd /work/xzy032/sequential_instruction

# 🔹 Load environment
module purge
module load anaconda3

# 🔹 Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seq_tuning

# 🔹 Set paths
export PYTHONPATH=/work/xzy032/sequential_instruction:${PYTHONPATH:-}

# 🔹 Optional but helpful for HF + caching
export TRANSFORMERS_CACHE=/work/xzy032/.cache/huggingface
export HF_HOME=/work/xzy032/.cache/huggingface

# 🔹 Prevent tokenizer parallelism warning spam
export TOKENIZERS_PARALLELISM=false

# 🔹 Debug info
echo "Python: $(which python)"
python --version
echo "Conda env: ${CONDA_DEFAULT_ENV:-not_set}"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

# 🔹 GPU check
nvidia-smi

# 🔹 Ensure directories exist
mkdir -p logs outputs/checkpoint1

# 🔹 Print key packages (debugging)
echo "Installed package snapshot:"
pip list | grep -E "transformers|torch|peft|bitsandbytes|datasets|accelerate|trl" || true

# 🔹 Run Stage 1 training
echo "===== Running Stage 1 Training ====="
python -m src.training.train_stage1

# 🔹 Check output exists (important)
if [ ! -d "outputs/checkpoint1" ]; then
    echo "ERROR: checkpoint1 not created!"
    exit 1
fi

echo "Checkpoint1 successfully created."

echo "===== Stage 1 Training Job Finished ====="
echo "End time: $(date)"