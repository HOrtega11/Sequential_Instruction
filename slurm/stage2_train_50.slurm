#!/bin/bash
#SBATCH --job-name=stage2_json50
#SBATCH --output=logs/stage2_json50_%j.out
#SBATCH --error=logs/stage2_json50_%j.err
#SBATCH --partition=gpu1v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

echo "===== Stage 2 (JSON 50) Training Job Started ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

cd /work/xzy032/sequential_instruction

module purge
module load anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seq_tuning

export PYTHONPATH=/work/xzy032/sequential_instruction:${PYTHONPATH:-}
export TRANSFORMERS_CACHE=/work/xzy032/.cache/huggingface
export HF_HOME=/work/xzy032/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Python: $(which python)"
python --version
echo "Conda env: ${CONDA_DEFAULT_ENV:-not_set}"
echo "Working directory: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

nvidia-smi

mkdir -p logs outputs/checkpoint2_json50

if [ ! -d "outputs/checkpoint1" ]; then
    echo "ERROR: outputs/checkpoint1 not found!"
    exit 1
fi

if [ ! -f "data/json_instruct/json_train_50.json" ]; then
    echo "ERROR: data/json_instruct/json_train_50.json not found!"
    exit 1
fi

echo "Checkpoint1 and json_train_50.json found. Proceeding."

echo "Installed package snapshot:"
pip list | grep -E "transformers|torch|peft|bitsandbytes|datasets|accelerate|trl" || true

echo "===== Running Stage 2 Training (json_train_50) ====="
python -m src.training.train_stage2 \
  --train_path data/json_instruct/json_train_50.json \
  --stage1_dir outputs/checkpoint1 \
  --output_dir outputs/checkpoint2_json50

if [ ! -d "outputs/checkpoint2_json50" ]; then
    echo "ERROR: checkpoint2_json50 not created!"
    exit 1
fi

echo "Checkpoint2_json50 successfully created."
echo "===== Stage 2 (JSON 50) Training Job Finished ====="
echo "End time: $(date)"