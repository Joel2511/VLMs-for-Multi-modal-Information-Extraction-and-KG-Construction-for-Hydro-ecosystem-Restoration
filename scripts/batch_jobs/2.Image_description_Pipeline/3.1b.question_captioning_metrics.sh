#!/bin/bash
#SBATCH --job-name=verif_captions
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=08:00:00
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/logs/vcap_job_%j.out
#SBATCH --error=/home2020/home/icube/janil/Rhine_proj/logs/vcap_job_%j.err

# ---------------------------------------------------------------------------
# 1. Activate Conda environment
# ---------------------------------------------------------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vlm_env

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# GPU allocation info
echo "===== Job node: $(hostname) ====="
echo "===== Allocated GPU: $CUDA_VISIBLE_DEVICES ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

echo "NLI and VQA metrics on Qwen27b using vidore"
# ---------------------------------------------------------------------------
python -u ~/Rhine_proj/scripts/code/2.Image_description_Pipeline/3.1.Question_captions_metrics.py > ~/Rhine_proj/outputs/live_logs/qwen27b_question_metrics_vidore.txt 2>&1