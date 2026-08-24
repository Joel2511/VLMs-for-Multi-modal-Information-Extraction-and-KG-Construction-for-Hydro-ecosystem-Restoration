#!/bin/bash
#SBATCH --job-name=merge_adapter
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-n983
#SBATCH --time=02:00:00
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/5.merge/logs/%j.out
#SBATCH --error=/home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/5.merge/logs/%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate qwen3vl_env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "[SLURM] Job started on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

python3 -u /home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/5.merge/merge_adapter.py

echo "[SLURM] Job finished at $(date)"