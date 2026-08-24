#!/bin/bash
#SBATCH --job-name=gen_synth_dataset
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodelist=hpc-n978
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/1.dataset_generation/logs/%j.out
#SBATCH --error=/home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/1.dataset_generation/logs/%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate qwen3vl_env

unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "[SLURM] Job started on $(hostname) at $(date)"
echo "[SLURM] Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

python3 -u /home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/1.dataset_generation/generate_dataset.py

echo "[SLURM] Job finished at $(date)"