#!/bin/bash
#SBATCH --job-name=nemotron_matching
#SBATCH --output=nemotron_matching_%j.out
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --nodelist=hpc-n739
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      
#SBATCH --mem=32G               
#SBATCH --gres=gpu:1   
#SBATCH --time=08:00:00         
#SBATCH --exclude=hpc-n967

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"



{
    echo "--- CHECKING GPU ALLOCATION ---"
    nvidia-smi

    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate barthez_env  
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    echo "Starting Nemotron extraction"
    python -u ~/Rhine_proj/scripts/code/1.Matching_img_texts_Pipeline/5.Nemotron_exp.py
} 2>&1 | tee ~/Rhine_proj/outputs/live_logs/1.Image_matching_output/nemotron_output.txt