#!/bin/bash
#SBATCH --job-name=dse_qwen2_matching
#SBATCH --partition=publicgpu   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       
#SBATCH --mem=32G               
#SBATCH --gres=gpu:1    
#SBATCH --time=02:00:00         
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/logs/dse_qwen2_%j.log

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"



{
    echo "--- CHECKING GPU ALLOCATION ---"
    nvidia-smi

    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate barthez_env  
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    echo "Starting ColQwen extraction ."
    python -u ~/Rhine_proj/scripts/code/1.Matching_img_texts_Pipeline/3.Dse-qwen.py

} 2>&1 | tee ~/Rhine_proj/outputs/live_logs/1.Image_matching_output/dse_qwen2_output.txt