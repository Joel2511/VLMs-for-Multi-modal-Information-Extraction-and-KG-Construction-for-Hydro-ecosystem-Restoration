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

# python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'MrLight/dse-qwen2-2b-mrl-v1', '--local-dir', '/home2020/home/icube/janil/Rhine_proj/hf_models/dse-qwen2-2b', '--token', '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'], env=env)"

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