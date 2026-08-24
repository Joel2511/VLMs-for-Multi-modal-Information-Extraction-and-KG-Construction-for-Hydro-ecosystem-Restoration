#!/bin/bash
#SBATCH --job-name=colqwen
#SBATCH --partition=publicgpu   
#SBATCH --nodes=1
#SBATCH --nodelist=hpc-n739
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       
#SBATCH --mem=32G               
#SBATCH --gres=gpu:1    
#SBATCH --time=02:00:00         
#SBATCH --exclude=hpc-n967
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/logs/colqwen_%j.log


# python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'vidore/colqwen2-v1.0-hf', '--local-dir', '/home2020/home/icube/janil/Rhine_proj/hf_models/colqwen2-v1.0-hf', '--token', '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'], env=env)"
{
    echo "--- CHECKING GPU ALLOCATION ---"
    nvidia-smi

    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate barthez_env  
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    echo "Starting ColQwen extraction ."
    python -u ~/Rhine_proj/scripts/code/1.Matching_img_texts_Pipeline/2b.colqwen_exp.py

} 2>&1 | tee ~/Rhine_proj/outputs/live_logs/1.Image_matching_output/colqwen_output.txt