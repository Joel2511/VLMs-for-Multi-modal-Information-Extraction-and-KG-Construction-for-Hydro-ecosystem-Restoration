#!/bin/bash
#SBATCH --job-name=clip_experiment
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/logs/slurm_clip_%j.log
#SBATCH --partition=publicgpu        
#SBATCH --gres=gpu:1                   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00


# 1. Activate Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate barthez_env          

# 2. model location
export HF_HOME=~/Rhine_proj/hf_models
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'openai/clip-vit-base-patch32', '--local-dir', '/home2020/home/icube/janil/Rhine_proj/hf_models/clip-vit-base-patch32', '--token', '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'], env=env)"

# 3. Run
echo "CLIP Experiment"
python -u ~/Rhine_proj/scripts/code/1.Matching_img_texts_Pipeline/1.clip_exp.py > ~/Rhine_proj/outputs/live_logs/1.Image_matching_output/clip_output.txt 2>&1