#!/bin/bash
#SBATCH --job-name=mistral31_Captioning
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=mistral31_job_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate vlm_env
unset CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES=0,1

# Download Mistral-Small-3.1-24B (Just-In-Time)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
#python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'mistralai/Mistral-Small-3.1-24B-Instruct-2503', '--local-dir', '/home2020/home/icube/janil/Rhine_proj/hf_models/Mistral-Small-3.1-24B-Instruct-2503', '--exclude', 'consolidated.safetensors', '--token', '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'], env=env)"
# Run Inference
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi --gpu-reset 2>/dev/null || true

python -u ~/Rhine_proj/scripts/code/2.1.Image_description_context_ocr/1g.Mistral3.1_24B_img_descp.py > ~/Rhine_proj/outputs/live_logs/2.Image_captioning_output/mistral_small31_ablation_ocr_context.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.0.Image_description_ablation_no_prompt/4.Mistral3.1_24B_ablation.py > ~/Rhine_proj/outputs/live_logs/2.1.Ablation_Image_captioning_output/Mistral3.1_24B_ablation_no_prompt.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.2.Image_captioning_with_summary_ablation/4.Mistral3.1_24B_captions_w_summary.py > ~/Rhine_proj/outputs/live_logs/2.2.Image_captioning_with_summary_output/mistral_with_summary.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.3.Weighted_context_captioning_ablation/4.Mistral3.1_24B_blended_summary.py > ~/Rhine_proj/outputs/live_logs/2.3.Weighted_summary_captioning_output/mistral_with_weighted_summary.txt 2>&1

# Cleanup Model to free space
# rm -rf /home2020/home/icube/janil/Rhine_proj/hf_models/Mistral-Small-3.1-24B-Instruct-2503