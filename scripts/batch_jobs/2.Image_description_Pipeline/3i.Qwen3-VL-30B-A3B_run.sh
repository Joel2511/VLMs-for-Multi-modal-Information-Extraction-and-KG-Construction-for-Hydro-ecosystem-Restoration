#!/bin/bash
#SBATCH --job-name=qwen3vl_Captioning
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --nodelist=hpc-n878
#SBATCH --gres=gpu:3
#SBATCH --time=04:00:00

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"


# CANDIDATES where EVERY individual GPU >= 24000MB: hpc-n863 hpc-n875 hpc-n878 hpc-n983
# 1. Activate Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qwen3vl_env
# unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_ENABLE_HF_TRANSFER=1

# python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'Qwen/Qwen3-VL-30B-A3B-Instruct', '--local-dir', '$RHINE_PROJECT_ROOT/hf_models/Qwen3-VL-30B-A3B-Instruct', '--token', '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'], env=env)"

# Run Inference
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


python -u ~/Rhine_proj/scripts/code/2.0.Image_description_ablation_no_prompt/6.Qwen3_VL_30B_A3b_ablation.py > ~/Rhine_proj/outputs/live_logs/2.1.Ablation_Image_captioning_output/qwen3vl_30B_A3b_ablation_no_prompt.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.1.Image_description_context_ocr/1i.Qwen3_VL_30B_A3b_img_descp.py > ~/Rhine_proj/outputs/live_logs/2.Image_captioning_output/qwen3vl_30b_ocr_context.txt 2>&1

# python -u ~/Rhine_proj/scripts/code/2.2.Image_captioning_with_summary_ablation/3.Qwen3_VL_30B_captions_w_summary.py > ~/Rhine_proj/outputs/live_logs/2.2.Image_captioning_with_summary_output/qwen3vl_with_summary.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.3.Weighted_context_captioning_ablation/3.Qwen3_VL_blended_summary.py > ~/Rhine_proj/outputs/live_logs/2.3.Weighted_summary_captioning_output/qwen3vl__with_weighted_summary.txt 2>&1

# Cleanup Model to free space
# rm -rf $RHINE_PROJECT_ROOT/hf_models/Qwen3-VL-30B-A3B-Instruct