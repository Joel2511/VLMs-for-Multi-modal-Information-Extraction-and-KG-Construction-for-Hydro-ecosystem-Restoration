#!/bin/bash
#SBATCH --job-name=qwen36_Captioning
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --exclude=hpc-n878,hpc-n870
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=qwen36_job_%j.out

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"


# 1. Activate Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate qwen36_env
unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_ENABLE_HF_TRANSFER=1


# Run Inference
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4

python -u ~/Rhine_proj/scripts/code/2.0.Image_description_ablation_no_prompt/2.Qwen3.6_27B_ablation.py > ~/Rhine_proj/outputs/live_logs/2.1.Ablation_Image_captioning_output/qwen3.6_27B_ablation_no_prompt.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.1.Image_description_context_ocr/1d.Qwen3.6_27B_img_descp.py > ~/Rhine_proj/outputs/live_logs/2.Image_captioning_output/qwen36_27b_ocr_context.txt 2>&1

# python -u ~/Rhine_proj/scripts/code/2.2.Image_captioning_with_summary_ablation/2.Qwen3.6_27B_captions_w_summary.py > ~/Rhine_proj/outputs/live_logs/2.2.Image_captioning_with_summary_output/qwen36_vidore_with_summary.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.3.Weighted_context_captioning_ablation/2.Qwen3.6_27B_blended_summary.py > ~/Rhine_proj/outputs/live_logs/2.3.Weighted_summary_captioning_output/qwen36_vidore_with_weighted_summary.txt 2>&1

# python -u ~/Rhine_proj/scripts/code/2.5.Decoding_ablation/3.Qwen3.6_27b_captioner_icd.py --cd_alpha 0.5 > ~/Rhine_proj/outputs/live_logs/2.5.Decoding_ablation_output/qwen36_icd_mild.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.5.Decoding_ablation/3.Qwen3.6_27b_captioner_icd.py --cd_alpha 1.0 > ~/Rhine_proj/outputs/live_logs/2.5.Decoding_ablation_output/qwen36_icd_strong.txt 2>&1
# Cleanup Model to free space
# rm -rf $RHINE_PROJECT_ROOT/hf_models/Qwen3.6-27B