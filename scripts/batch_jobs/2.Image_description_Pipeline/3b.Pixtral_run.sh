#!/bin/bash
#SBATCH --job-name=pixtral_Captioning
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --exclude=hpc-n868,hpc-n872,hpc-n879,hpc-n888,hpc-n889,hpc-n890,hpc-n891,hpc-n925,hpc-n871,hpc-n873,hpc-n875,hpc-n881,hpc-n882,hpc-n933,hpc-n983,hpc-n739
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"


# 1. Activate Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vlm_env            
unset CUDA_VISIBLE_DEVICES
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES=0,1


# 3. Run Inference
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -u ~/Rhine_proj/scripts/code/2.0.Image_description_ablation_no_prompt/1.Pixtral_ablation.py > ~/Rhine_proj/outputs/live_logs/2.1.Ablation_Image_captioning_output/pixtral12b_ablation_no_prompt.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.1.Image_description_context_ocr/1b.Pixtral_summarized_img_descp.py > ~/Rhine_proj/outputs/live_logs/2.Image_captioning_output/pixtral12b_ablation_ocr_context.txt 2>&1


# 4. Cleanup Model to free space
# rm -rf $RHINE_PROJECT_ROOT/hf_models/pixtral-12b-hf

