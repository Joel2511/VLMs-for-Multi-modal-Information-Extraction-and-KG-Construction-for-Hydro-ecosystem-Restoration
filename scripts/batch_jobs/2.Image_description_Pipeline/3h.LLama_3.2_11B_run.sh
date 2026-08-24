#!/bin/bash
#SBATCH --job-name=llama32vision_Captioning
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=llama32vision_job_%j.out

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"


source ~/miniforge3/etc/profile.d/conda.sh
conda activate qwen3vl_env

unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_ENABLE_HF_TRANSFER=1

export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# Optional first-time download
# python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit', '--local-dir', '$RHINE_PROJECT_ROOT/hf_models/Llama-3.2-11B-Vision-Instruct', '--token', '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'], env=env)"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -u ~/Rhine_proj/scripts/code/2.1.Image_description_context_ocr/1h.LLama_3.2_11B_img_descp.py > ~/Rhine_proj/outputs/live_logs/2.Image_captioning_output/llama32vision_11b_ocr_context.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.0.Image_description_ablation_no_prompt/5.LLama_3.2_11B_ablation.py > ~/Rhine_proj/outputs/live_logs/2.1.Ablation_Image_captioning_output/LLama_3.2_11B_ablation_no_prompt.txt 2>&1

# Optional cleanup
# rm -rf $RHINE_PROJECT_ROOT/hf_models/Llama-3.2-11B-Vision-Instruct