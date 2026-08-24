#!/bin/bash
#SBATCH --job-name=internvl25_Captioning
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=internvl25_job_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate colqwen_char
unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_ENABLE_HF_TRANSFER=1

# python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'OpenGVLab/InternVL2_5-8B', '--local-dir', '/home2020/home/icube/janil/Rhine_proj/hf_models/InternVL2_5-8B', '--token', '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'], env=env)"

# Run Inference
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -u ~/Rhine_proj/scripts/code/2.0.Image_description_ablation_no_prompt/3.InternVL2_8B_ablation.py > ~/Rhine_proj/outputs/live_logs/2.1.Ablation_Image_captioning_output/InternVL2_8B_ablation_no_prompt.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.1.Image_description_context_ocr/1e.InternVL2_8B_img_descp.py > ~/Rhine_proj/outputs/live_logs/2.Image_captioning_output/internvl25_8b_ocr_context.txt 2>&1

# Cleanup
#rm -rf /home2020/home/icube/janil/Rhine_proj/hf_models/InternVL2_5-8B