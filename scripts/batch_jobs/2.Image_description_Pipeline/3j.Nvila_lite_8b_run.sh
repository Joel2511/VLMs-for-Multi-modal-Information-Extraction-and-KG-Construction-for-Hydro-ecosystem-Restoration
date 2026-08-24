#!/bin/bash
#SBATCH --job-name=nvila_lite_8b_Captioning
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --output=nvila_lite_8b_job_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvila_env

unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_ENABLE_HF_TRANSFER=1


# python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'Efficient-Large-Model/NVILA-Lite-8B-hf', '--local-dir', '/home2020/home/icube/janil/Rhine_proj/hf_models/NVILA-Lite-8B-hf', '--token', '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'], env=env)"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


# python -u ~/Rhine_proj/scripts/code/2.0.Image_description_ablation_no_prompt/7.Nvila_lite_8B_ablation.py > ~/Rhine_proj/outputs/live_logs/2.1.Ablation_Image_captioning_output/Nvila_lite_8B_ablation_no_prompt.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.1.Image_description_context_ocr/1j.Nvila_lite_8B_img_descp.py > ~/Rhine_proj/outputs/live_logs/2.Image_captioning_output/nvila_lite_8b_ablation_ocr_context.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.2.Image_captioning_with_summary_ablation/1.Nvila_lite_8B_captions_w_summary.py > ~/Rhine_proj/outputs/live_logs/2.2.Image_captioning_with_summary_output/nvila_with_summary.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.3.Weighted_context_captioning_ablation/1.Nvila_lite_8B_blended_summary.py > ~/Rhine_proj/outputs/live_logs/2.3.Weighted_summary_captioning_output/nvila_vidore_with_weighted_summary.txt 2>&1

# python -u ~/Rhine_proj/scripts/code/2.4.Prompt_engineering_ablation/0.Nvila_A5-P0.py > ~/Rhine_proj/outputs/live_logs/2.4.Prompt_engineering_output/nvila_P0.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.4.Prompt_engineering_ablation/1.Nvila_A5-P1.py > ~/Rhine_proj/outputs/live_logs/2.4.Prompt_engineering_output/nvila_P1.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.4.Prompt_engineering_ablation/2.Nvila_A5-P2.py > ~/Rhine_proj/outputs/live_logs/2.4.Prompt_engineering_output/nvila_P2.txt 2>&1
python -u ~/Rhine_proj/scripts/code/2.4.Prompt_engineering_ablation/3.Nvila_A5-P3.py > ~/Rhine_proj/outputs/live_logs/2.4.Prompt_engineering_output/nvila_P3.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.4.Prompt_engineering_ablation/4.Nvila_A5-P4.py > ~/Rhine_proj/outputs/live_logs/2.4.Prompt_engineering_output/nvila_P4.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.4.Prompt_engineering_ablation/5.Nvila_A5-P5.py > ~/Rhine_proj/outputs/live_logs/2.4.Prompt_engineering_output/nvila_P5.txt 2>&1

# python -u ~/Rhine_proj/scripts/code/2.5.Decoding_ablation/1.Nvila_captioner_icd.py --cd_alpha 0.5 > ~/Rhine_proj/outputs/live_logs/2.5.Decoding_ablation_output/a.nvila_icd_ablation_mild.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.5.Decoding_ablation/1.Nvila_captioner_icd.py --cd_alpha 1.0 > ~/Rhine_proj/outputs/live_logs/2.5.Decoding_ablation_output/b.nvila_icd_ablation_strong.txt 2>&1

# python -u ~/Rhine_proj/scripts/code/2.5.Decoding_ablation/2.Nvila_captioner_vcd.py --cd_alpha 0.5 --noise_step 500 > ~/Rhine_proj/outputs/live_logs/2.5.Decoding_ablation_output/c.nvila_vcd_mild.txt 2>&1
# python -u ~/Rhine_proj/scripts/code/2.5.Decoding_ablation/2.Nvila_captioner_vcd.py --cd_alpha 1.0 --noise_step 500 > ~/Rhine_proj/outputs/live_logs/2.5.Decoding_ablation_output/d.nvila_vcd_strong.txt 2>&1

# rm -rf /home2020/home/icube/janil/Rhine_proj/hf_models/NVILA-Lite-8B-hf

#replace processing_nvila_lite.py file