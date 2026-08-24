#!/bin/bash
#SBATCH --job-name=qwen_sum
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --time=04:00:00
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/logs/qwen_sum_%j.out
#SBATCH --error=/home2020/home/icube/janil/Rhine_proj/logs/qwen_sum_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate vlm_env

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "===== Job node: $(hostname) ====="
echo "===== Allocated GPU: $CUDA_VISIBLE_DEVICES ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# python -u /home2020/home/icube/janil/Rhine_proj/scripts/code/0.1.Pdf_summary/0.summary.py > ~/Rhine_proj/outputs/live_logs/0.PDF_extraction_output/pdf_summary.txt 2>&1
# python -u /home2020/home/icube/janil/Rhine_proj/scripts/code/0.1.Pdf_summary/0.summary.py > ~/Rhine_proj/outputs/live_logs/0.PDF_extraction_output/pdf_summary_test_dataset.txt 2>&1

# python -u /home2020/home/icube/janil/Rhine_proj/scripts/code/0.1.Pdf_summary/1.Weighted_summary.py > ~/Rhine_proj/outputs/live_logs/0.PDF_extraction_output/weighted_pdf_summary.txt 2>&1
python -u /home2020/home/icube/janil/Rhine_proj/scripts/code/0.1.Pdf_summary/1.Weighted_summary.py > ~/Rhine_proj/outputs/live_logs/0.PDF_extraction_output/weighted_pdf_summary_test_dataset.txt 2>&1