#!/bin/bash
#SBATCH --job-name=kg_triple_extraction
#SBATCH --partition=publicgpu
#SBATCH --nodelist=hpc-n981
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/outputs/live_logs/4.Triples_output_ablation/kg_job_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate qwen3vl_env

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false

mkdir -p /home2020/home/icube/janil/Rhine_proj/outputs/live_logs/4.Triples_output_ablation

python -c "import torch; print('[GPU CHECK]', torch.cuda.device_count(), 'GPUs visible', flush=True)"

# # baseline on GPU 0
# CUDA_VISIBLE_DEVICES=0 python -u /home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/baseline_triples_conversion.py \
#     > /home2020/home/icube/janil/Rhine_proj/outputs/live_logs/4.Triples_output_ablation/triples_baseline.txt 2>&1 &
# BASELINE_PID=$!

# # fine-tuned on GPU 1
# CUDA_VISIBLE_DEVICES=1 python -u /home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuned_triples_conversion.py \
#     > /home2020/home/icube/janil/Rhine_proj/outputs/live_logs/4.Triples_output_ablation/fine_tuned_triples.txt 2>&1 &
# FT_PID=$!

# fine-tuned on GPU 1
CUDA_VISIBLE_DEVICES=1 python -u /home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuned_triples_conversion.py \
    > /home2020/home/icube/janil/Rhine_proj/outputs/live_logs/4.Triples_output_ablation/fine_tuned_triples_test_dataset.txt 2>&1 &
FT_PID=$!

# wait $BASELINE_PID
# BASELINE_EXIT=$?
wait $FT_PID
FT_EXIT=$?

# echo "[DONE] Baseline exit code: $BASELINE_EXIT"
echo "[DONE] Fine-tuned exit code: $FT_EXIT"