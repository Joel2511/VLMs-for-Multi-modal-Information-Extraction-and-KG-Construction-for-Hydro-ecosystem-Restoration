#!/bin/bash

FOLDERS=(
    "qwen3vl-30b_captions_vidore"
)

for FOLDER in "${FOLDERS[@]}"; do

    JOBSCRIPT=/tmp/clip_pacs_job_${FOLDER}.sh

    cat > $JOBSCRIPT << ENDJOB
#!/bin/bash
#SBATCH --job-name=clip_${FOLDER:0:20}
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home2020/home/icube/janil/clip_pacs_${FOLDER}_%j.out

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"


source /home2020/home/icube/janil/miniforge3/etc/profile.d/conda.sh
conda activate vlm_env

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH

echo "===== Job node: \$(hostname) ====="
echo "===== Allocated GPU: \$CUDA_VISIBLE_DEVICES ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output
python -u $RHINE_PROJECT_ROOT/scripts/code/2.Image_description_Pipeline/temp_3.Image_captions_metrics.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/Context_OCR_ablation/${FOLDER}_clip_pacs.txt 2>&1
# python -u $RHINE_PROJECT_ROOT/scripts/code/2.Image_description_Pipeline/temp_3.Image_captions_metrics.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/${FOLDER}_ablation_clip_pacs.txt 2>&1
# python -u $RHINE_PROJECT_ROOT/scripts/code/2.Image_description_Pipeline/temp_3.Image_captions_metrics.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/Summary_ablation/${FOLDER}_summary_clip_pacs.txt 2>&1
# python -u $RHINE_PROJECT_ROOT/scripts/code/2.Image_description_Pipeline/temp_3.Image_captions_metrics.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/Weighted_blended_summary/${FOLDER}_weighted_summary_clip_pacs.txt 2>&1
ENDJOB

    sbatch $JOBSCRIPT
    echo "Submitted: $FOLDER"
done

echo "All ${#FOLDERS[@]} jobs submitted."