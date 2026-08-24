#!/bin/bash

FOLDERS=(
    "internvl25-8b"
    # "llama32vision-11b"
    # "mistral-small-31-24b"
    # "nvila-lite-8b"
    # "pixtral-12b-hf"
    # "qwen36-27b"
    # "qwen3vl-30b"

    # "internvl25-8b_captions_vidore"
    # "llama32vision-11b_captions_vidore"
    # "mistral-small-31-24b_captions_vidore"
    # "nvila-lite-8b_captions_vidore"
    # "pixtral-12b-hf_captions_vidore"
    # "qwen36-27b_captions_vidore"
    # "qwen3vl-30b_captions_vidore"

    ##Ablation 5 files
    # "nvila_p0_baseline_captions"
    # "nvila_p1_primitive_captions"
    # "nvila_p2_refusal_captions"
    # "nvila_p3_contrast_captions"
    # "nvila_p4_selfcorrect_captions"
    # "nvila_p5_expert_captions"

    ## Ablation 6 files
    # "nvila-lite-8b_captions_vidore_icd_alpha05"
    # "nvila-lite-8b_captions_vidore_icd_alpha10"
    #  "nvila-lite-8b_captions_vidore_vcd_alpha05_t500"
    #  "nvila-lite-8b_captions_vidore_vcd_alpha10_t500"
)

for FOLDER in "${FOLDERS[@]}"; do

    JOBSCRIPT=/tmp/nli_job_${FOLDER}.sh

    cat > $JOBSCRIPT << ENDJOB
#!/bin/bash
#SBATCH --job-name=nli_${FOLDER:0:20}
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --exclude=hpc-n983
#SBATCH --time=04:00:00
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/outputs/live_logs/3.Metrics_output/1st_ablation/nli_vqa_${FOLDER}_%j.out
#SBATCH --error=/home2020/home/icube/janil/Rhine_proj/outputs/live_logs/3.Metrics_output/1st_ablation/nli_vqa_${FOLDER}_%j.err

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"


source /home2020/home/icube/janil/miniforge3/etc/profile.d/conda.sh
conda activate vlm_env
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -u $RHINE_PROJECT_ROOT/scripts/code/3.Metrics_Calculation/captioning_metrics_calculation.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/1st_ablation/${FOLDER}_no_prompt_ablation_vqa.txt 2>&1
# python -u $RHINE_PROJECT_ROOT/scripts/code/3.Metrics_Calculation/captioning_metrics_calculation.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/2nd_ablation/${FOLDER}_captioning_metrics.txt 2>&1
# python -u $RHINE_PROJECT_ROOT/scripts/code/3.Metrics_Calculation/captioning_metrics_calculation.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/3rd_ablation/${FOLDER}_summary_nli_vqa.txt 2>&1
# python -u $RHINE_PROJECT_ROOT/scripts/code/3.Metrics_Calculation/captioning_metrics_calculation.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/4th_ablation/${FOLDER}_weighted_summary_nli_vqa.txt 2>&1

# python -u $RHINE_PROJECT_ROOT/scripts/code/3.Metrics_Calculation/captioning_metrics_calculation.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/5th_ablation/${FOLDER}_prompt_nli_vqa.txt 2>&1
# python -u $RHINE_PROJECT_ROOT/scripts/code/3.Metrics_Calculation/captioning_metrics_calculation.py ${FOLDER} > $RHINE_PROJECT_ROOT/outputs/live_logs/3.Metrics_output/6th_ablation/${FOLDER}_decoder_nli_vqa.txt 2>&1
ENDJOB

    sbatch $JOBSCRIPT
    echo "Submitted: $FOLDER"
done

echo "All ${#FOLDERS[@]} jobs submitted."