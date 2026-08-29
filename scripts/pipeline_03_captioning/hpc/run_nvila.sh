#!/usr/bin/env bash
#SBATCH --job-name=pipeline03_nvila
#SBATCH --output=outputs/runs/logs/pipeline_03_nvila_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00

# =============================================================================
# Pipeline 3: NVILA-Lite-8B captioning experiments.
#
# Usage:
#   sbatch run_nvila.sh ablation_01
#   sbatch run_nvila.sh ablation_02
#   sbatch run_nvila.sh ablation_03
#   sbatch run_nvila.sh ablation_04
#   sbatch run_nvila.sh ablation_05_p0
#   sbatch run_nvila.sh ablation_05_p1
#   sbatch run_nvila.sh ablation_05_p2
#   sbatch run_nvila.sh ablation_05_p3
#   sbatch run_nvila.sh ablation_05_p4
#   sbatch run_nvila.sh ablation_05_p5
#   sbatch run_nvila.sh ablation_06_icd_mild
#   sbatch run_nvila.sh ablation_06_icd_strong
#   sbatch run_nvila.sh ablation_06_vcd_mild
#   sbatch run_nvila.sh ablation_06_vcd_strong
#
# The selected thesis configuration is:
#   ablation_05_p3  (Counterfactual Contrast)
#
# Adapt Slurm resources to the local cluster when necessary.
# =============================================================================

set -euo pipefail

ABLATION="${1:-ablation_05_p3}"

if [[ $# -gt 0 ]]; then
    shift
fi

FORWARD_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -f "${DEFAULT_PROJECT_ROOT}/config/project.env" ]]; then
    # shellcheck disable=SC1091
    source "${DEFAULT_PROJECT_ROOT}/config/project.env"
fi

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
export HF_MODELS_DIR="${HF_MODELS_DIR:-${RHINE_PROJECT_ROOT}/hf_models}"
export CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: Conda initialization script not found: ${CONDA_SH}"
    echo "Set CONDA_SH in config/project.env."
    exit 1
fi

source "${CONDA_SH}"
conda activate nvila_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
fi

mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/captioning"
mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/logs"

case "${ABLATION}" in
    ablation_01)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_01_no_prompt/run_nvila.py"
        ;;
    ablation_02)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_02_retrieval_context/run_nvila.py"
        ;;
    ablation_03)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_03_document_summary/run_nvila.py"
        ;;
    ablation_04)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_04_weighted_context/run_nvila.py"
        ;;
    ablation_05_p0)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_05_prompt_engineering/run_p0_baseline.py"
        ;;
    ablation_05_p1)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_05_prompt_engineering/run_p1_visual_primitives.py"
        ;;
    ablation_05_p2)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_05_prompt_engineering/run_p2_verification_refusal.py"
        ;;
    ablation_05_p3)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_05_prompt_engineering/run_p3_counterfactual_contrast.py"
        ;;
    ablation_05_p4)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_05_prompt_engineering/run_p4_self_correction.py"
        ;;
    ablation_05_p5)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_05_prompt_engineering/run_p5_domain_expert.py"
        ;;
    ablation_06_icd_mild)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_06_contrastive_decoding/run_instruction_contrastive_decoding.py"
        EXTRA_ARGS=(--cd_alpha 0.5)
        ;;
    ablation_06_icd_strong)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_06_contrastive_decoding/run_instruction_contrastive_decoding.py"
        EXTRA_ARGS=(--cd_alpha 1.0)
        ;;
    ablation_06_vcd_mild)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_06_contrastive_decoding/run_visual_contrastive_decoding.py"
        EXTRA_ARGS=(--cd_alpha 0.5 --noise_step 500)
        ;;
    ablation_06_vcd_strong)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_06_contrastive_decoding/run_visual_contrastive_decoding.py"
        EXTRA_ARGS=(--cd_alpha 1.0 --noise_step 500)
        ;;
    *)
        echo "ERROR: Unknown NVILA configuration: ${ABLATION}"
        echo "Supported values:"
        echo "  ablation_01, ablation_02, ablation_03, ablation_04"
        echo "  ablation_05_p0 ... ablation_05_p5"
        echo "  ablation_06_icd_mild, ablation_06_icd_strong"
        echo "  ablation_06_vcd_mild, ablation_06_vcd_strong"
        exit 1
        ;;
esac

EXTRA_ARGS=("${EXTRA_ARGS[@]:-}")

echo "============================================================"
echo "Pipeline 3: NVILA-Lite-8B captioning"
echo "Configuration: ${ABLATION}"
echo "Python script: ${PYTHON_SCRIPT}"
echo "Job started: $(date)"
echo "Project root: ${RHINE_PROJECT_ROOT}"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID:-local}"
echo "Allocated GPUs: ${SLURM_GPUS_ON_NODE:-not-reported}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "============================================================"

nvidia-smi || echo "nvidia-smi not available"

python -u \
    "${RHINE_PROJECT_ROOT}/${PYTHON_SCRIPT}" \
    "${EXTRA_ARGS[@]}" \
    "${FORWARD_ARGS[@]}" \
    2>&1 | tee "${RHINE_PROJECT_ROOT}/outputs/runs/logs/pipeline_03_nvila_${ABLATION}_${SLURM_JOB_ID:-local}.txt"

echo "Job finished: $(date)"