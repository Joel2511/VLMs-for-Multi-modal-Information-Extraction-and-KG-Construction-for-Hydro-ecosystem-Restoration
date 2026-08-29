#!/usr/bin/env bash
#SBATCH --job-name=pipeline03_caption_metrics
#SBATCH --output=outputs/runs/logs/pipeline_03_caption_metrics_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

# =============================================================================
# Pipeline 3 archived evaluation: caption quality metrics.
#
# This script evaluates caption outputs using the corresponding implementation
# in scripts/evaluation/evaluate_captioning.py. It supports evaluation runs
# after any captioning ablation.
#
# Usage:
#   sbatch evaluate_caption_metrics.sh <caption-output-folder>
#
# Example:
#   sbatch evaluate_caption_metrics.sh nvila_p3_contrast_captions
#
# The folder argument is passed to the Python evaluation implementation.
# Adapt Slurm resources to the local cluster when necessary.
# =============================================================================

set -euo pipefail

CAPTION_FOLDER="${1:-}"

if [[ -z "${CAPTION_FOLDER}" ]]; then
    echo "ERROR: Provide a caption output folder."
    echo "Usage: sbatch evaluate_caption_metrics.sh <caption-output-folder>"
    exit 1
fi

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
conda activate vlm_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/metrics"
mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/logs"

echo "============================================================"
echo "Pipeline 3: Caption-quality metric evaluation"
echo "Caption output folder: ${CAPTION_FOLDER}"
echo "Job started: $(date)"
echo "Project root: ${RHINE_PROJECT_ROOT}"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID:-local}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "============================================================"

nvidia-smi || echo "nvidia-smi not available"

python -u \
    "${RHINE_PROJECT_ROOT}/scripts/evaluation/evaluate_captioning.py" \
    "${CAPTION_FOLDER}" \
    2>&1 | tee "${RHINE_PROJECT_ROOT}/outputs/runs/logs/pipeline_03_caption_metrics_${CAPTION_FOLDER}_${SLURM_JOB_ID:-local}.txt"

echo "Job finished: $(date)"