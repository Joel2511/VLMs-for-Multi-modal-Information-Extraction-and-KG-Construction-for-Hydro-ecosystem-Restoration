#!/usr/bin/env bash
#SBATCH --job-name=pipeline01_extraction
#SBATCH --output=outputs/runs/logs/pipeline_01_extraction_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# =============================================================================
# Pipeline 1: Layout Parsing and Cross-Modal Extraction
#
# This Slurm script runs the PDF extraction workflow, including layout
# detection, figure cropping, text-block extraction, and OCR-based filtering.
#
# Adapt Slurm resources above to the policies and hardware of the local cluster.
# In particular, add a partition, account, GPU type, or time limit if required.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -f "${DEFAULT_PROJECT_ROOT}/config/project.env" ]]; then
    # shellcheck disable=SC1091
    source "${DEFAULT_PROJECT_ROOT}/config/project.env"
fi

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
export HF_MODELS_DIR="${HF_MODELS_DIR:-${RHINE_PROJECT_ROOT}/hf_models}"
export CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
export NUMA_PREFIX="${NUMA_PREFIX:-}"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: Conda initialization script not found: ${CONDA_SH}"
    echo "Set CONDA_SH in config/project.env."
    exit 1
fi

source "${CONDA_SH}"
conda activate kg_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PILLOW_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_ALLOW_TF32="${TORCH_ALLOW_TF32:-1}"

mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/extraction"
mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/logs"

echo "============================================================"
echo "Pipeline 1: Layout Parsing and Cross-Modal Extraction"
echo "Job started: $(date)"
echo "Project root: ${RHINE_PROJECT_ROOT}"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID:-local}"
echo "Allocated node list: ${SLURM_NODELIST:-local}"
echo "Allocated GPUs: ${SLURM_GPUS_ON_NODE:-not-reported}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-not-reported}"
echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Python: $(python --version)"
echo "============================================================"

nvidia-smi --query-gpu=name,index,memory.total --format=csv,noheader \
    || echo "nvidia-smi not available"

if [[ -n "${NUMA_PREFIX}" ]]; then
    ${NUMA_PREFIX} python -u \
        "${RHINE_PROJECT_ROOT}/scripts/pipeline_01_extraction/extract_document_content.py" \
        "$@" \
        2>&1 | tee "${RHINE_PROJECT_ROOT}/outputs/runs/logs/pipeline_01_extraction_${SLURM_JOB_ID:-local}.txt"
else
    python -u \
        "${RHINE_PROJECT_ROOT}/scripts/pipeline_01_extraction/extract_document_content.py" \
        "$@" \
        2>&1 | tee "${RHINE_PROJECT_ROOT}/outputs/runs/logs/pipeline_01_extraction_${SLURM_JOB_ID:-local}.txt"
fi

echo "Job finished: $(date)"