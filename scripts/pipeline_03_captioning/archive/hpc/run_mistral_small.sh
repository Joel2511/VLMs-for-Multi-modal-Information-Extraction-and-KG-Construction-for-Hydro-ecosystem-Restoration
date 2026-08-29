#!/usr/bin/env bash
#SBATCH --job-name=pipeline03_mistral_small
#SBATCH --output=outputs/runs/logs/pipeline_03_mistral_small_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00

# =============================================================================
# Pipeline 3 archived captioning ablation: Mistral-Small-3.1-24B.
#
# Usage:
#   sbatch run_mistral_small.sh ablation_01
#   sbatch run_mistral_small.sh ablation_02
#   sbatch run_mistral_small.sh ablation_03
#   sbatch run_mistral_small.sh ablation_04
# =============================================================================

set -euo pipefail

ABLATION="${1:-ablation_02}"

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

case "${ABLATION}" in
    ablation_01)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_01_no_prompt/run_mistral_small.py"
        ;;
    ablation_02)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_02_retrieval_context/run_mistral_small.py"
        ;;
    ablation_03)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_03_document_summary/run_mistral_small.py"
        ;;
    ablation_04)
        PYTHON_SCRIPT="scripts/pipeline_03_captioning/ablation_04_weighted_context/run_mistral_small.py"
        ;;
    *)
        echo "ERROR: Mistral-Small supports: ablation_01, ablation_02, ablation_03, ablation_04"
        exit 1
        ;;
esac

mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/captioning"
mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/logs"

echo "Pipeline 3 archived captioning ablation: Mistral-Small-3.1-24B"
echo "Configuration: ${ABLATION}"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID:-local}"
nvidia-smi || true

python -u "${RHINE_PROJECT_ROOT}/${PYTHON_SCRIPT}" \
    2>&1 | tee "${RHINE_PROJECT_ROOT}/outputs/runs/logs/pipeline_03_mistral_small_${ABLATION}_${SLURM_JOB_ID:-local}.txt"

echo "Finished: $(date)"