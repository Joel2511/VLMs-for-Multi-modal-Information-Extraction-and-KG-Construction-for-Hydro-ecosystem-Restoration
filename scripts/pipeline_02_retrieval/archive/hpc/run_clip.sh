#!/usr/bin/env bash
#SBATCH --job-name=pipeline02_clip
#SBATCH --output=outputs/runs/logs/pipeline_02_clip_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Pipeline 2 retrieval ablation: CLIP ViT-B/32.
# This is an archived baseline, not the selected VDR configuration.
# Adapt Slurm resources to the local cluster.

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

source "${CONDA_SH}"
conda activate barthez_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/retrieval"
mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/logs"

echo "Pipeline 2 retrieval ablation: CLIP ViT-B/32"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Slurm job ID: ${SLURM_JOB_ID:-local}"
nvidia-smi || true

python -u "${RHINE_PROJECT_ROOT}/scripts/pipeline_02_retrieval/run_clip.py" "$@" \
    2>&1 | tee "${RHINE_PROJECT_ROOT}/outputs/runs/logs/pipeline_02_clip_${SLURM_JOB_ID:-local}.txt"

echo "Finished: $(date)"
