#!/usr/bin/env bash
#SBATCH --job-name=pipeline04_merge_lora
#SBATCH --output=outputs/runs/logs/pipeline_04_merge_lora_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Pipeline 4 advanced workflow: merge the trained LoRA adapter into Qwen2.5-14B.
# This script is not required when using the separately hosted merged model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ -f "${DEFAULT_PROJECT_ROOT}/config/project.env" ]]; then
    # shellcheck disable=SC1091
    source "${DEFAULT_PROJECT_ROOT}/config/project.env"
fi

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
export CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"

source "${CONDA_SH}"
conda activate qwen3vl_env

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/logs"

echo "Pipeline 4: LoRA adapter merge"
echo "Started: $(date)"
echo "Node: $(hostname)"
nvidia-smi || true

python -u \
    "${RHINE_PROJECT_ROOT}/scripts/pipeline_04_knowledge_graph/fine_tuning/merge_lora_adapter.py" \
    2>&1 | tee "${RHINE_PROJECT_ROOT}/outputs/runs/logs/pipeline_04_merge_lora_${SLURM_JOB_ID:-local}.txt"

echo "Finished: $(date)"