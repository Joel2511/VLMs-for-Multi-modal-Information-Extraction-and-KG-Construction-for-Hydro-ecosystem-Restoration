#!/usr/bin/env bash
#SBATCH --job-name=pipeline04_triples
#SBATCH --output=outputs/runs/logs/pipeline_04_triples_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# =============================================================================
# Pipeline 4: Cross-Modal Knowledge-Graph Construction.
#
# Usage:
#   sbatch run_triple_extraction.sh baseline
#   sbatch run_triple_extraction.sh fine_tuned
#
# The baseline uses Qwen2.5-14B-Instruct.
# The fine-tuned mode uses the separately hosted merged checkpoint or a locally
# reproduced checkpoint, as documented in MODEL_SETUP.md.
# =============================================================================

set -euo pipefail

MODE="${1:-fine_tuned}"

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
conda activate qwen3vl_env

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"

if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
fi

mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/knowledge_graph"
mkdir -p "${RHINE_PROJECT_ROOT}/outputs/runs/logs"

case "${MODE}" in
    baseline)
        PYTHON_SCRIPT="scripts/pipeline_04_knowledge_graph/extract_triples_baseline.py"
        ;;
    fine_tuned)
        PYTHON_SCRIPT="scripts/pipeline_04_knowledge_graph/extract_triples_finetuned.py"
        ;;
    *)
        echo "ERROR: Unknown Pipeline 4 mode: ${MODE}"
        echo "Supported values: baseline, fine_tuned"
        exit 1
        ;;
esac

echo "============================================================"
echo "Pipeline 4: Cross-Modal Knowledge-Graph Construction"
echo "Mode: ${MODE}"
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
    2>&1 | tee "${RHINE_PROJECT_ROOT}/outputs/runs/logs/pipeline_04_${MODE}_${SLURM_JOB_ID:-local}.txt"

echo "Job finished: $(date)"