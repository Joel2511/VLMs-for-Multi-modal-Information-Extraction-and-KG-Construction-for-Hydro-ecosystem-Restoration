#!/usr/bin/env bash
#SBATCH --job-name=rhine_demo_fichier
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home2020/home/icube/janil/github-thesis-edit/outputs/slurm_logs/demo_fichier_%j.out
#SBATCH --error=/home2020/home/icube/janil/github-thesis-edit/outputs/slurm_logs/demo_fichier_%j.err

set -euo pipefail

PROJECT_ROOT="/home2020/home/icube/janil/github-thesis-edit"
DOCUMENT="fichier"
RUN_ROOT="${PROJECT_ROOT}/outputs/runs/demo/${DOCUMENT}"

echo "======================================================================"
echo "RHINE END-TO-END ONE-DOCUMENT DEMO"
echo "======================================================================"
echo "Started: $(date)"
echo "Host: $(hostname)"
echo "Project root: ${PROJECT_ROOT}"
echo "Document: ${DOCUMENT}"
echo "Run root: ${RUN_ROOT}"
echo "======================================================================"

cd "${PROJECT_ROOT}"

# Safety guard: never mix this demonstration with a previous attempt.
if [[ -e "${RUN_ROOT}" ]]; then
    echo "ERROR: Demo output directory already exists:"
    echo "       ${RUN_ROOT}"
    echo
    echo "Nothing was run and no existing output was changed."
    echo "Rename/remove that incomplete demo folder only after inspecting it,"
    echo "or use a new run name in scripts/run_demo.sh."
    exit 1
fi

source "${HOME}/miniforge3/etc/profile.d/conda.sh"

echo
echo "========== SLURM GPU ALLOCATION =========="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-not-set}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not-set}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader \
    || true

echo
echo "========== CONDA ENVIRONMENTS =========="
conda env list | grep -E \
    'kg_env|barthez_env|vlm_env|nvila_env|qwen3vl_env' \
    || true

echo
echo "========== STARTING END-TO-END DEMO =========="

bash scripts/run_demo.sh "${DOCUMENT}"

echo
echo "======================================================================"
echo "DEMO JOB FINISHED SUCCESSFULLY"
echo "Finished: $(date)"
echo "Expected final HTML graph:"
echo "${RUN_ROOT}/knowledge_graph/html/${DOCUMENT}/interactive_graph.html"
echo "======================================================================"
