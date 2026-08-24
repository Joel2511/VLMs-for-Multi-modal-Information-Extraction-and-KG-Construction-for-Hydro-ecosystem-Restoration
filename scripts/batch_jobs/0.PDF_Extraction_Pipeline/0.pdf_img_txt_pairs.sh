#!/bin/bash
#SBATCH --job-name=pdf_extraction
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/logs/slurm_extraction_%j.log
#SBATCH --partition=publicgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# ── Environment ──────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate kg_env

# ── NUMA-aware CPU pinning ────────────────────────────────────────────────────
# Bind to NUMA node 0 (even cores, affine to GPU0 which runs YOLO + OCR)
# This avoids cross-NUMA memory latency for GPU0-bound work
export NUMA_PIN="numactl --cpunodebind=0 --membind=0"

# ── PyTorch / CUDA tuning ─────────────────────────────────────────────────────
# Allow TF32 on V100 (Ampere+ only, no-op on V100 but harmless)
export TORCH_ALLOW_TF32=1
# Limit each process to GPU0 for this single-process script
# GPU1 is available for future multi-process pipelines
export CUDA_VISIBLE_DEVICES=0
# Deterministic CUDA memory allocator caching — reduces fragmentation on long jobs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# ── OpenMP / MKL threading ────────────────────────────────────────────────────
# Match thread count to Slurm allocation to avoid oversubscription / libgomp errors
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# Let OpenMP choose cores; avoid invalid affinity lists across nodes
unset GOMP_CPU_AFFINITY

# ── Parallel image I/O ────────────────────────────────────────────────────────
# PIL/Pillow uses this for PNG compression workers
export PILLOW_WORKERS=8

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPUs allocated by Slurm: ${SLURM_GPUS_ON_NODE:-unknown}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,index,memory.total --format=csv,noheader || echo "nvidia-smi not available"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

$NUMA_PIN python -u ~/Rhine_proj/scripts/code/0.PDF_Extraction_Pipeline/pdf_extraction.py \
    > ~/Rhine_proj/outputs/live_logs/0.PDF_extraction_output/test_dataset_output.txt 2>&1

echo "Job finished: $(date)"