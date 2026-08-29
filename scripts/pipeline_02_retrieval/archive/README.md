# Pipeline 2 Archived Retrieval Ablations

This directory contains the non-selected retrieval implementations and Slurm
wrappers evaluated during the Pipeline 2 retrieval benchmark.

The selected retrieval configuration is VDR 2B Multi:

```text
llamaindex/vdr-2b-multi-v1
```

The selected implementation remains in the parent directory:

```text
../run_vdr.py
../hpc/run_vdr.sh
```

## Archived retrieval implementations

| Model | Python implementation | HPC wrapper |
|---|---|---|
| CLIP ViT-B/32 | `retrieval_ablations/run_clip.py` | `hpc/run_clip.sh` |
| ColPali v1.2 | `retrieval_ablations/run_colpali.py` | `hpc/run_colpali.sh` |
| ColQwen2 v1.0 | `retrieval_ablations/run_colqwen2.py` | `hpc/run_colqwen2.sh` |
| DSE-Qwen2 2B | `retrieval_ablations/run_dse_qwen2.py` | `hpc/run_dse_qwen2.sh` |
| Nemotron ColEmbed 4B | `retrieval_ablations/run_nemotron.py` | `hpc/run_nemotron.sh` |

These files are retained to document the retrieval ablation study reported in
the thesis. They are not required for the recommended end-to-end workflow.
