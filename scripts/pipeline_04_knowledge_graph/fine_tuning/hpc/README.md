# Fine-Tuning HPC Launchers

This folder contains Slurm launchers for the fine-tuning workflow.

| Script | Stage |
|---|---|
| `generate_synthetic_dataset.sh` | Generate synthetic ontology-population training examples |
| `run_fine_tuning.sh` | Train the Qwen2.5-14B LoRA adapter |
| `merge_lora_adapter.sh` | Merge the trained adapter into the base model |

Before running, ensure the required model is available through `HF_MODELS_DIR`,
activate or configure the correct Conda environment, and adapt the Slurm
resource directives to the target HPC system.

Generated datasets, adapters, and merged checkpoints are written under
`outputs/runs/fine_tuning/` and are not tracked in Git.
