# Fine-Tuning Workflow

This folder contains the full reproducibility workflow used to create the
fine-tuned Pipeline 4 Qwen2.5-14B checkpoint.

## Workflow order

Run the stages in this order:

1. `generate_synthetic_dataset.py`
2. `validate_synthetic_dataset.py`
3. `build_hf_dataset.py`
4. `fine_tune_lora.py`
5. `merge_lora_adapter.py`

## Generated artifacts

The scripts write generated artifacts to:

```text
outputs/runs/fine_tuning/
```

This includes the synthetic dataset, cleaned dataset, Hugging Face dataset,
LoRA adapter checkpoints, merged model output, and run logs.

These artifacts are generated at runtime and are intentionally not tracked in
Git. The merged checkpoint used for the reported Pipeline 4 results is
distributed separately; see `MODEL_SETUP.md`.

## Requirements

- The Qwen2.5-14B-Instruct base model must be available under `HF_MODELS_DIR`.
- Run the workflow in the documented Qwen/Pipeline 4 Conda environment.
- Use a suitable GPU-equipped HPC allocation.
- The local output location can be overridden with
  `FINE_TUNING_OUTPUT_ROOT`.

See the `hpc/` directory for Slurm launchers.
