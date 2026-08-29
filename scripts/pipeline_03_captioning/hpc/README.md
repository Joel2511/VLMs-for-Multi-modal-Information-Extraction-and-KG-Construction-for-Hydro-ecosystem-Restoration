# Pipeline 3 HPC Launchers

This folder contains Slurm/HPC launchers for active Pipeline 3 operations.

- `run_document_summarization.sh` launches document summarisation.
- `build_weighted_context.sh` builds weighted per-image context.
- `run_nvila.sh` launches the selected NVILA captioning workflow.

These scripts require the documented Conda environments and local model paths.
Adapt scheduler directives such as partition, memory, GPU count, and time limit
to the target HPC system.
