# Pipeline 4 HPC Launcher

This folder contains the Slurm launcher for Pipeline 4 knowledge-graph
extraction.

- `run_triple_extraction.sh` launches triple extraction on an HPC GPU node.
- The active extraction implementations are in the parent directory:
  - `extract_triples_baseline.py`
  - `extract_triples_finetuned.py`

The fine-tuned merged model checkpoint is not stored in Git. Download and
verify it as described in `MODEL_SETUP.md`, then place it under
`HF_MODELS_DIR` before running the fine-tuned configuration.
