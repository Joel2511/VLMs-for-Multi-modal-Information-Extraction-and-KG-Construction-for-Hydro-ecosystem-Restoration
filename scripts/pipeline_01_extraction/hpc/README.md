# Pipeline 1 HPC Launcher

This folder contains Slurm/HPC execution support for Pipeline 1.

- `run_extraction.sh` launches document extraction on an HPC GPU node.
- The active extraction implementation is
  `../extract_document_content.py`.
- The script should be adapted to the target scheduler, partition names, GPU
  availability, and local Conda installation.

Pipeline 1 converts input PDFs into extracted figure images and structured text
JSON files used by the downstream retrieval and captioning stages.
