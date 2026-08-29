# Pipeline 2 HPC Launcher

This folder contains the active Slurm/HPC launcher for the selected retrieval
configuration.

- `run_vdr.sh` launches ViDoRe VDR image–text retrieval.
- The active retrieval implementation is `../run_vdr.py`.
- VDR requires the local `vdr-2b-multi-v1` model directory under
  `HF_MODELS_DIR`.

Alternative retrieval implementations are retained under `../archive/` because
they were evaluated during the thesis but are not part of the selected
end-to-end demonstration.
