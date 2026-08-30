# Scripts

This directory contains the implementation of the thesis workflow, organised by
pipeline.

## Start here

`README.md` explains the research workflow, required data, model setup, and the recommended reading order.

For local model placement, model download instructions, checksum verification,
and the NVILA patch, see `MODEL_SETUP.md`.

For Conda environment creation and the mapping between scripts and environments,
see `environments/README.md`.

## Selected end-to-end demonstration

The selected reproducibility workflow is:

```text
PDF extraction
→ VDR image–text retrieval
→ document summary
→ weighted context construction
→ NVILA-Lite-8B A5-P3 captioning
→ fine-tuned Qwen2.5-14B knowledge-graph extraction
→ GEXF conversion
→ interactive HTML graph
```

- `scripts/run_demo.sh` runs the one-document demonstration sequentially.
- `scripts/submit_demo_fichier.sh` is the tested Slurm wrapper for the example document
  named `fichier`.

The demonstration writes new derived results under:

```text
outputs/runs/demo/<document-name>/
```

It does not overwrite the historical thesis outputs stored elsewhere under
`outputs/`.

## Running the demonstration

After completing the environment and model setup described in
`MODEL_SETUP.md` and `environments/README.md`, run the tested Slurm
demonstration from the repository root:

```bash
sbatch scripts/submit_demo_fichier.sh
```

This submits one GPU job for the included example document, `fichier`. The
batch script allocates the tested resources and then invokes:

```bash
bash scripts/run_demo.sh fichier
```

Do not run the Slurm wrapper with `bash`; submit it with `sbatch`.

To monitor the submitted job:

```bash
squeue -u "$USER"
```

The Slurm standard output and error logs are written to:

```text
outputs/slurm_logs/demo_fichier_<job-id>.out
outputs/slurm_logs/demo_fichier_<job-id>.err
```

The demonstration creates a new isolated run directory:

```text
outputs/runs/demo/fichier/
```

When the job completes successfully, open the resulting interactive
knowledge graph at:

```text
outputs/runs/demo/fichier/knowledge_graph/html/fichier/interactive_graph.html
```

To run the same selected workflow for another PDF already placed in
`data/raw_pdfs/`, use `scripts/run_demo.sh` with that PDF filename or stem
inside a suitable GPU allocation. For example:

```bash
bash scripts/run_demo.sh example_document.pdf
```

For the tested HPC configuration, use `scripts/submit_demo_fichier.sh` as the
reference when creating a corresponding Slurm submission script for another
document. The wrapper requests one GPU, 8 CPU cores, 80 GB RAM, and a 12-hour
wall-time limit.
## Directory map

| Directory | Purpose |
|---|---|
| `pipeline_01_extraction/` | PDF layout parsing, figure extraction, OCR, and text-block extraction |
| `pipeline_02_retrieval/` | Image–text retrieval, with VDR as the selected configuration |
| `pipeline_03_captioning/` | Captioning workflow, context construction, and captioning ablations |
| `pipeline_04_knowledge_graph/` | Baseline/fine-tuned triple extraction, RDF graphs, and fine-tuning workflow |
| `evaluation/` | Evaluation and configuration-ranking utilities |

## Important

Large model artifacts are not tracked in Git. After cloning, create the
required Conda environments, download or place the required model directories,
and set `HF_MODELS_DIR` as documented in `MODEL_SETUP.md` before executing the
workflow.
