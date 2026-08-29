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
