# Pipeline 1: Layout Parsing and Cross-Modal Extraction

This directory contains the first stage of the thesis framework. Pipeline 1
converts each source PDF into structured text blocks, extracted figure crops,
and OCR output while retaining document-level organization.

## Main implementation

| File | Purpose |
|---|---|
| `extract_document_content.py` | Runs PDF text extraction, 300-DPI page rasterization, DocLayout-YOLO layout detection, figure merging, caption association, structural-noise filtering, OCR, and OCR-based cross-modal deduplication |
| `hpc/run_extraction.sh` | Slurm/HPC wrapper for the extraction implementation |

## Inputs

By default, the script reads PDFs from:

```text
data/raw_pdfs/
```

## Outputs

New runs write to:

```text
outputs/runs/extraction/
├── extracted_images/<document-name>/
└── json_results/<document-name>/
    ├── extracted_content.json
    └── ocr_results.json
```

The original thesis artifacts remain available separately under:

```text
data/extracted_images/
outputs/json_results/
```

They are not overwritten by the public execution path.

## Usage

Process all available PDFs:

```bash
sbatch scripts/pipeline_01_extraction/hpc/run_extraction.sh
```

Process one PDF:

```bash
sbatch scripts/pipeline_01_extraction/hpc/run_extraction.sh \
  --document <pdf-filename-or-stem>
```

For model setup, see [`../../MODEL_SETUP.md`](../../MODEL_SETUP.md).
For cluster-specific configuration, copy `config/project.env.example` to
`config/project.env` and edit it for the local environment.
