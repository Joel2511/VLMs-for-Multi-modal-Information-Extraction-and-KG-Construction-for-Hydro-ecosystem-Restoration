# Pipeline 2: Cross-Modal Image-Text Retrieval

This directory contains the image-text retrieval implementations evaluated in
the thesis. Given figure crops and extracted text blocks from Pipeline 1, the
retrieval stage identifies the most relevant text blocks for each image.

## Selected configuration

The selected retrieval configuration is VDR 2B Multi:

```text
llamaindex/vdr-2b-multi-v1
```

| File | Purpose |
|---|---|
| `run_vdr.py` | Selected VDR retrieval implementation |
| `hpc/run_vdr.sh` | Slurm/HPC wrapper for VDR |

The VDR implementation preserves the thesis retrieval configuration, including
dynamic-resolution image preprocessing, single-vector embedding extraction,
hybrid page-proximity re-ranking, and the reported \( \beta = 0.02 \) penalty.

## Inputs

Pipeline 2 requires:

```text
<image-root>/<document-name>/
<json-root>/<document-name>/extracted_content.json
```

The default inputs are the archived extraction artifacts:

```text
data/extracted_images/
outputs/json_results/
```

For a fresh Pipeline 1 run, pass:

```text
--image-root outputs/runs/extraction/extracted_images
--json-root outputs/runs/extraction/json_results
```

## Outputs

New VDR retrieval runs write to:

```text
outputs/runs/retrieval/vidore_results/<document-name>/
├── vidore_raw_matrix.csv
├── vidore_img_embeddings.npy
└── vidore_similarity_histogram.png
```

The archived thesis retrieval outputs remain under:

```text
outputs/Image_text_matching/vidore_results/
```

## Archived alternatives

The `archive/` directory contains Slurm wrappers for the non-selected
retrieval configurations evaluated in the thesis:

- CLIP ViT-B/32
- ColPali v1.2
- ColQwen2 v1.0
- DSE-Qwen2 2B
- Nemotron ColEmbed 4B

These alternatives are retained for ablation traceability. They are not needed
for the recommended end-to-end workflow.
