# Pipeline 3: Grounded Visual Captioning

This directory contains the captioning pipeline, its document-context
preparation steps, and the prompt/context/decoding ablations evaluated in the
thesis.

The selected configuration is:

```text
NVILA-Lite-8B + Ablation 5 P3 Counterfactual Contrast
```

## Execution order

For the selected configuration, execute the stages in this order:

```text
Pipeline 1 extraction
    ↓
Pipeline 2 VDR retrieval
    ↓
Document summarization
    ↓
Weighted-context construction
    ↓
NVILA A5-P3 Counterfactual Contrast captioning
    ↓
NLI and VQA-consistency evaluation
```

## Directory map

| Directory | Purpose |
|---|---|
| `context_preparation/` | MapReduce document summarization and weighted local/document-context fusion |
| `ablation_01_no_prompt/` | Seven-VLM no-prompt baseline |
| `ablation_02_retrieval_context/` | Structured prompt with retrieved local context |
| `ablation_03_document_summary/` | Parallel document-summary context injection |
| `ablation_04_weighted_context/` | Approximately 70/30 local/document weighted-context fusion |
| `ablation_05_prompt_engineering/` | NVILA prompt-engineering comparison, including selected P3 Counterfactual Contrast |
| `ablation_06_contrastive_decoding/` | NVILA instruction- and visual-contrastive-decoding experiments |
| `hpc/` | Recommended Pipeline 3 Slurm/HPC wrappers |
| `archive/` | Non-selected model-specific HPC wrappers and obsolete experimental utilities |

## Selected scripts

| File | Purpose |
|---|---|
| `context_preparation/summarize_documents.py` | Generates MapReduce summaries for source documents |
| `context_preparation/build_weighted_context.py` | Combines retrieved local blocks with document summaries |
| `ablation_05_prompt_engineering/run_p3_counterfactual_contrast.py` | Selected NVILA Counterfactual Contrast captioning configuration |
| `hpc/run_document_summarization.sh` | Slurm wrapper for document summarization |
| `hpc/build_weighted_context.sh` | Slurm wrapper for weighted-context construction |
| `hpc/run_nvila.sh` | Slurm wrapper for NVILA captioning configurations |
| `hpc/evaluate_question_caption_metrics.sh` | NLI and VQA-consistency evaluation wrapper |

## Inputs and outputs

The selected public workflow supports both archived artifacts and fresh runs.

Archived thesis inputs include:

```text
data/extracted_images/
outputs/json_results/
outputs/Summarized_pdfs/
outputs/Weighted_summary/
outputs/Image_text_matching/vidore_results/
```

New runs write to:

```text
outputs/runs/summarized_pdfs/
outputs/runs/weighted_context/
outputs/runs/captioning/
outputs/runs/metrics/
```

This separation prevents new executions from overwriting the committed thesis
artifacts.

## NVILA modes

The NVILA wrapper supports the following modes:

```text
ablation_01
ablation_02
ablation_03
ablation_04
ablation_05_p0
ablation_05_p1
ablation_05_p2
ablation_05_p3
ablation_05_p4
ablation_05_p5
ablation_06_icd_mild
ablation_06_icd_strong
ablation_06_vcd_mild
ablation_06_vcd_strong
```

The selected thesis configuration is:

```bash
sbatch scripts/pipeline_03_captioning/hpc/run_nvila.sh ablation_05_p3
```

The Python scripts must first receive matching input/output arguments when
running the fresh one-document demonstration.
