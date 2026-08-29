# Rhine River Restoration: Multimodal Knowledge Extraction and Knowledge-Graph Construction

**Master Thesis Research | ICube Laboratory, Université de Strasbourg, Big Data Management and Analytics(BDMA)**  
**TETRA Project | Franco-German AI Toolbox for the Water Domain**

## Supervisors

- Dr. Franco Giustozzi — INSA Strasbourg
- Dr. Florence Le Ber — ENGEES, Université de Strasbourg

---

## Overview

This repository accompanies a master thesis conducted at the ICube Laboratory within the Franco-German TETRA project. TETRA develops AI-based methodologies and tools for water-domain applications through collaboration among Fraunhofer IOSB, ICube, SEBA, and Thales.

The project extracts, structures, and organizes restoration knowledge from heterogeneous French technical and institutional documents. These sources include technical reports, maps, site photographs, diagrams, graphs, and project plans relating primarily to Rhine-basin restoration and hydro-ecological management.

The implemented framework integrates visual and textual document content through vision-language models (VLMs). It extends prior text-based ontology-population work by extracting visual evidence, linking it to relevant document context, generating grounded captions, and integrating text- and image-derived information into an ontology-conditioned knowledge graph.

The full methodology, experimental design, results, and limitations are documented in [`Thesis.pdf`](Thesis.pdf).

---

## Research Pipelines

The framework consists of four dependent pipelines.

### Pipeline 1 : Layout Parsing and Cross-Modal Extraction

Source PDFs are rasterized at 300 DPI and processed with DocLayout-YOLOv10 to detect figures, captions, text regions, and structural margins. Geometric block merging reconstructs multi-panel figures, and detected captions are associated with their figures. A layout-aware text parser, cross-page boilerplate filtering, and OCR-based cross-modal deduplication produce structured figure crops and cleaned text blocks for each document.

### Pipeline 2 : Cross-Modal Image-Text Retrieval

For each extracted figure, the retrieval stage identifies the most relevant text blocks from the same source document. The repository contains six evaluated retrieval configurations spanning single-vector dense retrieval and multi-vector late interaction:

The selected retrieval configuration for the downstream pipeline is **VDR 2B Multi** (`llamaindex/vdr-2b-multi-v1`). Retrieval evaluation uses Recall@5, Precision@1, MRR@5, and nDCG@5 against manually annotated image-text associations.

### Pipeline 3 — Grounded Visual Captioning

A generative VLM describes each extracted image through a two-phase, pixel-first prompting procedure:

- **Phase 1:** image-only visual observation, including visible components and legible text.
- **Phase 2:** controlled contextual enrichment using retrieved local context and document-level summary information; it retains only image-visible information that is absent from the retrieved text.

Seven candidate VLMs and several context, prompt-engineering, and contrastive-decoding ablations are included. Caption evaluation includes CLIPScore, PAC-S, NLI-based novelty/grounding, VQA-Consistency, and a weighted multi-criteria selection score.

### Pipeline 4 — Cross-Modal Knowledge-Graph Construction

Textual content and image captions are converted into structured JSON triples and deterministically serialized as RDF using the `TetraOnto.ttl` OWL ontology. The extraction prompt receives a compact bilingual schema representation to encourage use of authorized ontology classes and predicates.

Text and images are processed separately before cross-modal entity resolution merges aliases and duplicates into canonical entities. The repository includes both a zero-shot Qwen2.5-14B-Instruct baseline and a LoRA fine-tuned variant evaluated for ontology-predicate conformance. Pipeline 4 evaluation includes Schema Conformance Rate (SCR), Average Node Degree Shift, Cross-Modal Link Density (CMLD), and a prepared manual intrinsic triple-precision audit protocol.

---

## Selected Configuration

The recommended deployment path follows the selected configurations from the thesis evaluation:

| Stage | Selected configuration |
|---|---|
| Layout parsing | 300-DPI rasterization with DocLayout-YOLOv10 and layout-aware post-processing |
| Image-text retrieval | VDR 2B Multi (`llamaindex/vdr-2b-multi-v1`) |
| Image captioning | NVILA-Lite-8B with A5-P3 Counterfactual Contrast prompting |
| Knowledge-graph extraction | Qwen2.5-14B-Instruct, with the LoRA fine-tuned variant available for the Pipeline 4 comparison |

The repository retains all evaluated candidate models and ablation implementations for experimental traceability. They are not required to execute the recommended path.

---

## Reproducibility Scope

This repository supports three levels of reproducibility:

1. **Archived thesis artifacts.** includes the structured outputs underlying the thesis experiments

2. **Representative end-to-end demonstration.** A documented one-document execution path will run the selected extraction, retrieval, captioning, and knowledge-graph construction configuration. This demonstration is intended to verify the workflow without requiring every model or ablation to be re-executed.

3. **Full HPC re-execution.** The repository retains code, Conda environments, and batch scripts for the complete retrieval benchmark, captioning ablations, fine-tuning workflow, and graph-construction evaluation. Re-executing the complete study requires a compatible Linux/HPC environment with NVIDIA GPUs, CUDA, substantial storage, and access to the model checkpoints listed in [`MODEL_SETUP.md`](MODEL_SETUP.md).

Exact numerical replication is not guaranteed across environments because generative model outputs can depend on software versions, model revisions, random seeds, hardware, and decoding settings. The committed artifacts provide the original experimental outputs reported in the thesis.

---

## Repository Structure

The repository will be organized around the following logical structure.

| Path | Contents |
|---|---|
| `data/raw_pdfs/` | Five source PDF documents used in the thesis experiments |
| `data/ontology/` | `TetraOnto.ttl` and the compact ontology representation used for prompt conditioning |
| `scripts/` | Python implementation, HPC batch scripts, setup utilities, and archived ablation code |
| `environments/` | Conda environment specifications for pipeline stages and model families |
| `model_patches/` | Required local compatibility patches, including the NVILA processing patch |
| `outputs/` | Archived experimental artifacts: captions, metrics, retrieval results, logs, and knowledge graphs |
| `MODEL_SETUP.md` | Model download, patching, and fine-tuned-model setup instructions |
| `Thesis.pdf` | Submitted master thesis report |

The relationship between the thesis pipelines and source code is:

| Thesis component | Current source-code location |
|---|---|
| Pipeline 1: layout parsing and extraction | `scripts/pipeline_01_extraction/` |
| Supporting document summarization | `scripts/pipeline_03_captioning/context_preparation/` |
| Pipeline 2: image-text retrieval | `scripts/pipeline_02_retrieval/` |
| Pipeline 3: captioning and ablations | `scripts/pipeline_03_captioning/` |
| Evaluation and selection metrics | `scripts/evaluation/` |
| Pipeline 4: triples and KG construction | `scripts/pipeline_04_knowledge_graph/` |
| HPC job submission scripts | `scripts/**/hpc/` |

This mapping will be updated when the codebase is reorganized into stable pipeline-oriented directories.

---

## Data

The `data/raw_pdfs/` directory contains the five French technical and institutional documents used in the experimental corpus.

PDFs 1, 3, 4, and 5 concern the Rhine basin or associated restoration contexts. PDF-2 is included for its structurally diverse French hydro-ecological content rather than for direct geographical relevance to the Rhine basin.

The source documents are retained for academic reproducibility. Users are responsible for verifying that their intended use, redistribution, and derivative use comply with the original document terms and any applicable project or institutional agreements.

---

## Setup

The implementation uses separate Conda environments because the evaluated models require incompatible dependency versions.

Create a required environment from the corresponding specification:

```bash
conda env create -f environments/<environment-file>.yml
```

For example:

```bash
conda env create -f environments/nvila_env.yml
conda activate nvila_env
```

Model weights are not stored in this repository. Download, local model-directory, compatibility-patch, ColPali merge, and fine-tuned-model instructions are provided in [`MODEL_SETUP.md`](MODEL_SETUP.md).

A representative end-to-end demo command sequence will be added after the pipeline scripts are standardized and tested.

---

## Hardware Requirements

The experiments were run on a Linux HPC cluster managed with Slurm. At the time of verification, available GPU partitions provided nodes with 1–8 GPUs and approximately 128 GB to 2.3 TB of RAM; GPU model and CUDA-toolkit availability depend on the allocated node and local module environment.

GPU resources are required for the retrieval, VLM captioning, document summarization, knowledge-graph extraction, and fine-tuning stages. CPU execution is sufficient for data preparation, graph conversion, visualisation, and metric utilities. The tested end-to-end demonstration requests 1 GPU, 8 CPU cores, 80 GB RAM, and a 12-hour wall-time limit; individual pipeline scripts specify their own tested Slurm resource requests.

---

## Archived Results

The `outputs/` directory contains the machine-readable artifacts supporting the thesis results, including:

- extracted PDF content, image crops, and OCR outputs;
- annotated image-text matching ground truth;
- retrieval similarity matrices and similarity histograms;
- caption JSON files and evaluation results for the captioning ablations;
- document summaries and weighted context files;
- Weighted model-selection outputs;
- baseline and fine-tuned RDF/Turtle knowledge graphs;
- GEXF and interactive HTML knowledge-graph visualizations;
- metric summaries, execution logs, and selected manual-audit materials.

These outputs allow inspection of the reported results without re-running the resource-intensive models.
