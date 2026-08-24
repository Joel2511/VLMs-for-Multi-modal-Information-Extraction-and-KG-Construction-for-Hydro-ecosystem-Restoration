# Rhine River Restoration — Multimodal Knowledge Extraction Pipeline

**Master Thesis Research | ICube Laboratory, Université de Strasbourg**
**TETRA Project — Franco-German AI Toolbox for the Water Domain**

---

## Project Overview

This repository supports a master thesis conducted at ICube Laboratory as part of TETRA, a Franco-German collaborative initiative developing AI-based methodologies for the water domain.

The objective is to extract, structure, and organize restoration knowledge from heterogeneous documentary sources on Rhine River interventions — technical reports, maps, photographs, diagrams, and project plans, often unstructured, multilingual, and scattered across institutions. The project extends prior work on textual knowledge extraction into the visual domain, building a unified multimodal framework that integrates images and text through Vision-Language Models (VLMs).

The end goal is to populate a structured knowledge model (ontology) describing restoration cases such as intervention types, environmental context, methods, outcomes, constraints, and costs. And it also involves constructing knowledge graphs enabling querying, comparison, and visualization by researchers and practitioners.

---

## Research Pipelines

The project is organized into three sequential pipelines, each building on the previous stage's output.

### Pipeline 1 — Document Preprocessing

Raw PDF documents (French-language technical reports on Rhine restoration) are parsed and segmented. Images are extracted per page alongside surrounding text blocks, with OCR applied to capture embedded textual content within images. Output: a structured per-document corpus of images and associated text blocks, ready for downstream processing.

### Pipeline 2 — Image–Text Retrieval

For each extracted image, a retrieval stage identifies the most semantically relevant text blocks from the same document, providing grounded documentary context for the captioning stage. Both single-vector and multi-vector retrieval approaches are evaluated, with ablation studies comparing model families and retrieval strategies. Evaluation relies on standard information retrieval metrics against ground-truth image–text correspondences.

### Pipeline 3 — Multimodal Image Captioning

Each image is described using a structured, two-phase prompting strategy applied to a generative VLM:

- **Phase 1** — a purely visual, pixel-grounded description with no external knowledge injected.
- **Phase 2** — cross-references the visual description against retrieved text context and OCR output to identify complementary observations or contradictions.

Multiple VLMs are evaluated under this framework, with ablation studies assessing the contribution of each phase and the retrieval context. Caption quality is evaluated using VQA-based metrics and hallucination grounding measures.

---

## Evaluation Framework

- **Pipeline 2**: Information retrieval metrics (MRR, NDCG, Recall@k) against annotated ground-truth image–text pairs.
- **Pipeline 3**: VQA-based evaluation assessing factual grounding and hallucination rate of generated captions.

---

## Repository Structure

| Path | Contents |
|---|---|
| `scripts/code/` | Pipeline source code, organized by stage |
| `scripts/batch_jobs/` | HPC batch submission scripts per stage |
| `environments/` | Exported conda environment specs (`.yml`) for each pipeline stage |
| `model_patches/` | Source patches required for models with compatibility issues (e.g. NVILA) |
| `hf_models/` | Local model weights, downloaded per `MODEL_SETUP.md` |
| `data/`, `embeddings/` | Input corpora and generated embeddings |
| `outputs/` | Pipeline results (retrieval matrices, similarity plots, captions) |
| `logs/` | Run logs per pipeline stage |
| `MODEL_SETUP.md` | Model download and setup instructions (see below) |

---

## Setup

Each pipeline stage runs in its own conda environment, exported under `environments/`. Recreate an environment with:
