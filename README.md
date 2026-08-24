# Rhine River Restoration: Multimodal Knowledge Extraction Pipeline

**Master Thesis Research | ICube Laboratory, Universite de Strasbourg**
**TETRA Project, Franco-German AI Toolbox for the Water Domain**

---

## Project Overview

This repository supports a master thesis conducted at ICube Laboratory as part of TETRA, a Franco-German collaborative initiative developing AI-based methodologies for the water domain, involving four institutional partners: Fraunhofer IOSB, ICube, SEBA, and Thales.

The objective is to extract, structure, and organize restoration knowledge from heterogeneous documentary sources on Rhine River interventions, including technical reports, maps, photographs, diagrams, and project plans, often unstructured, multilingual, and scattered across institutions. The project builds a unified multimodal framework that integrates images and text through Vision-Language Models (VLMs), extending prior ICube work on textual knowledge extraction into the visual domain.

The end goal is to populate a structured knowledge model (ontology) describing restoration cases, covering intervention types, environmental context, methods, outcomes, constraints, and costs, and to construct knowledge graphs enabling querying, comparison, and visualization by researchers and practitioners.

---

## Research Pipelines

The project is organized into four sequential pipelines, each building on the previous stage's output.

### Pipeline 1: Document Layout Parsing and Cross-Modal Extraction

Raw PDF documents (French-language technical reports on Rhine restoration) are rasterized page by page at 300 DPI and processed through a fine-tuned DocLayout-YOLOv10 detector to segment figures, captions, and text regions. Detected figure boxes are geometrically merged to reconstruct multi-panel compositions, captions are absorbed into their associated figures, and text is assembled into logical paragraphs using layout-aware heuristics tuned for French technical writing. A multi-layer noise filtration stage removes institutional boilerplate, repeated headers, and duplicate captions using cross-page repeater detection and OCR-based deduplication. Output: a structured per-document corpus of image crops and associated text blocks.

### Pipeline 2: Layout-Agnostic Contextual Image-Text Matching

For each extracted image, a retrieval stage identifies the most semantically relevant text blocks from the same document, providing grounded documentary context for the captioning stage. Two architectural families are evaluated: single-vector dense bi-encoders and multi-vector late-interaction retrievers using a mean-normalized MaxSim operator. Evaluation relies on Recall@k, Precision@1, Mean Reciprocal Rank (MRR), and normalized Discounted Cumulative Gain (nDCG@k) against annotated ground-truth image-text correspondences.

### Pipeline 3: Grounded Visual Captioning

Each image is described using a structured, two-phase prompting strategy applied to a generative VLM:

- **Phase 1** produces a purely visual, pixel-grounded inventory of the image, with no external context, listing visible components and transcribing legible text.
- **Phase 2** cross-references that visual inventory against retrieved local text context and a MapReduce-generated document summary, surfacing only facts that are visible in the image but absent from the text.

Seven VLMs are evaluated across multiple prompt topologies and context-fusion strategies. Caption quality is assessed using CLIPScore, PAC-S, NLI-based novelty detection, and VQA-Consistency Score, with model selection guided by a Pareto-informed weighted scoring scheme.

### Pipeline 4: Cross-Modal Semantic Integration

Verified text and image captions are converted into RDF triples that populate an existing OWL ontology (TetraOnto.ttl). A compressed, bilingual (English/French) schema summary is injected into the extraction prompt to keep the model anchored to authorized classes and predicates. Extraction runs as two passes per document, text first and then images, sharing a running entity registry so that image-derived mentions resolve back to entities already established from text. Rather than generating raw RDF syntax directly, the model emits a flat JSON array of triples, which a deterministic Python script sanitizes, validates against a predicate and class whitelist, and serializes into the graph, demoting any non-conformant predicate to an auditable annotation instead of discarding it.

Cross-modal entity resolution then merges the text-derived and image-derived graphs through three passes: exact-label normalization, LLM-mediated alias resolution, and similarity-based alias merging. A LoRA fine-tuned variant of Qwen2.5-14B-Instruct is evaluated against the zero-shot baseline model for predicate conformance. Evaluation uses Schema Conformance Rate (SCR), Average Node Degree Shift, Cross-Modal Link Density (CMLD), and manually audited Intrinsic Triple Precision.

---

## Evaluation Framework

- **Pipeline 2**: Recall@k, Precision@1, MRR, and nDCG@k against annotated ground-truth image-text pairs.
- **Pipeline 3**: CLIPScore, PAC-S, NLI-based novelty, and VQA-Consistency Score, assessing factual grounding and hallucination rate.
- **Pipeline 4**: Schema Conformance Rate, node degree shift, cross-modal link density, and manually reviewed triple precision.

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
| `outputs/` | Pipeline results (retrieval matrices, similarity plots, captions, knowledge graph files) |
| `logs/` | Run logs per pipeline stage |
| `MODEL_SETUP.md` | Model download and setup instructions (see below) |

---

## Setup

Each pipeline stage runs in its own conda environment, exported under `environments/`. Recreate an environment with:
