# Rhine River Restoration — Multimodal Knowledge Extraction Pipeline

**Master Thesis Research | ICube Laboratory, Université de Strasbourg**
**TETRA Project — Franco-German AI Toolbox for the Water Domain**

***

## Project Overview

This repository supports a master thesis research project conducted at ICube Laboratory as part of the TETRA project, a Franco-German collaborative initiative involving academic and industrial partners working on AI-based methodologies for the water domain.

The central objective is to extract, structure, and organize restoration knowledge from heterogeneous documentary sources related to interventions on the Rhine River. These sources include technical reports, maps, photographs, diagrams, and project plans — often unstructured, multilingual, and distributed across institutions. The project extends prior work on textual knowledge extraction to the visual domain, building a unified multimodal framework that integrates images and text through Vision-Language Models (VLMs).

The ultimate goal is to populate a structured knowledge model (ontology) describing restoration cases — covering intervention types, environmental context, methods, outcomes, constraints, and costs — and to construct knowledge graphs that enable querying, comparison, and visualization by researchers and practitioners.

***

## Research Pipelines

The project is organized into three sequential pipelines, each building on the output of the previous.

### Pipeline 1 — Document Preprocessing

Raw PDF documents (French-language technical reports on Rhine restoration) are parsed and segmented. Images are extracted per page alongside their surrounding text blocks, with optical character recognition (OCR) applied to capture embedded textual content within images. The output is a structured per-document corpus of images and associated text blocks, ready for downstream processing.

### Pipeline 2 — Image–Text Retrieval

For each extracted image, a retrieval stage identifies the most semantically relevant text blocks from the same document. This provides grounded documentary context that can later inform the captioning stage. Both single-vector and multi-vector retrieval approaches are evaluated, with ablation studies comparing model families and retrieval strategies. Evaluation relies on standard information retrieval metrics computed against ground-truth image–text correspondences.

### Pipeline 3 — Multimodal Image Captioning

Each image is described using a structured prompting strategy applied to a generative Vision-Language Model. The prompt follows a two-phase design:

- **Phase 1** elicits a purely visual, pixel-grounded description of the image with no external knowledge injected.
- **Phase 2** cross-references the visual description against the retrieved text context and OCR output to identify complementary observations or contradictions.

Multiple VLMs are evaluated under this framework. Prompt design itself is subject to ablation studies to assess the contribution of each phase and the retrieval context. Caption quality is evaluated using Visual Question Answering (VQA)-based metrics and hallucination grounding measures.

***

## Evaluation Framework

Evaluation spans both retrieval and generation stages:

- **Pipeline 2**: Information retrieval metrics (MRR, NDCG, Recall@k) against annotated ground-truth image–text pairs
- **Pipeline 3**: VQA-based evaluation assessing factual grounding and hallucination rate of generated captions

***

## Project Context

This research is part of the TETRA project, which focuses on developing an AI-based toolbox and methodology for the water domain. Four partners are involved: ICube and Fraunhofer IOSB (public research), and SEBA and Thalès (private sector). The results of this internship will be integrated into the TETRA toolbox, contributing to the broader goal of making restoration experience accessible and reusable for future river restoration design, policy, and monitoring.

***

## Repository Status

| Stage | Status |
|---|---|
| Pipeline 1 — Document Preprocessing | Complete |
| Pipeline 2 — Image–Text Retrieval | Complete |
| Pipeline 3 — Image Captioning | In Progress |
| Knowledge Graph Construction | Planned |

***

## Supervisors

*[To be filled in]*

***

*Full methodology, experimental results, and analysis are documented in the thesis report.*
