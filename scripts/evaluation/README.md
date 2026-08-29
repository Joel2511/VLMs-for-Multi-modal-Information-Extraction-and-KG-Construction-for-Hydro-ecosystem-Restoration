# Evaluation Utilities

This directory contains scripts used to calculate and aggregate the metrics
reported across the thesis experiments.

| File | Purpose |
|---|---|
| `evaluate_retrieval.py` | Retrieval metrics, including Recall@5, Precision@1, MRR@5, and nDCG@5 |
| `evaluate_captioning.py` | Caption-related evaluation utilities retained from the experiment workflow |
| `evaluate_knowledge_graph.py` | Knowledge-graph enrichment and structural metrics |
| `rank_within_ablation.py` | Per-ablation Pareto-informed ranking |
| `rank_global_configurations.py` | Global model/configuration comparison across ablations |

The full evaluation outputs used in the thesis are preserved under `outputs/`.
New evaluation runs should write to `outputs/runs/metrics/` rather than changing
archived thesis artifacts.

Pipeline 3's completed evaluation relies on the NLI and VQA-consistency
procedure. CLIPScore and PAC-S scripts were exploratory/deprecated and are not
part of the recommended final workflow.
