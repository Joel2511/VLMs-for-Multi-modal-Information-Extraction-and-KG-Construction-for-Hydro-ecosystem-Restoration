# Evaluation Utilities

This directory contains scripts used to calculate and aggregate the metrics
reported across the thesis experiments.

| File | Purpose |
|---|---|
| `evaluate_retrieval.py` | Retrieval metrics, including Recall@5, Precision@1, MRR@5, and nDCG@5 |
| `evaluate_captioning.py` | Caption-related evaluation utilities retained from the experiment workflow |
| `evaluate_knowledge_graph.py` | Knowledge-graph enrichment and structural metrics |
| `model_selection_metric.py` | Model selection weighted ranking |
| `rank_global_configurations.py` | Global model/configuration comparison across ablations |

The full evaluation outputs used in the thesis are preserved under `outputs/`.
New evaluation runs should write to `outputs/runs/metrics/`.

Pipeline 3's completed evaluation relies on the NLI and VQA-consistency
procedure. CLIPScore and PAC-S scripts were exploratory/deprecated and are not
part of the recommended final workflow.

`model_selection_metric.py` ranks candidate captioning models using the global weighted metric applied to the consolidated results from Ablations 2–4. `rank+global_configurations.py` uses the same weighted metric logic for all the models used in ablation 2 to 6 for global winner.

It is a post-hoc analysis utility: The relevant caption outputs and evaluation logs have to be placed in the documented input directories. It does not rerun caption generation or automatically collect results from each ablation directory.
