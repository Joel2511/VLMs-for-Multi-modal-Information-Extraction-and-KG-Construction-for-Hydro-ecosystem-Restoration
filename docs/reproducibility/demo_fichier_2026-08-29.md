# End-to-End Demonstration Record

## Run identity

- **Date:** 2026-08-29
- **Slurm job ID:** `17819713`
- **Document:** `fichier`
- **Execution mode:** One-document, isolated end-to-end HPC demonstration
- **Output root:** `outputs/runs/demo/fichier/`

## Selected workflow

```text
PDF extraction
→ VDR 2B Multi image–text retrieval
→ Qwen2-VL document summarisation
→ weighted retrieval/summary context construction
→ NVILA-Lite-8B A5-P3 Counterfactual Contrast captioning
→ fine-tuned Qwen2.5-14B ontology-aware knowledge-graph extraction
→ RDF/Turtle to GEXF conversion
→ interactive HTML knowledge-graph visualisation
```

## Result

The job completed successfully. The complete Slurm output is preserved in:

```text
docs/reproducibility/logs/demo_fichier_job_17819713.out
```

The generated run outputs are intentionally excluded from Git because they are
derived artifacts. The local validation performed after execution confirmed:

| Artifact | Validation result |
|---|---|
| Extracted text JSON | Valid JSON |
| NVILA image-caption JSON | Valid JSON; 25 entries |
| Weighted-context JSON | Valid JSON; 25 entries |
| Text graph | Valid Turtle; 705 RDF triples |
| Image graph | Valid Turtle; 460 RDF triples |
| Merged fine-tuned graph | Valid Turtle; 1,041 RDF triples |
| Merged fine-tuned GEXF graph | Valid GEXF; 94 nodes; 337 edges |
| Interactive HTML graph | Present and non-empty; 92 nodes; 249 edges |
| Runtime logs | No traceback, missing-file/module, runtime, CUDA-OOM, or segmentation-fault marker found |

## Reproducing the demonstration

The runtime requires the documented Conda environments, access to an appropriate
GPU-equipped Slurm node, and local model artifacts configured through
`HF_MODELS_DIR`. The large merged fine-tuned model checkpoint is hosted
separately and is not tracked by Git.

Run:

```bash
bash scripts/run_demo.sh fichier
```

For an HPC submission example, consult the repository's demonstration job script
or create a Slurm wrapper appropriate to the target cluster.
