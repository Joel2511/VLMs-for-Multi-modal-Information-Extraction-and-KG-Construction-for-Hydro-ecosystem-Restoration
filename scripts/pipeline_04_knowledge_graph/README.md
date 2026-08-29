# Pipeline 4: Cross-Modal Knowledge-Graph Construction

This directory contains the final stage of the framework. Pipeline 4 converts
textual extraction output and image-caption output into ontology-conditioned
RDF triples, resolves cross-modal entities, and serializes text, image, and
merged knowledge graphs.

## Main implementations

| File | Purpose |
|---|---|
| `extract_triples_baseline.py` | Zero-shot Qwen2.5-14B-Instruct baseline extraction |
| `extract_triples_finetuned.py` | Fine-tuned Qwen2.5-14B extraction configuration |
| `convert_ttl_to_gexf.py` | Converts RDF/Turtle graph artifacts to GEXF |
| `visualize_knowledge_graph.py` | Generates static graph visualizations |
| `create_interactive_graph.py` | Creates a self-contained interactive HTML visualization from a user-supplied GEXF input path and HTML output path |
| `hpc/run_triple_extraction.sh` | Slurm/HPC wrapper for baseline and fine-tuned extraction |

## Extraction modes

The Pipeline 4 wrapper supports the controlled thesis comparison:

```bash
sbatch scripts/pipeline_04_knowledge_graph/hpc/run_triple_extraction.sh baseline
```

```bash
sbatch scripts/pipeline_04_knowledge_graph/hpc/run_triple_extraction.sh fine_tuned
```

The baseline uses:

```text
Qwen/Qwen2.5-14B-Instruct
```

The fine-tuned mode uses the merged checkpoint described in
[`../../MODEL_SETUP.md`](../../MODEL_SETUP.md).

## Inputs

Pipeline 4 consumes:

```text
<text-root>/<document-name>/extracted_content.json
<captions-root>/<document-name>/image_captions_p3_contrast.json
data/ontology/TetraOnto.ttl
data/ontology/TetraOnto_compact.txt
```

## Outputs

New graph outputs are written under:

```text
outputs/runs/knowledge_graph/
├── baseline/
└── fine_tuned/
```

Each document can produce:

```text
texts.ttl
images.ttl
merged_knowledge_graph.ttl
ft_merged_knowledge_graph.ttl
```

The archived thesis artifacts remain available under:

```text
outputs/triples_baseline/
outputs/triples_fine_tuned/
```
## Graph Conversion and Inspection

Convert a generated Turtle graph to GEXF:

```bash
python scripts/pipeline_04_knowledge_graph/convert_ttl_to_gexf.py \
  --mode fine_tuned \
  --document <document-name> \
  --input-root outputs/runs/knowledge_graph/fine_tuned
```

Create an interactive HTML graph from the merged GEXF file:

```bash
python scripts/pipeline_04_knowledge_graph/create_interactive_graph.py \
  outputs/runs/knowledge_graph/gexf/fine_tuned/<document-name>/ft_merged_knowledge_graph.gexf \
  outputs/runs/knowledge_graph/html/fine_tuned/<document-name>/interactive_graph.html
```

The generated HTML file is self-contained and can be opened locally in a web browser.

## Fine-tuning workflow

The `fine_tuning/` directory retains the five-step LoRA workflow used for the
baseline-versus-fine-tuned experiment:

1. Generate the synthetic training data.
2. Validate and clean generated triples.
3. Build the Hugging Face training dataset.
4. Fine-tune the LoRA adapter.
5. Merge the adapter with the baseline model.

The workflow is an advanced HPC reproduction path. The separately hosted merged
fine-tuned model can be used instead when available.
