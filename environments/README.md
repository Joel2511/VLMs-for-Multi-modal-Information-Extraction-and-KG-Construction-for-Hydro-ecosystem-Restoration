# Conda Environments

The project uses multiple Conda environments because the evaluated vision-language,
retrieval, and knowledge-graph components require incompatible package versions.

Create an environment with:

```bash
conda env create -f environments/<environment-file>.yml
```

Activate it with:

```bash
conda activate <environment-name>
```

## Environment mapping

| Environment file | Conda environment | Primary use |
|---|---|---|
| `kg_env.yml` | `kg_env` | Pipeline 1 document extraction, layout processing, OCR filtering, and KG-related utilities |
| `barthez_env.yml` | `barthez_env` | Pipeline 2 retrieval experiments, including VDR, CLIP, ColQwen2, DSE-Qwen2, and Nemotron workflows |
| `colpali_env.yml` | `colpali_env` | Pipeline 2 ColPali retrieval and ColPali adapter merging |
| `colqwen_char.yml` | `colqwen_char` | InternVL2.5 captioning experiments |
| `vlm_env.yml` | `vlm_env` | Document summarization, weighted-context preparation, Mistral/Pixtral captioning, and caption-evaluation utilities |
| `nvila_env.yml` | `nvila_env` | NVILA-Lite-8B captioning and Prompt/Decoding Ablations 1--6 |
| `qwen36_env.yml` | `qwen36_env` | Qwen3.6-27B captioning experiments |
| `qwen3vl_env.yml` | `qwen3vl_env` | Qwen3-VL, Llama-3.2 Vision, Pipeline 4 triple extraction, LoRA fine-tuning, and KG visualization |
| `triple_env.yml` | `triple_env` | Triple-extraction support utilities |

## Notes

- The YAML files record the tested package configurations used during the thesis experiments.
- The full suite requires Linux, CUDA-compatible NVIDIA GPUs, and sufficient storage for model checkpoints.
- Model checkpoints are not included in the repository. See [`../MODEL_SETUP.md`](../MODEL_SETUP.md).
- Users may need to adapt CUDA, PyTorch, and cluster-module settings to their local HPC environment.
- The environment files are intended for re-execution and traceability. Exact numerical replication of generative-model outputs is not guaranteed across different hardware, model revisions, or package versions.
