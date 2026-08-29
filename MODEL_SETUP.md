# Model Setup

This document describes the model download and local setup procedures used in the project.

The recommended end-to-end demonstration requires only the selected models:

- **Pipeline 1:** DocLayout-YOLO
- **Pipeline 2:** VDR 2B Multi
- **Pipeline 3:** NVILA-Lite-8B
- **Pipeline 3 support:** Qwen2-VL-7B-Instruct and mDeBERTa-v3-base-mnli-xnli
- **Pipeline 4:** Qwen2.5-14B-Instruct baseline and the separately hosted fine-tuned merged checkpoint

The remaining models are retained for re-executing the retrieval and captioning ablations reported in the thesis. They are not required for the recommended demonstration workflow.

Model weights are not tracked in this repository. Do not commit Hugging Face tokens, downloaded model files, Hugging Face caches, or locally generated checkpoints.

---

## Prerequisites

Run the following commands from the repository root after cloning the repository:

```bash
export RHINE_PROJECT_ROOT="$(pwd)"
export HF_TOKEN="your_token_here"
```

`HF_TOKEN` is required for authenticated or gated Hugging Face downloads. Set it in each new terminal session when needed. Do not add the token to source code, batch scripts, configuration files committed to Git, or shell history.

All model checkpoints are downloaded into the local `hf_models/` directory.

Before downloading models, create a local directory outside the repository and set `HF_MODELS_DIR` to it:

```bash
export HF_MODELS_DIR="$HOME/hf_models"
mkdir -p "$HF_MODELS_DIR"
```

The download commands then place each model inside this directory. `HF_MODELS_DIR` is intentionally excluded from Git.

---

## Recommended Demonstration Models

The following models implement the selected end-to-end configuration described in the thesis.

| Thesis stage | Model | Purpose | Hugging Face repository | Local directory |
|---|---|---|---|---|
| Pipeline 1 | DocLayout-YOLO | Figure, caption, text-region, and structural-margin detection | `juliozhao/DocLayout-YOLO-DocStructBench` | Automatically downloaded or stored by the extraction implementation |
| Pipeline 2 | **VDR 2B Multi** | **Selected image-text retrieval configuration** | `llamaindex/vdr-2b-multi-v1` | `hf_models/vdr-2b-multi-v1` |
| Pipeline 3 | **NVILA-Lite-8B** | **Selected VLM for the A5-P3 Counterfactual Contrast captioning configuration** | `Efficient-Large-Model/NVILA-Lite-8B-hf` | `hf_models/NVILA-Lite-8B-hf` |
| Pipeline 3 support | Qwen2-VL-7B-Instruct | MapReduce document summarization, weighted context fusion, and caption metric scoring | `Qwen/Qwen2-VL-7B-Instruct` | `hf_models/Qwen2-VL-7B-Instruct` |
| Pipeline 3 support | mDeBERTa-v3-base-mnli-xnli | NLI-based caption novelty and grounding evaluation | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | `hf_models/mDeBERTa-v3-mnli-xnli` |
| Pipeline 4 | Qwen2.5-14B-Instruct | Zero-shot baseline for ontology-conditioned triple extraction | `Qwen/Qwen2.5-14B-Instruct` | `hf_models/Qwen2.5-14B-Instruct` |
| Pipeline 4 | Qwen2.5-14B fine-tuned merged checkpoint | Fine-tuned Pipeline 4 configuration for ontology-predicate conformance | Separately hosted; see below | `hf_models/Qwen2.5-14B-populate-ontology-merged` |

### Download the recommended Hugging Face models

```bash
hf download llamaindex/vdr-2b-multi-v1 \
  --local-dir hf_models/vdr-2b-multi-v1 \
  --token $HF_TOKEN

hf download Efficient-Large-Model/NVILA-Lite-8B-hf \
  --local-dir hf_models/NVILA-Lite-8B-hf \
  --token $HF_TOKEN

hf download Qwen/Qwen2-VL-7B-Instruct \
  --local-dir hf_models/Qwen2-VL-7B-Instruct \
  --token $HF_TOKEN

hf download MoritzLaurer/mDeBERTa-v3-base-mnli-xnli \
  --local-dir hf_models/mDeBERTa-v3-mnli-xnli \
  --token $HF_TOKEN

hf download Qwen/Qwen2.5-14B-Instruct \
  --local-dir hf_models/Qwen2.5-14B-Instruct \
  --token $HF_TOKEN
```

### Pipeline 1: DocLayout-YOLO

The Pipeline 1 extraction implementation downloads or loads the following checkpoint when `pdf_extraction.py` is run:

| Field | Value |
|---|---|
| Hugging Face repository | `juliozhao/DocLayout-YOLO-DocStructBench` |
| Checkpoint filename | `doclayout_yolo_docstructbench_imgsz1024.pt` |
| Purpose | Layout detection of figures, captions, text regions, and structural margins |

No separate manual model-download command is required for this stage if the extraction script can access its configured model source.

### NVILA compatibility patch

NVILA-Lite-8B requires the included local processing-file patch. Apply it after downloading the checkpoint:

```bash
cp model_patches/NVILA-Lite-8B-hf/processing_nvila_lite.py \
  hf_models/NVILA-Lite-8B-hf/processing_nvila_lite.py
```

Do not skip this step: the NVILA model does not load correctly in the project environment without the patch.

---

## Fine-Tuned Pipeline 4 Model

Pipeline 4 compares the zero-shot `Qwen/Qwen2.5-14B-Instruct` baseline with a
LoRA fine-tuned configuration designed to improve ontology-predicate
conformance.

The merged fine-tuned checkpoint used for the reported Pipeline 4 results is
hosted separately because model artifacts are not tracked in GitHub.

### Download the merged checkpoint

1. Download the merged fine-tuned checkpoint archive from Google Drive:

   [Download the merged fine-tuned checkpoint archive from Google Drive](https://drive.google.com/drive/folders/16nRYdLA0_OSq84LxdGYbj6rFB1oIRMqk?usp=sharing)

2. Verify the archive integrity before extraction.

   SHA-256 of `Qwen2.5-14B-populate-ontology-merged.tar.gz`:

   ```text
   cdcee16032d66ffab02451fa743a84773f18d6ae712c6b3fd7e5dddeeb50babd
   ```

   The checksum is also provided in the accompanying
   `Qwen2.5-14B-populate-ontology-merged.tar.gz.sha256` file. Place both files
   in the same directory and run:

   ```bash
   sha256sum -c Qwen2.5-14B-populate-ontology-merged.tar.gz.sha256
   ```

3. From the repository root, create the local model directory and extract the
   archive:

   ```bash
   mkdir -p hf_models
   tar -xzf Qwen2.5-14B-populate-ontology-merged.tar.gz -C hf_models/
   ```

   The expected model path after extraction is:

   ```text
   hf_models/Qwen2.5-14B-populate-ontology-merged/
   ```

4. Confirm that the extracted directory contains the model configuration,
   tokenizer files, and model-weight files. Do not create an additional nested
   directory inside `hf_models/Qwen2.5-14B-populate-ontology-merged/`.

The fine-tuned triple-extraction implementation automatically uses this
directory when the fine-tuned Pipeline 4 configuration is selected.

### Local fine-tuning reproduction workflow

The repository also retains the complete fine-tuning workflow and HPC launchers for reproducing the merged checkpoint locally. This workflow is optional when the pre-trained merged checkpoint above can be downloaded and verified successfully; use it only when local retraining is required or the checkpoint is unavailable.

The workflow uses the baseline `Qwen/Qwen2.5-14B-Instruct` checkpoint and the `qwen3vl_env` Conda environment.

Activate the environment:

```bash
conda activate qwen3vl_env
```

Then execute the following steps from the repository root.

#### Step 1: Generate the synthetic training dataset

```bash
cd scripts/pipeline_04_knowledge_graph/fine_tuning/
bash run_data_generation.sh
```

Expected output:

```text
synthetic_dataset.jsonl
```

Approximate runtime reported during the original experiments: 45 minutes.

#### Step 2: Validate and clean generated triples

```bash
cd ../2.validation/
python3 validate_triples.py
```

Expected outputs:

```text
synthetic_dataset_clean.jsonl
validation_report.txt
```

This step is CPU-only and comparatively fast.

#### Step 3: Build the Hugging Face training dataset

```bash
cd ../3.dataset_prep/
python3 build_hf_dataset.py
```

Expected output:

```text
hf_dataset/
```

The resulting dataset uses the required chat-template format.

#### Step 4: Run LoRA fine-tuning

```bash
cd ../4.training/
bash run_fine_tune.sh
```

Expected output:

```text
Qwen2.5-14B-populate-ontology/
```

Approximate runtime reported during the original experiments: 30 minutes on the tested HPC environment.

#### Step 5: Merge the LoRA adapter into the base model

```bash
cd ../5.merge/
bash run_merge.sh
```

Expected output:

```text
Qwen2.5-14B-populate-ontology-merged/
```

Approximate runtime reported during the original experiments: 1--2 minutes.

> **Reproducibility note:** the fine-tuning workflow is re-runnable using the included code and synthetic training data. Exact bitwise reproduction cannot be guaranteed across systems because generated data, package versions, model revisions, GPU hardware, random seeds, and training conditions can affect the resulting checkpoint.

---

## Archived Retrieval Ablation Models

The following models were evaluated in the Pipeline 2 retrieval benchmark. Only VDR 2B Multi is required for the selected end-to-end workflow.

| Model | Purpose | Hugging Face repository | Local directory |
|---|---|---|---|
| CLIP ViT-B/32 | Single-vector retrieval baseline | `openai/clip-vit-base-patch32` | `hf_models/clip-vit-base-patch32` |
| ColPali v1.2 | Multi-vector late-interaction retrieval; requires local merge | `vidore/colpali-v1.2` | `hf_models/colpali-v1.2-merged` |
| ColQwen2 v1.0 | Multi-vector retrieval candidate | `vidore/colqwen2-v1.0-hf` | `hf_models/colqwen2-v1.0-hf` |
| DSE-Qwen2 2B | Dense single-vector retrieval candidate | `MrLight/dse-qwen2-2b-mrl-v1` | `hf_models/dse-qwen2-2b` |
| Nemotron ColEmbed 4B | Multi-vector retrieval candidate | `nvidia/nemotron-colembed-vl-4b-v2` | `hf_models/nemotron-colembed-vl-4b-v2` |

```bash
hf download openai/clip-vit-base-patch32 \
  --local-dir hf_models/clip-vit-base-patch32 \
  --token $HF_TOKEN

hf download vidore/colqwen2-v1.0-hf \
  --local-dir hf_models/colqwen2-v1.0-hf \
  --token $HF_TOKEN

hf download MrLight/dse-qwen2-2b-mrl-v1 \
  --local-dir hf_models/dse-qwen2-2b \
  --token $HF_TOKEN

hf download nvidia/nemotron-colembed-vl-4b-v2 \
  --local-dir hf_models/nemotron-colembed-vl-4b-v2 \
  --token $HF_TOKEN
```

ColPali uses a separate base backbone and LoRA adapter. Its download and merge procedure is documented in the [ColPali setup](#colpali-two-downloads-plus-local-merge) section below.

---

## Archived Captioning Ablation Models

The following models were evaluated in Pipeline 3 captioning ablations. NVILA-Lite-8B is the selected model for the recommended workflow.

| Model | Purpose | Hugging Face repository | Local directory |
|---|---|---|---|
| Qwen3-VL 30B | High-quality captioning candidate under unconstrained compute | `Qwen/Qwen3-VL-30B-A3B-Instruct` | `hf_models/Qwen3-VL-30B-A3B-Instruct` |
| Mistral-Small 3.1 24B | Captioning ablation candidate | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `hf_models/Mistral-Small-3.1-24B-Instruct-2503` |
| Pixtral 12B | Captioning ablation candidate | `mistral-community/pixtral-12b` | `hf_models/pixtral-12b-hf` |
| Qwen3.6 27B | Captioning ablation candidate | `Qwen/Qwen3.6-27B` | `hf_models/Qwen3.6-27B` |
| Llama-3.2 Vision 11B | Captioning ablation candidate | `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` | `hf_models/Llama-3.2-11B-Vision-Instruct` |
| InternVL2.5 8B | Captioning ablation candidate | `OpenGVLab/InternVL2_5-8B` | `hf_models/InternVL2_5-8B` |

```bash
hf download Qwen/Qwen3-VL-30B-A3B-Instruct \
  --local-dir hf_models/Qwen3-VL-30B-A3B-Instruct \
  --token $HF_TOKEN

hf download mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --local-dir hf_models/Mistral-Small-3.1-24B-Instruct-2503 \
  --exclude "consolidated.safetensors" \
  --token $HF_TOKEN

hf download mistral-community/pixtral-12b \
  --local-dir hf_models/pixtral-12b-hf \
  --token $HF_TOKEN

hf download Qwen/Qwen3.6-27B \
  --local-dir hf_models/Qwen3.6-27B \
  --token $HF_TOKEN

hf download unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit \
  --local-dir hf_models/Llama-3.2-11B-Vision-Instruct \
  --token $HF_TOKEN

hf download OpenGVLab/InternVL2_5-8B \
  --local-dir hf_models/InternVL2_5-8B \
  --token $HF_TOKEN
```

---

## ColPali: Two Downloads Plus Local Merge

ColPali v1.2 was evaluated as a Pipeline 2 retrieval candidate. The local setup used in this project downloads the LoRA adapter and base backbone separately, then merges them locally.

The following instructions assume that `RHINE_PROJECT_ROOT` has been set as described in [Prerequisites](#prerequisites).

### Step 1: Download the ColPali adapter

```bash
python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'vidore/colpali-v1.2', '--local-dir', os.path.join(os.environ['RHINE_PROJECT_ROOT'], 'hf_models/colpali-v1.2'), '--token', os.environ['HF_TOKEN']], env=env, check=True)"
```

### Step 2: Download the base backbone

```bash
python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'vidore/colpaligemma-3b-pt-448-base', '--local-dir', os.path.join(os.environ['RHINE_PROJECT_ROOT'], 'hf_models/colpaligemma-3b-pt-448-base'), '--token', os.environ['HF_TOKEN']], env=env, check=True)"
```

### Step 3: Merge the adapter into the base model

Run this command in the Conda environment that provides `torch`, `transformers`, `peft`, and `colpali_engine`:

```bash
python3 -c 'import os; import torch; from colpali_engine.models import ColPali; from peft import PeftModel; from transformers import AutoTokenizer; root=os.environ["RHINE_PROJECT_ROOT"]; base_dir=os.path.join(root, "hf_models/colpaligemma-3b-pt-448-base"); adapter_dir=os.path.join(root, "hf_models/colpali-v1.2"); output_dir=os.path.join(root, "hf_models/colpali-v1.2-merged"); base_model=ColPali.from_pretrained(base_dir, torch_dtype=torch.float32, local_files_only=True); model=PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=True); model=model.merge_and_unload(); model.save_pretrained(output_dir, safe_serialization=True); tokenizer=AutoTokenizer.from_pretrained(base_dir, local_files_only=True); tokenizer.save_pretrained(output_dir)'
```

> **Important:** do not delete `hf_models/colpaligemma-3b-pt-448-base/` after merging. The merged model stores the merged weights, but `ColPaliProcessor.from_pretrained()` loads image-preprocessing configuration from the base backbone directory during inference. Removing the base backbone therefore breaks the project retrieval implementation.

---

## Environment Mapping

The repository uses separate Conda environments because several evaluated model families require incompatible versions of PyTorch, Transformers, CUDA-related libraries, or model-specific packages.

| Environment file | Primary use |
|---|---|
| `environments/vlm_env.yml` | General vision-language model workflows |
| `environments/nvila_env.yml` | NVILA-Lite-8B captioning |
| `environments/colpali_env.yml` | ColPali retrieval and merging |
| `environments/colqwen_char.yml` | ColQwen2 retrieval |
| `environments/barthez_env.yml` | BARThez summarization experiments |
| `environments/qwen3vl_env.yml` | Qwen-based VLM workflows and Pipeline 4 fine-tuning |
| `environments/qwen36_env.yml` | Qwen3.6 captioning experiments |
| `environments/kg_env.yml` | Knowledge-graph construction and visualization |
| `environments/triple_env.yml` | Triple extraction and related utilities |

Create an environment with:

```bash
conda env create -f environments/<environment-file>.yml
```

For example:

```bash
conda env create -f environments/nvila_env.yml
conda activate nvila_env
```

The final demonstration instructions in `README.md` will state the environment required for each command.
