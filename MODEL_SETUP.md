# Model Setup

Download instructions for every model used in this project, organized by pipeline. Run once per session:

```bash
export HF_TOKEN="your_token_here"
```

---

## Pipeline 1: Document Layout Parsing

| Model | Purpose | Source |
|---|---|---|
| DocLayout-YOLO | Figure, caption, and text region detection | Auto-downloads on first `pdf_extraction.py` run |

| Field | Value |
|---|---|
| Repo ID | `juliozhao/DocLayout-YOLO-DocStructBench` |
| Filename | `doclayout_yolo_docstructbench_imgsz1024.pt` |

No manual download needed for this stage.

---

## Pipeline 2: Image-Text Retrieval

| Model | Purpose | Hugging Face Repo | Local Directory |
|---|---|---|---|
| CLIP ViT-B/32 | Baseline single-vector retrieval | `openai/clip-vit-base-patch32` | `hf_models/clip-vit-base-patch32` |
| ColPali v1.2 | Multi-vector late-interaction retrieval (requires merge, see Exceptions) | `vidore/colpali-v1.2` | `hf_models/colpali-v1.2-merged` |
| **ColQwen2 v1.0** | **Winner — best retrieval performance in evaluation** | `vidore/colqwen2-v1.0-hf` | `hf_models/colqwen2-v1.0-hf` |
| DSE-Qwen2 2B | Multi-vector retrieval candidate | `MrLight/dse-qwen2-2b-mrl-v1` | `hf_models/dse-qwen2-2b` |
| VDR 2B Multi | Multi-vector retrieval candidate | `llamaindex/vdr-2b-multi-v1` | `hf_models/vdr-2b-multi-v1` |
| Nemotron ColEmbed 4B | Multi-vector retrieval candidate | `nvidia/nemotron-colembed-vl-4b-v2` | `hf_models/nemotron-colembed-vl-4b-v2` |

```bash
hf download openai/clip-vit-base-patch32 --local-dir hf_models/clip-vit-base-patch32 --token $HF_TOKEN
hf download vidore/colqwen2-v1.0-hf --local-dir hf_models/colqwen2-v1.0-hf --token $HF_TOKEN
hf download MrLight/dse-qwen2-2b-mrl-v1 --local-dir hf_models/dse-qwen2-2b --token $HF_TOKEN
hf download llamaindex/vdr-2b-multi-v1 --local-dir hf_models/vdr-2b-multi-v1 --token $HF_TOKEN
hf download nvidia/nemotron-colembed-vl-4b-v2 --local-dir hf_models/nemotron-colembed-vl-4b-v2 --token $HF_TOKEN
```

ColPali download and merge steps are in the Exceptions section below.

---

## Pipeline 3: Image Captioning

| Model | Purpose | Hugging Face Repo | Local Directory |
|---|---|---|---|
| **NVILA-Lite 8B** | **Winner — best production-scenario captioning config** | `Efficient-Large-Model/NVILA-Lite-8B-hf` | `hf_models/NVILA-Lite-8B-hf` |
| Qwen3-VL 30B | Top unconstrained-compute captioning config | `Qwen/Qwen3-VL-30B-A3B-Instruct` | `hf_models/Qwen3-VL-30B-A3B-Instruct` |
| Mistral-Small 3.1 24B | Captioning ablation candidate | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `hf_models/Mistral-Small-3.1-24B-Instruct-2503` |
| Pixtral 12B | Captioning ablation candidate | `mistral-community/pixtral-12b` | `hf_models/pixtral-12b-hf` |
| Qwen3.6 27B | Captioning ablation candidate | `Qwen/Qwen3.6-27B` | `hf_models/Qwen3.6-27B` |
| Llama-3.2 Vision 11B | Captioning ablation candidate | `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` | `hf_models/Llama-3.2-11B-Vision-Instruct` |
| InternVL2.5 8B | Captioning ablation candidate | `OpenGVLab/InternVL2_5-8B` | `hf_models/InternVL2_5-8B` |

```bash
hf download Efficient-Large-Model/NVILA-Lite-8B-hf --local-dir hf_models/NVILA-Lite-8B-hf --token $HF_TOKEN
hf download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir hf_models/Qwen3-VL-30B-A3B-Instruct --token $HF_TOKEN
hf download mistralai/Mistral-Small-3.1-24B-Instruct-2503 --local-dir hf_models/Mistral-Small-3.1-24B-Instruct-2503 --exclude "consolidated.safetensors" --token $HF_TOKEN
hf download mistral-community/pixtral-12b --local-dir hf_models/pixtral-12b-hf --token $HF_TOKEN
hf download Qwen/Qwen3.6-27B --local-dir hf_models/Qwen3.6-27B --token $HF_TOKEN
hf download unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit --local-dir hf_models/Llama-3.2-11B-Vision-Instruct --token $HF_TOKEN
hf download OpenGVLab/InternVL2_5-8B --local-dir hf_models/InternVL2_5-8B --token $HF_TOKEN
```

**Pipeline 3 support models** (not part of the captioning ablation itself):

| Model | Purpose | Hugging Face Repo | Local Directory |
|---|---|---|---|
| Qwen2-VL 7B | MapReduce document summarization, weighted context fusion, caption metric scoring | `Qwen/Qwen2-VL-7B-Instruct` | `hf_models/Qwen2-VL-7B-Instruct` |
| mDeBERTa v3 (mnli-xnli) | NLI-based caption novelty metric | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | `hf_models/mDeBERTa-v3-mnli-xnli` |

```bash
hf download Qwen/Qwen2-VL-7B-Instruct --local-dir hf_models/Qwen2-VL-7B-Instruct --token $HF_TOKEN
hf download MoritzLaurer/mDeBERTa-v3-base-mnli-xnli --local-dir hf_models/mDeBERTa-v3-mnli-xnli --token $HF_TOKEN
```

---

## Pipeline 4: Knowledge Graph Construction

| Model | Purpose | Hugging Face Repo | Local Directory |
|---|---|---|---|
| Qwen2.5 14B (baseline) | RDF triple extraction, zero-shot | `Qwen/Qwen2.5-14B-Instruct` | `hf_models/Qwen2.5-14B-Instruct` |
| **Qwen2.5 14B (fine-tuned)** | **LoRA fine-tuned for predicate conformance, produced locally, see below** | n/a, built from the baseline checkpoint above | `scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/5.merge/Qwen2.5-14B-populate-ontology-merged` |

```bash
hf download Qwen/Qwen2.5-14B-Instruct --local-dir hf_models/Qwen2.5-14B-Instruct --token $HF_TOKEN
```

The fine-tuned checkpoint is not downloaded, it is built locally by running the five-step pipeline below. This was a small-scale ablation (150 synthetic triples) to demonstrate that fine-tuning improves ontology-predicate conformance over the zero-shot baseline, so the checkpoint is reproduced from scratch rather than distributed as a separate artifact. All five steps run in the `qwen3vl_env` conda environment and depend on the baseline `Qwen2.5-14B-Instruct` checkpoint above already being downloaded.

**Step 1: generate the synthetic training dataset (~45 minutes)**

```bash
cd scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/1.dataset_generation/
bash run_data_generation.sh
```

Produces `synthetic_dataset.jsonl`.

**Step 2: validate and clean the generated triples (CPU-only, fast)**

```bash
cd ../2.validation/
python3 validate_triples.py
```

Produces `synthetic_dataset_clean.jsonl` and `validation_report.txt`.

**Step 3: build the fine-tuning dataset (CPU-only, fast)**

```bash
cd ../3.dataset_prep/
python3 build_hf_dataset.py
```

Produces the `hf_dataset/` directory in chat-template format.

**Step 4: run LoRA fine-tuning (~30 minutes)**

```bash
cd ../4.training/
bash run_fine_tune.sh
```

Produces the LoRA adapter in `Qwen2.5-14B-populate-ontology`.

**Step 5: merge the adapter into the base model (1 to 2 minutes)**

```bash
cd ../5.merge/
bash run_merge.sh
```

Produces the final deployable checkpoint in `Qwen2.5-14B-populate-ontology-merged`, which is the path referenced directly in the extraction code.
---

## Exceptions

### ColPali (two downloads plus local merge)

The official pre-merged checkpoint did not work for this setup, so ColPali is downloaded in two parts and merged locally.

**Step 1: download the LoRA adapter and the base backbone**

```bash
python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'vidore/colpali-v1.2', '--local-dir', '$RHINE_PROJECT_ROOT/hf_models/colpali-v1.2', '--token', os.environ['HF_TOKEN']], env=env)"
```

```bash
python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'vidore/colpaligemma-3b-pt-448-base', '--local-dir', '$RHINE_PROJECT_ROOT/hf_models/colpaligemma-3b-pt-448-base', '--token', os.environ['HF_TOKEN']], env=env)"
```

**Step 2: merge the adapter into the base model**

```bash
python3 -c 'import torch; from colpali_engine.models import ColPali; from peft import PeftModel; from transformers import AutoTokenizer; import os; BASE_MODEL_DIR=os.environ["RHINE_PROJECT_ROOT"]+"/hf_models/colpaligemma-3b-pt-448-base"; ADAPTER_DIR=os.environ["RHINE_PROJECT_ROOT"]+"/hf_models/colpali-v1.2"; MERGED_OUTPUT_DIR=os.environ["RHINE_PROJECT_ROOT"]+"/hf_models/colpali-v1.2-merged"; base_model=ColPali.from_pretrained(BASE_MODEL_DIR,torch_dtype=torch.float32,local_files_only=True); model=PeftModel.from_pretrained(base_model,ADAPTER_DIR,local_files_only=True); model=model.merge_and_unload(); model.save_pretrained(MERGED_OUTPUT_DIR,safe_serialization=True); tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL_DIR,local_files_only=True); tokenizer.save_pretrained(MERGED_OUTPUT_DIR)'
```

> **Do not delete `colpaligemma-3b-pt-448-base` after merging.** The merged model only holds weights. `ColPaliProcessor.from_pretrained()` loads image preprocessing config from `colpaligemma-3b-pt-448-base` directly at inference time, so deleting it breaks the pipeline even though the model itself loads from `colpali-v1.2-merged`.

### NVILA (required source patch)

The model will not load without this patch applied.

```bash
cp model_patches/NVILA-Lite-8B-hf/processing_nvila_lite.py hf_models/NVILA-Lite-8B-hf/processing_nvila_lite.py
```
