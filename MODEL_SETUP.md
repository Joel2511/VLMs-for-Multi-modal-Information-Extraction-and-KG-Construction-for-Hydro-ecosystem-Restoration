# Model Setup

Download instructions for every model used in this project. Run once per session:

```bash
export HF_TOKEN="your_token_here"
```

## Standard Downloads

| Model | Hugging Face Repo | Local Directory |
|---|---|---|
| Qwen3-VL 30B | `Qwen/Qwen3-VL-30B-A3B-Instruct` | `hf_models/Qwen3-VL-30B-A3B-Instruct` |
| NVILA-Lite 8B | `Efficient-Large-Model/NVILA-Lite-8B-hf` | `hf_models/NVILA-Lite-8B-hf` |
| Mistral-Small 3.1 24B | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `hf_models/Mistral-Small-3.1-24B-Instruct-2503` |
| Pixtral 12B | `mistral-community/pixtral-12b` | `hf_models/pixtral-12b-hf` |
| Qwen3.6 27B | `Qwen/Qwen3.6-27B` | `hf_models/Qwen3.6-27B` |
| Llama-3.2 Vision 11B | `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` | `hf_models/Llama-3.2-11B-Vision-Instruct` |
| InternVL2.5 8B | `OpenGVLab/InternVL2_5-8B` | `hf_models/InternVL2_5-8B` |
| DSE-Qwen2 2B | `MrLight/dse-qwen2-2b-mrl-v1` | `hf_models/dse-qwen2-2b` |
| ColQwen2 v1.0 | `vidore/colqwen2-v1.0-hf` | `hf_models/colqwen2-v1.0-hf` |
| VDR 2B Multi | `llamaindex/vdr-2b-multi-v1` | `hf_models/vdr-2b-multi-v1` |
| Nemotron ColEmbed 4B | `nvidia/nemotron-colembed-vl-4b-v2` | `hf_models/nemotron-colembed-vl-4b-v2` |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `hf_models/clip-vit-base-patch32` |
| Qwen2-VL 7B | `Qwen/Qwen2-VL-7B-Instruct` | `hf_models/Qwen2-VL-7B-Instruct` |
| Qwen2.5 14B | `Qwen/Qwen2.5-14B-Instruct` | `hf_models/Qwen2.5-14B-Instruct` |
| mDeBERTa v3 (mnli-xnli) | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | `hf_models/mDeBERTa-v3-mnli-xnli` |

Run all fifteen with:

```bash
hf download Qwen/Qwen3-VL-30B-A3B-Instruct --local-dir hf_models/Qwen3-VL-30B-A3B-Instruct --token $HF_TOKEN
hf download Efficient-Large-Model/NVILA-Lite-8B-hf --local-dir hf_models/NVILA-Lite-8B-hf --token $HF_TOKEN
hf download mistralai/Mistral-Small-3.1-24B-Instruct-2503 --local-dir hf_models/Mistral-Small-3.1-24B-Instruct-2503 --exclude "consolidated.safetensors" --token $HF_TOKEN
hf download mistral-community/pixtral-12b --local-dir hf_models/pixtral-12b-hf --token $HF_TOKEN
hf download Qwen/Qwen3.6-27B --local-dir hf_models/Qwen3.6-27B --token $HF_TOKEN
hf download unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit --local-dir hf_models/Llama-3.2-11B-Vision-Instruct --token $HF_TOKEN
hf download OpenGVLab/InternVL2_5-8B --local-dir hf_models/InternVL2_5-8B --token $HF_TOKEN
hf download MrLight/dse-qwen2-2b-mrl-v1 --local-dir hf_models/dse-qwen2-2b --token $HF_TOKEN
hf download vidore/colqwen2-v1.0-hf --local-dir hf_models/colqwen2-v1.0-hf --token $HF_TOKEN
hf download llamaindex/vdr-2b-multi-v1 --local-dir hf_models/vdr-2b-multi-v1 --token $HF_TOKEN
hf download nvidia/nemotron-colembed-vl-4b-v2 --local-dir hf_models/nemotron-colembed-vl-4b-v2 --token $HF_TOKEN
hf download openai/clip-vit-base-patch32 --local-dir hf_models/clip-vit-base-patch32 --token $HF_TOKEN
hf download Qwen/Qwen2-VL-7B-Instruct --local-dir hf_models/Qwen2-VL-7B-Instruct --token $HF_TOKEN
hf download Qwen/Qwen2.5-14B-Instruct --local-dir hf_models/Qwen2.5-14B-Instruct --token $HF_TOKEN
hf download MoritzLaurer/mDeBERTa-v3-base-mnli-xnli --local-dir hf_models/mDeBERTa-v3-mnli-xnli --token $HF_TOKEN
```

## ColPali

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

## DocLayout-YOLO

Downloads automatically on the first run of `pdf_extraction.py`. No manual step needed.

| Field | Value |
|---|---|
| Repo ID | `juliozhao/DocLayout-YOLO-DocStructBench` |
| Filename | `doclayout_yolo_docstructbench_imgsz1024.pt` |

## NVILA Patch

**Required.** The model will not load without this.

```bash
cp model_patches/NVILA-Lite-8B-hf/processing_nvila_lite.py hf_models/NVILA-Lite-8B-hf/processing_nvila_lite.py
```
