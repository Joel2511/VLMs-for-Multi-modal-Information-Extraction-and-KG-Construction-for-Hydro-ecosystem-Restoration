# Model Download Instructions

Set your token once per session: export HF_TOKEN="your_token_here"

## Standard downloads
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

## ColPali (two downloads + local merge, official pre-merged repo did not work for this setup)

python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'vidore/colpali-v1.2', '--local-dir', '$RHINE_PROJECT_ROOT/hf_models/colpali-v1.2', '--token', os.environ['HF_TOKEN']], env=env)"

python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'vidore/colpaligemma-3b-pt-448-base', '--local-dir', '$RHINE_PROJECT_ROOT/hf_models/colpaligemma-3b-pt-448-base', '--token', os.environ['HF_TOKEN']], env=env)"

# Merge the LoRA adapter (colpali-v1.2) into the base backbone (colpaligemma-3b-pt-448-base)
python3 -c 'import torch; from colpali_engine.models import ColPali; from peft import PeftModel; from transformers import AutoTokenizer; import os; BASE_MODEL_DIR=os.environ["RHINE_PROJECT_ROOT"]+"/hf_models/colpaligemma-3b-pt-448-base"; ADAPTER_DIR=os.environ["RHINE_PROJECT_ROOT"]+"/hf_models/colpali-v1.2"; MERGED_OUTPUT_DIR=os.environ["RHINE_PROJECT_ROOT"]+"/hf_models/colpali-v1.2-merged"; print("[MERGE] Loading base ColPali backbone (fp32 for a clean merge)..."); base_model=ColPali.from_pretrained(BASE_MODEL_DIR,torch_dtype=torch.float32,local_files_only=True); print("[MERGE] Loading LoRA adapter..."); model=PeftModel.from_pretrained(base_model,ADAPTER_DIR,local_files_only=True); print("[MERGE] Merging adapter into base model..."); model=model.merge_and_unload(); print(f"[MERGE] Saving merged model to {MERGED_OUTPUT_DIR}"); model.save_pretrained(MERGED_OUTPUT_DIR,safe_serialization=True); tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL_DIR,local_files_only=True); tokenizer.save_pretrained(MERGED_OUTPUT_DIR); print("[DONE] Merge complete.")'

# IMPORTANT: keep colpaligemma-3b-pt-448-base after merging -- it is NOT redundant.
# The merged model only holds weights; ColPaliProcessor.from_pretrained() loads
# preprocessing config (image sizing, patching) from colpaligemma-3b-pt-448-base
# directly at inference time. Deleting the base folder will break the pipeline
# even though the model itself loads from colpali-v1.2-merged.

## DocLayout-YOLO (auto-downloads on first pdf_extraction.py run)
# repo_id: juliozhao/DocLayout-YOLO-DocStructBench
# filename: doclayout_yolo_docstructbench_imgsz1024.pt

## NVILA patch (REQUIRED -- model will not load without this)
cp model_patches/NVILA-Lite-8B-hf/processing_nvila_lite.py hf_models/NVILA-Lite-8B-hf/processing_nvila_lite.py
