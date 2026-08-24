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

## ColPali (two downloads + local merge required)
hf download vidore/colpali-v1.2 --local-dir hf_models/colpali-v1.2 --token $HF_TOKEN
hf download vidore/colpaligemma-3b-pt-448-base --local-dir hf_models/colpaligemma-3b-pt-448-base --token $HF_TOKEN
# TODO: run merge step to produce hf_models/colpali-v1.2-merged

## DocLayout-YOLO (auto-downloads on first pdf_extraction.py run)
# repo_id: juliozhao/DocLayout-YOLO-DocStructBench
# filename: doclayout_yolo_docstructbench_imgsz1024.pt

## NVILA patch (REQUIRED — model will not load without this)
cp model_patches/NVILA-Lite-8B-hf/processing_nvila_lite.py hf_models/NVILA-Lite-8B-hf/processing_nvila_lite.py
