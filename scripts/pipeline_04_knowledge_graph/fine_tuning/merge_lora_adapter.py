import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get(
        "RHINE_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
).expanduser().resolve()

HF_MODELS_DIR = Path(
    os.environ.get("HF_MODELS_DIR", str(PROJECT_ROOT / "hf_models"))
).expanduser().resolve()

FINE_TUNING_OUTPUT_ROOT = Path(
    os.environ.get(
        "FINE_TUNING_OUTPUT_ROOT",
        str(PROJECT_ROOT / "outputs" / "runs" / "fine_tuning"),
    )
).expanduser().resolve()


BASE_MODEL_DIR = HF_MODELS_DIR / "Qwen2.5-14B-Instruct"
ADAPTER_DIR = FINE_TUNING_OUTPUT_ROOT / "adapters" / "Qwen2.5-14B-populate-ontology"
MERGED_OUTPUT_DIR = FINE_TUNING_OUTPUT_ROOT / "merged_model" / "Qwen2.5-14B-populate-ontology-merged"

print("[MERGE] Loading base model in fp16 (no quantization for merge)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

print("[MERGE] Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

print("[MERGE] Merging adapter into base model...")
model = model.merge_and_unload()

print(f"[MERGE] Saving merged model to {MERGED_OUTPUT_DIR}")
MERGED_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

print("[DONE] Merge complete.")