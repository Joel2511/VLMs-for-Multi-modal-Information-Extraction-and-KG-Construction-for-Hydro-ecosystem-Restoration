import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_DIR = "/home2020/home/icube/janil/Rhine_proj/hf_models/Qwen2.5-14B-Instruct"
ADAPTER_DIR = "/home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/4.training/Qwen2.5-14B-populate-ontology"
MERGED_OUTPUT_DIR = "/home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/5.merge/Qwen2.5-14B-populate-ontology-merged"

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
model.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_OUTPUT_DIR)

print("[DONE] Merge complete.")