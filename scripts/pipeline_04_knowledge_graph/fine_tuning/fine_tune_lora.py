import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

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



def format_chat_template(dataset, tokenizer):
    def transform_row(row):
        row_json = json.loads(row["raw"])
        messages = row_json["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        assistant_content = messages[-1]["content"]
        row["prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        row["completion"] = assistant_content + tokenizer.eos_token
        return row
    return dataset.map(transform_row)


MODEL_NAME = "Qwen2.5-14B-Instruct"
model_directory = HF_MODELS_DIR / MODEL_NAME

DATASET_PATH = FINE_TUNING_OUTPUT_ROOT / "hf_dataset"
OUTPUT_DIR = FINE_TUNING_OUTPUT_ROOT / "adapters" / "Qwen2.5-14B-populate-ontology"

torch_dtype = torch.float16
attn_implementation = "eager"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    model_directory,
    quantization_config=bnb_config,
    device_map={"": torch.cuda.current_device()},
    attn_implementation=attn_implementation,
)

from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

tokenizer = AutoTokenizer.from_pretrained(model_directory)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"],
)


dataset = load_from_disk(DATASET_PATH)
dataset = dataset.shuffle(seed=65)
dataset = format_chat_template(dataset, tokenizer)
dataset = dataset.train_test_split(test_size=0.1)

training_arguments = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.25,
    save_strategy="steps",
    save_steps=0.25,
    save_total_limit=2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    max_length=8192,
    packing=False,
    completion_only_loss=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
)

trainer.train()

model.config.use_cache = True
trainer.model.save_pretrained(OUTPUT_DIR)