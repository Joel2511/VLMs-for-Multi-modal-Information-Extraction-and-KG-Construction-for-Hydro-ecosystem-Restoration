import os
import gc
import json
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration
import time
from datetime import datetime

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")


# --- CONFIGURATION ---
GENERATOR_MODEL_NAME = "llama32vision-11b"
RETRIEVAL_MODEL_NAME = "baseline_noprompt"

PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")
BASE_RETRIEVAL_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Image_text_matching", f"{RETRIEVAL_MODEL_NAME}_results")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Ablation_image_captions", GENERATOR_MODEL_NAME)
JUNK_FOLDER = os.path.join(PROJECT_ROOT, "data", "junk_images")

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise EnvironmentError("HF_TOKEN environment variable not set. Run 'export HF_TOKEN=...' before executing this script.")

MODEL_ID = os.path.join(PROJECT_ROOT, "hf_models/Llama-3.2-11B-Vision-Instruct")


class ContextMissingError(Exception):
    """Raised when retrieval mapping is missing for an image."""
    pass


def format_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def load_retrieval_index(pdf_base: str) -> dict:
    csv_path = os.path.join(BASE_RETRIEVAL_FOLDER, pdf_base, f"{RETRIEVAL_MODEL_NAME}_raw_matrix.csv")
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    index = {}
    for _, row in df.iterrows():
        img_name = row["Image_Name"]
        blocks = []
        for i in range(1, 4):
            col_name = f"Top_{i}_Block"
            if col_name in row and pd.notna(row[col_name]):
                blocks.append(str(row[col_name]))
        index[img_name] = blocks
    return index


def get_image_context(img_name: str, retrieval_index: dict, blocks_dict: dict) -> str:
    if img_name not in retrieval_index:
        raise ContextMissingError(f"Image '{img_name}' not found in CSV.")

    matched_keys = retrieval_index[img_name]
    context_parts = []

    for key in matched_keys:
        text = blocks_dict.get(key, "").strip()
        if text:
            context_parts.append(f"[{key}]: {text}")

    if not context_parts:
        raise ContextMissingError(f"No text found in JSON for blocks matched to {img_name}.")

    return "\n\n---\n\n".join(context_parts)


def build_prompt(image_context: str, image_ocr: str) -> str:
    return "Décris cette image."


def run_llama_captioning():
    pipeline_start = time.time()
    print(f"--- Loading {GENERATOR_MODEL_NAME} with {RETRIEVAL_MODEL_NAME} retrieval context ---")
    print(f"    Generator : {GENERATOR_MODEL_NAME}")
    print(f"    Retriever : {RETRIEVAL_MODEL_NAME}")

    max_memory_map = {0: "28GiB", 1: "28GiB"}


    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        max_memory=max_memory_map,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Model size   : {total_params/1e9:.1f}B parameters")
    print(f"    Loaded at    : {datetime.now().strftime('%H:%M:%S')}")

    if not os.path.exists(BASE_IMAGE_FOLDER):
        print(f"Error: Base image folder not found at {BASE_IMAGE_FOLDER}")
        return

    pdf_folders = [f for f in os.listdir(BASE_IMAGE_FOLDER) if os.path.isdir(os.path.join(BASE_IMAGE_FOLDER, f))]

    for pdf_base in pdf_folders:
        pdf_start = time.time()
        print(f"\n========== PROCESSING [{pdf_base}] ==========")

        IMAGE_FOLDER = os.path.join(BASE_IMAGE_FOLDER, pdf_base)
        OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, pdf_base)
        JUNK_PDF_FOLDER = os.path.join(JUNK_FOLDER, pdf_base)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(JUNK_PDF_FOLDER, exist_ok=True)

        json_output_path = os.path.join(OUTPUT_FOLDER, "image_captions.json")
        if os.path.exists(json_output_path):
            print(f" -> Already captioned. Skipping.")
            continue

        ocr_json_path = os.path.join(BASE_JSON_FOLDER, pdf_base, "ocr_results.json")
        ocr_dict = {}
        if os.path.exists(ocr_json_path):
            with open(ocr_json_path, "r", encoding="utf-8") as f:
                ocr_dict = json.load(f)
        else:
            print(f" -> WARNING: {ocr_json_path} not found. Proceeding with empty OCR context.")

        json_path = os.path.join(BASE_JSON_FOLDER, pdf_base, "extracted_content.json")
        if not os.path.exists(json_path):
            print(f" -> SKIPPING {pdf_base}: No extracted_content.json found.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            blocks_dict = json.load(f).get("pages", {})

        # retrieval_index = load_retrieval_index(pdf_base)
        # if not retrieval_index:
        #     print(f" -> SKIPPING {pdf_base}: CSV missing.")
        #     continue
        retrieval_index = {}

        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        results_dict = {}
        total_images = len(image_files)
        print(f"    Images to caption: {total_images}")

        for idx, img_name in enumerate(image_files, start=1):
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            torch.cuda.empty_cache()
            img_start = time.time()

            try:
                raw_image = Image.open(img_path).convert("RGB")

                image_context = ""
                image_ocr = ""
                caption_prompt = build_prompt(image_context, image_ocr)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": caption_prompt}
                        ],
                    }
                ]

                input_text = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )

                inputs = processor(
                    raw_image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                )

                model_device = next(model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )

                prompt_len = inputs["input_ids"].shape[-1]
                generated_text = processor.decode(
                    generated_ids[0][prompt_len:],
                    skip_special_tokens=True
                ).strip()

                img_elapsed = time.time() - img_start

                if generated_text.strip().lower().startswith("image non informative"):
                    print(f"    [{idx}/{total_images}] {img_name} -> DECORATIVE ({img_elapsed:.1f}s)")
                    results_dict[img_name] = "DECORATIVE"
                else:
                    results_dict[img_name] = generated_text
                    print(f"    [{idx}/{total_images}] {img_name} -> {generated_text[:100]}... ({img_elapsed:.1f}s)")

            except Exception as e:
                print(f"FAILED on {img_name}: {e}")
                results_dict[img_name] = f"ERROR: {str(e)}"
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)

        pdf_elapsed = time.time() - pdf_start
        print(f"========== COMPLETED [{pdf_base}] in {format_elapsed(pdf_elapsed)} ==========")

    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Generator model  : {GENERATOR_MODEL_NAME}")
    print(f"  Retrieval model  : {RETRIEVAL_MODEL_NAME}")
    print(f"  Total runtime    : {format_elapsed(total_elapsed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_llama_captioning()