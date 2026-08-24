#Pipeline 2 — Family 1 (single-vector dense bi-encoder) retrieval using VDR (vidore/vdr-2b-multi-v1)

import os
import json
import math
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
PDF_FOLDER = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs","Image_text_matching", "vidore_results")

SEARCH_MODE = 'hybrid'
BETA = 0.02

MAX_PIXELS = 768 * 28 * 28
MIN_PIXELS = 1 * 28 * 28
EMBED_DIM = 1536

DOCUMENT_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|endoftext|>"
QUERY_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Query: %s<|im_end|>\n<|endoftext|>"


def get_page_num_from_img(img_name):
    match = re.search(r'p(\d+)', img_name)
    return int(match.group(1)) if match else None


def get_page_num_from_block_key(block_key):
    match = re.search(r'page_(\d+)', block_key)
    return int(match.group(1)) if match else None


def normalize_scores(scores):
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max == s_min:
        return np.ones_like(scores)
    return (scores - s_min) / (s_max - s_min)


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor

# Dynamic-resolution patch resizing utilities (Qwen2-VL native resizing scheme)
def smart_resize(height, width):
    h_bar = max(28, round_by_factor(height, 28))
    w_bar = max(28, round_by_factor(width, 28))
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = floor_by_factor(height / beta, 28)
        w_bar = floor_by_factor(width / beta, 28)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = ceil_by_factor(height * beta, 28)
        w_bar = ceil_by_factor(width * beta, 28)
    return w_bar, h_bar


def resize_image(image):
    new_size = smart_resize(image.height, image.width)
    return image.resize(new_size)


def l2_normalize(x):
    return torch.nn.functional.normalize(x, p=2, dim=-1)

# Encode a text query into the shared embedding space; a blank placeholder image is required because the underlying VLM architecture expects a vision token even for text-only input
def encode_text(text, model, processor, device):
    dummy_image = Image.new("RGB", (56, 56))
    inputs = processor(
        text=[QUERY_PROMPT % text],
        images=[dummy_image],
        videos=None,
        padding="longest",
        return_tensors="pt"
    ).to(device)

    cache_position = torch.arange(0, 1, device=device)
    model_inputs = model.prepare_inputs_for_generation(
        **inputs,
        cache_position=cache_position,
        use_cache=False
    )

    with torch.no_grad():
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)

    emb = outputs.hidden_states[-1][:, -1][:, :EMBED_DIM]
    return l2_normalize(emb)

# Encode a document image into the shared embedding space using the fixed document prompt template
def encode_image(image, model, processor, device):
    image = resize_image(image)
    inputs = processor(
        text=[DOCUMENT_PROMPT],
        images=[image],
        videos=None,
        padding="longest",
        return_tensors="pt"
    ).to(device)

    cache_position = torch.arange(0, 1, device=device)
    model_inputs = model.prepare_inputs_for_generation(
        **inputs,
        cache_position=cache_position,
        use_cache=False
    )

    with torch.no_grad():
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)

    emb = outputs.hidden_states[-1][:, -1][:, :EMBED_DIM]
    return l2_normalize(emb)


def run_vidore_matching():
    print(f"--- Loading ViDoRe VDR Model [Mode: {SEARCH_MODE.upper()}] ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    model_id = "/home2020/home/icube/janil/Rhine_proj/hf_models/vdr-2b-multi-v1"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS
    )

    model.padding_side = "left"
    processor.tokenizer.padding_side = "left"

    if not os.path.exists(PDF_FOLDER):
        print(f"Error: PDF folder not found at {PDF_FOLDER}")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs to process: {pdf_files}\n")

    for pdf_filename in pdf_files:
        pdf_base = os.path.splitext(pdf_filename)[0]
        print(f"========== STARTING PROCESSING FOR: {pdf_base} ==========")

        IMAGE_FOLDER = os.path.join(BASE_IMAGE_FOLDER, pdf_base)
        JSON_PATH = os.path.join(BASE_JSON_FOLDER, pdf_base, "extracted_content.json")
        OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, pdf_base)

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        csv_path = os.path.join(OUTPUT_FOLDER, "vidore_raw_matrix.csv")
        if os.path.exists(csv_path):
            print(f"-> SKIPPING {pdf_base}: ViDoRe CSV already exists at {csv_path}.")
            continue

        if not os.path.exists(JSON_PATH) or not os.path.exists(IMAGE_FOLDER):
            print(f"-> SKIPPING {pdf_base}: Required files not found.")
            continue

        print(f"--- Loading Data for {pdf_base} ---")
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            pdf_data = json.load(f)
            raw_blocks = pdf_data["pages"]

        blocks_dict = {k: v for k, v in raw_blocks.items() if len(v.strip()) > 40}
        block_keys = list(blocks_dict.keys())
        block_texts = list(blocks_dict.values())
        block_pages = [get_page_num_from_block_key(k) for k in block_keys]

        print(f"Encoding {len(block_texts)} block texts...")
        all_text_embeddings = []
        for text in block_texts:
            all_text_embeddings.append(encode_text(text, model, processor, device))
        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"Found {len(image_files)} images to match.")

        results = []
        top_scores_for_plot = []
        embeddings_dict = {}

        for img_name in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            img_page = get_page_num_from_img(img_name)
            if img_page is None:
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                img_emb = encode_image(image, model, processor, device)

                embeddings_dict[img_name] = img_emb.cpu().float().numpy()

                raw_scores = torch.matmul(img_emb, all_text_embeddings.T).squeeze(0).cpu().float().numpy()
                # Exploratory strict page-window mode; not used for reported results
                if SEARCH_MODE == "local":
                    masked_scores = raw_scores.copy()
                    for idx in range(len(block_keys)):
                        if abs(img_page - block_pages[idx]) > 1:
                            masked_scores[idx] = -100.0
                    final_ranking_scores = masked_scores
                # Hybrid page-proximity re-weighting: soft penalty discourages implausible long-range matches without a rigid window
                elif SEARCH_MODE == "hybrid":
                    valid_mask = np.ones_like(raw_scores, dtype=bool)
                    norm_sims = normalize_scores(raw_scores[valid_mask])
                    distances = np.abs(img_page - np.array(block_pages)[valid_mask])
                    hybrid_scores = norm_sims - (BETA * distances)

                    final_ranking_scores = np.full_like(raw_scores, -100.0)
                    final_ranking_scores[valid_mask] = hybrid_scores

                else:
                    final_ranking_scores = raw_scores

                k_val = min(5, len(block_keys))
                top_k_indices = np.argsort(final_ranking_scores)[-k_val:][::-1]

                res_entry = {"Image_Name": img_name}
                for i, idx in enumerate(top_k_indices):
                    res_entry[f"Top_{i+1}_Block"] = block_keys[idx]
                    res_entry[f"Top_{i+1}_Score"] = round(float(final_ranking_scores[idx]), 4)

                results.append(res_entry)
                top_1_score = final_ranking_scores[top_k_indices[0]]
                top_scores_for_plot.append(float(top_1_score))
                print(f"Processed {img_name} -> Best: {res_entry['Top_1_Block']} ({res_entry['Top_1_Score']})")

            except Exception as e:
                print(f"Error {img_name}: {e}")

        np.save(os.path.join(OUTPUT_FOLDER, "vidore_img_embeddings.npy"), embeddings_dict, allow_pickle=True)

        print("--- Saving Output Data ---")
        df = pd.DataFrame(results).sort_values(by="Top_1_Score", ascending=False)
        csv_path = os.path.join(OUTPUT_FOLDER, "vidore_raw_matrix.csv")
        df.to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 6))
        plt.hist(top_scores_for_plot, bins=15, color="darkorange", edgecolor="black")
        plt.title(f"Top-1 ViDoRe Similarity Scores ({SEARCH_MODE.upper()}) - {pdf_base}")
        plt.xlabel("Similarity Score")
        plt.ylabel("Number of Images")
        plt.grid(axis="y", alpha=0.75)
        plot_path = os.path.join(OUTPUT_FOLDER, "vidore_similarity_histogram.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"CSV and Histogram saved to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    run_vidore_matching()