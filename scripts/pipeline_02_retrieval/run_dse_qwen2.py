import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from torch.nn.functional import cosine_similarity
import numpy as np
import re

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")

PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
PDF_FOLDER = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Image_text_matching", "dse_qwen2_results")

SEARCH_MODE = 'hybrid'
BETA = 0.02
EMBEDDING_DIM = 1536  # MRL supports 256-1536, 1536 = full quality

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

def get_embedding(last_hidden_state: torch.Tensor, dimension: int) -> torch.Tensor:
    # DSE: take last token hidden state, normalize to unit vector
    reps = last_hidden_state[:, -1]
    reps = torch.nn.functional.normalize(reps[:, :dimension], p=2, dim=-1)
    return reps

def encode_text_query(text: str, processor, model, device: str) -> torch.Tensor:
    # DSE requires a 1x1 dummy image alongside text queries
    dummy_image = Image.new('RGB', (28, 28))
    message = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': dummy_image, 'resized_height': 1, 'resized_width': 1},
                {'type': 'text', 'text': f'Query: {text}'},
            ]
        }
    ]
    text_input = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    ) + "<|endoftext|>"
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding='longest',
        return_tensors='pt'
    ).to(device)
    cache_position = torch.arange(0, 1)
    inputs = model.prepare_inputs_for_generation(
        **inputs, cache_position=cache_position, use_cache=False
    )
    with torch.no_grad():
        output = model(**inputs, return_dict=True, output_hidden_states=True)
    return get_embedding(output.hidden_states[-1], EMBEDDING_DIM)

def encode_image_doc(image: Image.Image, processor, model, device: str) -> torch.Tensor:
    message = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image, 'resized_height': 680, 'resized_width': 680},
                {'type': 'text', 'text': 'What is shown in this image?'}
            ]
        }
    ]
    text_input = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    ) + "<|endoftext|>"
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding='longest',
        return_tensors='pt'
    ).to(device)
    cache_position = torch.arange(0, 1)
    inputs = model.prepare_inputs_for_generation(
        **inputs, cache_position=cache_position, use_cache=False
    )
    with torch.no_grad():
        output = model(**inputs, return_dict=True, output_hidden_states=True)
    return get_embedding(output.hidden_states[-1], EMBEDDING_DIM)

def run_dse_matching():
    print(f"--- Loading DSE-Qwen2-2B [Mode: {SEARCH_MODE.upper()}] ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    model_id = os.path.join(PROJECT_ROOT, "hf_models/dse-qwen2-2b")

    min_pixels = 1 * 28 * 28
    max_pixels = 2560 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels
    )
    processor.tokenizer.padding_side = "left"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval()
    model.padding_side = "left"

    if not os.path.exists(PDF_FOLDER):
        print(f"Error: PDF folder not found at {PDF_FOLDER}")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs to process: {pdf_files}\n")

    for pdf_filename in pdf_files:
        pdf_base = os.path.splitext(pdf_filename)[0]
        print(f"========== STARTING PROCESSING FOR: {pdf_base} ==========")

        IMAGE_FOLDER = os.path.join(BASE_IMAGE_FOLDER, pdf_base)
        JSON_PATH = os.path.join(BASE_JSON_FOLDER, pdf_base, "extracted_content.json")
        OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, pdf_base)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        if not os.path.exists(JSON_PATH) or not os.path.exists(IMAGE_FOLDER):
            print(f"-> SKIPPING {pdf_base}: Required files not found.")
            continue

        print(f"--- Loading Data for {pdf_base} ---")
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)
            raw_blocks = pdf_data["pages"]

        blocks_dict = {k: v for k, v in raw_blocks.items() if len(v.strip()) > 40}
        block_keys = list(blocks_dict.keys())
        block_texts = list(blocks_dict.values())
        block_pages = [get_page_num_from_block_key(k) for k in block_keys]

        print(f"Encoding {len(block_texts)} text blocks...")
        all_text_embeddings = []
        for text in block_texts:
            emb = encode_text_query(text, processor, model, device)
            all_text_embeddings.append(emb)

        image_files = [
            f for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
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
                img_emb = encode_image_doc(image, processor, model, device)

                embeddings_dict[img_name] = img_emb.cpu().float().numpy()

                # Cosine similarity between image embedding and all text embeddings
                raw_sims = []
                for idx in range(len(block_keys)):
                    if SEARCH_MODE == 'local' and abs(img_page - block_pages[idx]) > 1:
                        raw_sims.append(-100.0)
                        continue
                    sim = cosine_similarity(img_emb, all_text_embeddings[idx]).item()
                    raw_sims.append(sim)

                raw_sims = np.array(raw_sims)
                valid_mask = raw_sims > -50.0

                if SEARCH_MODE == 'hybrid' and np.any(valid_mask):
                    norm_sims = normalize_scores(raw_sims[valid_mask])
                    distances = np.abs(img_page - np.array(block_pages)[valid_mask])
                    hybrid_scores = norm_sims - (BETA * distances)
                    final_ranking_scores = np.full_like(raw_sims, -100.0)
                    final_ranking_scores[valid_mask] = hybrid_scores
                else:
                    final_ranking_scores = raw_sims

                k_val = min(5, len(block_keys))
                top_k_indices = np.argsort(final_ranking_scores)[-k_val:][::-1]

                res_entry = {"Image_Name": img_name}
                for i, idx in enumerate(top_k_indices):
                    res_entry[f"Top_{i+1}_Block"] = block_keys[idx]
                    res_entry[f"Top_{i+1}_Score"] = round(final_ranking_scores[idx], 4)

                results.append(res_entry)
                top_1_score = final_ranking_scores[top_k_indices[0]]
                top_scores_for_plot.append(top_1_score)
                print(f"Processed {img_name} -> Best: {res_entry['Top_1_Block']} ({res_entry['Top_1_Score']})")

            except Exception as e:
                print(f"Error {img_name}: {e}")

        np.save(
            os.path.join(OUTPUT_FOLDER, "dse_qwen2_img_embeddings.npy"),
            embeddings_dict,
            allow_pickle=True
        )

        print("--- Saving Output Data ---")
        df = pd.DataFrame(results).sort_values(by="Top_1_Score", ascending=False)
        csv_path = os.path.join(OUTPUT_FOLDER, "dse_qwen2_raw_matrix.csv")
        df.to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 6))
        plt.hist(top_scores_for_plot, bins=15, color='steelblue', edgecolor='black')
        plt.title(f'Top-1 DSE-Qwen2 Cosine Scores ({SEARCH_MODE.upper()}) - {pdf_base}')
        plt.xlabel('Cosine Similarity Score')
        plt.ylabel('Number of Images')
        plt.grid(axis='y', alpha=0.75)
        plot_path = os.path.join(OUTPUT_FOLDER, "dse_qwen2_similarity_histogram.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"CSV and Histogram saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    run_dse_matching()
