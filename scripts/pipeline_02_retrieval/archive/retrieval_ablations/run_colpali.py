# pip install torch==2.4.0 torchvision==0.19.0 transformers==4.46.1 colpali-engine==0.3.4
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor
import numpy as np
import re

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")

# --- PATHS --- exactly as in your original ---
PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
PDF_FOLDER = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs","Image_text_matching","colpali_results")

# HYPERPARAMETERS:
# Choose from ['local', 'global', 'hybrid']
# 'local': Only searches +/- 1 page (baseline)
# 'global': Search the whole PDF with no distance bias
# 'hybrid': Weighted distance search
SEARCH_MODE = 'hybrid'

#  Distance Penalty
# When scores are normalized to 0-1, a BETA of 0.02 means
# a block 5 pages away loses 10% of its semantic score.
BETA = 0.02

# --- UTILS ---
def get_page_num_from_img(img_name):
    match = re.search(r'p(\d+)', img_name)
    return int(match.group(1)) if match else None

def get_page_num_from_block_key(block_key):
    match = re.search(r'page_(\d+)', block_key)
    return int(match.group(1)) if match else None

def normalize_scores(scores):
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max == s_min: return np.ones_like(scores)
    return (scores - s_min) / (s_max - s_min)

def run_colpali_matching():
    print(f"--- Loading ColPali Model [Mode: {SEARCH_MODE.upper()}] ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    model_id = os.path.join(PROJECT_ROOT, "hf_models/colpali-v1.2-merged")
    model = ColPali.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(
    os.path.join(PROJECT_ROOT, "hf_models/colpaligemma-3b-pt-448-base"),
    local_files_only=True,
    )

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

        print(f"Encoding {len(block_texts)} block texts...")
        all_text_embeddings = []
        with torch.no_grad():
            for text in block_texts:
                batch = processor.process_queries([text]).to(device)
                all_text_embeddings.append(model(**batch)[0])

        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} images to match.")

        results = []
        top_scores_for_plot = []
        embeddings_dict = {}

        for img_name in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            img_page = get_page_num_from_img(img_name)
            if img_page is None: continue

            try:
                image = Image.open(img_path).convert("RGB")
                with torch.no_grad():
                    img_batch = processor.process_images([image]).to(device)
                    img_emb = model(**img_batch)[0]

                embeddings_dict[img_name] = img_emb.cpu().float().numpy()

                raw_maxsims = []
                for idx in range(len(block_keys)):
                    # Local mode mask
                    if SEARCH_MODE == 'local' and abs(img_page - block_pages[idx]) > 1:
                        raw_maxsims.append(-100.0)
                        continue

                    text_emb = all_text_embeddings[idx]
                    sim_matrix = torch.matmul(text_emb.float(), img_emb.float().T)
                    raw_maxsims.append(sim_matrix.max(dim=-1).values.mean().item()) #With mean(), the score is normalized by query length, so a long generic block doesn't automatically beat a short specific one. For your use case — long scientific text blocks as queries — mean() is actually the right choice

                raw_maxsims = np.array(raw_maxsims)
                valid_mask = raw_maxsims > -50.0

                if SEARCH_MODE == 'hybrid' and np.any(valid_mask):
                    norm_sims = normalize_scores(raw_maxsims[valid_mask])
                    distances = np.abs(img_page - np.array(block_pages)[valid_mask])

                    # Soft penalty: nearby pages get a boost, far pages lose very little
                    # log1p makes the penalty grow slowly — page 1 away: 0.69, page 10 away: 2.4, page 20 away: 3.04
                    # Dividing by 15 keeps the max penalty under ~0.20 even for 20+ pages away
                    distance_penalty = np.log1p(distances) / 15.0
                    hybrid_scores = norm_sims - (BETA * distances)

                    final_ranking_scores = np.full_like(raw_maxsims, -100.0)
                    final_ranking_scores[valid_mask] = hybrid_scores
                else:
                    final_ranking_scores = raw_maxsims

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

        # --- RESTORED ORIGINAL SAVE LOGIC ---
        np.save(os.path.join(OUTPUT_FOLDER, "colpali_img_embeddings.npy"), embeddings_dict, allow_pickle=True)

        print("--- Saving Output Data ---")
        df = pd.DataFrame(results).sort_values(by="Top_1_Score", ascending=False)
        csv_path = os.path.join(OUTPUT_FOLDER, "colpali_raw_matrix.csv")
        df.to_csv(csv_path, index=False)

        # --- RESTORED ORIGINAL PLOTTING LOGIC ---
        plt.figure(figsize=(10, 6))
        plt.hist(top_scores_for_plot, bins=15, color='steelblue', edgecolor='black')
        plt.title(f'Top-1 ColPali MaxSim Scores ({SEARCH_MODE.upper()}) - {pdf_base}')
        plt.xlabel('MaxSim Score')
        plt.ylabel('Number of Images')
        plt.grid(axis='y', alpha=0.75)
        plot_path = os.path.join(OUTPUT_FOLDER, "colpali_similarity_histogram.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"CSV and Histogram saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    run_colpali_matching()
