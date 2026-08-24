# Nemotron ColEmbed VL-4B-V2 image–text matching for French PDFs
# Inputs and outputs follow the same conventions used across the other retrieval models in this pipeline.

import os
import json
import re
import math
from typing import List, Dict

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, AutoProcessor

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")


# --- PATHS ---
PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
PDF_FOLDER = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")

# Dedicated Nemotron output folder (parallel to colpali_results, etc.)
BASE_OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT, "outputs", "Image_text_matching","nemotron_results"
)


# HYPERPARAMETERS
# SEARCH_MODE: 'local' | 'global' | 'hybrid'
SEARCH_MODE = "hybrid"

# Distance penalty
BETA = 0.02


# --- UTILS  ---

def get_page_num_from_img(img_name: str) -> int | None:
    match = re.search(r"p(\d+)", img_name)
    return int(match.group(1)) if match else None


def get_page_num_from_block_key(block_key: str) -> int | None:
    match = re.search(r"page_(\d+)", block_key)
    return int(match.group(1)) if match else None


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max == s_min:
        return np.ones_like(scores)
    return (scores - s_min) / (s_max - s_min)


# --- NEMOTRON HELPERS ---

def load_nemotron_model(device: str):
    """
    Load Nemotron ColEmbed VL-4B-V2 with its custom remote code.

    We rely on the EmbeddingMixin + ColBERTScoringMixin defined in
    modeling_qwen3_vl_nemotron_embed.py:
      - forward_queries(...)
      - forward_images(...)
      - colbert_score(qs, ps)
    """
    model_id = os.path.join(PROJECT_ROOT, "hf_models/nemotron-colembed-vl-4b-v2")

    # trust_remote_code=True is required to load Qwen3VLNemotronEmbedForConditionalGeneration
    # with EmbeddingMixin + ColBERTScoringMixin.
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    ).eval()

    # Processor is only needed internally by model._get_processor(), but loading it here
    # ensures all assets are on disk and avoids first-call download surprises.
    AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    return model


def encode_text_blocks(
    model, texts: List[str], device: str, batch_size: int = 4
) -> List[torch.Tensor]:
    """
    Use Nemotron's forward_queries() to get per-block multi-vector embeddings.


    Returned as a list of 2D tensors, each [seq_len, dim], matching the
    expectations of colbert_score(qs, ps).
    """
    for i, text in enumerate(texts):
        approx_tokens = len(text.split())
        if approx_tokens > 400:
            print(f"  [LENGTH-CHECK] block {i}: ~{approx_tokens} words, may approach q_max_length=512 cap")

    # forward_queries returns a tensor [N, max_seq_len, hidden_dim]
    with torch.no_grad():
        q_emb = model.forward_queries(texts, batch_size=batch_size)


    # Split into list of [seq_len_i, dim] by trimming trailing all-zero rows
    q_list: List[torch.Tensor] = []
    for i in range(q_emb.shape[0]):
        emb = q_emb[i]  # [max_seq_len, dim]
        # Identify non-zero rows by L2 norm > 0
        norms = emb.norm(dim=-1)
        valid = norms > 0
        if valid.any():
            emb = emb[valid]
        q_list.append(emb.cpu())
    return q_list


def encode_images(
    model, images: List[Image.Image], device: str, batch_size: int = 4
) -> List[torch.Tensor]:
    """
    Use Nemotron's forward_images() to get per-image multi-vector embeddings.

    Returned as a list of 2D tensors, each [seq_len, dim], as required by
    colbert_score().
    """
    with torch.no_grad():
        img_emb = model.forward_images(images, batch_size=batch_size)

    img_list: List[torch.Tensor] = []
    for i in range(img_emb.shape[0]):
        emb = img_emb[i]
        norms = emb.norm(dim=-1)
        valid = norms > 0
        if valid.any():
            emb = emb[valid]
        print(f"  [PATCH-CHECK] image {i}: {emb.shape[0]} real visual tokens (raw padded shape: {img_emb.shape[1]}, cap=1024)")
        img_list.append(emb.cpu())
    return img_list


def run_nemotron_matching():
    print(f"--- Loading Nemotron ColEmbed VL-4B-V2 [Mode: {SEARCH_MODE.upper()}] ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    model = load_nemotron_model(device)

    if not os.path.exists(PDF_FOLDER):
        print(f"Error: PDF folder not found at {PDF_FOLDER}")
        return

    pdf_files = [
        f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")
    ]
    print(f"Found {len(pdf_files)} PDFs to process: {pdf_files}\n")

    for pdf_filename in pdf_files:
        pdf_base = os.path.splitext(pdf_filename)[0]
        print(f"========== STARTING PROCESSING FOR: {pdf_base} ==========")

        IMAGE_FOLDER = os.path.join(BASE_IMAGE_FOLDER, pdf_base)
        JSON_PATH = os.path.join(
            BASE_JSON_FOLDER, pdf_base, "extracted_content.json"
        )
        OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, pdf_base)

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        if not os.path.exists(JSON_PATH) or not os.path.exists(IMAGE_FOLDER):
            print(f"-> SKIPPING {pdf_base}: Required files not found.")
            continue

        print(f"--- Loading Data for {pdf_base} ---")
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            pdf_data = json.load(f)
            raw_blocks: Dict[str, str] = pdf_data["pages"]

        # Filter very short blocks identically to ColPali code
        blocks_dict = {
            k: v for k, v in raw_blocks.items() if len(v.strip()) > 40
        }
        block_keys = list(blocks_dict.keys())
        block_texts = list(blocks_dict.values())
        block_pages = [get_page_num_from_block_key(k) for k in block_keys]

        print(f"Encoding {len(block_texts)} block texts with Nemotron...")
        text_embeddings = encode_text_blocks(
            model, block_texts, device=device, batch_size=4
        )  # list of [seq_len, dim]

        image_files = [
            f
            for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"Found {len(image_files)} images to match.")

        results = []
        top_scores_for_plot = []
        embeddings_dict = {}

        # Images are processed one at a time here to keep results aligned with image_files; no cross-image batching is applied in this loop.
        for img_name in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            img_page = get_page_num_from_img(img_name)
            if img_page is None:
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                # Nemotron forward_images expects a list of PIL images
                img_emb_list = encode_images(
                    model, [image], device=device, batch_size=1
                )
                assert len(img_emb_list) == 1
                img_emb = img_emb_list[0]  # [seq_len, dim] on CPU

                embeddings_dict[img_name] = img_emb.numpy()

                raw_scores = []

                for idx in range(len(block_keys)):
                    if (
                        SEARCH_MODE == "local"
                        and abs(img_page - block_pages[idx]) > 1
                    ):
                        raw_scores.append(-100.0)
                        continue

                    q_emb = text_embeddings[idx].to(device)   # [q_len, dim]
                    p_emb = img_emb.to(device)                # [p_len, dim]

                    with torch.no_grad():
                        sim_matrix = torch.matmul(q_emb.float(), p_emb.float().T)   # [q_len, p_len]
                        maxsim_per_query_token = sim_matrix.max(dim=-1).values      # [q_len]
                        score = maxsim_per_query_token.mean().item()                # normalize by query length

                    raw_scores.append(score)

                raw_scores = np.array(raw_scores, dtype=np.float32)

                valid_mask = raw_scores > -50.0

                if SEARCH_MODE == "hybrid" and np.any(valid_mask):
                    norm_sims = normalize_scores(raw_scores[valid_mask])
                    distances = np.abs(
                        img_page - np.array(block_pages, dtype=np.int32)[valid_mask]
                    )

                    hybrid_scores = norm_sims - (BETA * distances)

                    final_ranking_scores = np.full_like(
                        raw_scores, -100.0, dtype=np.float32
                    )
                    final_ranking_scores[valid_mask] = hybrid_scores
                else:
                    final_ranking_scores = raw_scores

                k_val = min(5, len(block_keys))
                top_k_indices = np.argsort(final_ranking_scores)[-k_val:][::-1]

                res_entry = {"Image_Name": img_name}
                for i, idx in enumerate(top_k_indices):
                    res_entry[f"Top_{i+1}_Block"] = block_keys[idx]
                    res_entry[f"Top_{i+1}_Score"] = round(
                        float(final_ranking_scores[idx]), 4
                    )

                results.append(res_entry)
                top_1_score = float(final_ranking_scores[top_k_indices[0]])
                top_scores_for_plot.append(top_1_score)

                print(
                    f"Processed {img_name} -> Best: "
                    f"{res_entry['Top_1_Block']} ({res_entry['Top_1_Score']})"
                )

            except Exception as e:
                print(f"Error {img_name}: {e}")

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Save embeddings dict as NPY
        np.save(
            os.path.join(OUTPUT_FOLDER, "nemotron_img_embeddings.npy"),
            embeddings_dict,
            allow_pickle=True,
        )

        print("--- Saving Output Data ---")
        df = pd.DataFrame(results).sort_values(
            by="Top_1_Score", ascending=False
        )
        csv_path = os.path.join(OUTPUT_FOLDER, "nemotron_raw_matrix.csv")
        df.to_csv(csv_path, index=False)

        # Histogram of top-1 scores
        plt.figure(figsize=(10, 6))
        plt.hist(
            top_scores_for_plot,
            bins=15,
            color="darkgreen",
            edgecolor="black",
        )
        plt.title(
            f"Top-1 Nemotron ColEmbed Scores ({SEARCH_MODE.upper()}) - {pdf_base}"
        )
        plt.xlabel("Score")
        plt.ylabel("Number of Images")
        plt.grid(axis="y", alpha=0.75)
        plot_path = os.path.join(
            OUTPUT_FOLDER, "nemotron_similarity_histogram.png"
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"CSV and Histogram saved to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    run_nemotron_matching()