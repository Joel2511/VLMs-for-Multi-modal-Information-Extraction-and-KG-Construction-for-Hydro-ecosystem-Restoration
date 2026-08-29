import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")

# --- PATHS ---
PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
PDF_FOLDER = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Image_text_matching", "clip_results")


def run_clip_matching():
    print("--- Loading CLIP Model (Using SafeTensors) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device.upper()}")

    model = CLIPModel.from_pretrained(
        os.path.join(PROJECT_ROOT, "hf_models/clip-vit-base-patch32")
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        os.path.join(PROJECT_ROOT, "hf_models/clip-vit-base-patch32")
    )

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

        if not os.path.exists(JSON_PATH):
            print(f"-> SKIPPING {pdf_base}: JSON file not found at {JSON_PATH}")
            continue

        if not os.path.exists(IMAGE_FOLDER):
            print(f"-> SKIPPING {pdf_base}: Image folder not found at {IMAGE_FOLDER}")
            continue

        print(f"--- Loading Data for {pdf_base} ---")
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            pdf_data = json.load(f)
            raw_blocks = pdf_data["pages"]

        # Uses identical text-block granularity across all retrieval models for a fair comparison
        blocks_dict = {k: v for k, v in raw_blocks.items() if len(v.strip()) > 40}
        block_keys = list(blocks_dict.keys())
        block_texts = list(blocks_dict.values())

        if len(block_keys) == 0:
            print(f"-> SKIPPING {pdf_base}: No valid text blocks found in JSON.")
            continue
        # CLIP's fixed 77-token ceiling truncates long French paragraphs; this is a known architectural limit, not a pipeline design choice
        print(f"Encoding {len(block_keys)} text blocks...")
        text_inputs = processor(
            text=block_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)

        with torch.no_grad():
            text_outputs = model.get_text_features(**text_inputs)
            text_features = (
                text_outputs.pooler_output
                if hasattr(text_outputs, "pooler_output")
                else text_outputs
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_files = [
            f for f in os.listdir(IMAGE_FOLDER)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        print(f"Found {len(image_files)} images to match against {len(block_keys)} text blocks.")

        if len(image_files) == 0:
            print(f"-> SKIPPING {pdf_base}: No images inside folder.")
            continue

        results = []
        top_scores_for_plot = []
        embeddings_dict = {}

        print("--- Calculating Similarity Matrix ---")
        for img_name in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            try:
                image = Image.open(img_path).convert("RGB")

                image_inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    image_outputs = model.get_image_features(**image_inputs)
                    image_features = (
                        image_outputs.pooler_output
                        if hasattr(image_outputs, "pooler_output")
                        else image_outputs
                    )
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).squeeze(0)

                embeddings_dict[img_name] = image_features.cpu().numpy().flatten()

                k_val = min(5, len(block_keys))
                top_k_values, top_k_indices = similarity.topk(k_val)

                top_1_score = top_k_values[0].item()
                top_scores_for_plot.append(top_1_score)

                res_entry = {"Image_Name": img_name}
                for i in range(k_val):
                    idx = top_k_indices[i].item()
                    res_entry[f"Top_{i+1}_Block"] = block_keys[idx]
                    res_entry[f"Top_{i+1}_Score"] = round(top_k_values[i].item(), 4)

                results.append(res_entry)

                print(
                    f"Processed {img_name} -> Best Match: "
                    f"{res_entry['Top_1_Block']} ({round(top_1_score, 4)})"
                )

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

        np.save(
            os.path.join(OUTPUT_FOLDER, "clip_img_embeddings.npy"),
            embeddings_dict,
        )

        print("--- Saving Output Data ---")
        df = pd.DataFrame(results).sort_values(by="Top_1_Score", ascending=False)
        csv_path = os.path.join(OUTPUT_FOLDER, "clip_raw_matrix.csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV saved to: {csv_path}")

        plt.figure(figsize=(10, 6))
        plt.hist(top_scores_for_plot, bins=15, color="skyblue", edgecolor="black")
        plt.title(f"Distribution of Top-1 CLIP Similarity Scores ({pdf_base})")
        plt.xlabel("Cosine Similarity Score")
        plt.ylabel("Number of Images")
        plt.grid(axis="y", alpha=0.75)

        plot_path = os.path.join(OUTPUT_FOLDER, "similarity_histogram.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Histogram saved to: {plot_path}")


        print(f"========== COMPLETED PROCESSING FOR: {pdf_base} ==========\n")


if __name__ == "__main__":
    run_clip_matching()
