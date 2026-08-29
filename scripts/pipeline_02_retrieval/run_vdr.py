"""
Pipeline 2: single-vector dense image-text retrieval using VDR.

This implementation evaluates the VDR 2B Multi retrieval configuration used
for the selected downstream captioning pipeline. It preserves the original
dynamic-resolution preprocessing, shared-embedding extraction, hybrid
page-proximity re-ranking, CSV output schema, and similarity histogram logic.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ─── Repository and experiment configuration ─────────────────────────────────

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("RHINE_PROJECT_ROOT", str(SCRIPT_ROOT))
).expanduser().resolve()

HF_MODELS_DIR = Path(
    os.environ.get("HF_MODELS_DIR", str(PROJECT_ROOT / "hf_models"))
).expanduser().resolve()

SEARCH_MODE = "hybrid"
BETA = 0.02

MAX_PIXELS = 768 * 28 * 28
MIN_PIXELS = 1 * 28 * 28
EMBED_DIM = 1536

MODEL_ID = HF_MODELS_DIR / "vdr-2b-multi-v1"

DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "extracted_images"
DEFAULT_JSON_ROOT = PROJECT_ROOT / "outputs" / "json_results"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "retrieval"
    / "vidore_results"
)

DOCUMENT_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is shown in this image?<|im_end|>\n<|endoftext|>"
)

QUERY_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Query: %s<|im_end|>\n<|endoftext|>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run VDR image-text retrieval on one extracted document folder "
            "or every available document."
        )
    )

    parser.add_argument(
        "--document",
        default="all",
        help=(
            "Name of one document folder under --image-root, or 'all' to "
            "process every available document. Default: all."
        ),
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help=(
            "Root directory containing per-document extracted image folders. "
            f"Default: {DEFAULT_IMAGE_ROOT}"
        ),
    )

    parser.add_argument(
        "--json-root",
        type=Path,
        default=DEFAULT_JSON_ROOT,
        help=(
            "Root directory containing per-document extracted_content.json "
            f"files. Default: {DEFAULT_JSON_ROOT}"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory for newly generated VDR retrieval results. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )

    return parser.parse_args()


def get_page_num_from_img(img_name: str):
    match = re.search(r"p(\d+)", img_name)
    return int(match.group(1)) if match else None


def get_page_num_from_block_key(block_key: str):
    match = re.search(r"page_(\d+)", block_key)
    return int(match.group(1)) if match else None


def normalize_scores(scores):
    score_min, score_max = np.min(scores), np.max(scores)

    if score_max == score_min:
        return np.ones_like(scores)

    return (scores - score_min) / (score_max - score_min)


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


# Dynamic-resolution patch resizing utilities using the Qwen2-VL resizing scheme.
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


def l2_normalize(tensor):
    return torch.nn.functional.normalize(tensor, p=2, dim=-1)


# A blank placeholder image is required because the underlying VLM architecture
# expects a vision token even when encoding text-only retrieval queries.
def encode_text(text, model, processor, device):
    dummy_image = Image.new("RGB", (56, 56))

    inputs = processor(
        text=[QUERY_PROMPT % text],
        images=[dummy_image],
        videos=None,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    cache_position = torch.arange(0, 1, device=device)

    model_inputs = model.prepare_inputs_for_generation(
        **inputs,
        cache_position=cache_position,
        use_cache=False,
    )

    with torch.no_grad():
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )

    embedding = outputs.hidden_states[-1][:, -1][:, :EMBED_DIM]
    return l2_normalize(embedding)


def encode_image(image, model, processor, device):
    image = resize_image(image)

    inputs = processor(
        text=[DOCUMENT_PROMPT],
        images=[image],
        videos=None,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    cache_position = torch.arange(0, 1, device=device)

    model_inputs = model.prepare_inputs_for_generation(
        **inputs,
        cache_position=cache_position,
        use_cache=False,
    )

    with torch.no_grad():
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )

    embedding = outputs.hidden_states[-1][:, -1][:, :EMBED_DIM]
    return l2_normalize(embedding)


def select_document_folders(image_root: Path, document: str) -> list[Path]:
    if not image_root.exists():
        print(f"[ERROR] Image root does not exist: {image_root}")
        sys.exit(1)

    available_folders = sorted(
        folder for folder in image_root.iterdir() if folder.is_dir()
    )

    if not available_folders:
        print(f"[ERROR] No document folders found in {image_root}")
        sys.exit(1)

    if document == "all":
        return available_folders

    selected_folder = image_root / document

    if not selected_folder.is_dir():
        available_names = ", ".join(folder.name for folder in available_folders)
        print(f"[ERROR] Requested document folder not found: {document}")
        print(f"[INFO] Available document folders: {available_names}")
        sys.exit(1)

    return [selected_folder]


def run_vidore_matching(args: argparse.Namespace) -> None:
    print(f"--- Loading ViDoRe VDR Model [Mode: {SEARCH_MODE.upper()}] ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_root = args.image_root.expanduser().resolve()
    json_root = args.json_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    print(f"Running on device: {device.upper()}")
    print(f"Image root: {image_root}")
    print(f"JSON root: {json_root}")
    print(f"Output root: {output_root}")
    print(f"Document selection: {args.document}")

    if not MODEL_ID.exists():
        print(f"[ERROR] VDR model directory not found: {MODEL_ID}")
        print(
            "[INFO] Download the model as described in MODEL_SETUP.md or set "
            "HF_MODELS_DIR in config/project.env."
        )
        sys.exit(1)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    model.padding_side = "left"
    processor.tokenizer.padding_side = "left"

    pdf_folders = select_document_folders(image_root, args.document)

    print(
        f"Found {len(pdf_folders)} document folder(s) to process: "
        f"{[folder.name for folder in pdf_folders]}\n"
    )

    for pdf_folder in pdf_folders:
        pdf_base = pdf_folder.name

        print(f"========== STARTING PROCESSING FOR: {pdf_base} ==========")

        image_folder = pdf_folder
        json_path = json_root / pdf_base / "extracted_content.json"
        output_folder = output_root / pdf_base

        output_folder.mkdir(parents=True, exist_ok=True)

        csv_path = output_folder / "vidore_raw_matrix.csv"

        if csv_path.exists():
            print(
                f"-> SKIPPING {pdf_base}: ViDoRe CSV already exists at "
                f"{csv_path}."
            )
            continue

        if not json_path.exists() or not image_folder.exists():
            print(f"-> SKIPPING {pdf_base}: Required files not found.")
            continue

        print(f"--- Loading Data for {pdf_base} ---")

        with open(json_path, "r", encoding="utf-8") as file:
            pdf_data = json.load(file)
            raw_blocks = pdf_data["pages"]

        blocks_dict = {
            key: value
            for key, value in raw_blocks.items()
            if len(value.strip()) > 40
        }

        block_keys = list(blocks_dict.keys())
        block_texts = list(blocks_dict.values())
        block_pages = [
            get_page_num_from_block_key(key)
            for key in block_keys
        ]

        print(f"Encoding {len(block_texts)} block texts...")

        all_text_embeddings = []

        for text in block_texts:
            all_text_embeddings.append(
                encode_text(text, model, processor, device)
            )

        all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

        image_files = sorted(
            file.name
            for file in image_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        print(f"Found {len(image_files)} images to match.")

        results = []
        top_scores_for_plot = []
        embeddings_dict = {}

        for img_name in image_files:
            img_path = image_folder / img_name
            img_page = get_page_num_from_img(img_name)

            if img_page is None:
                continue

            try:
                image = Image.open(img_path).convert("RGB")

                img_emb = encode_image(
                    image,
                    model,
                    processor,
                    device,
                )

                embeddings_dict[img_name] = img_emb.cpu().float().numpy()

                raw_scores = (
                    torch.matmul(img_emb, all_text_embeddings.T)
                    .squeeze(0)
                    .cpu()
                    .float()
                    .numpy()
                )

                # Exploratory strict page-window mode; not used for thesis results.
                if SEARCH_MODE == "local":
                    masked_scores = raw_scores.copy()

                    for index in range(len(block_keys)):
                        if abs(img_page - block_pages[index]) > 1:
                            masked_scores[index] = -100.0

                    final_ranking_scores = masked_scores

                # Hybrid page-proximity re-weighting used for reported results.
                elif SEARCH_MODE == "hybrid":
                    valid_mask = np.ones_like(raw_scores, dtype=bool)
                    norm_sims = normalize_scores(raw_scores[valid_mask])

                    distances = np.abs(
                        img_page - np.array(block_pages)[valid_mask]
                    )

                    hybrid_scores = norm_sims - (BETA * distances)

                    final_ranking_scores = np.full_like(raw_scores, -100.0)
                    final_ranking_scores[valid_mask] = hybrid_scores

                else:
                    final_ranking_scores = raw_scores

                k_val = min(5, len(block_keys))
                top_k_indices = np.argsort(final_ranking_scores)[-k_val:][::-1]

                result_entry = {"Image_Name": img_name}

                for rank, index in enumerate(top_k_indices, start=1):
                    result_entry[f"Top_{rank}_Block"] = block_keys[index]
                    result_entry[f"Top_{rank}_Score"] = round(
                        float(final_ranking_scores[index]),
                        4,
                    )

                results.append(result_entry)

                top_1_score = final_ranking_scores[top_k_indices[0]]
                top_scores_for_plot.append(float(top_1_score))

                print(
                    f"Processed {img_name} -> Best: "
                    f"{result_entry['Top_1_Block']} "
                    f"({result_entry['Top_1_Score']})"
                )

            except Exception as error:
                print(f"Error {img_name}: {error}")

        np.save(
            output_folder / "vidore_img_embeddings.npy",
            embeddings_dict,
            allow_pickle=True,
        )

        print("--- Saving Output Data ---")

        dataframe = pd.DataFrame(results).sort_values(
            by="Top_1_Score",
            ascending=False,
        )

        dataframe.to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 6))
        plt.hist(
            top_scores_for_plot,
            bins=15,
            color="darkorange",
            edgecolor="black",
        )
        plt.title(
            f"Top-1 ViDoRe Similarity Scores "
            f"({SEARCH_MODE.upper()}) - {pdf_base}"
        )
        plt.xlabel("Similarity Score")
        plt.ylabel("Number of Images")
        plt.grid(axis="y", alpha=0.75)

        plot_path = output_folder / "vidore_similarity_histogram.png"
        plt.savefig(plot_path)
        plt.close()

        print(f"CSV and Histogram saved to {output_folder}")


if __name__ == "__main__":
    run_vidore_matching(parse_args())