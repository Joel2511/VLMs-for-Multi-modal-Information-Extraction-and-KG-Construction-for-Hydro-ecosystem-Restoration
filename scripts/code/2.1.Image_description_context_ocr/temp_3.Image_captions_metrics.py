import os
import sys
import json
import warnings
import multiprocessing

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PAC_REPO_ROOT = "/home2020/home/icube/janil/pacscore"
PAC_CODE_DIR = os.path.join(PAC_REPO_ROOT, "evaluation", "pac_score")
sys.path.insert(0, PAC_REPO_ROOT)
sys.path.insert(0, PAC_CODE_DIR)

import torch
import clip
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from pac_score import PACScore
except ImportError as e:
    print(f"Pathing Error: Could not find pac_score.py. Check: {PAC_CODE_DIR}")
    raise e

# --- 1. PATHS ---
PROJECT_ROOT       = os.path.join(os.path.expanduser("~"), "Rhine_proj")
BASE_IMAGE_FOLDER  = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Image_captioning")
# BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Ablation_image_captions")
#BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Image_captioning_w_summary")
# BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Image_captioning_w_weighted_context")

if len(sys.argv) < 2:
    print("Usage: python temp_3.2.clip_pacs_metrics.py <model_results_folder_name>")
    sys.exit(1)

FOLDER_NAME          = sys.argv[1]
MODEL_RESULTS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, FOLDER_NAME)
# SAVE_PATH            = os.path.join(MODEL_RESULTS_FOLDER, "scientific_metrics.json")
SAVE_PATH = os.path.join(MODEL_RESULTS_FOLDER, "context_ocr_clip_pacs_results.json")


def _redirect_stderr_to_devnull():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    return saved_stderr_fd


def _restore_stderr(saved_fd):
    os.dup2(saved_fd, 2)
    os.close(saved_fd)


def run_vision_language_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n" + "=" * 70)
    print(f"STARTING EVALUATION: {FOLDER_NAME}")
    print(f"COMPUTE DEVICE: {device.upper()}")
    print("=" * 70)

    if not os.path.isdir(MODEL_RESULTS_FOLDER):
        print(f"ERROR: Folder not found: {MODEL_RESULTS_FOLDER}")
        sys.exit(1)

    # Skip if already evaluated
    if os.path.exists(SAVE_PATH):
        print(f"Already evaluated — skipping. Delete {SAVE_PATH} to re-run.")
        sys.exit(0)

    print("Loading CLIP (ViT-B/32)...")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    all_clip_scores   = []
    all_pacs_scores   = []
    per_image_results = {}

    pdf_folders = [
        f for f in os.listdir(MODEL_RESULTS_FOLDER)
        if os.path.isdir(os.path.join(MODEL_RESULTS_FOLDER, f))
    ]

    for pdf_base in pdf_folders:
        print(f"\n>>> PROCESSING PDF: {pdf_base}")

        caption_json_path = os.path.join(MODEL_RESULTS_FOLDER, pdf_base, "image_captions.json")
        image_dir         = os.path.join(BASE_IMAGE_FOLDER, pdf_base)

        if not os.path.exists(caption_json_path):
            print(f"    [SKIP] No image_captions.json found at {caption_json_path}")
            continue

        with open(caption_json_path, "r", encoding="utf-8") as f:
            captions_dict = json.load(f)

        for img_name, caption in captions_dict.items():
            ignored_captions = [
                "DECORATIVE",
                "Image non informative : élément décoratif ou logo."
            ]
            if caption in ignored_captions or caption.startswith("ERROR"):
                continue

            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                continue

            try:
                image_pil   = Image.open(img_path).convert("RGB")
                image_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
                text_input  = clip.tokenize([caption], truncate=True).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    text_features  = clip_model.encode_text(text_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features  /= text_features.norm(dim=-1, keepdim=True)
                    raw_similarity  = (image_features @ text_features.T).item()
                    clip_score      = max(raw_similarity, 0) * 2.5

                saved_fd = _redirect_stderr_to_devnull()
                try:
                    try:
                        pacs_res   = PACScore(clip_model, clip_preprocess, [image_pil], [caption], device)
                        pacs_score = pacs_res[0]
                    except Exception:
                        pacs_res   = PACScore(clip_model, clip_preprocess, [img_path], [caption], device)
                        pacs_score = pacs_res[0]
                finally:
                    _restore_stderr(saved_fd)

                clip_score = float(clip_score)
                pacs_score = float(pacs_score)

                all_clip_scores.append(clip_score)
                all_pacs_scores.append(pacs_score)
                per_image_results[img_name] = {
                    "pdf":  pdf_base,
                    "clip": round(clip_score, 4),
                    "pacs": round(pacs_score, 4)
                }

                print(f"    [{img_name}] CLIP: {clip_score:.4f} | PAC-S: {pacs_score:.4f}")

            except Exception as e:
                print(f"    [ERROR] Failed to evaluate {img_name}: {e}")

    # --- AGGREGATION ---
    if all_clip_scores:
        pdf_aggregates = {}
        for img_key, img_data in per_image_results.items():
            pdf = img_data["pdf"]
            if pdf not in pdf_aggregates:
                pdf_aggregates[pdf] = {"clip_scores": [], "pacs_scores": [], "image_count": 0}
            pdf_aggregates[pdf]["clip_scores"].append(img_data["clip"])
            pdf_aggregates[pdf]["pacs_scores"].append(img_data["pacs"])
            pdf_aggregates[pdf]["image_count"] += 1

        per_pdf_summary = {}
        for pdf, data in pdf_aggregates.items():
            per_pdf_summary[pdf] = {
                "image_count":     data["image_count"],
                "mean_clip_score": round(sum(data["clip_scores"]) / len(data["clip_scores"]), 4),
                "mean_pacs_score": round(sum(data["pacs_scores"]) / len(data["pacs_scores"]), 4),
            }

        mean_clip = sum(all_clip_scores) / len(all_clip_scores)
        mean_pacs = sum(all_pacs_scores) / len(all_pacs_scores)

        summary_results = {
            "model_name":               FOLDER_NAME,
            "total_images_evaluated":   len(all_clip_scores),
            "total_pdfs_evaluated":     len(pdf_aggregates),
            "global_mean_clip_score":   round(mean_clip, 4),
            "global_mean_pacs_score":   round(mean_pacs, 4),
            "per_pdf_summary":          per_pdf_summary,
            "per_image_raw_data":       per_image_results
        }

        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=4, ensure_ascii=False)

        print("\n" + "=" * 70)
        print(f"FINAL REPORT: {FOLDER_NAME}")
        print("-" * 70)
        print(f"Total Images Evaluated : {len(all_clip_scores)}")
        print(f"Total PDFs Evaluated   : {len(pdf_aggregates)}")
        print("-" * 70)
        print(f"Global Mean CLIPScore  : {mean_clip:.4f}")
        print(f"Global Mean PAC-S      : {mean_pacs:.4f}")
        print("-" * 70)
        print("Per-PDF Breakdown:")
        for pdf, data in per_pdf_summary.items():
            print(f"  {pdf} ({data['image_count']} imgs) | CLIP: {data['mean_clip_score']:.4f} | PAC-S: {data['mean_pacs_score']:.4f}")
        print("-" * 70)
        print(f"Detailed results saved to: {SAVE_PATH}")
        print("=" * 70 + "\n")
    else:
        print("\n[!] No valid images found. Check folder paths.")


if __name__ == "__main__":
    run_vision_language_evaluation()