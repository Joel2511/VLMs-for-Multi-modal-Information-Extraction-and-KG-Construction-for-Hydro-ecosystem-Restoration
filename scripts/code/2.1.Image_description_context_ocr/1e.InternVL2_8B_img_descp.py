import os
import gc
import re
import math
import json
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import time
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

GENERATOR_MODEL_NAME = "internvl25-8b"
RETRIEVAL_MODEL_NAME = "vidore"
GENERATOR_MODEL_PATH = "/home2020/home/icube/janil/Rhine_proj/hf_models/InternVL2_5-8B"

PROJECT_ROOT              = os.path.join(os.path.expanduser("~"), "Rhine_proj")
IMAGE_INPUT_DIR           = os.path.join(PROJECT_ROOT, "data", "extracted_images")
JSON_INPUT_DIR            = os.path.join(PROJECT_ROOT, "outputs", "json_results")
JUNK_OUTPUT_DIR           = os.path.join(PROJECT_ROOT, "data", "junk_images")
RETRIEVAL_RESULTS_DIR     = os.path.join(PROJECT_ROOT, "outputs", "Image_text_matching", f"{RETRIEVAL_MODEL_NAME}_results")
RETRIEVAL_MATRIX_FILENAME = f"{RETRIEVAL_MODEL_NAME}_raw_matrix.csv"
CAPTION_OUTPUT_DIR        = os.path.join(PROJECT_ROOT, "outputs", "2nd_ablation_Image_captioning", f"{GENERATOR_MODEL_NAME}_captions_{RETRIEVAL_MODEL_NAME}")

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise EnvironmentError("HF_TOKEN environment variable not set. Run 'export HF_TOKEN=...' before executing this script.")
GPU_MAX_MEMORY = {0: "14GiB", 1: "14GiB"}


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ContextMissingError(Exception):
    pass


def format_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


# =============================================================================
# RETRIEVAL INDEX LOADING
# =============================================================================

def load_retrieval_index(pdf_base: str) -> dict:
    csv_path = os.path.join(RETRIEVAL_RESULTS_DIR, pdf_base, RETRIEVAL_MATRIX_FILENAME)
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


# =============================================================================
# CONTEXT RETRIEVAL
# =============================================================================

def get_image_context(img_name: str, retrieval_index: dict, blocks_dict: dict) -> str:
    if img_name not in retrieval_index:
        raise ContextMissingError(f"Image '{img_name}' not found in retrieval index.")
    matched_keys = retrieval_index[img_name]
    context_parts = []
    for key in matched_keys:
        text = blocks_dict.get(key, "").strip()
        if text:
            context_parts.append(f"[{key}]: {text}")
    if not context_parts:
        raise ContextMissingError(f"No text found in JSON for blocks matched to '{img_name}'.")
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================

def build_prompt(image_context: str, image_ocr: str) -> str:
    return f"""Réponds exclusivement en français. Tu travailles sur des documents environnementaux français portant sur la restauration de cours d'eau, le Rhin, et l'hydromorphologie.


PHASE 1 — DESCRIPTION LIBRE (obligatoire, sans référence à aucun texte externe) :
Décris uniquement ce que tu vois dans les pixels de cette image. Sois factuel et court.
- Si l'image est un logo ou un élément purement décoratif, réponds uniquement : "Image non informative : élément décoratif ou logo." et arrête-toi.
- Indique le type visuel : PHOTO / MAP / GRAPH / DIAGRAM / TABLE
- Décris le sujet principal visible
- Mentionne tout élément physique clairement observable (eau, végétation, structures, terrain, texte dans l'image)
- Si un élément n'est pas visible ou trop petit pour être identifié, écris explicitement "non observable à cette échelle"
- Ne complète pas les lacunes avec des suppositions


PHASE 2 — COMPARAISON CONTEXTUELLE :
Contexte du rapport lié à cette image :
<contexte>{image_context}</contexte>
OCR détecté dans l'image :
<ocr>{image_ocr}</ocr>


En te basant exclusivement sur ta Phase 1 :
- Identifie 1 à 2 faits physiques présents dans ton observation mais absents du contexte ci-dessus (ex: une dégradation visible non mentionnée)
- Si aucun fait nouveau n'est visible : "Aucun enrichissement détecté"
- Si l'image contredit le contexte : note la discordance en une phrase


RÈGLE FINALE : Tu n'as le droit de mentionner en Phase 2 que des éléments que tu as déjà écrits en Phase 1. Zéro ajout."""


# =============================================================================
# INTERNVL-SPECIFIC IMAGE PREPROCESSING
# =============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width  = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width  // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width  // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image_internvl(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values

def split_model():
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = 32
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map


# =============================================================================
# MAIN CAPTIONING PIPELINE
# =============================================================================

def run_internvl_captioning():
    pipeline_start = time.time()
    print(f"--- Loading {GENERATOR_MODEL_NAME} with {RETRIEVAL_MODEL_NAME} retrieval context ---")
    print(f"    Generator : {GENERATOR_MODEL_NAME}")
    print(f"    Retriever : {RETRIEVAL_MODEL_NAME}")

    model_load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        GENERATOR_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
        token=HF_TOKEN
    )

    model = AutoModel.from_pretrained(
        GENERATOR_MODEL_PATH,
        torch_dtype=torch.float16,   
        load_in_8bit=True,           
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        token=HF_TOKEN,
        device_map="auto"
    ).eval()

    model_load_elapsed = time.time() - model_load_start
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Model size   : {total_params/1e9:.1f}B parameters")
    print(f"    Loaded at    : {datetime.now().strftime('%H:%M:%S')}")
    print(f"    Load time    : {format_elapsed(model_load_elapsed)}")

    if not os.path.exists(IMAGE_INPUT_DIR):
        print(f"Error: Image input folder not found at {IMAGE_INPUT_DIR}")
        return

    pdf_folders = [
        f for f in os.listdir(IMAGE_INPUT_DIR)
        if os.path.isdir(os.path.join(IMAGE_INPUT_DIR, f))
    ]

    for pdf_base in pdf_folders:
        pdf_start = time.time()
        print(f"\n========== PROCESSING [{pdf_base}] ==========")

        image_dir       = os.path.join(IMAGE_INPUT_DIR, pdf_base)
        caption_out_dir = os.path.join(CAPTION_OUTPUT_DIR, pdf_base)
        junk_pdf_dir    = os.path.join(JUNK_OUTPUT_DIR, pdf_base)
        os.makedirs(caption_out_dir, exist_ok=True)
        os.makedirs(junk_pdf_dir, exist_ok=True)

        json_output_path = os.path.join(caption_out_dir, "image_captions.json")
        if os.path.exists(json_output_path):
            print(f"    Already captioned — skipping.")
            continue

        ocr_json_path = os.path.join(JSON_INPUT_DIR, pdf_base, "ocr_results.json")
        ocr_dict = {}
        if os.path.exists(ocr_json_path):
            with open(ocr_json_path, "r", encoding="utf-8") as f:
                ocr_dict = json.load(f)
        else:
            print(f"    WARNING: OCR results not found. Proceeding without OCR context.")

        extracted_json_path = os.path.join(JSON_INPUT_DIR, pdf_base, "extracted_content.json")
        if not os.path.exists(extracted_json_path):
            print(f"    SKIPPING: extracted_content.json not found.")
            continue

        with open(extracted_json_path, "r", encoding="utf-8") as f:
            blocks_dict = json.load(f).get("pages", {})

        retrieval_index = load_retrieval_index(pdf_base)
        if not retrieval_index:
            print(f"    SKIPPING: Retrieval index ({RETRIEVAL_MODEL_NAME} CSV) not found.")
            continue

        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        results_dict = {}
        total_images = len(image_files)
        print(f"    Images to caption: {total_images}")

        for idx, img_name in enumerate(image_files, start=1):
            img_path = os.path.join(image_dir, img_name)
            torch.cuda.empty_cache()
            img_start = time.time()

            try:
                image_context  = get_image_context(img_name, retrieval_index, blocks_dict)
                image_ocr      = ocr_dict.get(img_name, "Aucun texte détecté.")
                caption_prompt = build_prompt(image_context, image_ocr)

                question = "<image>\n" + caption_prompt

                # FIX: max_num=6 kept for memory safety on V100, matches working config
                pixel_values = load_image_internvl(img_path, max_num=6).to(torch.float16).to('cuda:0')

                generation_config = dict(
                    do_sample=False,
                    max_length=8192       
                )
                generated_text = model.chat(
                    tokenizer, pixel_values, question, generation_config
                ).strip()

                img_elapsed = time.time() - img_start

                if generated_text.strip().lower().startswith("image non informative"):
                    print(f"    [{idx}/{total_images}] {img_name} -> DECORATIVE ({img_elapsed:.1f}s)")
                    results_dict[img_name] = "DECORATIVE"
                else:
                    results_dict[img_name] = generated_text
                    print(f"    [{idx}/{total_images}] {img_name} -> {generated_text[:100]}... ({img_elapsed:.1f}s)")

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"    FAILED on {img_name}: {e}")
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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_internvl_captioning()