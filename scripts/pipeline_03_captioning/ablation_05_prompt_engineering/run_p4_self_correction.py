"""
Pipeline 3, Ablation 5 P4: Internal Self-Correction caption generation.

This script runs NVILA-Lite-8B with the Internal Self-Correction prompt used
in the prompt-engineering ablation. It supports processing one document or
every available document while preserving the original captioning,
retrieval-context, weighted-context, and JSON-output logic.
"""

import argparse
import gc
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


# ─── Repository and experiment configuration ─────────────────────────────────

SCRIPT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("RHINE_PROJECT_ROOT", str(SCRIPT_ROOT))
).expanduser().resolve()

HF_MODELS_DIR = Path(
    os.environ.get("HF_MODELS_DIR", str(PROJECT_ROOT / "hf_models"))
).expanduser().resolve()

GENERATOR_MODEL_NAME = "nvila-lite-8b"
RETRIEVAL_MODEL_NAME = "vidore"
PROMPT_ID = "p4_selfcorrect"

MODEL_ID = HF_MODELS_DIR / "NVILA-Lite-8B-hf"

DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "extracted_images"
DEFAULT_JSON_ROOT = PROJECT_ROOT / "outputs" / "json_results"
DEFAULT_RETRIEVAL_ROOT = PROJECT_ROOT / "outputs" / "Image_text_matching"
DEFAULT_WEIGHTED_CONTEXT_ROOT = PROJECT_ROOT / "outputs" / "Weighted_summary"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "captioning"
    / f"nvila_{PROMPT_ID}_captions"
)
DEFAULT_JUNK_ROOT = PROJECT_ROOT / "data" / "junk_images"


class ContextMissingError(Exception):
    """Raised when no usable retrieved text context exists for an image."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run NVILA-Lite-8B Internal Self-Correction captioning with "
            "retrieval and optional weighted-context inputs."
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
        "--retrieval-root",
        type=Path,
        default=DEFAULT_RETRIEVAL_ROOT,
        help=(
            "Root directory containing retrieval result folders. "
            f"Default: {DEFAULT_RETRIEVAL_ROOT}"
        ),
    )

    parser.add_argument(
        "--weighted-context-root",
        type=Path,
        default=DEFAULT_WEIGHTED_CONTEXT_ROOT,
        help=(
            "Directory containing weighted-context JSON files. "
            f"Default: {DEFAULT_WEIGHTED_CONTEXT_ROOT}"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory for newly generated caption outputs. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )

    parser.add_argument(
        "--junk-root",
        type=Path,
        default=DEFAULT_JUNK_ROOT,
        help=(
            "Directory used for per-document junk-image folders. "
            f"Default: {DEFAULT_JUNK_ROOT}"
        ),
    )

    return parser.parse_args()


def format_elapsed(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = int(seconds % 60)
    return f"{hours:02d}h {minutes:02d}m {remaining_seconds:02d}s"


def load_retrieval_index(pdf_base: str, retrieval_root: Path) -> dict:
    csv_path = (
        retrieval_root
        / f"{RETRIEVAL_MODEL_NAME}_results"
        / pdf_base
        / f"{RETRIEVAL_MODEL_NAME}_raw_matrix.csv"
    )

    if not csv_path.exists():
        return {}

    dataframe = pd.read_csv(csv_path)
    index = {}

    for _, row in dataframe.iterrows():
        img_name = row["Image_Name"]
        blocks = []

        for rank in range(1, 4):
            column_name = f"Top_{rank}_Block"

            if column_name in row and pd.notna(row[column_name]):
                blocks.append(str(row[column_name]))

        index[img_name] = blocks

    return index


def get_image_context(
    img_name: str,
    retrieval_index: dict,
    blocks_dict: dict,
) -> str:
    if img_name not in retrieval_index:
        raise ContextMissingError(f"Image '{img_name}' not found in CSV.")

    matched_keys = retrieval_index[img_name]
    context_parts = []

    for key in matched_keys:
        text = blocks_dict.get(key, "").strip()

        if text:
            context_parts.append(f"[{key}]: {text}")

    if not context_parts:
        raise ContextMissingError(
            f"No text found in JSON for blocks matched to {img_name}."
        )

    return "\n\n---\n\n".join(context_parts)


def build_prompt(image_context: str) -> str:
    return f"""Réponds exclusivement en français. Tu travailles sur des documents environnementaux français portant sur la restauration de cours d'eau, le Rhin, et l'hydromorphologie.

CONTEXTE SPÉCIFIQUE À CETTE IMAGE (source principale) :
Le passage suivant est extrait du rapport et décrit directement ce que représente cette image. Lis-le attentivement avant d'analyser l'image.
<contexte>{image_context}</contexte>

PHASE 1 — DESCRIPTION VISUELLE (obligatoire) :
Décris uniquement ce que tu vois dans les pixels de cette image. Sois factuel et précis.
- Si l'image est un logo ou un élément purement décoratif, réponds uniquement : "Image non informative : élément décoratif ou logo." et arrête-toi.
- Indique le type visuel : PHOTO / MAP / GRAPH / DIAGRAM / TABLE
- Décris le sujet principal visible
- Mentionne tout élément physique clairement observable (eau, végétation, structures, terrain, texte dans l'image)
- Si un élément n'est pas visible ou trop petit, écris explicitement "non observable à cette échelle"
- Ne complète pas les lacunes avec des suppositions

PHASE 2 — ANCRAGE CONTEXTUEL :
En confrontant ta Phase 1 avec le contexte fourni ci-dessus :
- Identifie les éléments visuels qui correspondent directement au contexte
- Identifie 1 à 2 faits physiques visibles dans l'image mais absents du contexte (ex: dégradation visible, structure non décrite)
- Si aucun fait nouveau : "Aucun enrichissement détecté"
- Si l'image contredit le contexte : note la discordance en une phrase
Après avoir rédigé ta Phase 2, relis chaque affirmation mot à mot en la confrontant uniquement à ce que tu as écrit en Phase 1. Supprime activement toute affirmation de Phase 2 qui introduit un concept, un diagnostic hydromorphologique, ou une caractéristique physique absent de ta Phase 1. N'émets que la version vérifiée et épurée.

RÈGLE FINALE : Tu n'as le droit de mentionner en Phase 2 que des éléments que tu as déjà écrits en Phase 1. Zéro ajout."""


def select_document_folders(image_root: Path, document: str) -> list[Path]:
    if not image_root.exists():
        raise FileNotFoundError(
            f"Base image folder not found: {image_root}"
        )

    available_folders = sorted(
        folder for folder in image_root.iterdir() if folder.is_dir()
    )

    if not available_folders:
        raise FileNotFoundError(
            f"No document folders found in: {image_root}"
        )

    if document == "all":
        return available_folders

    selected_folder = image_root / document

    if not selected_folder.is_dir():
        available_names = ", ".join(folder.name for folder in available_folders)
        raise FileNotFoundError(
            f"Requested document folder not found: {document}\n"
            f"Available document folders: {available_names}"
        )

    return [selected_folder]


def run_nvila_captioning(args: argparse.Namespace) -> None:
    pipeline_start = time.time()

    image_root = args.image_root.expanduser().resolve()
    json_root = args.json_root.expanduser().resolve()
    retrieval_root = args.retrieval_root.expanduser().resolve()
    weighted_context_root = args.weighted_context_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    junk_root = args.junk_root.expanduser().resolve()

    print(
        f"--- Loading {GENERATOR_MODEL_NAME} with "
        f"{RETRIEVAL_MODEL_NAME} retrieval context ---"
    )
    print(f"    Generator : {GENERATOR_MODEL_NAME}")
    print(f"    Retriever : {RETRIEVAL_MODEL_NAME}")
    print(f"    Document  : {args.document}")
    print(f"    Images    : {image_root}")
    print(f"    JSON      : {json_root}")
    print(f"    Retrieval : {retrieval_root}")
    print(f"    Context   : {weighted_context_root}")
    print(f"    Output    : {output_root}")

    if not MODEL_ID.exists():
        raise FileNotFoundError(
            f"NVILA model directory not found: {MODEL_ID}\n"
            "Download the model and apply the required patch described in "
            "MODEL_SETUP.md, or set HF_MODELS_DIR in config/project.env."
        )

    model_load_start = time.time()
    hf_token = os.environ.get("HF_TOKEN")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=hf_token,
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.eval()

    model_load_elapsed = time.time() - model_load_start
    total_params = sum(parameter.numel() for parameter in model.parameters())

    print(f"    Model size : {total_params / 1e9:.1f}B parameters")
    print(f"    Loaded at  : {datetime.now().strftime('%H:%M:%S')}")
    print(f"    Load time  : {format_elapsed(model_load_elapsed)}")

    pdf_folders = select_document_folders(image_root, args.document)

    for pdf_folder in pdf_folders:
        pdf_base = pdf_folder.name
        pdf_start = time.time()

        print(f"\n========== PROCESSING [{pdf_base}] ==========")

        image_folder = pdf_folder
        output_folder = output_root / pdf_base
        junk_pdf_folder = junk_root / pdf_base

        output_folder.mkdir(parents=True, exist_ok=True)
        junk_pdf_folder.mkdir(parents=True, exist_ok=True)

        json_output_path = output_folder / f"image_captions_{PROMPT_ID}.json"

        if json_output_path.exists():
            print(" -> Already captioned. Skipping.")
            continue

        weighted_context_path = (
            weighted_context_root
            / f"{pdf_base}_{RETRIEVAL_MODEL_NAME}_weighted_context.json"
        )

        weighted_context_index = {}

        if weighted_context_path.exists():
            with open(weighted_context_path, "r", encoding="utf-8") as file:
                weighted_context_index = json.load(file)

            print(
                f" -> Loaded weighted context: "
                f"{len(weighted_context_index)} images"
            )
        else:
            print(
                f" -> WARNING: {weighted_context_path} not found. "
                "Will fallback to raw blocks."
            )

        json_path = json_root / pdf_base / "extracted_content.json"

        if not json_path.exists():
            print(f" -> SKIPPING {pdf_base}: No extracted_content.json found.")
            continue

        with open(json_path, "r", encoding="utf-8") as file:
            blocks_dict = json.load(file).get("pages", {})

        retrieval_index = load_retrieval_index(pdf_base, retrieval_root)

        if not retrieval_index:
            print(f" -> SKIPPING {pdf_base}: Retrieval CSV missing.")
            continue

        image_files = sorted(
            file.name
            for file in image_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        results_dict = {}
        total_images = len(image_files)

        print(f"    Images to caption: {total_images}")

        for index, img_name in enumerate(image_files, start=1):
            img_path = image_folder / img_name
            torch.cuda.empty_cache()
            img_start = time.time()

            try:
                raw_image = Image.open(img_path).convert("RGB")

                img_stem = Path(img_name).stem

                if img_stem in weighted_context_index:
                    image_context = weighted_context_index[img_stem][
                        "weighted_context"
                    ]
                else:
                    image_context = get_image_context(
                        img_name,
                        retrieval_index,
                        blocks_dict,
                    )

                caption_prompt = build_prompt(image_context)

                image_token = processor.tokenizer.image_token
                messages = [
                    {
                        "role": "user",
                        "content": f"{image_token}\n{caption_prompt}",
                    }
                ]

                text_with_image = processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = processor(
                    text=text_with_image,
                    images=raw_image,
                    return_tensors="pt",
                )

                model_device = next(model.parameters()).device
                inputs = {
                    key: value.to(model_device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )

                prompt_len = inputs["input_ids"].shape[-1]

                generated_text = processor.tokenizer.decode(
                    generated_ids[0][prompt_len:],
                    skip_special_tokens=True,
                ).strip()

                img_elapsed = time.time() - img_start

                if generated_text.lower().startswith("image non informative"):
                    print(
                        f"    [{index}/{total_images}] {img_name} -> "
                        f"DECORATIVE ({img_elapsed:.1f}s)"
                    )
                    results_dict[img_name] = "DECORATIVE"
                else:
                    results_dict[img_name] = generated_text
                    print(
                        f"    [{index}/{total_images}] {img_name} -> "
                        f"{generated_text[:100]}... ({img_elapsed:.1f}s)"
                    )

            except Exception as error:
                traceback.print_exc()
                print(f"FAILED on {img_name}: {error}")
                results_dict[img_name] = f"ERROR: {str(error)}"

            finally:
                gc.collect()
                torch.cuda.empty_cache()

        with open(json_output_path, "w", encoding="utf-8") as file:
            json.dump(results_dict, file, indent=4, ensure_ascii=False)

        pdf_elapsed = time.time() - pdf_start

        print(
            f"========== COMPLETED [{pdf_base}] in "
            f"{format_elapsed(pdf_elapsed)} =========="
        )

    total_elapsed = time.time() - pipeline_start

    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE")
    print(f"  Generator model : {GENERATOR_MODEL_NAME}")
    print(f"  Retrieval model : {RETRIEVAL_MODEL_NAME}")
    print(f"  Total runtime   : {format_elapsed(total_elapsed)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_nvila_captioning(parse_args())
