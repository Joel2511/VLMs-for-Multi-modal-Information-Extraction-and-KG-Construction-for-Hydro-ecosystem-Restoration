#Verification & Refusal Gate
import os
import gc
import json
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel
import time
from datetime import datetime

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")




# --- CONFIGURATION ---
GENERATOR_MODEL_NAME = "nvila-lite-8b"
RETRIEVAL_MODEL_NAME = "vidore"
PROMPT_ID = "p2_refusal"


PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")
BASE_RETRIEVAL_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Image_text_matching", f"{RETRIEVAL_MODEL_NAME}_results")
BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "5th_ablation_Image_captioning_prompt_engineering", f"nvila_{PROMPT_ID}_captions")
JUNK_FOLDER = os.path.join(PROJECT_ROOT, "data", "junk_images")


HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise EnvironmentError("HF_TOKEN environment variable not set. Run 'export HF_TOKEN=...' before executing this script.")
MODEL_ID = os.path.join(PROJECT_ROOT, "hf_models/NVILA-Lite-8B-hf")



class ContextMissingError(Exception):
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
Pour chaque affirmation que tu formules en Phase 2, identifie dans ta réponse l'élément visuel précis de ta Phase 1 (couleur, forme, localisation dans l'image) qui la justifie. Si tu ne peux pas identifier un tel élément directement observable dans ta Phase 1, écris : "Non confirmé visuellement" et passe à l'affirmation suivante. Ne reformule jamais les informations du contexte sans ancrage visuel explicite dans ta Phase 1.


RÈGLE FINALE : Tu n'as le droit de mentionner en Phase 2 que des éléments que tu as déjà écrits en Phase 1. Zéro ajout."""



def run_nvila_captioning():
    pipeline_start = time.time()
    print(f"--- Loading {GENERATOR_MODEL_NAME} with {RETRIEVAL_MODEL_NAME} retrieval context ---")
    print(f"    Generator : {GENERATOR_MODEL_NAME}")
    print(f"    Retriever : {RETRIEVAL_MODEL_NAME}")


    model_load_start = time.time()


    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=HF_TOKEN
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )
    model.eval()


    model_load_elapsed = time.time() - model_load_start
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Model size   : {total_params/1e9:.1f}B parameters")
    print(f"    Loaded at    : {datetime.now().strftime('%H:%M:%S')}")
    print(f"    Load time    : {format_elapsed(model_load_elapsed)}")


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


        json_output_path = os.path.join(OUTPUT_FOLDER, f"image_captions_{PROMPT_ID}.json")
        if os.path.exists(json_output_path):
            print(f" -> Already captioned. Skipping.")
            continue


        weighted_context_path = os.path.join(PROJECT_ROOT, "outputs", "Weighted_summary", f"{pdf_base}_{RETRIEVAL_MODEL_NAME}_weighted_context.json")
        weighted_context_index = {}
        if os.path.exists(weighted_context_path):
            with open(weighted_context_path, "r", encoding="utf-8") as f:
                weighted_context_index = json.load(f)
            print(f" -> Loaded weighted context: {len(weighted_context_index)} images")
        else:
            print(f" -> WARNING: {weighted_context_path} not found. Will fallback to raw blocks.")


        json_path = os.path.join(BASE_JSON_FOLDER, pdf_base, "extracted_content.json")
        if not os.path.exists(json_path):
            print(f" -> SKIPPING {pdf_base}: No extracted_content.json found.")
            continue


        with open(json_path, "r", encoding="utf-8") as f:
            blocks_dict = json.load(f).get("pages", {})


        retrieval_index = load_retrieval_index(pdf_base)
        if not retrieval_index:
            print(f" -> SKIPPING {pdf_base}: CSV missing.")
            continue


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

                img_stem = os.path.splitext(img_name)[0]
                if img_stem in weighted_context_index:
                    image_context = weighted_context_index[img_stem]["weighted_context"]
                else:
                    image_context = get_image_context(img_name, retrieval_index, blocks_dict)

                caption_prompt = build_prompt(image_context)

                # NVILA-specific chat template construction and text generation
                image_token = processor.tokenizer.image_token
                messages = [
                    {
                        "role": "user",
                        "content": f"{image_token}\n{caption_prompt}"
                    }
                ]
                text_with_image = processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = processor(
                    text=text_with_image,
                    images=raw_image,
                    return_tensors="pt"
                )

                model_device = next(model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )

                prompt_len = inputs["input_ids"].shape[-1]
                generated_text = processor.tokenizer.decode(
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
                import traceback
                traceback.print_exc()
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
    run_nvila_captioning()