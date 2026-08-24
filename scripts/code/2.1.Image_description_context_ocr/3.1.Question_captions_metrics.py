import os
import sys
import json
import torch
import pandas as pd
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForSequenceClassification,
    pipeline,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

# --- 1. PATHS ---
PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")
BASE_COLPALI_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "vidore_results")
MODEL_RESULTS_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "qwen36_27b_expert_results_using_vidore")
EVAL_SAVE_PATH = os.path.join(MODEL_RESULTS_FOLDER, "enrichment_nli_vqa_results.json")


# --- 2. UTILS ---
def load_colpali_index(pdf_base: str) -> dict:
    """Reads ColPali CSV and dynamically returns a mapping for ALL matched blocks."""

    csv_path = os.path.join(BASE_COLPALI_FOLDER, pdf_base, "vidore_raw_matrix.csv")
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)

    # dynamically detect all Top_X_Block columns
    block_cols = [c for c in df.columns if c.startswith("Top_") and c.endswith("_Block")]

    index = {}

    for _, row in df.iterrows():
        img_name = row["Image_Name"]

        blocks = []
        for col_name in block_cols:
            if pd.notna(row[col_name]):
                blocks.append(str(row[col_name]))

        index[img_name] = blocks

    return index


def get_image_context(img_name: str, colpali_index: dict, blocks_dict: dict) -> str:
    """Fetches all matched raw text blocks (2, 3, or 5+) for the NLI comparison."""

    matched_keys = colpali_index.get(img_name, [])

    if not matched_keys:
        return ""

    # Join all retrieved blocks into a single NLI premise
    context_parts = [
        blocks_dict.get(key, "").strip()
        for key in matched_keys
        if blocks_dict.get(key)
    ]

    return " ".join(context_parts)

# --- 3. MODELS ---
def load_eval_models(device):
    print("--- Loading NLI Model ---")

# --- NLI MODEL ---
# The 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli' is the standard for French NLI.

    nli_model = "/home2020/home/icube/janil/Rhine_proj/hf_models/mDeBERTa-v3-mnli-xnli"

    nli_pipe = pipeline(
        "text-classification",
        model=nli_model,
        device=device
    )

    print("--- Loading Qwen2-VL-7B (4-bit) ---")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "/home2020/home/icube/janil/Rhine_proj/hf_models/Qwen2-VL-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    qwen_processor = AutoProcessor.from_pretrained(
        "/home2020/home/icube/janil/Rhine_proj/hf_models/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        model_max_length=2048,
        min_pixels=256 * 28 * 28,   # minimum visual token budget
        max_pixels=1280 * 28 * 28   # caps image at ~1280 visual tokens, prevents OOM
    )
    qwen_processor.tokenizer.model_max_length = 32768

    return nli_pipe, qwen_model, qwen_processor


# --- 4. MAIN EVALUATION ---
def run_enrichment_evaluation():

    # SAFE device detection (SLURM-aware)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    nli_pipe, qwen_model, qwen_processor = load_eval_models(device)

    pdf_folders = [
        f for f in os.listdir(MODEL_RESULTS_FOLDER)
        if os.path.isdir(os.path.join(MODEL_RESULTS_FOLDER, f))
    ]

    final_metrics = []

    for pdf_base in pdf_folders:
        print(f"\n>>> PDF: {pdf_base}")

        caption_path = os.path.join(MODEL_RESULTS_FOLDER, pdf_base, "image_captions.json")
        content_path = os.path.join(BASE_JSON_FOLDER, pdf_base, "extracted_content.json")

        colpali_index = load_colpali_index(pdf_base)

        if not (os.path.exists(caption_path) and os.path.exists(content_path)):
            print(f"SKIP {pdf_base}")
            continue

        with open(caption_path, "r", encoding="utf-8") as f:
            captions = json.load(f)

        with open(content_path, "r", encoding="utf-8") as f:
            blocks_dict = json.load(f).get("pages", {})

        for img_name, caption in captions.items():

            if caption in [
                "DECORATIVE",
                "Image non informative : élément décoratif ou logo."
            ] or "ERROR" in caption:
                continue

            try:
                # --- NLI (TEXT NOVELTY) ---
                context = get_image_context(img_name, colpali_index, blocks_dict)

                nli_result = nli_pipe(
                    {"text": context, "text_pair": caption}
                )

                if isinstance(nli_result, list):
                    nli_result = nli_result[0]

                nli_label = nli_result["label"].lower()

                # --- VQA (VISUAL CHECK) ---
                img_path = os.path.join(BASE_IMAGE_FOLDER, pdf_base, img_name)
                if not os.path.exists(img_path):
                    print(f"  [SKIP] Image file not found: {img_path}")
                    continue

                # --- STEP 1: DECOMPOSE CAPTION INTO ATOMIC CLAIMS ---
                decomp_messages = [
                    {
                        "role": "user",
                        "content": (
                            "Tu es un extracteur de faits techniques. "
                            "Extrais exactement 3 à 5 affirmations factuelles courtes, indépendantes et vérifiables à partir de la description technique suivante. "
                            "Chaque affirmation doit tenir en une seule phrase. Réponds uniquement avec une liste numérotée, sans rien d'autre.\n\n"
                            f"Description : {caption}"
                        )
                    }
                ]

                decomp_text = qwen_processor.apply_chat_template(
                    decomp_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                decomp_inputs = qwen_processor(
                    text=[decomp_text],
                    padding=True,
                    return_tensors="pt"
                )

                model_device = next(qwen_model.parameters()).device
                decomp_inputs = decomp_inputs.to(model_device)

                decomp_ids = qwen_model.generate(**decomp_inputs, max_new_tokens=200)
                decomp_trimmed = [
                    out[len(inp):]
                    for inp, out in zip(decomp_inputs.input_ids, decomp_ids)
                ]
                claims_text = qwen_processor.batch_decode(
                    decomp_trimmed,
                    skip_special_tokens=True
                )[0]
                torch.cuda.empty_cache()

                claims = [
                    line.strip().lstrip("0123456789.-) ")
                    for line in claims_text.strip().split("\n")
                    if line.strip() and line.strip()[0].isdigit()
                ]

                if not claims:
                    claims = [caption]

                # --- STEP 2: VERIFY EACH ATOMIC CLAIM AGAINST THE IMAGE ---
                verified_count = 0
                total_claims = len(claims)

                # AFTER (load once before the loop, reuse inside):
                pil_img = Image.open(img_path).convert("RGB")  # loaded ONCE per image

                for claim in claims:
                    inputs = None
                    generated_ids = None
                    generated_ids_trimmed = None

                    try:

                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": pil_img},
                                    {
                                        "type": "text",
                                        "text": (
                                        "Tu es un analyste visuel forensique spécialisé en géomorphologie fluviale et restauration de cours d'eau. "
                                        "Ton expertise couvre : la dynamique sédimentaire (colmatage, granulométrie, dépôts de sédiments fins), "
                                        "les indicateurs hydrauliques (turbulence, écoulement laminaire, turbidité de l'eau), "
                                        "les caractéristiques morphologiques (sinuosité, érosion des berges, géométrie de la section transversale), "
                                        "et les indicateurs biologiques (biofilm, macrophytes, développement d'algues). "
                                        "Tu évalues les affirmations techniques strictement à partir des preuves visuelles présentes dans les pixels. "
                                        "Tu es rigoureux et ne confirmes pas les affirmations que tu ne peux pas observer directement.\n\n"
                                        f"Affirmation à vérifier : '{claim}'\n\n"
                                        "Étape 1 - Décris en une phrase ce que tu observes réellement dans l'image en lien avec cette affirmation.\n"
                                        "Étape 2 - Les preuves visuelles soutiennent-elles cette affirmation ?\n"
                                        "Réponse finale : CONFIRMÉ ou NON-CONFIRMÉ"
                                        )
                                    }
                                ],
                            }
                        ]

                        text = qwen_processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        image_inputs, video_inputs = process_vision_info(messages)

                        inputs = qwen_processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        )

                        inputs = inputs.to(model_device)

                        with torch.inference_mode():
                            generated_ids = qwen_model.generate(
                                **inputs,
                                max_new_tokens=150
                            )

                        generated_ids_trimmed = [
                            out[len(inp):]
                            for inp, out in zip(inputs.input_ids, generated_ids)
                        ]

                        claim_answer = qwen_processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True
                        )[0]

                        if (
                            "non-confirmé" not in claim_answer.lower() and
                            "confirmé" in claim_answer.lower()
                        ):
                            verified_count += 1

                    except torch.cuda.OutOfMemoryError:
                        print(f"  [OOM SKIP] claim skipped for {img_name}, counting as unverified.")
                    finally:
                        # pil_img is NOT deleted here anymore - it's reused across claims
                        if inputs is not None:
                            del inputs
                        if generated_ids is not None:
                            del generated_ids
                        if generated_ids_trimmed is not None:
                            del generated_ids_trimmed
                        torch.cuda.empty_cache()
                        import gc; gc.collect()
                pil_img = None  # free AFTER the full claims loop for this image is done

                vqa_score = verified_count / total_claims if total_claims > 0 else 0.0
                vqa_answer = f"{verified_count}/{total_claims} confirmed"

                # --- METRIC LOGIC ---
                is_neutral = (nli_label == "neutral")

                is_verified = vqa_score >= 0.5

                final_metrics.append({
                    "image": img_name,
                    "pdf": pdf_base,
                    "nli_status": nli_label,
                    "vqa_verification": vqa_answer.strip(),
                    "vqa_atomic_score": round(vqa_score, 4),
                    "is_novel": int(is_neutral),
                    "is_visually_grounded": int(is_verified),
                    "is_verified_enrichment": int(is_neutral and is_verified)
                })

                print(
                    f"{img_name} | NLI: {nli_label} | "
                    f"VQA: {is_verified} ({vqa_answer}) | ENR: {is_neutral and is_verified}"
                )

            except Exception as e:
                    import traceback
                    print(f"\n[FAILED IMAGE] {img_name}")
                    print(traceback.format_exc())

# --- SUMMARY ---
    if final_metrics:

        df = pd.DataFrame(final_metrics)

        # Per-PDF aggregates (correct granularity)
        per_pdf_summary = {}
        for pdf_name, pdf_group in df.groupby("pdf"):
            nli_dist = pdf_group["nli_status"].value_counts().to_dict()
            per_pdf_summary[pdf_name] = {
                "image_count": len(pdf_group),
                "novelty_rate": round(float(pdf_group["is_novel"].mean()), 4),
                "visual_grounding_rate": round(float(pdf_group["is_visually_grounded"].mean()), 4),
                "verified_enrichment_rate": round(float(pdf_group["is_verified_enrichment"].mean()), 4),
                "mean_vqa_atomic_score": round(float(pdf_group["vqa_atomic_score"].mean()), 4),
                "nli_distribution": nli_dist
            }

        # Global aggregate = mean over ALL images (not mean of PDF means)
        summary = {
            "model": os.path.basename(MODEL_RESULTS_FOLDER),
            "total_images_evaluated": len(df),
            "total_pdfs_evaluated": len(per_pdf_summary),
            "global_novelty_rate": round(float(df["is_novel"].mean()), 4),
            "global_visual_grounding_rate": round(float(df["is_visually_grounded"].mean()), 4),
            "global_verified_enrichment_rate": round(float(df["is_verified_enrichment"].mean()), 4),
            "global_mean_vqa_atomic_score": round(float(df["vqa_atomic_score"].mean()), 4),
            "global_nli_distribution": df["nli_status"].value_counts().to_dict(),
            "per_pdf_summary": per_pdf_summary,
            "details": final_metrics
        }

        with open(EVAL_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("ENRICHMENT REPORT")
        print(f"Total Images Evaluated : {summary['total_images_evaluated']}")
        print(f"Total PDFs Evaluated   : {summary['total_pdfs_evaluated']}")
        print("-" * 60)
        print(f"Global Novelty Rate    : {summary['global_novelty_rate']:.3f}")
        print(f"Global Visual Grounding: {summary['global_visual_grounding_rate']:.3f}")
        print(f"Global Enrichment Rate : {summary['global_verified_enrichment_rate']:.3f}")
        print(f"Global VQA Atomic Score: {summary['global_mean_vqa_atomic_score']:.3f}")
        print("-" * 60)
        print("Per-PDF Breakdown:")
        for pdf, data in per_pdf_summary.items():
            print(f"  {pdf} ({data['image_count']} imgs) | "
                  f"Novelty: {data['novelty_rate']:.3f} | "
                  f"Grounding: {data['visual_grounding_rate']:.3f} | "
                  f"Enrichment: {data['verified_enrichment_rate']:.3f} | "
                  f"VQA: {data['mean_vqa_atomic_score']:.3f}")
        print("=" * 60)


if __name__ == "__main__":
    run_enrichment_evaluation()