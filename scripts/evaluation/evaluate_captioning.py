import os
import sys
import json
import re
import torch
import pandas as pd
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
import spacy

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")

_nlp = spacy.load("fr_core_news_sm", disable=["ner", "tagger", "parser"])
_nlp.add_pipe("sentencizer")


# --- 1. PATHS ---
PROJECT_ROOT       = os.path.join(os.path.expanduser("~"), "Rhine_proj")
BASE_IMAGE_FOLDER  = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER   = os.path.join(PROJECT_ROOT, "outputs", "json_results")

BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "1st_ablation_image_captions_no_prompt")
# BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "2nd_ablation_Image_captioning")
# BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "3rd_ablation_Image_captioning_w_summary")
# BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "4th_ablation_Image_captioning_w_weighted_context")
# BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "5th_ablation_Image_captioning_prompt_engineering")
# BASE_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "6th_ablation_Image_captioning_decoding")

if len(sys.argv) < 2:
    print("Usage: python 3a.nli_vqa_metrics.py <model_results_folder_name>")
    print("Example: python 3a.nli_vqa_metrics.py internvl25-8b_captions_colqwen")
    sys.exit(1)

FOLDER_NAME          = sys.argv[1]
MODEL_RESULTS_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, FOLDER_NAME)
EVAL_SAVE_PATH       = os.path.join(MODEL_RESULTS_FOLDER, "1st_ablation_vqa_nli_results.json")

if "colqwen" in FOLDER_NAME:
    RETRIEVAL_MODEL = "colqwen"
    RETRIEVAL_CSV    = "colqwen_raw_matrix.csv"
    RETRIEVAL_SUBDIR = "colqwen_results"
else:
    RETRIEVAL_MODEL  = "vidore"
    RETRIEVAL_CSV    = "vidore_raw_matrix.csv"
    RETRIEVAL_SUBDIR = "vidore_results"

BASE_RETRIEVAL_FOLDER = os.path.join(
    PROJECT_ROOT, "outputs", "Image_text_matching", RETRIEVAL_SUBDIR
)


# --- 2. UTILS ---
def load_retrieval_index(pdf_base: str) -> dict:
    csv_path = os.path.join(BASE_RETRIEVAL_FOLDER, pdf_base, RETRIEVAL_CSV)
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    block_cols = [c for c in df.columns if c.startswith("Top_") and c.endswith("_Block")]
    index = {}
    for _, row in df.iterrows():
        img_name = row["Image_Name"]
        blocks = [str(row[col]) for col in block_cols if pd.notna(row[col])]
        index[img_name] = blocks
    return index


def get_image_context(img_name: str, retrieval_index: dict, blocks_dict: dict) -> str:
    matched_keys = retrieval_index.get(img_name, [])
    if not matched_keys:
        return ""
    context_parts = [
        blocks_dict.get(key, "").strip()
        for key in matched_keys
        if blocks_dict.get(key)
    ]
    return " ".join(context_parts)


def split_sentences(text: str) -> list:
    """Split text into sentences using spaCy French model (SummaC-ZS)."""
    doc = _nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 15]


def extract_phase_text(caption: str):
    """
    Split structured captions into Phase 1 (free visual description)
    and Phase 2 (contextual enrichment) if markers are present.
    Falls back to (full caption, "") if no phase markers are found.
    """
    p1_match = re.search(
        r'###?\s*(?:PHASE\s*1|Phase\s*1)[^\n]*\n(.*?)(?=###?\s*(?:PHASE\s*2|Phase\s*2)|$)',
        caption, re.IGNORECASE | re.DOTALL
    )
    p2_match = re.search(
        r'###?\s*(?:PHASE\s*2|Phase\s*2)[^\n]*\n(.*)',
        caption, re.IGNORECASE | re.DOTALL
    )
    phase1 = p1_match.group(1).strip() if p1_match else caption.strip()
    phase2 = p2_match.group(1).strip() if p2_match else ""
    return phase1, phase2


# --- 3. NLI: SummaC-ZS sentence-level scoring ---
def load_nli_model(nli_model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(nli_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_path)
    model.eval()
    model.to(device)
    # Detect entailment label index from model config
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    entailment_id = next(
        (i for i, lbl in id2label.items() if "entail" in lbl), 0
    )
    return tokenizer, model, entailment_id


def summac_zs_score(
    context: str,
    caption: str,
    nli_tokenizer,
    nli_model,
    entailment_id: int,
    device: str,
    max_tokens: int = 400
) -> float:
    """
    SummaC-ZS: split both context and caption into sentences, score every
    (context_sentence, caption_sentence) pair, take the maximum entailment
    probability per caption sentence, then average across caption sentences.
    Higher score = caption is well-grounded in context.
    """
    context_sentences = split_sentences(context)
    caption_sentences = split_sentences(caption)
    if not context_sentences or not caption_sentences:
        return 0.0

    per_sentence_scores = []
    for cap_sent in caption_sentences:
        best_score = 0.0
        for ctx_sent in context_sentences:
            enc = nli_tokenizer(
                ctx_sent, cap_sent,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = nli_model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            score = probs[0, entailment_id].item()
            if score > best_score:
                best_score = score
        per_sentence_scores.append(best_score)

    return float(sum(per_sentence_scores) / len(per_sentence_scores))


# --- 4. MODELS ---
def load_eval_models(device):
    print("--- Loading NLI Model (SummaC-ZS sentence scorer) ---")
    nli_model_path = os.path.join(PROJECT_ROOT, "hf_models/mDeBERTa-v3-mnli-xnli")
    nli_tokenizer, nli_model, entailment_id = load_nli_model(nli_model_path, device)

    print("--- Loading Qwen2-VL-7B (8-bit) ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        os.path.join(PROJECT_ROOT, "hf_models/Qwen2-VL-7B-Instruct"),
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    qwen_processor = AutoProcessor.from_pretrained(
        os.path.join(PROJECT_ROOT, "hf_models/Qwen2-VL-7B-Instruct"),
        trust_remote_code=True,
        model_max_length=2048,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28
    )
    qwen_processor.tokenizer.model_max_length = 32768
    return nli_tokenizer, nli_model, entailment_id, qwen_model, qwen_processor


# --- 5. VQA HELPERS ---
def decompose_claims(caption_text: str, qwen_model, qwen_processor, phase: str = "visual") -> list:
    """
    Decompose a caption or sub-caption into atomic verifiable claims.
    phase="visual"  → only directly observable visual elements (Phase 1)
    phase="context" → contextual interpretive links (Phase 2)
    """
    if phase == "visual":
        instruction = (
            "Tu es un extracteur de faits visuels. "
            "Extrais exactement 3 à 5 affirmations factuelles courtes et UNIQUEMENT les éléments "
            "directement observables dans une image (couleurs, formes, présence d'objets, textures, "
            "positions relatives). Ne pas inclure d'interprétations ou d'inférences contextuelles. "
            "Chaque affirmation doit tenir en une seule phrase. "
            "Réponds uniquement avec une liste numérotée, sans rien d'autre.\n\n"
            f"Description : {caption_text}"
        )
    else:
        instruction = (
            "Tu es un extracteur de liens contextuels. "
            "Extrais exactement 2 à 4 affirmations qui relient ce qui est visible dans l'image "
            "à des informations contextuelles (noms de lieux, mesures, processus hydrologiques, "
            "interprétations géomorphologiques). "
            "Chaque affirmation doit tenir en une seule phrase. "
            "Réponds uniquement avec une liste numérotée, sans rien d'autre.\n\n"
            f"Description : {caption_text}"
        )

    messages = [{"role": "user", "content": instruction}]
    decomp_text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    decomp_inputs = qwen_processor(text=[decomp_text], padding=True, return_tensors="pt")
    model_device = next(qwen_model.parameters()).device
    decomp_inputs = decomp_inputs.to(model_device)

    decomp_ids = qwen_model.generate(**decomp_inputs, max_new_tokens=200)
    decomp_trimmed = [
        out[len(inp):]
        for inp, out in zip(decomp_inputs.input_ids, decomp_ids)
    ]
    claims_text = qwen_processor.batch_decode(decomp_trimmed, skip_special_tokens=True)[0]
    torch.cuda.empty_cache()

    claims = [
        line.strip().lstrip("0123456789.-) ")
        for line in claims_text.strip().split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]
    return claims if claims else [caption_text]


def verify_claims_against_image(
    claims: list,
    pil_img,
    context_sentences: list,
    qwen_model,
    qwen_processor,
    img_name: str,
    phase: str = "visual"
) -> tuple:
    """
    Verify each atomic claim against the image (and optionally context sentences for Phase 2).
    Returns (verified_count, total_claims).
    """
    model_device = next(qwen_model.parameters()).device
    verified_count = 0
    total_claims = len(claims)

    for claim in claims:
        inputs = None
        generated_ids = None
        generated_ids_trimmed = None
        try:
            if phase == "visual":
                verify_text = (
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
            else:
                ctx_snippet = " ".join(context_sentences[:5]) if context_sentences else ""
                verify_text = (
                    "Tu es un analyste visuel forensique spécialisé en géomorphologie fluviale et restauration de cours d'eau. "
                    "Tu disposes d'un extrait du document source pour t'aider à interpréter l'image.\n\n"
                    f"Extrait du document : \"{ctx_snippet}\"\n\n"
                    f"Affirmation contextuelle à vérifier : '{claim}'\n\n"
                    "Étape 1 - Décris en une phrase ce que tu observes dans l'image qui est pertinent pour cette affirmation.\n"
                    "Étape 2 - L'image et le contexte documentaire soutiennent-ils ensemble cette affirmation ?\n"
                    "Réponse finale : CONFIRMÉ ou NON-CONFIRMÉ"
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": verify_text}
                    ],
                }
            ]
            text = qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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
                generated_ids = qwen_model.generate(**inputs, max_new_tokens=150)
            generated_ids_trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            claim_answer = qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0]
            if (
                "non-confirmé" not in claim_answer.lower() and
                "confirmé" in claim_answer.lower()
            ):
                verified_count += 1

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM SKIP] claim skipped for {img_name}, counting as unverified.")
        finally:
            if inputs is not None:
                del inputs
            if generated_ids is not None:
                del generated_ids
            if generated_ids_trimmed is not None:
                del generated_ids_trimmed
            torch.cuda.empty_cache()
            import gc; gc.collect()

    return verified_count, total_claims


# --- 6. MAIN EVALUATION ---
def run_enrichment_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Evaluating folder: {FOLDER_NAME}")
    print(f"Retrieval model  : {RETRIEVAL_MODEL}")
    print(f"Results will be saved to: {EVAL_SAVE_PATH}")

    if not os.path.isdir(MODEL_RESULTS_FOLDER):
        print(f"ERROR: Folder not found: {MODEL_RESULTS_FOLDER}")
        sys.exit(1)

    if os.path.exists(EVAL_SAVE_PATH):
        print(f"Already evaluated — skipping. Delete {EVAL_SAVE_PATH} to re-run.")
        sys.exit(0)

    nli_tokenizer, nli_model, entailment_id, qwen_model, qwen_processor = load_eval_models(device)

    pdf_folders = [
        f for f in os.listdir(MODEL_RESULTS_FOLDER)
        if os.path.isdir(os.path.join(MODEL_RESULTS_FOLDER, f))
    ]

    final_metrics = []

    for pdf_base in pdf_folders:
        print(f"\n>>> PDF: {pdf_base}")

        caption_path = os.path.join(MODEL_RESULTS_FOLDER, pdf_base, "image_captions.json")
        # Auto-detect whichever caption JSON exists for this prompt variant
        # _candidate_names = [
        #     "image_captions_p1_primitive.json",
        #     "image_captions_p2_refusal.json",
        #     "image_captions_p3_contrast.json",
        #     "image_captions_p4_selfcorrect.json",
        #     "image_captions_p5_expert.json",
        # ]
        # caption_path = None
        # for _name in _candidate_names:
        #     _p = os.path.join(MODEL_RESULTS_FOLDER, pdf_base, _name)
        #     if os.path.exists(_p):
        #         caption_path = _p
        #         break
        # if caption_path is None:
        #     print(f"SKIP {pdf_base} — no caption JSON found")
        #     continue
        content_path  = os.path.join(BASE_JSON_FOLDER, pdf_base, "extracted_content.json")
        retrieval_index = load_retrieval_index(pdf_base)

        if not os.path.exists(caption_path):
            print(f"SKIP {pdf_base} — no caption JSON found")
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
                final_metrics.append({
                    "image": img_name,
                    "pdf": pdf_base,
                    "nli_status": "excluded",
                    "nli_grounding_score": 0.0,
                    "nli_novelty_score": 0.0,
                    "vqa_verification": "n/a",
                    "vqa_atomic_score": 0.0,
                    "vqa_p1_atomic_score": 0.0,
                    "vqa_p2_atomic_score": None,
                    "vqa_p2_verification": "n/a",
                    "is_novel": 0,
                    "is_visually_grounded": 0,
                    "is_verified_enrichment": 0
                })
                continue

            try:
                # --- Resolve context ---
                img_stem = os.path.splitext(img_name)[0]
                weighted_ctx_path = os.path.join(
                    PROJECT_ROOT, "outputs", "Weighted_summary",
                    f"{pdf_base}_{RETRIEVAL_MODEL}_weighted_context.json"
                )
                if os.path.exists(weighted_ctx_path):
                    with open(weighted_ctx_path, "r", encoding="utf-8") as wf:
                        _wctx = json.load(wf)
                    context = _wctx.get(img_stem, {}).get(
                        "weighted_context",
                        get_image_context(img_name, retrieval_index, blocks_dict)
                    )
                else:
                    context = get_image_context(img_name, retrieval_index, blocks_dict)

                # --- NLI: SummaC-ZS continuous grounding score ---
                nli_grounding_score = summac_zs_score(
                    context, caption,
                    nli_tokenizer, nli_model, entailment_id, device
                )
                nli_novelty_score = round(1.0 - nli_grounding_score, 4)
                nli_grounding_score = round(nli_grounding_score, 4)
                # Keep legacy label field for backwards compatibility
                if nli_grounding_score >= 0.7:
                    nli_label = "entailment"
                elif nli_grounding_score >= 0.4:
                    nli_label = "neutral"
                else:
                    nli_label = "contradiction"

                # --- Image path ---
                img_path = os.path.join(BASE_IMAGE_FOLDER, pdf_base, img_name)
                if not os.path.exists(img_path):
                    print(f"  [SKIP] Image file not found: {img_path}")
                    continue

                pil_img = Image.open(img_path).convert("RGB")
                context_sentences = split_sentences(context)

                # --- VQA Phase 1: purely visual claims ---
                phase1_text, phase2_text = extract_phase_text(caption)

                p1_claims = decompose_claims(phase1_text, qwen_model, qwen_processor, phase="visual")
                p1_verified, p1_total = verify_claims_against_image(
                    p1_claims, pil_img, context_sentences, qwen_model, qwen_processor,
                    img_name, phase="visual"
                )
                vqa_p1_score = round(p1_verified / p1_total, 4) if p1_total > 0 else 0.0

                # --- VQA Phase 2: contextual claims (skip if no Phase 2) ---
                if phase2_text:
                    p2_claims = decompose_claims(phase2_text, qwen_model, qwen_processor, phase="context")
                    p2_verified, p2_total = verify_claims_against_image(
                        p2_claims, pil_img, context_sentences, qwen_model, qwen_processor,
                        img_name, phase="context"
                    )
                    vqa_p2_score = round(p2_verified / p2_total, 4) if p2_total > 0 else 0.0
                    p2_answer = f"{p2_verified}/{p2_total} confirmed"
                else:
                    vqa_p2_score = None
                    p2_answer = "n/a"

                pil_img = None

                # --- Composite scores (legacy fields kept) ---
                # Visual grounding: Phase 1 VQA >= 0.5
                is_verified = (vqa_p1_score >= 0.5)
                is_novel = int(nli_label == "neutral")
                is_verified_enrichment = int(is_novel and is_verified)

                vqa_answer = f"{p1_verified}/{p1_total} confirmed"

                final_metrics.append({
                    "image":                  img_name,
                    "pdf":                    pdf_base,
                    # NLI fields
                    "nli_status":             nli_label,
                    "nli_grounding_score":    nli_grounding_score,
                    "nli_novelty_score":      nli_novelty_score,
                    # VQA fields
                    "vqa_verification":       vqa_answer.strip(),
                    "vqa_atomic_score":       vqa_p1_score,
                    "vqa_p1_atomic_score":    vqa_p1_score,
                    "vqa_p2_atomic_score":    vqa_p2_score,
                    "vqa_p2_verification":    p2_answer,
                    # Legacy composite flags
                    "is_novel":               is_novel,
                    "is_visually_grounded":   int(is_verified),
                    "is_verified_enrichment": is_verified_enrichment
                })

                print(
                    f"{img_name} | NLI_grounding: {nli_grounding_score:.3f} | "
                    f"VQA_P1: {vqa_p1_score:.3f} ({vqa_answer}) | "
                    f"VQA_P2: {vqa_p2_score if vqa_p2_score is not None else 'n/a'} | "
                    f"ENR: {bool(is_verified_enrichment)}"
                )

            except Exception as e:
                import traceback
                print(f"\n[FAILED IMAGE] {img_name}")
                print(traceback.format_exc())

    # --- SUMMARY ---
    if final_metrics:
        df = pd.DataFrame(final_metrics)
        per_pdf_summary = {}
        for pdf_name, pdf_group in df.groupby("pdf"):
            nli_dist = pdf_group["nli_status"].value_counts().to_dict()
            per_pdf_summary[pdf_name] = {
                "image_count":               len(pdf_group),
                "novelty_rate":              round(float(pdf_group["is_novel"].mean()), 4),
                "mean_nli_grounding_score":  round(float(pdf_group["nli_grounding_score"].mean()), 4),
                "mean_nli_novelty_score":    round(float(pdf_group["nli_novelty_score"].mean()), 4),
                "visual_grounding_rate":     round(float(pdf_group["is_visually_grounded"].mean()), 4),
                "verified_enrichment_rate":  round(float(pdf_group["is_verified_enrichment"].mean()), 4),
                "mean_vqa_atomic_score":     round(float(pdf_group["vqa_atomic_score"].mean()), 4),
                "mean_vqa_p1_score":         round(float(pdf_group["vqa_p1_atomic_score"].mean()), 4),
                "mean_vqa_p2_score":         round(float(
                    pdf_group["vqa_p2_atomic_score"].dropna().mean()
                ), 4) if pdf_group["vqa_p2_atomic_score"].notna().any() else None,
                "nli_distribution":          nli_dist
            }

        summary = {
            "model":                          FOLDER_NAME,
            "total_images_evaluated":         len(df),
            "total_pdfs_evaluated":           len(per_pdf_summary),
            "global_novelty_rate":            round(float(df["is_novel"].mean()), 4),
            "global_mean_nli_grounding":      round(float(df["nli_grounding_score"].mean()), 4),
            "global_mean_nli_novelty":        round(float(df["nli_novelty_score"].mean()), 4),
            "global_visual_grounding_rate":   round(float(df["is_visually_grounded"].mean()), 4),
            "global_verified_enrichment_rate":round(float(df["is_verified_enrichment"].mean()), 4),
            "global_mean_vqa_atomic_score":   round(float(df["vqa_atomic_score"].mean()), 4),
            "global_mean_vqa_p1_score":       round(float(df["vqa_p1_atomic_score"].mean()), 4),
            "global_mean_vqa_p2_score":       round(float(
                df["vqa_p2_atomic_score"].dropna().mean()
            ), 4) if df["vqa_p2_atomic_score"].notna().any() else None,
            "global_nli_distribution":        df["nli_status"].value_counts().to_dict(),
            "per_pdf_summary":                per_pdf_summary,
            "details":                        final_metrics
        }
        with open(EVAL_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print("\n" + "=" * 60)
        print(f"ENRICHMENT REPORT — {FOLDER_NAME}")
        print(f"Total Images Evaluated     : {summary['total_images_evaluated']}")
        print(f"Total PDFs Evaluated       : {summary['total_pdfs_evaluated']}")
        print("-" * 60)
        print(f"Global NLI Grounding Score : {summary['global_mean_nli_grounding']:.3f}")
        print(f"Global NLI Novelty Score   : {summary['global_mean_nli_novelty']:.3f}")
        print(f"Global Novelty Rate        : {summary['global_novelty_rate']:.3f}")
        print(f"Global Visual Grounding    : {summary['global_visual_grounding_rate']:.3f}")
        print(f"Global Enrichment Rate     : {summary['global_verified_enrichment_rate']:.3f}")
        print(f"Global VQA P1 Score        : {summary['global_mean_vqa_p1_score']:.3f}")
        if summary["global_mean_vqa_p2_score"] is not None:
            print(f"Global VQA P2 Score        : {summary['global_mean_vqa_p2_score']:.3f}")
        print("-" * 60)
        print("Per-PDF Breakdown:")
        for pdf, data in per_pdf_summary.items():
            p2_str = f"{data['mean_vqa_p2_score']:.3f}" if data.get("mean_vqa_p2_score") is not None else "n/a"
            print(
                f"  {pdf} ({data['image_count']} imgs) | "
                f"NLI_grd: {data['mean_nli_grounding_score']:.3f} | "
                f"NLI_nov: {data['mean_nli_novelty_score']:.3f} | "
                f"VQA_P1: {data['mean_vqa_p1_score']:.3f} | "
                f"VQA_P2: {p2_str} | "
                f"Enrichment: {data['verified_enrichment_rate']:.3f}"
            )
        print("=" * 60)
    else:
        print("No metrics computed — check that caption files exist and are non-empty.")


if __name__ == "__main__":
    run_enrichment_evaluation()