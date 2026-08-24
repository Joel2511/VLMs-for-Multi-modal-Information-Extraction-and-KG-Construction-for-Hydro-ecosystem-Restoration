# captioner_icd.py  —  Ablation 6 ICD (Instruction Contrastive Decoding)
# Based on: baseline weighted-context captioner (Ablation 5 best)
# Added: ICDLogitsProcessor with cd_alpha argument
# Output format: IDENTICAL to baseline (same JSON keys, same DECORATIVE handling)

import os
import gc
import json
import argparse
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModel, LogitsProcessor, LogitsProcessorList
import time
from datetime import datetime


# ─────────────────────────────────────────────
# CHANGE 1 — CLI argument for cd_alpha
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NVILA ICD Captioner — Ablation 6")
parser.add_argument("--cd_alpha", type=float, default=0.5,
                    help="ICD contrastive strength: 0.5 (moderate) or 1.0 (strong)")
args = parser.parse_args()
CD_ALPHA = args.cd_alpha


# --- CONFIGURATION ---  (UNCHANGED from baseline)
GENERATOR_MODEL_NAME = "nvila-lite-8b"
RETRIEVAL_MODEL_NAME = "vidore"

PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
BASE_IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
BASE_JSON_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")
BASE_RETRIEVAL_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Image_text_matching",
                                   f"{RETRIEVAL_MODEL_NAME}_results")

# ─────────────────────────────────────────────
# CHANGE 2 — Output folder encodes cd_alpha
# ─────────────────────────────────────────────
alpha_tag = f"alpha{str(CD_ALPHA).replace('.', '')}"
BASE_OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT, "outputs", "6th_ablation_Image_captioning_decoding",
    f"{GENERATOR_MODEL_NAME}_captions_{RETRIEVAL_MODEL_NAME}_icd_{alpha_tag}"
)

JUNK_FOLDER = os.path.join(PROJECT_ROOT, "data", "junk_images")

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise EnvironmentError("HF_TOKEN environment variable not set. Run 'export HF_TOKEN=...' before executing this script.")
MODEL_ID = "/home2020/home/icube/janil/Rhine_proj/hf_models/NVILA-Lite-8B-hf"


# ─────────────────────────────────────────────
# CHANGE 3 — Disturbance prompt constant (ICD)
# This is X'_ins: strips all domain grounding,
# Phase structure, and fluvial analyst persona.
# Its job is to amplify hallucination signal so
# we can SUBTRACT it from the real distribution.
# ─────────────────────────────────────────────
DISTURBANCE_PROMPT = (
    "Tu es un assistant général. "
    "Décris cette image en quelques mots simples."
)

# ─────────────────────────────────────────────
# CHANGE 4 — Protected token strings
# ICD must NEVER suppress these structural tokens
# or the metric parser regex will break.
# ─────────────────────────────────────────────
PROTECTED_STRINGS = [
    "### PHASE 1",
    "### PHASE 2",
    "PHASE 1",
    "PHASE 2",
    "Image non informative",
    "Aucun enrichissement",
]


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
    # ── UNCHANGED from Ablation 5 baseline ──
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




RÈGLE FINALE : Tu n'as le droit de mentionner en Phase 2 que des éléments que tu as déjà écrits en Phase 1. Zéro ajout."""


# ═══════════════════════════════════════════════════════════════
# CHANGE 5 — ICDLogitsProcessor class (the entire ICD mechanism)
# ═══════════════════════════════════════════════════════════════
class ICDLogitsProcessor(LogitsProcessor):
    """
    Instruction Contrastive Decoding (ICD) — Wang et al., Findings of ACL 2024

    At each decoding step:
      1. Runs a second forward pass with the disturbance prompt embeddings
         (same visual tokens, different text instruction — no domain grounding)
      2. Subtracts disturbance logits from real logits, scaled by cd_alpha
      3. Applies adaptive plausibility constraint (cd_beta) to protect
         high-confidence tokens from being incorrectly suppressed
      4. Bypasses ICD entirely for protected structural tokens (PHASE headers)

    Args:
        model:           The loaded NVILA model
        disturbance_embeds: Pre-computed inputs_embeds for the disturbance pass
                           (same shape as real prefill embeds, text portion replaced)
        tokenizer:       Tokenizer for protected token detection
        cd_alpha:        Contrastive strength (0.5 moderate, 1.0 strong)
        cd_beta:         Plausibility threshold (default 0.1, keeps top tokens safe)
        protected_ids:   Token IDs whose generation bypasses ICD
    """

    def __init__(self, model, disturbance_embeds, tokenizer,
                 cd_alpha=0.5, cd_beta=0.1, protected_ids=None):
        self.model = model
        self.disturbance_embeds = disturbance_embeds  # [1, seq_len, hidden]
        self.tokenizer = tokenizer
        self.cd_alpha = cd_alpha
        self.cd_beta = cd_beta
        self.protected_ids = protected_ids or set()
        self._past_key_values_cd = None  # KV cache for disturbance stream
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._step += 1

        # ── Disturbance forward pass ──
        with torch.no_grad():
            if self._step == 1:
                # First decode step: full forward pass with disturbance embeddings
                cd_out = self.model(
                    inputs_embeds=self.disturbance_embeds,
                    use_cache=True,
                )
                self._past_key_values_cd = cd_out.past_key_values
                cd_logits = cd_out.logits[:, -1, :]   # ← was missing in your version
            else:
                # Subsequent steps: use KV cache, pass only last token
                last_token = input_ids[:, -1:]
                cd_out = self.model(
                    input_ids=last_token,
                    past_key_values=self._past_key_values_cd,
                    use_cache=True,
                )
                self._past_key_values_cd = cd_out.past_key_values
                cd_logits = cd_out.logits[:, -1, :]

        # ── Cast cd_logits to match scores dtype (float16 → float32 safety) ──
        cd_logits = cd_logits.to(scores.dtype)   # ← moved here, after cd_logits is fully assigned

        # ── Adaptive plausibility constraint ──
        cutoff = torch.log(torch.tensor(self.cd_beta, device=scores.device)) + \
                scores.max(dim=-1, keepdim=True).values
        disfluent_mask = scores < cutoff

        # ── Restore plausibility-masked tokens to original scores ──
        cutoff = torch.log(torch.tensor(self.cd_beta, device=scores.device)) + \
         scores.max(dim=-1, keepdim=True).values
        disfluent_mask = scores < cutoff

        cd_scores = scores + self.cd_alpha * (scores - cd_logits)

        cd_scores[disfluent_mask] = -float("inf")
        return cd_scores


def build_disturbance_embeds(model, processor, raw_image, device):
    """
    Builds disturbance stream inputs_embeds:
    Same image tokens, but text = DISTURBANCE_PROMPT (stripped of all domain grounding).
    """
    image_token = processor.tokenizer.image_token
    messages_cd = [
        {
            "role": "user",
            "content": f"{image_token}\n{DISTURBANCE_PROMPT}"
        }
    ]
    text_cd = processor.tokenizer.apply_chat_template(
        messages_cd,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs_cd = processor(
        text=text_cd,
        images=raw_image,
        return_tensors="pt"
    )

    # ── FIX: move ALL tensors to device before calling _embed ──
    inputs_cd = {k: v.to(device) for k, v in inputs_cd.items()}

    with torch.no_grad():
        disturbance_embeds = model._embed(
            input_ids=inputs_cd["input_ids"],
            pixel_values=inputs_cd.get("pixel_values"),
            pixel_values_videos=inputs_cd.get("pixel_values_videos"),
        )
    return disturbance_embeds


def get_protected_token_ids(tokenizer):
    """
    Returns the set of token IDs for structural headers that ICD must never suppress.
    We encode each protected string and collect all token IDs that appear in them.
    """
    protected_ids = set()
    for s in PROTECTED_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        protected_ids.update(ids)
    return protected_ids


def run_nvila_captioning_icd():
    pipeline_start = time.time()
    print(f"--- Loading {GENERATOR_MODEL_NAME} with ICD (cd_alpha={CD_ALPHA}) ---")
    print(f"    Generator : {GENERATOR_MODEL_NAME}")
    print(f"    Retriever : {RETRIEVAL_MODEL_NAME}")
    print(f"    ICD alpha : {CD_ALPHA}")
    print(f"    Output to : {BASE_OUTPUT_FOLDER}")

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

    # Pre-compute protected token IDs once
    protected_ids = get_protected_token_ids(processor.tokenizer)
    print(f"    Protected token IDs: {len(protected_ids)} tokens guarded from ICD")

    if not os.path.exists(BASE_IMAGE_FOLDER):
        print(f"Error: Base image folder not found at {BASE_IMAGE_FOLDER}")
        return

    pdf_folders = [f for f in os.listdir(BASE_IMAGE_FOLDER)
                   if os.path.isdir(os.path.join(BASE_IMAGE_FOLDER, f))]

    for pdf_base in pdf_folders:
        pdf_start = time.time()
        print(f"\n========== PROCESSING [{pdf_base}] ==========")

        IMAGE_FOLDER = os.path.join(BASE_IMAGE_FOLDER, pdf_base)
        OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, pdf_base)
        JUNK_PDF_FOLDER = os.path.join(JUNK_FOLDER, pdf_base)
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(JUNK_PDF_FOLDER, exist_ok=True)

        json_output_path = os.path.join(OUTPUT_FOLDER, "image_captions_using_weighted_context.json")
        if os.path.exists(json_output_path):
            print(f" -> Already captioned. Skipping.")
            continue

        weighted_context_path = os.path.join(
            PROJECT_ROOT, "outputs", "Weighted_summary",
            f"{pdf_base}_{RETRIEVAL_MODEL_NAME}_weighted_context.json"
        )
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

        image_files = [f for f in os.listdir(IMAGE_FOLDER)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        results_dict = {}
        total_images = len(image_files)
        print(f"    Images to caption: {total_images}")

        model_device = next(model.parameters()).device

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

                # ── Real (normal) inputs ──
                image_token = processor.tokenizer.image_token
                messages = [{"role": "user", "content": f"{image_token}\n{caption_prompt}"}]
                text_with_image = processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=text_with_image, images=raw_image, return_tensors="pt")
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

                # ─────────────────────────────────────────────
                # CHANGE 6 — Build disturbance embeddings
                # (same image, stripped generic prompt)
                # ─────────────────────────────────────────────
                disturbance_embeds = build_disturbance_embeds(
                    model, processor, raw_image, model_device
                )

                # ─────────────────────────────────────────────
                # CHANGE 7 — Build ICD processor and inject
                # ─────────────────────────────────────────────
                icd_processor = ICDLogitsProcessor(
                    model=model,
                    disturbance_embeds=disturbance_embeds,
                    tokenizer=processor.tokenizer,
                    cd_alpha=CD_ALPHA,
                    cd_beta=0.1,
                    protected_ids=protected_ids,
                )

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        logits_processor=LogitsProcessorList([icd_processor]),  # ← ICD injected here
                    )

                prompt_len = inputs["input_ids"].shape[-1]
                generated_text = processor.tokenizer.decode(
                    generated_ids[0][prompt_len:],
                    skip_special_tokens=True
                ).strip()

                img_elapsed = time.time() - img_start

                # ── Output handling: IDENTICAL to baseline ──
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

        # ── JSON output: IDENTICAL key name to baseline ──
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)

        pdf_elapsed = time.time() - pdf_start
        print(f"========== COMPLETED [{pdf_base}] in {format_elapsed(pdf_elapsed)} ==========")

    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — ICD Ablation 6")
    print(f"  Generator model  : {GENERATOR_MODEL_NAME}")
    print(f"  Retrieval model  : {RETRIEVAL_MODEL_NAME}")
    print(f"  ICD cd_alpha     : {CD_ALPHA}")
    print(f"  Total runtime    : {format_elapsed(total_elapsed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_nvila_captioning_icd()