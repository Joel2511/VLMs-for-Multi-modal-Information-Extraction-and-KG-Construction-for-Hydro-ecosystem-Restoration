import os
import sys
import json
import csv
import torch
import warnings
from pathlib import Path
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME     = "qwen2vl-7b-sum"
MODEL_PATH     = "/home2020/home/icube/janil/Rhine_proj/hf_models/Qwen2-VL-7B-Instruct"
JSON_ROOT      = "/home2020/home/icube/janil/Rhine_proj/outputs/json_results"
SUMMARY_ROOT   = "/home2020/home/icube/janil/Rhine_proj/outputs/Summarized_pdfs"
RETRIEVAL_ROOT = "/home2020/home/icube/janil/Rhine_proj/outputs/Image_text_matching"
OUTPUT_ROOT    = "/home2020/home/icube/janil/Rhine_proj/outputs/Weighted_summary"

RETRIEVERS     = ["vidore"]
MAX_NEW_TOKENS = 300
TOP_K          = 3

# ─── PROMPTS (French only) ────────────────────────────────────────────────────

CHUNK_PROMPT = (
    "Tu es un outil de résumé automatique. "
    "Résume uniquement le passage suivant en 60 à 80 mots en français. "
    "Sois factuel et n'utilise que les informations présentes dans ce passage. "
    "N'ajoute rien qui ne soit pas dans le texte."
)

REDUCE_PROMPT = (
    "Tu es un outil de résumé automatique. "
    "Tu reçois plusieurs résumés partiels d'un même document. "
    "Produis un résumé final cohérent en 100 à 150 mots en français, "
    "en couvrant : (1) le sujet principal et la zone d'étude, "
    "(2) les résultats ou constats clés, "
    "(3) les recommandations ou conclusions si présentes. "
    "Règles absolues : n'utilise que les informations des résumés partiels fournis ; "
    "ne généralise pas un résultat partiel à l'ensemble du document ; "
    "ne déduis rien qui ne soit pas explicitement écrit."
)

SYSTEM_PROMPT = (
    "Tu es un outil de résumé automatique. "
    "Ta seule tâche est de produire un résumé fidèle et factuel du texte fourni, "
    "en 100 à 150 mots en français. "
    "Avant de rédiger le résumé, identifie mentalement : "
    "(1) le sujet principal et la zone géographique ou d'étude, "
    "(2) les résultats ou constats clés mentionnés dans le texte, "
    "(3) les recommandations ou conclusions présentes. "
    "Rédige ensuite un résumé en prose continue couvrant ces trois points. "
    "Règles absolues : "
    "n'utilise que les informations présentes dans le texte fourni ; "
    "ne généralise pas un résultat partiel à l'ensemble du document ; "
    "ne cite aucune source externe ; "
    "ne déduis rien qui ne soit pas explicitement écrit."
)

FUSION_SYSTEM_PROMPT = (
    "Tu es un assistant spécialisé dans la création de contextes descriptifs pour des images scientifiques. "
    "Tu reçois deux sources d'information : "
    "(1) un RÉSUMÉ DU DOCUMENT — contexte global, à utiliser uniquement pour le cadre thématique, "
    "la terminologie et la zone d'étude ; "
    "(2) des BLOCS DE CONTEXTE SPÉCIFIQUES À L'IMAGE — extraits du document directement liés à cette image, "
    "qui constituent ta source principale. "
    "Ta tâche : rédige un passage de contexte de 4 à 6 phrases en français "
    "qui ancre cette image spécifique dans le contenu du document. "
    "Règles absolues : "
    "accorde 70 % de ton contenu aux blocs spécifiques à l'image ; "
    "utilise le résumé du document uniquement pour ajouter la terminologie ou le cadrage thématique (30 % maximum) ; "
    "n'invente aucun fait — utilise uniquement les informations présentes dans les deux sources fournies ; "
    "ne mentionne pas les numéros de blocs ni les métadonnées ; "
    "rédige uniquement le passage de contexte, sans introduction ni conclusion."
)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, chunk_words: int = 2500, overlap_words: int = 150) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


def generate_text(prompt_system, user_text, tokenizer, model, device, max_new_tokens):
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    torch.cuda.empty_cache()
    return result


def summarize(text: str, tokenizer, model, device: str) -> str:
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    print(f"    [MapReduce] {total_chunks} chunk(s) detected")

    if total_chunks == 1:
        print(f"    [MapReduce] Single chunk — direct summarization")
        return generate_text(
            SYSTEM_PROMPT,
            f"Voici le texte du document à résumer :\n\n{chunks[0]}",
            tokenizer, model, device, MAX_NEW_TOKENS
        )

    partial_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"    [Map {i+1}/{total_chunks}] Summarizing chunk ({count_words(chunk)} words)...")
        partial = generate_text(
            CHUNK_PROMPT,
            f"Passage {i+1}/{total_chunks} :\n\n{chunk}",
            tokenizer, model, device,
            max_new_tokens=150
        )
        partial_summaries.append(partial)
        print(f"    [Map {i+1}/{total_chunks}] Done — {count_words(partial)} words")

    combined = "\n\n".join([f"Résumé partiel {i+1} :\n{s}" for i, s in enumerate(partial_summaries)])
    print(f"    [Reduce] Combining {total_chunks} partial summaries...")
    final_summary = generate_text(
        REDUCE_PROMPT,
        f"Voici les résumés partiels à combiner :\n\n{combined}",
        tokenizer, model, device,
        max_new_tokens=MAX_NEW_TOKENS
    )
    print(f"    [Reduce] Final summary: {count_words(final_summary)} words")
    return final_summary


def extract_text_from_json(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text_blocks = []
    pages = data.get("pages", {})
    for block_key, block_text in pages.items():
        if isinstance(block_text, str) and block_text.strip():
            text_blocks.append(block_text.strip())
    return " ".join(text_blocks)


def get_block_texts_from_pages(pages: dict, block_keys: list) -> list:
    texts = []
    for bk in block_keys:
        text = pages.get(bk, "").strip()
        if text:
            texts.append(text)
        else:
            print(f"    [WARN] Block key '{bk}' not found in extracted_content.json")
    return texts


def load_existing_summary(pdf_name: str) -> str:
    summary_path = Path(SUMMARY_ROOT) / MODEL_NAME / pdf_name / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("summary", "")
    return ""


def parse_retrieval_csv(csv_path: str, top_k: int = TOP_K) -> dict:
    image_to_blocks = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",") 
        for row in reader:
            img_name = row.get("Image_Name", "").strip()
            if not img_name:
                continue
            img_stem = Path(img_name).stem
            blocks = []
            for k in range(1, top_k + 1):
                block_key = row.get(f"Top_{k}_Block", "").strip()
                if block_key:
                    blocks.append(block_key)
            image_to_blocks[img_stem] = blocks
    return image_to_blocks


def generate_weighted_context(doc_summary: str, block_texts: list,
                               tokenizer, model, device: str) -> str:
    if not block_texts:
        return doc_summary

    blocks_formatted = "\n\n".join(
        [f"Bloc {i+1} :\n{t}" for i, t in enumerate(block_texts)]
    )
    user_text = (
        f"RÉSUMÉ DU DOCUMENT (contexte global — cadrage thématique uniquement) :\n"
        f"{doc_summary}\n\n"
        f"---\n\n"
        f"BLOCS DE CONTEXTE SPÉCIFIQUES À L'IMAGE (source principale — 70% du contenu) :\n"
        f"{blocks_formatted}"
    )
    return generate_text(
        FUSION_SYSTEM_PROMPT,
        user_text,
        tokenizer, model, device,
        max_new_tokens=MAX_NEW_TOKENS
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"WEIGHTED CONTEXT GENERATION — Model: {MODEL_NAME}")
    print(f"Device: {device.upper()}")
    print("=" * 70)

    print(f"\nLoading tokenizer and model from {MODEL_PATH} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded.\n")

    pdf_folders = [d for d in Path(JSON_ROOT).iterdir() if d.is_dir()]
    if not pdf_folders:
        print(f"[ERROR] No PDF folders found in {JSON_ROOT}")
        sys.exit(1)

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    for retriever in RETRIEVERS:
        retriever_dir = Path(RETRIEVAL_ROOT) / f"{retriever}_results"
        if not retriever_dir.exists():
            print(f"[SKIP] Retriever directory not found: {retriever_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"RETRIEVER: {retriever.upper()}")
        print(f"{'='*70}")

        for pdf_folder in sorted(pdf_folders):
            pdf_name  = pdf_folder.name
            json_path = pdf_folder / "extracted_content.json"

            if not json_path.exists():
                print(f"[SKIP] No extracted_content.json in {pdf_name}")
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                pages_dict = json.load(f).get("pages", {})

            csv_path = retriever_dir / pdf_name / f"{retriever}_raw_matrix.csv"
            if not csv_path.exists():
                print(f"[SKIP] No retrieval CSV found at {csv_path}")
                continue

            print(f"\n>>> PDF: {pdf_name} | Retriever: {retriever}")

            out_filename = f"{pdf_name}_{retriever}_weighted_context.json"
            out_path = Path(OUTPUT_ROOT) / out_filename
            if out_path.exists():
                print(f"    [SKIP] Weighted context already exists for {pdf_name} ({retriever})")
                continue

            doc_summary = load_existing_summary(pdf_name)
            if doc_summary:
                print(f"    [Summary] Loaded existing summary ({count_words(doc_summary)} words)")
            else:
                print(f"    [Summary] No existing summary found — generating...")
                full_text = extract_text_from_json(str(json_path))
                if not full_text or len(full_text.strip()) < 100:
                    print(f"    [SKIP] Insufficient text in {pdf_name}")
                    continue
                doc_summary = summarize(full_text, tokenizer, model, device)
                print(f"    [Summary] Generated ({count_words(doc_summary)} words)")

            image_to_blocks = parse_retrieval_csv(str(csv_path), top_k=TOP_K)
            print(f"    [Retrieval] {len(image_to_blocks)} images found in CSV")

            weighted_contexts = {}
            for img_stem, block_keys in sorted(image_to_blocks.items()):
                block_texts = get_block_texts_from_pages(pages_dict, block_keys)
                print(f"    [{img_stem}] Blocks: {block_keys} → {len(block_texts)} text(s) found")

                fused = generate_weighted_context(
                    doc_summary, block_texts, tokenizer, model, device
                )
                weighted_contexts[img_stem] = {
                    "weighted_context": fused,
                    "doc_summary_used": doc_summary,
                    "blocks_used":      block_keys,
                    "block_texts":      block_texts,
                    "word_count":       count_words(fused)
                }
                print(f"    [{img_stem}] Fused context: {count_words(fused)} words — {fused[:120]}...")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(weighted_contexts, f, indent=4, ensure_ascii=False)
            print(f"    Saved → {out_path}")

    print("\n" + "=" * 70)
    print("WEIGHTED CONTEXT GENERATION COMPLETE")
    print(f"Outputs saved to: {OUTPUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    run()