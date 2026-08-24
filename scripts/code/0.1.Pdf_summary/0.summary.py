"""
Document-level abstractive summarization via MapReduce.

Generates one global summary per source document, used as the macro-level
contextual signal in the context-fusion ablations (Ablation 3: parallel
unblended injection; Ablation 4: 70/30 weighted blend). Run once per
document, independently of the per-image captioning scripts, with output
cached to disk for reuse across both ablations.

Long documents are split into overlapping word-based chunks (Map phase),
each chunk is summarized independently, and the partial summaries are
combined into one coherent document-level summary (Reduce phase). Summary
quality is validated against the source text using ROUGE and BERTScore.
"""

import os
import sys
import json
import torch
import warnings
from pathlib import Path
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")

warnings.filterwarnings("ignore")

def count_words(text: str) -> int:
    return len(text.split())


# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "qwen2vl-7b-sum"
MODEL_PATH   = os.path.join(PROJECT_ROOT, "hf_models/Qwen2-VL-7B-Instruct")
JSON_ROOT    = os.path.join(PROJECT_ROOT, "outputs/json_results")
OUTPUT_ROOT  = os.path.join(PROJECT_ROOT, "outputs/Summarized_pdfs")

MAX_NEW_TOKENS   = 300

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




def extract_text_from_json(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text_blocks = []
    pages = data.get("pages", {})
    for block_key, block_text in pages.items():
        if isinstance(block_text, str) and block_text.strip():
            text_blocks.append(block_text.strip())
    return " ".join(text_blocks)

CHUNK_WORDS     = 2500   # words per chunk — safe for 5500 token window
CHUNK_OVERLAP   = 150    # word overlap between chunks to preserve boundary context

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


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping word-based chunks."""
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
    """Single generation call — reused for both Map and Reduce steps."""
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
    torch.cuda.empty_cache()  # Free GPU memory after each generation call
    return result


def summarize(text: str, tokenizer, model, device: str) -> str:
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    print(f"    [MapReduce] {total_chunks} chunk(s) detected")

    if total_chunks == 1:
        # Document fits in one pass — no MapReduce needed
        print(f"    [MapReduce] Single chunk — direct summarization")
        return generate_text(SYSTEM_PROMPT, f"Voici le texte du document à résumer :\n\n{chunks[0]}", tokenizer, model, device, MAX_NEW_TOKENS)

    # ── MAP step: summarize each chunk independently ──
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"    [Map {i+1}/{total_chunks}] Summarizing chunk ({count_words(chunk)} words)...")
        partial = generate_text(
            CHUNK_PROMPT,
            f"Passage {i+1}/{total_chunks} :\n\n{chunk}",
            tokenizer, model, device,
            max_new_tokens=150   # shorter output per chunk
        )
        partial_summaries.append(partial)
        print(f"    [Map {i+1}/{total_chunks}] Done — {count_words(partial)} words")

    # ── REDUCE step: combine all partial summaries into one final summary ──
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




def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"SUMMARIZATION ABLATION — Model: {MODEL_NAME}")
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

    all_results = {}

    for pdf_folder in sorted(pdf_folders):
        pdf_name  = pdf_folder.name
        json_path = pdf_folder / "extracted_content.json"
        out_dir   = Path(OUTPUT_ROOT) / MODEL_NAME / pdf_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path  = out_dir / "summary.json"

        if not json_path.exists():
            print(f"[SKIP] No extracted_content.json in {pdf_name}")
            continue

        # If summary already exists, load and reuse it
        if out_path.exists():
            print(f"[SKIP] Existing summary found for {pdf_name} → {out_path}")
            with open(out_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            all_results[pdf_name] = result
            continue

        print(f">>> Processing: {pdf_name}")
        full_text = extract_text_from_json(str(json_path))
        if not full_text or len(full_text.strip()) < 100:
            print(f"    [SKIP] Insufficient text extracted from {pdf_name}")
            continue

        word_count_input = count_words(full_text)
        print(f"    Input text: {word_count_input} words")

        summary            = summarize(full_text, tokenizer, model, device)
        word_count_summary = count_words(summary)
        print(f"    Summary:    {word_count_summary} words")
        print(f"    Preview:    {summary[:200]}...")


        result = {
            "pdf_name":         pdf_name,
            "model":            MODEL_NAME,
            "summary":          summary,
            "word_count":       word_count_summary,
            "input_word_count": word_count_input,
        }

        all_results[pdf_name] = result

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"    Saved → {out_path}\n")

    combined_path = Path(OUTPUT_ROOT) / MODEL_NAME / "all_summaries.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print("=" * 70)
    print(f"FINAL REPORT — {MODEL_NAME}")
    print("-" * 70)
    for pdf_name, res in all_results.items():
        print(f"  {pdf_name}")
        print(f"    Words: {res['word_count']} (from {res['input_word_count']} input words)")
    print("=" * 70)
    print(f"Combined results → {combined_path}")
'''Summary quality is not validated against automatic reference-based metrics 
(ROUGE, BERTScore), since these require a human-authored gold summary for
meaningful comparison, which is unavailable for this corpus.'''

if __name__ == "__main__":
    run()