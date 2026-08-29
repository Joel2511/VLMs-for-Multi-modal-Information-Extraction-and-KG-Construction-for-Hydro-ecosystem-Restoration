"""
Document-level abstractive summarization via MapReduce.

Generates one global summary per source document, used as the macro-level
contextual signal in the context-fusion ablations (Ablation 3: parallel
unblended injection; Ablation 4: 70/30 weighted blend). Run once per
document, independently of the per-image captioning scripts, with output
cached to disk for reuse across both ablations.

Long documents are split into overlapping word-based chunks (Map phase),
each chunk is summarized independently, and the partial summaries are
combined into one coherent document-level summary (Reduce phase).

Summary quality is not evaluated with automatic reference-based metrics such
as ROUGE or BERTScore because the corpus has no human-authored gold summaries.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

warnings.filterwarnings("ignore")


def count_words(text: str) -> int:
    return len(text.split())


# ─── Repository and runtime configuration ────────────────────────────────────
SCRIPT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(
    os.environ.get("RHINE_PROJECT_ROOT", str(SCRIPT_ROOT))
).expanduser().resolve()

HF_MODELS_DIR = Path(
    os.environ.get("HF_MODELS_DIR", str(PROJECT_ROOT / "hf_models"))
).expanduser().resolve()

MODEL_NAME = "qwen2vl-7b-sum"
MODEL_PATH = HF_MODELS_DIR / "Qwen2-VL-7B-Instruct"

DEFAULT_JSON_ROOT = PROJECT_ROOT / "outputs" / "json_results"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs" / "summarized_pdfs"

MAX_NEW_TOKENS = 300

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

CHUNK_WORDS = 2500
CHUNK_OVERLAP = 150

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MapReduce document summaries from Pipeline 1 "
            "extracted-content JSON files."
        )
    )
    parser.add_argument(
        "--document",
        default="all",
        help=(
            "Name of one document folder under --input-root, or 'all' "
            "to process every available document. Default: all."
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_JSON_ROOT,
        help=(
            "Directory containing per-document extracted_content.json files. "
            f"Default: {DEFAULT_JSON_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Directory for newly generated summaries. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )
    return parser.parse_args()


def extract_text_from_json(json_path: Path) -> str:
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    text_blocks = []
    pages = data.get("pages", {})

    for block_text in pages.values():
        if isinstance(block_text, str) and block_text.strip():
            text_blocks.append(block_text.strip())

    return " ".join(text_blocks)


def chunk_text(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP,
) -> list[str]:
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
    """Single generation call reused for both Map and Reduce steps."""
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    torch.cuda.empty_cache()
    return result


def summarize(text: str, tokenizer, model, device: str) -> str:
    chunks = chunk_text(text)
    total_chunks = len(chunks)

    print(f"    [MapReduce] {total_chunks} chunk(s) detected")

    if total_chunks == 1:
        print("    [MapReduce] Single chunk — direct summarization")
        return generate_text(
            SYSTEM_PROMPT,
            f"Voici le texte du document à résumer :\n\n{chunks[0]}",
            tokenizer,
            model,
            device,
            MAX_NEW_TOKENS,
        )

    partial_summaries = []

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"    [Map {index}/{total_chunks}] "
            f"Summarizing chunk ({count_words(chunk)} words)..."
        )

        partial = generate_text(
            CHUNK_PROMPT,
            f"Passage {index}/{total_chunks} :\n\n{chunk}",
            tokenizer,
            model,
            device,
            max_new_tokens=150,
        )

        partial_summaries.append(partial)
        print(
            f"    [Map {index}/{total_chunks}] "
            f"Done — {count_words(partial)} words"
        )

    combined = "\n\n".join(
        f"Résumé partiel {index} :\n{summary}"
        for index, summary in enumerate(partial_summaries, start=1)
    )

    print(f"    [Reduce] Combining {total_chunks} partial summaries...")

    final_summary = generate_text(
        REDUCE_PROMPT,
        f"Voici les résumés partiels à combiner :\n\n{combined}",
        tokenizer,
        model,
        device,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print(f"    [Reduce] Final summary: {count_words(final_summary)} words")
    return final_summary


def select_document_folders(input_root: Path, document: str) -> list[Path]:
    if not input_root.exists():
        print(f"[ERROR] Input directory does not exist: {input_root}")
        sys.exit(1)

    available_folders = sorted(
        directory for directory in input_root.iterdir() if directory.is_dir()
    )

    if not available_folders:
        print(f"[ERROR] No document folders found in {input_root}")
        sys.exit(1)

    if document == "all":
        return available_folders

    selected_folder = input_root / document

    if not selected_folder.is_dir():
        available_names = ", ".join(folder.name for folder in available_folders)
        print(f"[ERROR] Requested document folder was not found: {document}")
        print(f"[INFO] Available document folders: {available_names}")
        sys.exit(1)

    return [selected_folder]


def load_existing_combined_results(combined_path: Path) -> dict:
    if not combined_path.exists():
        return {}

    try:
        with open(combined_path, "r", encoding="utf-8") as file:
            existing_results = json.load(file)

        if isinstance(existing_results, dict):
            return existing_results

        print(
            f"[WARNING] Existing combined file is not a JSON object: "
            f"{combined_path}. It will be replaced."
        )
        return {}

    except (json.JSONDecodeError, OSError) as error:
        print(
            f"[WARNING] Could not load existing combined results from "
            f"{combined_path}: {error}. It will be replaced."
        )
        return {}


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    print("=" * 70)
    print(f"SUMMARIZATION ABLATION — Model: {MODEL_NAME}")
    print(f"Device: {device.upper()}")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Document selection: {args.document}")
    print("=" * 70)

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model directory not found: {MODEL_PATH}")
        print(
            "[INFO] Download the model as described in MODEL_SETUP.md or set "
            "HF_MODELS_DIR in config/project.env."
        )
        sys.exit(1)

    print(f"\nLoading tokenizer and model from {MODEL_PATH} ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    print("Model loaded.\n")

    pdf_folders = select_document_folders(input_root, args.document)

    combined_path = output_root / MODEL_NAME / "all_summaries.json"
    all_results = load_existing_combined_results(combined_path)

    for pdf_folder in pdf_folders:
        pdf_name = pdf_folder.name
        json_path = pdf_folder / "extracted_content.json"
        out_dir = output_root / MODEL_NAME / pdf_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "summary.json"

        if not json_path.exists():
            print(f"[SKIP] No extracted_content.json in {pdf_name}")
            continue

        if out_path.exists():
            print(f"[SKIP] Existing summary found for {pdf_name} → {out_path}")

            with open(out_path, "r", encoding="utf-8") as file:
                result = json.load(file)

            all_results[pdf_name] = result
            continue

        print(f">>> Processing: {pdf_name}")

        full_text = extract_text_from_json(json_path)

        if not full_text or len(full_text.strip()) < 100:
            print(f"    [SKIP] Insufficient text extracted from {pdf_name}")
            continue

        word_count_input = count_words(full_text)
        print(f"    Input text: {word_count_input} words")

        summary = summarize(full_text, tokenizer, model, device)
        word_count_summary = count_words(summary)

        print(f"    Summary: {word_count_summary} words")
        print(f"    Preview: {summary[:200]}...")

        result = {
            "pdf_name": pdf_name,
            "model": MODEL_NAME,
            "summary": summary,
            "word_count": word_count_summary,
            "input_word_count": word_count_input,
        }

        all_results[pdf_name] = result

        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)

        print(f"    Saved → {out_path}\n")

    combined_path.parent.mkdir(parents=True, exist_ok=True)

    with open(combined_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4, ensure_ascii=False)

    print("=" * 70)
    print(f"FINAL REPORT — {MODEL_NAME}")
    print("-" * 70)

    for pdf_name, result in all_results.items():
        print(f"  {pdf_name}")
        print(
            f"    Words: {result['word_count']} "
            f"(from {result['input_word_count']} input words)"
        )

    print("=" * 70)
    print(f"Combined results → {combined_path}")


if __name__ == "__main__":
    run(parse_args())