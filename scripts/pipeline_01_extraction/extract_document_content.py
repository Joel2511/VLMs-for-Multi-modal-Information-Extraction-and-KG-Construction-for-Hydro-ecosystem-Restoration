"""
Pipeline 1: Layout Parsing and Cross-Modal Extraction.

Processes French technical PDF documents through layout-aware text extraction,
300-DPI page rasterization, DocLayout-YOLO figure detection, geometric figure
merging, caption association, structural-noise filtering, OCR, and OCR-based
caption deduplication.

The extraction and filtering logic reproduces the procedure used in the thesis.
This version adds portable path handling and allows processing one document or
the complete input PDF collection without overwriting archived thesis outputs.
"""

import argparse
import gc
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import fitz
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doclayout_yolo import YOLOv10
from PIL import Image


# ─── Repository and runtime configuration ────────────────────────────────────

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("RHINE_PROJECT_ROOT", str(SCRIPT_ROOT))
).expanduser().resolve()

HF_MODELS_DIR = Path(
    os.environ.get("HF_MODELS_DIR", str(PROJECT_ROOT / "hf_models"))
).expanduser().resolve()

DEFAULT_PDF_ROOT = PROJECT_ROOT / "data" / "raw_pdfs"
DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT / "outputs" / "runs" / "extraction" / "extracted_images"
)
DEFAULT_JSON_ROOT = (
    PROJECT_ROOT / "outputs" / "runs" / "extraction" / "json_results"
)

YOLO_BATCH_SIZE = 8
OCR_BATCH_SIZE = 16

REPEATER_PAGE_RATIO = 0.6
REPEATER_FUZZY_THRESHOLD = 0.88

_ABBREVIATIONS = (
    r"(?:M|Mme|Mlle|Dr|Prof|p|fig|id|etc|av|bd|ca|cf|vol|et\s*al|ibid|"
    r"p\.\s*ex|art|al|ch|sect|tab)\.?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract figures, text blocks, and OCR content from one PDF or "
            "all PDFs in the input directory."
        )
    )

    parser.add_argument(
        "--document",
        default="all",
        help=(
            "PDF filename, PDF stem, or 'all'. When a PDF stem is supplied, "
            "the script resolves it to <stem>.pdf. Default: all."
        ),
    )

    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=DEFAULT_PDF_ROOT,
        help=(
            "Directory containing input PDF documents. "
            f"Default: {DEFAULT_PDF_ROOT}"
        ),
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help=(
            "Directory for newly extracted image crops. "
            f"Default: {DEFAULT_IMAGE_ROOT}"
        ),
    )

    parser.add_argument(
        "--json-root",
        type=Path,
        default=DEFAULT_JSON_ROOT,
        help=(
            "Directory for newly generated extracted-content and OCR JSON "
            f"files. Default: {DEFAULT_JSON_ROOT}"
        ),
    )

    return parser.parse_args()


def normalize_block(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[header\]", "", text)
    text = re.sub(r"\s+\d{1,4}\s*$", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_boilerplate_ngrams(
    temp_blocks_store: dict,
    total_pages: int,
) -> tuple:
    from collections import defaultdict

    def _collect(ngram_size, threshold):
        ngram_page_presence = defaultdict(set)

        for page_num, blocks in temp_blocks_store.items():
            full_page_text = " ".join(blocks)
            words = re.sub(r"\s+", " ", full_page_text).split()
            seen = set()

            for index in range(len(words) - ngram_size + 1):
                ngram = " ".join(words[index : index + ngram_size]).lower()
                seen.add(ngram)

            for ngram in seen:
                ngram_page_presence[ngram].add(page_num)

        return {
            ngram
            for ngram, pages in ngram_page_presence.items()
            if len(pages) / total_pages >= threshold
        }

    boilerplate_6grams = _collect(6, 0.35)
    boilerplate_3grams = _collect(3, 0.60)

    if boilerplate_6grams or boilerplate_3grams:
        print(
            "  [FILTER-NGR] Boilerplate n-grams: "
            f"{len(boilerplate_6grams)} (6-gram) + "
            f"{len(boilerplate_3grams)} (3-gram)"
        )

    return boilerplate_6grams, boilerplate_3grams


def strip_boilerplate_from_text(text: str, bp6: set, bp3: set) -> str:
    def _strip(current_text, ngrams, ngram_size):
        words = re.sub(r"\s+", " ", current_text).split()

        if len(words) < ngram_size:
            return current_text

        is_boilerplate = [False] * len(words)

        for index in range(len(words) - ngram_size + 1):
            ngram = " ".join(words[index : index + ngram_size]).lower()

            if ngram in ngrams:
                for word_index in range(index, index + ngram_size):
                    is_boilerplate[word_index] = True

        return " ".join(
            word
            for index, word in enumerate(words)
            if not is_boilerplate[index]
        )

    text = _strip(text, bp6, 6)
    text = _strip(text, bp3, 3)
    text = re.sub(r"^\s*\d{1,4}\s+", "", text)
    text = re.sub(r"\s+\d{1,4}\s*$", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def is_numeric_row(text):
    words = text.split()

    if not words:
        return False

    numeric_like = sum(
        1 for word in words if re.match(r"^[\d.,%]+$", word)
    )

    return numeric_like / len(words) > 0.4


def is_sentence_ender(text, next_text=None):
    if is_numeric_row(text):
        return True

    text = text.strip()

    if next_text and next_text.strip() and next_text.strip()[0].islower():
        return False

    sticky_words = [
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "du",
        "de",
        "d'",
        "au",
        "aux",
        "et",
        "ou",
        "mais",
        "en",
        "pour",
        "dans",
        "une",
        "plus",
    ]

    last_word = (
        re.sub(r"[^\w]", "", text.split()[-1]).lower()
        if text.split()
        else ""
    )

    if last_word in sticky_words:
        return False

    if re.search(_ABBREVIATIONS, text, re.IGNORECASE):
        return False

    if re.match(
        r"^[\(\[]?(?:\d+|[a-z]|[ivxlcdm]+)[\.)\]]\s",
        text,
        re.IGNORECASE,
    ):
        return True

    if text.count("«") > text.count("»"):
        return False

    if re.search(r"(?<!\.)\.\.\.(?!\.)|…$", text):
        return True

    return bool(re.search(r"[\.!\?…]['\"»\s\)]*$", text))


def is_figure_block(text):
    if not text or not text.strip():
        return False

    text = text.strip()

    keywords = (
        r"(Figure|Fig\.|Tableau|Source|Planche|Carte|Photo|Graphique|"
        r"Schéma|Encadré|Photos)s?"
    )

    pattern = (
        rf"^(?:\[HEADER\]\s*)?{keywords}\s*"
        rf"(?:\d+[\.d]*[a-z]?)?\s*[:\-\.]"
    )

    return bool(re.match(pattern, text, re.IGNORECASE))


def is_toc_entry(text: str) -> bool:
    if not is_figure_block(text):
        return False

    if len(text) > 160:
        return False

    return bool(re.search(r"[\.\s]{2,}\s*\d{1,3}\s*$", text))


def clean_text_content(text):
    text = re.sub(r"^\s*\[HEADER\]\s*", "", text)
    text = re.sub(r"\s+([:;!\?])", r" \1", text)
    text = re.sub(r"\.{4,}", " ", text)
    return " ".join(text.split())


def rasterize_pages(pdf_path, dpi=300):
    document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages = []

    for page in document:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )
        pages.append(image)

    document.close()
    return pages


def detect_repeating_blocks(
    temp_blocks_store: dict,
    total_pages: int,
) -> set:
    from collections import defaultdict

    normalized_to_pages = defaultdict(set)

    for page_num, blocks in temp_blocks_store.items():
        for block in blocks:
            if not block:
                continue

            normalized = normalize_block(block)

            if len(normalized) < 10:
                continue

            normalized_to_pages[normalized].add(page_num)

    threshold_pages = max(2, int(total_pages * REPEATER_PAGE_RATIO))

    exact_repeaters = {
        normalized
        for normalized, pages in normalized_to_pages.items()
        if len(pages) >= threshold_pages
    }

    from collections import defaultdict as dd

    buckets = dd(list)

    for normalized in normalized_to_pages:
        if normalized not in exact_repeaters:
            buckets[normalized[:20]].append(normalized)

    fuzzy_repeaters = set()

    for norms in buckets.values():
        if len(norms) < 2:
            continue

        for first_index in range(len(norms)):
            for second_index in range(first_index + 1, len(norms)):
                similarity = SequenceMatcher(
                    None,
                    norms[first_index],
                    norms[second_index],
                ).ratio()

                if similarity >= REPEATER_FUZZY_THRESHOLD:
                    combined_pages = (
                        normalized_to_pages[norms[first_index]]
                        | normalized_to_pages[norms[second_index]]
                    )

                    if len(combined_pages) >= threshold_pages:
                        fuzzy_repeaters.add(norms[first_index])
                        fuzzy_repeaters.add(norms[second_index])

    all_repeaters = exact_repeaters | fuzzy_repeaters

    if all_repeaters:
        print(
            "  [FILTER] Cross-page deduplication: "
            f"{len(all_repeaters)} repeating block pattern(s) flagged as "
            "structural noise."
        )

    return all_repeaters


def is_structural_repeater(text: str, repeater_norms: set) -> bool:
    normalized = normalize_block(text)

    if normalized in repeater_norms:
        return True

    if len(normalized) < 15:
        return False

    for repeater in repeater_norms:
        if abs(len(normalized) - len(repeater)) > 40:
            continue

        if (
            SequenceMatcher(None, normalized, repeater).ratio()
            >= REPEATER_FUZZY_THRESHOLD
        ):
            return True

    return False


def _pick_yolo_device():
    """Choose a stable device for DocLayout-YOLO, falling back to CPU."""
    if not torch.cuda.is_available():
        return "cpu"

    try:
        compute_capability = torch.cuda.get_device_capability(0)

        if compute_capability[0] < 7:
            print(
                "  [YOLO-DEV] GPU compute capability "
                f"{compute_capability} < 7.0 — using CPU."
            )
            return "cpu"

        dummy = torch.zeros(1, 3, 64, 64, device="cuda:0")
        torch.nn.functional.avg_pool2d(dummy, 2)

        del dummy
        torch.cuda.empty_cache()

        return "cuda:0"

    except Exception as error:
        print(
            "  [YOLO-DEV] CUDA smoke-test failed "
            f"({error}) — falling back to CPU."
        )
        return "cpu"


def extract_pdf_data(pdf_path, img_out_dir, json_out_dir, layout_model):
    print(f"\n--- Final Extraction Pipeline: {os.path.basename(pdf_path)} ---")

    for filename in os.listdir(img_out_dir):
        if filename.lower().endswith((".jpeg", ".jpg", ".png")):
            os.remove(os.path.join(img_out_dir, filename))

    document = fitz.open(pdf_path)

    extracted_data = {
        "metadata": document.metadata,
        "total_pages": len(document),
        "pages": {},
    }

    temp_blocks_store = {}
    temp_blocks_with_bbox = {}

    for page_num in range(len(document)):
        page = document[page_num]
        width, height = page.rect.width, page.rect.height
        margin_y = height * 0.025

        page_dict = page.get_text("dict")
        raw_blocks = page_dict["blocks"]

        styles = {}
        line_heights = []

        for block in raw_blocks:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                line_heights.append(line["bbox"][3] - line["bbox"][1])

                for span in line["spans"]:
                    font_base = span["font"].split("+")[-1]
                    style_key = (
                        round(span["size"] * 2) / 2,
                        font_base,
                    )
                    styles[style_key] = (
                        styles.get(style_key, 0) + len(span["text"])
                    )

        body_style = max(styles, key=styles.get) if styles else (None, None)

        median_line_height = (
            sorted(line_heights)[len(line_heights) // 2]
            if line_heights
            else 15
        )

        dynamic_gap_threshold = median_line_height * 1.4

        all_lines = []

        for block in raw_blocks:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                line_y0, line_y1 = line["bbox"][1], line["bbox"][3]

                if line_y0 < margin_y or line_y1 > (height - margin_y):
                    continue

                spans = line["spans"]
                last_span_text = spans[-1]["text"] if spans else ""
                has_line_break_hyphen = last_span_text.rstrip().endswith("-")

                line_text = "".join(
                    span["text"] for span in spans
                ).strip()

                if not line_text:
                    continue

                main_span = max(spans, key=lambda span: len(span["text"]))
                font_base = main_span["font"].split("+")[-1]

                line_style = (
                    round(main_span["size"] * 2) / 2,
                    font_base,
                )

                all_lines.append(
                    {
                        "text": line_text,
                        "bbox": line["bbox"],
                        "style": line_style,
                        "line_break_hyphen": has_line_break_hyphen,
                    }
                )

        all_lines.sort(
            key=lambda line: (
                round(line["bbox"][1] / 5) * 5,
                line["bbox"][0],
            )
        )

        processed_page_blocks = []
        processed_page_blocks_bbox = []

        current_block_text = ""
        current_block_bbox = None

        last_y1 = -1
        last_x0 = -1
        last_style = (None, None)

        def flush_block():
            nonlocal current_block_text, current_block_bbox

            if current_block_text:
                processed_page_blocks.append(current_block_text)
                processed_page_blocks_bbox.append(current_block_bbox)

            current_block_text = ""
            current_block_bbox = None

        for line_index, line in enumerate(all_lines):
            line_x0, line_y0, line_x1, line_y1 = line["bbox"]
            text = line["text"]
            current_style = line["style"]

            sub_parts = re.split(
                r"(?i)(?=(?:Figure|Fig\.|Tableau|Source|Planche|Carte)"
                r"\s*(?:\d+)?\s*[:\-\.])",
                text,
            )

            sub_parts = [
                part.strip()
                for part in sub_parts
                if part.strip()
            ]

            for part in sub_parts:
                is_splitter = is_figure_block(part)

                is_vertically_close = (
                    last_y1 != -1
                    and (line_y0 - last_y1) < dynamic_gap_threshold
                )

                is_aligned = (
                    last_x0 != -1
                    and abs(line_x0 - last_x0) < 5
                )

                is_style_match = last_style == current_style

                is_unfinished = (
                    current_block_text
                    and not is_sentence_ender(current_block_text, part)
                )

                next_is_fresh_start = False

                if line_index + 1 < len(all_lines):
                    next_line = all_lines[line_index + 1]

                    next_gap = next_line["bbox"][1] - line_y1

                    if (
                        next_gap > (median_line_height * 0.9)
                        and next_line["text"]
                        and next_line["text"][0].isupper()
                    ):
                        if (
                            abs(next_line["bbox"][0] - line_x0) > 10
                            or next_gap > median_line_height
                        ):
                            next_is_fresh_start = True

                is_header_style = (
                    current_style[0] > body_style[0] * 1.25
                    if body_style[0]
                    else False
                )

                if is_splitter:
                    force_break = True
                elif is_unfinished:
                    force_break = False
                else:
                    force_break = (
                        is_header_style
                        or not is_aligned
                        or not is_style_match
                        or next_is_fresh_start
                        or not is_vertically_close
                    )

                if force_break:
                    flush_block()

                    current_block_text = (
                        f"[HEADER] {part}"
                        if is_header_style
                        else part
                    )

                    current_block_bbox = list(line["bbox"])
                    last_x0 = line_x0

                else:
                    if (
                        line.get("line_break_hyphen")
                        and current_block_text.endswith("-")
                    ):
                        current_block_text = (
                            current_block_text[:-1] + part
                        )
                    else:
                        current_block_text += " " + part

                    if current_block_bbox is None:
                        current_block_bbox = list(line["bbox"])
                    else:
                        current_block_bbox[0] = min(
                            current_block_bbox[0],
                            line_x0,
                        )
                        current_block_bbox[1] = min(
                            current_block_bbox[1],
                            line_y0,
                        )
                        current_block_bbox[2] = max(
                            current_block_bbox[2],
                            line_x1,
                        )
                        current_block_bbox[3] = max(
                            current_block_bbox[3],
                            line_y1,
                        )

                last_y1 = line_y1
                last_style = current_style

        flush_block()

        temp_blocks_store[page_num] = processed_page_blocks

        temp_blocks_with_bbox[page_num] = list(
            zip(processed_page_blocks, processed_page_blocks_bbox)
        )

    # --- CROSS-PAGE STITCHING ---
    for page_num in range(len(document)):
        current_blocks = temp_blocks_store[page_num]

        if page_num < len(document) - 1:
            next_blocks = temp_blocks_store[page_num + 1]

            if current_blocks and next_blocks:
                last_index = len(current_blocks) - 1

                if (
                    not is_figure_block(current_blocks[last_index])
                    and not is_sentence_ender(
                        current_blocks[last_index],
                        next_blocks[0] if next_blocks else None,
                    )
                ):
                    next_text_index = -1

                    for index in range(len(next_blocks)):
                        if (
                            next_blocks[index]
                            and not is_figure_block(next_blocks[index])
                        ):
                            next_text_index = index
                            break

                    if next_text_index != -1:
                        first_text = next_blocks[next_text_index]

                        match = re.search(r"([.!?])(?:\s|$)", first_text)

                        if match:
                            split_position = match.end()

                            current_blocks[last_index] += (
                                " " + first_text[:split_position]
                            )

                            remainder = first_text[split_position:].strip()

                            if remainder:
                                next_blocks[next_text_index] = remainder
                            else:
                                next_blocks[next_text_index] = ""
                        else:
                            current_blocks[last_index] += " " + first_text
                            next_blocks[next_text_index] = ""

    document.close()

    existing_title = (
        extracted_data["metadata"].get("title") or ""
    ).strip()

    if not existing_title:
        first_page_blocks = temp_blocks_store.get(0, [])

        for block in first_page_blocks:
            if not block:
                continue

            candidate = re.sub(
                r"^\s*\[HEADER\]\s*",
                "",
                block,
            ).strip()

            candidate = clean_text_content(candidate)

            if len(candidate) > 8:
                extracted_data["metadata"]["title"] = candidate

                print(
                    "  [TITLE-FALLBACK] Derived title from page 1: "
                    f"{candidate[:70]}..."
                )

                break

    # --- CROSS-PAGE REPEATER DETECTION ---
    repeater_norms = detect_repeating_blocks(
        temp_blocks_store,
        len(temp_blocks_store),
    )

    bp6, bp3 = build_boilerplate_ngrams(
        temp_blocks_store,
        len(temp_blocks_store),
    )

    # --- RASTERIZE + YOLO ---
    print("Rasterizing pages at 300 DPI...")

    page_images = rasterize_pages(pdf_path, dpi=300)
    dpi = 300
    zoom = dpi / 72.0

    yolo_device = _pick_yolo_device()

    print(f"Running DocLayout-YOLO (batched) on {yolo_device}...")

    all_detection_results = []

    for batch_start in range(0, len(page_images), YOLO_BATCH_SIZE):
        batch = page_images[batch_start : batch_start + YOLO_BATCH_SIZE]

        detection_results = layout_model.predict(
            batch,
            imgsz=1024,
            conf=0.2,
            device=yolo_device,
            verbose=False,
        )

        all_detection_results.extend(detection_results)

    abandon_regions_pdf = {}
    caption_regions_pdf = {}

    for page_num, (page_image, detection_result) in enumerate(
        zip(page_images, all_detection_results)
    ):
        figure_boxes = []
        caption_boxes = []
        abandon_boxes_pixel = []

        for box in detection_result.boxes:
            class_id = int(box.cls[0])
            class_name = layout_model.names[class_id].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if class_name in ["figure", "picture", "image"]:
                figure_boxes.append([x1, y1, x2, y2])

            elif class_name in [
                "figure_caption",
                "figurecaption",
                "caption",
            ]:
                caption_boxes.append([x1, y1, x2, y2])

            elif class_name == "abandon":
                abandon_boxes_pixel.append([x1, y1, x2, y2])

        if abandon_boxes_pixel:
            abandon_regions_pdf[page_num] = [
                [x1 / zoom, y1 / zoom, x2 / zoom, y2 / zoom]
                for x1, y1, x2, y2 in abandon_boxes_pixel
            ]

        if caption_boxes:
            caption_regions_pdf[page_num] = [
                [x1 / zoom, y1 / zoom, x2 / zoom, y2 / zoom]
                for x1, y1, x2, y2 in caption_boxes
            ]

        if not figure_boxes:
            continue

        page_width, page_height = page_image.size
        horizontal_merge_margin = page_width * 0.08
        vertical_merge_margin = page_height * 0.02

        changed = True

        while changed:
            changed = False
            merged_boxes = []
            used = set()

            for first_index, first_box in enumerate(figure_boxes):
                if first_index in used:
                    continue

                for second_index, second_box in enumerate(figure_boxes):
                    if second_index <= first_index or second_index in used:
                        continue

                    horizontal_distance = max(
                        0,
                        second_box[0] - first_box[2],
                        first_box[0] - second_box[2],
                    )

                    vertical_distance = max(
                        0,
                        second_box[1] - first_box[3],
                        first_box[1] - second_box[3],
                    )

                    if (
                        horizontal_distance < horizontal_merge_margin
                        and vertical_distance < vertical_merge_margin
                    ):
                        first_box = [
                            min(first_box[0], second_box[0]),
                            min(first_box[1], second_box[1]),
                            max(first_box[2], second_box[2]),
                            max(first_box[3], second_box[3]),
                        ]

                        used.add(second_index)
                        changed = True

                merged_boxes.append(first_box)
                used.add(first_index)

            figure_boxes = merged_boxes

        for caption_box in caption_boxes:
            best_figure = None
            best_distance = float("inf")

            for figure_box in figure_boxes:
                horizontal_overlap = max(
                    0,
                    min(figure_box[2], caption_box[2])
                    - max(figure_box[0], caption_box[0]),
                )

                vertical_distance = caption_box[1] - figure_box[3]

                if (
                    horizontal_overlap > 0
                    and 0 <= vertical_distance < page_height * 0.08
                ):
                    if vertical_distance < best_distance:
                        best_distance = vertical_distance
                        best_figure = figure_box

            if best_figure is not None:
                best_figure[0] = min(best_figure[0], caption_box[0])
                best_figure[2] = max(best_figure[2], caption_box[2])
                best_figure[3] = max(best_figure[3], caption_box[3])

        image_index = 1

        for figure_box in figure_boxes:
            crop_rect = (
                max(0, figure_box[0] - 30),
                max(0, figure_box[1] - 30),
                min(page_image.width, figure_box[2] + 30),
                min(page_image.height, figure_box[3] + 30),
            )

            page_image.crop(crop_rect).save(
                os.path.join(
                    img_out_dir,
                    f"img_p{page_num + 1}_{image_index}.png",
                ),
                "PNG",
            )

            image_index += 1

    # --- FINAL BLOCK ASSEMBLY WITH ALL FILTERS ---
    def bbox_overlap_ratio(first_box, second_box):
        intersection_x0 = max(first_box[0], second_box[0])
        intersection_y0 = max(first_box[1], second_box[1])
        intersection_x1 = min(first_box[2], second_box[2])
        intersection_y1 = min(first_box[3], second_box[3])

        if intersection_x1 <= intersection_x0 or intersection_y1 <= intersection_y0:
            return 0.0

        intersection = (
            (intersection_x1 - intersection_x0)
            * (intersection_y1 - intersection_y0)
        )

        first_area = max(
            1e-6,
            (first_box[2] - first_box[0])
            * (first_box[3] - first_box[1]),
        )

        return intersection / first_area

    for page_num in range(len(temp_blocks_with_bbox)):
        blocks_with_bbox = temp_blocks_with_bbox.get(page_num, [])
        stitched_texts = temp_blocks_store[page_num]

        abandon_regions = abandon_regions_pdf.get(page_num, [])
        caption_regions = caption_regions_pdf.get(page_num, [])

        block_counter = 0

        for index, raw_text in enumerate(stitched_texts):
            if not raw_text:
                continue

            stripped = strip_boilerplate_from_text(raw_text, bp6, bp3)
            clean = clean_text_content(stripped)

            if is_toc_entry(clean):
                print(
                    "  [FILTER-TOC] Dropped TOC entry on page "
                    f"{page_num + 1}: {clean[:70]}..."
                )
                continue

            if is_structural_repeater(clean, repeater_norms):
                print(
                    "  [FILTER-RPT] Dropped repeating block on page "
                    f"{page_num + 1}: {clean[:70]}..."
                )
                continue

            block_bbox = None

            if index < len(blocks_with_bbox):
                _, block_bbox = blocks_with_bbox[index]

            if block_bbox is not None:
                dropped_abandon = False

                for abandon_box in abandon_regions:
                    if bbox_overlap_ratio(block_bbox, abandon_box) > 0.5:
                        print(
                            "  [FILTER-ABN] Dropped abandon-region block "
                            f"on page {page_num + 1}: {clean[:70]}..."
                        )
                        dropped_abandon = True
                        break

                if dropped_abandon:
                    continue

                dropped_caption = False

                for caption_box in caption_regions:
                    if bbox_overlap_ratio(block_bbox, caption_box) > 0.5:
                        print(
                            "  [FILTER-CAP] Dropped figure caption block "
                            f"on page {page_num + 1}: {clean[:70]}..."
                        )
                        dropped_caption = True
                        break

                if dropped_caption:
                    continue

            if len(clean) > 40:
                block_counter += 1

                block_key = (
                    f"page_{page_num + 1}_block_{block_counter}"
                )

                extracted_data["pages"][block_key] = clean

                print(
                    f"  [LOG] Finalized {block_key}: "
                    f"{clean[:60]}..."
                )

    with open(
        os.path.join(json_out_dir, "extracted_content.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(extracted_data, file, ensure_ascii=False, indent=4)


def postprocess_extracted_with_ocr(json_out_dir: str):
    extracted_path = os.path.join(json_out_dir, "extracted_content.json")
    ocr_path = os.path.join(json_out_dir, "ocr_results.json")

    if not os.path.exists(extracted_path) or not os.path.exists(ocr_path):
        print("  [OCR-DEDUP] Skipping post-process: missing files.")
        return

    with open(extracted_path, "r", encoding="utf-8") as file:
        extracted = json.load(file)

    ocr_token_sets = load_ocr_texts(ocr_path)

    if not ocr_token_sets:
        print("  [OCR-DEDUP] No OCR data found — skipping.")
        return

    original_count = len(extracted["pages"])
    clean_pages = {}
    dropped = 0

    for block_key, block_text in extracted["pages"].items():
        word_count = len(block_text.split())

        if word_count <= 22 and is_ocr_caption(block_text, ocr_token_sets):
            print(
                f"  [OCR-DEDUP] Dropped caption block '{block_key}': "
                f"{block_text[:70]}..."
            )
            dropped += 1
        else:
            clean_pages[block_key] = block_text

    renumbered = {}
    page_counters = {}

    for old_key, text in clean_pages.items():
        parts = old_key.rsplit("_block_", 1)
        page_prefix = parts[0]

        page_counters[page_prefix] = (
            page_counters.get(page_prefix, 0) + 1
        )

        new_key = f"{page_prefix}_block_{page_counters[page_prefix]}"
        renumbered[new_key] = text

    extracted["pages"] = renumbered

    with open(extracted_path, "w", encoding="utf-8") as file:
        json.dump(extracted, file, ensure_ascii=False, indent=4)

    print(
        f"  [OCR-DEDUP] Removed {dropped}/{original_count} blocks "
        "matching OCR image content."
    )


def run_ocr_on_folder(image_folder, output_json):
    if (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 7
    ):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"--- Running OCR on {device} ---")

    model = ocr_predictor(pretrained=True).to(device)

    image_files = sorted(
        filename
        for filename in os.listdir(image_folder)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    ocr_results = {}

    for batch_start in range(0, len(image_files), OCR_BATCH_SIZE):
        batch_files = image_files[
            batch_start : batch_start + OCR_BATCH_SIZE
        ]

        batch_paths = [
            os.path.join(image_folder, filename)
            for filename in batch_files
        ]

        batch_number = batch_start // OCR_BATCH_SIZE + 1
        total_batches = -(-len(image_files) // OCR_BATCH_SIZE)

        print(
            f"  [OCR] Batch {batch_number} / {total_batches} "
            f"({len(batch_files)} images)"
        )

        try:
            document = DocumentFile.from_images(batch_paths)
            result = model(document)

            for index, page in enumerate(result.pages):
                text = " ".join(
                    word.value
                    for block in page.blocks
                    for line in block.lines
                    for word in line.words
                )

                ocr_results[batch_files[index]] = text

        except Exception as error:
            print(f"  [OCR BATCH ERROR] {error} — marking batch as empty.")

            for filename in batch_files:
                ocr_results[filename] = ""

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(ocr_results, file, ensure_ascii=False, indent=4)


def load_ocr_texts(ocr_json_path: str) -> list:
    if not os.path.exists(ocr_json_path):
        return []

    with open(ocr_json_path, "r", encoding="utf-8") as file:
        ocr_data = json.load(file)

    token_sets = []

    for ocr_text in ocr_data.values():
        if not ocr_text or not ocr_text.strip():
            continue

        tokens = frozenset(
            re.sub(r"[^\w\s]", " ", ocr_text.lower()).split()
        )

        if len(tokens) >= 4:
            token_sets.append(tokens)

    return token_sets


def is_ocr_caption(
    block_text: str,
    ocr_token_sets: list,
    threshold: float = 0.72,
) -> bool:
    if not ocr_token_sets or not block_text.strip():
        return False

    block_tokens = frozenset(
        re.sub(r"[^\w\s]", " ", block_text.lower()).split()
    )

    word_count = len(block_text.split())

    if word_count < 4 or word_count > 22:
        return False

    for ocr_tokens in ocr_token_sets:
        intersection = len(block_tokens & ocr_tokens)
        overlap = intersection / len(block_tokens)

        if overlap >= threshold:
            return True

    return False


def select_pdf_files(pdf_root: Path, document: str) -> list[Path]:
    if not pdf_root.exists():
        print(f"[ERROR] PDF root does not exist: {pdf_root}")
        sys.exit(1)

    available_pdfs = sorted(
        file
        for file in pdf_root.iterdir()
        if file.is_file() and file.suffix.lower() == ".pdf"
    )

    if not available_pdfs:
        print(f"[ERROR] No PDF files found in {pdf_root}")
        sys.exit(1)

    if document == "all":
        return available_pdfs

    requested_name = document
    requested_path = pdf_root / requested_name

    if requested_path.is_file():
        return [requested_path]

    if not requested_name.lower().endswith(".pdf"):
        requested_path = pdf_root / f"{requested_name}.pdf"

        if requested_path.is_file():
            return [requested_path]

    available_names = ", ".join(file.name for file in available_pdfs)

    print(f"[ERROR] Requested PDF was not found: {document}")
    print(f"[INFO] Available PDFs: {available_names}")
    sys.exit(1)


def load_layout_model() -> YOLOv10:
    DOCLAYOUT_MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
    model_path = HF_MODELS_DIR / DOCLAYOUT_MODEL_FILENAME

    HF_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print("Downloading DocLayout-YOLO model from Hugging Face...")

        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename=DOCLAYOUT_MODEL_FILENAME,
            local_dir=str(HF_MODELS_DIR),
        )

        model_path = Path(downloaded_path)
        print(f"Model saved to {model_path}")

    try:
        return YOLOv10(str(model_path))

    except Exception as error:
        print(f"DocLayout-YOLO Load Error: {error}")
        sys.exit(1)


def run(args: argparse.Namespace) -> None:
    pdf_root = args.pdf_root.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    json_root = args.json_root.expanduser().resolve()

    print("=" * 70)
    print("PIPELINE 1: LAYOUT PARSING AND CROSS-MODAL EXTRACTION")
    print(f"PDF root: {pdf_root}")
    print(f"Image output root: {image_root}")
    print(f"JSON output root: {json_root}")
    print(f"Document selection: {args.document}")
    print("=" * 70)

    pdf_files = select_pdf_files(pdf_root, args.document)

    image_root.mkdir(parents=True, exist_ok=True)
    json_root.mkdir(parents=True, exist_ok=True)

    layout_model = load_layout_model()

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    for pdf_path in pdf_files:
        pdf_base = pdf_path.stem
        pdf_img_dir = image_root / pdf_base
        pdf_json_dir = json_root / pdf_base

        extracted_json = pdf_json_dir / "extracted_content.json"
        ocr_json = pdf_json_dir / "ocr_results.json"

        has_images = pdf_img_dir.is_dir() and any(
            filename.suffix.lower() in {".png", ".jpg", ".jpeg"}
            for filename in pdf_img_dir.iterdir()
            if filename.is_file()
        )

        if has_images and extracted_json.exists() and ocr_json.exists():
            print(f"-> Skipping {pdf_base} (images + JSON already present).")
            continue

        pdf_img_dir.mkdir(parents=True, exist_ok=True)
        pdf_json_dir.mkdir(parents=True, exist_ok=True)

        print(f"--- PyMuPDF4LLM Layout Pre-Pass: {pdf_base} ---")

        extract_pdf_data(
            str(pdf_path),
            str(pdf_img_dir),
            str(pdf_json_dir),
            layout_model,
        )

        run_ocr_on_folder(
            str(pdf_img_dir),
            str(ocr_json),
        )

        postprocess_extracted_with_ocr(str(pdf_json_dir))

        torch.cuda.empty_cache()
        gc.collect()

        print(f"--- Pipeline complete for {pdf_base} ---\n")


if __name__ == "__main__":
    run(parse_args())