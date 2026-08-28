import fitz
import unicodedata
import json
import os
import re
import gc
import torch
from difflib import SequenceMatcher
from PIL import Image
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doclayout_yolo import YOLOv10


# --- HPC DIRECTORY STRUCTURE ---
PROJECT_ROOT = os.path.join(os.path.expanduser("~"), "Rhine_proj")
PDF_FOLDER = os.path.join(PROJECT_ROOT, "data", "raw_pdfs")
IMAGE_FOLDER = os.path.join(PROJECT_ROOT, "data", "extracted_images")
JSON_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "json_results")


YOLO_BATCH_SIZE = 8
OCR_BATCH_SIZE = 16

REPEATER_PAGE_RATIO = 0.6
REPEATER_FUZZY_THRESHOLD = 0.88

_ABBREVIATIONS = r"(?:M|Mme|Mlle|Dr|Prof|p|fig|id|etc|av|bd|ca|cf|vol|et\s*al|ibid|p\.\s*ex|art|al|ch|sect|tab)\.?$"


def normalize_block(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[header\]', '', text)
    text = re.sub(r'\s+\d{1,4}\s*$', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def build_boilerplate_ngrams(temp_blocks_store: dict, total_pages: int) -> tuple:
    from collections import defaultdict

    def _collect(ngram_size, threshold):
        ngram_page_presence = defaultdict(set)
        for page_num, blocks in temp_blocks_store.items():
            full_page_text = " ".join(blocks)
            words = re.sub(r'\s+', ' ', full_page_text).split()
            seen = set()
            for i in range(len(words) - ngram_size + 1):
                ng = " ".join(words[i:i + ngram_size]).lower()
                seen.add(ng)
            for ng in seen:
                ngram_page_presence[ng].add(page_num)
        return {ng for ng, pages in ngram_page_presence.items()
                if len(pages) / total_pages >= threshold}

    bp6 = _collect(6, 0.35)
    bp3 = _collect(3, 0.60)
    if bp6 or bp3:
        print(f"  [FILTER-NGR] Boilerplate n-grams: {len(bp6)} (6-gram) + {len(bp3)} (3-gram)")
    return bp6, bp3


def strip_boilerplate_from_text(text: str, bp6: set, bp3: set) -> str:
    def _strip(t, ngrams, n):
        words = re.sub(r'\s+', ' ', t).split()
        if len(words) < n:
            return t
        is_bp = [False] * len(words)
        for i in range(len(words) - n + 1):
            ng = " ".join(words[i:i + n]).lower()
            if ng in ngrams:
                for j in range(i, i + n):
                    is_bp[j] = True
        return " ".join(w for i, w in enumerate(words) if not is_bp[i])

    text = _strip(text, bp6, 6)
    text = _strip(text, bp3, 3)
    text = re.sub(r'^\s*\d{1,4}\s+', '', text)
    text = re.sub(r'\s+\d{1,4}\s*$', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def is_numeric_row(text):
    words = text.split()
    if not words:
        return False
    numeric_like = sum(1 for w in words if re.match(r'^[\d.,%]+$', w))
    return numeric_like / len(words) > 0.4

def is_sentence_ender(text, next_text=None):
    if is_numeric_row(text):
        return True
    text = text.strip()

    if next_text and next_text.strip() and next_text.strip()[0].islower():
        return False

    sticky_words = ["le", "la", "les", "un", "une", "des", "du", "de", "d'", "au", "aux",
                    "et", "ou", "mais", "en", "pour", "dans", "une", "plus"]
    last_word = re.sub(r'[^\w]', '', text.split()[-1]).lower() if text.split() else ""
    if last_word in sticky_words:
        return False

    if re.search(_ABBREVIATIONS, text, re.IGNORECASE):
        return False

    if re.match(r'^[\(\[]?(?:\d+|[a-z]|[ivxlcdm]+)[\.)\]]\s', text, re.IGNORECASE):
        return True

    if text.count('«') > text.count('»'):
        return False

    if re.search(r'(?<!\.)\.\.\.(?!\.)|…$', text):
        return True

    return bool(re.search(r"[\.!\?…]['\"»\s\)]*$", text))


def is_figure_block(text):
    if not text or not text.strip():
        return False
    text = text.strip()
    keywords = r"(Figure|Fig\.|Tableau|Source|Planche|Carte|Photo|Graphique|Schéma|Encadré|Photos)s?"
    pattern = rf"^(?:\[HEADER\]\s*)?{keywords}\s*(?:\d+[\.d]*[a-z]?)?\s*[:\-\.]"
    return bool(re.match(pattern, text, re.IGNORECASE))


def is_toc_entry(text: str) -> bool:
    if not is_figure_block(text):
        return False
    if len(text) > 160:
        return False
    return bool(re.search(r'[\.\s]{2,}\s*\d{1,3}\s*$', text))


def clean_text_content(text):
    text = re.sub(r'^\s*\[HEADER\]\s*', '', text)
    text = re.sub(r'\s+([:;!\?])', r' \1', text)
    text = re.sub(r'\.{4,}', ' ', text)
    return " ".join(text.split())


def rasterize_pages(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    doc.close()
    return pages


def detect_repeating_blocks(temp_blocks_store: dict, total_pages: int) -> set:
    from collections import defaultdict
    norm_to_pages = defaultdict(set)

    for page_num, blocks in temp_blocks_store.items():
        for block in blocks:
            if not block:
                continue
            norm = normalize_block(block)
            if len(norm) < 10:
                continue
            norm_to_pages[norm].add(page_num)

    threshold_pages = max(2, int(total_pages * REPEATER_PAGE_RATIO))
    exact_repeaters = {norm for norm, pages in norm_to_pages.items() if len(pages) >= threshold_pages}

    from collections import defaultdict as dd
    buckets = dd(list)
    for norm in norm_to_pages.keys():
        if norm in exact_repeaters:
            continue
        buckets[norm[:20]].append(norm)

    fuzzy_repeaters = set()
    for bucket_key, norms in buckets.items():
        if len(norms) < 2:
            continue
        for i in range(len(norms)):
            for j in range(i + 1, len(norms)):
                sim = SequenceMatcher(None, norms[i], norms[j]).ratio()
                if sim >= REPEATER_FUZZY_THRESHOLD:
                    combined_pages = norm_to_pages[norms[i]] | norm_to_pages[norms[j]]
                    if len(combined_pages) >= threshold_pages:
                        fuzzy_repeaters.add(norms[i])
                        fuzzy_repeaters.add(norms[j])

    all_repeaters = exact_repeaters | fuzzy_repeaters
    if all_repeaters:
        print(f"  [FILTER] Cross-page deduplication: {len(all_repeaters)} repeating block pattern(s) flagged as structural noise.")
    return all_repeaters


def is_structural_repeater(text: str, repeater_norms: set) -> bool:
    norm = normalize_block(text)
    if norm in repeater_norms:
        return True
    if len(norm) < 15:
        return False
    for rep in repeater_norms:
        if abs(len(norm) - len(rep)) > 40:
            continue
        if SequenceMatcher(None, norm, rep).ratio() >= REPEATER_FUZZY_THRESHOLD:
            return True
    return False


def _pick_yolo_device():
    """Pick the safest available device for DocLayout-YOLO.
    Returns 'cuda:0' only when a lightweight smoke-test succeeds.
    Falls back to 'cpu' on any failure so the pipeline never crashes here.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        cc = torch.cuda.get_device_capability(0)
        if cc[0] < 7:
            print(f"  [YOLO-DEV] GPU compute capability {cc} < 7.0 — using CPU.")
            return "cpu"
        dummy = torch.zeros(1, 3, 64, 64, device="cuda:0")
        torch.nn.functional.avg_pool2d(dummy, 2)
        del dummy
        torch.cuda.empty_cache()
        return "cuda:0"
    except Exception as e:
        print(f"  [YOLO-DEV] CUDA smoke-test failed ({e}) — falling back to CPU.")
        return "cpu"


def extract_pdf_data(pdf_path, img_out_dir, json_out_dir, layout_model):
    print(f"\n--- Final Extraction Pipeline: {os.path.basename(pdf_path)} ---")

    for f in os.listdir(img_out_dir):
        if f.lower().endswith(('.jpeg', '.jpg', '.png')):
            os.remove(os.path.join(img_out_dir, f))

    doc = fitz.open(pdf_path)
    extracted_data = {"metadata": doc.metadata, "total_pages": len(doc), "pages": {}}
    temp_blocks_store = {}
    temp_blocks_with_bbox = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        width, height = page.rect.width, page.rect.height
        margin_y = height * 0.025

        page_dict = page.get_text("dict")
        raw_blocks = page_dict["blocks"]

        styles = {}
        line_heights = []
        for b in raw_blocks:
            if b["type"] != 0: continue
            for line in b["lines"]:
                line_heights.append(line["bbox"][3] - line["bbox"][1])
                for span in line["spans"]:
                    font_base = span["font"].split("+")[-1]
                    s_key = (round(span["size"] * 2) / 2, font_base)
                    styles[s_key] = styles.get(s_key, 0) + len(span["text"])

        body_style = max(styles, key=styles.get) if styles else (None, None)
        median_lh = sorted(line_heights)[len(line_heights)//2] if line_heights else 15
        dynamic_gap_threshold = median_lh * 1.4

        all_lines = []
        for b in raw_blocks:
            if b["type"] != 0: continue
            for line in b["lines"]:
                ly0, ly1 = line["bbox"][1], line["bbox"][3]
                if ly0 < margin_y or ly1 > (height - margin_y): continue

                spans = line["spans"]
                last_span_text = spans[-1]["text"] if spans else ""
                has_line_break_hyphen = last_span_text.rstrip().endswith("-")

                line_text = "".join([span["text"] for span in spans]).strip()
                if not line_text: continue

                main_span = max(spans, key=lambda s: len(s["text"]))
                font_base = main_span["font"].split("+")[-1]
                line_style = (round(main_span["size"] * 2) / 2, font_base)

                all_lines.append({
                    "text": line_text,
                    "bbox": line["bbox"],
                    "style": line_style,
                    "line_break_hyphen": has_line_break_hyphen
                })

        all_lines.sort(key=lambda l: (round(l["bbox"][1] / 5) * 5, l["bbox"][0]))

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

        for i, line in enumerate(all_lines):
            lx0, ly0, lx1, ly1 = line["bbox"]
            text = line["text"]
            curr_style = line["style"]

            sub_parts = re.split(r'(?i)(?=(?:Figure|Fig\.|Tableau|Source|Planche|Carte)\s*(?:\d+)?\s*[:\-\.])', text)
            sub_parts = [p.strip() for p in sub_parts if p.strip()]

            for part in sub_parts:
                is_splitter = is_figure_block(part)

                is_vertically_close = (last_y1 != -1 and (ly0 - last_y1) < dynamic_gap_threshold)
                is_aligned = (last_x0 != -1 and abs(lx0 - last_x0) < 5)
                is_style_match = (last_style == curr_style)
                is_unfinished = current_block_text and not is_sentence_ender(current_block_text, part)

                next_is_fresh_start = False
                if i + 1 < len(all_lines):
                    next_l = all_lines[i+1]
                    next_gap = next_l["bbox"][1] - ly1
                    if next_gap > (median_lh * 0.9) and next_l["text"] and next_l["text"][0].isupper():
                        if abs(next_l["bbox"][0] - lx0) > 10 or next_gap > median_lh:
                            next_is_fresh_start = True

                is_header_style = curr_style[0] > body_style[0] * 1.25 if body_style[0] else False

                if is_splitter:
                    force_break = True
                elif is_unfinished:
                    force_break = False
                else:
                    force_break = is_header_style or not is_aligned or not is_style_match or next_is_fresh_start or not is_vertically_close

                if force_break:
                    flush_block()
                    current_block_text = f"[HEADER] {part}" if is_header_style else part
                    current_block_bbox = list(line["bbox"])
                    last_x0 = lx0
                else:
                    if line.get("line_break_hyphen") and current_block_text.endswith("-"):
                        current_block_text = current_block_text[:-1] + part
                    else:
                        current_block_text += " " + part
                    if current_block_bbox is None:
                        current_block_bbox = list(line["bbox"])
                    else:
                        current_block_bbox[0] = min(current_block_bbox[0], lx0)
                        current_block_bbox[1] = min(current_block_bbox[1], ly0)
                        current_block_bbox[2] = max(current_block_bbox[2], lx1)
                        current_block_bbox[3] = max(current_block_bbox[3], ly1)

                last_y1 = ly1
                last_style = curr_style

        flush_block()
        temp_blocks_store[page_num] = processed_page_blocks
        temp_blocks_with_bbox[page_num] = list(zip(processed_page_blocks, processed_page_blocks_bbox))

    # --- CROSS-PAGE STITCHING ---
    for page_num in range(len(doc)):
        current_blocks = temp_blocks_store[page_num]
        if page_num < len(doc) - 1:
            next_blocks = temp_blocks_store[page_num + 1]
            if current_blocks and next_blocks:
                last_idx = len(current_blocks) - 1
                if not is_figure_block(current_blocks[last_idx]) and not is_sentence_ender(current_blocks[last_idx], next_blocks[0] if next_blocks else None):
                    next_text_idx = -1
                    for idx in range(len(next_blocks)):
                        # FIX: Skip slots already consumed by stitching, and skip figure-only blocks
                        if next_blocks[idx] and not is_figure_block(next_blocks[idx]):
                            next_text_idx = idx
                            break
                    if next_text_idx != -1:
                        first_text = next_blocks[next_text_idx]
                        match = re.search(r'([.!?])(?:\s|$)', first_text)
                        if match:
                            split_pos = match.end()
                            current_blocks[last_idx] += " " + first_text[:split_pos]
                            remainder = first_text[split_pos:].strip()
                            if remainder:
                                next_blocks[next_text_idx] = remainder
                            else:
                                # Blank the consumed slot rather than removing it, to preserve page-block index alignment
                                next_blocks[next_text_idx] = ""
                        else:
                            current_blocks[last_idx] += " " + first_text
                            next_blocks[next_text_idx] = ""

    doc.close()

    existing_title = (extracted_data["metadata"].get("title") or "").strip()
    if not existing_title:
        first_page_blocks = temp_blocks_store.get(0, [])
        for block in first_page_blocks:
            if not block:
                continue
            candidate = re.sub(r'^\s*\[HEADER\]\s*', '', block).strip()
            candidate = clean_text_content(candidate)
            if len(candidate) > 8:
                extracted_data["metadata"]["title"] = candidate
                print(f"  [TITLE-FALLBACK] Derived title from page 1: {candidate[:70]}...")
                break

    # --- CROSS-PAGE REPEATER DETECTION ---
    repeater_norms = detect_repeating_blocks(temp_blocks_store, len(temp_blocks_store))
    bp6, bp3 = build_boilerplate_ngrams(temp_blocks_store, len(temp_blocks_store))

    # --- RASTERIZE + YOLO ---
    print("Rasterizing pages at 300 DPI...")
    page_images = rasterize_pages(pdf_path, dpi=300)
    dpi = 300
    zoom = dpi / 72.0

    yolo_device = _pick_yolo_device()
    print(f"Running DocLayout-YOLO (batched) on {yolo_device}...")
    all_det_results = []
    for batch_start in range(0, len(page_images), YOLO_BATCH_SIZE):
        batch = page_images[batch_start:batch_start + YOLO_BATCH_SIZE]
        det_results = layout_model.predict(
            batch,
            imgsz=1024,
            conf=0.2,
            device=yolo_device,
            verbose=False
        )
        all_det_results.extend(det_results)

    abandon_regions_pdf = {}
    caption_regions_pdf = {}

    for page_num, (page_img, det_res) in enumerate(zip(page_images, all_det_results)):
        figure_boxes = []
        caption_boxes = []
        abandon_boxes_pixel = []

        for box in det_res.boxes:
            class_id = int(box.cls[0])
            class_name = layout_model.names[class_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if class_name in ["figure", "picture", "image"]:
                figure_boxes.append([x1, y1, x2, y2])
            elif class_name in ["figure_caption", "figurecaption", "caption"]:
                caption_boxes.append([x1, y1, x2, y2])
            elif class_name == "abandon":
                abandon_boxes_pixel.append([x1, y1, x2, y2])

        if abandon_boxes_pixel:
            abandon_regions_pdf[page_num] = [
                [x1/zoom, y1/zoom, x2/zoom, y2/zoom]
                for x1, y1, x2, y2 in abandon_boxes_pixel
            ]

        if caption_boxes:
            caption_regions_pdf[page_num] = [
                [x1/zoom, y1/zoom, x2/zoom, y2/zoom]
                for x1, y1, x2, y2 in caption_boxes
            ]

        if not figure_boxes:
            continue

        page_w, page_h = page_img.size
        H_MERGE_MARGIN = page_w * 0.08
        V_MERGE_MARGIN = page_h * 0.02

        changed = True
        while changed:
            changed = False
            merged_boxes = []
            used = set()
            for i, b1 in enumerate(figure_boxes):
                if i in used:
                    continue
                for j, b2 in enumerate(figure_boxes):
                    if j <= i or j in used:
                        continue
                    h_dist = max(0, b2[0] - b1[2], b1[0] - b2[2])
                    v_dist = max(0, b2[1] - b1[3], b1[1] - b2[3])
                    if h_dist < H_MERGE_MARGIN and v_dist < V_MERGE_MARGIN:
                        b1 = [min(b1[0], b2[0]), min(b1[1], b2[1]),
                              max(b1[2], b2[2]), max(b1[3], b2[3])]
                        used.add(j)
                        changed = True
                merged_boxes.append(b1)
                used.add(i)
            figure_boxes = merged_boxes

        for c_box in caption_boxes:
            best_fig = None
            best_dist = float("inf")
            for f_box in figure_boxes:
                h_overlap = max(0, min(f_box[2], c_box[2]) - max(f_box[0], c_box[0]))
                v_dist = c_box[1] - f_box[3]
                if h_overlap > 0 and 0 <= v_dist < page_h * 0.08:
                    if v_dist < best_dist:
                        best_dist = v_dist
                        best_fig = f_box
            if best_fig is not None:
                best_fig[0] = min(best_fig[0], c_box[0])
                best_fig[2] = max(best_fig[2], c_box[2])
                best_fig[3] = max(best_fig[3], c_box[3])

        img_index = 1
        for f_box in figure_boxes:
            crop_rect = (
                max(0, f_box[0] - 30),
                max(0, f_box[1] - 30),
                min(page_img.width, f_box[2] + 30),
                min(page_img.height, f_box[3] + 30)
            )
            page_img.crop(crop_rect).save(
                os.path.join(img_out_dir, f"img_p{page_num + 1}_{img_index}.png"),
                "PNG"
            )
            img_index += 1

    # --- FINAL BLOCK ASSEMBLY WITH ALL FILTERS ---
    def bbox_overlap_ratio(b1, b2):
        ix0 = max(b1[0], b2[0])
        iy0 = max(b1[1], b2[1])
        ix1 = min(b1[2], b2[2])
        iy1 = min(b1[3], b2[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        intersection = (ix1 - ix0) * (iy1 - iy0)
        b1_area = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
        return intersection / b1_area

    for page_num in range(len(temp_blocks_with_bbox)):
        blocks_with_bbox = temp_blocks_with_bbox.get(page_num, [])
        stitched_texts = temp_blocks_store[page_num]

        abandon_regions = abandon_regions_pdf.get(page_num, [])
        caption_regions = caption_regions_pdf.get(page_num, [])

        block_counter = 0
        for i, raw_text in enumerate(stitched_texts):
            # Skip blocks emptied during cross-page stitching
            if not raw_text:
                continue

            stripped = strip_boilerplate_from_text(raw_text, bp6, bp3)
            clean = clean_text_content(stripped)

            if is_toc_entry(clean):
                print(f"  [FILTER-TOC] Dropped TOC entry on page {page_num+1}: {clean[:70]}...")
                continue

            if is_structural_repeater(clean, repeater_norms):
                print(f"  [FILTER-RPT] Dropped repeating block on page {page_num+1}: {clean[:70]}...")
                continue

            block_bbox = None
            if i < len(blocks_with_bbox):
                _, block_bbox = blocks_with_bbox[i]

            if block_bbox is not None:
                dropped_abandon = False
                for ab in abandon_regions:
                    if bbox_overlap_ratio(block_bbox, ab) > 0.5:
                        print(f"  [FILTER-ABN] Dropped abandon-region block on page {page_num+1}: {clean[:70]}...")
                        dropped_abandon = True
                        break
                if dropped_abandon:
                    continue

                dropped_caption = False
                for cb in caption_regions:
                    if bbox_overlap_ratio(block_bbox, cb) > 0.5:
                        print(f"  [FILTER-CAP] Dropped figure caption block on page {page_num+1}: {clean[:70]}...")
                        dropped_caption = True
                        break
                if dropped_caption:
                    continue

            if len(clean) > 40:
                block_counter += 1
                block_key = f"page_{page_num + 1}_block_{block_counter}"
                extracted_data["pages"][block_key] = clean
                print(f"  [LOG] Finalized {block_key}: {clean[:60]}...")

    with open(os.path.join(json_out_dir, "extracted_content.json"), "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)


def postprocess_extracted_with_ocr(json_out_dir: str):
    extracted_path = os.path.join(json_out_dir, "extracted_content.json")
    ocr_path = os.path.join(json_out_dir, "ocr_results.json")

    if not os.path.exists(extracted_path) or not os.path.exists(ocr_path):
        print("  [OCR-DEDUP] Skipping post-process: missing files.")
        return

    with open(extracted_path, "r", encoding="utf-8") as f:
        extracted = json.load(f)

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
            print(f"  [OCR-DEDUP] Dropped caption block '{block_key}': {block_text[:70]}...")
            dropped += 1
        else:
            clean_pages[block_key] = block_text

    renumbered = {}
    page_counters = {}
    for old_key, text in clean_pages.items():
        parts = old_key.rsplit("_block_", 1)
        page_prefix = parts[0]
        page_counters[page_prefix] = page_counters.get(page_prefix, 0) + 1
        new_key = f"{page_prefix}_block_{page_counters[page_prefix]}"
        renumbered[new_key] = text

    extracted["pages"] = renumbered
    with open(extracted_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=4)

    print(f"  [OCR-DEDUP] Removed {dropped}/{original_count} blocks matching OCR image content.")


def run_ocr_on_folder(image_folder, output_json):
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"--- Running OCR on {device} ---")
    model = ocr_predictor(pretrained=True).to(device)

    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    ocr_results = {}

    for batch_start in range(0, len(image_files), OCR_BATCH_SIZE):
        batch_files = image_files[batch_start:batch_start + OCR_BATCH_SIZE]
        batch_paths = [os.path.join(image_folder, f) for f in batch_files]
        print(f"  [OCR] Batch {batch_start // OCR_BATCH_SIZE + 1} / {-(-len(image_files) // OCR_BATCH_SIZE)} ({len(batch_files)} images)")
        try:
            doc = DocumentFile.from_images(batch_paths)
            result = model(doc)
            for j, page in enumerate(result.pages):
                text = " ".join([
                    word.value
                    for block in page.blocks
                    for line in block.lines
                    for word in line.words
                ])
                ocr_results[batch_files[j]] = text
        except Exception as e:
            print(f"  [OCR BATCH ERROR] {e} — marking batch as empty.")
            for f in batch_files:
                ocr_results[f] = ""

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=4)


def load_ocr_texts(ocr_json_path: str) -> list:
    if not os.path.exists(ocr_json_path):
        return []
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    token_sets = []
    for img_name, ocr_text in ocr_data.items():
        if not ocr_text or not ocr_text.strip():
            continue
        tokens = frozenset(re.sub(r'[^\w\s]', ' ', ocr_text.lower()).split())
        if len(tokens) >= 4:
            token_sets.append(tokens)
    return token_sets


def is_ocr_caption(block_text: str, ocr_token_sets: list, threshold: float = 0.72) -> bool:
    if not ocr_token_sets or not block_text.strip():
        return False
    block_tokens = frozenset(re.sub(r'[^\w\s]', ' ', block_text.lower()).split())
    word_count = len(block_text.split())
    if word_count < 4 or word_count > 22:
        return False
    for ocr_tokens in ocr_token_sets:
        intersection = len(block_tokens & ocr_tokens)
        overlap = intersection / len(block_tokens)
        if overlap >= threshold:
            return True
    return False


if __name__ == "__main__":
    os.makedirs(PDF_FOLDER, exist_ok=True)
    HF_MODEL_DIR = os.path.join(os.path.expanduser("~"), "Rhine_proj", "hf_models")
    os.makedirs(HF_MODEL_DIR, exist_ok=True)
    DOCLAYOUT_MODEL_PATH = os.path.join(HF_MODEL_DIR, "doclayout_yolo_docstructbench_imgsz1024.pt")

    if not os.path.exists(DOCLAYOUT_MODEL_PATH):
        print("Downloading DocLayout-YOLO model from HuggingFace...")
        from huggingface_hub import hf_hub_download
        DOCLAYOUT_MODEL_PATH = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            local_dir=HF_MODEL_DIR
        )
        print(f"Model saved to {DOCLAYOUT_MODEL_PATH}")

    try:
        layout_model = YOLOv10(DOCLAYOUT_MODEL_PATH)
    except Exception as e:
        print(f"DocLayout-YOLO Load Error: {e}"); exit()

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs.")

    for pdf_filename in pdf_files:
        pdf_base = os.path.splitext(pdf_filename)[0]
        pdf_img_dir = os.path.join(IMAGE_FOLDER, pdf_base)
        pdf_json_dir = os.path.join(JSON_OUTPUT_FOLDER, pdf_base)

        extracted_json = os.path.join(pdf_json_dir, "extracted_content.json")
        ocr_json = os.path.join(pdf_json_dir, "ocr_results.json")

        has_images = os.path.isdir(pdf_img_dir) and any(
            fn.lower().endswith((".png", ".jpg", ".jpeg"))
            for fn in os.listdir(pdf_img_dir)
        )

        if has_images and os.path.exists(extracted_json) and os.path.exists(ocr_json):
            print(f"-> Skipping {pdf_base} (images + JSON already present).")
            continue

        os.makedirs(pdf_img_dir, exist_ok=True)
        os.makedirs(pdf_json_dir, exist_ok=True)

        pdf_path = os.path.join(PDF_FOLDER, pdf_filename)

        print(f"--- PyMuPDF4LLM Layout Pre-Pass: {pdf_base} ---")

        extract_pdf_data(pdf_path, pdf_img_dir, pdf_json_dir, layout_model)
        run_ocr_on_folder(pdf_img_dir, os.path.join(pdf_json_dir, "ocr_results.json"))
        postprocess_extracted_with_ocr(pdf_json_dir)

        torch.cuda.empty_cache()
        gc.collect()

        print(f"--- Pipeline complete for {pdf_base} ---\n")
