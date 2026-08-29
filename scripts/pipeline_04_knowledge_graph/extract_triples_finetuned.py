import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ─── Repository and runtime configuration ────────────────────────────────────

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("RHINE_PROJECT_ROOT", str(SCRIPT_ROOT))
).expanduser().resolve()

HF_MODELS_DIR = Path(
    os.environ.get("HF_MODELS_DIR", str(PROJECT_ROOT / "hf_models"))
).expanduser().resolve()

HF_TOKEN = os.environ.get("HF_TOKEN")

DEFAULT_CAPTIONS_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "5th_ablation_Image_captioning_prompt_engineering"
    / "nvila_p3_contrast_captions"
)

DEFAULT_TEXT_ROOT = PROJECT_ROOT / "outputs" / "json_results"

DEFAULT_TTL_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "knowledge_graph"
    / "fine_tuned"
)

ONTOLOGY_TTL_PATH = PROJECT_ROOT / "data" / "ontology" / "TetraOnto_compact.txt"
FULL_ONTOLOGY_PATH = PROJECT_ROOT / "data" / "ontology" / "TetraOnto.ttl"

MODEL_ID = HF_MODELS_DIR / "Qwen2.5-14B-populate-ontology-merged"


TETRA_NS = "http://www.semanticweb.org/fethi/ontologies/2024/1/TetraOnto.owl#"
TETRA    = Namespace(TETRA_NS)


PREFIX_HEADER = f"""@prefix : <{TETRA_NS}> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fine-tuned Pipeline 4 knowledge-graph extraction on one "
            "caption document folder or all available caption folders."
        )
    )

    parser.add_argument(
        "--document",
        default="all",
        help=(
            "Name of one document folder under --captions-root, or 'all' to "
            "process every available document. Default: all."
        ),
    )

    parser.add_argument(
        "--captions-root",
        type=Path,
        default=DEFAULT_CAPTIONS_ROOT,
        help=(
            "Directory containing per-document A5-P3 caption folders. "
            f"Default: {DEFAULT_CAPTIONS_ROOT}"
        ),
    )

    parser.add_argument(
        "--text-root",
        type=Path,
        default=DEFAULT_TEXT_ROOT,
        help=(
            "Directory containing per-document extracted_content.json files. "
            f"Default: {DEFAULT_TEXT_ROOT}"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_TTL_OUTPUT_ROOT,
        help=(
            "Directory for newly generated fine-tuned RDF/Turtle graph "
            f"outputs. Default: {DEFAULT_TTL_OUTPUT_ROOT}"
        ),
    )

    return parser.parse_args()

# --- LOAD FULL ONTOLOGY ---
def load_full_ontology(ttl_path: str) -> str:
    try:
        with open(ttl_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"[ONTOLOGY] Loaded compact summary: {len(content)} chars, ~{len(content)//4} tokens", flush=True)
        return content
    except Exception as e:
        print(f"[ONTOLOGY ERROR] {e}", flush=True)
        return ""


FULL_ONTOLOGY_TTL = load_full_ontology(ONTOLOGY_TTL_PATH)

CHUNK_WORDS   = 2500
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP) -> list:
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

def _trailing_number(s):
    m = re.search(r'(\d+)\s*$', s.strip())
    return m.group(1) if m else None

def _merge_json_arrays(json_a: str, json_b: str) -> str:
    def _extract(s):
        s = s.strip()
        start, end = s.find("["), s.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return []
    combined = _extract(json_a) + _extract(json_b)
    seen, deduped = set(), []
    for t in combined:
        key = (str(t.get("subject","")).strip().lower(),
               str(t.get("predicate","")).strip().lower(),
               str(t.get("object","")).strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(t)
    return json.dumps(deduped, ensure_ascii=False)


def _extract_subjects_for_registry(json_str: str) -> list:
    s = json_str.strip()
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        triples = json.loads(s[start:end + 1])
        return [str(t.get("subject","")).strip() for t in triples if isinstance(t, dict) and t.get("subject")]
    except Exception:
        return []


def process_chunked_content(text: str, source_type: str, page_num: str, entity_registry: dict,
                             label: str, tokenizer, model) -> str:
    chunks = chunk_text(text)
    if len(chunks) == 1:
        return call_deepseek(chunks[0], source_type, page_num, entity_registry, label, tokenizer, model)

    print(f"  [CHUNK] {label}: {len(chunks)} chunks (proactive, overlap={CHUNK_OVERLAP}w)", flush=True)
    merged_json = "[]"
    for i, chunk in enumerate(chunks):
        chunk_label = f"{label}_chunk{i+1}of{len(chunks)}"
        raw = call_deepseek(chunk, source_type, page_num, entity_registry, chunk_label, tokenizer, model)
        merged_json = _merge_json_arrays(merged_json, raw)
        for subj in _extract_subjects_for_registry(raw):
            entity_registry[subj] = None
    return merged_json


# --- PREDICATE WHITELIST ---
def load_valid_predicates(compact_txt_path: str) -> set:
    valid = set()
    if not os.path.exists(compact_txt_path):
        print(f"[WARN] Compact ontology not found: {compact_txt_path}")
        return valid
    with open(compact_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^:(\w+)", line.strip())
            if m:
                valid.add(m.group(1))
    print(f"[ONTOLOGY] {len(valid)} valid predicates loaded for whitelisting.", flush=True)
    return valid

def load_valid_classes(compact_txt_path: str) -> set:
    valid = set()
    with open(compact_txt_path, "r", encoding="utf-8") as f:
        in_classes = False
        for line in f:
            if line.strip() == "=== CLASSES ===":
                in_classes = True; continue
            if line.strip().startswith("=== "):
                in_classes = False; continue
            if in_classes:
                m = re.match(r"^:(\w[\w-]*)", line.strip())
                if m:
                    valid.add(m.group(1))
    print(f"[ONTOLOGY] {len(valid)} valid classes loaded for type whitelisting.", flush=True)
    return valid

VALID_CLASSES = load_valid_classes(ONTOLOGY_TTL_PATH)

def load_altlabel_to_class_map(ontology_full_ttl_path: str) -> list:
    ontg = Graph()
    ontg.parse(ontology_full_ttl_path, format="turtle")
    mapping = {}
    for s, p, o in ontg.triples((None, SKOS.altLabel, None)):
        mapping[str(o).strip().lower()] = str(s).split("#")[-1]
    sorted_mapping = sorted(mapping.items(), key=lambda x: -len(x[0]))
    print(f"[ONTOLOGY] {len(sorted_mapping)} altLabel-to-class fallback mappings loaded.", flush=True)
    return sorted_mapping


ALTLABEL_TO_CLASS = load_altlabel_to_class_map(str(FULL_ONTOLOGY_PATH))

VALID_PREDICATES = load_valid_predicates(ONTOLOGY_TTL_PATH)
_demoted_predicates_count = 0  # global counter for thesis stats


def make_base_graph() -> Graph:
    g = Graph()
    g.bind("",     TETRA)
    g.bind("skos", SKOS)
    g.bind("xsd",  XSD)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)
    g.bind("rdf",  RDF)
    ontology_uri = URIRef(TETRA_NS.rstrip("#/") + "/instance_graph")
    g.add((ontology_uri, RDF.type, OWL.Ontology))
    return g


def format_entity_registry(registry: dict, max_entities: int = 40) -> str:
    """Formats the current running entity registry for prompt injection.
    Caps at max_entities most-recently-added to control token growth."""
    if not registry:
        return ""
    entities = list(registry.keys())[-max_entities:]
    return ", ".join(entities)

def build_prompt(text_input: str, source_type: str, page_num: str, entity_registry: set = None) -> str:
    registry_instruction = ""
    registry_str = format_entity_registry(entity_registry) if entity_registry else ""
    if registry_str:
        registry_instruction = f"""
ENTITÉS DÉJÀ IDENTIFIÉES DANS CE DOCUMENT (registre courant) :
{registry_str}

RÈGLE DE RÉSOLUTION D'ENTITÉ (IMPORTANT) :
Si le texte fait référence de façon générique ou implicite à une entité déjà listée ci-dessus
(ex: "le cours d'eau", "cet ouvrage", "ce site", "la structure", pronoms, etc.),
utilise le nom EXACT de cette entité tel qu'il apparaît dans le registre, au lieu du terme générique ou du pronom.
Ne crée PAS une nouvelle entité si elle correspond déjà à une entité du registre.
"""

    return f"""Tu es un expert en extraction de connaissances pour un graphe de connaissances scientifique sur l'hydromorphologie fluviale et la restauration de cours d'eau.


ONTOLOGIE DISPONIBLE (utilise UNIQUEMENT ces classes et propriétés) :
{FULL_ONTOLOGY_TTL}
{registry_instruction}

CONTENU À ANALYSER :
Source: {source_type} | Page: {page_num}
{text_input}


MISSION : Extrais les faits de ce texte sous forme de triplets JSON.


RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec un tableau JSON valide. Aucun texte avant ou après.
2. Chaque triplet a exactement 4 clés : "subject", "subject_type", "predicate", "object"
3. "subject" : nom court en français de l'entité principale (ex: "Ruisseau de Norroy")
3b. "subject_type" : la classe EXACTE de l'ontologie ci-dessus qui correspond au sujet (ex: "waterBody", "river", "structure"). Si aucune classe ne correspond, écris "unknown".
4. "predicate" : EXACTEMENT un terme de l'ontologie ci-dessus (ex: "isLocatedOn", "hasPageNumber", "dataSource")
5. "object" : valeur ou nom court de l'entité liée (ex: "Tronçon 1", "restauration", "{page_num}")
6. Maximum 15 triplets par bloc.
7. Commence DIRECTEMENT par [ sans aucune introduction.
8. Si aucun triplet pertinent n'existe dans ce texte, réponds UNIQUEMENT avec: []


Exemple de sortie valide :
[
  {{"subject": "Ruisseau de Norroy", "subject_type": "waterBody", "predicate": "isLocatedOn", "object": "Tronçon 1"}},
  {{"subject": "Ruisseau de Norroy", "subject_type": "waterBody", "predicate": "hasPageNumber", "object": "{page_num}"}},
  {{"subject": "Ruisseau de Norroy", "subject_type": "waterBody", "predicate": "dataSource", "object": "{source_type}"}}
]"""





print(f"[GPU CHECK] Available: {torch.cuda.device_count()}", flush=True)
print(f"[GPU CHECK] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)

def pick_free_gpu(min_free_gb: float = 10.0) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this pipeline inside a GPU Slurm job.")

    visible_count = torch.cuda.device_count()
    if visible_count < 1:
        raise RuntimeError("No CUDA GPU is visible to this job.")

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    print(
        f"[GPU CHECK] CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}",
        flush=True,
    )
    print(
        f"[GPU CHECK] Using Slurm-visible CUDA device 0: "
        f"{free_gb:.2f} GiB free / {total_gb:.2f} GiB total.",
        flush=True,
    )

    if free_gb < min_free_gb:
        print(
            f"[WARN] Visible GPU has only {free_gb:.2f} GiB free; "
            "the model may run out of memory.",
            flush=True,
        )

    return 0

def load_deepseek_model():
    print("[MODEL] Loading Qwen2.5-14B-populate-ontology (fine-tuned, merged) 4bit...", flush=True)
    device_id = pick_free_gpu()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": device_id},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    model.eval()
    print(f"[MODEL] Fine-tuned merged model loaded and ready on GPU {device_id}.", flush=True)
    return tokenizer, model


def call_deepseek(
    text_input: str,
    source_type: str,
    page_num: str,
    entity_registry,
    source_label: str,
    tokenizer,
    model,
) -> str:
    prompt = build_prompt(text_input, source_type, page_num, entity_registry)
    print(f" Processing: {source_label}", flush=True)
    try:
        start = time.time()
        messages = [
            {"role": "system", "content": "Translate the user text into an TTL graph based on the TetraOnto ontology."},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=900, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = outputs[0][inputs.input_ids.shape[1]:]
        result = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"  Done in {time.time()-start:.1f}s ({len(result)} chars)", flush=True)
        del inputs, outputs, generated
        torch.cuda.empty_cache()
        return result
    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM-RETRY] {source_label}: waiting, retrying same unmodified content once.", flush=True)
        torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(5)
        try:
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=900, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs.input_ids.shape[1]:]
            result = tokenizer.decode(generated, skip_special_tokens=True).strip()
            del inputs, outputs, generated
            torch.cuda.empty_cache()
            print(f"  [OOM-RETRY OK] {source_label}", flush=True)
            return result
        except Exception as e2:
            print(
                f"[OOM-RETRY FAIL] {source_label}: {e2}. "
                "No triples were produced for this item.",
                flush=True,
            )
            return ""
    except Exception as e:
        print(f"  [ERROR] Failed for {source_label}: {e}", flush=True)
        return ""


def call_deepseek_raw_prompt(prompt: str, source_label: str, tokenizer, model) -> str:
    print(f" Processing: {source_label}", flush=True)
    try:
        start = time.time()
        messages = [
            {"role": "system", "content": "Translate the user text into an TTL graph based on the TetraOnto ontology."},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=900, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = outputs[0][inputs.input_ids.shape[1]:]
        result = tokenizer.decode(generated, skip_special_tokens=True).strip()
        print(f"  Done in {time.time()-start:.1f}s ({len(result)} chars)", flush=True)
        del inputs, outputs, generated
        torch.cuda.empty_cache()
        return result
    except torch.cuda.OutOfMemoryError:
        print(f"  [OOM-RETRY] {source_label}: waiting, retrying once.", flush=True)
        torch.cuda.empty_cache(); torch.cuda.synchronize(); time.sleep(5)
        try:
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=900, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = outputs[0][inputs.input_ids.shape[1]:]
            result = tokenizer.decode(generated, skip_special_tokens=True).strip()
            del inputs, outputs, generated
            torch.cuda.empty_cache()
            return result
        except Exception as e2:
            print(f"  [OOM-RETRY FAIL] {source_label}: {e2}", flush=True)
            return ""
    except Exception as e:
        print(f"  [ERROR] Failed for {source_label}: {e}", flush=True)
        return ""


def call_deepseek_batch(items: list, tokenizer, model) -> list:
    if not items:
        return []

    prompts = [build_prompt(text_input, source_type, page_num, entity_registry)
               for (text_input, source_type, page_num, entity_registry, label) in items]

    all_texts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "Translate the user text into an TTL graph based on the TetraOnto ontology."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_texts.append(text)

    inputs = tokenizer(
        all_texts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=900,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        results = []
        for i in range(len(items)):
            total_len  = inputs["input_ids"].shape[1]
            generated  = outputs[i][total_len:]
            decoded    = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append(decoded)
            print(f"  [OK] {items[i][4]} ({len(decoded)} chars)", flush=True)

        del inputs, outputs
        torch.cuda.empty_cache()
        return results

    except torch.cuda.OutOfMemoryError:
        print("  [OOM on batch] Falling back to sequential per-item processing...", flush=True)
        torch.cuda.empty_cache()
        results = []
        for (text_input, source_type, page_num, entity_registry, label) in items:
            results.append(call_deepseek(text_input, source_type, page_num, entity_registry, label, tokenizer, model))
        return results

    except torch.cuda.OutOfMemoryError:
        print("  [OOM on batch] Falling back to sequential...", flush=True)
        torch.cuda.empty_cache()
        results = []
        for prompt, label in items:
            results.append(call_deepseek(prompt, label, tokenizer, model))
        return results


def normalize_uri_fragment(label: str) -> str:
    """Converts any label to a clean, safe URI fragment. No brackets, no special chars."""
    label = unicodedata.normalize("NFD", label)
    label = "".join(c for c in label if unicodedata.category(c) != "Mn")
    label = re.sub(r"[^\w\s]", "", label)
    label = label.strip().title().replace(" ", "_")
    label = re.sub(r"_+", "_", label).strip("_")
    return label[:80] if label else "UnknownEntity"


def clean_literal(value: str) -> str:
    """
    Sanitizes any string before it becomes an rdflib Literal.
    """
    if not isinstance(value, str):
        value = str(value)
    value = unicodedata.normalize("NFC", value)
    value = value.replace("'", "\u2019")
    value = value.replace("\u2018", "\u2019")
    value = value.replace("\u0060", "\u2019")
    value = value.replace("\\", "")
    value = value.replace("\u20ac", "EUR")
    value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    value = value.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    value = re.sub(r" +", " ", value).strip()
    return value


def json_to_rdf_graph(json_str, source_type, page_num, uri_prefix, source_label, g: Graph, document_title: str = None) -> int:
    """Parses LLM JSON triples, validates, converts to RDF with clean entity URIs and provenance attributes."""

    json_str = json_str.strip()
    start = json_str.find("[")
    end   = json_str.rfind("]")
    if start == -1 or end == -1:
        print(f"  [JSON WARN] No JSON array found for {source_label}", flush=True)
        return 0

    json_str = json_str[start:end + 1]

    try:
        triples = json.loads(json_str)
    except json.JSONDecodeError as e:
        last_brace = json_str.rfind("},")
        if last_brace > 0:
            try:
                triples = json.loads(json_str[:last_brace + 1] + "]")
                print(f"  [JSON SALVAGE] Partial JSON recovered for {source_label}", flush=True)
            except Exception:
                print(f"  [JSON ERROR] Cannot parse for {source_label}: {e}", flush=True)
                return 0
        else:
            print(f"  [JSON ERROR] Cannot parse for {source_label}: {e}", flush=True)
            return 0

    source_doc = document_title if document_title else (source_label.split(" page ")[0].strip() if " page " in source_label else source_label.strip())
    page_int   = int(page_num) if str(page_num).isdigit() else 0

    count = 0
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        subj_label = clean_literal(str(triple.get("subject", "")).strip())
        pred_label = str(triple.get("predicate", "")).strip()
        subj_type = str(triple.get("subject_type", "")).strip()
        obj_label  = clean_literal(str(triple.get("object",  "")).strip())

        if not subj_label or not pred_label or not obj_label:
            continue

        if is_section_number_artifact(subj_label) or is_section_number_artifact(obj_label):
            print(f"  [FILTER-SECNUM] Skipped section-number artifact in {source_label}: {subj_label} / {obj_label}", flush=True)
            continue

        try:
            subj_fragment = normalize_uri_fragment(subj_label)
            subj_uri      = TETRA[subj_fragment]

            g.add((subj_uri, RDF.type, OWL.NamedIndividual))
            resolved_type = subj_type if subj_type in VALID_CLASSES else None
            if not resolved_type:
                subj_label_norm = subj_label.lower().strip()
                for alt_label, cls in ALTLABEL_TO_CLASS:
                    if subj_label_norm == alt_label:
                        resolved_type = cls
                        break
            if resolved_type:
                g.add((subj_uri, RDF.type, TETRA[resolved_type]))
            g.add((subj_uri, SKOS.prefLabel, Literal(subj_label, lang="fr")))

            g.add((subj_uri, TETRA.hasPageNumber, Literal(page_int, datatype=XSD.integer)))
            g.add((subj_uri, TETRA.dataSource,    Literal(source_doc, datatype=XSD.string)))
            g.add((subj_uri, TETRA.extractedFrom, Literal(source_type, datatype=XSD.string)))

            pred_clean = re.sub(r"[^\w]", "", pred_label)
            if not pred_clean:
                continue

            pred_lower = pred_clean.lower()
            DATE_PREDICATE_HINTS = ("date", "startsat", "endsat")
            if pred_lower in ("haspagenum", "haspagenumber", "pagenumber"):
                g.add((subj_uri, TETRA.hasPageNumber, Literal(page_int, datatype=XSD.integer)))
            elif pred_lower in ("datasource", "source"):
                g.add((subj_uri, TETRA.dataSource, Literal(source_doc, datatype=XSD.string)))
            elif pred_lower in ("label", "name", "preflabel"):
                g.add((subj_uri, SKOS.prefLabel, Literal(obj_label, lang="fr")))
            elif any(h in pred_lower for h in DATE_PREDICATE_HINTS) and (obj_label.strip() == str(page_int) or obj_label.strip().isdigit() and len(obj_label.strip()) <= 3):
                print(f"  [FILTER-PAGEASDATE] Rejected page-number-as-date for predicate '{pred_clean}' in {source_label}: {obj_label}", flush=True)
                note_value = clean_literal(f"{pred_clean}: {obj_label}")
                g.add((subj_uri, SKOS.note, Literal(note_value, lang="fr")))
                continue
            else:
                if pred_clean not in VALID_PREDICATES:
                    global _demoted_predicates_count
                    _demoted_predicates_count += 1
                    note_value = clean_literal(f"{pred_clean}: {obj_label}")
                    g.add((subj_uri, SKOS.note, Literal(note_value, lang="fr")))
                    continue

                pred_uri = TETRA[pred_clean]

                if source_type == "Image":
                    g.add((subj_uri, TETRA.extractedFrom,
                           Literal("VQA-verified caption", datatype=XSD.string)))

                obj_is_entity = (
                    len(obj_label.split()) <= 6
                    and not is_numeric_literal(obj_label)
                    and obj_label not in (source_type, page_num)
                )

                if obj_is_entity:
                    obj_fragment = normalize_uri_fragment(obj_label)
                    obj_uri      = TETRA[obj_fragment]
                    g.add((subj_uri, pred_uri, obj_uri))
                    g.add((obj_uri, RDF.type, OWL.NamedIndividual))
                    g.add((obj_uri, SKOS.prefLabel, Literal(obj_label, lang="fr")))
                    g.add((obj_uri, TETRA.hasPageNumber, Literal(page_int, datatype=XSD.integer)))
                    g.add((obj_uri, TETRA.dataSource,    Literal(source_doc, datatype=XSD.string)))
                    g.add((obj_uri, TETRA.extractedFrom, Literal(source_type, datatype=XSD.string)))
                    g.add((pred_uri, RDF.type, OWL.ObjectProperty))
                else:
                    g.add((subj_uri, pred_uri, Literal(obj_label, datatype=XSD.string)))
                    g.add((pred_uri, RDF.type, OWL.DatatypeProperty))

            count += 1

        except Exception as e:
            print(f"  [TRIPLE SKIP] Bad triple in {source_label}: {e}", flush=True)
            continue

    return count


def parse_page_number(key: str, source: str) -> str:
    if source == "text":
        match = re.search(r"page_(\d+)", key)
    else:
        match = re.search(r"img_p(\d+)", key)
    return match.group(1) if match else "0"


def merge_same_label_entities(g: Graph) -> Graph:
    """
    Two-pass entity resolution: merges cross-page duplicate nodes by skos:prefLabel.
    """
    label_to_uris = defaultdict(list)

    for s, p, o in g.triples((None, SKOS.prefLabel, None)):
        label_key = " ".join(str(o).strip().lower().split())
        label_key = re.sub(r"^(la |le |les |l'|l’|un |une |des |du |de la )", "", label_key)
        if label_key:
            label_to_uris[label_key].append(s)

    merged_count = 0

    for label, uris in label_to_uris.items():
        unique_uris = list(dict.fromkeys(uris))
        if len(unique_uris) < 2:
            continue

        canonical = max(unique_uris, key=lambda u: len(list(g.triples((u, None, None)))))

        for duplicate in unique_uris:
            if duplicate == canonical:
                continue

            for s, p, o in list(g.triples((duplicate, None, None))):
                if p == SKOS.prefLabel:
                    continue
                g.add((canonical, p, o))

            for s, p, o in list(g.triples((None, None, duplicate))):
                g.add((s, p, canonical))

            for triple in list(g.triples((duplicate, None, None))):
                g.remove(triple)
            for triple in list(g.triples((None, None, duplicate))):
                g.remove(triple)

            merged_count += 1

    print(f"  [MERGE] {merged_count} duplicate entities resolved into canonical nodes.", flush=True)
    return g


NUMERIC_LITERAL_PATTERN = re.compile(r'^[\(\[]?[-<>~]?\s*\d')
SECTION_NUMBER_PATTERN = re.compile(r'^\d{1,3}(\.\d{1,3}){1,4}$')


def is_numeric_literal(label: str) -> bool:
    """Generic check: any object value starting with a digit is a measurement/count,
    never a real-world entity — regardless of unit or language."""
    return bool(NUMERIC_LITERAL_PATTERN.match(label.strip()))


def is_section_number_artifact(label: str) -> bool:
    """Catches TOC-style section numbers like '1.2.3' that should never become entities."""
    return bool(SECTION_NUMBER_PATTERN.match(label.strip()))


def resolve_entity_aliases_by_class(g: Graph) -> dict:
    """
    Label-similarity based resolution: merges entities whose labels are
    near-identical strings (after normalization), not just same-class.
    This prevents geographically distinct entities (e.g. river vs floodplain)
    from being wrongly merged just because they share an ontology class.
    """
    from difflib import SequenceMatcher

    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r"^(la |le |les |l'|l\u2019|un |une |des |du |de la )", "", s)
        return s

    label_to_uri = {}
    label_counts = {}
    for s, p, o in g.triples((None, SKOS.prefLabel, None)):
        lbl = str(o).strip()
        label_to_uri[lbl] = s
        label_counts[lbl] = len(list(g.triples((s, None, None))))

    labels = list(label_to_uri.keys())
    alias_map = {}

    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            na, nb = normalize(a), normalize(b)
            num_a, num_b = _trailing_number(a), _trailing_number(b)
            if num_a is not None and num_b is not None and num_a != num_b:
                continue
            ratio = SequenceMatcher(None, na, nb).ratio()
            if ratio >= 0.92:
                canonical, duplicate = (a, b) if label_counts[a] >= label_counts[b] else (b, a)
                alias_map[duplicate] = canonical
                print(f"  [ALIAS-SIM] '{duplicate}' → '{canonical}' (similarity={ratio:.2f})", flush=True)

    print(f"  [ALIAS-CLASS] {len(alias_map)} near-identical label entities merged.", flush=True)
    return alias_map


def apply_alias_mapping(g: Graph, alias_map: dict) -> Graph:
    if not alias_map:
        return g

    label_to_uri = {}
    for s, p, o in g.triples((None, SKOS.prefLabel, None)):
        label_key = " ".join(str(o).strip().lower().split())
        label_to_uri.setdefault(label_key, s)

    merged = 0
    for alias_label, canonical_label in alias_map.items():
        alias_key     = " ".join(alias_label.strip().lower().split())
        canonical_key = " ".join(canonical_label.strip().lower().split())

        alias_uri     = label_to_uri.get(alias_key)
        canonical_uri = label_to_uri.get(canonical_key)

        if alias_uri is None or canonical_uri is None or alias_uri == canonical_uri:
            continue

        for s, p, o in list(g.triples((alias_uri, None, None))):
            if p == SKOS.prefLabel:
                continue
            g.add((canonical_uri, p, o))
        for s, p, o in list(g.triples((None, None, alias_uri))):
            g.add((s, p, canonical_uri))

        g.remove((alias_uri, None, None))
        g.remove((None, None, alias_uri))
        merged += 1

    print(f"  [ALIAS MERGE] {merged} generic-term entities folded into canonical entities.", flush=True)
    return g

def build_alias_resolution_prompt(entity_labels: list) -> str:
    labels_str = "\n".join(f"- {l}" for l in entity_labels)
    return f"""Tu es un expert en résolution de coréférences pour un graphe de connaissances.

Voici la liste de toutes les entités extraites d'un même document :
{labels_str}

MISSION : Identifie les entités qui désignent en réalité LE MÊME objet réel, où l'une est un terme générique
(ex: "cours d'eau", "le site", "cet ouvrage", "la structure", "ce tronçon") et l'autre est le nom propre
spécifique de ce même objet dans ce document (ex: "Ruisseau de Norroy", "Tronçon 5").

RÈGLES :
1. Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ou après.
2. Format : {{"terme_generique": "nom_propre_canonique", ...}}
3. Ne mappe QUE si tu es CERTAIN qu'il s'agit du même objet réel dans ce document.
4. N'invente aucune entité absente de la liste ci-dessus.
5. Si aucune correspondance fiable n'existe, réponds avec: {{}}

Exemple de sortie valide :
{{"cours d'eau": "Ruisseau de Norroy", "le site": "Ruisseau de Norroy"}}
"""


def resolve_entity_aliases(tokenizer, model, entity_labels: list) -> dict:
    if not entity_labels:
        return {}
    labels_capped = entity_labels[:300]
    prompt = build_alias_resolution_prompt(labels_capped)
    raw = call_deepseek_raw_prompt(prompt, "alias_resolution_pass", tokenizer, model)
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        mapping = json.loads(raw[start:end + 1])
        if not isinstance(mapping, dict):
            return {}
        return {k.strip().lower(): v.strip() for k, v in mapping.items() if k and v}
    except Exception:
        return {}

# --- MAIN PDF PROCESSOR ---
def process_pdf(
    pdf_base: str,
    tokenizer,
    ds_model,
    captions_root: Path,
    text_root: Path,
    ttl_output_root: Path,
):
    print(f"\n{'=' * 60}\nPROCESSING: {pdf_base}\n{'=' * 60}", flush=True)

    captions_path = (
        captions_root / pdf_base / "image_captions_p3_contrast.json"
    )

    text_path = text_root / pdf_base / "extracted_content.json"
    output_dir = ttl_output_root / pdf_base
    output_dir.mkdir(parents=True, exist_ok=True)

    text_ttl_path = output_dir / "texts.ttl"
    image_ttl_path = output_dir / "images.ttl"

    document_title = pdf_base

    if text_path.exists():
        with open(text_path, "r", encoding="utf-8") as file:
            metadata_probe = json.load(file)

        document_title = (
            metadata_probe.get("metadata", {}).get("title") or pdf_base
        ).strip()

        print(
            f"  [DOC-TITLE] Using document title: {document_title}",
            flush=True,
        )

    entity_registry = {}

    if document_title and document_title != pdf_base:
        entity_registry[document_title] = None

    river_match = re.search(
        r"ruisseau\s+\w+(?:\s+\w+)?",
        document_title.lower(),
    )

    if river_match:
        entity_registry[document_title] = None

    # ----------------------------------------------------------------
    # PASS 1: TEXT BLOCKS
    # ----------------------------------------------------------------
    print("\n--- PASS 1: TEXT ---", flush=True)

    text_graph = make_base_graph()
    text_total_triples = 0

    if text_ttl_path.exists():
        raise FileExistsError(
            f"Refusing to reuse an existing output file: {text_ttl_path}\n"
            "Choose a new --output-root or remove the previous test output "
            "before starting a fresh run."
        )

    elif text_path.exists():
        with open(text_path, "r", encoding="utf-8") as file:
            pages_dict = json.load(file).get("pages", {})

        normal_items, long_items = [], []

        for page_key, page_text in pages_dict.items():
            if len(page_text.split()) < 15:
                continue

            page_num = parse_page_number(page_key, "text")

            if len(page_text.split()) > CHUNK_WORDS:
                long_items.append((page_key, page_num, page_text))
            else:
                normal_items.append((page_key, page_num, page_text))

        batch_size = 4

        for batch_start in range(0, len(normal_items), batch_size):
            batch = normal_items[batch_start : batch_start + batch_size]

            prompts_and_labels = [
                (page_text, "Text", page_num, entity_registry, page_key)
                for page_key, page_num, page_text in batch
            ]

            raw_results = call_deepseek_batch(
                prompts_and_labels,
                tokenizer,
                ds_model,
            )

            for (
                page_key,
                page_num,
                _,
            ), raw_json in zip(batch, raw_results):
                if not raw_json:
                    continue

                count = json_to_rdf_graph(
                    raw_json,
                    "Text",
                    page_num,
                    None,
                    page_key,
                    text_graph,
                    document_title,
                )

                if count > 0:
                    text_total_triples += count

                    print(
                        f"  [OK] {page_key}: {count} triples",
                        flush=True,
                    )

            for subject, predicate, obj in text_graph.triples(
                (None, SKOS.prefLabel, None)
            ):
                entity_registry[str(obj)] = None

        for page_key, page_num, page_text in long_items:
            raw_json = process_chunked_content(
                page_text,
                "Text",
                page_num,
                entity_registry,
                page_key,
                tokenizer,
                ds_model,
            )

            count = json_to_rdf_graph(
                raw_json,
                "Text",
                page_num,
                None,
                page_key,
                text_graph,
                document_title,
            )

            if count > 0:
                text_total_triples += count

                print(
                    f"  [OK-CHUNKED] {page_key}: {count} triples",
                    flush=True,
                )

            for subject, predicate, obj in text_graph.triples(
                (None, SKOS.prefLabel, None)
            ):
                entity_registry[str(obj)] = None

    if text_total_triples > 0:
        text_graph.serialize(destination=str(text_ttl_path), format="turtle")

        print(
            f"\n-> texts.ttl saved: {text_total_triples} valid triples",
            flush=True,
        )

    # ----------------------------------------------------------------
    # PASS 2: IMAGE CAPTIONS
    # ----------------------------------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n--- PASS 2: IMAGES ---", flush=True)

    image_graph = make_base_graph()
    image_total_triples = 0

    if captions_path.exists():
        with open(captions_path, "r", encoding="utf-8") as file:
            captions_dict = json.load(file)

        normal_image_items, long_image_items = [], []

        for image_name, caption in captions_dict.items():
            if (
                not caption
                or "DECORATIVE" in str(caption)
                or "non informative" in str(caption).lower()
                or len(str(caption).split()) < 15
            ):
                continue

            page_num = parse_page_number(image_name, "image")
            caption_string = str(caption)

            if len(caption_string.split()) > CHUNK_WORDS:
                long_image_items.append(
                    (image_name, page_num, caption_string)
                )
            else:
                normal_image_items.append(
                    (image_name, page_num, caption_string)
                )

        batch_size = 2

        for batch_start in range(
            0,
            len(normal_image_items),
            batch_size,
        ):
            batch = normal_image_items[
                batch_start : batch_start + batch_size
            ]

            prompts_and_labels = [
                (caption, "Image", page_num, entity_registry, image_name)
                for image_name, page_num, caption in batch
            ]

            raw_results = call_deepseek_batch(
                prompts_and_labels,
                tokenizer,
                ds_model,
            )

            for (
                image_name,
                page_num,
                _,
            ), raw_json in zip(batch, raw_results):
                if not raw_json:
                    continue

                count = json_to_rdf_graph(
                    raw_json,
                    "Image",
                    page_num,
                    None,
                    image_name,
                    image_graph,
                    document_title,
                )

                if count > 0:
                    image_total_triples += count

                    print(
                        f"  [OK] {image_name}: {count} triples",
                        flush=True,
                    )
                else:
                    print(
                        f"  [SKIP] {image_name}: no valid triples",
                        flush=True,
                    )

            for subject, predicate, obj in image_graph.triples(
                (None, SKOS.prefLabel, None)
            ):
                entity_registry[str(obj)] = None

        for image_name, page_num, caption_string in long_image_items:
            raw_json = process_chunked_content(
                caption_string,
                "Image",
                page_num,
                entity_registry,
                image_name,
                tokenizer,
                ds_model,
            )

            count = json_to_rdf_graph(
                raw_json,
                "Image",
                page_num,
                None,
                image_name,
                image_graph,
                document_title,
            )

            if count > 0:
                image_total_triples += count

                print(
                    f"  [OK-CHUNKED] {image_name}: {count} triples",
                    flush=True,
                )
            else:
                print(
                    f"  [SKIP] {image_name}: no valid triples",
                    flush=True,
                )

            for subject, predicate, obj in image_graph.triples(
                (None, SKOS.prefLabel, None)
            ):
                entity_registry[str(obj)] = None

    if image_total_triples > 0:
        image_graph.serialize(destination=str(image_ttl_path), format="turtle")

        print(
            f"\n-> images.ttl saved: {image_total_triples} valid triples",
            flush=True,
        )

    # ----------------------------------------------------------------
    # MERGED GRAPH
    # ----------------------------------------------------------------
    merged_path = output_dir / "ft_merged_knowledge_graph.ttl"

    try:
        merged_graph = make_base_graph()

        for ttl_path in [text_ttl_path, image_ttl_path]:
            if ttl_path.exists():
                try:
                    merged_graph.parse(str(ttl_path), format="turtle")

                except Exception as error:
                    print(
                        f"  [MERGE WARN] Could not parse {ttl_path}: "
                        f"{error}",
                        flush=True,
                    )

        if len(merged_graph) > 0:
            text_entities = {
                subject
                for subject, predicate, obj in text_graph.triples(
                    (None, RDF.type, OWL.NamedIndividual)
                )
            }

            image_entities = {
                subject
                for subject, predicate, obj in image_graph.triples(
                    (None, RDF.type, OWL.NamedIndividual)
                )
            }

            new_entities = image_entities - text_entities
            cross_modal = text_entities & image_entities

            print(f"  Text-only entities : {len(text_entities)}")
            print(f"  Image-only new     : {len(new_entities)}")
            print(f"  Cross-modal linked : {len(cross_modal)}")

            merged_graph = merge_same_label_entities(merged_graph)

            all_labels = sorted(
                {
                    str(obj)
                    for subject, predicate, obj in merged_graph.triples(
                        (None, SKOS.prefLabel, None)
                    )
                }
            )

            alias_map_llm = resolve_entity_aliases(
                tokenizer,
                ds_model,
                all_labels,
            )

            merged_graph = apply_alias_mapping(
                merged_graph,
                alias_map_llm,
            )

            merged_graph = merge_same_label_entities(merged_graph)

            alias_map_similarity = resolve_entity_aliases_by_class(
                merged_graph
            )

            merged_graph = apply_alias_mapping(
                merged_graph,
                alias_map_similarity,
            )

            merged_graph = merge_same_label_entities(merged_graph)

            merged_graph.serialize(
                destination=str(merged_path),
                format="turtle",
            )

            print(
                "\n-> ft_merged_knowledge_graph.ttl saved: "
                f"{len(merged_graph)} total triples",
                flush=True,
            )

            print(
                "   Demoted off-ontology predicates : "
                f"{_demoted_predicates_count}",
                flush=True,
            )

            print(
                "   Open ft_merged_knowledge_graph.ttl in Gephi or yEd "
                "for visualization.",
                flush=True,
            )

    except Exception as error:
        print(f"  [MERGE ERROR] {error}", flush=True)


def select_pdf_folders(captions_root: Path, document: str) -> list[str]:
    if not captions_root.exists():
        print(
            f"[ERROR] Captions directory does not exist: {captions_root}",
            flush=True,
        )
        sys.exit(1)

    available_folders = sorted(
        folder.name
        for folder in captions_root.iterdir()
        if folder.is_dir()
    )

    if not available_folders:
        print(
            "[ERROR] No document folders found in captions directory.",
            flush=True,
        )
        sys.exit(1)

    if document == "all":
        return available_folders

    if document not in available_folders:
        print(
            f"[ERROR] Requested document folder not found: {document}",
            flush=True,
        )
        print(
            "[INFO] Available document folders: "
            + ", ".join(available_folders),
            flush=True,
        )
        sys.exit(1)

    return [document]


def main(args: argparse.Namespace):
    captions_root = args.captions_root.expanduser().resolve()
    text_root = args.text_root.expanduser().resolve()
    ttl_output_root = args.output_root.expanduser().resolve()

    print("=" * 70, flush=True)
    print("PIPELINE 4: FINE-TUNED KNOWLEDGE-GRAPH EXTRACTION", flush=True)
    print(f"Captions root: {captions_root}", flush=True)
    print(f"Text root: {text_root}", flush=True)
    print(f"Output root: {ttl_output_root}", flush=True)
    print(f"Document selection: {args.document}", flush=True)
    print("=" * 70, flush=True)

    if not ONTOLOGY_TTL_PATH.exists():
        print(
            f"[ERROR] Compact ontology file not found: {ONTOLOGY_TTL_PATH}",
            flush=True,
        )
        sys.exit(1)

    if not FULL_ONTOLOGY_PATH.exists():
        print(
            f"[ERROR] Full ontology file not found: {FULL_ONTOLOGY_PATH}",
            flush=True,
        )
        sys.exit(1)

    if not MODEL_ID.exists():
        print(
            f"[ERROR] Fine-tuned merged model directory not found: {MODEL_ID}",
            flush=True,
        )
        print(
            "[INFO] Download the merged checkpoint from the location "
            "documented in MODEL_SETUP.md, or reproduce it through the "
            "included fine-tuning workflow.",
            flush=True,
        )
        sys.exit(1)

    ttl_output_root.mkdir(parents=True, exist_ok=True)

    pdf_folders = select_pdf_folders(captions_root, args.document)

    print(f"[INFO] Found {len(pdf_folders)} PDF folders: {pdf_folders}", flush=True)

    tokenizer, ds_model = load_deepseek_model()

    for pdf_base in pdf_folders:
        process_pdf(
            pdf_base,
            tokenizer,
            ds_model,
            captions_root,
            text_root,
            ttl_output_root,
        )

    print("[DONE] All requested PDFs processed.", flush=True)


if __name__ == "__main__":
    main(parse_args())