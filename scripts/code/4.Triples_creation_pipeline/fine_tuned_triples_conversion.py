import torch
import os
import json
import time
import re
import unicodedata
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS
from collections import defaultdict

# --- PATHS ---
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise EnvironmentError("HF_TOKEN environment variable not set. Run 'export HF_TOKEN=...' before executing this script.")
BASE_IMAGE_CAPTIONS = "/home2020/home/icube/janil/Rhine_proj/outputs/5th_ablation_Image_captioning_prompt_engineering/nvila_p3_contrast_captions"
BASE_TEXT_JSON      = "/home2020/home/icube/janil/Rhine_proj/outputs/json_results"

BASE_TTL_OUTPUT = "/home2020/home/icube/janil/Rhine_proj/outputs/triples_fine_tuned"
ONTOLOGY_TTL_PATH = "/home2020/home/icube/janil/Rhine_proj/data/ontology/TetraOnto_compact.txt"


TETRA_NS = "http://www.semanticweb.org/fethi/ontologies/2024/1/TetraOnto.owl#"
TETRA    = Namespace(TETRA_NS)


PREFIX_HEADER = f"""@prefix : <{TETRA_NS}> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
"""


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


ALTLABEL_TO_CLASS = load_altlabel_to_class_map(ONTOLOGY_TTL_PATH.replace("_compact.txt", ".ttl"))

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


MODEL_ID = "/home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/5.merge/Qwen2.5-14B-populate-ontology-merged"


print(f"[GPU CHECK] Available: {torch.cuda.device_count()}", flush=True)
print(f"[GPU CHECK] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)

def pick_free_gpu(min_free_gb=10):
    best_device = 0
    best_free = -1
    for i in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free_gb = free_bytes / (1024 ** 3)
        print(f"[GPU CHECK] GPU {i}: {free_gb:.2f} GiB free", flush=True)
        if free_gb > best_free:
            best_free = free_gb
            best_device = i
    if best_free < min_free_gb:
        print(f"[WARN] Best available GPU ({best_device}) only has {best_free:.2f} GiB free — may still OOM.", flush=True)
    else:
        print(f"[GPU CHECK] Selected GPU {best_device} with {best_free:.2f} GiB free.", flush=True)
    return best_device

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


def call_deepseek(text_input: str, source_type: str, page_num: str, entity_registry, source_label: str,
                   tokenizer, model) -> str:
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
            print(f"  [OOM-RETRY FAIL] {source_label}: {e2}", flush=True)
            with open(os.path.join(BASE_TTL_OUTPUT, "oom_failures.txt"), "a") as f:
                f.write(f"{source_label}\n")
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
def process_pdf(pdf_base: str, tokenizer, ds_model):
    print(f"\n{'='*60}\nPROCESSING: {pdf_base}\n{'='*60}", flush=True)

    captions_path = os.path.join(BASE_IMAGE_CAPTIONS, pdf_base, "image_captions_p3_contrast.json")
    text_path     = os.path.join(BASE_TEXT_JSON, pdf_base, "extracted_content.json")
    output_dir    = os.path.join(BASE_TTL_OUTPUT, pdf_base)
    os.makedirs(output_dir, exist_ok=True)

    text_ttl_path  = os.path.join(output_dir, "texts.ttl")
    image_ttl_path = os.path.join(output_dir, "images.ttl")

    document_title = pdf_base
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            _meta_probe = json.load(f)
        document_title = (_meta_probe.get("metadata", {}).get("title") or pdf_base).strip()
        print(f"  [DOC-TITLE] Using document title: {document_title}", flush=True)

    entity_registry = {}

    # Pre-seed registry with document title as the canonical water body name
    # so the model resolves "cours d'eau" to the real name from block 1
    if document_title and document_title != pdf_base:
        entity_registry[document_title] = None
    river_match = re.search(r'ruisseau\s+\w+(?:\s+\w+)?', document_title.lower())
    if river_match:
        entity_registry[document_title] = None

    # ----------------------------------------------------------------
    # PASS 1: TEXT BLOCKS
    # ----------------------------------------------------------------
    print("\n--- PASS 1: TEXT ---", flush=True)
    text_graph = make_base_graph()
    text_total_triples = 0

    if os.path.exists(text_ttl_path):
        print(f"  [SKIP-PASS1] {text_ttl_path} already exists, loading instead of regenerating.", flush=True)
        text_graph.parse(text_ttl_path, format="turtle")
        text_total_triples = len(text_graph)
        for s, p, o in text_graph.triples((None, SKOS.prefLabel, None)):
            entity_registry[str(o)] = None
    elif os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            pages_dict = json.load(f).get("pages", {})

        normal_items, long_items = [], []
        for page_key, page_text in pages_dict.items():
            if len(page_text.split()) < 15:
                continue
            page_num = parse_page_number(page_key, "text")
            if len(page_text.split()) > CHUNK_WORDS:
                long_items.append((page_key, page_num, page_text))
            else:
                normal_items.append((page_key, page_num, page_text))

        BATCH_SIZE = 4
        for batch_start in range(0, len(normal_items), BATCH_SIZE):
            batch = normal_items[batch_start:batch_start + BATCH_SIZE]
            prompts_and_labels = [
                (page_text, "Text", page_num, entity_registry, page_key)
                for (page_key, page_num, page_text) in batch
            ]
            raw_results = call_deepseek_batch(prompts_and_labels, tokenizer, ds_model)

            for (page_key, page_num, _), raw_json in zip(batch, raw_results):
                if not raw_json:
                    continue
                count = json_to_rdf_graph(raw_json, "Text", page_num, None, page_key, text_graph, document_title)
                if count > 0:
                    text_total_triples += count
                    print(f"  [OK] {page_key}: {count} triples", flush=True)

            for s, p, o in text_graph.triples((None, SKOS.prefLabel, None)):
                entity_registry[str(o)] = None

        for page_key, page_num, page_text in long_items:
            raw_json = process_chunked_content(page_text, "Text", page_num, entity_registry, page_key, tokenizer, ds_model)
            count = json_to_rdf_graph(raw_json, "Text", page_num, None, page_key, text_graph, document_title)
            if count > 0:
                text_total_triples += count
                print(f"  [OK-CHUNKED] {page_key}: {count} triples", flush=True)
            for s, p, o in text_graph.triples((None, SKOS.prefLabel, None)):
                entity_registry[str(o)] = None

    if text_total_triples > 0:
        text_graph.serialize(destination=text_ttl_path, format="turtle")
        print(f"\n-> texts.ttl saved: {text_total_triples} valid triples", flush=True)

    # ----------------------------------------------------------------
    # PASS 2: IMAGE CAPTIONS
    # Inherits entity_registry from text pass so captions can resolve
    # generic references against entities already found in text.
    # ----------------------------------------------------------------
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("\n--- PASS 2: IMAGES ---", flush=True)
    image_graph         = make_base_graph()
    image_total_triples = 0

    if os.path.exists(captions_path):
        with open(captions_path, "r", encoding="utf-8") as f:
            captions_dict = json.load(f)

        normal_img_items, long_img_items = [], []
        for img_name, caption in captions_dict.items():
            if (
                not caption
                or "DECORATIVE" in str(caption)
                or "non informative" in str(caption).lower()
                or len(str(caption).split()) < 15
            ):
                continue
            page_num = parse_page_number(img_name, "image")
            caption_str = str(caption)
            if len(caption_str.split()) > CHUNK_WORDS:
                long_img_items.append((img_name, page_num, caption_str))
            else:
                normal_img_items.append((img_name, page_num, caption_str))

        BATCH_SIZE = 2
        for batch_start in range(0, len(normal_img_items), BATCH_SIZE):
            batch = normal_img_items[batch_start:batch_start + BATCH_SIZE]
            prompts_and_labels = [
                (caption, "Image", page_num, entity_registry, img_name)
                for (img_name, page_num, caption) in batch
            ]
            raw_results = call_deepseek_batch(prompts_and_labels, tokenizer, ds_model)

            for (img_name, page_num, _), raw_json in zip(batch, raw_results):
                if not raw_json:
                    continue
                count = json_to_rdf_graph(raw_json, "Image", page_num, None, img_name, image_graph, document_title)
                if count > 0:
                    image_total_triples += count
                    print(f"  [OK] {img_name}: {count} triples", flush=True)
                else:
                    print(f"  [SKIP] {img_name}: no valid triples", flush=True)

            for s, p, o in image_graph.triples((None, SKOS.prefLabel, None)):
                entity_registry[str(o)] = None

        for img_name, page_num, caption_str in long_img_items:
            raw_json = process_chunked_content(caption_str, "Image", page_num, entity_registry, img_name, tokenizer, ds_model)
            count = json_to_rdf_graph(raw_json, "Image", page_num, None, img_name, image_graph, document_title)
            if count > 0:
                image_total_triples += count
                print(f"  [OK-CHUNKED] {img_name}: {count} triples", flush=True)
            else:
                print(f"  [SKIP] {img_name}: no valid triples", flush=True)
            for s, p, o in image_graph.triples((None, SKOS.prefLabel, None)):
                entity_registry[str(o)] = None

    if image_total_triples > 0:
        image_graph.serialize(destination=image_ttl_path, format="turtle")
        print(f"\n-> images.ttl saved: {image_total_triples} valid triples", flush=True)

    # ----------------------------------------------------------------
    # MERGED GRAPH
    # ----------------------------------------------------------------
    merged_path = os.path.join(output_dir, "ft_merged_knowledge_graph.ttl")
    try:
        g_merged = make_base_graph()
        for ttl_path in [text_ttl_path, image_ttl_path]:
            if os.path.exists(ttl_path):
                try:
                    g_merged.parse(ttl_path, format="turtle")
                except Exception as e:
                    print(f"  [MERGE WARN] Could not parse {ttl_path}: {e}", flush=True)

        if len(g_merged) > 0:
            text_entities  = set(s for s, p, o in text_graph.triples((None, RDF.type, OWL.NamedIndividual)))
            image_entities = set(s for s, p, o in image_graph.triples((None, RDF.type, OWL.NamedIndividual)))

            new_entities = image_entities - text_entities
            cross_modal  = text_entities & image_entities
            print(f"  Text-only entities : {len(text_entities)}")
            print(f"  Image-only new     : {len(new_entities)}")
            print(f"  Cross-modal linked : {len(cross_modal)}")

            g_merged = merge_same_label_entities(g_merged)
            all_labels = sorted(set(str(o) for s, p, o in g_merged.triples((None, SKOS.prefLabel, None))))
            alias_map_llm = resolve_entity_aliases(tokenizer, ds_model, all_labels)
            g_merged = apply_alias_mapping(g_merged, alias_map_llm)
            g_merged = merge_same_label_entities(g_merged)
            alias_map_sim = resolve_entity_aliases_by_class(g_merged)
            g_merged = apply_alias_mapping(g_merged, alias_map_sim)
            g_merged = merge_same_label_entities(g_merged)

            g_merged.serialize(destination=merged_path, format="turtle")
            print(f"\n-> ft_merged_knowledge_graph.ttl saved: {len(g_merged)} total triples", flush=True)
            print(f"   Demoted off-ontology predicates : {_demoted_predicates_count}", flush=True)
            print(f"   Open merged_knowledge_graph.ttl in Gephi or yEd for visualization.", flush=True)
    except Exception as e:
        print(f"  [MERGE ERROR] {e}", flush=True)


def main():
    CHECKPOINT_FILE = os.path.join(BASE_TTL_OUTPUT, "completed_pdfs.txt")

    def load_completed():
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE) as f:
                return set(line.strip() for line in f)
        return set()

    def mark_completed(pdf_name):
        with open(CHECKPOINT_FILE, "a") as f:
            f.write(pdf_name + "\n")

    tokenizer, ds_model = load_deepseek_model()

    pdf_folders = [
        f for f in os.listdir(BASE_IMAGE_CAPTIONS)
        if os.path.isdir(os.path.join(BASE_IMAGE_CAPTIONS, f))
    ]
    print(f"[INFO] Found {len(pdf_folders)} PDF folders: {pdf_folders}", flush=True)

    if not pdf_folders:
        print("[ERROR] No PDF folders found in captions directory.", flush=True)
        return

    completed = load_completed()
    print(f"[CHECKPOINT] Already completed: {completed}", flush=True)

    for pdf_base in pdf_folders:
        if pdf_base in completed:
            print(f"[SKIP] Already completed: {pdf_base}", flush=True)
            continue
        process_pdf(pdf_base, tokenizer, ds_model)
        mark_completed(pdf_base)

    print("\n[DONE] All PDFs processed.", flush=True)


if __name__ == "__main__":
    main()