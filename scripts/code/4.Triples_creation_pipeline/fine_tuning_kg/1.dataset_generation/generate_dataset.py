import os
import json
import re
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ONTOLOGY_TTL_PATH = "/home2020/home/icube/janil/Rhine_proj/data/ontology/TetraOnto_compact.txt"
SEED_TEXT_PATH = "/home2020/home/icube/janil/Rhine_proj/outputs/json_results/fichier/extracted_content.json"
OUTPUT_JSONL = "/home2020/home/icube/janil/Rhine_proj/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/1.dataset_generation/synthetic_dataset.jsonl"
MODEL_ID = "/home2020/home/icube/janil/Rhine_proj/hf_models/Qwen2.5-14B-Instruct"

N_GENERAL_EXAMPLES = 130
N_CONSISTENCY_PAIRS = 25  # -> 50 examples total (2 per pair)


def load_ontology(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def pick_free_gpu(min_free_gb=10):
    best_device, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(i)
        free_gb = free_bytes / (1024 ** 3)
        if free_gb > best_free:
            best_free, best_device = free_gb, i
    return best_device


def load_model():
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
    )
    model.eval()
    return tokenizer, model


def call_model(prompt, tokenizer, model, max_new_tokens=1200):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    del inputs, outputs, generated
    torch.cuda.empty_cache()
    return result

def call_model_batch(prompts, tokenizer, model, max_new_tokens=400):
    input_texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for i in range(len(prompts)):
        gen = outputs[i][inputs.input_ids.shape[1]:]
        results.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    del inputs, outputs
    torch.cuda.empty_cache()
    return results


SCENARIO_GEN_PROMPT = """Tu es un expert en restauration hydromorphologique de cours d'eau.

Génère UN scénario réaliste et FICTIF (n'utilise pas de vrais noms de lieux français réels) de restauration
de cours d'eau, en français, de 4 à 8 phrases, riche en détails concrets (nom d'un cours d'eau fictif,
structure concernée, maître d'ouvrage, dates, coûts, hauteur, monitoring, etc.).

Réponds UNIQUEMENT avec le texte du scénario, sans introduction ni commentaire."""


TRIPLE_EXTRACTION_META_PROMPT = """Tu es un expert en extraction de connaissances pour un graphe de connaissances
scientifique sur l'hydromorphologie fluviale et la restauration de cours d'eau.

ONTOLOGIE DISPONIBLE (utilise UNIQUEMENT ces classes et propriétés) :
{ontology}
{registry_instruction}

CONTENU À ANALYSER :
Source: {source_type} | Page: {page_num}
{text_input}

MISSION : Extrais les faits de ce texte sous forme de triplets JSON.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT avec un tableau JSON valide. Aucun texte avant ou après.
2. Chaque triplet a exactement 3 clés : "subject", "predicate", "object"
3. "subject" : nom court en français de l'entité principale
4. "predicate" : EXACTEMENT un terme de l'ontologie ci-dessus
5. "object" : valeur ou nom court de l'entité liée
6. Maximum 15 triplets par bloc.
7. Commence DIRECTEMENT par [ sans aucune introduction.
8. Si aucun triplet pertinent n'existe dans ce texte, réponds UNIQUEMENT avec: []
"""


def build_registry_instruction(registry_list):
    if not registry_list:
        return ""
    registry_str = ", ".join(registry_list)
    return f"""
ENTITÉS DÉJÀ IDENTIFIÉES DANS CE DOCUMENT (registre courant) :
{registry_str}

RÈGLE DE RÉSOLUTION D'ENTITÉ (IMPORTANT) :
Si le texte fait référence de façon générique, implicite, OU sous un AUTRE NOM/LIBELLÉ à une entité
déjà listée ci-dessus (ex: "le cours d'eau", "cet ouvrage", pronoms, mais aussi un nom différent
désignant la même chose), utilise le nom EXACT de cette entité tel qu'il apparaît dans le registre.
Ne crée PAS une nouvelle entité si elle correspond déjà à une entité du registre.
"""


def extract_json_array(raw_text):
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def generate_general_examples(tokenizer, model, ontology, n, output_path, start_from=0):
    BATCH_SIZE = 8
    i = start_from
    while i < n:
        batch_n = min(BATCH_SIZE, n - i)
        scenarios = call_model_batch([SCENARIO_GEN_PROMPT] * batch_n, tokenizer, model, max_new_tokens=400)
        prompts = [
            TRIPLE_EXTRACTION_META_PROMPT.format(
                ontology=ontology, registry_instruction="", source_type="Text",
                page_num=str(random.randint(1, 30)), text_input=s
            ) for s in scenarios
        ]
        raw_outputs = call_model_batch(prompts, tokenizer, model, max_new_tokens=1200)
        for j, raw_output in enumerate(raw_outputs):
            triples = extract_json_array(raw_output)
            if triples is None:
                print(f"[SKIP general #{i+j}] invalid JSON")
                continue
            example = {"system": "Translate the user text into an TTL graph based on the TetraOnto ontology.",
                       "prompt": prompts[j], "triples": triples}
            append_jsonl(example, output_path)
            print(f"[OK general #{i+j}] {len(triples)} triples", flush=True)
        i += batch_n
        
def to_jsonl_record(example):
    return {
        "messages": [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": json.dumps(example["triples"], ensure_ascii=False)},
        ]
    }

def generate_consistency_pairs(tokenizer, model, ontology, n_pairs, output_path, start_from=0):
    for i in range(start_from, n_pairs):
        base_scenario = call_model(SCENARIO_GEN_PROMPT, tokenizer, model, max_new_tokens=400)

        prompt_a = TRIPLE_EXTRACTION_META_PROMPT.format(
            ontology=ontology, registry_instruction="", source_type="Text",
            page_num=str(random.randint(1, 30)), text_input=base_scenario,
        )
        raw_a = call_model(prompt_a, tokenizer, model)
        triples_a = extract_json_array(raw_a)
        if not triples_a:
            print(f"[SKIP pair #{i}] invalid JSON for text pass")
            continue

        entity_name = triples_a[0].get("subject", "").strip()
        if not entity_name:
            continue

        alias_prompt = f"""Voici un scénario de restauration de cours d'eau :
{base_scenario}

Le nom de l'entité principale identifiée est : "{entity_name}"

Écris UNE SEULE phrase courte (type légende d'image), qui parle de la MÊME entité,
mais en utilisant un terme différent, plus générique ou une autre formulation
(sans répéter le nom exact), comme si c'était une légende d'image extraite du même rapport.

Réponds au format JSON strict :
{{"caption_sentence": "..."}}"""
        alias_raw = call_model(alias_prompt, tokenizer, model, max_new_tokens=200)
        alias_match = re.search(r"\{.*\}", alias_raw, re.DOTALL)
        if not alias_match:
            print(f"[SKIP pair #{i}] no alias JSON")
            continue
        try:
            alias_data = json.loads(alias_match.group(0))
        except json.JSONDecodeError:
            print(f"[SKIP pair #{i}] malformed alias JSON")
            continue

        caption_sentence = alias_data.get("caption_sentence", "").strip()
        if not caption_sentence:
            continue

        registry_instruction = build_registry_instruction([entity_name])
        prompt_b = TRIPLE_EXTRACTION_META_PROMPT.format(
            ontology=ontology, registry_instruction=registry_instruction, source_type="Image",
            page_num=str(random.randint(1, 30)), text_input=caption_sentence,
        )
        raw_b = call_model(prompt_b, tokenizer, model)
        triples_b = extract_json_array(raw_b)
        if triples_b is None:
            print(f"[SKIP pair #{i}] invalid JSON for image pass")
            continue
        for t in triples_b:
            if isinstance(t, dict) and "subject" in t:
                t["subject"] = entity_name

        append_jsonl({"system": "Translate the user text into an TTL graph based on the TetraOnto ontology.",
                      "prompt": prompt_a, "triples": triples_a}, output_path)
        append_jsonl({"system": "Translate the user text into an TTL graph based on the TetraOnto ontology.",
                      "prompt": prompt_b, "triples": triples_b}, output_path)
        print(f"[OK pair #{i}] entity='{entity_name}'", flush=True)


def append_jsonl(record, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonl_record(record), ensure_ascii=False) + "\n")


def count_existing_examples(path, prefix_tag):
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if prefix_tag in line:
                count += 1
    return count

def main():
    ontology = load_ontology(ONTOLOGY_TTL_PATH)
    tokenizer, model = load_model()

    general_done = count_existing_examples(OUTPUT_JSONL, '"page_num"') if False else 0
    # simple resume: count lines already written, assume all general first
    existing_lines = 0
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            existing_lines = sum(1 for _ in f)

    print(f"[RESUME] {existing_lines} examples already in file, skipping ahead", flush=True)

    if existing_lines < N_GENERAL_EXAMPLES:
        generate_general_examples(tokenizer, model, ontology, N_GENERAL_EXAMPLES, OUTPUT_JSONL, start_from=existing_lines)

    existing_pairs = count_existing_examples(OUTPUT_JSONL, "Source: Image")
    generate_consistency_pairs(tokenizer, model, ontology, N_CONSISTENCY_PAIRS, OUTPUT_JSONL, start_from=existing_pairs)

    print(f"\n[DONE] Generation complete. File: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()