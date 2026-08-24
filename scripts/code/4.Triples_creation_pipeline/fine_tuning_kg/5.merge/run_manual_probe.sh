#!/usr/bin/env bash

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"

set -euo pipefail

# EDIT: confirm this is your real merged model path
# MODEL_DIR="$RHINE_PROJECT_ROOT/hf_models/Qwen2.5-14B-Instruct" #baseline
MODEL_DIR="$RHINE_PROJECT_ROOT/scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/5.merge/Qwen2.5-14B-populate-ontology-merged" #finetuned

ONTOLOGY_PATH="$RHINE_PROJECT_ROOT/data/ontology/TetraOnto_compact.txt"

# EDIT: base dir under which your 5 dossiers' caption JSONs live
CAPTIONS_BASE="$RHINE_PROJECT_ROOT/outputs/5th_ablation_Image_captioning_prompt_engineering/nvila_p3_contrast_captions"

# How many captions to sample per document
N_PER_DOC=3

python3 - "$MODEL_DIR" "$ONTOLOGY_PATH" "$CAPTIONS_BASE" "$N_PER_DOC" <<'PYEOF'
import json, sys, re, glob, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR, ONTOLOGY_PATH, CAPTIONS_BASE, N_PER_DOC = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

ontology = open(ONTOLOGY_PATH, encoding="utf-8").read()

# Extract valid predicate names from the ontology file (":predName  [EN:predName]  domain:...")
valid_predicates = set(re.findall(r'^:(\w+)\s+\[EN:', ontology, flags=re.MULTILINE))
print(f"[ONTOLOGY] {len(valid_predicates)} predicates parsed for compliance check\n")

# Find all caption JSON files across the 5 dossiers
caption_files = sorted(glob.glob(os.path.join(CAPTIONS_BASE, "*", "image_captions_p3_contrast.json")))
if not caption_files:
    caption_files = sorted(glob.glob(os.path.join(CAPTIONS_BASE, "**", "*.json"), recursive=True))
print(f"[FOUND] {len(caption_files)} caption files:")
for f in caption_files:
    print("  -", f)
print()

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=bnb, device_map="auto")

total_triples = 0
total_bad_predicates = 0

for cf in caption_files:
    dossier = os.path.basename(os.path.dirname(cf))
    captions = json.load(open(cf, encoding="utf-8"))
    sample_captions = list(captions.items())[:N_PER_DOC]

    print("#" * 70)
    print(f"DOSSIER: {dossier}  ({cf})")
    print("#" * 70)

    for img_name, caption in sample_captions:
        prompt = f"""Tu es un expert en extraction de connaissances pour un graphe de connaissances scientifique sur l'hydromorphologie fluviale et la restauration de cours d'eau.

ONTOLOGIE DISPONIBLE (utilise UNIQUEMENT ces classes et proprietes) :
{ontology}

CONTENU A ANALYSER :
Source: Image | Page: 1
{caption}

MISSION : Extrais les faits de ce texte sous forme de triplets JSON.
Reponds UNIQUEMENT avec un tableau JSON valide contenant subject, subject_type, predicate, object."""

        messages = [
            {"role": "system", "content": "Translate the user text into an TTL graph based on the TetraOnto ontology."},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False, pad_token_id=tok.eos_token_id)
        result = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print("=" * 60)
        print("IMAGE:", img_name)
        print("CAPTION:", caption[:150].replace("\n", " "))
        print("RAW OUTPUT:", result)

        # Automated predicate-compliance check (does NOT check factual grounding)
        try:
            json_match = re.search(r'\[.*\]', result, flags=re.DOTALL)
            triples = json.loads(json_match.group(0)) if json_match else []
        except Exception as e:
            triples = None
            print(f"[PARSE ERROR] could not parse JSON: {e}")

        if triples:
            for t in triples:
                pred = t.get("predicate", "")
                total_triples += 1
                status = "OK" if pred in valid_predicates else "NOT IN ONTOLOGY"
                if status != "OK":
                    total_bad_predicates += 1
                print(f"    predicate check: '{pred}' -> {status}")
        print()

print("=" * 70)
print(f"SUMMARY: {total_triples} triples generated, {total_bad_predicates} used out-of-ontology predicates")
print("NOTE: this only checks predicate names against the ontology.")
print("It does NOT check whether the object/fact is actually grounded in the caption text.")
print("You still need to manually eyeball each RAW OUTPUT against its CAPTION for hallucinated content.")
PYEOF