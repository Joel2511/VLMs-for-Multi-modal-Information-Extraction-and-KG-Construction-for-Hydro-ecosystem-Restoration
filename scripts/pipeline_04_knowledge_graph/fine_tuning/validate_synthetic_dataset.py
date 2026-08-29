import json
import re

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")

ONTOLOGY_COMPACT_PATH = os.path.join(PROJECT_ROOT, "data/ontology/TetraOnto_compact.txt")
INPUT_JSONL = os.path.join(PROJECT_ROOT, "scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/1.dataset_generation/synthetic_dataset.jsonl")
OUTPUT_CLEAN_JSONL = os.path.join(PROJECT_ROOT, "scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/2.validation/synthetic_dataset_clean.jsonl")
OUTPUT_REPORT = os.path.join(PROJECT_ROOT, "scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/2.validation/validation_report.txt")

MIN_VALID_RATIO = 0.5


def parse_ontology(path):
    classes = set()
    predicates = set()
    section = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== CLASSES"):
                section = "classes"
                continue
            if line.startswith("=== OBJECT PROPERTIES") or line.startswith("=== DATATYPE PROPERTIES"):
                section = "predicates"
                continue
            if not line.startswith(":"):
                continue
            m = re.match(r"^:(\S+)", line)
            if not m:
                continue
            name = m.group(1)
            if section == "classes":
                classes.add(name.lower())
            elif section == "predicates":
                predicates.add(name.lower())
    return classes, predicates


def is_placeholder(value, classes, predicates):
    v = value.strip().lower()
    return v in classes or v in predicates


def validate_example(example_messages, classes, predicates):
    assistant_content = example_messages[-1]["content"]
    try:
        triples = json.loads(assistant_content)
    except json.JSONDecodeError:
        return None, 0, 0

    if not isinstance(triples, list):
        return None, 0, 0

    clean_triples = []
    total = len(triples)
    for t in triples:
        if not isinstance(t, dict):
            continue
        subj = str(t.get("subject", "")).strip()
        pred = str(t.get("predicate", "")).strip()
        obj = str(t.get("object", "")).strip()

        if not subj or not pred or not obj:
            continue
        if pred.lower() not in predicates:
            continue
        if is_placeholder(subj, classes, predicates):
            continue
        if is_placeholder(obj, classes, predicates):
            continue

        clean_triples.append({"subject": subj, "predicate": pred, "object": obj})

    return clean_triples, len(clean_triples), total


def main():
    classes, predicates = parse_ontology(ONTOLOGY_COMPACT_PATH)
    print(f"[ONTOLOGY] {len(classes)} classes, {len(predicates)} predicates loaded")

    kept, dropped_examples, dropped_triples, total_triples = 0, 0, 0, 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_CLEAN_JSONL, "w", encoding="utf-8") as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record["messages"]

            clean_triples, n_clean, n_total = validate_example(messages, classes, predicates)
            total_triples += n_total

            if clean_triples is None:
                dropped_examples += 1
                continue

            if n_total == 0:
                messages[-1]["content"] = "[]"
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                kept += 1
                continue

            ratio = n_clean / n_total if n_total > 0 else 0
            dropped_triples += (n_total - n_clean)

            if ratio < MIN_VALID_RATIO or n_clean == 0:
                dropped_examples += 1
                continue

            messages[-1]["content"] = json.dumps(clean_triples, ensure_ascii=False)
            f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            kept += 1

    report = (
        f"Total examples processed lines: {line_num}\n"
        f"Examples kept: {kept}\n"
        f"Examples dropped (bad JSON or below {MIN_VALID_RATIO*100:.0f}% valid): {dropped_examples}\n"
        f"Total triples seen: {total_triples}\n"
        f"Total triples dropped (bad predicate/placeholder): {dropped_triples}\n"
    )
    print(report)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()