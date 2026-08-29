import json
from datasets import Dataset

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")

INPUT_JSONL = os.path.join(PROJECT_ROOT, "scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/2.validation/synthetic_dataset_clean.jsonl")
OUTPUT_DATASET_DIR = os.path.join(PROJECT_ROOT, "scripts/code/4.Triples_creation_pipeline/fine_tuning_kg/3.dataset_prep/hf_dataset")


def main():
    records = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({"raw": json.dumps(obj, ensure_ascii=False)})

    print(f"[INFO] Loaded {len(records)} records")
    dataset = Dataset.from_list(records)
    dataset.save_to_disk(OUTPUT_DATASET_DIR)
    print(f"[DONE] Saved HF dataset to {OUTPUT_DATASET_DIR}")


if __name__ == "__main__":
    main()