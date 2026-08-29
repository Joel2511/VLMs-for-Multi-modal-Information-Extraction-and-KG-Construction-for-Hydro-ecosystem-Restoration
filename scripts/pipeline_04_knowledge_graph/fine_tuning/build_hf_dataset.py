import json
from datasets import Dataset

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get(
        "RHINE_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[3]),
    )
).expanduser().resolve()

HF_MODELS_DIR = Path(
    os.environ.get("HF_MODELS_DIR", str(PROJECT_ROOT / "hf_models"))
).expanduser().resolve()

FINE_TUNING_OUTPUT_ROOT = Path(
    os.environ.get(
        "FINE_TUNING_OUTPUT_ROOT",
        str(PROJECT_ROOT / "outputs" / "runs" / "fine_tuning"),
    )
).expanduser().resolve()


INPUT_JSONL = FINE_TUNING_OUTPUT_ROOT / "synthetic_dataset_clean.jsonl"
OUTPUT_DATASET_DIR = FINE_TUNING_OUTPUT_ROOT / "hf_dataset"


def main():
    OUTPUT_DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
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