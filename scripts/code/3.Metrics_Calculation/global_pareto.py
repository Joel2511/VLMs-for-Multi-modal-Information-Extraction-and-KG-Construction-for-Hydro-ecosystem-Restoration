#!/usr/bin/env python3


import math


# ══════════════════════════════════════════════════════════════
# CONFIGURATION — Edit this block for each ablation
# ══════════════════════════════════════════════════════════════


SCENARIOS = {
    "A": {
        "name":        "Grand Final — Global Pareto Comparison | Scenario A: Production",
        "weights":     {"quality": 0.40, "speed": 0.35, "size": 0.25},
        "sub_weights": {"vqa": 0.33, "enrichment": 0.34, "novelty": 0.33}
    },
    "B": {
        "name":        "Grand Final — Global Pareto Comparison | Scenario B: Oracle",
        "weights":     {"quality": 0.60, "speed": 0.25, "size": 0.15},
        "sub_weights": {"vqa": 0.50, "enrichment": 0.35, "novelty": 0.15}
    }
}


ACTIVE_MODE  = "B"   # ← choose scenario


CONFIG        = SCENARIOS[ACTIVE_MODE]
ABLATION_NAME = CONFIG["name"]
WEIGHTS       = CONFIG["weights"]
SUB_WEIGHTS   = CONFIG["sub_weights"]
CAPTIONING_ROOT = "/home2020/home/icube/janil/Rhine_proj/outputs/Global_pareto_comparison_captions"
METRICS_JSON    = "vqa_results.json"
LOG_ROOT        = "/home2020/home/icube/janil/Rhine_proj/outputs/live_logs/2.Global_output"

# grep -E "Model size|Total runtime" /home2020/home/icube/janil/Rhine_proj/outputs/live_logs/2.0.Global_output/*.txt
OVERRIDES       = {
    # "1st_ablation_pixtral-12b-hf":            {"params_b": 7.0,  "total_sec": 7893},   # 02h11m33s
    # "2nd_ablation_mistral24b":                {"params_b": 12.7, "total_sec": 5360},   # 01h29m20s
    # "2nd_ablation_nvila_lite_8b":             {"params_b": 8.1,  "total_sec": 1501},   # 00h25m01s
    # "2nd_ablation_qwen3vl_30b":               {"params_b": 30.3, "total_sec": 3119},   # 00h51m59s
    # "3rd_ablation_qwen3vl_30b":               {"params_b": 30.3, "total_sec": 2775},   # 00h46m15s
    # "4th_ablation_nvila-lite-8b":             {"params_b": 8.1,  "total_sec": 1753},   # 00h29m13s
    # "5th_ablation_nvila_lite_8b_p3":          {"params_b": 8.1,  "total_sec": 1448},   # 00h24m08s
    # "6th_ablation_nvila_lite-8b_vcd_alpha05": {"params_b": 8.1,  "total_sec": 3078},   # 00h51m18s
}


import os, json, re


def _tokenize(name):
    """Split a filename or model name into lowercase tokens, normalising separators."""
    name = os.path.splitext(name)[0]          # drop .txt
    name = name.lower()
    name = re.sub(r'[-_]+', ' ', name)        # dashes and underscores → space
    # NEW: split between letters and digits so 'mistral24b' → 'mistral 24b'
    name = re.sub(r'(?<=\d)(?=\D)|(?<=\D)(?=\d)', ' ', name)
    return set(name.split())


def _parse_summary_block(text):
    """
    Extract structured fields from the PIPELINE COMPLETE summary block.
    Returns a dict of normalised key→value strings, e.g.:
      {"generator_model": "nvila-lite-8b",
       "retrieval_model": "vidore",
       "icd_cd_alpha":    "alpha05"}
    """
    fields = {}
    # Generator model
    m = re.search(r"Generator model\s*:\s*(.+)", text)
    if m:
        fields["generator_model"] = m.group(1).strip().lower()
    # Retrieval model
    m = re.search(r"Retrieval model\s*:\s*(.+)", text)
    if m:
        fields["retrieval_model"] = m.group(1).strip().lower()
    # ICD cd_alpha  →  normalise 0.5 → "alpha05", 1.0 → "alpha10", etc.
    m = re.search(r"ICD cd_alpha\s*:\s*([\d.]+)", text)
    if m:
        raw = m.group(1).strip()          # e.g. "0.5"
        normalised = "alpha" + raw.replace(".", "")   # "alpha05"
        fields["icd_cd_alpha"] = normalised
    return fields


def find_log_for_model(model_name, log_root):
    """
    Universal log matcher — works for all ablations regardless of naming convention.

    Strategy:
      1. Parse the PIPELINE COMPLETE summary block from every log in log_root.
      2. Build a score for each log: count how many tokens from the JSON model name
         appear in (a) the log's summary field values, or (b) the log's filename.
      3. Return the log with the highest score (ties broken by filename token count).
    """
    model_tokens = _tokenize(model_name)

    best_path  = None
    best_score = -1

    for fname in sorted(os.listdir(log_root)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(log_root, fname)

        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue

        summary = _parse_summary_block(text)
        fname_tokens = _tokenize(fname)

        # Build a flat set of all matchable tokens from this log
        log_token_pool = set(fname_tokens)
        for v in summary.values():
            # tokenise each field value the same way
            log_token_pool |= _tokenize(v)

        score = len(model_tokens & log_token_pool)

        if score > best_score:
            best_score = score
            best_path  = fpath

    return best_path  # None only if log_root is empty


def parse_log(log_path):
    """Extract model size (params_b) and total runtime (seconds) from a captioning log."""
    params_b, total_sec = None, None
    if log_path is None or not os.path.exists(log_path):
        return params_b, total_sec
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    m = re.search(r"Model size\s*:\s*([\d.]+)B", text)
    if m:
        params_b = float(m.group(1))
    m = re.search(r"Total runtime\s*:\s*(\d+)h\s*(\d+)m\s*(\d+)s", text)
    if m:
        total_sec = int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))
    return params_b, total_sec


def load_models_from_folder(captioning_root, metrics_json, log_root, overrides):
    models = []
    excluded = []
    for folder_name in sorted(os.listdir(captioning_root)):
        folder_path = os.path.join(captioning_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        json_path = os.path.join(folder_path, metrics_json)
        if not os.path.exists(json_path):
            print(f"  [SKIP] {folder_name} — no {metrics_json} found")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        log_path = find_log_for_model(folder_name, log_root)
        print(f"  [READ] {folder_name} → {os.path.basename(log_path) if log_path else 'NO LOG FOUND'}")
        params_b, total_sec = parse_log(log_path)

        ov = overrides.get(folder_name, {})
        if "params_b" in ov:
            params_b = ov["params_b"]
        if "total_sec" in ov:
            total_sec = ov["total_sec"]

        # Thesis-level guardrail: a model with unresolved size or runtime is
        # excluded from ranking rather than defaulted to placeholder values.
        # A defaulted params_b=0.0 or total_sec=1 would make an unmatched
        # model appear artificially fastest/smallest, silently corrupting
        # the speed and size sub-scores. Add a manual entry to OVERRIDES
        # for this folder_name to include it instead of excluding it.
        if params_b is None or total_sec is None:
            print(f"  [EXCLUDED] {folder_name} — missing params_b or total_sec "
                  f"and no override supplied; excluded to prevent a distorted score.")
            excluded.append(folder_name)
            continue

        total_imgs = d.get("total_images_evaluated", 105)
        models.append({
            "name":        folder_name,
            "params_b":    params_b,
            "total_sec":   total_sec,
            "total_imgs":  total_imgs,
            "nli_novelty": d.get("global_mean_nli_novelty", 0.0),
            "vqa_p1":      d.get("global_mean_vqa_p1_score", 0.0),
            "vqa_p2":      d.get("global_mean_vqa_p2_score") or None,
            "enrichment":  d.get("global_verified_enrichment_rate", 0.0),
        })

    if excluded:
        print(f"\n  NOTE: {len(excluded)} model(s) excluded from ranking due to "
              f"unresolved metadata: {excluded}\n")

    return models


# ══════════════════════════════════════════════════════════════
# COMPUTATION — Do not edit below this line
# ══════════════════════════════════════════════════════════════


def compute_combined_vqa(vqa_p1, vqa_p2):
    if vqa_p2 is not None:
        return (vqa_p1 + vqa_p2) / 2
    return vqa_p1


def minmax_norm(values, invert=False):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    normed = [(v - lo) / (hi - lo) for v in values]
    if invert:
        normed = [1.0 - n for n in normed]
    return normed


def compute_pareto(models, weights):
    for m in models:
        m["s_per_img"]    = m["total_sec"] / m["total_imgs"]
        m["combined_vqa"] = compute_combined_vqa(m["vqa_p1"], m["vqa_p2"])
        m["quality_raw"]  = (
            SUB_WEIGHTS["vqa"]        * m["combined_vqa"] +
            SUB_WEIGHTS["enrichment"] * m["enrichment"] +
            SUB_WEIGHTS["novelty"]    * m["nli_novelty"]
        )

    q_norm = minmax_norm([m["quality_raw"] for m in models], invert=False)
    s_norm = minmax_norm([m["s_per_img"]   for m in models], invert=True)
    z_norm = minmax_norm([m["params_b"]    for m in models], invert=True)

    for i, m in enumerate(models):
        m["q_norm"] = q_norm[i]
        m["s_norm"] = s_norm[i]
        m["z_norm"] = z_norm[i]
        m["pareto_score"] = (
            weights["quality"] * m["q_norm"] +
            weights["speed"]   * m["s_norm"] +
            weights["size"]    * m["z_norm"]
        )

    models.sort(key=lambda x: x["pareto_score"], reverse=True)
    return models


# ══════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════


def print_table(models, ablation_name, weights):
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  PARETO RESULTS — {ablation_name}")
    print(f"  Weights: Quality={weights['quality']:.0%}  Speed={weights['speed']:.0%}  Size={weights['size']:.0%}")
    print(f"Scenario A (Production): Quality = 0.33*VQA + 0.34*Enrichment + 0.33*NLI  | Weights: Q=40% S=35% Z=25%")
    print(f"Scenario B (Oracle):     Quality = 0.50*VQA + 0.35*Enrichment + 0.15*NLI  | Weights: Q=60% S=25% Z=15%")
    print(sep)

    header = (
        f"  {'Rank':<5} {'Model':<25} {'Params':>7} {'s/img':>7} "
        f"{'NLI Nov':>8} {'Comb VQA':>9} {'Enrich':>8} "
        f"{'Q_raw':>7} {'Q_norm':>7} {'S_norm':>7} {'Z_norm':>7} "
        f"{'PARETO':>8}"
    )
    print(header)
    print("-" * 120)

    for rank, m in enumerate(models, 1):
        print(
            f"  {rank:<5} "
            f"{m['name']:<25} "
            f"{m['params_b']:>6.1f}B "
            f"{m['s_per_img']:>7.1f} "
            f"{m['nli_novelty']:>8.3f} "
            f"{m['combined_vqa']:>9.3f} "
            f"{m['enrichment']:>8.3f} "
            f"{m['quality_raw']:>7.3f} "
            f"{m['q_norm']:>7.3f} "
            f"{m['s_norm']:>7.3f} "
            f"{m['z_norm']:>7.3f} "
            f"{m['pareto_score']:>8.4f}"
        )

    print("=" * 120)
    winner = models[0]
    print(f"\n Global winner: {winner['name']}  |  Pareto Score: {winner['pareto_score']:.4f}")
    print(f"     Quality: {winner['quality_raw']:.3f}  |  s/img: {winner['s_per_img']:.1f}  |  Params: {winner['params_b']:.1f}B")
    print(f"{sep}\n")


if __name__ == "__main__":
    MODELS = load_models_from_folder(CAPTIONING_ROOT, METRICS_JSON, LOG_ROOT, OVERRIDES)
    if not MODELS:
        print(f"ERROR: No models found in {CAPTIONING_ROOT}.")
    else:
        print(f"  Found {len(MODELS)} model(s): {[m['name'] for m in MODELS]}")
        results = compute_pareto(MODELS, WEIGHTS)
        print_table(results, ABLATION_NAME, WEIGHTS)