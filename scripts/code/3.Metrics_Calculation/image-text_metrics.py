import pandas as pd
import json
import math
import os

pdf_bases = [
    "fichier",
    "2025 12 OFB - FICHE TECHNIQUE Définition, identification, préservation et restauration des zones de sources des cours d’eau en Bretagne et Pays de la Loire",
    "dossier_unio_crassus_zorn_v4_19_11_2024",
    "Rapport-stage-M2-Floriane",
    "reconstituer-rhin-disparu-eca2fa4191154675939c8b4e07593bb7"
]

# Compute per-image retrieval metrics for one model against ground-truth relevance labels
def evaluate_one_method(csv_path, gt_data, method_name):
    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: {method_name} results CSV not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print(f"Error: {method_name} results CSV is empty.")
        return None

    df = df[df["Image_Name"].isin(gt_data.keys())].copy()

    if len(df) == 0:
        print(f"Check Image Names: No overlap between {method_name} CSV and GT JSON.")
        return None

    total_recall_5 = 0
    total_p1 = 0
    total_mrr = 0
    total_ndcg5 = 0
    evaluated_count = 0

    for _, row in df.iterrows():
        img_name = row["Image_Name"]

        if img_name not in gt_data:
            continue

        gt_blocks = gt_data.get(img_name, [])
        if not gt_blocks:
            continue

        retrieved = []
        scores = []
        for i in range(1, 6):
            b_col, s_col = f"Top_{i}_Block", f"Top_{i}_Score"
            if b_col in row and pd.notna(row[b_col]):
                retrieved.append(row[b_col])
                scores.append(row[s_col])

        if not retrieved:
            continue

        evaluated_count += 1

        found_at_rank = None
        for rank, block in enumerate(retrieved, 1):
            if block in gt_blocks:
                found_at_rank = rank
                break

        if found_at_rank is not None:
            if found_at_rank == 1:
                total_p1 += 1
            total_mrr += (1.0 / found_at_rank)

        num_relevant_retrieved = len(set(retrieved) & set(gt_blocks))
        total_recall_5 += (num_relevant_retrieved / len(gt_blocks))

        dcg = 0
        for rank, block in enumerate(retrieved, 1):
            rel = 1 if block in gt_blocks else 0
            dcg += rel / math.log2(rank + 1)

        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gt_blocks), 5) + 1))
        total_ndcg5 += (dcg / idcg)



    if evaluated_count == 0:
        print(f"No valid evaluated rows found for {method_name}.")
        return None

    return {
        "Method": method_name,
        "Total Queries Matched in GT": evaluated_count,
        "Recall@5": round(total_recall_5 / evaluated_count, 4), # Recall@5: fraction of ground-truth relevant blocks recovered within the top-5 results
        "Precision@1": round(total_p1 / evaluated_count, 4),
        "MRR": round(total_mrr / evaluated_count, 4), # MRR: reciprocal rank of the first relevant result within the top-5 (0 if not found)
        "nDCG@5": round(total_ndcg5 / evaluated_count, 4), # nDCG@5: binary-relevance normalized discounted cumulative gain, capped at rank 5
    }


def print_method_block(name, metrics):
    if metrics is None:
        return
    print(name)
    print("-" * 40)
    print(f"Total Queries Matched in GT: {metrics['Total Queries Matched in GT']}")
    print(f"Recall@5:     {metrics['Recall@5']}")
    print(f"Precision@1:  {metrics['Precision@1']}")
    print(f"MRR:          {metrics['MRR']}")
    print(f"nDCG@5:       {metrics['nDCG@5']}") 
    print("-" * 40)


def compare_results(clip_metrics, colpali_metrics, colqwen_metrics,
                    dse_metrics, vidore_metrics, nemotron_metrics, pdf_base):
    print("\n" + "=" * 70)
    print(f"      SCIENTIFIC EVALUATION: SINGLE-VECTOR VS MULTI-VECTOR VS GT on {pdf_base}")
    print("=" * 70)

    if all(m is None for m in [clip_metrics, colpali_metrics, colqwen_metrics, dse_metrics, vidore_metrics, nemotron_metrics]):
        print("All evaluations failed.")
        print("=" * 70 + "\n")
        return

    print("\nSINGLE-VECTOR FAMILY (CLIP / DSE-QWEN2 / VDR-2B-MULTI-V1)")
    print("-" * 70)
    print_method_block("CLIP", clip_metrics)
    print_method_block("DSE-QWEN2", dse_metrics)
    print_method_block("VIDORE-VDR", vidore_metrics)

    print("\nMULTI-VECTOR FAMILY (COLPALI / COLQWEN / NEMOTRON)")
    print("-" * 70)
    print_method_block("COLPALI", colpali_metrics)
    print_method_block("COLQWEN", colqwen_metrics)
    print_method_block("NEMOTRON", nemotron_metrics)

    all_metrics = {
        "CLIP": clip_metrics,
        "ColPali": colpali_metrics,
        "ColQwen": colqwen_metrics,
        "DSE": dse_metrics,
        "ViDoRe": vidore_metrics,
        "Nemotron": nemotron_metrics,
    }
    all_metrics = {k: v for k, v in all_metrics.items() if v is not None}

    if len(all_metrics) > 1:
        print("\nOVERALL COMPARISON (ALL MODELS)")
        print("-" * 40)

        metrics_to_compare = ["Recall@5", "Precision@1", "MRR", "nDCG@5"]

        for metric in metrics_to_compare:
            values = {k: v[metric] for k, v in all_metrics.items()}
            best_val = max(values.values())
            winners = [k for k, v in values.items() if v == best_val]
            winner = "/".join(winners)
            print(f"{metric}: {winner}")

        print("-" * 40)
        print("Evaluation Complete.")

    print("=" * 70 + "\n")

# Final ranking prioritizes macro-averaged nDCG@5, with MRR as tiebreaker
def compute_and_print_macro_averages(aggregated_data):
    print("\n" + "=" * 80)
    print("GLOBAL LEADERBOARD: ARITHMETIC MACRO-AVERAGES ACROSS ALL FILES")
    print("=" * 80)

    summary_list = []
    metrics_to_average = ["Recall@5", "Precision@1", "MRR", "nDCG@5"]

    for model_name, files_list in aggregated_data.items():
        if not files_list:
            continue
        
        num_files = len(files_list)
        model_summary = {"Method": model_name, "Processed Files": num_files}
        
        for metric in metrics_to_average:
            total_sum = sum(f[metric] for f in files_list)
            model_summary[metric] = round(total_sum / num_files, 4)
            
        summary_list.append(model_summary)

    if not summary_list:
        print("No macro-averaged data available.")
        print("=" * 80 + "\n")
        return

    df_summary = pd.DataFrame(summary_list)
    df_summary = df_summary.sort_values(by=["nDCG@5", "MRR"], ascending=False).reset_index(drop=True)
    
    print(df_summary.to_string(index=False))
    print("=" * 80)
    
    print(f"\nOVERALL WINNER BASED ON GLOBAL MACRO-NDCG@5: {df_summary.iloc[0]['Method']}")
    print("=" * 80 + "\n")


def main():
    base_output = "/home2020/home/icube/janil/Rhine_proj/outputs/Image_text_matching"
    
    macro_registry = {
        "CLIP": [], "COLPALI": [], "COLQWEN": [], 
        "DSE-QWEN2": [], "VIDORE-VDR": [], "NEMOTRON": []
    }

    for pdf_base in pdf_bases:
        gt_json_path = f"/home2020/home/icube/janil/Rhine_proj/outputs/json_results/{pdf_base}/GT_Img_text_matching_{pdf_base}.json"
        clip_results_csv_path    = os.path.join(base_output, "clip_results",         pdf_base, "clip_raw_matrix.csv")
        colpali_results_csv_path = os.path.join(base_output, "colpali_results",      pdf_base, "colpali_raw_matrix.csv")
        colqwen_results_csv_path = os.path.join(base_output, "colqwen_results",      pdf_base, "colqwen_raw_matrix.csv")
        dse_results_csv_path      = os.path.join(base_output, "dse_qwen2_results",    pdf_base, "dse_qwen2_raw_matrix.csv")
        vidore_results_csv_path   = os.path.join(base_output, "vidore_results",       pdf_base, "vidore_raw_matrix.csv")
        nemotron_results_csv_path = os.path.join(base_output, "nemotron_results",     pdf_base, "nemotron_raw_matrix.csv")

        if not os.path.exists(gt_json_path):
            print(f"CRITICAL ERROR: GT JSON not found at {gt_json_path}")
            continue

        with open(gt_json_path, "r") as f:
            gt_data = json.load(f)

        clip_metrics     = evaluate_one_method(clip_results_csv_path,     gt_data, "CLIP")
        colpali_metrics  = evaluate_one_method(colpali_results_csv_path,  gt_data, "COLPALI")
        colqwen_metrics  = evaluate_one_method(colqwen_results_csv_path,  gt_data, "COLQWEN")
        dse_metrics      = evaluate_one_method(dse_results_csv_path,      gt_data, "DSE-QWEN2")
        vidore_metrics   = evaluate_one_method(vidore_results_csv_path,   gt_data, "VIDORE-VDR")
        nemotron_metrics = evaluate_one_method(nemotron_results_csv_path, gt_data, "NEMOTRON")

        compare_results(clip_metrics, colpali_metrics, colqwen_metrics,
                        dse_metrics, vidore_metrics, nemotron_metrics, pdf_base)

        if clip_metrics: macro_registry["CLIP"].append(clip_metrics)
        if colpali_metrics: macro_registry["COLPALI"].append(colpali_metrics)
        if colqwen_metrics: macro_registry["COLQWEN"].append(colqwen_metrics)
        if dse_metrics: macro_registry["DSE-QWEN2"].append(dse_metrics)
        if vidore_metrics: macro_registry["VIDORE-VDR"].append(vidore_metrics)
        if nemotron_metrics: macro_registry["NEMOTRON"].append(nemotron_metrics)

    compute_and_print_macro_averages(macro_registry)


if __name__ == "__main__":
    main()