import os
import json
import random
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, OWL, SKOS

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")
random.seed(42)

# ─────────────────────────────────────────────
# PATHS — change BASE_TTL_OUTPUT for other runs
# ─────────────────────────────────────────────
RUN_LABEL        = "finetuned"   # change to "baseline"/"finetuned" for the second run
MERGED_FILENAME  = "merged_knowledge_graph.ttl" if RUN_LABEL == "baseline" else "ft_merged_knowledge_graph.ttl"

BASE_TTL_OUTPUT  = os.path.join(PROJECT_ROOT, "outputs/triples_baseline") if RUN_LABEL == "baseline" \
                   else os.path.join(PROJECT_ROOT, "outputs/triples_fine_tuned")
METRICS_OUT_JSON = os.path.join(BASE_TTL_OUTPUT, f"kg_metrics_{RUN_LABEL}.json")
METRICS_OUT_TXT  = os.path.join(BASE_TTL_OUTPUT, f"kg_metrics_{RUN_LABEL}.txt")
PRECISION_SAMPLE = os.path.join(BASE_TTL_OUTPUT, f"precision_sample_{RUN_LABEL}.txt")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def uri_nodes(g: Graph) -> set:
    """All unique URI nodes (subjects + URI objects) — excludes literals."""
    return set(s for s, p, o in g) | set(o for s, p, o in g if isinstance(o, URIRef))


def avg_degree(g: Graph) -> float:
    """Average node degree over URI nodes only: 2*|object-property edges| / |V|."""
    nodes = uri_nodes(g)
    if not nodes:
        return 0.0
    n_object_edges = sum(1 for s, p, o in g if isinstance(o, URIRef))
    return (2 * n_object_edges) / len(nodes)


def named_individuals(g: Graph) -> set:
    """All OWL NamedIndividuals — entities declared by the pipeline."""
    return set(s for s, p, o in g.triples((None, RDF.type, OWL.NamedIndividual)))


def count_skos_notes(g: Graph) -> int:
    """Count demoted predicates stored as skos:note."""
    return sum(1 for _ in g.triples((None, SKOS.note, None)))


def count_tetra_predicates(g: Graph) -> int:
    TETRA_NS = "http://www.semanticweb.org/fethi/ontologies/2024/1/TetraOnto.owl#"
    EXCLUDED = {
        TETRA_NS + "dataSource",
        TETRA_NS + "hasPageNumber",
        TETRA_NS + "extractedFrom",
    }
    return sum(1 for s, p, o in g
               if str(p).startswith(TETRA_NS) and str(p) not in EXCLUDED)


def cross_modal_edges(g_merged: Graph, text_individuals: set, image_individuals: set) -> int:
    """
    Count edges in the merged graph that connect to a node which was
    resolved (appears in BOTH text and image individual sets).
    These are the edges that prove structural fusion.
    """
    TETRA_NS = "http://www.semanticweb.org/fethi/ontologies/2024/1/TetraOnto.owl#"
    EXCLUDED = {TETRA_NS + "dataSource", TETRA_NS + "hasPageNumber", TETRA_NS + "extractedFrom"}
    resolved = text_individuals & image_individuals
    count = 0
    for s, p, o in g_merged:
        if not str(p).startswith(TETRA_NS) or str(p) in EXCLUDED:
            continue
        s_resolved = s in resolved
        o_resolved = isinstance(o, URIRef) and o in resolved
        if s_resolved or o_resolved:
            count += 1
    return count

def sample_triples_for_precision(g: Graph, n: int = 50) -> list:
    """
    Random sample of n triples for manual Intrinsic Triple Precision check.
    Filters out boilerplate rdf:type / skos:prefLabel triples — only real domain triples.
    """
    TETRA_NS = "http://www.semanticweb.org/fethi/ontologies/2024/1/TetraOnto.owl#"
    PROVENANCE_EXCLUDED = {
        TETRA_NS + "dataSource",
        TETRA_NS + "hasPageNumber",
        TETRA_NS + "extractedFrom",
    }
    domain_triples = [
        (str(s), str(p), str(o))
        for s, p, o in g
        if str(p).startswith(TETRA_NS) and str(p) not in PROVENANCE_EXCLUDED
    ]
    sample_size = min(n, len(domain_triples))
    return random.sample(domain_triples, sample_size) if domain_triples else []


# ─────────────────────────────────────────────
# PER-PDF METRIC COMPUTATION
# ─────────────────────────────────────────────
def compute_metrics_for_pdf(pdf_name: str, ttl_dir: str) -> dict:
    text_ttl   = os.path.join(ttl_dir, pdf_name, "texts.ttl")
    image_ttl  = os.path.join(ttl_dir, pdf_name, "images.ttl")
    merged_ttl = os.path.join(ttl_dir, pdf_name, MERGED_FILENAME)

    # Load graphs — safe fallback to empty graph if file missing
    g_text   = Graph()
    g_image  = Graph()
    g_merged = Graph()

    if os.path.exists(text_ttl):
        g_text.parse(text_ttl, format="turtle")
    else:
        print(f"  [WARN] texts.ttl not found for {pdf_name}")

    if os.path.exists(image_ttl):
        g_image.parse(image_ttl, format="turtle")
    else:
        print(f"  [WARN] images.ttl not found for {pdf_name}")

    if os.path.exists(merged_ttl):
        g_merged.parse(merged_ttl, format="turtle")
    else:
        print(f"  [WARN] merged_knowledge_graph.ttl not found for {pdf_name}")

    # ── Counts ──
    text_triple_count   = len(g_text)
    image_triple_count  = len(g_image)
    merged_triple_count = len(g_merged)

    text_individuals  = named_individuals(g_text)
    image_individuals = named_individuals(g_image)
    merged_individuals = named_individuals(g_merged)

    new_from_images   = image_individuals - text_individuals   # entities only images added
    cross_modal_nodes = text_individuals & image_individuals   # entities present in both

    # ── Metric 1: Schema Conformance Rate ──
    # conformant = TETRA predicates used; demoted = skos:note triples
    # Total attempted = conformant + demoted
    conformant_count = count_tetra_predicates(g_merged)
    demoted_count    = count_skos_notes(g_merged)
    total_attempted  = conformant_count + demoted_count
    scr = conformant_count / total_attempted if total_attempted > 0 else 0.0

    # ── Metric 2: Average Node Degree Shift ──
    avg_deg_text   = avg_degree(g_text)
    avg_deg_merged = avg_degree(g_merged)
    delta_degree   = avg_deg_merged - avg_deg_text

    # ── Metric 3: Cross-Modal Link Density ──
    cm_edges = cross_modal_edges(g_merged, text_individuals, image_individuals)
    cmld = cm_edges / conformant_count if conformant_count > 0 else 0.0

    # ── Metric 4: Precision sample (for manual review) ──
    precision_sample = sample_triples_for_precision(g_merged, n=50)

    return {
        "pdf": pdf_name,
        "triple_counts": {
            "text_only":    text_triple_count,
            "image_only":   image_triple_count,
            "merged_total": merged_triple_count,
        },
        "entity_counts": {
            "text_individuals":    len(text_individuals),
            "image_individuals":   len(image_individuals),
            "merged_individuals":  len(merged_individuals),
            "new_from_images":     len(new_from_images),
            "cross_modal_nodes":   len(cross_modal_nodes),
        },
        "metrics": {
            "SCR":          round(scr, 4),
            "SCR_pct":      round(scr * 100, 2),
            "demoted":      demoted_count,
            "conformant":   conformant_count,
            "delta_degree": round(delta_degree, 4),
            "avg_deg_text": round(avg_deg_text, 4),
            "avg_deg_merged": round(avg_deg_merged, 4),
            "CMLD":         round(cmld, 4),
            "CMLD_pct":     round(cmld * 100, 2),
            "cm_edges":     cm_edges,
        },
        "precision_sample": precision_sample,
    }


# ─────────────────────────────────────────────
# AGGREGATE ACROSS ALL PDFs
# ─────────────────────────────────────────────
def aggregate_metrics(all_results: list) -> dict:
    """Compute corpus-level averages across all PDFs."""
    if not all_results:
        return {}

    def avg(key_path):
        vals = []
        for r in all_results:
            v = r
            for k in key_path:
                v = v.get(k, 0)
            vals.append(v)
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    _ratios = [r["triple_counts"]["merged_total"] / r["triple_counts"]["text_only"]
               for r in all_results if r["triple_counts"]["text_only"] > 0]
    avg_enrichment_ratio_value = round(sum(_ratios) / len(_ratios), 4) if _ratios else 0.0

    return {
        "total_pdfs":           len(all_results),
        "total_text_triples":   sum(r["triple_counts"]["text_only"]    for r in all_results),
        "total_image_triples":  sum(r["triple_counts"]["image_only"]   for r in all_results),
        "total_merged_triples": sum(r["triple_counts"]["merged_total"] for r in all_results),
        "total_new_entities_from_images": sum(r["entity_counts"]["new_from_images"] for r in all_results),
        "avg_SCR":              avg(["metrics", "SCR"]),
        "avg_delta_degree":     avg(["metrics", "delta_degree"]),
        "avg_CMLD":             avg(["metrics", "CMLD"]),
        "avg_enrichment_ratio": avg_enrichment_ratio_value,
    }


# ─────────────────────────────────────────────
# PRETTY PRINT TO TXT
# ─────────────────────────────────────────────
def format_report(all_results: list, aggregate: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"  KNOWLEDGE GRAPH ENRICHMENT METRICS — {RUN_LABEL.upper()} PIPELINE")
    lines.append("=" * 70)

    for r in all_results:
        m = r["metrics"]
        e = r["entity_counts"]
        t = r["triple_counts"]
        lines.append(f"\n{'─'*60}")
        lines.append(f"  PDF: {r['pdf']}")
        lines.append(f"{'─'*60}")
        lines.append(f"  Triple Counts:")
        lines.append(f"    Text-only triples    : {t['text_only']}")
        lines.append(f"    Image-only triples   : {t['image_only']}")
        lines.append(f"    Total merged RDF triples : {t['merged_total']}")
        lines.append(f"")
        lines.append(f"  Entity Counts:")
        lines.append(f"    Text individuals     : {e['text_individuals']}")
        lines.append(f"    Image individuals    : {e['image_individuals']}")
        lines.append(f"    New from images      : {e['new_from_images']}  ← enrichment")
        lines.append(f"    Cross-modal nodes    : {e['cross_modal_nodes']}  ← fusion")
        lines.append(f"    Merged individuals   : {e['merged_individuals']}")
        lines.append(f"")
        lines.append(f"  ── Metric 1: Schema Conformance Rate (SCR) ──")
        lines.append(f"    Conformant predicates : {m['conformant']}")
        lines.append(f"    Demoted predicates    : {m['demoted']}")
        lines.append(f"    SCR                   : {m['SCR_pct']:.2f}%")
        lines.append(f"")
        lines.append(f"  ── Metric 2: Average Node Degree Shift (Δd̄) ──")
        lines.append(f"    Avg degree (text-only): {m['avg_deg_text']:.4f}")
        lines.append(f"    Avg degree (merged)   : {m['avg_deg_merged']:.4f}")
        lines.append(f"    Δd̄                    : {m['delta_degree']:+.4f}  ← positive = enriched")
        lines.append(f"")
        lines.append(f"  ── Metric 3: Cross-Modal Link Density (CMLD) ──")
        lines.append(f"    Cross-modal edges     : {m['cm_edges']}")
        lines.append(f"    Total merged edges    : {t['merged_total']}")
        lines.append(f"    CMLD                  : {m['CMLD_pct']:.2f}%")
        lines.append(f"")
        lines.append(f"  ── Metric 4: Intrinsic Triple Precision (PT) ──")
        lines.append(f"    Sample size           : {len(r['precision_sample'])} triples")
        lines.append(f"    [MANUAL REVIEW REQUIRED — see precision_sample file]")

    lines.append(f"\n{'='*70}")
    lines.append(f"  CORPUS AGGREGATES ({aggregate['total_pdfs']} PDFs)")
    lines.append(f"{'='*70}")
    lines.append(f"  Total text triples          : {aggregate['total_text_triples']}")
    lines.append(f"  Total image triples         : {aggregate['total_image_triples']}")
    lines.append(f"  Total merged triples        : {aggregate['total_merged_triples']}")
    lines.append(f"  Total new entities (images) : {aggregate['total_new_entities_from_images']}")
    lines.append(f"  Avg SCR                     : {aggregate['avg_SCR']*100:.2f}%")
    lines.append(f"  Avg Δd̄                      : {aggregate['avg_delta_degree']:+.4f}")
    lines.append(f"  Avg CMLD                    : {aggregate['avg_CMLD']*100:.2f}%")
    lines.append(f"{'='*70}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"[METRICS] Scanning: {BASE_TTL_OUTPUT}", flush=True)

    if not os.path.exists(BASE_TTL_OUTPUT):
        print(f"[ERROR] Output directory not found: {BASE_TTL_OUTPUT}")
        return

    pdf_folders = [
        f for f in sorted(os.listdir(BASE_TTL_OUTPUT))
        if os.path.isdir(os.path.join(BASE_TTL_OUTPUT, f))
    ]

    if not pdf_folders:
        print("[ERROR] No PDF subfolders found.")
        return

    print(f"[METRICS] Found {len(pdf_folders)} PDF folders: {pdf_folders}", flush=True)

    all_results = []

    for pdf in pdf_folders:
        print(f"\n[METRICS] Processing: {pdf}", flush=True)
        result = compute_metrics_for_pdf(pdf, BASE_TTL_OUTPUT)
        all_results.append(result)

    aggregate = aggregate_metrics(all_results)

    # ── Extract precision samples BEFORE popping from results ──
    precision_samples = {r["pdf"]: r.get("precision_sample", []) for r in all_results}

    # ── Build precision sample file ──
    all_precision_lines = []
    for r in all_results:
        pdf = r["pdf"]
        sample = precision_samples.get(pdf, [])
        all_precision_lines.append(f"\n{'='*60}")
        all_precision_lines.append(f"PDF: {pdf}  — {len(sample)} triples sampled")
        all_precision_lines.append(f"{'='*60}")
        all_precision_lines.append(f"{'#':<4} {'SUBJECT':<40} {'PREDICATE':<30} {'OBJECT'}")
        all_precision_lines.append(f"{'-'*4} {'-'*40} {'-'*30} {'-'*40}")
        for i, (s, p, o) in enumerate(sample, 1):
            s_short = s.split("#")[-1][:38]
            p_short = p.split("#")[-1][:28]
            o_short = str(o).split("#")[-1][:38]
            all_precision_lines.append(f"{i:<4} {s_short:<40} {p_short:<30} {o_short}")
        all_precision_lines.append(f"\n[REVIEWER]: Mark each triple T (true) / F (false) / U (unsure)")
        all_precision_lines.append(f"[PT = true_count / {len(sample)}]")

    # ── Generate report (needs precision_sample key still intact) ──
    report = format_report(all_results, aggregate)

    # ── NOW pop precision_sample before saving JSON ──
    output_data = {"per_pdf": all_results, "aggregate": aggregate}
    for r in output_data["per_pdf"]:
        r.pop("precision_sample", None)

    # ── Save JSON ──
    with open(METRICS_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] JSON metrics: {METRICS_OUT_JSON}", flush=True)

    # ── Save TXT report ──
    with open(METRICS_OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[SAVED] TXT report  : {METRICS_OUT_TXT}", flush=True)
    print(report)

    # ── Save precision sample ──
    with open(PRECISION_SAMPLE, "w", encoding="utf-8") as f:
        f.write("INTRINSIC TRIPLE PRECISION SAMPLE — Nvila\n")
        f.write("Review each triple and mark T/F/U. PT = true_count / total_sampled\n")
        f.write("\n".join(all_precision_lines))
    print(f"[SAVED] Precision sample: {PRECISION_SAMPLE}", flush=True)

if __name__ == "__main__":
    main()
