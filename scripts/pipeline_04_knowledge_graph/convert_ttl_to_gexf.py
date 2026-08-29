"""
Convert Pipeline 4 RDF/Turtle graph outputs to GEXF.

This utility converts the text-only, image-only, and merged RDF graphs of one
document into GEXF files for graph inspection tools such as Gephi. It preserves
the original RDF filtering, literal-property handling, TBox exclusion, and
text/image datasource attribution logic.

The script supports both baseline and fine-tuned Pipeline 4 graph outputs.
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
from rdflib import Graph, URIRef


# ─── Repository configuration ────────────────────────────────────────────────

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("RHINE_PROJECT_ROOT", str(SCRIPT_ROOT))
).expanduser().resolve()

DEFAULT_BASELINE_INPUT_ROOT = PROJECT_ROOT / "outputs" / "triples_baseline"
DEFAULT_FINE_TUNED_INPUT_ROOT = PROJECT_ROOT / "outputs" / "triples_fine_tuned"

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "runs" / "knowledge_graph" / "gexf"
)

TETRA_NS = "http://www.semanticweb.org/fethi/ontologies/2024/1/TetraOnto.owl#"
DS_PRED = URIRef(TETRA_NS + "extractedFrom")

SKIP_PREDICATES = [
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    TETRA_NS + "dataSource",
    TETRA_NS + "hasPageNumber",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Pipeline 4 RDF/Turtle graph files to GEXF for one "
            "document."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "fine_tuned"],
        default="fine_tuned",
        help=(
            "Pipeline 4 graph configuration to convert. "
            "Default: fine_tuned."
        ),
    )

    parser.add_argument(
        "--document",
        required=True,
        help=(
            "Document folder name under --input-root. Use the exact folder "
            "name used by Pipeline 4 graph output."
        ),
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing Pipeline 4 per-document TTL outputs. "
            "Defaults to archived baseline or fine-tuned results according "
            "to --mode."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root directory for newly generated GEXF files. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )

    return parser.parse_args()


def clean_label(uri_str: str) -> str:
    label = uri_str.strip()
    label = re.sub(r"[\n\r\t]", " ", label)
    label = label.split("#")[-1].split("/")[-1]
    return label


def get_datasource(graph: Graph, subject: URIRef) -> str:
    values = set()

    for _, _, obj in graph.triples((subject, DS_PRED, None)):
        value_lower = str(obj).lower()

        if (
            "image" in value_lower
            or "vqa" in value_lower
            or "caption" in value_lower
        ):
            values.add("Image")
        else:
            values.add("Text")

    if not values:
        return "unknown"

    if len(values) == 1:
        return values.pop()

    return "Text+Image"


def ttl_files_for_mode(mode: str) -> dict[str, str]:
    if mode == "baseline":
        return {
            "texts": "texts.ttl",
            "images": "images.ttl",
            "merged_knowledge_graph": "merged_knowledge_graph.ttl",
        }

    return {
        "texts": "texts.ttl",
        "images": "images.ttl",
        "ft_merged_knowledge_graph": "ft_merged_knowledge_graph.ttl",
    }


def convert_ttl_to_gexf(ttl_path: Path, output_path: Path) -> None:
    print(f"\n[INFO] Parsing: {ttl_path}")

    graph = Graph()
    graph.parse(str(ttl_path), format="turtle")

    print(f"[INFO] Loaded {len(graph)} triples")

    # First pass: collect text/image datasource per subject URI.
    uri_datasource = {}

    for subject in set(graph.subjects()):
        if isinstance(subject, URIRef):
            label = clean_label(str(subject))
            uri_datasource[label] = get_datasource(graph, subject)

    # Collect literal datatype-property attributes per subject URI.
    uri_attributes = {}

    for subject, predicate, obj in graph:
        if isinstance(obj, URIRef):
            continue

        predicate_string = str(predicate)

        if any(
            predicate_string.startswith(skip_predicate)
            for skip_predicate in SKIP_PREDICATES
        ):
            continue

        subject_label = clean_label(str(subject))
        attribute_key = clean_label(predicate_string)
        attribute_value = str(obj).replace("\n", " ").strip()[:200]

        uri_attributes.setdefault(subject_label, {})

        existing_value = uri_attributes[subject_label].get(
            attribute_key,
            "",
        )

        uri_attributes[subject_label][attribute_key] = (
            f"{existing_value}; {attribute_value}"
            if existing_value
            else attribute_value
        )

    nx_graph = nx.MultiDiGraph()

    # Identify TBox meta-elements declared inside the RDF graph.
    rdf_type = URIRef(
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    )

    owl_class = URIRef("http://www.w3.org/2002/07/owl#Class")
    owl_object_property = URIRef(
        "http://www.w3.org/2002/07/owl#ObjectProperty"
    )
    owl_datatype_property = URIRef(
        "http://www.w3.org/2002/07/owl#DatatypeProperty"
    )
    owl_annotation_property = URIRef(
        "http://www.w3.org/2002/07/owl#AnnotationProperty"
    )

    property_uris = set()
    class_uris = set()

    for subject, predicate, obj in graph.triples((None, rdf_type, None)):
        if obj in (
            owl_object_property,
            owl_datatype_property,
            owl_annotation_property,
        ):
            property_uris.add(subject)

        elif obj == owl_class:
            class_uris.add(subject)

    # Store RDF class membership as a node attribute, not an edge.
    for subject, predicate, obj in graph.triples((None, rdf_type, None)):
        if obj in class_uris:
            subject_label = clean_label(str(subject))
            class_label = clean_label(str(obj))

            uri_attributes.setdefault(subject_label, {})

            existing_type = uri_attributes[subject_label].get(
                "rdf_type",
                "",
            )

            uri_attributes[subject_label]["rdf_type"] = (
                f"{existing_type}; {class_label}"
                if existing_type
                else class_label
            )

    # Build the NetworkX graph from URI-to-URI relations.
    for subject, predicate, obj in graph:
        if not isinstance(obj, URIRef):
            continue

        predicate_string = str(predicate)

        if any(
            predicate_string.startswith(skip_predicate)
            for skip_predicate in SKIP_PREDICATES
        ):
            continue

        # Exclude schema/TBox declarations.
        if subject in property_uris or subject in class_uris:
            continue

        if predicate == rdf_type and obj in class_uris:
            continue

        subject_label = clean_label(str(subject))
        object_label = clean_label(str(obj))
        predicate_label = clean_label(str(predicate))

        if subject_label not in nx_graph.nodes:
            nx_graph.add_node(
                subject_label,
                datasource=uri_datasource.get(subject_label, "unknown"),
                **uri_attributes.get(subject_label, {}),
            )

        if object_label not in nx_graph.nodes:
            nx_graph.add_node(
                object_label,
                datasource=uri_datasource.get(object_label, "unknown"),
                **uri_attributes.get(object_label, {}),
            )

        nx_graph.add_edge(
            subject_label,
            object_label,
            label=predicate_label,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(nx_graph, str(output_path))

    print(
        f"[DONE] Nodes: {nx_graph.number_of_nodes()} | "
        f"Edges: {nx_graph.number_of_edges()}"
    )

    print(f"[DONE] Saved: {output_path}")

    datasource_counts = Counter(
        nx_graph.nodes[node].get("datasource", "unknown")
        for node in nx_graph.nodes
    )

    print(f"[INFO] Datasource breakdown: {dict(datasource_counts)}")


def main(args: argparse.Namespace) -> None:
    if args.input_root is None:
        input_root = (
            DEFAULT_BASELINE_INPUT_ROOT
            if args.mode == "baseline"
            else DEFAULT_FINE_TUNED_INPUT_ROOT
        )
    else:
        input_root = args.input_root.expanduser().resolve()

    output_root = args.output_root.expanduser().resolve()

    ttl_directory = input_root / args.document

    if not ttl_directory.is_dir():
        print(
            f"[ERROR] Document TTL directory not found: {ttl_directory}"
        )
        print(
            "[INFO] Set --input-root to the appropriate Pipeline 4 output "
            "directory and provide the exact document folder name."
        )
        sys.exit(1)

    destination_directory = (
        output_root / args.mode / args.document
    )

    print("=" * 70)
    print("PIPELINE 4: TTL TO GEXF CONVERSION")
    print(f"Mode: {args.mode}")
    print(f"Input directory: {ttl_directory}")
    print(f"Output directory: {destination_directory}")
    print("=" * 70)

    ttl_files = ttl_files_for_mode(args.mode)
    found_files = 0

    for output_name, ttl_filename in ttl_files.items():
        ttl_path = ttl_directory / ttl_filename

        if not ttl_path.exists():
            print(f"[SKIP] TTL file not found: {ttl_path}")
            continue

        found_files += 1
        output_path = destination_directory / f"{output_name}.gexf"

        convert_ttl_to_gexf(ttl_path, output_path)

    if found_files == 0:
        print(
            "[ERROR] No expected Turtle files were found for the selected "
            "document and mode."
        )
        sys.exit(1)


if __name__ == "__main__":
    main(parse_args())