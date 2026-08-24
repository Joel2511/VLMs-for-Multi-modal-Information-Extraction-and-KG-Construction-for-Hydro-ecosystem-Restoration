import os
from rdflib import Graph, URIRef
import networkx as nx
import re
from collections import Counter

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")

# ─────────────────────────────────────────────
# PATHS — change PDF for other documents
# ─────────────────────────────────────────────
# TTL_DIR = os.path.join(PROJECT_ROOT, "outputs/triples_fine_tuned/fichier")
# TTL_DIR = os.path.join(PROJECT_ROOT, "outputs/triples_fine_tuned/reconstituer-rhin-disparu-eca2fa4191154675939c8b4e07593bb7")
# TTL_DIR = os.path.join(PROJECT_ROOT, "outputs/triples_fine_tuned/Rapport-stage-M2-Floriane")
# TTL_DIR = os.path.join(PROJECT_ROOT, "outputs/triples_fine_tuned/2025 12 OFB - FICHE TECHNIQUE Définition, identification, préservation et restauration des zones de sources des cours d’eau en Bretagne et Pays de la Loire")
# TTL_DIR = os.path.join(PROJECT_ROOT, "outputs/triples_fine_tuned/dossier_unio_crassus_zorn_v4_19_11_2024")
TTL_DIR = os.path.join(PROJECT_ROOT, "outputs/triples_fine_tuned/test_dataset_1")

TTL_FILES = {
    "texts":                     "texts.ttl",
    "images":                    "images.ttl",
    "ft_merged_knowledge_graph": "ft_merged_knowledge_graph.ttl",
}

TETRA_NS = "http://www.semanticweb.org/fethi/ontologies/2024/1/TetraOnto.owl#"
DS_PRED = URIRef(TETRA_NS + "extractedFrom")

SKIP_PREDICATES = [
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    TETRA_NS + "dataSource",
    TETRA_NS + "hasPageNumber",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def clean_label(uri_str: str) -> str:
    label = uri_str.strip()
    label = re.sub(r'[\n\r\t]', ' ', label)
    label = label.split("#")[-1].split("/")[-1]
    return label

def get_datasource(g: Graph, subject: URIRef) -> str:
    values = set()
    for _, _, o in g.triples((subject, DS_PRED, None)):
        val_lower = str(o).lower()
        if "image" in val_lower or "vqa" in val_lower or "caption" in val_lower:
            values.add("Image")
        else:
            values.add("Text")
    if not values:
        return "unknown"
    if len(values) == 1:
        return values.pop()
    return "Text+Image"

# ─────────────────────────────────────────────
# PROCESS EACH FILE
# ─────────────────────────────────────────────
for out_name, ttl_filename in TTL_FILES.items():
    ttl_path = os.path.join(TTL_DIR, ttl_filename)
    out_path = os.path.join(TTL_DIR, f"{out_name}.gexf")

    print(f"\n[INFO] Parsing: {ttl_path}")
    g = Graph()
    g.parse(ttl_path, format="turtle")
    print(f"[INFO] Loaded {len(g)} triples")

    # First pass — collect datasource per subject URI
    uri_datasource = {}
    for s in set(g.subjects()):
        if isinstance(s, URIRef):
            label = clean_label(str(s))
            uri_datasource[label] = get_datasource(g, s)

    # Collect literal (datatype-property) attributes per subject URI
    uri_attributes = {}
    for s, p, o in g:
        if isinstance(o, URIRef):
            continue
        p_str = str(p)
        if any(p_str.startswith(skip) for skip in SKIP_PREDICATES):
            continue
        s_label = clean_label(str(s))
        attr_key = clean_label(p_str)
        attr_val = str(o).replace("\n", " ").strip()[:200]
        uri_attributes.setdefault(s_label, {})
        existing = uri_attributes[s_label].get(attr_key, "")
        uri_attributes[s_label][attr_key] = f"{existing}; {attr_val}" if existing else attr_val

    # Second pass — build NetworkX graph
    nx_graph = nx.MultiDiGraph()

    # NEW: identify TBox meta-elements (classes and properties declared in the schema)
    RDF_TYPE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    OWL_CLASS = URIRef("http://www.w3.org/2002/07/owl#Class")
    OWL_OBJ_PROP = URIRef("http://www.w3.org/2002/07/owl#ObjectProperty")
    OWL_DATA_PROP = URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty")
    OWL_ANN_PROP = URIRef("http://www.w3.org/2002/07/owl#AnnotationProperty")

    property_uris = set()
    class_uris = set()
    for s, p, o in g.triples((None, RDF_TYPE, None)):
        if o in (OWL_OBJ_PROP, OWL_DATA_PROP, OWL_ANN_PROP):
            property_uris.add(s)
        elif o == OWL_CLASS:
            class_uris.add(s)

    # NEW: capture rdf:type-as-class-membership as an attribute, not an edge
    for s, p, o in g.triples((None, RDF_TYPE, None)):
        if o in class_uris:
            s_label = clean_label(str(s))
            o_label = clean_label(str(o))
            uri_attributes.setdefault(s_label, {})
            existing = uri_attributes[s_label].get("rdf_type", "")
            uri_attributes[s_label]["rdf_type"] = f"{existing}; {o_label}" if existing else o_label

    for s, p, o in g:
        if not isinstance(o, URIRef):
            continue
        p_str = str(p)
        if any(p_str.startswith(skip) for skip in SKIP_PREDICATES):
            continue

        # NEW: skip pure schema/TBox declarations
        if s in property_uris or s in class_uris:
            continue
        if p == RDF_TYPE and o in class_uris:
            continue  # already captured as attribute above

        s_label = clean_label(str(s))
        o_label = clean_label(str(o))
        p_label = clean_label(str(p))

        if s_label not in nx_graph.nodes:
            nx_graph.add_node(s_label, datasource=uri_datasource.get(s_label, "unknown"),
                               **uri_attributes.get(s_label, {}))
        if o_label not in nx_graph.nodes:
            nx_graph.add_node(o_label, datasource=uri_datasource.get(o_label, "unknown"),
                               **uri_attributes.get(o_label, {}))

        nx_graph.add_edge(s_label, o_label, label=p_label)

    # Save
    nx.write_gexf(nx_graph, out_path)
    print(f"[DONE] Nodes: {nx_graph.number_of_nodes()} | Edges: {nx_graph.number_of_edges()}")
    print(f"[DONE] Saved: {out_path}")

    ds_counts = Counter(nx_graph.nodes[n].get("datasource", "unknown") for n in nx_graph.nodes)
    print(f"[INFO] datasource breakdown: {dict(ds_counts)}")