# run this once: generate_ontology_summary.py
from rdflib import Graph, Namespace, RDF, RDFS, OWL
from rdflib.namespace import SKOS
from collections import defaultdict


ONTOLOGY_TTL_PATH = "/home2020/home/icube/janil/Rhine_proj/data/ontology/TetraOnto.ttl"
OUTPUT_PATH = "/home2020/home/icube/janil/Rhine_proj/data/ontology/TetraOnto_compact.txt"


g = Graph()
g.parse(ONTOLOGY_TTL_PATH, format="turtle")


TETRA_NS = "http://www.semanticweb.org/fethi/ontologies/2024/1/TetraOnto.owl#"
TETRA = Namespace(TETRA_NS)


def get_labels_suffix(entity, g):
    """Compact suffix combining prefLabel (EN) and altLabel(s) (usually FR),
    so the LLM sees canonical aliases without bloating token count."""
    pref = g.value(entity, SKOS.prefLabel)
    alts = [str(a) for a in g.objects(entity, SKOS.altLabel)]
    parts = []
    if pref:
        parts.append(f"EN:{pref}")
    if alts:
        parts.append(f"FR/alt:{'|'.join(alts)}")
    return f"  [{', '.join(parts)}]" if parts else ""


lines = []
lines.append("=== CLASSES ===")
for cls in sorted(g.subjects(RDF.type, OWL.Class)):
    label = g.value(cls, RDFS.label)
    comment = g.value(cls, RDFS.comment)
    name = str(cls).replace(TETRA_NS, ":")
    entry = name
    if label:
        entry += f"  # {label}"
    entry += get_labels_suffix(cls, g)
    if comment:
        entry += f" — {comment}"
    parents = [str(p).replace(TETRA_NS, ":") for p in g.objects(cls, RDFS.subClassOf)
           if str(p).startswith(TETRA_NS)]
    if parents:
        entry += f"  subClassOf:{','.join(parents)}"
    lines.append(entry)


lines.append("\n=== OBJECT PROPERTIES ===")
for prop in sorted(g.subjects(RDF.type, OWL.ObjectProperty)):
    label = g.value(prop, RDFS.label)
    domain = g.value(prop, RDFS.domain)
    range_ = g.value(prop, RDFS.range)
    name = str(prop).replace(TETRA_NS, ":")
    entry = name
    if label:
        entry += f"  # {label}"
    entry += get_labels_suffix(prop, g)
    if domain:
        entry += f"  domain:{str(domain).replace(TETRA_NS, ':')}"
    if range_:
        entry += f"  range:{str(range_).replace(TETRA_NS, ':')}"
    lines.append(entry)


lines.append("\n=== DATATYPE PROPERTIES ===")
for prop in sorted(g.subjects(RDF.type, OWL.DatatypeProperty)):
    label = g.value(prop, RDFS.label)
    domain = g.value(prop, RDFS.domain)
    name = str(prop).replace(TETRA_NS, ":")
    entry = name
    if label:
        entry += f"  # {label}"
    entry += get_labels_suffix(prop, g)
    if domain:
        entry += f"  domain:{str(domain).replace(TETRA_NS, ':')}"
    lines.append(entry)


output = "\n".join(lines)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(output)


print(f"Done. {len(lines)} lines, ~{len(output)//4} tokens")
print(f"Saved to: {OUTPUT_PATH}")