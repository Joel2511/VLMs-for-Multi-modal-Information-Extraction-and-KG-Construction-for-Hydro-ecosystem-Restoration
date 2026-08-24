cat > ~/make_ego_viz.py << 'PYEOF'
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import textwrap
import os

import os
PROJECT_ROOT = os.environ.get("RHINE_PROJECT_ROOT", "/home2020/home/icube/janil/Rhine_proj")

EXAMPLES = [
    {
        "id": "ex1",
        "ttl_dir": os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl_ft_merged/reconstituer-rhin-disparu-eca2fa4191154675939c8b4e07593bb7"),
        "anchor": "Rhin",
        "focus": "Projet_Kembs",
        "title": "Example 1 — Dense Hub: 'Rhin' (fine-tuned run)"
    },
    {
        "id": "ex2",
        "ttl_dir": os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl_ft_merged/reconstituer-rhin-disparu-eca2fa4191154675939c8b4e07593bb7"),
        "anchor": "Projet_Kembs",
        "focus": "Rhin",
        "title": "Example 2 — Named Project: 'Projet Kembs' (fine-tuned run)"
    },
    {
        "id": "ex3",
        "ttl_dir": os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl_ft_merged/reconstituer-rhin-disparu-eca2fa4191154675939c8b4e07593bb7"),
        "anchor": "Edf",
        "focus": "Rhin",
        "title": "Example 3 — Cross-Modal Bridge: 'EDF' links Text and Image evidence"
    },
    {
        "id": "ex4",
        "ttl_dir": os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl_ft_merged/reconstituer-rhin-disparu-eca2fa4191154675939c8b4e07593bb7"),
        "anchor": "Rocher_Distein",
        "focus": "Rhin",
        "title": "Example 4 — Pure Image Extraction: 'Rocher d'Istein' (Figure 5 caption only)"
    }
]
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl_ft_merged/visual_proof")

FILES = {
    "texts": "texts.gexf",
    "images": "images.gexf",
    "merged": "ft_merged_knowledge_graph.gexf"
}

SKIP_NODES = {
    "NamedIndividual", "ObjectProperty", "DatatypeProperty", "Class",
    "Ontology", "Thing", "topObjectProperty", "topDataProperty"
}

COLOR_MAP = {
    "Text": "#2B7BB9",
    "Image": "#D94F3D",
    "Text+Image": "#27A663",
    "unknown": "#8C8C8C"
}

BG_COLOR = "#FFFFFF"
PANEL_BG = "#F7F9FC"
ANCHOR_RING = "#1A1A2E"
FOCUS_RING = "#6C2BD9"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.dpi": 150
})


def load_graph_safe(path):
    if not os.path.exists(path):
        return nx.DiGraph()
    return nx.read_gexf(path)


def load_full_graphs(ttl_dir):
    return {name: load_graph_safe(os.path.join(ttl_dir, fname)) for name, fname in FILES.items()}


def clean_ego(G, anchor, radius=1):
    if anchor not in G.nodes:
        return set()
    return set(nx.ego_graph(G, anchor, radius=radius).nodes) - SKIP_NODES


def provenance_of(n, merged_graph):
    ds = merged_graph.nodes[n].get("datasource", "unknown") if n in merged_graph.nodes else "unknown"
    return ds


def load_ego(full_graphs, source_name, anchor, radius=1):
    G = full_graphs[source_name]
    if anchor not in G.nodes:
        return None
    ego = nx.ego_graph(G, anchor, radius=radius)
    ego.remove_nodes_from([n for n in ego.nodes if n in SKIP_NODES])
    return ego


def shorten(label, maxlen=12):
    label = label.replace("_", " ")
    return "\n".join(textwrap.wrap(label, width=maxlen))


def draw_subgraph(ax, subG, anchor, focus, title, merged_graph, show_legend=False):
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#1A1A2E")

    if subG is None or subG.number_of_nodes() == 0:
        ax.text(
            0.5, 0.5, f"'{anchor}'\nnot found",
            ha="center", va="center", fontsize=12, color="#999",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#ddd")
        )
        return

    pos = nx.spring_layout(subG, seed=42, k=6.0, iterations=250)

    node_colors, node_sizes, node_edge_colors, node_lw = [], [], [], []
    for n in subG.nodes:
        prov = provenance_of(n, merged_graph)
        node_colors.append(COLOR_MAP.get(prov, COLOR_MAP["unknown"]))

        if n == anchor:
            node_sizes.append(5600)
            node_edge_colors.append(ANCHOR_RING)
            node_lw.append(4.0)
        elif focus and n == focus:
            node_sizes.append(3000)
            node_edge_colors.append(FOCUS_RING)
            node_lw.append(3.2)
        else:
            node_sizes.append(2100)
            node_edge_colors.append("white")
            node_lw.append(1.8)

    nx.draw_networkx_nodes(
        subG, pos,
        node_color=node_colors, node_size=node_sizes,
        linewidths=node_lw, edgecolors=node_edge_colors,
        alpha=1.0, ax=ax
    )

    nx.draw_networkx_edges(
        subG, pos,
        edge_color="#555555", arrows=True, arrowsize=20,
        width=1.6, connectionstyle="arc3,rad=0.1",
        min_source_margin=20, min_target_margin=20,
        alpha=0.75, ax=ax
    )

    label_artists = nx.draw_networkx_labels(
        subG, pos,
        labels={n: shorten(n) for n in subG.nodes},
        font_size=8.5, font_weight="bold", font_color="white", ax=ax
    )
    for t in label_artists.values():
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    edge_labels = {}
    for u, v, d in subG.edges(data=True):
        lbl = str(d.get("label", "")).replace("_", " ").strip()
        lbl = lbl.replace("has", "").replace("is", "").replace("Located", "loc.")[:20]
        if lbl:
            edge_labels[(u, v)] = lbl

    nx.draw_networkx_edge_labels(
        subG, pos,
        edge_labels=edge_labels,
        font_size=7.5, font_color="#222222", font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.92, linewidth=0.8),
        ax=ax
    )

    nn, ne = subG.number_of_nodes(), subG.number_of_edges()
    focus_msg = f" | focus: {focus}" if focus else ""
    ax.text(
        0.02, 0.02, f"{nn} nodes · {ne} edges{focus_msg}",
        transform=ax.transAxes, fontsize=8.5, color="#333333", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.95, linewidth=0.8)
    )

    if show_legend:
        handles = [
            mpatches.Patch(facecolor=c, edgecolor="#555", linewidth=0.8, label=l)
            for l, c in COLOR_MAP.items()
        ]
        handles.append(mpatches.Patch(facecolor="white", edgecolor=FOCUS_RING, linewidth=2.2, label="Target node"))
        leg = ax.legend(
            handles=handles, loc="upper right", fontsize=9,
            framealpha=0.95, edgecolor="#CCCCCC",
            title="Node Type", title_fontsize=9, borderpad=0.8
        )
        leg.get_title().set_fontweight("bold")


def save_single(out_dir, full_graphs, anchor, focus, source_name, prefix, radius=1):
    subG = load_ego(full_graphs, source_name, anchor, radius)

    labels = {"texts": "Text KG", "images": "Image KG", "merged": "Merged KG"}
    title = f"{labels[source_name]}\n1-hop neighborhood of '{anchor}'"

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.patch.set_facecolor(BG_COLOR)

    draw_subgraph(ax, subG, anchor, focus, title, full_graphs["merged"], show_legend=True)

    if subG is not None:
        fig.suptitle(
            f"{labels[source_name]} — '{anchor}'",
            fontsize=11, color="#555", y=0.98
        )

    plt.tight_layout(pad=1.5)
    out = os.path.join(out_dir, f"{prefix}_{source_name}_{anchor}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[SAVED single] {out}")


def make_trio(out_dir, full_graphs, anchor, focus, suptitle, out_filename, radius=1):
    pass

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    fig.patch.set_facecolor(BG_COLOR)

    panel_titles = {
        "texts": "Text KG\n1-hop neighborhood of '{}'",
        "images": "Image KG\n1-hop neighborhood of '{}'",
        "merged": "Merged KG\n1-hop neighborhood of '{}'"
    }

    for i, name in enumerate(FILES.keys()):
        subG = load_ego(full_graphs, name, anchor, radius)
        draw_subgraph(
            axes[i], subG, anchor, focus,
            panel_titles[name].format(anchor),
            full_graphs["merged"],
            show_legend=(i == 2)
        )

    for ax in axes[:2]:
        ax.annotate(
            "", xy=(1.06, 0.5), xytext=(0.94, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5, mutation_scale=22)
        )

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01, color="#1A1A2E")
    plt.tight_layout(pad=2.5, w_pad=3.0)

    out = os.path.join(out_dir, out_filename)
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[SAVED trio]   {out}")


for ex in EXAMPLES:
    full_graphs = load_full_graphs(ex["ttl_dir"])
    prefix = ex["id"]
    anchor = ex["anchor"]
    focus = ex["focus"]

    for src in FILES.keys():
        save_single(OUT_DIR, full_graphs, anchor, focus, src, prefix)

    make_trio(
        OUT_DIR,
        full_graphs,
        anchor,
        focus,
        ex["title"],
        f"{prefix}_trio_{anchor}.png"
    )

print("\n[ALL DONE]")
PYEOF

python3 ~/make_ego_viz.py

# ============================================================
cat > ~/make_rare_tail_viz.py << 'PYEOF'
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import textwrap
import os
import copy
import rdflib
from rdflib import Namespace, RDF

EXAMPLES = [
    {
        "id": "ex1",
        "anchor": "Etang",
        "relation": "isLocatedOn",
        "target": "Geographiczone",
        "merged_gexf": os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl/fichier/merged_knowledge_graph.gexf"),
        "predicted_ttl": os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl/tagged/fichier/merged_knowledge_graph_predicted.ttl"),
        "title": "Example 1 — proof of new predicted edge"
    },
    {
        "id": "ex2",
        "anchor": "Cours_Deau",
        "relation": "isLocatedIn",
        "target": "Paysage_Agricole",
        "merged_gexf": os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl/Rapport-stage-M2-Floriane/merged_knowledge_graph.gexf"),
        "predicted_ttl": os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl/tagged/Rapport-stage-M2-Floriane/merged_knowledge_graph_predicted.ttl"),
        "title": "Example 2 — proof of new predicted edge"
    }
]

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/ttl_results_qwen3vl_ft_merged/visual_proof_reconstituer")
os.makedirs(OUT_DIR, exist_ok=True)

NS = Namespace("http://www.semanticweb.org/fethi/ontologies/2024/1/TetraOnto.owl#")

SKIP_NODES = {
    "NamedIndividual", "ObjectProperty", "DatatypeProperty", "Class",
    "Ontology", "Thing", "topObjectProperty", "topDataProperty"
}

COLOR_MAP = {
    "Text": "#2B7BB9",
    "Image": "#D94F3D",
    "Text+Image": "#27A663",
    "Predicted": "#D4A017",
    "Ghost": "#FBE7A1",
    "unknown": "#8C8C8C"
}

BG_COLOR = "#FFFFFF"
PANEL_BG = "#F7F9FC"
ANCHOR_RING = "#1A1A2E"
TARGET_RING = "#6C2BD9"
PRED_EDGE = "#D4A017"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.dpi": 150
})


def local_name(uri_or_str):
    s = str(uri_or_str)
    s = s.split("#")[-1] if "#" in s else s.rstrip("/").split("/")[-1]
    return s.strip("<>.").strip()


def shorten(label, maxlen=12):
    label = label.replace("_", " ")
    return "\n".join(textwrap.wrap(label, width=maxlen))


def ensure_defaults(G):
    for n in G.nodes:
        G.nodes[n].setdefault("datasource", "unknown")
        G.nodes[n].setdefault("predicted_node", False)
        G.nodes[n].setdefault("ghost_node", False)
    for u, v in G.edges:
        G.edges[u, v].setdefault("predicted_edge", False)
        G.edges[u, v].setdefault("label", G.edges[u, v].get("label", ""))


def edge_has_relation(G, u, v, rel):
    if not G.has_edge(u, v):
        return False
    data = G.get_edge_data(u, v)
    if isinstance(data, dict) and "label" in data:
        return str(data.get("label", "")) == rel
    return False


def parse_predicted_meta(pred_path, anchor, relation, target):
    meta = {"exists_in_tagged": False, "predicted_flag": None, "score": None, "rank": None}
    if not os.path.exists(pred_path):
        return meta

    g = rdflib.Graph()
    g.parse(pred_path, format="turtle")

    s = NS[anchor]
    p = NS[relation]
    o = NS[target]

    meta["exists_in_tagged"] = ((s, p, o) in g)

    for stmt in g.subjects(RDF.type, RDF.Statement):
        subj = g.value(stmt, RDF.subject)
        pred = g.value(stmt, RDF.predicate)
        obj = g.value(stmt, RDF.object)
        if subj == s and pred == p and obj == o:
            meta["predicted_flag"] = g.value(stmt, NS.predicted)
            meta["score"] = g.value(stmt, NS.confidenceScore)
            meta["rank"] = g.value(stmt, NS.rank)
            break

    return meta


def load_graph_safe(path):
    if not os.path.exists(path):
        return nx.DiGraph()
    return nx.read_gexf(path)


def build_before_after(example):
    G_before = load_graph_safe(example["merged_gexf"])
    ensure_defaults(G_before)

    G_after = copy.deepcopy(G_before)
    ensure_defaults(G_after)

    anchor = example["anchor"]
    relation = example["relation"]
    target = example["target"]

    meta = parse_predicted_meta(example["predicted_ttl"], anchor, relation, target)
    before_has = edge_has_relation(G_before, anchor, target, relation)

    if target not in G_after.nodes:
        G_after.add_node(target, datasource="Predicted", predicted_node=True, ghost_node=False)

    if anchor not in G_after.nodes:
        G_after.add_node(anchor, datasource="unknown", predicted_node=False, ghost_node=False)

    if meta["exists_in_tagged"] and not before_has:
        G_after.add_edge(anchor, target, label=relation, predicted_edge=True)
    elif meta["exists_in_tagged"] and edge_has_relation(G_after, anchor, target, relation):
        G_after.edges[anchor, target]["predicted_edge"] = True

    ensure_defaults(G_after)

    return G_before, G_after, meta, before_has


def clean_ego(G, anchor, radius=1):
    if anchor not in G.nodes:
        return None
    ego = nx.ego_graph(G, anchor, radius=radius)
    ego.remove_nodes_from([n for n in ego.nodes if n in SKIP_NODES])
    return ego


def add_ghost_target_if_needed(subG, target):
    if subG is None:
        return None
    H = subG.copy()
    if target not in H.nodes:
        H.add_node(target, datasource="Ghost", predicted_node=False, ghost_node=True)
    return H


def draw_panel(ax, subG, anchor, target, title, pos, meta=None, before_has=False, after_mode=False):
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#1A1A2E")

    if subG is None or subG.number_of_nodes() == 0:
        ax.text(
            0.5, 0.5, f"'{anchor}'\nnot found",
            ha="center", va="center", fontsize=12, color="#999",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#ddd")
        )
        return

    node_colors, node_sizes, node_edge_colors, node_lw = [], [], [], []
    for n in subG.nodes:
        datasource = subG.nodes[n].get("datasource", "unknown")
        ghost = subG.nodes[n].get("ghost_node", False)

        if ghost:
            node_colors.append(COLOR_MAP["Ghost"])
        else:
            node_colors.append(COLOR_MAP.get(datasource, COLOR_MAP["unknown"]))

        if n == anchor:
            node_sizes.append(5600)
            node_edge_colors.append(ANCHOR_RING)
            node_lw.append(4.0)
        elif n == target:
            node_sizes.append(3000)
            node_edge_colors.append(TARGET_RING)
            node_lw.append(3.0)
        else:
            node_sizes.append(2100)
            node_edge_colors.append("white")
            node_lw.append(1.8)

    nx.draw_networkx_nodes(
        subG, pos,
        node_color=node_colors, node_size=node_sizes,
        linewidths=node_lw, edgecolors=node_edge_colors,
        alpha=1.0, ax=ax
    )

    normal_edges = [(u, v) for u, v, d in subG.edges(data=True) if not d.get("predicted_edge", False)]
    predicted_edges = [(u, v) for u, v, d in subG.edges(data=True) if d.get("predicted_edge", False)]

    nx.draw_networkx_edges(
        subG, pos,
        edgelist=normal_edges,
        edge_color="#555555", arrows=True, arrowsize=20,
        width=1.6, connectionstyle="arc3,rad=0.1",
        min_source_margin=20, min_target_margin=20,
        alpha=0.75, ax=ax
    )

    nx.draw_networkx_edges(
        subG, pos,
        edgelist=predicted_edges,
        edge_color=PRED_EDGE, arrows=True, arrowsize=24,
        width=2.8, style="dashed",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=20, min_target_margin=20,
        alpha=0.98, ax=ax
    )

    label_artists = nx.draw_networkx_labels(
        subG, pos,
        labels={n: shorten(n) for n in subG.nodes},
        font_size=8.5, font_weight="bold", font_color="white", ax=ax
    )
    for t in label_artists.values():
        t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])

    edge_labels = {}
    for u, v, d in subG.edges(data=True):
        lbl = str(d.get("label", "")).replace("_", " ").strip()
        lbl = lbl[:24]
        if lbl:
            edge_labels[(u, v)] = lbl + (" *" if d.get("predicted_edge", False) else "")

    nx.draw_networkx_edge_labels(
        subG, pos,
        edge_labels=edge_labels,
        font_size=7.5, font_color="#222222", font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.92, linewidth=0.8),
        ax=ax
    )

    nn, ne = subG.number_of_nodes(), subG.number_of_edges()
    pred_edges = sum(1 for _, _, d in subG.edges(data=True) if d.get("predicted_edge", False))

    if after_mode and meta is not None:
        score_txt = f"{meta['score']}" if meta.get("score") is not None else "NA"
        rank_txt = f"{meta['rank']}" if meta.get("rank") is not None else "NA"
        footer = f"{nn} nodes · {ne} edges | predicted edge(s): {pred_edges} | rank={rank_txt} | score={score_txt}"
    else:
        footer = f"{nn} nodes · {ne} edges | edge existed before: {before_has}"

    ax.text(
        0.02, 0.02, footer,
        transform=ax.transAxes, fontsize=8.5, color="#333333", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", alpha=0.95, linewidth=0.8)
    )

    handles = [
        mpatches.Patch(facecolor=COLOR_MAP["Text"], edgecolor="#555", linewidth=0.8, label="Text"),
        mpatches.Patch(facecolor=COLOR_MAP["Image"], edgecolor="#555", linewidth=0.8, label="Image"),
        mpatches.Patch(facecolor=COLOR_MAP["Text+Image"], edgecolor="#555", linewidth=0.8, label="Text+Image"),
        mpatches.Patch(facecolor=COLOR_MAP["Ghost"], edgecolor=TARGET_RING, linewidth=1.6, label="Target absent before"),
        mpatches.Patch(facecolor="none", edgecolor=PRED_EDGE, linewidth=2.4, label="New predicted edge (dashed)")
    ]
    leg = ax.legend(
        handles=handles, loc="upper right", fontsize=8.5,
        framealpha=0.95, edgecolor="#CCCCCC",
        title="Legend", title_fontsize=9, borderpad=0.8
    )
    leg.get_title().set_fontweight("bold")


def save_before_after(example, radius=1):
    G_before, G_after, meta, before_has = build_before_after(example)

    sub_before = clean_ego(G_before, example["anchor"], radius)
    sub_after = clean_ego(G_after, example["anchor"], radius)

    display_before = add_ghost_target_if_needed(sub_before, example["target"])
    display_after = sub_after

    layout_graph = display_after if display_after is not None and display_after.number_of_nodes() > 0 else display_before
    pos = nx.spring_layout(layout_graph, seed=42, k=6.0, iterations=250)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor(BG_COLOR)

    draw_panel(
        axes[0], display_before,
        example["anchor"], example["target"],
        f"BEFORE\nMerged KG around '{example['anchor']}'",
        pos, meta=meta, before_has=before_has, after_mode=False
    )

    draw_panel(
        axes[1], display_after,
        example["anchor"], example["target"],
        f"AFTER\nMerged KG + predicted triple",
        pos, meta=meta, before_has=before_has, after_mode=True
    )

    fig.suptitle(
        f"{example['title']}\n{example['anchor']} --{example['relation']}--> {example['target']}",
        fontsize=14, fontweight="bold", y=1.01, color="#1A1A2E"
    )

    plt.tight_layout(pad=2.0, w_pad=2.5)
    out = os.path.join(OUT_DIR, f"{example['id']}_proof_before_after_{example['anchor']}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[SAVED compare] {out}")

    return G_before, G_after, meta, before_has


def save_final_collage(examples, radius=1):
    prepared = []
    for ex in examples:
        G_before, G_after, meta, before_has = build_before_after(ex)
        sub_before = clean_ego(G_before, ex["anchor"], radius)
        sub_after = clean_ego(G_after, ex["anchor"], radius)
        display_before = add_ghost_target_if_needed(sub_before, ex["target"])
        display_after = sub_after
        layout_graph = display_after if display_after is not None and display_after.number_of_nodes() > 0 else display_before
        pos = nx.spring_layout(layout_graph, seed=42, k=6.0, iterations=250)
        prepared.append((ex, display_before, display_after, meta, before_has, pos))

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.patch.set_facecolor(BG_COLOR)

    for row, (ex, display_before, display_after, meta, before_has, pos) in enumerate(prepared):
        draw_panel(
            axes[row, 0], display_before,
            ex["anchor"], ex["target"],
            f"{ex['anchor']} — BEFORE",
            pos, meta=meta, before_has=before_has, after_mode=False
        )
        draw_panel(
            axes[row, 1], display_after,
            ex["anchor"], ex["target"],
            f"{ex['anchor']} — AFTER",
            pos, meta=meta, before_has=before_has, after_mode=True
        )

    fig.suptitle(
        "Final proof collage — new predicted triples highlighted in gold dashed edges",
        fontsize=16, fontweight="bold", y=1.01, color="#1A1A2E"
    )

    plt.tight_layout(pad=2.2, w_pad=2.0, h_pad=2.5)
    out = os.path.join(OUT_DIR, "final_proof_collage_2examples.png")
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[SAVED final collage] {out}")


for ex in EXAMPLES:
    save_before_after(ex)

save_final_collage(EXAMPLES)

print("\n[ALL DONE]")
PYEOF

python3 ~/make_rare_tail_viz.py