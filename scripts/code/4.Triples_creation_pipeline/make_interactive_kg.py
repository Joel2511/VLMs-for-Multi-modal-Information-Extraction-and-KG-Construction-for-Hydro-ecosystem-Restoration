# python3 make_interactive_kg.py /home2020/home/icube/janil/Rhine_proj/outputs/triples_fine_tuned/test_dataset/ft_merged_knowledge_graph.gexf /home2020/home/icube/janil/Rhine_proj/outputs/triples_fine_tuned/test_dataset/kg_finetuned_rhin_test_1.html
import argparse
import networkx as nx
from pyvis.network import Network

COLOR_MAP = {
    "Text": "#2B7BB9",
    "Image": "#D94F3D",
    "Text+Image": "#27A663",
    "unknown": "#8C8C8C"
}

SKIP_NODES = {
    "NamedIndividual", "ObjectProperty", "DatatypeProperty", "Class",
    "Ontology", "Thing", "topObjectProperty", "topDataProperty"
}


def build_interactive_html(gexf_path, out_html, height="900px"):
    G = nx.read_gexf(gexf_path)
    G.remove_nodes_from([n for n in list(G.nodes) if n in SKIP_NODES])

    net = Network(
        height=height, width="100%",
        bgcolor="#ffffff", font_color="#1a1a2e",
        directed=True, notebook=False,
        cdn_resources="in_line"
    )

    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "scaling": {"min": 10, "max": 45},
        "font": {"size": 14, "face": "arial", "strokeWidth": 3, "strokeColor": "#ffffff"}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
        "color": {"color": "#999999", "highlight": "#1A1A2E"},
        "font": {"size": 10, "align": "middle"},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {"gravitationalConstant": -60, "springLength": 120, "avoidOverlap": 0.6},
        "stabilization": {"iterations": 300, "fit": true},
        "enabled": true
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    degree = dict(G.degree())

    for n, data in G.nodes(data=True):
        prov = data.get("datasource", "unknown")
        color = COLOR_MAP.get(prov, COLOR_MAP["unknown"])
        size = 15 + min(degree.get(n, 1), 40) * 1.3

        extra_attrs = {k: v for k, v in data.items() if k not in ("datasource", "label", "viz")}
        attrs_html = "".join(f"<br><b>{k}:</b> {v}" for k, v in extra_attrs.items())
        title = f"<b>{n}</b><br>Source: {prov}<br>Degree: {degree.get(n,0)}{attrs_html}"

        net.add_node(n, label=n.replace("_", " "), color=color, size=size, title=title)

    for u, v, data in G.edges(data=True):
        lbl = str(data.get("label", "")).replace("_", " ")
        net.add_edge(u, v, label=lbl, title=lbl)

    net.write_html(out_html, open_browser=False, notebook=False)

    legend_html = """
    <div id="kg-legend" style="position:fixed; top:20px; right:20px; background:white;
         border:1px solid #ccc; border-radius:8px; padding:14px 18px; font-family:Arial;
         font-size:13px; box-shadow:0 2px 8px rgba(0,0,0,0.15); z-index:9999;">
      <b>Node Type</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:#2B7BB9;border-radius:50%;margin-right:6px;"></span>Text<br>
      <span style="display:inline-block;width:12px;height:12px;background:#D94F3D;border-radius:50%;margin-right:6px;"></span>Image<br>
      <span style="display:inline-block;width:12px;height:12px;background:#27A663;border-radius:50%;margin-right:6px;"></span>Text+Image<br>
      <span style="display:inline-block;width:12px;height:12px;background:#8C8C8C;border-radius:50%;margin-right:6px;"></span>Unknown
    </div>
    <input id="kg-search" placeholder="Search node..." style="position:fixed; top:20px; left:20px;
           padding:8px 12px; font-size:14px; border:1px solid #ccc; border-radius:6px; z-index:9999; width:250px;">
    """

    freeze_script = """
    <script type="text/javascript">
      network.once("stabilizationIterationsDone", function () {
        network.setOptions({ physics: false });
      });
    </script>
    """

    click_script = """
    <button id="kg-reset" style="position:fixed; top:70px; left:20px; padding:8px 14px;
            font-size:13px; border:1px solid #ccc; border-radius:6px; z-index:9999;
            background:#1A1A2E; color:white; cursor:pointer;">Reset View</button>

    <script type="text/javascript">
      var allNodesDataset = nodes.get();
      var allEdgesDataset = edges.get();

      function resetView() {
        var nodeUpdates = allNodesDataset.map(function(n) {
          return {id: n.id, hidden: false};
        });
        var edgeUpdates = allEdgesDataset.map(function(e) {
          return {id: e.id, hidden: false};
        });
        nodes.update(nodeUpdates);
        edges.update(edgeUpdates);
        network.fit({animation: true});
      }

      network.on("click", function(params) {
        if (params.nodes.length > 0) {
          var selected = params.nodes[0];
          var connected = network.getConnectedNodes(selected);
          var keepSet = new Set(connected);
          keepSet.add(selected);

          var nodeUpdates = allNodesDataset.map(function(n) {
            return {id: n.id, hidden: !keepSet.has(n.id)};
          });

          var connectedEdges = network.getConnectedEdges(selected);
          var edgeUpdates = allEdgesDataset.map(function(e) {
            return {id: e.id, hidden: !connectedEdges.includes(e.id)};
          });

          nodes.update(nodeUpdates);
          edges.update(edgeUpdates);
          network.fit({nodes: Array.from(keepSet), animation: true});
        } else {
          resetView();
        }
      });

      document.getElementById('kg-reset').addEventListener('click', resetView);

      document.getElementById('kg-search').addEventListener('input', function(e) {
        var query = e.target.value.toLowerCase();
        if (!query) { resetView(); return; }
        var matchIds = allNodesDataset.filter(function(n){
          return n.label.toLowerCase().includes(query);
        }).map(function(n){ return n.id; });
        if (matchIds.length > 0) {
          var selected = matchIds[0];
          var connected = network.getConnectedNodes(selected);
          var keepSet = new Set(connected);
          keepSet.add(selected);
          var nodeUpdates = allNodesDataset.map(function(n) {
            return {id: n.id, hidden: !keepSet.has(n.id)};
          });
          var connectedEdges = network.getConnectedEdges(selected);
          var edgeUpdates = allEdgesDataset.map(function(e) {
            return {id: e.id, hidden: !connectedEdges.includes(e.id)};
          });
          nodes.update(nodeUpdates);
          edges.update(edgeUpdates);
          network.fit({nodes: Array.from(keepSet), animation: true});
        }
      });
    </script>
    """

    with open(out_html, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</body>", legend_html + click_script + freeze_script + "</body>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print("[SAVED interactive KG] " + out_html)
    print("  Nodes: " + str(G.number_of_nodes()) + "  Edges: " + str(G.number_of_edges()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gexf_path", help="Path to merged .gexf file")
    parser.add_argument("out_html", help="Output HTML file path")
    args = parser.parse_args()
    build_interactive_html(args.gexf_path, args.out_html)