"""
Create an interactive HTML visualization from a Pipeline 4 GEXF graph.

The input GEXF file is typically created by convert_ttl_to_gexf.py. The output
is a self-contained interactive HTML graph with datasource colouring, node
search, click-to-focus exploration, reset controls, and inline JavaScript/CSS.
"""

import argparse
from pathlib import Path

import networkx as nx
from pyvis.network import Network


COLOR_MAP = {
    "Text": "#2B7BB9",
    "Image": "#D94F3D",
    "Text+Image": "#27A663",
    "unknown": "#8C8C8C",
}

SKIP_NODES = {
    "NamedIndividual",
    "ObjectProperty",
    "DatatypeProperty",
    "Class",
    "Ontology",
    "Thing",
    "topObjectProperty",
    "topDataProperty",
}


def build_interactive_html(
    gexf_path: Path,
    output_html: Path,
    height: str = "900px",
) -> None:
    graph = nx.read_gexf(str(gexf_path))

    graph.remove_nodes_from(
        [
            node
            for node in list(graph.nodes)
            if node in SKIP_NODES
        ]
    )

    network = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#1a1a2e",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )

    network.set_options(
        """
        {
          "nodes": {
            "shape": "dot",
            "scaling": {"min": 10, "max": 45},
            "font": {
              "size": 14,
              "face": "arial",
              "strokeWidth": 3,
              "strokeColor": "#ffffff"
            }
          },
          "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
            "color": {"color": "#999999", "highlight": "#1A1A2E"},
            "font": {"size": 10, "align": "middle"},
            "smooth": {"type": "continuous"}
          },
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -60,
              "springLength": 120,
              "avoidOverlap": 0.6
            },
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
        """
    )

    degree = dict(graph.degree())

    for node, data in graph.nodes(data=True):
        provenance = data.get("datasource", "unknown")
        color = COLOR_MAP.get(provenance, COLOR_MAP["unknown"])
        size = 15 + min(degree.get(node, 1), 40) * 1.3

        extra_attributes = {
            key: value
            for key, value in data.items()
            if key not in ("datasource", "label", "viz")
        }

        attributes_html = "".join(
            f"<br><b>{key}:</b> {value}"
            for key, value in extra_attributes.items()
        )

        title = (
            f"<b>{node}</b><br>"
            f"Source: {provenance}<br>"
            f"Degree: {degree.get(node, 0)}"
            f"{attributes_html}"
        )

        network.add_node(
            node,
            label=str(node).replace("_", " "),
            color=color,
            size=size,
            title=title,
        )

    for source, target, data in graph.edges(data=True):
        label = str(data.get("label", "")).replace("_", " ")

        network.add_edge(
            source,
            target,
            label=label,
            title=label,
        )

    output_html.parent.mkdir(parents=True, exist_ok=True)

    network.write_html(
        str(output_html),
        open_browser=False,
        notebook=False,
    )

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

      document.getElementById("kg-reset").addEventListener(
        "click",
        resetView
      );

      document.getElementById("kg-search").addEventListener(
        "input",
        function(event) {
          var query = event.target.value.toLowerCase();

          if (!query) {
            resetView();
            return;
          }

          var matchIds = allNodesDataset.filter(function(n) {
            return n.label.toLowerCase().includes(query);
          }).map(function(n) {
            return n.id;
          });

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
              return {
                id: e.id,
                hidden: !connectedEdges.includes(e.id)
              };
            });

            nodes.update(nodeUpdates);
            edges.update(edgeUpdates);
            network.fit({
              nodes: Array.from(keepSet),
              animation: true
            });
          }
        }
      );
    </script>
    """

    with open(output_html, "r", encoding="utf-8") as file:
        html = file.read()

    html = html.replace(
        "</body>",
        legend_html + click_script + freeze_script + "</body>",
    )

    with open(output_html, "w", encoding="utf-8") as file:
        file.write(html)

    print(f"[SAVED interactive KG] {output_html}")
    print(
        f"  Nodes: {graph.number_of_nodes()}  "
        f"Edges: {graph.number_of_edges()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a self-contained interactive HTML visualization from a "
            "Pipeline 4 GEXF graph."
        )
    )

    parser.add_argument(
        "gexf_path",
        type=Path,
        help="Path to the input GEXF graph file.",
    )

    parser.add_argument(
        "output_html",
        type=Path,
        help="Path where the generated interactive HTML file will be saved.",
    )

    parser.add_argument(
        "--height",
        default="900px",
        help="HTML network height. Default: 900px.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_gexf = args.gexf_path.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()

    if not input_gexf.is_file():
        raise FileNotFoundError(
            f"Input GEXF graph not found: {input_gexf}"
        )

    build_interactive_html(
        input_gexf,
        output_html,
        height=args.height,
    )