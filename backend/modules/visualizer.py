"""
modules/visualizer.py
=====================
Renders interactive HTML graph visualizations using PyVis.
Falls back to static matplotlib if PyVis is unavailable.
"""

from __future__ import annotations
import logging
import os
from typing import Dict, Any

import networkx as nx

logger = logging.getLogger(__name__)

PYVIS_AVAILABLE = False
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    logger.warning("PyVis not installed. Run: pip install pyvis")


def visualize_graph(
    G: nx.DiGraph,
    output_path: str = "graph.html",
    height: str = "580px",
    width: str = "100%",
    bgcolor: str = "#0d1117",
    font_color: str = "#c9d1d9",
    max_nodes: int = 300,
    physics: bool = True,
) -> str:
    """
    Create an interactive PyVis HTML visualization of the graph.

    Parameters
    ----------
    G : nx.DiGraph
    output_path : str   Path to write the HTML file.
    max_nodes : int     Truncate graph to top N nodes by degree.

    Returns
    -------
    str — path to generated HTML file (empty string on failure)
    """
    if G.number_of_nodes() == 0:
        logger.warning("Empty graph — skipping visualization.")
        return ""

    if not PYVIS_AVAILABLE:
        logger.warning("PyVis unavailable — skipping visualization.")
        return ""

    # Truncate to most connected nodes for performance
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(
            G.nodes, key=lambda n: G.degree(n), reverse=True
        )[:max_nodes]
        G = G.subgraph(top_nodes).copy()
        logger.info("Truncated graph to top %d nodes by degree.", max_nodes)

    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        directed=True,
    )

    # Degree-based node sizing
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1

    for node in G.nodes:
        degree = degrees.get(node, 1)
        size = 8 + 30 * (degree / max_degree)
        color = _degree_color(degree, max_degree)
        net.add_node(
            node,
            label=str(node)[:25],  # Truncate long labels
            title=str(node),
            size=size,
            color=color,
            font={"size": 12, "color": font_color},
        )

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)
        relations = data.get("relations", [])
        rel_label = relations[0] if relations else ""
        title = ", ".join(relations[:5]) if relations else ""
        edge_color = _relation_color(rel_label)
        net.add_edge(
            u, v,
            value=max(float(weight), 0.5),
            title=title,
            label=rel_label[:18] if rel_label not in {"co-occurrence", "near", "related-to", "page-bridge", "cross-page", "continues-in"} else "",
            color={"color": edge_color, "highlight": "#f0c040", "opacity": 0.75},
            arrows="to",
            width=max(1.0, min(float(weight), 5.0)),
        )

    if physics:
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.3
            },
            "solver": "barnesHut",
            "stabilization": { "enabled": true, "iterations": 300 },
            "minVelocity": 0.75
          },
          "edges": {
            "smooth": { "enabled": true, "type": "dynamic" },
            "scaling": { "min": 1, "max": 6 },
            "font": { "size": 9, "color": "#aaaaaa", "align": "middle" }
          },
          "nodes": { "scaling": { "min": 8, "max": 40 } },
          "interaction": {
            "hover": true,
            "tooltipDelay": 80,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true
          }
        }
        """)

    net.save_graph(output_path)
    logger.info("Graph visualization saved to: %s", output_path)
    return output_path


def graph_stats_summary(G: nx.DiGraph) -> Dict[str, Any]:
    """Return a dict of basic graph statistics."""
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "density": 0.0, "components": 0}

    undirected = G.to_undirected()
    components = nx.number_connected_components(undirected)
    density = nx.density(G)

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(density, 6),
        "components": components,
        "avg_degree": round(
            sum(d for _, d in G.degree()) / G.number_of_nodes(), 2
        ),
    }


def _degree_color(degree: int, max_degree: int) -> str:
    """Map node degree to a color on a blue→red gradient."""
    ratio = degree / max(max_degree, 1)
    r = int(255 * ratio)
    b = int(255 * (1 - ratio))
    return f"#{r:02x}44{b:02x}"


def _relation_color(relation: str) -> str:
    """Color-code edges by relation type."""
    palette = {
        "co-occurrence": "#2a3a4a",
        "near":          "#2a3a4a",
        "related-to":    "#334455",
        "page-bridge":   "#223322",
        "cross-page":    "#223322",
        "continues-in":  "#332233",
    }
    if relation in palette:
        return palette[relation]
    # Semantic verbs get a brighter color
    return "#4a7a9b"
