"""
modules/graph_builder.py
========================
Builds and merges NetworkX directed graphs from entity/relation lists.

Key improvements for dense connectivity:
- Bidirectional co-occurrence edges (both directions)
- Cross-page entity bridging (nodes seen on multiple pages get linked)
- Sliding-window page-level proximity edges
- Degree-based pruning of isolated singletons
"""

from __future__ import annotations
import logging
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple, Dict

import networkx as nx

logger = logging.getLogger(__name__)


def build_graph(
    entities: List[str],
    relations: List[Tuple[str, str, str]],
    add_co_occurrence: bool = True,
    co_occurrence_window: int = 10,
) -> nx.DiGraph:
    """
    Build a directed knowledge graph from entities and relations.

    Parameters
    ----------
    entities : List[str]
    relations : List[Tuple[str, str, str]]
    add_co_occurrence : bool
        Connect entities that appear on the same page.
    co_occurrence_window : int
        For ordered entity lists, only connect entities within this
        positional window (avoids O(n²) explosion on large pages).

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()

    for entity in entities:
        e = entity.strip()
        if e:
            G.add_node(e, label=e)

    for rel in relations:
        if len(rel) != 3:
            continue
        source, relation, target = [r.strip() for r in rel]
        if not source or not target or source == target:
            continue

        G.add_node(source)
        G.add_node(target)

        # Accumulate edge weight if edge already exists
        for s, t in [(source, target), (target, source)]:  # bidirectional
            if G.has_edge(s, t):
                G[s][t]["weight"] = G[s][t].get("weight", 1) + 1
                rels = G[s][t].get("relations", [])
                if relation not in rels:
                    rels.append(relation)
            else:
                G.add_edge(s, t, weight=1, relations=[relation])

    # ── Windowed co-occurrence edges ───────────────────────────
    if add_co_occurrence and len(entities) > 1:
        n = len(entities)
        for i in range(n):
            e1 = entities[i].strip()
            window_end = min(i + co_occurrence_window + 1, n)
            for j in range(i + 1, window_end):
                e2 = entities[j].strip()
                if e1 == e2:
                    continue
                # Add both directions
                for s, t in [(e1, e2), (e2, e1)]:
                    if not G.has_edge(s, t):
                        G.add_edge(s, t, weight=0.5, relations=["co-occurrence"])
                    else:
                        G[s][t]["weight"] = G[s][t].get("weight", 0.5) + 0.1

    logger.debug("Built graph: %d nodes, %d edges",
                 G.number_of_nodes(), G.number_of_edges())
    return G


def merge_page_graphs(
    graphs: List[nx.DiGraph],
    add_cross_page_bridges: bool = True,
) -> nx.DiGraph:
    """
    Merge per-page graphs into a unified graph.

    Cross-page bridging: if the same entity node appears on multiple pages,
    connect it to all entities it was co-located with on each page.
    This prevents isolated per-page clusters.

    Parameters
    ----------
    graphs : List[nx.DiGraph]
    add_cross_page_bridges : bool
        Connect entities shared between consecutive pages.

    Returns
    -------
    nx.DiGraph
    """
    merged = nx.DiGraph()

    # Track which nodes appeared on which pages
    node_to_pages: Dict[str, List[int]] = defaultdict(list)

    for page_idx, G in enumerate(graphs):
        for node, data in G.nodes(data=True):
            node_to_pages[node].append(page_idx)
            if not merged.has_node(node):
                merged.add_node(node, **data)

        for u, v, data in G.edges(data=True):
            if merged.has_edge(u, v):
                merged[u][v]["weight"] = (
                    merged[u][v].get("weight", 1) + data.get("weight", 1)
                )
                existing = merged[u][v].get("relations", [])
                for r in data.get("relations", []):
                    if r not in existing:
                        existing.append(r)
            else:
                merged.add_edge(u, v, **data)

    # ── Cross-page bridges ─────────────────────────────────────
    if add_cross_page_bridges and len(graphs) > 1:
        # For each pair of consecutive pages, connect shared or
        # neighboring entities to stitch pages together.
        for page_idx in range(len(graphs) - 1):
            G_curr = graphs[page_idx]
            G_next = graphs[page_idx + 1]

            nodes_curr = set(G_curr.nodes)
            nodes_next = set(G_next.nodes)

            shared = nodes_curr & nodes_next
            if shared:
                # Shared nodes already exist — add cross edges between
                # their neighbors across pages
                for shared_node in list(shared)[:20]:   # cap
                    neighbors_curr = list(G_curr.neighbors(shared_node))[:5]
                    neighbors_next = list(G_next.neighbors(shared_node))[:5]
                    for nc in neighbors_curr:
                        for nn in neighbors_next:
                            if nc != nn and not merged.has_edge(nc, nn):
                                merged.add_edge(
                                    nc, nn,
                                    weight=0.3,
                                    relations=["cross-page"]
                                )
            else:
                # No shared nodes — connect the highest-degree nodes from
                # each page to prevent total disconnection
                top_curr = _top_degree_nodes(G_curr, k=3)
                top_next = _top_degree_nodes(G_next, k=3)
                for nc in top_curr:
                    for nn in top_next:
                        if not merged.has_edge(nc, nn):
                            merged.add_edge(
                                nc, nn,
                                weight=0.2,
                                relations=["page-bridge"]
                            )

    logger.info(
        "Merged %d graphs → %d nodes, %d edges",
        len(graphs), merged.number_of_nodes(), merged.number_of_edges()
    )
    return merged


def prune_isolates(G: nx.DiGraph, min_degree: int = 1) -> nx.DiGraph:
    """
    Remove nodes with degree below threshold.
    Useful for cleaning up noisy singleton nodes.
    """
    isolates = [n for n, d in G.degree() if d < min_degree]
    G.remove_nodes_from(isolates)
    if isolates:
        logger.debug("Pruned %d isolated nodes.", len(isolates))
    return G


def _top_degree_nodes(G: nx.DiGraph, k: int = 5) -> List[str]:
    if G.number_of_nodes() == 0:
        return []
    return sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:k]
