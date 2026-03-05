"""
Test Suite — PDF Graph GAT Backend
===================================
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.graph_builder import build_graph, graph_to_dict, get_graph_stats
from modules.adjacency_builder import (
    create_adjacency_matrix,
    adjacency_to_edge_index,
    _normalize_adjacency,
)
from modules.feature_builder import create_feature_matrix, create_tfidf_features


# ─── Graph Builder ────────────────────────────────────────────────────────────

class TestGraphBuilder:

    def test_build_graph_basic(self):
        entities = ["Alice", "Bob", "Acme Corp"]
        relations = [("Alice", "works_at", "Acme Corp"), ("Bob", "knows", "Alice")]
        G = build_graph(entities, relations)
        assert G.number_of_nodes() >= 3
        assert G.number_of_edges() == 2

    def test_build_graph_empty(self):
        G = build_graph([], [])
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_graph_to_dict(self):
        entities = ["A", "B"]
        relations = [("A", "rel", "B")]
        G = build_graph(entities, relations)
        d = graph_to_dict(G)
        assert "nodes" in d
        assert "edges" in d
        assert d["num_nodes"] >= 2
        assert d["num_edges"] == 1

    def test_graph_stats(self):
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C")])
        stats = get_graph_stats(G)
        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 2

    def test_duplicate_entities(self):
        entities = ["Alice", "Alice", "Bob"]
        G = build_graph(entities, [])
        # Deduplication via Counter
        assert G.number_of_nodes() == 2


# ─── Adjacency Matrix ─────────────────────────────────────────────────────────

class TestAdjacencyBuilder:

    def setup_method(self):
        self.G = nx.DiGraph()
        self.G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

    def test_shape(self):
        adj, node_map = create_adjacency_matrix(self.G)
        assert adj.shape == (3, 3)

    def test_node_map(self):
        _, node_map = create_adjacency_matrix(self.G)
        assert set(node_map.keys()) == {"A", "B", "C"}

    def test_self_loops(self):
        adj, _ = create_adjacency_matrix(self.G, add_self_loops=True)
        assert adj[0, 0] == 1.0

    def test_no_self_loops(self):
        adj, _ = create_adjacency_matrix(self.G, add_self_loops=False)
        # Off-diagonal edges only
        assert adj[0, 0] == 0.0

    def test_normalization(self):
        adj, _ = create_adjacency_matrix(self.G, normalize=True)
        # Normalized values should be in [0, 1]
        assert np.all(adj >= 0)
        assert np.all(adj <= 1)

    def test_edge_index_shape(self):
        adj, _ = create_adjacency_matrix(self.G)
        ei = adjacency_to_edge_index(adj)
        assert ei.shape[0] == 2
        assert ei.shape[1] > 0

    def test_empty_graph(self):
        G = nx.DiGraph()
        adj, node_map = create_adjacency_matrix(G)
        assert adj.shape == (0, 0)


# ─── Feature Builder ─────────────────────────────────────────────────────────

class TestFeatureBuilder:

    def test_tfidf_features_shape(self):
        nodes = ["machine learning", "neural network", "graph attention"]
        features = create_tfidf_features(nodes)
        assert features.shape[0] == 3
        assert features.shape[1] > 0

    def test_tfidf_empty(self):
        features = create_tfidf_features([])
        assert features.shape[0] == 0

    def test_feature_dtype(self):
        nodes = ["entity one", "entity two"]
        features = create_tfidf_features(nodes)
        assert features.dtype == np.float32


# ─── Integration ─────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_small(self):
        """Simulate a mini pipeline without PDF/NLP."""
        entities = ["Paris", "France", "Eiffel Tower", "Tourism"]
        relations = [
            ("Paris", "capital_of", "France"),
            ("Eiffel Tower", "located_in", "Paris"),
            ("France", "known_for", "Tourism"),
        ]

        G = build_graph(entities, relations)
        assert G.number_of_nodes() >= 4

        adj, node_map = create_adjacency_matrix(G)
        assert adj.shape[0] == G.number_of_nodes()

        nodes = list(G.nodes())
        features = create_tfidf_features(nodes)
        assert features.shape[0] == len(nodes)

        graph_dict = graph_to_dict(G)
        assert graph_dict["num_nodes"] == G.number_of_nodes()
