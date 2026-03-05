"""
Adjacency Matrix Builder
========================
Converts a NetworkX graph to a numpy adjacency matrix.

Supports:
- Binary adjacency (unweighted)
- Weighted adjacency
- Normalized adjacency (D^-1/2 * A * D^-1/2) for GNNs
- Sparse representation for large graphs
"""

from typing import Tuple, Dict
import numpy as np
import networkx as nx


def create_adjacency_matrix(
    G: nx.DiGraph,
    weighted: bool = False,
    add_self_loops: bool = True,
    normalize: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Build an adjacency matrix from a graph.

    Args:
        G: NetworkX directed graph.
        weighted: If True, use edge weights. Otherwise binary (0/1).
        add_self_loops: Add identity to adjacency (helps GNN training).
        normalize: Apply symmetric normalization (GCN-style).

    Returns:
        adj_matrix: np.ndarray of shape [N, N]
        node_to_id: Dict mapping node label → integer index
    """
    nodes = list(G.nodes())
    node_to_id: Dict[str, int] = {node: i for i, node in enumerate(nodes)}
    size = len(nodes)

    adj = np.zeros((size, size), dtype=np.float32)

    for u, v, data in G.edges(data=True):
        i = node_to_id[u]
        j = node_to_id[v]
        weight = float(data.get("weight", 1)) if weighted else 1.0
        adj[i][j] = weight
        # Make undirected copy for GNN (symmetric)
        adj[j][i] = weight

    if add_self_loops:
        np.fill_diagonal(adj, 1.0)

    if normalize:
        adj = _normalize_adjacency(adj)

    return adj, node_to_id


def _normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}
    Standard for GCN-style models.
    """
    degree = np.array(adj.sum(axis=1), dtype=np.float32)
    # Avoid division by zero
    degree = np.where(degree == 0, 1e-10, degree)
    d_inv_sqrt = np.diag(np.power(degree, -0.5))
    return d_inv_sqrt @ adj @ d_inv_sqrt


def adjacency_to_edge_index(adj: np.ndarray) -> np.ndarray:
    """
    Convert adjacency matrix to edge_index format for PyTorch Geometric.

    Returns:
        edge_index: np.ndarray of shape [2, num_edges]
    """
    rows, cols = np.where(adj > 0)
    return np.array([rows, cols], dtype=np.int64)


def get_edge_weights(adj: np.ndarray) -> np.ndarray:
    """
    Extract non-zero edge weights from adjacency matrix.

    Returns:
        weights: np.ndarray of shape [num_edges]
    """
    rows, cols = np.where(adj > 0)
    return adj[rows, cols]
