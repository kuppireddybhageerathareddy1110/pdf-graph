"""
modules/gat_model.py
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=dropout, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x


def run_gat_inference(
    feature_matrix: np.ndarray,
    adj_matrix: np.ndarray,
    hidden_dim: int = 64,
    output_dim: int = 32,
    heads: int = 4,
    dropout: float = 0.0,
) -> np.ndarray:
    """Run GAT in inference mode. Falls back to random embeddings if PyTorch/PyG not installed."""
    n = feature_matrix.shape[0]

    if n == 0:
        return np.zeros((0, output_dim), dtype=np.float32)

    if not (TORCH_AVAILABLE and PYG_AVAILABLE):
        logger.warning("PyTorch/PyG unavailable — returning placeholder embeddings.")
        return np.random.default_rng(42).standard_normal((n, output_dim)).astype(np.float32)

    from modules.adjacency_builder import adj_to_edge_index
    edge_index_np = adj_to_edge_index(adj_matrix)

    if edge_index_np.shape[1] == 0:
        pad = output_dim - feature_matrix.shape[1]
        return feature_matrix[:, :output_dim] if pad <= 0 else np.pad(feature_matrix, ((0,0),(0,pad)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)

    model = GAT(feature_matrix.shape[1], hidden_dim, output_dim, heads, dropout).to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model(x, edge_index)
    return embeddings.cpu().numpy()


def train_gat(
    feature_matrix: np.ndarray,
    adj_matrix: np.ndarray,
    labels: np.ndarray,
    hidden_dim: int = 64,
    output_dim: int = 32,
    heads: int = 4,
    epochs: int = 200,
    lr: float = 0.005,
    dropout: float = 0.3,
    train_mask: Optional[np.ndarray] = None,
) -> Tuple:
    if not (TORCH_AVAILABLE and PYG_AVAILABLE):
        raise ImportError("PyTorch and PyTorch Geometric are required for training.")

    from modules.adjacency_builder import adj_to_edge_index
    n = feature_matrix.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
    edge_index = torch.tensor(adj_to_edge_index(adj_matrix), dtype=torch.long).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    mask = torch.ones(n, dtype=torch.bool).to(device) if train_mask is None else torch.tensor(train_mask, dtype=torch.bool).to(device)

    model = GAT(feature_matrix.shape[1], hidden_dim, int(labels.max())+1, heads, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[mask], y[mask])
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            logger.info("Epoch %d  Loss: %.4f", epoch, loss.item())

    model.eval()
    with torch.no_grad():
        embeddings = model(x, edge_index).cpu().numpy()
    return model, embeddings
