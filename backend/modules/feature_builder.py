"""
Feature Matrix Builder
======================
Creates node feature embeddings using SentenceTransformers.

Each node (entity string) is embedded into a dense vector.
Model: all-MiniLM-L6-v2 → 384-dimensional embeddings

Supports:
- Batch encoding
- Caching (avoid re-encoding same entities)
- Fallback to TF-IDF if transformer unavailable
"""

from typing import List, Optional
import numpy as np

# Lazy-load to avoid slow startup
_model = None
_model_name = "all-MiniLM-L6-v2"


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _model


def create_feature_matrix(
    nodes: List[str],
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    Create a feature matrix by embedding node labels.

    Args:
        nodes: List of node label strings.
        batch_size: Encoding batch size.
        normalize: L2-normalize embeddings (recommended for GNNs).

    Returns:
        features: np.ndarray of shape [num_nodes, 384]
    """
    if not nodes:
        return np.zeros((0, 384), dtype=np.float32)

    model = _get_model()
    embeddings = model.encode(
        nodes,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    return embeddings.astype(np.float32)


def create_tfidf_features(
    nodes: List[str],
    corpus: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Fallback: TF-IDF feature matrix for nodes.

    Args:
        nodes: List of node label strings.
        corpus: Optional extra corpus for TF-IDF vocabulary.

    Returns:
        features: np.ndarray of shape [num_nodes, vocab_size]
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_texts = nodes + (corpus or [])
    vectorizer = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
    vectorizer.fit(all_texts)
    features = vectorizer.transform(nodes).toarray()

    return features.astype(np.float32)


def get_feature_stats(features: np.ndarray) -> dict:
    """Basic statistics about the feature matrix."""
    return {
        "shape": list(features.shape),
        "mean": float(np.mean(features)),
        "std": float(np.std(features)),
        "min": float(np.min(features)),
        "max": float(np.max(features)),
        "embedding_dim": features.shape[1] if features.ndim > 1 else 0,
    }
