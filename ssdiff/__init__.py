# ssdiff/__init__.py

# --- Public API: core classes and loaders ---
from .embeddings import Embeddings, load_embeddings
from .core import SSD
from .crossgroup import SSDGroup, SSDContrast
from .sweep import pca_sweep, PCAKSelectionResult

# --- Public API: lexicon utilities ---
from .lexicon import suggest_lexicon, coverage_by_lexicon, token_presence_stats

# --- Public API: preprocessing ---
from .preprocess import (
    load_spacy,
    load_stopwords,
    preprocess_texts,
    build_docs_from_preprocessed,
)

# --- Internal helpers (exported for power users / backwards compat) ---
# These are implementation details used by SSD internals.
# They are not part of the stable public API and may change without notice.
from .clusters import cluster_top_neighbors  # used by SSD.cluster_neighbors*
from .utils import (
    normalize_kv,           # embedding pre-processing
    compute_global_sif,     # SIF weight computation
    build_doc_vectors,      # document vector construction
    filtered_neighbors,     # neighbor filtering with bad-token regex
)

__all__ = [
    # Public API
    "Embeddings",
    "load_embeddings",
    "SSD",
    "SSDGroup",
    "SSDContrast",
    "pca_sweep",
    "PCAKSelectionResult",
    "suggest_lexicon",
    "coverage_by_lexicon",
    "token_presence_stats",
    "load_spacy",
    "load_stopwords",
    "preprocess_texts",
    "build_docs_from_preprocessed",
    # Internal helpers (exported for backwards compat)
    "cluster_top_neighbors",
    "normalize_kv",
    "compute_global_sif",
    "build_doc_vectors",
    "filtered_neighbors",
]
