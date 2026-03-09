# ssdiff/__init__.py
from .embeddings import Embeddings, load_embeddings
from .core import SSD
from .crossgroup import SSDGroup, SSDContrast
from .clusters import cluster_top_neighbors
from .utils import (
    normalize_kv,
    compute_global_sif,
    build_doc_vectors,
    filtered_neighbors,
)
from .lexicon import suggest_lexicon, coverage_by_lexicon, token_presence_stats
from .preprocess import (
    load_spacy,
    load_stopwords,
    preprocess_texts,
    build_docs_from_preprocessed,
)
from .sweep import pca_sweep

__all__ = [
    "Embeddings",
    "SSD",
    "SSDGroup",
    "SSDContrast",
    "cluster_top_neighbors",
    "load_embeddings",
    "normalize_kv",
    "compute_global_sif",
    "build_doc_vectors",
    "filtered_neighbors",
    "build_docs_from_preprocessed",
    "suggest_lexicon",
    "coverage_by_lexicon",
    "token_presence_stats",
    "load_spacy",
    "load_stopwords",
    "preprocess_texts",
    "pca_sweep",
]
