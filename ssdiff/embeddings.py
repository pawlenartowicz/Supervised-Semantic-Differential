"""Lightweight word-embedding container — drop-in replacement for gensim KeyedVectors."""

from __future__ import annotations

import gzip
import os
import pickle
import numpy as np
from typing import List, Optional, Sequence, Tuple


class Embeddings:
    """Stores word vectors and provides lookup / nearest-neighbor search.

    Designed as a minimal, API-compatible replacement for
    ``gensim.models.KeyedVectors`` covering only the surface used by ssdiff.
    """

    def __init__(
        self,
        keys: Sequence[str],
        vectors: np.ndarray,
    ) -> None:
        self.index_to_key: List[str] = list(keys)
        self.key_to_index: dict[str, int] = {
            w: i for i, w in enumerate(self.index_to_key)
        }
        self.vectors: np.ndarray = np.asarray(vectors, dtype=np.float32)
        self.vector_size: int = self.vectors.shape[1] if self.vectors.ndim == 2 else 0
        self._norms: Optional[np.ndarray] = None
        self._normed_vectors: Optional[np.ndarray] = None

    # ---- construction helpers ----

    @classmethod
    def empty(cls, vector_size: int) -> "Embeddings":
        """Create an empty Embeddings (for add_vectors pattern)."""
        return cls([], np.zeros((0, vector_size), dtype=np.float32))

    def add_vectors(self, keys: Sequence[str], vectors: np.ndarray) -> None:
        """Append words + vectors (used by normalize_kv)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.shape[0] == 0:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        for w in keys:
            self.key_to_index[w] = len(self.index_to_key)
            self.index_to_key.append(w)
        self.vector_size = self.vectors.shape[1]
        self._norms = None
        self._normed_vectors = None

    def fill_norms(self) -> None:
        """Precompute L2 norms."""
        self._norms = np.linalg.norm(self.vectors, axis=1)
        self._normed_vectors = None

    # ---- properties ----

    @property
    def norms(self) -> np.ndarray:
        if self._norms is None:
            self.fill_norms()
        return self._norms  # type: ignore[return-value]

    def get_normed_vectors(self) -> np.ndarray:
        """Return L2-normalized copy of the vector matrix."""
        if self._normed_vectors is None:
            n = self.norms.copy()
            n[n == 0] = 1e-12
            self._normed_vectors = self.vectors / n[:, None]
        return self._normed_vectors

    # ---- lookup ----

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_index

    def __len__(self) -> int:
        return len(self.index_to_key)

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.key_to_index[word]]

    def get_vector(self, word: str, norm: bool = False) -> np.ndarray:
        idx = self.key_to_index[word]
        if norm:
            return self.get_normed_vectors()[idx]
        return self.vectors[idx]

    # ---- persistence ----

    def save(self, path: str) -> None:
        """Save to pickle + ``.vectors.npy`` sidecar (round-trips with ``_load_pickle``)."""
        npy_path = path + ".vectors.npy"
        np.save(npy_path, self.vectors)
        saved_vectors = self.vectors
        self.vectors = np.zeros((0, self.vector_size), dtype=np.float32)
        self._norms = None
        self._normed_vectors = None
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self.vectors = saved_vectors
            self._norms = None
            self._normed_vectors = None

    # ---- similarity ----

    def similar_by_vector(
        self,
        vector: np.ndarray,
        topn: int = 10,
        restrict_vocab: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Return (word, cosine) pairs, most similar first."""
        vec = np.asarray(vector, dtype=np.float32)
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return []
        vec = vec / vec_norm

        vecs = self.get_normed_vectors()
        if restrict_vocab is not None:
            vecs = vecs[:restrict_vocab]

        if len(vecs) == 0:
            return []

        sims = vecs @ vec
        count = min(topn, len(sims))
        top_idx = np.argpartition(-sims, min(count, len(sims) - 1))[:count]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        keys = self.index_to_key
        return [(keys[i], float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# File loaders
# ---------------------------------------------------------------------------


def _first_line_tokens(path: str) -> list[str]:
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        return f.readline().strip().split()


def _load_text(path: str, binary: bool = False) -> Embeddings:
    """Load word2vec text or binary format."""
    if binary:
        return _load_word2vec_binary(path, is_gz=path.lower().endswith(".gz"))

    opener = gzip.open if path.lower().endswith(".gz") else open
    words: list[str] = []
    vecs: list[list[float]] = []

    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
        toks = first_line.split()

        if len(toks) == 2 and toks[0].isdigit() and toks[1].isdigit():
            pass  # skip header
        else:
            # No header — first line is data (GloVe-style)
            word, *vals = toks
            words.append(word)
            vecs.append([float(v) for v in vals])

        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            word = parts[0]
            vals = [float(v) for v in parts[1:]]
            words.append(word)
            vecs.append(vals)

    mat = np.array(vecs, dtype=np.float32)
    return Embeddings(words, mat)


def _load_word2vec_binary(path: str, is_gz: bool = False) -> Embeddings:
    """Load word2vec binary format (.bin)."""
    opener = gzip.open if is_gz else open
    words: list[str] = []
    vecs: list[np.ndarray] = []

    with opener(path, "rb") as f:
        header = f.readline().decode("utf-8").strip()
        vocab_size, dim = (int(x) for x in header.split())

        for _ in range(vocab_size):
            word_bytes = bytearray()
            while True:
                ch = f.read(1)
                if ch == b" " or ch == b"\t":
                    break
                if ch == b"\n" or ch == b"":
                    continue
                word_bytes.extend(ch)
            word = word_bytes.decode("utf-8", errors="ignore")
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
            words.append(word)
            vecs.append(vec)

    mat = np.vstack(vecs)
    return Embeddings(words, mat)


class _GensimUnpickler(pickle.Unpickler):
    """Intercept gensim classes during unpickling and map to Embeddings."""

    def find_class(self, module: str, name: str) -> type:
        if "KeyedVectors" in name or "Word2VecKeyedVectors" in name:
            return _GensimKVShim
        if module.startswith("gensim"):
            # Gensim pickles reference helper classes (e.g. gensim.utils.SaveLoad).
            # When gensim is not installed we need a harmless stand-in.
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, ImportError):
                return _GensimKVShim
        return super().find_class(module, name)


class _GensimKVShim:
    """Temporary shim that receives gensim pickle state, then converts to Embeddings."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        # Handle gensim 3.x → 4.x attribute renames
        if hasattr(self, "index2word") and not hasattr(self, "index_to_key"):
            self.index_to_key = self.index2word
        if hasattr(self, "index2entity") and not hasattr(self, "index_to_key"):
            self.index_to_key = self.index2entity
        if hasattr(self, "syn0") and not hasattr(self, "vectors"):
            self.vectors = self.syn0

    def to_embeddings(self) -> Embeddings:
        keys = list(self.index_to_key)
        vecs = np.asarray(self.vectors, dtype=np.float32)
        return Embeddings(keys, vecs)


def _load_pickle(path: str) -> Embeddings:
    """Load pickle-based embeddings: .sddemb (ssdiff) or .kv (gensim) format."""
    # Sidecar is always next to the uncompressed file, not the .gz wrapper
    base = path[: -len(".gz")] if path.lower().endswith(".gz") else path
    vectors_npy = base + ".vectors.npy"
    has_sidecar = os.path.exists(vectors_npy)

    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rb") as f:
        shim = _GensimUnpickler(f).load()

    if isinstance(shim, _GensimKVShim):
        if has_sidecar and (
            not hasattr(shim, "vectors")
            or shim.vectors is None
            or (hasattr(shim.vectors, "shape") and shim.vectors.shape[0] == 0)
        ):
            shim.vectors = np.load(vectors_npy, mmap_mode="r")
        return shim.to_embeddings()

    if isinstance(shim, Embeddings):
        if has_sidecar and shim.vectors.shape[0] == 0:
            shim.vectors = np.asarray(np.load(vectors_npy), dtype=np.float32)
            shim.vector_size = shim.vectors.shape[1]
            shim._norms = None
            shim._normed_vectors = None
        return shim

    raise ValueError(f"Cannot load pickle embeddings: unexpected object type {type(shim)}")


def load_embeddings(path: str) -> Embeddings:
    """
    Load pre-trained word embeddings from file.

    Supports:
      - .sddemb     (ssdiff native pickle + .npy sidecar — fastest)
      - .kv         (gensim KeyedVectors pickle — legacy, also supported)
      - .bin        (word2vec binary)
      - .txt, .vec  (word2vec/GloVe text, auto-detects header)
      - .gz         (gzip-compressed variants of the above)
    """
    low = path.lower()
    ext = os.path.splitext(low)[1]

    if ext == ".sddemb" or low.endswith(".sddemb.gz"):
        return _load_pickle(path)

    if ext == ".kv" or low.endswith(".kv.gz"):
        return _load_pickle(path)

    if ext == ".bin" or low.endswith(".bin.gz"):
        return _load_text(path, binary=True)

    if ext in {".txt", ".vec"} or low.endswith(".txt.gz") or low.endswith(".vec.gz"):
        return _load_text(path, binary=False)

    if ext == ".gz":
        raise ValueError(
            f"Cannot determine embedding format for '{path}'. "
            "Rename to .txt.gz, .vec.gz, .bin.gz, or .sddemb.gz."
        )

    # Fallback: try as pickle
    return _load_pickle(path)
