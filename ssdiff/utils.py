# ssdiff/utils.py
import numpy as np
from .embeddings import Embeddings, load_embeddings as _load_embeddings
from typing import Iterable, List, Tuple, Dict, Union
import re

_bad_token = re.compile(r".*\d|^[A-ZĄĆĘŁŃÓŚŹŻ]")


def normalize_kv(
    kv: Embeddings,
    *,
    l2: bool = True,
    abtt_m: int = 0,
    re_normalize: bool = True,
) -> Embeddings:
    """
    Return a NEW Embeddings with optional:
      1) L2 normalization of rows
      2) ABTT: center & remove top-m PCs
      3) re-normalize rows (recommended)
    """
    keys = list(kv.index_to_key)
    V = (
        kv.get_normed_vectors().astype(np.float64)
        if l2
        else kv.vectors.astype(np.float64)
    )

    if abtt_m > 0:
        mu = V.mean(axis=0)
        Vc = V - mu
        U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
        m = min(abtt_m, Vt.shape[0])
        top = Vt[:m, :]
        P = np.eye(Vt.shape[1]) - top.T @ top
        V = Vc @ P

    if re_normalize:
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        V = V / norms

    kv_t = Embeddings.empty(V.shape[1])
    kv_t.add_vectors(keys, V.astype(np.float32))
    kv_t.fill_norms()
    return kv_t


def compute_global_sif(sentences: List[List[str]]) -> Tuple[Dict[str, int], int]:
    wc: Dict[str, int] = {}
    for sent in sentences:
        for t in sent:
            wc[t] = wc.get(t, 0) + 1
    return wc, sum(wc.values())


def build_doc_vectors(docs, kv, lexicon, global_wc, total_tokens, window, sif_a):
    X_list = []
    keep_mask = []
    for doc in docs:
        v = _doc_vector(doc, kv, lexicon, global_wc, total_tokens, window, sif_a)
        if v is None:
            keep_mask.append(False)
        else:
            keep_mask.append(True)
            X_list.append(v)
    X = np.vstack(X_list) if X_list else np.zeros((0, kv.vector_size), dtype=np.float64)
    return X, np.array(keep_mask, dtype=bool)


def _doc_vector(doc, kv, lexicon, wc, tot, window, sif_a) -> np.ndarray | None:
    occ = []
    D = kv.vector_size
    for i, token in enumerate(doc):
        if token not in lexicon:
            continue
        start, end = max(0, i - window), min(len(doc), i + window + 1)
        sum_v = np.zeros(D, dtype=np.float64)
        w_sum = 0.0
        for j in range(start, end):
            if j == i:
                continue
            c = doc[j]
            if c not in kv:
                continue
            a = sif_a / (sif_a + wc.get(c, 0) / tot)
            sum_v += a * kv[c]
            w_sum += a
        if w_sum > 0:
            occ.append(sum_v / w_sum)

    if not occ:
        return None

    return np.mean(occ, axis=0).astype(np.float64)


def load_embeddings(path: str) -> Embeddings:
    """Load embeddings from file. Delegates to ssdiff.embeddings.load_embeddings."""
    return _load_embeddings(path)


def _iter_token_lists(docs: list) -> Iterable[list[str]]:
    """
    Yield token lists from either:
      - docs = List[List[str]]               (single-level)
      - docs = List[List[List[str]]]         (profiles with multiple posts)
    Empty lists are safely skipped.
    """
    for item in docs:
        if not item:
            continue
        # single document (list[str])
        if isinstance(item[0], str):
            yield item
        else:
            # list of posts (list[list[str]])
            for sub in item:
                if sub:
                    yield sub


def _occ_vectors_in_doc(doc, kv, lexicon, wc, tot, window, sif_a):
    """
    Return a list of SIF-averaged context vectors for EACH occurrence of a seed in `doc`.
    Mirrors _doc_vector logic but exposes each occurrence separately.
    """
    occ = []
    D = kv.vector_size
    for i, token in enumerate(doc):
        if token not in lexicon:
            continue
        start, end = max(0, i - window), min(len(doc), i + window + 1)
        sum_v = np.zeros(D, dtype=np.float64)
        w_sum = 0.0
        for j in range(start, end):
            if j == i:
                continue
            c = doc[j]
            if c not in kv:
                continue
            a = sif_a / (sif_a + wc.get(c, 0) / tot)
            sum_v += a * kv[c]
            w_sum += a
        if w_sum > 0:
            occ.append(sum_v / w_sum)
    return occ


def build_doc_vectors_grouped(
    docs,
    kv,
    lexicon,
    global_wc,
    total_tokens,
    window,
    sif_a,
    *,
    mode: str = "seed",  # NEW: "seed" (default) or "full"
):
    """
    docs can be:
      - List[List[str]]          -> one PCV per doc
      - List[List[List[str]]]    -> one PCV per profile (aggregate over posts)

    mode:
      - "seed" : old behavior, use SIF windows around lexicon seeds
      - "full" : ignore lexicon, use full-doc SIF vectors
    """
    if mode not in {"seed", "full"}:
        raise ValueError("mode must be 'seed' or 'full'.")

    use_seeds = mode == "seed"

    X_list = []
    keep_mask = []

    # Detect mode from first non-empty item (flat vs grouped docs)
    mode_docs = None  # "flat" or "grouped"
    for it in docs:
        if it:
            mode_docs = "flat" if isinstance(it[0], str) else "grouped"
            break
    if mode_docs is None:
        return np.zeros((0, kv.vector_size), dtype=np.float64), np.zeros(
            (0,), dtype=bool
        )

    # --- FLAT DOCS: List[List[str]] ---
    if mode_docs == "flat":
        for d in docs:
            if not d:
                keep_mask.append(False)
                continue

            if use_seeds:
                # existing behavior via per-occurrence contexts
                occ = _occ_vectors_in_doc(
                    d, kv, lexicon, global_wc, total_tokens, window, sif_a
                )
                if not occ:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)
                    X_list.append(np.mean(occ, axis=0).astype(np.float64))
            else:
                # NEW: full-doc SIF, no lexicon
                v = _full_doc_vector(d, kv, global_wc, total_tokens, sif_a)
                if v is None:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)
                    X_list.append(v)

    # --- GROUPED DOCS: List[List[List[str]]] ---
    else:
        for posts in docs:
            if not posts:
                keep_mask.append(False)
                continue

            if use_seeds:
                occ_all = []
                for p in posts:
                    if not p:
                        continue
                    occ_all.extend(
                        _occ_vectors_in_doc(
                            p, kv, lexicon, global_wc, total_tokens, window, sif_a
                        )
                    )
                if not occ_all:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)
                    X_list.append(np.mean(occ_all, axis=0).astype(np.float64))
            else:
                # full-doc variant: aggregate all tokens from all posts
                tokens_all = [t for p in posts for t in p]
                v = _full_doc_vector(tokens_all, kv, global_wc, total_tokens, sif_a)
                if v is None:
                    keep_mask.append(False)
                else:
                    keep_mask.append(True)
                    X_list.append(v)

    X = np.vstack(X_list) if X_list else np.zeros((0, kv.vector_size), dtype=np.float64)
    return X, np.array(keep_mask, dtype=bool)


def _full_doc_vector(tokens, kv, wc, tot, sif_a) -> np.ndarray | None:
    """
    SIF-weighted mean of *all* tokens in a doc.

    tokens : list[str]
    wc     : global word counts
    tot    : global token count
    """
    D = kv.vector_size
    sum_v = np.zeros(D, dtype=np.float64)
    w_sum = 0.0

    for c in tokens:
        if c not in kv:
            continue
        a = sif_a / (sif_a + wc.get(c, 0) / tot)
        sum_v += a * kv[c]
        w_sum += a

    if w_sum == 0.0:
        return None

    return (sum_v / w_sum).astype(np.float64)


def filtered_neighbors(
    kv: Embeddings,
    vec: Union[List[float], np.ndarray],
    topn: int = 20,
    cand: int = 2000,
    restrict: int = 10000,
):
    nbrs = kv.similar_by_vector(vec, topn=cand, restrict_vocab=restrict)
    out = []
    for w, sim in nbrs:
        if not _bad_token.match(w):
            out.append((w, sim))
            if len(out) >= topn:
                break
    return out
