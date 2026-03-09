# ssdiff/clusters.py
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from .utils import filtered_neighbors

if TYPE_CHECKING:
    # Only used for hints; not imported at runtime → no circular import
    from .core import SSD
from ._math import kmeans as _kmeans, silhouette_score as _silhouette_score


def cluster_top_neighbors(
    ssd: SSD,
    *,
    topn: int = 100,
    k: int | None = None,
    k_min: int = 2,
    k_max: int = 10,
    restrict_vocab: int = 50000,
    random_state: int = 13,
    min_cluster_size: int = 2,
    side: str = "pos",  # "pos" → +β̂, "neg" → −β̂
):
    """Cluster the top neighbors of ±β̂ into themes using KMeans.
    Parameters
    ----------
    ssd : SSD
        Fitted SSD instance.
    topn : int, optional
        Number of top neighbors to consider for clustering (default is 100).
    k : int | None, optional
        Number of clusters to form. If None, k is chosen automatically using silhouette score (default is None).
    k_min : int, optional
        Minimum number of clusters to consider when auto-selecting k (default is 2).
    k_max : int, optional
        Maximum number of clusters to consider when auto-selecting k (default is 10).
    restrict_vocab : int, optional
        Limit neighbors search to the top N most frequent words in the vocabulary (default is 50000).
    random_state : int, optional
        Random seed for KMeans (default is 13).
    min_cluster_size : int, optional
        Minimum size of clusters to keep (default is 2).
    side : str, optional
        Which side to cluster: "pos" for +β̂, "neg" for −β̂ (default is "pos").
    Returns
    -------
    List[dict]
        A list of clusters with their details.

    ️Raises
    ------
    ValueError
        If there are not enough neighbors to form clusters.
    """

    b = ssd.beta_unit if ssd.use_unit_beta else ssd.beta
    vec = b if side == "pos" else -b

    pairs = filtered_neighbors(ssd.kv, vec, topn=topn, restrict=restrict_vocab)
    words = [w for (w, _s) in pairs]
    if len(words) < max(2, k_min):
        raise ValueError("Not enough neighbors to cluster.")

    W = np.vstack(
        [ssd.kv.get_vector(w, norm=True).astype(np.float64) for w in words]
    )  # unit rows

    def choose_k_auto(W, kmin, kmax):
        best_k, best_s = None, -1.0
        upper = min(kmax, max(kmin, W.shape[0] - 1))
        for kk in range(max(2, kmin), max(2, upper) + 1):
            labels, _, _ = _kmeans(W, k=kk, random_state=random_state)
            if len(set(labels)) <= 1 or np.max(np.bincount(labels)) <= 1:
                continue
            s = _silhouette_score(W, labels)
            if s > best_s:
                best_s, best_k = s, kk
        return best_k if best_k is not None else max(2, kmin)

    k_use = int(k) if k is not None else choose_k_auto(W, k_min, k_max)
    labels, _, _ = _kmeans(W, k=k_use, random_state=random_state)

    bu = b / max(float(np.linalg.norm(b)), 1e-12)
    clusters = []
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        if len(idx) < min_cluster_size:
            continue
        Wc = W[idx]
        centroid = Wc.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        cos_beta = float(centroid @ bu)
        cos_to_centroid = (Wc @ centroid).astype(float)
        coherence = float(np.mean(cos_to_centroid))

        rows = []
        for j in idx:
            w = words[j]
            ccent = float(W[j] @ centroid)
            cbeta = float(W[j] @ bu)
            rows.append((w, ccent, cbeta))
        rows.sort(key=lambda t: t[1], reverse=True)

        clusters.append(
            {
                "id": int(cid),
                "size": int(len(idx)),
                "centroid_cos_beta": cos_beta,
                "coherence": coherence,
                "words": rows,
            }
        )

    if side == "pos":
        clusters.sort(key=lambda C: C["centroid_cos_beta"], reverse=True)
    else:
        clusters.sort(key=lambda C: C["centroid_cos_beta"], reverse=False)

    return clusters
