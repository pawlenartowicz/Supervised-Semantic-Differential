"""Pure-numpy replacements for sklearn StandardScaler, PCA, KMeans, silhouette_score."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardize columns. Returns (X_standardized, mean, scale)."""
    mean = X.mean(axis=0, dtype=np.float64)
    scale = X.std(axis=0, dtype=np.float64, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    Xs = (X - mean) / scale
    return Xs, mean, scale


def pca_fit_transform(
    X: np.ndarray, n_components: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via full SVD (matches sklearn svd_solver='full').

    Parameters
    ----------
    X : (n, d) array.  Centering is applied internally.
    n_components : number of components to keep.

    Returns
    -------
    z : (n, n_components) projected data
    components : (n_components, d) principal axes
    explained_variance_ratio : (n_components,) fraction of variance per component
    """
    n = X.shape[0]
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    components = Vt[:n_components]
    z = Xc @ components.T

    explained_var = (S**2) / (n - 1)
    total_var = explained_var.sum()
    evr = (
        explained_var[:n_components] / total_var
        if total_var > 0
        else np.zeros(n_components)
    )

    return z, components, evr


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between rows of X (n,d) and C (k,d).

    Uses BLAS matrix-multiply: ||x-c||^2 = ||x||^2 - 2*x.c + ||c||^2
    Avoids materializing an (n, k, d) intermediate array.
    """
    X_sq = np.einsum("ij,ij->i", X, X)  # (n,)
    C_sq = np.einsum("ij,ij->i", C, C)  # (k,)
    D = X_sq[:, None] + C_sq[None, :] - 2.0 * (X @ C.T)
    np.maximum(D, 0.0, out=D)  # clamp rounding noise
    return D


def _kmeans_plus_plus(
    X: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """K-means++ initialization: picks initial centers spread apart.

    O(n*k*d) but avoids large temporaries by updating distances incrementally.
    """
    n = X.shape[0]
    first = rng.integers(n)
    centers = [first]
    min_dists = np.full(n, np.inf, dtype=np.float64)

    for _ in range(1, k):
        last = X[centers[-1]]
        d = np.sum((X - last) ** 2, axis=1)
        np.minimum(min_dists, d, out=min_dists)

        total = min_dists.sum()
        if total == 0:
            # All remaining points coincide with existing centers
            idx = rng.integers(n)
        else:
            probs = min_dists / total
            idx = rng.choice(n, p=probs)
        centers.append(idx)

    return X[centers].copy()


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    random_state: int | None = None,
    max_iter: int = 300,
    n_init: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """K-Means clustering via Lloyd's algorithm with k-means++ initialization.

    Returns (labels, centers, inertia).
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    if k > n:
        raise ValueError(f"Cannot request k={k} clusters from n={n} samples.")
    best_inertia = np.inf
    best_labels = np.zeros(n, dtype=np.intp)
    best_centers = np.zeros((k, d), dtype=X.dtype)

    for _ in range(n_init):
        centers = _kmeans_plus_plus(X, k, rng)

        for _ in range(max_iter):
            dists = _sq_dists(X, centers)
            labels = np.argmin(dists, axis=1)

            # Vectorized center update via bincount
            new_centers = np.zeros((k, d), dtype=np.float64)
            counts = np.bincount(labels, minlength=k)
            for dim in range(d):
                new_centers[:, dim] = np.bincount(
                    labels, weights=X[:, dim], minlength=k
                )
            nonempty = counts > 0
            new_centers[nonempty] /= counts[nonempty, None]

            # Reassign empty clusters to random points
            empty = ~nonempty
            if empty.any():
                new_centers[empty] = X[rng.integers(n, size=int(empty.sum()))]

            new_centers = new_centers.astype(X.dtype)
            if np.allclose(centers, new_centers, rtol=1e-6, atol=1e-6):
                break
            centers = new_centers

        # Final assignment with accurate inertia
        dists = _sq_dists(X, centers)
        labels = np.argmin(dists, axis=1)
        inertia = float(np.min(dists, axis=1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


# ---------------------------------------------------------------------------
# Silhouette score
# ---------------------------------------------------------------------------


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (Euclidean).

    Vectorized: loops over clusters (k) instead of samples (n).
    """
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Pairwise Euclidean distances via BLAS trick
    X_sq = np.einsum("ij,ij->i", X, X)
    D_sq = X_sq[:, None] + X_sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D_sq, 0.0, out=D_sq)
    dists = np.sqrt(D_sq)

    a = np.zeros(n, dtype=np.float64)
    b = np.full(n, np.inf, dtype=np.float64)

    for lab in unique_labels:
        mask = labels == lab
        cluster_size = int(mask.sum())

        # Intra-cluster: mean distance to same-cluster members (excluding self)
        if cluster_size > 1:
            a[mask] = dists[np.ix_(mask, mask)].sum(axis=1) / (cluster_size - 1)

        # Inter-cluster: mean distance from non-members to this cluster
        not_mask = ~mask
        if not_mask.any() and cluster_size > 0:
            mean_to_cluster = dists[np.ix_(not_mask, mask)].mean(axis=1)
            b[not_mask] = np.minimum(b[not_mask], mean_to_cluster)

    # Points whose b stayed inf (sole member, only 1 other cluster) → silhouette 0
    finite = np.isfinite(b)
    denom = np.maximum(a, b)
    sil = np.zeros(n, dtype=np.float64)
    valid = finite & (denom > 0)
    sil[valid] = (b[valid] - a[valid]) / denom[valid]
    return float(sil.mean())


# ---------------------------------------------------------------------------
# F-distribution survival function (replaces scipy.stats.f.cdf)
# ---------------------------------------------------------------------------


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction.

    Uses the modified Lentz algorithm (Numerical Recipes) which converges
    rapidly for x < (a+1)/(a+b+2). For x above that threshold we use the
    symmetry relation I_x(a,b) = 1 - I_{1-x}(b,a).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1.0 - x)

    from math import lgamma, exp, log

    lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(a * log(x) + b * log(1.0 - x) - lbeta_ab) / a

    TINY = 1e-30
    EPS = 1e-14

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < TINY:
        d = TINY
    d = 1.0 / d
    h = d

    for m in range(1, 201):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        h *= d * c

        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + aa / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < EPS:
            break

    return front * h


def f_sf(f_val: float, dfn: float, dfd: float) -> float:
    """Survival function (1 - CDF) of the F-distribution.

    Equivalent to ``1 - scipy.stats.f.cdf(f_val, dfn, dfd)``.
    """
    if f_val <= 0 or dfn <= 0 or dfd <= 0:
        return 1.0
    x = dfn * f_val / (dfn * f_val + dfd)
    return 1.0 - _betainc(dfn / 2.0, dfd / 2.0, x)
