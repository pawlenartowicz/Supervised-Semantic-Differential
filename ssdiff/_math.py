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
    X : (n, d) array.  Centering is applied internally (matching sklearn).
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


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    random_state: int | None = None,
    max_iter: int = 300,
    n_init: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """K-Means clustering via Lloyd's algorithm.

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
        idx = rng.choice(n, size=k, replace=False)
        centers = X[idx].copy()

        for _ in range(max_iter):
            diffs = X[:, None, :] - centers[None, :, :]
            dists = (diffs**2).sum(axis=2)
            labels = np.argmin(dists, axis=1)

            new_centers = np.empty_like(centers)
            for j in range(k):
                members = X[labels == j]
                if len(members) == 0:
                    new_centers[j] = X[rng.integers(n)]
                else:
                    new_centers[j] = members.mean(axis=0)

            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        # Recompute distances against final centers for accurate inertia
        diffs = X[:, None, :] - centers[None, :, :]
        dists = (diffs**2).sum(axis=2)
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
    """Mean silhouette coefficient (Euclidean)."""
    n = len(X)
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)

    sil = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask_same = labels == labels[i]
        mask_same[i] = False
        n_same = mask_same.sum()

        if n_same == 0:
            sil[i] = 0.0
            continue

        a_i = dists[i, mask_same].sum() / n_same

        b_i = np.inf
        for lab in unique:
            if lab == labels[i]:
                continue
            mask_other = labels == lab
            b_i = min(b_i, dists[i, mask_other].mean())

        denom = max(a_i, b_i)
        sil[i] = (b_i - a_i) / denom if denom > 0 else 0.0

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
