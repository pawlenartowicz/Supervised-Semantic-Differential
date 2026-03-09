import numpy as np
import pytest
from ssdiff._math import kmeans, silhouette_score, f_sf, _betainc


# ── KMeans edge cases ─────────────────────────────────────────────────


def test_kmeans_k_greater_than_n():
    """Requesting more clusters than samples should raise ValueError."""
    X = np.array([[0, 0], [1, 1]], dtype=np.float64)
    with pytest.raises(ValueError, match="Cannot request k=5"):
        kmeans(X, k=5, random_state=0)


def test_kmeans_single_init():
    """n_init=1 should still produce valid results."""
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    labels, centers, inertia = kmeans(X, k=2, random_state=42, n_init=1)
    assert len(np.unique(labels)) == 2


def test_kmeans_max_iter_1():
    """max_iter=1 should not crash; may not converge but should return."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (30, 2))
    labels, centers, inertia = kmeans(X, k=3, random_state=0, max_iter=1, n_init=1)
    assert labels.shape == (30,)
    assert centers.shape == (3, 2)


def test_kmeans_convergence():
    """Well-separated clusters should converge quickly."""
    rng = np.random.default_rng(7)
    X = np.vstack(
        [
            rng.normal(0, 0.01, (10, 2)),
            rng.normal(100, 0.01, (10, 2)),
        ]
    )
    labels, centers, inertia = kmeans(X, k=2, random_state=7, n_init=1)
    # Inertia should be very small
    assert inertia < 1.0


# ── Silhouette edge cases ─────────────────────────────────────────────


def test_silhouette_single_cluster():
    """All same label → silhouette should be 0.0."""
    X = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float64)
    labels = np.array([0, 0, 0])
    s = silhouette_score(X, labels)
    assert s == 0.0


def test_silhouette_singleton_cluster():
    """One cluster has only 1 member → that member's silhouette = 0."""
    X = np.array([[0, 0], [0.1, 0], [10, 10]], dtype=np.float64)
    labels = np.array([0, 0, 1])
    s = silhouette_score(X, labels)
    assert -1 <= s <= 1


# ── betainc edge cases ────────────────────────────────────────────────


def test_betainc_zero():
    assert _betainc(1.0, 1.0, 0.0) == 0.0


def test_betainc_one():
    assert _betainc(1.0, 1.0, 1.0) == 1.0


def test_betainc_symmetry():
    """I_x(a,b) = 1 - I_{1-x}(b,a)."""
    val = _betainc(2.0, 3.0, 0.4)
    complement = _betainc(3.0, 2.0, 0.6)
    np.testing.assert_allclose(val + complement, 1.0, atol=1e-10)


def test_betainc_uniform():
    """For a=1, b=1: I_x(1,1) = x."""
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        np.testing.assert_allclose(_betainc(1.0, 1.0, x), x, atol=1e-10)


# ── f_sf edge cases ───────────────────────────────────────────────────


def test_f_sf_zero_f():
    """F=0 → p=1."""
    assert f_sf(0.0, 5, 50) == 1.0


def test_f_sf_negative_f():
    """Negative F → p=1."""
    assert f_sf(-1.0, 5, 50) == 1.0


def test_f_sf_zero_dof():
    """Zero degrees of freedom → p=1."""
    assert f_sf(5.0, 0, 50) == 1.0
    assert f_sf(5.0, 5, 0) == 1.0


def test_f_sf_large_f():
    """Very large F → p near 0."""
    p = f_sf(1000.0, 5, 100)
    assert p < 1e-10


# ── Original tests ────────────────────────────────────────────────────


def test_kmeans_basic():
    """Two well-separated clusters."""
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(0, 0.1, (30, 2)), rng.normal(5, 0.1, (30, 2))])
    labels, centers, inertia = kmeans(X, k=2, random_state=42)
    assert len(np.unique(labels)) == 2
    assert centers.shape == (2, 2)


def test_kmeans_reproducible():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 3))
    l1, _, _ = kmeans(X, k=3, random_state=7)
    l2, _, _ = kmeans(X, k=3, random_state=7)
    np.testing.assert_array_equal(l1, l2)


def test_silhouette_perfect_clusters():
    """Well-separated clusters should have high silhouette."""
    X = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    s = silhouette_score(X, labels)
    assert s > 0.9


def test_silhouette_matches_sklearn():
    sklearn_sil = pytest.importorskip("sklearn.metrics").silhouette_score
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 3))
    labels = rng.integers(0, 3, 40)
    ours = silhouette_score(X, labels)
    theirs = sklearn_sil(X, labels)
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_f_sf_basic():
    """Known value: F(1, 100) with large F should give small p."""
    p = f_sf(10.0, 1, 100)
    assert 0 < p < 0.01


def test_f_sf_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    # Test several F values
    for f_val, d1, d2 in [(3.5, 5, 50), (1.2, 3, 100), (10.0, 2, 30), (0.5, 4, 80)]:
        ours = f_sf(f_val, d1, d2)
        theirs = 1 - scipy_stats.f.cdf(f_val, d1, d2)
        np.testing.assert_allclose(
            ours,
            theirs,
            atol=1e-10,
            err_msg=f"Mismatch for F={f_val}, d1={d1}, d2={d2}",
        )
