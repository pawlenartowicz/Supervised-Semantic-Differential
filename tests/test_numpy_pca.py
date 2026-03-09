import numpy as np
import pytest
from ssdiff._math import standardize, pca_fit_transform


# ── Edge cases for standardize ────────────────────────────────────────


def test_standardize_zero_variance_column():
    """Constant columns should get scale=1 (not divide by zero)."""
    X = np.array([[5, 1], [5, 2], [5, 3]], dtype=np.float64)
    Xs, mean, scale = standardize(X)
    # First column is constant → scale should be 1.0
    assert scale[0] == 1.0
    # Output should not contain NaN or inf
    assert not np.any(np.isnan(Xs))
    assert not np.any(np.isinf(Xs))


def test_standardize_single_sample():
    """Single sample: std=0 for all columns, scale should be 1.0."""
    X = np.array([[3, 7, 2]], dtype=np.float64)
    Xs, mean, scale = standardize(X)
    np.testing.assert_array_equal(scale, [1.0, 1.0, 1.0])


# ── Edge cases for PCA ────────────────────────────────────────────────


def test_pca_zero_variance():
    """All identical rows → zero total variance."""
    X = np.ones((10, 5), dtype=np.float64)
    z, components, evr = pca_fit_transform(X, n_components=2)
    assert z.shape == (10, 2)
    # All projected values should be zero (no variance)
    np.testing.assert_allclose(z, 0, atol=1e-10)
    # EVR should be zeros
    np.testing.assert_allclose(evr, 0, atol=1e-10)


def test_pca_n_components_equals_features():
    """n_components == n_features should work."""
    X = np.random.randn(20, 5)
    z, components, evr = pca_fit_transform(X, n_components=5)
    assert z.shape == (20, 5)
    assert components.shape == (5, 5)
    # All variance captured
    np.testing.assert_allclose(evr.sum(), 1.0, atol=1e-10)


# ── Original tests ────────────────────────────────────────────────────


def test_standardize_roundtrip():
    X = np.random.randn(50, 10) * 3 + 5
    Xs, mean, scale = standardize(X)
    np.testing.assert_allclose(Xs.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(Xs.std(axis=0), 1, atol=1e-10)


def test_standardize_1d():
    y = np.random.randn(50) * 2 + 10
    ys, mean, scale = standardize(y.reshape(-1, 1))
    assert ys.shape == (50, 1)
    np.testing.assert_allclose(ys.mean(), 0, atol=1e-10)


def test_pca_explained_variance_sums_to_1():
    X = np.random.randn(100, 20)
    z, components, evr = pca_fit_transform(X, n_components=5)
    assert z.shape == (100, 5)
    assert components.shape == (5, 20)
    assert len(evr) == 5
    assert all(0 <= r <= 1 for r in evr)


def test_pca_matches_sklearn():
    """Verify our PCA matches sklearn (up to sign flips)."""
    sklearn_pca = pytest.importorskip("sklearn.decomposition").PCA
    X = np.random.randn(80, 15)
    k = 5

    z_ours, comp_ours, evr_ours = pca_fit_transform(X, n_components=k)

    pca = sklearn_pca(n_components=k, svd_solver="full")
    pca.fit_transform(X)

    # Components may differ by sign — compare absolute values
    np.testing.assert_allclose(np.abs(comp_ours), np.abs(pca.components_), atol=1e-6)
    np.testing.assert_allclose(evr_ours, pca.explained_variance_ratio_, atol=1e-6)
