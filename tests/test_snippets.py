# tests/test_snippets.py
"""Tests for ssdiff.snippets — snippets_along_beta, cluster_snippets_by_centroids."""

import pandas as pd

from ssdiff.snippets import snippets_along_beta, cluster_snippets_by_centroids


# ── snippets_along_beta ─────────────────────────────────────────────────


class TestSnippetsAlongBeta:
    def test_returns_dict_with_both_sides(self, fitted_ssd, sample_preprocessed_docs):
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd,
            token_window=3,
            seeds=fitted_ssd.lexicon,
            sif_a=1e-3,
            top_per_side=10,
            n_jobs=1,
            progress=False,
        )
        assert isinstance(result, dict)
        assert "beta_pos" in result
        assert "beta_neg" in result

    def test_returns_dataframes(self, fitted_ssd, sample_preprocessed_docs):
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd,
            token_window=3,
            seeds=fitted_ssd.lexicon,
            n_jobs=1,
            progress=False,
        )
        assert isinstance(result["beta_pos"], pd.DataFrame)
        assert isinstance(result["beta_neg"], pd.DataFrame)

    def test_columns_present(self, fitted_ssd, sample_preprocessed_docs):
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd,
            token_window=3,
            seeds=fitted_ssd.lexicon,
            n_jobs=1,
            progress=False,
        )
        expected_cols = {"cosine", "seed", "snippet_anchor", "profile_id", "post_id"}
        for key in ("beta_pos", "beta_neg"):
            if not result[key].empty:
                assert expected_cols <= set(result[key].columns)

    def test_top_per_side_limit(self, fitted_ssd, sample_preprocessed_docs):
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd,
            token_window=3,
            seeds=fitted_ssd.lexicon,
            top_per_side=2,
            n_jobs=1,
            progress=False,
        )
        assert len(result["beta_pos"]) <= 2
        assert len(result["beta_neg"]) <= 2

    def test_no_seeds_fallback(self, fitted_ssd, sample_preprocessed_docs):
        """When seeds is empty, uses sentence-based fallback."""
        result = snippets_along_beta(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd,
            token_window=3,
            seeds=set(),
            n_jobs=1,
            progress=False,
        )
        assert isinstance(result, dict)

    def test_empty_pre_docs(self, fitted_ssd):
        result = snippets_along_beta(
            pre_docs=[],
            ssd=fitted_ssd,
            token_window=3,
            seeds=fitted_ssd.lexicon,
            n_jobs=1,
            progress=False,
        )
        assert result["beta_pos"].empty
        assert result["beta_neg"].empty


# ── cluster_snippets_by_centroids ───────────────────────────────────────


class TestClusterSnippetsByCentroids:
    def test_empty_clusters_returns_empty(self, fitted_ssd, sample_preprocessed_docs):
        result = cluster_snippets_by_centroids(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd,
            pos_clusters=[],
            neg_clusters=[],
            n_jobs=1,
            progress=False,
        )
        assert result["pos"].empty
        assert result["neg"].empty

    def test_with_clusters(self, fitted_ssd_large, sample_preprocessed_docs):
        """Build some clusters, then extract snippets."""
        from ssdiff.clusters import cluster_top_neighbors

        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, restrict_vocab=50
        )
        result = cluster_snippets_by_centroids(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd_large,
            pos_clusters=clusters,
            neg_clusters=[],
            n_jobs=1,
            progress=False,
        )
        assert isinstance(result, dict)
        assert isinstance(result["pos"], pd.DataFrame)
        assert isinstance(result["neg"], pd.DataFrame)

    def test_returns_correct_keys(self, fitted_ssd_large, sample_preprocessed_docs):
        from ssdiff.clusters import cluster_top_neighbors

        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, restrict_vocab=50
        )
        result = cluster_snippets_by_centroids(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd_large,
            pos_clusters=clusters,
            neg_clusters=[],
            n_jobs=1,
            progress=False,
        )
        assert set(result.keys()) == {"pos", "neg"}

    def test_no_seeds_fallback(self, fitted_ssd_large, sample_preprocessed_docs):
        from ssdiff.clusters import cluster_top_neighbors

        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, restrict_vocab=50
        )
        result = cluster_snippets_by_centroids(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd_large,
            pos_clusters=clusters,
            neg_clusters=[],
            seeds=set(),
            n_jobs=1,
            progress=False,
        )
        assert isinstance(result, dict)

    def test_top_per_cluster_limit(self, fitted_ssd_large, sample_preprocessed_docs):
        from ssdiff.clusters import cluster_top_neighbors

        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, restrict_vocab=50
        )
        result = cluster_snippets_by_centroids(
            pre_docs=sample_preprocessed_docs,
            ssd=fitted_ssd_large,
            pos_clusters=clusters,
            neg_clusters=[],
            top_per_cluster=1,
            n_jobs=1,
            progress=False,
        )
        if not result["pos"].empty:
            counts = result["pos"].groupby("centroid_label").size()
            assert counts.max() <= 1
