# tests/test_clusters.py
"""Tests for ssdiff.clusters — KMeans clustering of beta neighbors."""

from ssdiff.clusters import cluster_top_neighbors


class TestClusterTopNeighbors:
    def test_returns_list_of_dicts(self, fitted_ssd_large):
        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, restrict_vocab=50
        )
        assert isinstance(clusters, list)
        assert all(isinstance(c, dict) for c in clusters)

    def test_cluster_dict_keys(self, fitted_ssd_large):
        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, restrict_vocab=50
        )
        for c in clusters:
            assert "id" in c
            assert "size" in c
            assert "centroid_cos_beta" in c
            assert "coherence" in c
            assert "words" in c

    def test_auto_k_selection(self, fitted_ssd_large):
        """k=None triggers silhouette-based auto selection."""
        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=None, k_min=2, k_max=4, restrict_vocab=50
        )
        assert len(clusters) >= 1

    def test_min_cluster_size_filter(self, fitted_ssd_large):
        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=3, min_cluster_size=2, restrict_vocab=50
        )
        for c in clusters:
            assert c["size"] >= 2

    def test_neg_side(self, fitted_ssd_large):
        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, side="neg", restrict_vocab=50
        )
        assert isinstance(clusters, list)

    def test_pos_sorted_descending(self, fitted_ssd_large):
        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, side="pos", restrict_vocab=50
        )
        if len(clusters) >= 2:
            cos_vals = [c["centroid_cos_beta"] for c in clusters]
            assert cos_vals == sorted(cos_vals, reverse=True)

    def test_neg_sorted_ascending(self, fitted_ssd_large):
        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, side="neg", restrict_vocab=50
        )
        if len(clusters) >= 2:
            cos_vals = [c["centroid_cos_beta"] for c in clusters]
            assert cos_vals == sorted(cos_vals)

    def test_words_are_tuples(self, fitted_ssd_large):
        clusters = cluster_top_neighbors(
            fitted_ssd_large, topn=20, k=2, restrict_vocab=50
        )
        for c in clusters:
            for word_entry in c["words"]:
                assert len(word_entry) == 3  # (word, cos_centroid, cos_beta)
