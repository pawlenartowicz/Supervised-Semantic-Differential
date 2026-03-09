# tests/test_crossgroup.py
"""Tests for ssdiff.crossgroup — SSDGroup, SSDContrast, permutation tests."""

import numpy as np
import pandas as pd
import pytest

from ssdiff import SSDGroup, SSDContrast


# ── SSDGroup construction ──────────────────────────────────────────────


class TestSSDGroupBasic:
    def test_has_centroids(self, fitted_ssd_group):
        assert isinstance(fitted_ssd_group.centroids, dict)
        assert len(fitted_ssd_group.centroids) == 2  # A and B

    def test_centroid_unit_length(self, fitted_ssd_group):
        for g, c in fitted_ssd_group.centroids.items():
            norm = np.linalg.norm(c)
            np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_omnibus_T_positive(self, fitted_ssd_group):
        assert fitted_ssd_group.omnibus_T >= 0.0

    def test_omnibus_p_in_range(self, fitted_ssd_group):
        assert 0.0 <= fitted_ssd_group.omnibus_p <= 1.0

    def test_pairwise_keys(self, fitted_ssd_group):
        assert len(fitted_ssd_group.pairwise) == 1  # 2 groups → 1 pair
        key = list(fitted_ssd_group.pairwise.keys())[0]
        assert len(key) == 2

    def test_pairwise_contains_required_fields(self, fitted_ssd_group):
        for key, val in fitted_ssd_group.pairwise.items():
            assert "T" in val
            assert "p_raw" in val
            assert "p_corrected" in val
            assert "contrast_unit" in val
            assert "cohens_d" in val

    def test_group_labels_sorted(self, fitted_ssd_group):
        assert fitted_ssd_group.group_labels == sorted(fitted_ssd_group.group_labels)

    def test_G_count(self, fitted_ssd_group):
        assert fitted_ssd_group.G == 2


# ── SSDGroup with 3 groups ─────────────────────────────────────────────


class TestSSDGroup3Groups:
    def test_3_groups_construction(
        self, tiny_kv, sample_docs, sample_groups_3, sample_lexicon
    ):
        grp = SSDGroup(
            kv=tiny_kv,
            docs=sample_docs,
            groups=sample_groups_3,
            lexicon=sample_lexicon,
            n_perm=50,
        )
        assert grp.G == 3
        assert len(grp.pairwise) == 3  # C(3,2) = 3

    def test_3_groups_omnibus_different(
        self, tiny_kv, sample_docs, sample_groups_3, sample_lexicon
    ):
        grp = SSDGroup(
            kv=tiny_kv,
            docs=sample_docs,
            groups=sample_groups_3,
            lexicon=sample_lexicon,
            n_perm=50,
        )
        assert np.isfinite(grp.omnibus_T)
        assert 0.0 <= grp.omnibus_p <= 1.0


# ── SSDGroup validation ────────────────────────────────────────────────


class TestSSDGroupValidation:
    def test_mismatched_lengths(self, tiny_kv, sample_docs, sample_lexicon):
        bad_groups = np.array(["A", "B"])  # too short
        with pytest.raises(ValueError, match="len"):
            SSDGroup(
                kv=tiny_kv,
                docs=sample_docs,
                groups=bad_groups,
                lexicon=sample_lexicon,
                n_perm=10,
            )

    def test_single_group_after_filter(self, tiny_kv, sample_docs, sample_lexicon):
        all_same = np.array(["X"] * len(sample_docs), dtype=object)
        with pytest.raises(ValueError, match="at least 2"):
            SSDGroup(
                kv=tiny_kv,
                docs=sample_docs,
                groups=all_same,
                lexicon=sample_lexicon,
                n_perm=10,
            )


# ── SSDGroup public API ────────────────────────────────────────────────


class TestSSDGroupAPI:
    def test_results_table(self, fitted_ssd_group):
        df = fitted_ssd_group.results_table()
        assert isinstance(df, pd.DataFrame)
        assert "group_A" in df.columns
        assert len(df) == 1

    def test_print_results(self, fitted_ssd_group, capsys):
        fitted_ssd_group.print_results()
        out = capsys.readouterr().out
        assert "SSDGroup" in out

    def test_contrast_scores(self, fitted_ssd_group):
        df = fitted_ssd_group.contrast_scores("A", "B")
        assert isinstance(df, pd.DataFrame)
        assert "cos_to_contrast" in df.columns
        assert len(df) == fitted_ssd_group.n_kept


# ── SSDContrast ─────────────────────────────────────────────────────────


class TestSSDContrast:
    def test_get_contrast_returns_contrast(self, fitted_ssd_group):
        c = fitted_ssd_group.get_contrast("A", "B")
        assert isinstance(c, SSDContrast)

    def test_contrast_flipped(self, fitted_ssd_group):
        c_ab = fitted_ssd_group.get_contrast("A", "B")
        c_ba = fitted_ssd_group.get_contrast("B", "A")
        # beta_unit should be flipped
        np.testing.assert_allclose(c_ab.beta_unit, -c_ba.beta_unit, atol=1e-10)

    def test_contrast_nbrs(self, fitted_ssd_group):
        c = fitted_ssd_group.get_contrast("A", "B")
        nbrs = c.nbrs(sign=+1, n=3, restrict_vocab=20)
        assert isinstance(nbrs, list)

    def test_contrast_top_words(self, fitted_ssd_group):
        c = fitted_ssd_group.get_contrast("A", "B")
        df = c.top_words(n=3)
        assert isinstance(df, pd.DataFrame)
        assert "side" in df.columns

    def test_contrast_invalid_pair(self, fitted_ssd_group):
        with pytest.raises(KeyError, match="not found"):
            fitted_ssd_group.get_contrast("X", "Y")


# ── SSDContrast clustering & snippets ─────────────────────────────────


class TestSSDContrastClustering:
    def test_cluster_neighbors_sign(self, fitted_ssd_group):
        c = fitted_ssd_group.get_contrast("A", "B")
        df_c, df_m = c.cluster_neighbors_sign(
            side="pos", topn=10, k=2, restrict_vocab=20
        )
        assert isinstance(df_c, pd.DataFrame)
        assert isinstance(df_m, pd.DataFrame)
        assert c.pos_clusters_raw is not None

    def test_cluster_neighbors_both(self, fitted_ssd_group):
        c = fitted_ssd_group.get_contrast("A", "B")
        df_c, df_m = c.cluster_neighbors(topn=10, k=2, restrict_vocab=20)
        assert isinstance(df_c, pd.DataFrame)
        assert c.pos_clusters_raw is not None
        assert c.neg_clusters_raw is not None

    def test_cluster_neighbors_verbose(self, fitted_ssd_group, capsys):
        c = fitted_ssd_group.get_contrast("A", "B")
        c.cluster_neighbors_sign(
            side="pos", topn=10, k=2, restrict_vocab=20, verbose=True
        )
        out = capsys.readouterr().out
        assert "Cluster" in out


class TestSSDContrastSnippets:
    def test_beta_snippets(self, fitted_ssd_group, sample_preprocessed_docs):
        c = fitted_ssd_group.get_contrast("A", "B")
        result = c.beta_snippets(pre_docs=sample_preprocessed_docs, top_per_side=10)
        assert "beta_pos" in result
        assert "beta_neg" in result

    def test_cluster_snippets_without_clustering_raises(
        self, fitted_ssd_group, sample_preprocessed_docs
    ):
        c = fitted_ssd_group.get_contrast("A", "B")
        with pytest.raises(RuntimeError, match="first"):
            c.cluster_snippets(pre_docs=sample_preprocessed_docs, side="pos")

    def test_cluster_snippets_after_clustering(
        self, fitted_ssd_group, sample_preprocessed_docs
    ):
        c = fitted_ssd_group.get_contrast("A", "B")
        c.cluster_neighbors_sign(side="pos", topn=10, k=2, restrict_vocab=20)
        result = c.cluster_snippets(
            pre_docs=sample_preprocessed_docs, side="pos", top_per_cluster=10
        )
        assert "pos" in result


class TestSSDContrastTopWords:
    def test_top_words_verbose(self, fitted_ssd_group, capsys):
        c = fitted_ssd_group.get_contrast("A", "B")
        df = c.top_words(n=3, verbose=True)
        out = capsys.readouterr().out
        assert isinstance(df, pd.DataFrame)
        assert "group" in df.columns
        # Verbose should mention groups
        assert "A" in out or "B" in out


# ── contrast_scores edge case ─────────────────────────────────────────


class TestContrastScoresDict:
    def test_contrast_scores_dict(self, fitted_ssd_group):
        result = fitted_ssd_group.contrast_scores("A", "B", return_df=False)
        assert isinstance(result, dict)
        assert "group" in result
        assert "cos_to_contrast" in result
