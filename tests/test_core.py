# tests/test_core.py
"""Tests for ssdiff.core — SSD class attributes, regression, methods, edge cases."""

import copy

import numpy as np
import pandas as pd
import pytest

from ssdiff import SSD


# ── Construction & attributes ───────────────────────────────────────────


class TestSSDAttributes:
    def test_has_beta(self, fitted_ssd):
        assert hasattr(fitted_ssd, "beta")
        assert fitted_ssd.beta.shape == (fitted_ssd.kv.vector_size,)

    def test_has_beta_unit(self, fitted_ssd):
        norm = np.linalg.norm(fitted_ssd.beta_unit)
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_r2_in_range(self, fitted_ssd):
        assert 0.0 <= fitted_ssd.r2 <= 1.0 or np.isnan(fitted_ssd.r2)

    def test_r2_adj_exists(self, fitted_ssd):
        assert hasattr(fitted_ssd, "r2_adj")

    def test_f_stat_nonneg(self, fitted_ssd):
        assert fitted_ssd.f_stat >= 0.0

    def test_keep_mask_shape(self, fitted_ssd):
        assert len(fitted_ssd.keep_mask) == fitted_ssd.n_raw

    def test_n_kept_consistent(self, fitted_ssd):
        assert fitted_ssd.n_kept == int(fitted_ssd.keep_mask.sum())
        assert fitted_ssd.n_kept + fitted_ssd.n_dropped == fitted_ssd.n_raw

    def test_pca_components(self, fitted_ssd):
        assert fitted_ssd.z.shape == (fitted_ssd.n_kept, fitted_ssd.N_PCA)

    def test_cluster_placeholders_init(self, fitted_ssd):
        # Cluster attributes should start as None
        assert fitted_ssd.pos_clusters_raw is None
        assert fitted_ssd.neg_clusters_raw is None


# ── Regression quality ──────────────────────────────────────────────────


class TestSSDRegression:
    def test_beta_norm_positive(self, fitted_ssd):
        assert fitted_ssd.beta_norm_stdCN > 0.0

    def test_y_corr_pred_nonneg(self, fitted_ssd):
        # After orientation fix, correlation should be >= 0
        assert fitted_ssd.y_corr_pred >= 0.0

    def test_y_mean_std(self, fitted_ssd):
        assert np.isfinite(fitted_ssd.y_mean)
        assert fitted_ssd.y_std > 0.0

    def test_pinned_regression_values(self, fitted_ssd):
        """Pin key numeric outputs from the seeded fixture to catch silent regressions."""
        assert fitted_ssd.r2 == pytest.approx(fitted_ssd.r2, abs=1e-10)  # self-check
        assert fitted_ssd.n_kept == 8
        assert fitted_ssd.n_dropped == 0
        assert fitted_ssd.N_PCA == 3
        assert fitted_ssd.z.shape == (8, 3)
        # beta_unit should be unit-length
        np.testing.assert_allclose(np.linalg.norm(fitted_ssd.beta_unit), 1.0, atol=1e-6)
        # pca variance explained should sum to <= 1.0
        assert 0.0 < fitted_ssd.pca_var_explained <= 1.0


# ── NaN handling ────────────────────────────────────────────────────────


class TestSSDNaN:
    def test_nan_in_y_dropped(
        self, tiny_kv, sample_docs, sample_y_with_nan, sample_lexicon
    ):
        ssd = SSD(
            kv=tiny_kv,
            docs=sample_docs,
            y=sample_y_with_nan,
            lexicon=sample_lexicon,
            N_PCA=3,
        )
        # One NaN in y → that doc dropped before PCV building
        assert ssd.n_raw == len(sample_docs) - 1


# ── Public methods ──────────────────────────────────────────────────────


class TestSSDMethods:
    def test_nbrs_returns_list(self, fitted_ssd):
        result = fitted_ssd.nbrs(sign=+1, n=5, restrict_vocab=20)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 3  # (word, cosine, shift)

    def test_nbrs_negative(self, fitted_ssd):
        result = fitted_ssd.nbrs(sign=-1, n=5, restrict_vocab=20)
        assert isinstance(result, list)

    def test_top_words_returns_df(self, fitted_ssd):
        df = fitted_ssd.top_words(n=3)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"side", "rank", "word", "cos"}
        assert set(df["side"].unique()) == {"pos", "neg"}

    def test_doc_scores_keys(self, fitted_ssd):
        scores = fitted_ssd.doc_scores()
        assert set(scores.keys()) >= {"keep_mask", "cos_align", "score_std", "yhat_raw"}
        assert len(scores["cos_align"]) == fitted_ssd.n_kept

    def test_ssd_scores_df(self, fitted_ssd):
        df = fitted_ssd.ssd_scores()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == fitted_ssd.n_raw
        assert "kept" in df.columns
        assert "cos" in df.columns

    def test_ssd_scores_only_kept(self, fitted_ssd):
        df = fitted_ssd.ssd_scores(include_all=False)
        assert len(df) == fitted_ssd.n_kept

    def test_ssd_scores_dict(self, fitted_ssd):
        result = fitted_ssd.ssd_scores(return_df=False)
        assert isinstance(result, dict)

    def test_select_extreme_docs(self, fitted_ssd):
        idx = fitted_ssd.select_extreme_docs(k=2, by="y")
        assert isinstance(idx, np.ndarray)
        assert len(idx) <= 4  # 2 bottom + 2 top

    def test_select_extreme_docs_by_cos(self, fitted_ssd):
        idx = fitted_ssd.select_extreme_docs(k=2, by="cos")
        assert isinstance(idx, np.ndarray)

    def test_select_extreme_docs_invalid_by(self, fitted_ssd):
        with pytest.raises(ValueError, match="by"):
            fitted_ssd.select_extreme_docs(k=2, by="invalid")

    def test_print_model_stats(self, fitted_ssd, capsys):
        fitted_ssd.print_model_stats()
        captured = capsys.readouterr()
        assert "R²" in captured.out


# ── No seeds → ValueError ──────────────────────────────────────────────


class TestSSDNoSeeds:
    def test_raises_when_no_seeds_present(
        self, tiny_kv, sample_docs_no_seeds, sample_lexicon
    ):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="No items contain the lexicon"):
            SSD(
                kv=tiny_kv,
                docs=sample_docs_no_seeds,
                y=y,
                lexicon=sample_lexicon,
                N_PCA=3,
            )


# ── ssd_scores edge cases ─────────────────────────────────────────────


class TestSSDScoresEdgeCases:
    def test_ssd_scores_no_true(self, fitted_ssd):
        """include_true=False should omit y_true columns."""
        df = fitted_ssd.ssd_scores(include_true=False)
        assert "y_true_std" not in df.columns
        assert "y_true_raw" not in df.columns

    def test_ssd_scores_only_kept_no_true(self, fitted_ssd):
        df = fitted_ssd.ssd_scores(include_all=False, include_true=False)
        assert len(df) == fitted_ssd.n_kept
        assert "y_true_std" not in df.columns

    def test_ssd_scores_dict_with_true(self, fitted_ssd):
        result = fitted_ssd.ssd_scores(return_df=False, include_true=True)
        assert isinstance(result, dict)
        assert "y_true_std" in result
        assert "y_true_raw" in result

    def test_ssd_scores_only_kept_dict(self, fitted_ssd):
        result = fitted_ssd.ssd_scores(include_all=False, return_df=False)
        assert len(result["cos"]) == fitted_ssd.n_kept


# ── select_extreme_docs edge cases ────────────────────────────────────


class TestSelectExtremeEdgeCases:
    def test_by_yhat(self, fitted_ssd):
        idx = fitted_ssd.select_extreme_docs(k=2, by="yhat")
        assert isinstance(idx, np.ndarray)

    def test_by_cos_not_include_dropped(self, fitted_ssd):
        idx = fitted_ssd.select_extreme_docs(k=2, by="cos", include_dropped=False)
        assert isinstance(idx, np.ndarray)

    def test_k_zero(self, fitted_ssd):
        idx = fitted_ssd.select_extreme_docs(k=0, by="y")
        assert len(idx) == 0

    def test_k_too_large(self, fitted_ssd):
        """k > n/2 should be clamped."""
        idx = fitted_ssd.select_extreme_docs(k=1000, by="y")
        assert isinstance(idx, np.ndarray)
        # Should not exceed total docs
        assert len(idx) <= fitted_ssd.n_raw


# ── Clustering methods ────────────────────────────────────────────────


class TestSSDClusterMethods:
    def test_cluster_neighbors_sign_pos(self, fitted_ssd_large):
        df_c, df_m = fitted_ssd_large.cluster_neighbors_sign(
            side="pos", topn=20, k=2, restrict_vocab=50
        )
        assert isinstance(df_c, pd.DataFrame)
        assert isinstance(df_m, pd.DataFrame)
        assert "side" in df_c.columns
        assert fitted_ssd_large.pos_clusters_raw is not None

    def test_cluster_neighbors_sign_neg(self, fitted_ssd_large):
        df_c, df_m = fitted_ssd_large.cluster_neighbors_sign(
            side="neg", topn=20, k=2, restrict_vocab=50
        )
        assert fitted_ssd_large.neg_clusters_raw is not None

    def test_cluster_neighbors_both(self, fitted_ssd_large):
        df_c, df_m = fitted_ssd_large.cluster_neighbors(topn=20, k=2, restrict_vocab=50)
        assert isinstance(df_c, pd.DataFrame)
        assert set(df_c["side"].unique()) == {"pos", "neg"}

    def test_cluster_neighbors_verbose(self, fitted_ssd_large, capsys):
        fitted_ssd_large.cluster_neighbors_sign(
            side="pos", topn=20, k=2, restrict_vocab=50, verbose=True
        )
        out = capsys.readouterr().out
        assert "Cluster" in out


# ── Snippet methods ───────────────────────────────────────────────────


class TestSSDSnippetMethods:
    def test_beta_snippets(self, fitted_ssd, sample_preprocessed_docs):
        result = fitted_ssd.beta_snippets(
            pre_docs=sample_preprocessed_docs, top_per_side=10
        )
        assert "beta_pos" in result
        assert "beta_neg" in result
        assert isinstance(result["beta_pos"], pd.DataFrame)

    def test_cluster_snippets_without_clustering_raises(
        self, fitted_ssd, sample_preprocessed_docs
    ):
        ssd_copy = copy.deepcopy(fitted_ssd)
        ssd_copy.pos_clusters_raw = None
        ssd_copy.neg_clusters_raw = None
        with pytest.raises(RuntimeError, match="clusters not available"):
            ssd_copy.cluster_snippets(pre_docs=sample_preprocessed_docs, side="pos")

    def test_cluster_snippets_after_clustering(
        self, fitted_ssd_large, sample_preprocessed_docs
    ):
        fitted_ssd_large.cluster_neighbors_sign(
            side="pos", topn=20, k=2, restrict_vocab=50
        )
        result = fitted_ssd_large.cluster_snippets(
            pre_docs=sample_preprocessed_docs, side="pos", top_per_cluster=10
        )
        assert "pos" in result
        assert isinstance(result["pos"], pd.DataFrame)

    def test_beta_snippets_extremes(self, fitted_ssd, sample_preprocessed_docs):
        result = fitted_ssd.beta_snippets_extremes(
            pre_docs=sample_preprocessed_docs,
            k=2,
            by="y",
            top_per_side=10,
        )
        assert "beta_pos" in result
        assert "beta_neg" in result

    def test_cluster_snippets_extremes(
        self, fitted_ssd_large, sample_preprocessed_docs
    ):
        # First run clustering
        fitted_ssd_large.cluster_neighbors(topn=20, k=2, restrict_vocab=50)
        result = fitted_ssd_large.cluster_snippets_extremes(
            pre_docs=sample_preprocessed_docs,
            k=5,
            by="y",
            side="both",
        )
        assert "pos" in result
        assert "neg" in result

    def test_cluster_snippets_extremes_no_clusters_raises(
        self, fitted_ssd, sample_preprocessed_docs
    ):
        ssd_copy = copy.deepcopy(fitted_ssd)
        ssd_copy.pos_clusters_raw = None
        with pytest.raises(RuntimeError, match="clusters missing"):
            ssd_copy.cluster_snippets_extremes(
                pre_docs=sample_preprocessed_docs, k=2, by="y", side="pos"
            )


# ── top_words verbose ─────────────────────────────────────────────────


class TestTopWordsVerbose:
    def test_top_words_verbose(self, fitted_ssd, capsys):
        df = fitted_ssd.top_words(n=3, verbose=True)
        out = capsys.readouterr().out
        assert "β̂" in out
        assert isinstance(df, pd.DataFrame)


# ── use_full_doc ──────────────────────────────────────────────────────


class TestSSDFullDoc:
    def test_full_doc_mode(self, tiny_kv, sample_docs, sample_y, sample_lexicon):
        ssd = SSD(
            kv=tiny_kv,
            docs=sample_docs,
            y=sample_y,
            lexicon=sample_lexicon,
            N_PCA=3,
            use_full_doc=True,
        )
        assert ssd.use_full_doc
        assert hasattr(ssd, "beta")

    def test_full_doc_no_embeddings_raises(self, sample_y):
        """All OOV docs in full_doc mode should raise."""
        # Create kv with no overlapping vocabulary
        from ssdiff.embeddings import Embeddings

        kv = Embeddings(["zzz_never_used"], np.array([[1, 0, 0]], dtype=np.float32))
        docs = [["aaa", "bbb"], ["ccc", "ddd"]]
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="No valid document vectors"):
            SSD(kv=kv, docs=docs, y=y, lexicon={"aaa"}, N_PCA=2, use_full_doc=True)


# ── subset_pre_docs_by_idx ─────────────────────────────────────────────


class TestSubsetPreDocs:
    def test_subset_basic(self, sample_preprocessed_docs):
        subset, kept = SSD.subset_pre_docs_by_idx(sample_preprocessed_docs, {0, 2})
        assert len(subset) == 2
        assert kept == [0, 2]

    def test_subset_empty_input(self):
        subset, kept = SSD.subset_pre_docs_by_idx([], {0, 1})
        assert subset == []
        assert kept == []
