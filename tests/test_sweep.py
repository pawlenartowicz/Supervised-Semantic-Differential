# tests/test_sweep.py
"""Tests for ssdiff.sweep — pca_sweep function."""

import numpy as np
import pandas as pd
import pytest

from ssdiff.sweep import pca_sweep, PCAKSelectionResult


class TestPCASweep:
    def test_returns_result_object(self, tiny_kv_large, sample_lexicon):
        rng = np.random.default_rng(7)
        docs = []
        vocab = list(tiny_kv_large.index_to_key)
        seeds = list(sample_lexicon & set(vocab))
        for _ in range(20):
            doc = list(rng.choice(seeds, size=1)) + list(rng.choice(vocab, size=6))
            docs.append(doc)
        y = rng.normal(1.0, 0.3, size=20)

        result = pca_sweep(
            kv=tiny_kv_large,
            docs=docs,
            y=y,
            lexicon=sample_lexicon,
            pca_k_values=[3, 4, 5],
            cluster_topn=20,
            k_min=2,
            k_max=3,
            verbose=False,
        )

        assert isinstance(result, PCAKSelectionResult)
        assert isinstance(result.best_k, int)
        assert result.best_k in [3, 4, 5]
        assert isinstance(result.df_joined, pd.DataFrame)
        assert "PCA_K" in result.df_joined.columns
        assert "joint_score" in result.df_joined.columns
        assert len(result.df_joined) == 3

    def test_save_raises_without_outdir(self, tiny_kv_large, sample_lexicon):
        rng = np.random.default_rng(7)
        docs = [
            list(rng.choice(list(sample_lexicon), size=1))
            + list(rng.choice(list(tiny_kv_large.index_to_key), size=4))
            for _ in range(10)
        ]
        y = rng.normal(size=10)

        with pytest.raises(ValueError, match="out_dir"):
            pca_sweep(
                kv=tiny_kv_large,
                docs=docs,
                y=y,
                lexicon=sample_lexicon,
                pca_k_values=[3],
                save_tables=True,
                verbose=False,
            )
