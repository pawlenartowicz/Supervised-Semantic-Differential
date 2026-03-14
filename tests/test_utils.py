# tests/test_utils.py
"""Tests for ssdiff.utils — SIF, normalize, neighbors, doc vectors, load dispatch."""

import numpy as np
import pytest

from ssdiff.embeddings import Embeddings, load_embeddings
from ssdiff.utils import (
    compute_global_sif,
    normalize_kv,
    filtered_neighbors,
    build_doc_vectors,
    build_doc_vectors_grouped,
    _iter_token_lists,
    _full_doc_vector,
)


# ── compute_global_sif ──────────────────────────────────────────────────


class TestComputeGlobalSIF:
    def test_basic_counts(self):
        sents = [["a", "b", "a"], ["b", "c"]]
        wc, tot = compute_global_sif(sents)
        assert wc == {"a": 2, "b": 2, "c": 1}
        assert tot == 5

    def test_empty(self):
        wc, tot = compute_global_sif([])
        assert wc == {}
        assert tot == 0

    def test_single_token(self):
        wc, tot = compute_global_sif([["x"]])
        assert wc == {"x": 1}
        assert tot == 1


# ── normalize_kv ────────────────────────────────────────────────────────


class TestNormalizeKV:
    def test_l2_normalization(self, tiny_kv):
        kv_n = normalize_kv(tiny_kv, l2=True, abtt_m=0)
        norms = np.linalg.norm(kv_n.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_returns_new_kv(self, tiny_kv):
        kv_n = normalize_kv(tiny_kv)
        assert kv_n is not tiny_kv
        assert set(kv_n.index_to_key) == set(tiny_kv.index_to_key)

    def test_abtt_removes_component(self, tiny_kv):
        kv_n = normalize_kv(tiny_kv, l2=True, abtt_m=1, re_normalize=True)
        norms = np.linalg.norm(kv_n.vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_no_l2(self, tiny_kv):
        kv_n = normalize_kv(tiny_kv, l2=False, abtt_m=0, re_normalize=False)
        # Should be same as original (float64 -> float32 rounding aside)
        np.testing.assert_allclose(kv_n.vectors, tiny_kv.vectors, atol=1e-5)


# ── filtered_neighbors ──────────────────────────────────────────────────


class TestFilteredNeighbors:
    def test_returns_list_of_tuples(self, tiny_kv):
        vec = tiny_kv["kraj"]
        result = filtered_neighbors(tiny_kv, vec, topn=5, restrict=20)
        assert isinstance(result, list)
        assert all(len(t) == 2 for t in result)

    def test_filters_bad_tokens(self, tiny_kv):
        vec = tiny_kv["kraj"]
        result = filtered_neighbors(tiny_kv, vec, topn=20, restrict=20)
        words = [w for w, _ in result]
        # "ABC123" contains digits — should be filtered
        assert "ABC123" not in words
        # "Warszawa" starts with uppercase — should NOT be filtered
        # (uppercase filter was removed to support non-Polish languages)

    def test_topn_respected(self, tiny_kv):
        vec = tiny_kv["piekny"]
        result = filtered_neighbors(tiny_kv, vec, topn=3, restrict=20)
        assert len(result) <= 3

    def test_similarity_descending(self, tiny_kv):
        vec = tiny_kv["dom"]
        result = filtered_neighbors(tiny_kv, vec, topn=10, restrict=20)
        sims = [s for _, s in result]
        assert sims == sorted(sims, reverse=True)


# ── build_doc_vectors ───────────────────────────────────────────────────


class TestBuildDocVectors:
    def test_shape_and_mask(self, tiny_kv, sample_docs, sample_lexicon):
        wc, tot = compute_global_sif(sample_docs)
        X, mask = build_doc_vectors(
            sample_docs, tiny_kv, sample_lexicon, wc, tot, window=3, sif_a=1e-3
        )
        assert X.ndim == 2
        assert X.shape[1] == tiny_kv.vector_size
        assert mask.sum() == X.shape[0]
        assert len(mask) == len(sample_docs)

    def test_no_seeds_returns_empty(
        self, tiny_kv, sample_docs_no_seeds, sample_lexicon
    ):
        wc, tot = compute_global_sif(sample_docs_no_seeds)
        X, mask = build_doc_vectors(
            sample_docs_no_seeds, tiny_kv, sample_lexicon, wc, tot, window=3, sif_a=1e-3
        )
        assert X.shape[0] == 0
        assert mask.sum() == 0

    def test_all_docs_kept_when_all_have_seeds(
        self, tiny_kv, sample_docs, sample_lexicon
    ):
        wc, tot = compute_global_sif(sample_docs)
        _, mask = build_doc_vectors(
            sample_docs, tiny_kv, sample_lexicon, wc, tot, window=3, sif_a=1e-3
        )
        assert mask.all()


# ── build_doc_vectors_grouped ───────────────────────────────────────────


class TestBuildDocVectorsGrouped:
    def test_flat_mode_seed(self, tiny_kv, sample_docs, sample_lexicon):
        wc, tot = compute_global_sif(sample_docs)
        X, mask = build_doc_vectors_grouped(
            sample_docs,
            tiny_kv,
            sample_lexicon,
            wc,
            tot,
            window=3,
            sif_a=1e-3,
            mode="seed",
        )
        assert X.shape[0] == mask.sum()
        assert X.shape[1] == tiny_kv.vector_size

    def test_flat_mode_full(self, tiny_kv, sample_docs, sample_lexicon):
        wc, tot = compute_global_sif(sample_docs)
        X, mask = build_doc_vectors_grouped(
            sample_docs,
            tiny_kv,
            sample_lexicon,
            wc,
            tot,
            window=3,
            sif_a=1e-3,
            mode="full",
        )
        # In full mode, all docs with any kv token are kept
        assert X.shape[0] == mask.sum()

    def test_grouped_mode_seed(self, tiny_kv, sample_lexicon):
        """Profile-style docs: List[List[List[str]]]."""
        grouped_docs = [
            [["kraj", "piekny", "dom"], ["narod", "silny"]],
            [["panstwo", "zly", "maly"]],
        ]
        wc, tot = compute_global_sif([t for profile in grouped_docs for t in profile])
        X, mask = build_doc_vectors_grouped(
            grouped_docs,
            tiny_kv,
            sample_lexicon,
            wc,
            tot,
            window=3,
            sif_a=1e-3,
            mode="seed",
        )
        assert X.shape[0] == mask.sum()
        assert len(mask) == 2  # one per profile

    def test_invalid_mode_raises(self, tiny_kv, sample_docs, sample_lexicon):
        wc, tot = compute_global_sif(sample_docs)
        with pytest.raises(ValueError, match="mode must be"):
            build_doc_vectors_grouped(
                sample_docs,
                tiny_kv,
                sample_lexicon,
                wc,
                tot,
                window=3,
                sif_a=1e-3,
                mode="invalid",
            )

    def test_empty_docs(self, tiny_kv, sample_lexicon):
        wc, tot = compute_global_sif([])
        X, mask = build_doc_vectors_grouped(
            [], tiny_kv, sample_lexicon, wc, tot, window=3, sif_a=1e-3
        )
        assert X.shape[0] == 0


# ── load_embeddings dispatch ────────────────────────────────────────────


class TestLoadEmbeddings:
    def test_txt_extension_with_header(self, tmp_path, tiny_kv):
        """Write a word2vec-style .txt (with header line)."""
        p = str(tmp_path / "test.txt")
        words = tiny_kv.index_to_key
        with open(p, "w") as f:
            f.write(f"{len(words)} {tiny_kv.vector_size}\n")
            for w in words:
                vec = tiny_kv[w]
                f.write(w + " " + " ".join(f"{v:.6f}" for v in vec) + "\n")
        loaded = load_embeddings(p)
        assert set(loaded.index_to_key) == set(words)

    def test_txt_extension_no_header(self, tmp_path, tiny_kv):
        """Write a GloVe-style .txt (no header line)."""
        p = str(tmp_path / "test.txt")
        for w in tiny_kv.index_to_key:
            pass  # get last word
        with open(p, "w") as f:
            for w in tiny_kv.index_to_key:
                vec = tiny_kv[w]
                f.write(w + " " + " ".join(f"{v:.6f}" for v in vec) + "\n")
        loaded = load_embeddings(p)
        assert w in loaded

    def test_bin_extension(self, tmp_path, tiny_kv):
        """Write a word2vec binary .bin file."""
        p = str(tmp_path / "test.bin")
        words = tiny_kv.index_to_key
        with open(p, "wb") as f:
            f.write(f"{len(words)} {tiny_kv.vector_size}\n".encode("utf-8"))
            for w in words:
                vec = tiny_kv[w]
                f.write(w.encode("utf-8") + b" ")
                f.write(vec.astype(np.float32).tobytes())
        loaded = load_embeddings(p)
        assert set(loaded.index_to_key) == set(words)
        np.testing.assert_allclose(loaded[words[0]], tiny_kv[words[0]], atol=1e-5)


# ── _iter_token_lists ─────────────────────────────────────────────────


class TestIterTokenLists:
    def test_flat_docs(self):
        docs = [["a", "b"], ["c", "d"]]
        result = list(_iter_token_lists(docs))
        assert result == [["a", "b"], ["c", "d"]]

    def test_grouped_docs(self):
        docs = [[["a", "b"], ["c"]], [["d"]]]
        result = list(_iter_token_lists(docs))
        assert result == [["a", "b"], ["c"], ["d"]]

    def test_empty_items_skipped(self):
        docs = [[], ["a", "b"], []]
        result = list(_iter_token_lists(docs))
        assert result == [["a", "b"]]

    def test_empty_sublists_skipped(self):
        docs = [[[], ["a"]], [["b"]]]
        result = list(_iter_token_lists(docs))
        assert result == [["a"], ["b"]]

    def test_empty_input(self):
        assert list(_iter_token_lists([])) == []


# ── _full_doc_vector ──────────────────────────────────────────────────


class TestFullDocVector:
    def test_all_oov_returns_none(self, tiny_kv):
        """Tokens not in kv → None."""
        tokens = ["zzz_oov1", "zzz_oov2"]
        wc = {"zzz_oov1": 1, "zzz_oov2": 1}
        result = _full_doc_vector(tokens, tiny_kv, wc, 2, 1e-3)
        assert result is None

    def test_valid_tokens(self, tiny_kv):
        tokens = ["kraj", "dom", "piekny"]
        wc = {"kraj": 1, "dom": 1, "piekny": 1}
        result = _full_doc_vector(tokens, tiny_kv, wc, 3, 1e-3)
        assert result is not None
        assert result.shape == (tiny_kv.vector_size,)

    def test_mixed_oov_and_valid(self, tiny_kv):
        tokens = ["kraj", "zzz_never", "dom"]
        wc = {"kraj": 1, "zzz_never": 1, "dom": 1}
        result = _full_doc_vector(tokens, tiny_kv, wc, 3, 1e-3)
        assert result is not None


# ── grouped docs with full mode ──────────────────────────────────────


class TestBuildDocVectorsGroupedFull:
    def test_grouped_full_mode(self, tiny_kv, sample_lexicon):
        grouped_docs = [
            [["kraj", "piekny", "dom"], ["narod", "silny"]],
            [["panstwo", "zly", "maly"]],
        ]
        wc, tot = compute_global_sif([t for p in grouped_docs for t in p])
        X, mask = build_doc_vectors_grouped(
            grouped_docs,
            tiny_kv,
            sample_lexicon,
            wc,
            tot,
            window=3,
            sif_a=1e-3,
            mode="full",
        )
        assert X.shape[0] == mask.sum()
        assert len(mask) == 2

    def test_flat_all_oov_full_mode(self):
        """Full mode with all OOV tokens → all docs dropped."""
        kv = Embeddings(["zzz"], np.array([[1, 0]], dtype=np.float32))
        docs = [["aaa", "bbb"], ["ccc"]]
        wc, tot = compute_global_sif(docs)
        X, mask = build_doc_vectors_grouped(
            docs, kv, set(), wc, tot, window=3, sif_a=1e-3, mode="full"
        )
        assert X.shape[0] == 0
        assert not mask.any()

    def test_flat_empty_doc_in_list(self, tiny_kv, sample_lexicon):
        """Empty docs in list should be dropped."""
        docs = [["kraj", "dom"], [], ["narod", "piekny"]]
        wc, tot = compute_global_sif([d for d in docs if d])
        X, mask = build_doc_vectors_grouped(
            docs, tiny_kv, sample_lexicon, wc, tot, window=3, sif_a=1e-3, mode="seed"
        )
        assert not mask[1]  # empty doc dropped
