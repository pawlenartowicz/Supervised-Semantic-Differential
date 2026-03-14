import gzip
import os
import pickle
import numpy as np
import pytest
from ssdiff.embeddings import (
    Embeddings,
    load_embeddings,
    _GensimKVShim,
    _load_pickle,
)


class TestEmbeddingsConstruction:
    def test_from_arrays(self):
        words = ["cat", "dog", "fish"]
        vecs = np.random.randn(3, 50).astype(np.float32)
        emb = Embeddings(words, vecs)
        assert emb.vector_size == 50
        assert len(emb) == 3

    def test_index_to_key(self):
        words = ["a", "b", "c"]
        vecs = np.random.randn(3, 10).astype(np.float32)
        emb = Embeddings(words, vecs)
        assert emb.index_to_key == ["a", "b", "c"]

    def test_getitem(self):
        words = ["cat", "dog"]
        vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        emb = Embeddings(words, vecs)
        np.testing.assert_array_equal(emb["cat"], vecs[0])

    def test_contains(self):
        words = ["cat", "dog"]
        vecs = np.random.randn(2, 10).astype(np.float32)
        emb = Embeddings(words, vecs)
        assert "cat" in emb
        assert "fish" not in emb

    def test_get_normed_vectors(self):
        words = ["a"]
        vecs = np.array([[3, 4, 0]], dtype=np.float32)
        emb = Embeddings(words, vecs)
        normed = emb.get_normed_vectors()
        np.testing.assert_allclose(np.linalg.norm(normed, axis=1), 1.0, atol=1e-6)

    def test_get_vector_norm(self):
        words = ["a"]
        vecs = np.array([[3, 4, 0]], dtype=np.float32)
        emb = Embeddings(words, vecs)
        v = emb.get_vector("a", norm=True)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-6)

    def test_similar_by_vector(self):
        words = ["a", "b", "c"]
        vecs = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
        emb = Embeddings(words, vecs)
        results = emb.similar_by_vector(np.array([1, 0], dtype=np.float32), topn=2)
        assert len(results) == 2
        assert results[0][0] == "a"  # closest to [1,0]

    def test_similar_by_vector_restrict_vocab(self):
        words = ["a", "b", "c", "d"]
        vecs = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1]], dtype=np.float32)
        emb = Embeddings(words, vecs)
        results = emb.similar_by_vector(
            np.array([1, 0], dtype=np.float32), topn=2, restrict_vocab=2
        )
        # Only considers first 2 words in vocabulary
        for word, _ in results:
            assert word in ("a", "b")


class TestEmbeddingsEdgeCases:
    """Cover empty(), add_vectors(), zero-vector, OOV, and zero-norm."""

    def test_empty_factory(self):
        emb = Embeddings.empty(50)
        assert len(emb) == 0
        assert emb.vector_size == 50

    def test_add_vectors_to_empty(self):
        emb = Embeddings.empty(3)
        words = ["a", "b"]
        vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        emb.add_vectors(words, vecs)
        assert len(emb) == 2
        assert emb.vector_size == 3
        assert "a" in emb
        np.testing.assert_array_equal(emb["a"], vecs[0])

    def test_add_vectors_to_existing(self):
        words1 = ["a"]
        vecs1 = np.array([[1, 0]], dtype=np.float32)
        emb = Embeddings(words1, vecs1)
        emb.add_vectors(["b"], np.array([[0, 1]], dtype=np.float32))
        assert len(emb) == 2
        assert "b" in emb
        # Norms cache should be invalidated
        assert emb._norms is None
        assert emb._normed_vectors is None

    def test_similar_by_zero_vector(self):
        words = ["a", "b"]
        vecs = np.array([[1, 0], [0, 1]], dtype=np.float32)
        emb = Embeddings(words, vecs)
        results = emb.similar_by_vector(np.zeros(2, dtype=np.float32))
        assert results == []

    def test_getitem_oov_raises(self):
        emb = Embeddings(["a"], np.array([[1, 0]], dtype=np.float32))
        with pytest.raises(KeyError):
            emb["nonexistent"]

    def test_get_vector_oov_raises(self):
        emb = Embeddings(["a"], np.array([[1, 0]], dtype=np.float32))
        with pytest.raises(KeyError):
            emb.get_vector("nonexistent")

    def test_zero_norm_vector_normalization(self):
        """Zero-norm vectors should not produce NaN after normalization."""
        words = ["a", "b"]
        vecs = np.array([[0, 0, 0], [3, 4, 0]], dtype=np.float32)
        emb = Embeddings(words, vecs)
        normed = emb.get_normed_vectors()
        assert not np.any(np.isnan(normed))
        # Zero vector stays near-zero (divided by 1e-12)
        np.testing.assert_allclose(np.linalg.norm(normed[1]), 1.0, atol=1e-6)

    def test_fill_norms_invalidates_normed_cache(self):
        emb = Embeddings(["a"], np.array([[3, 4]], dtype=np.float32))
        _ = emb.get_normed_vectors()  # populate cache
        assert emb._normed_vectors is not None
        emb.fill_norms()
        assert emb._normed_vectors is None  # invalidated

    def test_norms_property_lazy(self):
        emb = Embeddings(["a"], np.array([[3, 4]], dtype=np.float32))
        assert emb._norms is None
        norms = emb.norms
        np.testing.assert_allclose(norms[0], 5.0, atol=1e-5)


class TestEmbeddingsSave:
    """Test Embeddings.save() / load round-trip."""

    def test_save_load_roundtrip(self, tmp_path):
        words = ["cat", "dog", "fish"]
        vecs = np.random.randn(3, 5).astype(np.float32)
        emb = Embeddings(words, vecs)
        p = str(tmp_path / "test.kv")
        emb.save(p)
        # Check files exist
        assert os.path.exists(p)
        assert os.path.exists(p + ".vectors.npy")
        # Round-trip
        loaded = load_embeddings(p)
        assert len(loaded) == 3
        assert loaded.vector_size == 5
        assert loaded.index_to_key == words
        np.testing.assert_allclose(loaded["cat"], vecs[0], atol=1e-6)
        np.testing.assert_allclose(loaded["fish"], vecs[2], atol=1e-6)

    def test_save_preserves_original(self, tmp_path):
        """After save(), the in-memory object should be intact."""
        words = ["a", "b"]
        vecs = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        emb = Embeddings(words, vecs)
        p = str(tmp_path / "test.kv")
        emb.save(p)
        # Original object still works
        assert len(emb) == 2
        np.testing.assert_array_equal(emb["a"], vecs[0])
        assert emb.vectors.shape == (2, 3)

    def test_save_load_empty(self, tmp_path):
        """Saving/loading an empty Embeddings should work."""
        emb = Embeddings.empty(10)
        p = str(tmp_path / "empty.kv")
        emb.save(p)
        loaded = load_embeddings(p)
        assert len(loaded) == 0
        assert loaded.vector_size == 10


class TestSimilarByVectorEmpty:
    """Test similar_by_vector with empty/restricted embeddings."""

    def test_empty_embeddings(self):
        emb = Embeddings.empty(3)
        results = emb.similar_by_vector(np.array([1, 0, 0], dtype=np.float32))
        assert results == []

    def test_restrict_vocab_zero(self):
        emb = Embeddings(["a"], np.array([[1, 0]], dtype=np.float32))
        results = emb.similar_by_vector(
            np.array([1, 0], dtype=np.float32), restrict_vocab=0
        )
        assert results == []


class TestLoadEmbeddingsGzFallback:
    """Test standalone .gz error handling."""

    def test_standalone_gz_raises(self, tmp_path):
        p = str(tmp_path / "model.gz")
        with gzip.open(p, "wt") as f:
            f.write("some data\n")
        with pytest.raises(ValueError, match="Cannot determine embedding format"):
            load_embeddings(p)


class TestLoadKV:
    """Test .kv pickle loader and gensim shim."""

    def _make_fake_kv_pickle(self, path, words, vecs):
        """Create a pickle file that mimics gensim KeyedVectors structure."""
        shim = _GensimKVShim()
        shim.__dict__.update(
            {
                "index_to_key": list(words),
                "vectors": np.asarray(vecs, dtype=np.float32),
            }
        )
        with open(path, "wb") as f:
            pickle.dump(shim, f)

    def test_load_kv_basic(self, tmp_path):
        words = ["alpha", "beta"]
        vecs = np.random.randn(2, 5).astype(np.float32)
        p = str(tmp_path / "test.kv")
        self._make_fake_kv_pickle(p, words, vecs)
        emb = load_embeddings(p)
        assert "alpha" in emb
        assert emb.vector_size == 5
        np.testing.assert_allclose(emb["alpha"], vecs[0], atol=1e-6)

    def test_load_kv_with_npy_sidecar(self, tmp_path):
        """When pickle has empty vectors, load from .npy sidecar."""
        words = ["x", "y"]
        vecs = np.random.randn(2, 4).astype(np.float32)
        p = str(tmp_path / "test.kv")
        # Pickle with empty vectors
        shim = _GensimKVShim()
        shim.__dict__.update(
            {
                "index_to_key": list(words),
                "vectors": np.zeros((0, 4), dtype=np.float32),
            }
        )
        with open(p, "wb") as f:
            pickle.dump(shim, f)
        # Save sidecar
        np.save(p + ".vectors.npy", vecs)
        emb = _load_pickle(p)
        assert len(emb) == 2
        np.testing.assert_allclose(emb["x"], vecs[0], atol=1e-6)

    def test_gensim_shim_setstate_gensim3(self):
        """Test gensim 3.x attribute renames (index2word → index_to_key)."""
        shim = _GensimKVShim()
        state = {
            "index2word": ["a", "b"],
            "vectors": np.array([[1, 0], [0, 1]], dtype=np.float32),
        }
        shim.__setstate__(state)
        emb = shim.to_embeddings()
        assert "a" in emb
        assert emb.vector_size == 2

    def test_gensim_shim_setstate_index2entity(self):
        """Test gensim index2entity fallback."""
        shim = _GensimKVShim()
        state = {
            "index2entity": ["c", "d"],
            "vectors": np.array([[1, 0], [0, 1]], dtype=np.float32),
        }
        shim.__setstate__(state)
        emb = shim.to_embeddings()
        assert "c" in emb

    def test_gensim_shim_setstate_syn0(self):
        """Test gensim syn0 → vectors rename."""
        shim = _GensimKVShim()
        state = {
            "index_to_key": ["e", "f"],
            "syn0": np.array([[1, 0], [0, 1]], dtype=np.float32),
        }
        shim.__setstate__(state)
        emb = shim.to_embeddings()
        assert emb.vector_size == 2

    def test_load_kv_invalid_object_raises(self, tmp_path):
        """Loading a pickle that isn't a KeyedVectors-like should raise."""
        p = str(tmp_path / "bad.kv")
        with open(p, "wb") as f:
            pickle.dump({"not": "a keyed vectors"}, f)
        with pytest.raises(ValueError, match="unexpected object type"):
            load_embeddings(p)

    def test_load_embeddings_fallback_kv(self, tmp_path):
        """Unknown extension falls back to .kv loader."""
        words = ["g", "h"]
        vecs = np.random.randn(2, 3).astype(np.float32)
        p = str(tmp_path / "test.dat")
        self._make_fake_kv_pickle(p, words, vecs)
        emb = load_embeddings(p)
        assert "g" in emb

    def test_load_embeddings_vec_extension(self, tmp_path):
        """Explicitly test .vec extension dispatch."""
        words = ["i", "j"]
        vecs = np.random.randn(2, 3).astype(np.float32)
        p = str(tmp_path / "emb.vec")
        with open(p, "w") as f:
            f.write("2 3\n")
            for w, v in zip(words, vecs):
                f.write(w + " " + " ".join(f"{x:.6f}" for x in v) + "\n")
        emb = load_embeddings(p)
        assert len(emb) == 2


class TestLoadTextSkippedLines:
    """Test that lines with <2 parts are skipped in text loader."""

    def test_skip_empty_lines(self, tmp_path):
        p = str(tmp_path / "emb.txt")
        with open(p, "w") as f:
            f.write("2 3\n")
            f.write("cat 1.0 0.0 0.0\n")
            f.write("\n")  # empty line
            f.write("dog 0.0 1.0 0.0\n")
        emb = load_embeddings(p)
        assert len(emb) == 2


class TestLoadText:
    """Load .txt / .vec (word2vec text format)."""

    def _write_text(self, path, words, vecs, header=True):
        with open(path, "w") as f:
            if header:
                f.write(f"{len(words)} {vecs.shape[1]}\n")
            for w, v in zip(words, vecs):
                f.write(w + " " + " ".join(f"{x:.6f}" for x in v) + "\n")

    def test_load_txt_with_header(self, tmp_path):
        words = ["cat", "dog"]
        vecs = np.random.randn(2, 5).astype(np.float32)
        p = str(tmp_path / "emb.txt")
        self._write_text(p, words, vecs)
        emb = load_embeddings(p)
        assert "cat" in emb
        assert emb.vector_size == 5

    def test_load_txt_no_header(self, tmp_path):
        """GloVe-style: no header line."""
        words = ["cat", "dog"]
        vecs = np.random.randn(2, 5).astype(np.float32)
        p = str(tmp_path / "emb.txt")
        self._write_text(p, words, vecs, header=False)
        emb = load_embeddings(p)
        assert len(emb) == 2

    def test_load_vec_gz(self, tmp_path):
        words = ["a", "b"]
        vecs = np.random.randn(2, 3).astype(np.float32)
        p = str(tmp_path / "emb.vec.gz")
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write("2 3\n")
            for w, v in zip(words, vecs):
                f.write(w + " " + " ".join(f"{x:.6f}" for x in v) + "\n")
        emb = load_embeddings(p)
        assert len(emb) == 2


class TestLoadBinary:
    """Load .bin (word2vec binary format)."""

    def _write_bin(self, path, words, vecs):
        with open(path, "wb") as f:
            f.write(f"{len(words)} {vecs.shape[1]}\n".encode("utf-8"))
            for w, v in zip(words, vecs):
                f.write(w.encode("utf-8") + b" ")
                f.write(v.astype(np.float32).tobytes())

    def test_load_bin(self, tmp_path):
        words = ["hello", "world"]
        vecs = np.random.randn(2, 4).astype(np.float32)
        p = str(tmp_path / "emb.bin")
        self._write_bin(p, words, vecs)
        emb = load_embeddings(p)
        assert len(emb) == 2
        np.testing.assert_allclose(emb["hello"], vecs[0], atol=1e-6)
