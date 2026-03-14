# tests/test_io_utils.py
"""Tests for ssdiff.io_utils — save/load preprocessed bundles."""

import pytest
from ssdiff.io_utils import save_preprocessed_bundle, load_preprocessed_bundle
from ssdiff.preprocess import PreprocessedDoc


@pytest.fixture
def sample_pre_docs():
    return [
        PreprocessedDoc(
            raw="Kraj jest piekny.",
            sents_surface=["Kraj jest piekny."],
            sents_lemmas=[["kraj", "piekny"]],
            doc_lemmas=["kraj", "piekny"],
            sent_char_spans=[(0, 17)],
            token_to_sent=[0, 0],
            sents_kept_idx=[[0, 2]],
        ),
        PreprocessedDoc(
            raw="Narod jest wielki.",
            sents_surface=["Narod jest wielki."],
            sents_lemmas=[["narod", "wielki"]],
            doc_lemmas=["narod", "wielki"],
            sent_char_spans=[(0, 18)],
            token_to_sent=[0, 0],
            sents_kept_idx=[[0, 2]],
        ),
    ]


class TestSaveLoadBundle:
    def test_roundtrip_gzip(self, tmp_path, sample_pre_docs):
        path = str(tmp_path / "test_bundle")
        saved = save_preprocessed_bundle(sample_pre_docs, path, compress="gzip")
        assert saved.endswith(".pkl.gz")

        payload = load_preprocessed_bundle(saved)
        assert "meta" in payload
        assert "pre_docs" in payload
        assert len(payload["pre_docs"]) == 2
        assert payload["meta"]["kind"] == "doc"

    def test_roundtrip_raw(self, tmp_path, sample_pre_docs):
        path = str(tmp_path / "test_bundle")
        saved = save_preprocessed_bundle(sample_pre_docs, path, compress="raw")
        assert saved.endswith(".pkl")

        payload = load_preprocessed_bundle(saved)
        assert len(payload["pre_docs"]) == 2

    def test_metadata_preserved(self, tmp_path, sample_pre_docs):
        path = str(tmp_path / "bundle_meta")
        saved = save_preprocessed_bundle(
            sample_pre_docs,
            path,
            authors=["Alice", "Bob"],
            spaCy_model="pl_core_news_sm",
            extra_meta={"version": "1.0"},
        )
        payload = load_preprocessed_bundle(saved)
        meta = payload["meta"]
        assert meta["spaCy_model"] == "pl_core_news_sm"
        assert meta["authors_len"] == 2
        assert meta["extra"]["version"] == "1.0"
        assert payload["authors"] == ["Alice", "Bob"]

    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_preprocessed_bundle([], str(tmp_path / "empty"))

    def test_doc_content_preserved(self, tmp_path, sample_pre_docs):
        path = str(tmp_path / "content_check")
        saved = save_preprocessed_bundle(sample_pre_docs, path)
        payload = load_preprocessed_bundle(saved)
        doc = payload["pre_docs"][0]
        assert doc.raw == "Kraj jest piekny."
        assert doc.doc_lemmas == ["kraj", "piekny"]
