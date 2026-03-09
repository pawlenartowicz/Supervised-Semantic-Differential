# tests/test_preprocess.py
"""Tests for ssdiff.preprocess — load_spacy/stopwords (mocked), preprocess_texts, build_docs."""

from unittest.mock import patch, MagicMock

from ssdiff.preprocess import (
    PreprocessedDoc,
    PreprocessedProfile,
    preprocess_texts,
    build_docs_from_preprocessed,
    load_spacy,
    load_stopwords,
)


# ── load_spacy (mocked) ────────────────────────────────────────────────


class TestLoadSpacy:
    def test_empty_model_returns_none(self, capsys):
        result = load_spacy(None)
        assert result is None
        assert "Provide" in capsys.readouterr().out

    def test_empty_string_returns_none(self, capsys):
        result = load_spacy("")
        assert result is None

    @patch("ssdiff.preprocess.spacy")
    def test_successful_load(self, mock_spacy):
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = ["parser"]
        mock_spacy.load.return_value = mock_nlp
        result = load_spacy("en_core_web_sm")
        assert result is mock_nlp
        mock_spacy.load.assert_called_once()

    @patch("ssdiff.preprocess.spacy")
    def test_adds_sentencizer_when_missing(self, mock_spacy):
        mock_nlp = MagicMock()
        mock_nlp.pipe_names = []  # no parser, no sentencizer
        mock_spacy.load.return_value = mock_nlp
        load_spacy("test_model")
        mock_nlp.add_pipe.assert_called_once_with("sentencizer")

    @patch("ssdiff.preprocess.spacy")
    def test_load_failure(self, mock_spacy, capsys):
        mock_spacy.load.side_effect = OSError("not found")
        result = load_spacy("nonexistent_model")
        assert result is None
        assert "Could not load" in capsys.readouterr().out


# ── load_stopwords (mocked) ────────────────────────────────────────────


class TestLoadStopwords:
    @patch("ssdiff.preprocess.requests")
    def test_polish_fetches_from_github(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.text = "i\nw\nna\ndo\n"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        # Clear lru_cache for this test
        load_stopwords.cache_clear()
        words = load_stopwords("pl")
        assert "i" in words
        assert "w" in words
        load_stopwords.cache_clear()

    @patch("ssdiff.preprocess.spacy")
    def test_non_polish_uses_spacy(self, mock_spacy):
        mock_blank = MagicMock()
        mock_blank.Defaults.stop_words = {"the", "a", "is"}
        mock_spacy.blank.return_value = mock_blank

        load_stopwords.cache_clear()
        words = load_stopwords("en")
        assert "the" in words
        load_stopwords.cache_clear()


# ── preprocess_texts (with fake_nlp from conftest) ──────────────────────


class TestPreprocessTexts:
    def test_returns_empty_without_nlp(self, capsys):
        result = preprocess_texts(["hello world"], nlp=None)
        assert result == []
        assert "Call load_spacy" in capsys.readouterr().out

    def test_flat_mode(self, fake_nlp):
        texts = ["Kraj jest piekny", "Narod jest silny"]
        result = preprocess_texts(texts, nlp=fake_nlp)
        assert len(result) == 2
        assert all(isinstance(r, PreprocessedDoc) for r in result)

    def test_flat_mode_doc_structure(self, fake_nlp):
        texts = ["Kraj jest piekny"]
        result = preprocess_texts(texts, nlp=fake_nlp)
        doc = result[0]
        assert doc.raw == "Kraj jest piekny"
        assert len(doc.sents_surface) >= 1
        assert isinstance(doc.doc_lemmas, list)

    def test_profile_mode(self, fake_nlp):
        profiles = [
            ["Post one here", "Post two here"],
            ["Single post"],
        ]
        result = preprocess_texts(profiles, nlp=fake_nlp)
        assert len(result) == 2
        assert all(isinstance(r, PreprocessedProfile) for r in result)

    def test_profile_empty_posts(self, fake_nlp):
        profiles = [[], ["Some text"]]
        result = preprocess_texts(profiles, nlp=fake_nlp)
        assert len(result) == 2
        assert result[0].raw_posts == []

    def test_handles_none_text(self, fake_nlp):
        texts = [None, "Valid text"]
        result = preprocess_texts(texts, nlp=fake_nlp)
        assert len(result) == 2


# ── build_docs_from_preprocessed ────────────────────────────────────────


class TestBuildDocsFromPreprocessed:
    def test_flat_docs(self, sample_preprocessed_docs):
        docs = build_docs_from_preprocessed(sample_preprocessed_docs)
        assert isinstance(docs, list)
        assert len(docs) == 4
        assert all(isinstance(d, list) for d in docs)
        assert all(isinstance(t, str) for d in docs for t in d)

    def test_empty_input(self):
        result = build_docs_from_preprocessed([])
        assert result == []

    def test_profile_mode(self):
        profiles = [
            PreprocessedProfile(
                raw_posts=["A", "B"],
                post_sents_surface=[["A"], ["B"]],
                post_sents_lemmas=[[["a"]], [["b"]]],
                post_doc_lemmas=[["a"], ["b"]],
                post_sent_char_spans=[[(0, 1)], [(0, 1)]],
                post_token_to_sent=[[0], [0]],
                post_sents_kept_idx=[[[0]], [[0]]],
            )
        ]
        docs = build_docs_from_preprocessed(profiles)
        assert isinstance(docs, list)
        assert len(docs) == 1
        assert len(docs[0]) == 2  # 2 posts
