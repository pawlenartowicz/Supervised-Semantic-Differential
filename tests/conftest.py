# tests/conftest.py
"""Shared fixtures for ssdiff test suite.

All fixtures use tiny in-memory Embeddings so tests run without
real embeddings, network access, or spaCy downloads.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass
from ssdiff.embeddings import Embeddings

from ssdiff.preprocess import PreprocessedDoc


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
_VOCAB_20 = [
    "kraj",
    "narod",
    "panstwo",  # seeds (lexicon)
    "piekny",
    "silny",
    "zly",
    "dobry",  # context words
    "wielki",
    "maly",
    "stary",
    "nowy",
    "dom",
    "szkola",
    "praca",
    "miasto",
    "rzeka",
    "gora",
    "las",
    "ABC123",
    "Warszawa",  # "bad" tokens for filter tests
]

_VOCAB_50 = _VOCAB_20 + [
    "ludzie",
    "czas",
    "swiat",
    "dzien",
    "noc",
    "woda",
    "ogien",
    "ziemia",
    "niebo",
    "slonce",
    "droga",
    "pole",
    "morze",
    "kwiat",
    "drzewo",
    "kamien",
    "wiatr",
    "deszcz",
    "snieg",
    "chmura",
    "ptak",
    "ryba",
    "kon",
    "pies",
    "kot",
    "serce",
    "reka",
    "glowa",
    "oko",
    "usta",
]

# ---------------------------------------------------------------------------
# KeyedVectors helpers
# ---------------------------------------------------------------------------


def _make_kv(words: list[str], dim: int, seed: int = 42) -> Embeddings:
    """Build a tiny Embeddings with unit-normalized random vectors."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(len(words), dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return Embeddings(words, mat)


@pytest.fixture(scope="session")
def tiny_kv() -> Embeddings:
    """20-word, 8-dimensional Embeddings (unit-normalized)."""
    return _make_kv(_VOCAB_20, dim=8, seed=42)


@pytest.fixture(scope="session")
def tiny_kv_large() -> Embeddings:
    """50-word, 10-dimensional Embeddings for clustering tests."""
    return _make_kv(_VOCAB_50, dim=10, seed=99)


# ---------------------------------------------------------------------------
# Lexicon / docs / outcome
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_lexicon() -> set[str]:
    return {"kraj", "narod", "panstwo"}


@pytest.fixture(scope="session")
def sample_docs() -> list[list[str]]:
    """8 documents, each containing >=1 seed from sample_lexicon."""
    return [
        ["kraj", "piekny", "dom", "silny"],
        ["narod", "wielki", "miasto", "dobry"],
        ["panstwo", "silny", "szkola", "nowy"],
        ["kraj", "zly", "praca", "maly"],
        ["narod", "piekny", "rzeka", "stary"],
        ["panstwo", "dobry", "gora", "wielki"],
        ["kraj", "nowy", "las", "silny"],
        ["narod", "maly", "dom", "zly"],
    ]


@pytest.fixture(scope="session")
def sample_docs_no_seeds() -> list[list[str]]:
    """4 documents with NO lexicon seeds."""
    return [
        ["piekny", "dom", "silny"],
        ["wielki", "miasto", "dobry"],
        ["szkola", "nowy", "maly"],
        ["rzeka", "stary", "gora"],
    ]


@pytest.fixture(scope="session")
def sample_y() -> np.ndarray:
    return np.array([1.0, 1.2, 0.9, 0.8, 1.5, 1.1, 0.7, 1.3])


@pytest.fixture(scope="session")
def sample_y_with_nan() -> np.ndarray:
    return np.array([1.0, 1.2, 0.9, np.nan, 1.5, 1.1, 0.7, 1.3])


@pytest.fixture(scope="session")
def sample_groups() -> np.ndarray:
    """8-element categorical labels (2 groups)."""
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)


@pytest.fixture(scope="session")
def sample_groups_3() -> np.ndarray:
    """8-element categorical labels (3 groups)."""
    return np.array(["X", "X", "X", "Y", "Y", "Z", "Z", "Z"], dtype=object)


# ---------------------------------------------------------------------------
# Fitted SSD instances
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fitted_ssd(tiny_kv, sample_docs, sample_y, sample_lexicon):
    from ssdiff import SSD

    return SSD(
        kv=tiny_kv,
        docs=sample_docs,
        y=sample_y,
        lexicon=sample_lexicon,
        l2_normalize_docs=True,
        use_unit_beta=True,
        N_PCA=3,
    )


@pytest.fixture(scope="session")
def fitted_ssd_large(tiny_kv_large, sample_lexicon):
    """SSD with 50-word embeddings (for cluster tests needing more neighbors)."""
    from ssdiff import SSD

    rng = np.random.default_rng(7)
    # 20 docs to give clustering enough material
    docs = []
    vocab = list(tiny_kv_large.index_to_key)
    seeds = list(sample_lexicon & set(vocab))
    for _ in range(20):
        doc = list(rng.choice(seeds, size=1)) + list(rng.choice(vocab, size=6))
        docs.append(doc)
    y = rng.normal(1.0, 0.3, size=20)
    return SSD(
        kv=tiny_kv_large,
        docs=docs,
        y=y,
        lexicon=sample_lexicon,
        N_PCA=3,
    )


@pytest.fixture(scope="session")
def fitted_ssd_group(tiny_kv, sample_docs, sample_groups, sample_lexicon):
    from ssdiff import SSDGroup

    return SSDGroup(
        kv=tiny_kv,
        docs=sample_docs,
        groups=sample_groups,
        lexicon=sample_lexicon,
        n_perm=50,
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Preprocessed docs (manually built, no spaCy needed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_preprocessed_docs() -> list[PreprocessedDoc]:
    """4 PreprocessedDoc objects built by hand (no spaCy required)."""
    return [
        PreprocessedDoc(
            raw="Kraj jest piekny i silny.",
            sents_surface=["Kraj jest piekny i silny."],
            sents_lemmas=[["kraj", "piekny", "silny"]],
            doc_lemmas=["kraj", "piekny", "silny"],
            sent_char_spans=[(0, 26)],
            token_to_sent=[0, 0, 0],
            sents_kept_idx=[[0, 2, 4]],
        ),
        PreprocessedDoc(
            raw="Narod jest wielki. Miasto jest duze.",
            sents_surface=["Narod jest wielki.", "Miasto jest duze."],
            sents_lemmas=[["narod", "wielki"], ["miasto"]],
            doc_lemmas=["narod", "wielki", "miasto"],
            sent_char_spans=[(0, 18), (19, 35)],
            token_to_sent=[0, 0, 1],
            sents_kept_idx=[[0, 2], [0]],
        ),
        PreprocessedDoc(
            raw="Panstwo i szkola sa nowe.",
            sents_surface=["Panstwo i szkola sa nowe."],
            sents_lemmas=[["panstwo", "szkola", "nowy"]],
            doc_lemmas=["panstwo", "szkola", "nowy"],
            sent_char_spans=[(0, 24)],
            token_to_sent=[0, 0, 0],
            sents_kept_idx=[[0, 2, 4]],
        ),
        PreprocessedDoc(
            raw="Dom i praca w miescie.",
            sents_surface=["Dom i praca w miescie."],
            sents_lemmas=[["dom", "praca", "miasto"]],
            doc_lemmas=["dom", "praca", "miasto"],
            sent_char_spans=[(0, 22)],
            token_to_sent=[0, 0, 0],
            sents_kept_idx=[[0, 2, 4]],
        ),
    ]


# ---------------------------------------------------------------------------
# Mock spaCy objects (for preprocess tests)
# ---------------------------------------------------------------------------


@dataclass
class MockToken:
    text: str
    lemma_: str
    is_space: bool = False
    is_punct: bool = False
    is_quote: bool = False
    is_currency: bool = False
    is_digit: bool = False


class MockSent:
    def __init__(
        self, tokens: list[MockToken], text: str, start_char: int, end_char: int
    ):
        self._tokens = tokens
        self.text = text
        self.start_char = start_char
        self.end_char = end_char

    def __iter__(self):
        return iter(self._tokens)


class MockDoc:
    def __init__(self, sents: list[MockSent], text: str):
        self._sents = sents
        self.text = text

    @property
    def sents(self):
        return iter(self._sents)


def _fake_nlp_pipe(texts, batch_size=64, n_process=1):
    """Yield MockDoc objects that mimic spaCy's pipe output."""
    for t in texts:
        words = t.split()
        tokens = [MockToken(text=w, lemma_=w.lower()) for w in words]
        sent = MockSent(tokens, text=t, start_char=0, end_char=len(t))
        yield MockDoc(sents=[sent], text=t)


class FakeNlp:
    """Minimal mock of a spaCy Language object."""

    pipe_names = ["sentencizer"]

    def pipe(self, texts, batch_size=64, n_process=1):
        return _fake_nlp_pipe(texts, batch_size, n_process)


@pytest.fixture
def fake_nlp():
    return FakeNlp()
