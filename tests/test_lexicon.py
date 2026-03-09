# tests/test_lexicon.py
"""Tests for ssdiff.lexicon — suggest_lexicon, token_presence_stats, coverage_by_lexicon."""

import numpy as np
import pandas as pd
import pytest

from ssdiff.lexicon import suggest_lexicon, token_presence_stats, coverage_by_lexicon


# ── Shared test data ────────────────────────────────────────────────────


@pytest.fixture
def lex_texts():
    return [
        ["kraj", "piekny", "dom"],
        ["narod", "silny", "dom"],
        ["kraj", "wielki", "szkola"],
        ["panstwo", "maly", "dom"],
        ["kraj", "nowy", "miasto"],
        ["narod", "dobry", "szkola"],
        ["panstwo", "stary", "dom"],
        ["kraj", "silny", "miasto"],
    ]


@pytest.fixture
def lex_y():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


@pytest.fixture
def lex_groups():
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)


# ── suggest_lexicon ─────────────────────────────────────────────────────


class TestSuggestLexicon:
    def test_returns_dataframe(self, lex_texts, lex_y):
        df = suggest_lexicon((lex_texts, lex_y), top_k=10, min_docs=2)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {
            "token",
            "docs",
            "cov_all",
            "cov_bal",
            "corr",
            "rank",
        }

    def test_top_k_limit(self, lex_texts, lex_y):
        df = suggest_lexicon((lex_texts, lex_y), top_k=3, min_docs=1)
        assert len(df) <= 3

    def test_min_docs_filter(self, lex_texts, lex_y):
        df = suggest_lexicon((lex_texts, lex_y), top_k=50, min_docs=4)
        # "kraj" appears 4 times, "dom" 4 times
        assert all(row["docs"] >= 4 for _, row in df.iterrows())

    def test_sorted_by_rank(self, lex_texts, lex_y):
        df = suggest_lexicon((lex_texts, lex_y), top_k=10, min_docs=1)
        if len(df) >= 2:
            ranks = df["rank"].tolist()
            assert ranks == sorted(ranks, reverse=True)

    def test_categorical_mode(self, lex_texts, lex_groups):
        df = suggest_lexicon(
            (lex_texts, lex_groups), top_k=10, min_docs=1, var_type="categorical"
        )
        assert isinstance(df, pd.DataFrame)
        assert "corr" in df.columns  # Cramér's V

    def test_dataframe_input(self, lex_texts, lex_y):
        df_input = pd.DataFrame(
            {
                "text": [" ".join(t) for t in lex_texts],
                "score": lex_y,
            }
        )
        result = suggest_lexicon(
            df_input, text_col="text", score_col="score", top_k=10, min_docs=1
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_invalid_var_type(self, lex_texts, lex_y):
        with pytest.raises(ValueError, match="var_type"):
            suggest_lexicon((lex_texts, lex_y), var_type="wrong")


# ── token_presence_stats ────────────────────────────────────────────────


class TestTokenPresenceStats:
    def test_returns_dict(self, lex_texts, lex_y):
        result = token_presence_stats(lex_texts, lex_y, "kraj")
        assert isinstance(result, dict)
        assert set(result.keys()) >= {
            "token",
            "docs",
            "cov_all",
            "cov_bal",
            "corr",
            "rank",
            "q1",
            "q4",
        }

    def test_token_name(self, lex_texts, lex_y):
        result = token_presence_stats(lex_texts, lex_y, "kraj")
        assert result["token"] == "kraj"

    def test_docs_count(self, lex_texts, lex_y):
        result = token_presence_stats(lex_texts, lex_y, "kraj")
        assert result["docs"] == 4  # "kraj" in docs 0,2,4,7

    def test_absent_token(self, lex_texts, lex_y):
        result = token_presence_stats(lex_texts, lex_y, "zzzzz")
        assert result["docs"] == 0
        assert result["cov_all"] == 0.0

    def test_categorical(self, lex_texts, lex_groups):
        result = token_presence_stats(
            lex_texts, lex_groups, "dom", var_type="categorical"
        )
        assert "group_cov" in result
        assert isinstance(result["group_cov"], dict)

    def test_verbose_output(self, lex_texts, lex_y, capsys):
        token_presence_stats(lex_texts, lex_y, "kraj", verbose=True)
        assert "kraj" in capsys.readouterr().out


# ── coverage_by_lexicon ─────────────────────────────────────────────────


class TestCoverageByLexicon:
    def test_returns_tuple(self, lex_texts, lex_y):
        summary, per_token = coverage_by_lexicon(
            (lex_texts, lex_y), lexicon=["kraj", "narod"]
        )
        assert isinstance(summary, dict)
        assert isinstance(per_token, pd.DataFrame)

    def test_summary_keys(self, lex_texts, lex_y):
        summary, _ = coverage_by_lexicon((lex_texts, lex_y), lexicon=["kraj"])
        assert "docs_any" in summary
        assert "cov_all" in summary
        assert "hits_mean" in summary

    def test_per_token_columns(self, lex_texts, lex_y):
        _, per_token = coverage_by_lexicon((lex_texts, lex_y), lexicon=["kraj", "dom"])
        assert "word" in per_token.columns
        assert "docs" in per_token.columns
        assert len(per_token) == 2

    def test_categorical_mode(self, lex_texts, lex_groups):
        summary, per_token = coverage_by_lexicon(
            (lex_texts, lex_groups), lexicon=["kraj"], var_type="categorical"
        )
        assert "group_cov" in summary

    def test_dataframe_input(self, lex_texts, lex_y):
        df = pd.DataFrame(
            {
                "text": [" ".join(t) for t in lex_texts],
                "score": lex_y,
            }
        )
        summary, per_token = coverage_by_lexicon(
            df, text_col="text", score_col="score", lexicon=["kraj"]
        )
        assert summary["docs_any"] > 0

    def test_empty_lexicon(self, lex_texts, lex_y):
        summary, per_token = coverage_by_lexicon((lex_texts, lex_y), lexicon=[])
        assert summary["docs_any"] == 0
        assert len(per_token) == 0
