# ===== ssdiff/preprocess.py (UNIFIED, BACKWARDS-COMPATIBLE) =====
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional, Union, Tuple
import re
import requests
from functools import lru_cache

import spacy

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None


# ---------- stopwords ----------
@lru_cache(maxsize=16)
def load_stopwords(
    lang: str = "pl",
    *,
    lowercase: bool = True,
    timeout: float = 5.0,
) -> List[str]:
    """
    Load stopwords.

    - For Polish ("pl"): fetch the original list from GitHub (bieli/stopwords).
    - For other languages: use spaCy's built-in stopwords
      (spacy.blank(lang).Defaults.stop_words).
    """
    lang = (lang or "pl").strip().lower()

    if lang == "pl":
        url = "https://raw.githubusercontent.com/bieli/stopwords/master/polish.stopwords.txt"
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
        except (requests.RequestException, OSError) as exc:
            raise RuntimeError(
                f"Failed to fetch Polish stopwords from {url}: {exc}. "
                "Provide stopwords manually via the `stopwords` parameter, "
                "or check your internet connection."
            ) from exc
        words = [s.strip() for s in r.text.splitlines() if s.strip()]
        if not words:
            raise ValueError("Fetched Polish stopword list is empty.")
        return [w.lower() for w in words] if lowercase else words

    nlp_blank = spacy.blank(lang)
    sw = getattr(nlp_blank.Defaults, "stop_words", None)
    if not sw:
        raise LookupError(f"No stopwords available in spaCy for language '{lang}'.")
    words = list(sw)
    return [w.lower() for w in words] if lowercase else words


# ---------- spaCy loader ----------
def load_spacy(
    model: Optional[str] = None,
    *,
    disable: Sequence[str] = ("ner",),
) -> Optional["spacy.language.Language"]:
    """
    Load a spaCy model with light sanity checks.

    Backwards compatible with older usage: load_spacy("pl_core_news_sm").
    """
    if not model or not isinstance(model, str) or not model.strip():
        raise ValueError(
            "Provide a spaCy model name (e.g., 'pl_core_news_sm' or 'en_core_web_sm')."
        )

    try:
        nlp = spacy.load(model, disable=list(disable))
        # ensure we have sentence boundaries
        if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception as e:
        raise RuntimeError(
            f"Could not load spaCy model '{model}': {e}. See https://spacy.io/models"
        ) from e


# ---------- token filter ----------
_URL = re.compile(r"https?://\S+")
_AT = re.compile(r"@\S+")


def _keep_token(tok, stopset: set[str]) -> bool:
    if tok.is_space or tok.is_punct or tok.is_quote or tok.is_currency:
        return False
    if _URL.match(tok.text) or _AT.match(tok.text):
        return False
    if tok.is_digit:
        return False
    lem = tok.lemma_.lower()
    if not lem:
        return False
    if lem in stopset:
        return False
    return True


# ----------------- Data structures -----------------
@dataclass
class PreprocessedDoc:
    raw: str
    sents_surface: List[str]
    sents_lemmas: List[List[str]]
    doc_lemmas: List[str]
    sent_char_spans: List[Tuple[int, int]]
    token_to_sent: List[int]
    sents_kept_idx: List[List[int]]


@dataclass
class PreprocessedProfile:
    raw_posts: List[str]
    post_sents_surface: List[List[str]]
    post_sents_lemmas: List[List[List[str]]]
    post_doc_lemmas: List[List[str]]
    post_sent_char_spans: List[List[Tuple[int, int]]]
    post_token_to_sent: List[List[int]]
    post_sents_kept_idx: List[List[List[int]]]


# ----------------- helpers -----------------
def _pipe(nlp, texts: Sequence[str], batch_size: int, n_process: int):
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    # n_process=1 by default to avoid heavy multiproc startup
    return nlp.pipe(texts, batch_size=batch_size, n_process=n_process)


def _extract_from_doc(doc, stopset: set[str]):
    s_surface, s_lemmas, s_spans, s_kept_idx = [], [], [], []
    doc_lemmas, token_to_sent = [], []
    for si, sent in enumerate(doc.sents):
        s_surface.append(sent.text)
        s_spans.append((sent.start_char, sent.end_char))
        kept_lemmas, kept_idx = [], []
        for j, tok in enumerate(sent):
            if _keep_token(tok, stopset):
                kept_lemmas.append(tok.lemma_.lower())
                kept_idx.append(j)
        s_lemmas.append(kept_lemmas)
        s_kept_idx.append(kept_idx)
        start_flat = len(doc_lemmas)
        doc_lemmas.extend(kept_lemmas)
        token_to_sent.extend([si] * (len(doc_lemmas) - start_flat))
    return s_surface, s_lemmas, doc_lemmas, s_spans, token_to_sent, s_kept_idx


def _is_profile_input(texts: Sequence[Union[str, Sequence[str]]]) -> bool:
    """
    Detect whether `texts` is:
      - Sequence[str]                -> False (single-doc mode)
      - Sequence[Sequence[str]]     -> True  (profile mode)
    """
    for x in texts:
        if x is None:
            continue
        return not isinstance(x, (str, bytes))
    return False


def _sanitize_posts(posts: Sequence[Union[str, bytes, object]]) -> List[str]:
    out: List[str] = []
    for p in posts or []:
        if isinstance(p, bytes):
            p = p.decode(errors="ignore")
        if isinstance(p, str):
            s = p.strip()
            if s:
                out.append(s)
    return out


# ----------------- Public API -----------------
def preprocess_texts(
    texts: Sequence[Union[str, Sequence[str]]],
    nlp=None,
    stopwords: Optional[Sequence[str]] = None,
    batch_size: int = 64,
    n_process: int = 1,
    show_progress: bool = False,
    progress_desc: str = "Preprocessing",
) -> List[Union[PreprocessedDoc, PreprocessedProfile]]:
    """
    Preprocess texts.

    Backwards compatible modes:

    1) Old simple mode (non-grouped):
       texts: Sequence[str]
       → returns List[PreprocessedDoc]

    2) Profile mode (grouped):
       texts: Sequence[Sequence[str]]
       → returns List[PreprocessedProfile]

    Parameters
    ----------
    n_process : int
        Number of processes for spaCy.pipe.
        - Default 1  → avoids multiprocessing startup overhead.
        - Set to -1  → use all cores (like your FAST version).
    """
    if nlp is None:
        raise ValueError("nlp is None. Call load_spacy(model) first and pass the result.")

    stopset = set(stopwords or [])
    out: List[Union[PreprocessedDoc, PreprocessedProfile]] = []

    tq = _tqdm if show_progress and _tqdm is not None else None

    # -------- detect mode --------
    if not _is_profile_input(texts):
        # -------- single-doc mode (old behavior) --------
        texts_str = []
        for t in texts:
            if t is None or (isinstance(t, float) and t != t):
                texts_str.append("")
                continue
            if isinstance(t, bytes):
                t = t.decode(errors="ignore")
            texts_str.append(t if isinstance(t, str) else str(t))

        it = _pipe(nlp, texts_str, batch_size, n_process)
        if tq:
            it = tq(it, total=len(texts_str), desc=progress_desc)

        for doc in it:
            s_surface, s_lemmas, doc_lemmas, s_spans, token_to_sent, s_kept_idx = (
                _extract_from_doc(doc, stopset)
            )
            out.append(
                PreprocessedDoc(
                    raw=doc.text,
                    sents_surface=s_surface,
                    sents_lemmas=s_lemmas,
                    doc_lemmas=doc_lemmas,
                    sent_char_spans=s_spans,
                    token_to_sent=token_to_sent,
                    sents_kept_idx=s_kept_idx,
                )
            )
        return out

    # ================== PROFILE MODE (nested lists) ==================
    profiles: Sequence[Sequence[Union[str, bytes, object]]] = texts  # type: ignore
    posts_per_profile: List[List[str]] = [_sanitize_posts(p) for p in profiles]
    lengths: List[int] = [len(p) for p in posts_per_profile]

    # 2) make one flat list of posts
    flat_posts: List[str] = [pp for plist in posts_per_profile for pp in plist]

    # 3) parallel spaCy pass over flat list
    pit = _pipe(nlp, flat_posts, batch_size, n_process)
    if tq:
        pit = tq(pit, total=len(flat_posts), desc=f"{progress_desc} [flat]")

    out_profiles: List[PreprocessedProfile] = []
    prof_idx = 0
    remaining = lengths[0] if lengths else 0

    # emit leading empty profiles (length==0)
    while prof_idx < len(lengths) and remaining == 0:
        out_profiles.append(
            PreprocessedProfile(
                raw_posts=[],
                post_sents_surface=[],
                post_sents_lemmas=[],
                post_doc_lemmas=[],
                post_sent_char_spans=[],
                post_token_to_sent=[],
                post_sents_kept_idx=[],
            )
        )
        prof_idx += 1
        remaining = lengths[prof_idx] if prof_idx < len(lengths) else 0

    # accumulators for current profile
    cur_surface: List[List[str]] = []
    cur_lemmas: List[List[List[str]]] = []
    cur_docs: List[List[str]] = []
    cur_spans: List[List[Tuple[int, int]]] = []
    cur_tok2sent: List[List[int]] = []
    cur_kept_idx: List[List[List[int]]] = []

    for doc in pit:
        s_surface, s_lemmas, doc_lemmas, s_spans, token_to_sent, s_kept_idx = (
            _extract_from_doc(doc, stopset)
        )

        cur_surface.append(s_surface)
        cur_lemmas.append(s_lemmas)
        cur_docs.append(doc_lemmas)
        cur_spans.append(s_spans)
        cur_tok2sent.append(token_to_sent)
        cur_kept_idx.append(s_kept_idx)
        remaining -= 1

        if remaining == 0:
            out_profiles.append(
                PreprocessedProfile(
                    raw_posts=posts_per_profile[prof_idx],
                    post_sents_surface=cur_surface,
                    post_sents_lemmas=cur_lemmas,
                    post_doc_lemmas=cur_docs,
                    post_sent_char_spans=cur_spans,
                    post_token_to_sent=cur_tok2sent,
                    post_sents_kept_idx=cur_kept_idx,
                )
            )
            prof_idx += 1

            # reset accumulators
            cur_surface, cur_lemmas, cur_docs = [], [], []
            cur_spans, cur_tok2sent, cur_kept_idx = [], [], []

            # skip over any *consecutive* empty profiles
            if prof_idx < len(lengths):
                remaining = lengths[prof_idx]
                while prof_idx < len(lengths) and remaining == 0:
                    out_profiles.append(
                        PreprocessedProfile(
                            raw_posts=[],
                            post_sents_surface=[],
                            post_sents_lemmas=[],
                            post_doc_lemmas=[],
                            post_sent_char_spans=[],
                            post_token_to_sent=[],
                            post_sents_kept_idx=[],
                        )
                    )
                    prof_idx += 1
                    remaining = lengths[prof_idx] if prof_idx < len(lengths) else 0

    # trailing empty profiles
    while prof_idx < len(lengths):
        out_profiles.append(
            PreprocessedProfile(
                raw_posts=[],
                post_sents_surface=[],
                post_sents_lemmas=[],
                post_doc_lemmas=[],
                post_sent_char_spans=[],
                post_token_to_sent=[],
                post_sents_kept_idx=[],
            )
        )
        prof_idx += 1

    return out_profiles


def build_docs_from_preprocessed(
    pre_docs: List[Union[PreprocessedDoc, PreprocessedProfile]],
) -> Union[List[List[str]], List[List[List[str]]]]:
    """
    Backwards compatible:

    - If pre_docs is List[PreprocessedDoc]
        → return List[List[str]] (one lemma list per doc)
    - If pre_docs is List[PreprocessedProfile]
        → return List[List[List[str]]] (profiles × posts × lemmas)
    """
    if not pre_docs:
        return []
    first = pre_docs[0]
    if isinstance(first, PreprocessedDoc):
        return [P.doc_lemmas for P in pre_docs]  # type: ignore
    # profiles
    prof_out: List[List[List[str]]] = []
    for P in pre_docs:  # type: ignore
        prof_out.append([lemmas for lemmas in P.post_doc_lemmas])
    return prof_out
