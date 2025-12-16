# ssdiff/io_utils.py
from __future__ import annotations
import pickle, gzip, hashlib, time
from typing import Any, Dict, List, Optional, Union

# import your classes so isinstance checks work
from ssdiff.preprocess import PreprocessedDoc, PreprocessedProfile

def _hash_list_str(xs: List[str]) -> str:
    h = hashlib.sha1()
    for x in xs:
        h.update((x+'\n').encode('utf-8', 'ignore'))
    return h.hexdigest()

def save_preprocessed_bundle(
    pre_docs: List[Union[PreprocessedDoc, PreprocessedProfile]],
    path: str,
    *,
    authors: Optional[List[str]] = None,
    spaCy_model: Optional[str] = None,
    stopwords: Optional[List[str]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    compress: str = "gzip"  # "gzip" only here, simple and everywhere available
) -> str:
    """
    Save preprocessed objects + optional metadata in one compressed file.
    Returns the path actually written.
    """
    if not pre_docs:
        raise ValueError("pre_docs is empty")

    kind = "profile" if isinstance(pre_docs[0], PreprocessedProfile) else "doc"

    meta = {
        "schema": "ssdiff.preprocessed.v1",
        "kind": kind,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spaCy_model": spaCy_model,
        "authors_len": len(authors) if authors is not None else None,
        "stopwords_hash": _hash_list_str(stopwords) if stopwords else None,
        "extra": extra_meta or {},
    }

    # pickle the dataclass objects directly (keeps types intact)
    payload = {"meta": meta, "authors": authors, "pre_docs": pre_docs}

    if compress == "gzip":
        if not path.endswith(".pkl.gz"):
            path = path + ".pkl.gz"
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # raw pickle if you prefer (larger)
        if not path.endswith(".pkl"):
            path = path + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path

def load_preprocessed_bundle(path: str) -> Dict[str, Any]:
    """
    Load a bundle created by save_preprocessed_bundle.
    Returns dict with keys: 'meta', 'authors', 'pre_docs'
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        payload = pickle.load(f)

    # quick sanity
    meta = payload.get("meta", {})
    kind = meta.get("kind")
    pre_docs = payload.get("pre_docs", [])
    if not pre_docs:
        raise ValueError("Loaded bundle has no pre_docs")

    # Validate types (not strictly required but nice to catch mismatches)
    if kind == "profile" and not isinstance(pre_docs[0], PreprocessedProfile):
        raise TypeError("Bundle says 'profile' but data items are not PreprocessedProfile")
    if kind == "doc" and not isinstance(pre_docs[0], PreprocessedDoc):
        raise TypeError("Bundle says 'doc' but data items are not PreprocessedDoc")

    return payload
