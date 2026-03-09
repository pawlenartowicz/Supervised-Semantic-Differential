# tests/test_basic_pipeline.py


from ssdiff import SSD
from ssdiff.embeddings import Embeddings
import numpy as np


def _tiny_kv():
    dim = 5
    words = ["kraj", "naród", "państwo", "piękny", "silny", "zły", "dobry"]
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(len(words), dim)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return Embeddings(words, mat)


def test_ssd_smoke():
    kv = _tiny_kv()
    docs = [
        ["kraj", "piękny"],
        ["naród", "silny"],
        ["państwo", "dobry"],
        ["kraj", "zły"],
    ]
    y = np.array([1.0, 1.2, 1.1, 0.8])
    lex = {"kraj", "naród", "państwo"}
    ssd = SSD(
        kv=kv,
        docs=docs,
        y=y,
        lexicon=lex,
        l2_normalize_docs=True,
        use_unit_beta=True,
        N_PCA=3,
    )
    # basic attributes exist
    assert hasattr(ssd, "beta")
    assert hasattr(ssd, "beta_unit")
    assert hasattr(ssd, "r2")
    # per-essay scores compute
    scores = ssd.ssd_scores()
    assert len(scores) == len(docs)
