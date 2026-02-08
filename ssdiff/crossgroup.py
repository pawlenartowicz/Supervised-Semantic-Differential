# ssdiff/crossgroup.py
from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from gensim.models import KeyedVectors

from .core import _SSDBase
from .utils import filtered_neighbors, load_embeddings


class SSDGroup(_SSDBase):
    """
    Cross-group SSD via centroid contrasts and permutation inference.

    Builds PCVs identically to SSD (shared _SSDBase pipeline), then:
    - computes group centroid PCVs
    - runs omnibus permutation test (mean pairwise cosine distance)
    - runs pairwise permutation tests
    - constructs centroid contrast vectors for interpretation

    Parameters
    ----------
    kv : KeyedVectors or str
        Pretrained embeddings (or path to load).
    docs : list
        Documents as token lists. Same format as SSD (flat or grouped/profile).
    groups : array-like
        Group label per document (same length as docs). Supports 2+ groups.
        Any hashable type (str, int). NaN/None entries drop that doc.
    lexicon : set or list
        Seed words for the concept.
    n_perm : int
        Number of permutations for inference (default 5000).
    random_state : int
        Seed for reproducibility of permutations (default 42).
    l2_normalize_docs, window, sif_a, use_full_doc :
        Same as SSD — passed through to _build_pcvs.
    """

    def __init__(
        self,
        kv: Union[KeyedVectors, str],
        docs: List[Any],
        groups: Sequence,
        lexicon: Any,
        *,
        n_perm: int = 5000,
        random_state: int = 42,
        l2_normalize_docs: bool = True,
        window: int = 3,
        sif_a: float = 1e-3,
        use_full_doc: bool = False,
    ) -> None:
        # --- Validate and align groups with docs ---
        groups_raw = np.asarray(groups, dtype=object)
        if len(groups_raw) != len(docs):
            raise ValueError(
                f"len(groups)={len(groups_raw)} != len(docs)={len(docs)}"
            )

        # Drop docs where group label is missing (None, NaN, empty string)
        group_valid = np.array([
            g is not None and g != "" and (not isinstance(g, float) or np.isfinite(g))
            for g in groups_raw
        ], dtype=bool)

        if not group_valid.all():
            docs = [d for d, v in zip(docs, group_valid) if v]
            groups_raw = groups_raw[group_valid]

        # --- Build PCVs (shared pipeline) ---
        # N_PCA is not used by SSDGroup (centroids/contrasts operate in
        # doc-vector space), but _build_pcvs expects it for the shared
        # StandardScaler + PCA step.  Keep a small default internally.
        self._build_pcvs(
            kv, docs, lexicon,
            l2_normalize_docs=l2_normalize_docs,
            N_PCA=20, window=window, sif_a=sif_a,
            use_full_doc=use_full_doc,
        )

        # --- Apply keep_mask to groups ---
        self.groups_kept = groups_raw[self.keep_mask]
        self.group_labels = sorted(set(self.groups_kept))
        self.G = len(self.group_labels)

        if self.G < 2:
            raise ValueError(
                f"Need at least 2 groups after filtering, got {self.G}: {self.group_labels}"
            )

        self.n_perm = int(n_perm)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        # --- Compute group centroids (unit-length, in doc-vector space) ---
        # Work in self.x space (L2-normed doc vectors, pre-PCA)
        self.centroids = self._compute_centroids(self.x, self.groups_kept)

        # --- Permutation tests ---
        if self.G == 2:
            # Omnibus is identical to the single pairwise test; skip redundant loop
            self.pairwise = self._pairwise_tests()
            only = list(self.pairwise.values())[0]
            self.omnibus_T = only["T"]
            self.omnibus_p = only["p_raw"]
            self.omnibus_null = only["null_dist"]
        else:
            self.omnibus_T, self.omnibus_p, self.omnibus_null = self._omnibus_permutation_test()
            self.pairwise = self._pairwise_tests()

    # ----------------------------------------------------------------
    # Centroids
    # ----------------------------------------------------------------
    def _compute_centroids(
        self, X: np.ndarray, groups: np.ndarray
    ) -> Dict[Any, np.ndarray]:
        """Compute unit-length centroid PCV for each group."""
        centroids = {}
        for g in self.group_labels:
            mask = groups == g
            c = X[mask].mean(axis=0)
            norm = float(np.linalg.norm(c))
            centroids[g] = c / max(norm, 1e-12)
        return centroids

    # ----------------------------------------------------------------
    # Test statistics
    # ----------------------------------------------------------------
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """1 - cos(a, b), where a and b are assumed unit-length."""
        return 1.0 - float(np.dot(a, b))

    def _compute_omnibus_T(self, X: np.ndarray, groups: np.ndarray) -> float:
        """Mean pairwise cosine distance between group centroids."""
        centroids = self._compute_centroids(X, groups)
        pairs = list(combinations(self.group_labels, 2))
        if not pairs:
            return 0.0
        total = sum(
            self._cosine_distance(centroids[a], centroids[b])
            for a, b in pairs
        )
        return total / len(pairs)

    def _compute_pairwise_T(
        self, X: np.ndarray, groups: np.ndarray, g1, g2
    ) -> float:
        """Cosine distance between two specific group centroids."""
        mask = (groups == g1) | (groups == g2)
        X_sub = X[mask]
        g_sub = groups[mask]
        c1 = X_sub[g_sub == g1].mean(axis=0)
        c1 /= max(float(np.linalg.norm(c1)), 1e-12)
        c2 = X_sub[g_sub == g2].mean(axis=0)
        c2 /= max(float(np.linalg.norm(c2)), 1e-12)
        return self._cosine_distance(c1, c2)

    # ----------------------------------------------------------------
    # Permutation tests
    # ----------------------------------------------------------------
    def _omnibus_permutation_test(self):
        """
        Permutation test on T = mean pairwise cosine distance.
        Permute group labels, recompute centroids and T.
        One-sided p = proportion of null >= observed.
        """
        T_obs = self._compute_omnibus_T(self.x, self.groups_kept)
        null_dist = np.empty(self.n_perm, dtype=np.float64)

        groups_perm = self.groups_kept.copy()
        for i in range(self.n_perm):
            self._rng.shuffle(groups_perm)
            null_dist[i] = self._compute_omnibus_T(self.x, groups_perm)

        p_value = float(np.mean(null_dist >= T_obs))
        return T_obs, p_value, null_dist

    def _pairwise_tests(self) -> Dict[Tuple, Dict]:
        """
        For each pair of groups, compute:
        - pairwise cosine distance T
        - permutation p-value (permuting labels within the pair only)
        - centroid contrast vector v_{A-B} (unit-length)
        - raw contrast vector (before normalization)
        - Cohen's d on projections onto contrast vector
        """
        results = {}
        pairs = list(combinations(self.group_labels, 2))
        n_pairs = len(pairs)

        for g1, g2 in pairs:
            mask = (self.groups_kept == g1) | (self.groups_kept == g2)
            X_pair = self.x[mask]
            g_pair = self.groups_kept[mask]

            T_obs = self._compute_pairwise_T(self.x, self.groups_kept, g1, g2)

            # Permutation within pair
            null_dist = np.empty(self.n_perm, dtype=np.float64)
            g_perm = g_pair.copy()
            for i in range(self.n_perm):
                self._rng.shuffle(g_perm)
                c1 = X_pair[g_perm == g1].mean(axis=0)
                c1 /= max(float(np.linalg.norm(c1)), 1e-12)
                c2 = X_pair[g_perm == g2].mean(axis=0)
                c2 /= max(float(np.linalg.norm(c2)), 1e-12)
                null_dist[i] = 1.0 - float(np.dot(c1, c2))

            p_raw = float(np.mean(null_dist >= T_obs))

            # Bonferroni correction
            p_corrected = min(p_raw * n_pairs, 1.0)

            # Contrast vector: c_g1 - c_g2
            contrast_raw = self.centroids[g1] - self.centroids[g2]
            contrast_norm = float(np.linalg.norm(contrast_raw))
            contrast_unit = contrast_raw / max(contrast_norm, 1e-12)

            # Cohen's d on projections
            proj = (self.x @ contrast_unit).ravel()
            proj_g1 = proj[self.groups_kept == g1]
            proj_g2 = proj[self.groups_kept == g2]
            pooled_std = np.sqrt(
                ((len(proj_g1)-1)*np.var(proj_g1, ddof=1) +
                 (len(proj_g2)-1)*np.var(proj_g2, ddof=1))
                / (len(proj_g1) + len(proj_g2) - 2)
            )
            cohens_d = (
                (np.mean(proj_g1) - np.mean(proj_g2)) / max(pooled_std, 1e-12)
            )

            results[(g1, g2)] = {
                "T": T_obs,
                "p_raw": p_raw,
                "p_corrected": p_corrected,
                "null_dist": null_dist,
                "contrast_raw": contrast_raw,
                "contrast_unit": contrast_unit,
                "contrast_norm": contrast_norm,
                "cohens_d": float(cohens_d),
                "n_g1": int(np.sum(self.groups_kept == g1)),
                "n_g2": int(np.sum(self.groups_kept == g2)),
            }

        return results

    # ----------------------------------------------------------------
    # Public API: summary
    # ----------------------------------------------------------------
    def print_results(self) -> None:
        """Pretty-print omnibus and pairwise results."""
        print(f"SSDGroup: {self.G} groups, {self.n_kept} participants "
              f"(dropped {self.n_dropped})")
        print(f"Groups: {', '.join(str(g) for g in self.group_labels)}")
        for g in self.group_labels:
            n_g = int(np.sum(self.groups_kept == g))
            print(f"  {g}: n={n_g}")
        print(f"\nOmnibus permutation test ({self.n_perm} permutations):")
        print(f"  T = {self.omnibus_T:.6f}")
        print(f"  p = {self.omnibus_p:.4f}")
        print(f"\nPairwise contrasts:")
        for (g1, g2), r in self.pairwise.items():
            print(f"\n  {g1} vs {g2}:")
            print(f"    cosine distance = {r['T']:.6f}")
            print(f"    p_raw = {r['p_raw']:.4f}, "
                  f"p_corrected = {r['p_corrected']:.4f}")
            print(f"    Cohen's d = {r['cohens_d']:.3f}")
            print(f"    ||contrast|| = {r['contrast_norm']:.4f}")

    def results_table(self) -> pd.DataFrame:
        """Return pairwise results as a DataFrame."""
        rows = []
        for (g1, g2), r in self.pairwise.items():
            rows.append({
                "group_A": g1,
                "group_B": g2,
                "n_A": r["n_g1"],
                "n_B": r["n_g2"],
                "cosine_distance": r["T"],
                "p_raw": r["p_raw"],
                "p_corrected": r["p_corrected"],
                "cohens_d": r["cohens_d"],
                "contrast_norm": r["contrast_norm"],
            })
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------
    # Public API: extract SSDContrast for interpretation
    # ----------------------------------------------------------------
    def get_contrast(self, group_a, group_b) -> "SSDContrast":
        """
        Return an SSDContrast for the pair (group_a, group_b).

        The contrast vector points from B toward A: words aligned with
        +contrast are more A-like, words aligned with -contrast are
        more B-like.

        If (group_a, group_b) was not in the original combinations
        ordering, the contrast is flipped automatically.

        The returned SSDContrast duck-types with SSD for use with:
        - cluster_top_neighbors(contrast, ...)
        - snippets_along_beta(pre_docs, ssd=contrast, ...)
        - contrast.nbrs(sign=+1)  →  words more A-like
        - contrast.nbrs(sign=-1)  →  words more B-like
        """
        key = (group_a, group_b)
        flipped = False
        if key not in self.pairwise:
            key = (group_b, group_a)
            flipped = True
            if key not in self.pairwise:
                raise KeyError(
                    f"Pair ({group_a}, {group_b}) not found. "
                    f"Available: {list(self.pairwise.keys())}"
                )

        r = self.pairwise[key]
        contrast_unit = -r["contrast_unit"] if flipped else r["contrast_unit"]
        contrast_raw = -r["contrast_raw"] if flipped else r["contrast_raw"]

        # Subset x and groups to just these two groups for scoring
        mask_pair = (self.groups_kept == group_a) | (self.groups_kept == group_b)

        return SSDContrast(
            kv=self.kv,
            beta=contrast_raw,
            beta_unit=contrast_unit,
            use_unit_beta=True,
            x=self.x,
            x_pair=self.x[mask_pair],
            groups_kept=self.groups_kept,
            groups_pair=self.groups_kept[mask_pair],
            group_a=group_a,
            group_b=group_b,
            lexicon=self.lexicon,
            window=self.window,
            sif_a=self.sif_a,
            docs_kept=self.docs_kept,
            perm_result=r,
            pca=self.pca,
            scaler_X=self.scaler_X,
        )

    # ----------------------------------------------------------------
    # Public API: per-participant scores along a contrast
    # ----------------------------------------------------------------
    def contrast_scores(
        self, group_a, group_b, *, return_df: bool = True
    ) -> Union[pd.DataFrame, Dict]:
        """
        Project all kept participants onto the (A-B) contrast vector.
        Returns cos_to_contrast and group membership.
        Useful for violin/density plots.
        """
        contrast = self.get_contrast(group_a, group_b)
        bu = contrast.beta_unit

        # Cosine of each kept doc to contrast direction
        x_unit = self._row_l2_normalize(self.x)
        cos_vals = (x_unit @ bu).ravel().astype(float)

        result = {
            "group": self.groups_kept.tolist(),
            "cos_to_contrast": cos_vals.tolist(),
        }

        if return_df:
            return pd.DataFrame(result)
        return result


class SSDContrast:
    """
    A pairwise centroid contrast that duck-types with a fitted SSD
    for downstream interpretation (neighbors, clustering, snippets).

    Exposes the same attributes that clusters.py and snippets.py read:
        .kv, .beta, .beta_unit, .use_unit_beta, .lexicon,
        .window, .sif_a, .x (doc vectors)

    Plus convenience methods that mirror SSD:
        .nbrs(), .top_words(), .cluster_neighbors(), etc.
    """

    def __init__(
        self,
        kv: KeyedVectors,
        beta: np.ndarray,
        beta_unit: np.ndarray,
        use_unit_beta: bool,
        x: np.ndarray,
        x_pair: np.ndarray,
        groups_kept: np.ndarray,
        groups_pair: np.ndarray,
        group_a,
        group_b,
        lexicon: set,
        window: int,
        sif_a: float,
        docs_kept: list,
        perm_result: dict,
        pca,
        scaler_X,
    ):
        self.kv = kv
        self.beta = beta
        self.beta_unit = beta_unit
        self.use_unit_beta = use_unit_beta
        self.x = x               # all participants (for snippet scoring)
        self.x_pair = x_pair     # only A+B participants
        self.groups_kept = groups_kept
        self.groups_pair = groups_pair
        self.group_a = group_a
        self.group_b = group_b
        self.lexicon = lexicon
        self.window = window
        self.sif_a = sif_a
        self.docs_kept = docs_kept
        self.perm_result = perm_result
        self.pca = pca
        self.scaler_X = scaler_X

        # Cluster storage (mirrors SSD)
        self.pos_clusters_raw = None
        self.neg_clusters_raw = None

    # --- Neighbors (mirrors SSD.nbrs) ---
    def nbrs(self, sign: int = +1, n: int = 16, restrict_vocab: int = 10000):
        """Nearest neighbors to +/-contrast direction.
        sign=+1 → more group_a-like, sign=-1 → more group_b-like.
        """
        b = self.beta_unit if self.use_unit_beta else self.beta
        vec = b if sign > 0 else -b
        out = []
        for w, sim in filtered_neighbors(self.kv, vec, topn=n, restrict=restrict_vocab):
            out.append((w, sim, float(self.kv[w].dot(vec))))
        return out

    # --- Top words table (mirrors SSD.top_words) ---
    def top_words(self, n: int = 10, *, verbose: bool = False) -> pd.DataFrame:
        """DataFrame of top-N neighbors on both poles of the contrast."""
        b = self.beta_unit if self.use_unit_beta else self.beta
        rows = []
        for side_label, group_label, vec in [
            ("pos", self.group_a, b),
            ("neg", self.group_b, -b),
        ]:
            pairs = filtered_neighbors(self.kv, vec, topn=n)
            for rank, (w, sim) in enumerate(pairs, start=1):
                rows.append({
                    "side": side_label,
                    "group": group_label,
                    "rank": rank,
                    "word": w,
                    "cos": float(sim),
                })
        df = pd.DataFrame(rows)

        if verbose:
            print(f"\n--- Words more {self.group_a}-like (+contrast) ---")
            for _, r in df[df["side"] == "pos"].iterrows():
                print(f"  {r['word']:<18} {r['cos']:.4f}")
            print(f"\n--- Words more {self.group_b}-like (-contrast) ---")
            for _, r in df[df["side"] == "neg"].iterrows():
                print(f"  {r['word']:<18} {r['cos']:.4f}")

        return df

    # --- Clustering (mirrors SSD.cluster_neighbors_sign) ---
    def cluster_neighbors_sign(
        self,
        *,
        side: str = "pos",
        topn: int = 100,
        k=None,
        k_min: int = 2,
        k_max: int = 10,
        restrict_vocab: int = 50000,
        random_state: int = 13,
        min_cluster_size: int = 2,
        top_words: int = 10,
        verbose: bool = False,
    ):
        """Cluster top neighbors. Delegates to clusters.cluster_top_neighbors."""
        from .clusters import cluster_top_neighbors

        clusters = cluster_top_neighbors(
            self, topn=topn, k=k, k_min=k_min, k_max=k_max,
            restrict_vocab=restrict_vocab, random_state=random_state,
            min_cluster_size=min_cluster_size, side=side,
        )

        if side == "pos":
            self.pos_clusters_raw = clusters
        else:
            self.neg_clusters_raw = clusters

        # Build DataFrames (same format as SSD.cluster_neighbors_sign)
        rows_summary, rows_members = [], []
        for rank, C in enumerate(clusters, start=1):
            top = [w for (w, _cc, _cb) in C["words"][:top_words]]
            side_label = f"{self.group_a}" if side == "pos" else f"{self.group_b}"
            rows_summary.append({
                "side": side,
                "group": side_label,
                "cluster_rank": rank,
                "size": C.get("size", len(C["words"])),
                "centroid_cos_beta": C.get("centroid_cos_beta", float("nan")),
                "coherence": C.get("coherence", float("nan")),
                "top_words": ", ".join(top),
            })
            for (w, ccent, cbeta) in C["words"]:
                rows_members.append({
                    "side": side,
                    "group": side_label,
                    "cluster_rank": rank,
                    "word": w,
                    "cos_to_centroid": float(ccent),
                    "cos_to_beta": float(cbeta),
                })

        df_clusters = pd.DataFrame(rows_summary)
        df_members = pd.DataFrame(rows_members)

        if verbose:
            pole = f"more {self.group_a}-like" if side == "pos" else f"more {self.group_b}-like"
            print(f"\nThemes: {pole}")
            print("-" * 40)
            for _, row in df_clusters.iterrows():
                print(f"\n  Cluster {int(row.cluster_rank)}")
                print(f"    size={int(row['size'])}  cos_b={row.centroid_cos_beta:.3f}  "
                      f"coherence={row.coherence:.3f}")
                print(f"    top: {row.top_words}")

        return df_clusters, df_members

    def cluster_neighbors(self, *, topn=100, k=None, k_min=2, k_max=10,
                           restrict_vocab=50000, random_state=13,
                           min_cluster_size=2, top_words=10, verbose=False):
        """Run clustering for both sides, return concatenated DFs."""
        df_pos_c, df_pos_m = self.cluster_neighbors_sign(
            side="pos", topn=topn, k=k, k_min=k_min, k_max=k_max,
            restrict_vocab=restrict_vocab, random_state=random_state,
            min_cluster_size=min_cluster_size, top_words=top_words,
            verbose=verbose,
        )
        df_neg_c, df_neg_m = self.cluster_neighbors_sign(
            side="neg", topn=topn, k=k, k_min=k_min, k_max=k_max,
            restrict_vocab=restrict_vocab, random_state=random_state,
            min_cluster_size=min_cluster_size, top_words=top_words,
            verbose=verbose,
        )
        return (
            pd.concat([df_pos_c, df_neg_c], ignore_index=True),
            pd.concat([df_pos_m, df_neg_m], ignore_index=True),
        )

    # --- Snippets (mirrors SSD.beta_snippets) ---
    def beta_snippets(self, *, pre_docs, seeds=None,
                       top_per_side=200, min_cosine=None):
        """Snippets along the contrast direction."""
        from .snippets import snippets_along_beta
        seeds = set(seeds or self.lexicon)
        return snippets_along_beta(
            pre_docs=pre_docs, ssd=self,
            token_window=self.window, seeds=seeds,
            sif_a=self.sif_a, top_per_side=top_per_side,
            min_cosine=min_cosine,
        )

    def cluster_snippets(self, *, pre_docs, side="both",
                          seeds=None, top_per_cluster=100):
        """Snippets per cluster centroid."""
        from .snippets import cluster_snippets_by_centroids
        need_pos = side in ("pos", "both")
        need_neg = side in ("neg", "both")
        pos_c = self.pos_clusters_raw if need_pos else []
        neg_c = self.neg_clusters_raw if need_neg else []
        if need_pos and not pos_c:
            raise RuntimeError("Run cluster_neighbors_sign(side='pos') first.")
        if need_neg and not neg_c:
            raise RuntimeError("Run cluster_neighbors_sign(side='neg') first.")
        seeds = set(seeds or self.lexicon)
        return cluster_snippets_by_centroids(
            pre_docs=pre_docs, ssd=self,
            pos_clusters=pos_c, neg_clusters=neg_c,
            token_window=self.window, seeds=seeds,
            sif_a=self.sif_a, top_per_cluster=top_per_cluster,
        )

    @staticmethod
    def _unit(v, eps=1e-12):
        n = float(np.linalg.norm(v))
        return v / max(n, eps)
