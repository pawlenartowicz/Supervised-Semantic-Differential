# ssdiff/pca_k_selector.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import SSD  # adjust if needed


# =========================
# Small math utilities
# =========================
def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


def _zscore_ignore_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    return (x - m) / s


def _rolling_smooth(x: np.ndarray, window: int = 7, kind: str = "median") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)

    if window <= 1:
        return x.copy()

    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = x[lo:hi]
        w = w[np.isfinite(w)]
        if len(w) == 0:
            continue
        out[i] = float(np.nanmedian(w) if kind == "median" else np.nanmean(w))
    return out


def _compute_auck(z: np.ndarray, radius: int) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    n = len(z)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        w = z[lo:hi]
        w = w[np.isfinite(w)]
        if len(w) == 0:
            continue
        out[i] = float(np.nanmean(w))
    return out


def _detrend_by_variance(var_explained_percent: np.ndarray, y: np.ndarray):
    """
    Fit y ~ a + b*log(var_explained_percent) and return y_hat, residuals, (a,b).
    """
    v = np.asarray(var_explained_percent, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(v) & np.isfinite(y) & (v > 0)
    v2 = v[mask]
    y2 = y[mask]

    if len(v2) < 3:
        y_hat = np.full_like(y, np.nan, dtype=float)
        resid = np.full_like(y, np.nan, dtype=float)
        return y_hat, resid, (float("nan"), float("nan"))

    X = np.column_stack([np.ones_like(v2), np.log(v2)])
    coef, _, _, _ = np.linalg.lstsq(X, y2, rcond=None)
    a, b = coef

    y_hat = np.full_like(y, np.nan, dtype=float)
    y_hat[mask] = a + b * np.log(v2)
    resid = y - y_hat
    return y_hat, resid, (float(a), float(b))


# =========================
# Interpretability aggregate
# =========================
def _overall_interpretability(df_clusters: pd.DataFrame, weight_by_size: bool = True) -> dict:
    if df_clusters is None or len(df_clusters) == 0:
        return dict(
            mean_coherence=np.nan,
            mean_abs_cosb=np.nan,
            aggregate=np.nan,
            n_clusters=0,
            total_size=0,
        )

    need_cols = {"size", "coherence", "centroid_cos_beta"}
    missing = need_cols - set(df_clusters.columns)
    if missing:
        raise RuntimeError(f"df_clusters missing columns: {missing}")

    d = df_clusters.copy()
    coherence = d["coherence"].to_numpy(dtype=float)
    abs_cosb = np.abs(d["centroid_cos_beta"].to_numpy(dtype=float))

    if weight_by_size:
        w = d["size"].to_numpy(dtype=float)
        wsum = np.nansum(w)
        if wsum > 0:
            w = w / wsum
            mean_coh = float(np.nansum(coherence * w))
            mean_abs = float(np.nansum(abs_cosb * w))
        else:
            mean_coh = float(np.nanmean(coherence))
            mean_abs = float(np.nanmean(abs_cosb))
    else:
        mean_coh = float(np.nanmean(coherence))
        mean_abs = float(np.nanmean(abs_cosb))

    return dict(
        mean_coherence=mean_coh,
        mean_abs_cosb=mean_abs,
        aggregate=float(mean_coh * mean_abs),
        n_clusters=int(len(d)),
        total_size=int(np.nansum(d["size"].to_numpy(dtype=float))),
    )


# =========================
# One-pass sweep + selection
# =========================
@dataclass(frozen=True)
class PCAKSelectionResult:
    best_k: int
    df_joined: pd.DataFrame  # includes raw + detrended + AUCK + joint_score


def pca_sweep(
    *,
    kv,
    docs,
    y: np.ndarray,
    lexicon=None,
    use_full_doc: bool = True,
    pca_k_values: Optional[Sequence[int]] = None,
    sif_a: float = 1e-3,
    window: Optional[int] = None,
    cluster_topn: int = 100,
    k_min: int = 2,
    k_max: int = 5,
    top_words: int = 20,
    weight_by_size: bool = True,
    auck_radius: int = 3,
    beta_smooth_win: int = 7,
    beta_smooth_kind: str = "median",
    out_dir: Optional[str] = None,
    prefix: str = "pca_k",
    save_tables: bool = False,
    save_figures: bool = False,
    verbose: bool = True,
) -> PCAKSelectionResult:
    """
    SINGLE-PASS sweep over PCA_K.

    Metrics per K:
      - interpretability aggregate (cluster-based)
      - beta_delta_1_minus_cos between consecutive beta_unit (RAW)
      - var_explained

    Scoring:
      - interpretability: detrend aggregate by log(var_explained) -> z -> AUCK
      - stability: RAW, use stab_good_raw = -beta_delta -> z -> AUCK
      - joint_score = mean(interp_auck, stab_auck_raw)
      - pick smallest K among ties
    """
    if pca_k_values is None:
        pca_k_values = list(range(20, 121, 2))

    if (save_tables or save_figures) and out_dir is None:
        raise ValueError("If save_tables or save_figures is True, out_dir must be provided.")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    beta_prev = None

    for K in pca_k_values:
        if verbose:
            print(f"  [pca_sweep] PCA_K={K}")

        try:
            ssd_kwargs = dict(
                lexicon=lexicon,
                use_full_doc=use_full_doc,
                N_PCA=int(K),
                SIF_a=sif_a,
            )
            if window is not None:
                ssd_kwargs["window"] = window

            ssd = SSD(kv, docs, y, **ssd_kwargs)

            # var explained
            var_expl = float(ssd.pca.explained_variance_ratio_.sum() * 100)

            # beta delta
            beta = getattr(ssd, "beta_unit", None)
            if beta is None:
                raise RuntimeError("ssd.beta_unit missing")
            beta = np.asarray(beta, dtype=float).ravel()

            if beta_prev is None:
                beta_delta = np.nan
            else:
                beta_delta = 1.0 - _cosine(beta_prev, beta)

            beta_prev = beta

            # clustering -> interpretability
            df_clusters, _ = ssd.cluster_neighbors(
                topn=cluster_topn,
                k=None,
                k_min=k_min,
                k_max=k_max,
                verbose=False,
                top_words=top_words,
            )
            overall = _overall_interpretability(df_clusters, weight_by_size=weight_by_size)

            rows.append(
                dict(
                    PCA_K=int(K),
                    var_explained=var_expl,
                    mean_coherence=overall["mean_coherence"],
                    mean_abs_cosb=overall["mean_abs_cosb"],
                    aggregate=overall["aggregate"],
                    n_clusters=overall["n_clusters"],
                    total_size=overall["total_size"],
                    beta_delta_1_minus_cos=float(beta_delta) if np.isfinite(beta_delta) else np.nan,
                )
            )

        except Exception as e:
            if verbose:
                print(f"    [skip] PCA_K={K} failed: {type(e).__name__}: {e}")
            rows.append(
                dict(
                    PCA_K=int(K),
                    var_explained=np.nan,
                    mean_coherence=np.nan,
                    mean_abs_cosb=np.nan,
                    aggregate=np.nan,
                    n_clusters=0,
                    total_size=0,
                    beta_delta_1_minus_cos=np.nan,
                )
            )
            beta_prev = None  # break beta chain after failure for safety

    df = pd.DataFrame(rows).sort_values("PCA_K").reset_index(drop=True)

    # --- Interpretability detrend + AUCK ---
    interp_hat, interp_resid, _ = _detrend_by_variance(
        df["var_explained"].to_numpy(dtype=float),
        df["aggregate"].to_numpy(dtype=float),
    )
    interp_z = _zscore_ignore_nan(interp_resid)
    interp_auck = _compute_auck(interp_z, radius=auck_radius)

    df["interp_hat"] = interp_hat
    df["interp_resid"] = interp_resid
    df["interp_resid_z"] = interp_z
    df["interp_auck"] = interp_auck

    # --- Stability RAW (no detrend) + AUCK ---
    delta = df["beta_delta_1_minus_cos"].to_numpy(dtype=float)
    stab_good_raw = -delta  # smaller delta => better stability
    stab_z_raw = _zscore_ignore_nan(stab_good_raw)
    stab_auck_raw = _compute_auck(stab_z_raw, radius=auck_radius)

    df["stab_good_raw"] = stab_good_raw
    df["stab_z_raw"] = stab_z_raw
    df["stab_auck_raw"] = stab_auck_raw

    # --- Joint score + choose best K ---
    df["joint_score"] = 0.5 * (df["interp_auck"] + df["stab_auck_raw"])

    mask = np.isfinite(df["joint_score"].to_numpy(dtype=float))
    if not mask.any():
        raise RuntimeError("No finite joint_score values; cannot select best PCA_K.")

    joint_vals = df.loc[mask, "joint_score"].to_numpy(dtype=float)
    ks = df.loc[mask, "PCA_K"].to_numpy(dtype=int)

    best_val = float(np.nanmax(joint_vals))
    tied = ks[np.isclose(joint_vals, best_val, rtol=0, atol=1e-12)]
    best_k = int(np.min(tied))

    if verbose:
        print("\n=== BEST PCA_K (JOINT AUCK, single-pass) ===")
        print(f"PCA_K        : {best_k}")
        print(f"joint_score  : {best_val:.6f}")
        row = df[df["PCA_K"] == best_k].iloc[0]
        print(f"interp_auck  : {row['interp_auck']:.6f}")
        print(f"stab_auck_raw: {row['stab_auck_raw']:.6f}")

    # --- Optional outputs ---
    if save_tables and out_dir is not None:
        out_xlsx = os.path.join(out_dir, f"{prefix}_pca_k_joint_auck_table.xlsx")
        df.to_excel(out_xlsx, index=False)
        if verbose:
            print(f"Saved table → {out_xlsx}")

    if save_figures and out_dir is not None:
        x = df["PCA_K"].to_numpy(dtype=int)
        y_left = df["interp_resid_z"].to_numpy(dtype=float)

        beta_delta = df["beta_delta_1_minus_cos"].to_numpy(dtype=float)
        beta_delta_s = _rolling_smooth(beta_delta, window=beta_smooth_win, kind=beta_smooth_kind)

        c_left = "tab:blue"
        c_right = "tab:orange"

        fig, ax1 = plt.subplots()
        ax1.plot(x, y_left, marker="o", color=c_left, label="detrended interpretability (z)")
        ax1.axhline(0.0, linewidth=1, color="0.6")
        ax1.set_xlabel("PCA_K")
        ax1.set_ylabel("Detrended interpretability (z)", color=c_left)
        ax1.tick_params(axis="y", labelcolor=c_left)

        ax2 = ax1.twinx()
        ax2.plot(
            x, beta_delta_s,
            linewidth=2,
            color=c_right,
            label=f"beta_unit change (smoothed 1-cos, {beta_smooth_kind}, w={beta_smooth_win})",
        )
        ax2.set_ylabel("Beta_unit change (smoothed 1 - cosine)", color=c_right)
        ax2.tick_params(axis="y", labelcolor=c_right)

        ax1.axvline(best_k, color="red", linewidth=2, label=f"best_K={best_k}")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title(f"{prefix}: detrended interpretability + beta stability (RAW, joint AUCK)")
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{prefix}_pca_k_joint_auck_ONEPLOT.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        if verbose:
            print(f"Saved figure → {out_png}")

    return PCAKSelectionResult(best_k=best_k, df_joined=df)
