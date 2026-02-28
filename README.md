# Supervised Semantic Differential (SSD)

[![PyPI version](https://img.shields.io/pypi/v/ssdiff.svg)](https://pypi.org/project/ssdiff/)


**SSD**  lets you recover **interpretable semantic directions** related to specific concepts directly from open-ended text and relate them to **numeric outcomes**
(e.g., psychometric scales, judgments). It builds per-essay concept vectors from **local contexts around seed words**,
learns a **semantic gradient (β̂)** that best predicts the outcome, and then provides multiple interpretability layers:

- **Nearest neighbors** of each pole (+β̂ / −β̂)
- **Clustering** of neighbors into themes
- **Text snippets**: top sentences whose local contexts align with each cluster centroid or the β̂ axis
- **Per-essay scores** (cosine alignments) for further analysis

**SSDGroup** extends SSD to **cross-group comparisons** (e.g., clinical vs. control, different nationalities). Instead of regressing onto a continuous outcome, it computes **group centroid contrasts** in the same concept-vector space and uses **permutation inference** to test whether groups differ in how they represent the concept. The resulting contrast vectors plug directly into the same interpretation pipeline (neighbors, clusters, snippets).

The goal of the package is to allow psycholinguistic researchers to draw data-driven insights about
how people use language depending on their attitudes, traits, or other numeric variables of interest.

The method has been presented in the following preprint:
https://doi.org/10.31234/osf.io/gvrsb_v1

> **No-code option:** a GUI desktop application for SSD is available at [hplisiecki/SSD_APP](https://github.com/hplisiecki/SSD_APP). It wraps this package into a point-and-click interface with a guided three-stage workflow, interactive lexicon builder, and APA-formatted export — pre-built binaries for Windows, Linux, and macOS are available with no Python installation required.

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Core Concepts](#core-concepts)
- [Preprocessing (spaCy)](#preprocessing-spacy)
- [Lexicon Utilities](#lexicon-utilities)
- [Choosing PCA dimensionality (PCA Sweep)](#choosing-pca-dimensionality-pca-sweep)
- [Fitting SSD](#fitting-ssd)
- [Neighbors & Clustering](#neighbors--clustering)
- [Interpreting with Snippets](#interpreting-with-snippets)
- [Per-Essay SSD Scores](#per-essay-ssd-scores)
- [Cross-Group Comparison (SSDGroup)](#cross-group-comparison-ssdgroup)
- [API Summary](#api-summary)
- [Citing & License](#citing--license)

---

## Installation

```bash
pip install ssdiff
```

Dependencies (installed automatically): `numpy`, `pandas`, `scikit-learn`, `gensim`, `spacy`.


---

## Quickstart

Below is an end-to-end minimal example using the Polish model and an example dataset. 
Adjust paths and column names to your data.

```python
from ssdiff import (
    SSD, load_embeddings, normalize_kv,
    load_spacy, load_stopwords, preprocess_texts, build_docs_from_preprocessed,
    suggest_lexicon, token_presence_stats, coverage_by_lexicon, pca_sweep,
)

import pandas as pd

MODEL_PATH = r"NLPResources\word2vec_model.kv"
DATA_PATH  = r"data\example_dataset.csv"

# 1) Load and normalize embeddings (L2 + ABTT on word space)
kv = normalize_kv(load_embeddings(MODEL_PATH), l2=True, abtt_m=1)

# 2) Load your data
df = pd.read_csv(DATA_PATH)
text_raw_col = "text_raw" # column with raw text
y_col        = "questionnaire_result" # numeric outcome column

# 3) Preprocess (spaCy) — keep original sentences and lemmas linked
nlp = load_spacy("pl_core_news_lg") # polish spacy model
stopwords = load_stopwords("pl") # polish stopwords
texts_raw = df[text_raw_col].fillna("").astype(str).tolist()
pre_docs = preprocess_texts(texts_raw, nlp, stopwords)

# 4) Build lemma docs for modeling and filter to non-NaN y
docs = build_docs_from_preprocessed(pre_docs)       # list[list[str]]
y = pd.to_numeric(df[y_col], errors="coerce")
mask = ~y.isna()
docs = [docs[i] for i in range(len(docs)) if mask.iat[i]]
pre_docs = [pre_docs[i] for i in range(len(pre_docs)) if mask.iat[i]]
y = y[mask].to_numpy()

# 5) Define a lexicon (tokens must match your preprocessing) Check lexicon utilities section for data-driven selection.
lexicon = {"concept_keyword_1", "concept_keyword_2", "concept_keyword_3", "concept_keyword_4"} # keywords that define your concept

# 6) Choose PCA dimensionality based on interpretability + stability
# This method was not a part of the original SSD paper. It was developed later.

sel = pca_sweep(
    kv=kv,
    docs=docs,
    y=y,
    lexicon={"concept_keyword_1", "concept_keyword_2", "concept_keyword_3"},
    use_full_doc=False,
    pca_k_values=list(range(1, 121, 2)),
    window=3,      # context window ±3 tokens (same meaning as in SSD)
    SIF_a=1e-3,
    prefix="concept",
)

PCA_K = sel.best_k

# 7) Fit SSD
ssd = SSD(
    kv=kv,
    docs=docs,
    y=y,
    lexicon=lexicon,
    l2_normalize_docs=True, # normalize per-essay vectors
    N_PCA=PCA_K,
    use_unit_beta=True, # unit β̂ for neighbors/interpretation
    window = 3, # context window ±3 tokens
    SIF_a = 1e-3, # SIF weighting parameter
)

# 8) Inspect regression readout
print({
    "R2": ssd.r2,
    "adj_R2": float(getattr(ssd, "r2_adj", float("nan"))),
    "F": ssd.f_stat,
    "p": ssd.f_pvalue,
    "beta_norm": ssd.beta_norm_stdCN,        # ||β|| in SD(y) per +1.0 cosine
    "delta_per_0.10_raw": ssd.delta_per_0p10_raw,
    "IQR_effect_raw": ssd.iqr_effect_raw,
    "corr_y_pred": ssd.y_corr_pred,
    "n_raw": int(getattr(ssd, "n_raw", len(docs))),
    "n_kept": int(getattr(ssd, "n_kept", len(docs))),
    "n_dropped": int(getattr(ssd, "n_dropped", 0)),    
})

# 9) Neighbors
ssd.top_words(n = 20, verbose = True)

# 10) Cluster themes (e.g., Clustering by silhouette)
df_clusters, df_members = ssd.cluster_neighbors(topn = 100, k=None, k_min = 2, k_max = 10, verbose = True, top_words = 5)

# 11) Snippets for interpretation
snips = ssd.cluster_snippets(pre_docs=pre_docs, side="both", window_sentences=1, top_per_cluster=100)
df_pos_snip = snips["pos"]
df_neg_snip = snips["neg"]

beta_snips = ssd.beta_snippets(pre_docs=pre_docs, window_sentences=1, top_per_side=200)
df_beta_pos = beta_snips["beta_pos"]
df_beta_neg = beta_snips["beta_neg"]

# 12) Per-essay SSD scores
scores = ssd.ssd_scores(docs, include_all=True)

```
---

## Core Concepts

- **Seed lexicon**: a small set of tokens (lemmas) indicating the concept of interest (e.g., {klimat, klimatyczny, zmiana}).
- **Per-essay vector**: SIF-weighted average of context vectors around each seed occurrence (±3 tokens), then averaged across occurrences.
- **SSD fitting**: PCA on standardized doc vectors, OLS from components to standardized outcome 𝑦, then back-project to doc space to get β (the semantic gradient).
- **Interpretation**: nearest neighbors to +β̂/−β̂, clustering neighbors into themes, and showing original sentences whose local context aligns with centroids or β̂.




---
## Word2Vec Embeddings

The method requires pre-trained word embeddings in either the .kv, .bin, .txt, or a .gz compression of the previous formats.
In order to capture only the semantic information, without frequency-based artifacts, it is recommended to apply L2 normalization 
and All-But-The-Top (ABTT) transformation to the embeddings before fitting SSD.

```python
from ssdiff import load_embeddings, normalize_kv

MODEL_PATH = r"NLPResources\word2vec_model.kv"

kv = load_embeddings(MODEL_PATH) # load embeddings

kv = normalize_kv(kv, l2=True, abtt_m=1)  # L2 + ABTT (remove top-1 PC)
```

The model is not included in the package, and will differ depending on your language and domain.
Look for pre-trained embeddings in your language, the more data they were trained on, the better.
Pay attention to the vocabulary coverage of your texts.

For polish, the nkjp+wiki-lemmas-all-300-cbow-hs.txt.gz (no. 25) from the [Polish Word2Vec model list](https://dsmodels.nlp.ipipan.waw.pl) was found to work well.

---
## Preprocessing (spaCy)

SSD uses spaCy to keep original sentences and lemmas aligned for later snippet extraction.

```python
from ssdiff import load_spacy, load_stopwords, preprocess_texts, build_docs_from_preprocessed

nlp = load_spacy("pl_core_news_lg")   # or another language model
stopwords = load_stopwords("pl")      # same stopword source across app & package

pre_docs = preprocess_texts(texts_raw, nlp, stopwords)
docs = build_docs_from_preprocessed(pre_docs)  # → list[list[str]] (lemmas without stopwords/punct)
```

Each PreprocessedDoc stores:

- **raw**: original raw text
- **sents_surface**: list[str], original sentences
- **sents_lemmas**: list[list[str]]
- **doc_lemmas**: flattened lemmas (list[str])
- **sent_char_spans**: list of (start_char, end_char) per sentence
- **token_to_sent**: index mapping lemma positions → sentence index

Spacy models for various languages can be found [here](https://spacy.io/models).

To install a spaCy model, run e.g.:
```bash
python -m spacy download pl_core_news_lg
```

---
## Lexicon Utilities

These helpers make lexicon selection transparent and data-driven (you can also hand-pick tokens).

### `suggest_lexicon(...)`

Rank tokens by balanced coverage with a mild penalty for strong association with the outcome.

All three lexicon utilities accept `var_type='continuous'` (default) or `var_type='categorical'`:

| | `var_type='continuous'` | `var_type='categorical'` |
|---|---|---|
| `cov_bal` | average presence across 𝑛 quantile bins of 𝑦 | average presence across group labels |
| `corr` | Pearson correlation between 0/1 presence and standardized 𝑦 | Cramér's V between 0/1 presence and group label |
| `q1` / `q4` | coverage in lowest / highest 𝑦 quantile bin | min / max group coverage |
| `rank` | `cov_bal * (1 - min(1, \|corr\|/corr_cap))` | same formula (Cramér's V replaces Pearson) |

Accepts a DataFrame (`text_col`, `score_col`) or a `(texts, y)` tuple where texts can be raw strings or token lists.

```python
from ssdiff import suggest_lexicon

# Continuous outcome (default)
cands_df = suggest_lexicon(df, text_col="lemmatized", score_col="questionnaire_result", top_k=150)

# Or using a tuple (texts, y)
texts = [" ".join(doc) for doc in docs]
cands_df2 = suggest_lexicon((docs, y), top_k=150)

# Categorical groups
cands_cat = suggest_lexicon(df, text_col="lemmatized", score_col="diagnosis", top_k=150, var_type="categorical")
cands_cat2 = suggest_lexicon((docs, groups), top_k=150, var_type="categorical")
```
### `token_presence_stats(...)`

Per-token coverage & association diagnostics:
```python
from ssdiff import token_presence_stats

# Continuous
stats = token_presence_stats(texts, y, token="concept_keyword_1", n_bins=4, verbose=True)
print(stats)  # dict: token, docs, cov_all, cov_bal, corr, rank, q1, q4

# Categorical — output also includes group_cov (per-group coverage dict)
stats = token_presence_stats(texts, groups, token="concept_keyword_1", var_type="categorical", verbose=True)
print(stats["group_cov"])  # e.g. {"control": 0.45, "depression": 0.62}
```

### `coverage_by_lexicon(...)`

Summary for your chosen lexicon:
- `summary` : `docs_any`, `cov_all`, `q1`, `q4`, `corr_any`, `hits_mean`, `hits_median`, `types_mean`, `types_median`
  - `q1` / `q4`: coverage within the lowest/highest 𝑦 bins (continuous) or min/max group coverage (categorical)
  - when `var_type='categorical'`, summary also includes `group_cov` (per-group coverage dict)
- `per_token_df`: per-token stats

```python
from ssdiff import coverage_by_lexicon

# Continuous
summary, per_tok = coverage_by_lexicon(
    (texts, y),
    lexicon={"concept_keyword_1", "concept_keyword_2", "concept_keyword_3", "concept_keyword_4"},
    n_bins=4,
    verbose=True,
)

# Categorical
summary, per_tok = coverage_by_lexicon(
    (texts, groups),
    lexicon={"concept_keyword_1", "concept_keyword_2"},
    var_type="categorical",
    verbose=True,
)
print(summary["group_cov"])  # e.g. {"control": 0.80, "depression": 0.75}
```

---


## Fitting SSD

Instantiate `SSD` with normalized embeddings, tokenized documents, a numeric outcome, and a lexicon
(defining the concept of interest):

```python
from ssdiff import SSD, load_embeddings, normalize_kv

kv = normalize_kv(load_embeddings(MODEL_PATH), l2=True, abtt_m=1)

PCA_K = min(20, max(3, n_docs // 20))  # simple heuristic, or use pca_sweep(...)
ssd = SSD(
    kv=kv,
    docs=docs,
    y=y,
    lexicon={"concept_keyword_1", "concept_keyword_2", "concept_keyword_3"},
    N_PCA=PCA_K,
    window=3,       # context window ±3 tokens around lexicon hits
    SIF_a=1e-3,      # SIF weighting parameter
    use_full_doc=False,
    use_unit_beta=True,
)

print(ssd.r2, ssd.f_stat, ssd.f_pvalue)
```

Key outputs attached to the instance:
- `beta` / `beta_unit` — semantic gradient (doc space)
- `r2`, `f_stat`, `f_pvalue`, 'r2_adj' — regression fit stats
- `beta_norm_stdCN` — ||β|| in SD(y) per +1.0 cosine
- `delta_per_0p10_raw` — change in raw 𝑦 per +0.10 cosine
- `iqr_effect_raw` — IQR(of cosine)*slope in raw 𝑦
- `y_corr_pred` — correlation of standardized 𝑦 with predicted values


--- 
## Choosing PCA dimensionality (PCA Sweep)

The original SSD pipeline applies PCA to document vectors before regression to reduce redundancy and enable fitting on small corpora.
However, selecting the number of components (`N_PCA = K`) can otherwise become a researcher degree of freedom.

To make this choice more systematic and transparent, `ssdiff` includes a **PCA sweep procedure** that evaluates a sequence of `K` values and selects the most robust solution.

### What PCA Sweep optimizes

For each candidate PCA dimensionality `K`, the sweep fits SSD and tracks:

1) **Interpretability quality**
   - based on clustering the nearest neighbors at each pole of the semantic gradient (β̂)
   - aggregate interpretability combines:
     - cluster coherence (how semantically tight clusters are)
     - alignment with the semantic gradient (|cos(centroid, β̂)|)

2) **Stability of the semantic gradient**
   - measured as the cosine change between consecutive gradients:
     - `beta_delta = 1 - cos(beta_unit(K-Δ), beta_unit(K))`
   - smaller values mean **more stable** gradients as `K` increases

These signals are smoothed across nearby K values using an **AUCK window**:
for `auck_radius=r`, the sweep averages across a sliding window of `2r+1` values (edge-safe).

The sweep returns the selected `best_k`, and also provides per-K tables that can be saved for transparency.


### Minimal example (PCA Sweep → final SSD)

```python
from ssdiff import SSD, load_embeddings, normalize_kv, pca_sweep

kv = normalize_kv(load_embeddings(MODEL_PATH), l2=True, abtt_m=1)

# Pick PCA_K automatically with a sweep (robust: interpretability + beta stability)
sel = pca_sweep(
    kv=kv,
    docs=docs,
    y=y,
    lexicon={"concept_keyword_1", "concept_keyword_2", "concept_keyword_3"},
    use_full_doc=False,
    pca_k_values=list(range(1, 121, 2)),
    window=3,      # context window ±3 tokens (same meaning as in SSD)
    SIF_a=1e-3,
    save_figures=True,
    out_dir=RESULTS_DIR,
    prefix="climate",
)

PCA_K = sel.best_k

ssd = SSD(
    kv=kv,
    docs=docs,
    y=y,
    lexicon={"concept_keyword_1", "concept_keyword_2", "concept_keyword_3"},
    N_PCA=PCA_K,
    use_unit_beta=True,
    windpow=3,
    SIF_a=1e-3,
)

print("PCA_K:", PCA_K)
print(ssd.r2, ssd.f_stat, ssd.f_pvalue)
```



### Sweep outputs

If `save_tables=True`, PCA Sweep saves:

- `{prefix}_pca_k_joint_auck_table.xlsx`

If `save_figures=True`, PCA Sweep saves:

- `{prefix}_sweep_plot.png`

These files document the sweep for transparency and reproducibility.


### PCA sweep plot example

<img src="https://github.com/hplisiecki/Supervised-Semantic-Differential/blob/main/images/sweep_plot.png" width="500">

**Figure. PCA sweep for SSD.**  
The **blue curve** shows **detrended interpretability** of the SSD solution as a function of the PCA dimensionality **K**.  
For each **K**, SSD clusters the nearest neighbors of the learned semantic gradient and computes an interpretability score:

**aggregate(K) = weighted_mean(coherence) × weighted_mean(|cos(beta_hat, centroid)|)**,

where both terms are **weighted by cluster size**. The aggregate score is then **detrended** by regressing it on
**log(% variance explained)** and plotting the resulting residuals (z-scored), so the blue curve reflects interpretability
**beyond what is trivially expected from retaining more variance at larger K**.

The **orange curve** shows **solution stability**, measured as the change of the unit semantic gradient between consecutive
K values: **delta_beta(K) = 1 − cos(beta_hat(K−Δ), beta_hat(K))** (smoothed for readability).
Lower values indicate that increasing K does not substantially rotate the inferred semantic direction.

The **red vertical line** marks the selected **K**, chosen by maximizing a **robust joint score** that averages
(1) local AUCK-smoothed detrended interpretability and (2) local AUCK-smoothed stability (favoring small delta_beta),
with ties resolved by selecting the **smallest** K on a plateau.

---
## Neighbors & Clustering

### Nearest neighbors

Get the top N nearest neighbors of +β̂/−β̂:

```python
top_words = ssd.top_words(n = 20, verbose = True)
```

### Clustering neighbors into themes
Use `cluster_neighbors_sign` to group the top N neighbors of +β̂/−β̂ into k clusters (k-means; Euclidean on unit vectors ≈ cosine):

```python
df_clusters, df_members = ssd.cluster_neighbors(topn = 100, 
                                                k=None,
                                                k_min = 2, 
                                                k_max = 10, 
                                                verbose = True,
                                                random_state = 13, # for reproducibility
                                                top_words = 5,
                                                verbose = True)
```

Returns
- df_clusters (one row per cluster):
- side, cluster_rank, size, centroid_cos_beta, coherence, top_words
- df_members (one row per word):
  side, cluster_rank, word, cos_to_centroid, cos_to_beta

The raw clusters (with all per-word cosines and internal ids) are kept internally as:
- ssd.pos_clusters_raw
- ssd.neg_clusters_raw

---
## Interpreting with Snippets
After clustering, SSD lets you **link the abstract directions in embedding space back to actual language** by inspecting **text snippets**.  
 The script:
1. Locates each **occurrence of a seed word** (from your lexicon) in the corpus.  
2. Extracts a **small window of surrounding context** (±3 tokens).  
3. Represents that window as a **SIF-weighted context vector** in the same embedding space as β̂ and the cluster centroids.  
4. Computes the **cosine similarity** between each such local context vector and  
   - a **cluster centroid** (to find passages representative of that theme), or  
   - the overall **semantic gradient β̂** (to find passages aligned with the global direction).

### Snippets by cluster centroids

```python
snips = ssd.cluster_snippets(
    pre_docs=pre_docs,    # from preprocess_texts(...)
    side="both",          # "pos", "neg", or "both"
    window_sentences=1,   # [sent-1, sent, sent+1]
    top_per_cluster=100,  # keep best K per cluster
)

df_pos_snip = snips["pos"]  # columns: centroid_label, doc_id, cosine, seed, sentence_before, sentence_anchor, sentence_after, window_text_surface, ...
df_neg_snip = snips["neg"]



df_pos_snip = snips["pos"] 
df_neg_snip = snips["neg"]
```
Each returned row represents a seed occurrence window, not a whole essay.  
The `cosine` column is the similarity between the context vector (built around that seed occurrence) and the cluster centroid.  
Surface text (`sentence_before`, `sentence_anchor`, `sentence_after`) lets you read the passage in context.

### Snippets along β̂
You can also extract windows that best illustrate the main semantic direction (rather than specific clusters):
```python

beta_snips = ssd.beta_snippets(
    pre_docs=pre_docs,
    window_sentences=1,
    seeds=ssd.lexicon,
    top_per_side=200,
)

df_beta_pos = beta_snips["beta_pos"]
df_beta_neg = beta_snips["beta_neg"]
```
Here, the cosine is taken between each seed-centered context vector and β̂ (the main semantic gradient).
Sorting by this cosine reveals which local language usages most strongly express the positive or negative pole of your concept.

---
## Per-Essay SSD Scores

The **SSD score** for each essay quantifies **how closely the text’s meaning aligns with the main semantic direction (β̂)** discovered by the model.  
These scores can be used for individual-difference analyses, correlations with psychological scales, or visualization of semantic alignment across groups.

Internally, each essay is represented by a **SIF-weighted average of local context vectors** (around the lexicon seeds).  
The SSD score is then computed as the **cosine similarity between that essay’s vector and β̂**.  
In addition, the model’s regression weights allow you to compute the **predicted outcome** for each essay — both in standardized units and in the original scale of your dependent variable.


### How scores are computed

For each document \(i\):
- (x_i) — document vector (normalized if `l2_normalize_docs=True`)
- (β̂) — unit semantic gradient in embedding space  
- `cos[i] = cos(x_i, β̂)` → **semantic alignment score**  
- `yhat_std[i] = x_i · β` → predicted standardized outcome  
- `yhat_raw[i] = mean(y) + std(y) * yhat_std[i]` → prediction in original units  

These are available for **all documents**, with NaNs for those that did not contain any lexicon occurrences (i.e., were dropped before fitting).

```python
scores = ssd.ssd_scores(
    docs, # list[list[str]]
    include_all=True) # include all docs, even those dropped due to no seed contexts

```

Returned columns:
- `doc_index`	Original document index (0-based)
- `kept`	Whether the essay had valid seed contexts (True/False)
- `cos`	Cosine alignment of essay vector to β̂
- `yhat_std`	Predicted outcome (standardized units)
- `yhat_raw`	Predicted outcome (original scale of your dependent variable)
- `y_true_std`	True standardized outcome (NaN for dropped docs)
- `y_true_raw`	True raw outcome (NaN for dropped docs)

---
## Cross-Group Comparison (SSDGroup)

When your research question involves **categorical groups** rather than a continuous outcome (e.g., clinical diagnosis, experimental condition, nationality), use `SSDGroup` instead of `SSD`.

`SSDGroup` builds per-essay concept vectors using the same pipeline as `SSD`, then:
1. Computes **unit-length group centroids** in doc-vector space.
2. Runs an **omnibus permutation test** (mean pairwise cosine distance between centroids) to test whether any groups differ.
3. Runs **pairwise permutation tests** with Bonferroni correction.
4. Constructs **centroid contrast vectors** for each pair, which plug into the same interpretation tools (neighbors, clusters, snippets).

For two groups, the omnibus test is skipped (it is identical to the single pairwise test).

The cross-group extension was introduced in:
Plisiecki, H., Sterna, A., Maciejewska, E., & Moskalewicz, M. (2026). Computational phenomenology of self and time in borderline and narcissistic personality disorders: Cross-group supervised semantic differential. *PsyArXiv*. [https://doi.org/10.31234/osf.io/r8y6b_v1](https://doi.org/10.31234/osf.io/r8y6b_v1)

### When to use SSDGroup vs SSD

| Scenario | Use |
|---|---|
| Continuous outcome (scale score, rating) | `SSD` |
| Categorical groups (diagnosis, condition) | `SSDGroup` |
| Continuous outcome AND group labels | Both — `SSD` for the continuous analysis, `SSDGroup` for the group comparison |

Do **not** discretize a continuous variable (e.g., median split) just to use `SSDGroup`. `SSD` with the continuous outcome is strictly more powerful in that case.

### Fitting SSDGroup

```python
from ssdiff import SSDGroup

sg = SSDGroup(
    kv=kv,
    docs=docs,
    groups=diagnosis_labels,   # e.g. ["BPD", "NPD", "HC", "BPD", ...]
    lexicon=lexicon,
    n_perm=5000,               # number of permutations (default 5000)
    random_state=42,           # for reproducibility
    window=3,
    sif_a=1e-3,
)
```

Parameters:
- `kv` — pretrained embeddings (KeyedVectors or path)
- `docs` — documents as token lists (same format as `SSD`)
- `groups` — group label per document (same length as `docs`); any hashable type. `None`/`NaN`/empty string entries are dropped.
- `lexicon` — seed words for the concept
- `n_perm` — number of permutations for inference (default 5000)
- `random_state` — RNG seed for reproducibility
- `l2_normalize_docs`, `window`, `sif_a`, `use_full_doc` — same as `SSD`

### Omnibus and pairwise results

```python
# Pretty-print all results
sg.print_results()

# Pairwise results as a DataFrame
sg.results_table()
#   group_A  group_B  n_A  n_B  cosine_distance  p_raw  p_corrected  cohens_d  contrast_norm
```

Key attributes:
- `sg.omnibus_T` — observed test statistic (mean pairwise cosine distance)
- `sg.omnibus_p` — permutation p-value
- `sg.pairwise` — dict mapping `(group_A, group_B)` tuples to result dicts containing `T`, `p_raw`, `p_corrected`, `cohens_d`, `contrast_unit`, etc.

### Interpreting a contrast

Extract a contrast between two groups using `get_contrast()`. The returned `SSDContrast` object exposes the same interpretation API as `SSD` (neighbors, clusters, snippets):

```python
c = sg.get_contrast("BPD", "NPD")

# Top words along the contrast direction
# +contrast → more BPD-like, −contrast → more NPD-like
c.top_words(n=15, verbose=True)

# Cluster themes on both poles
c.cluster_neighbors(topn=100, verbose=True)

# Text snippets aligned with the contrast
snips = c.beta_snippets(pre_docs=pre_docs, top_per_side=100)
df_pos = snips["beta_pos"]   # passages more BPD-like
df_neg = snips["beta_neg"]   # passages more NPD-like

# Snippets per cluster centroid (requires cluster_neighbors first)
cluster_snips = c.cluster_snippets(pre_docs=pre_docs, side="both", top_per_cluster=50)
```

Requesting the reversed contrast flips the direction automatically:

```python
c_flipped = sg.get_contrast("NPD", "BPD")
# +contrast is now NPD-like, −contrast is now BPD-like
```

### Per-participant scores

Project all participants onto a contrast direction for visualization (e.g., violin or density plots):

```python
scores = sg.contrast_scores("BPD", "NPD")
# DataFrame with columns: group, cos_to_contrast
```

---
## API Summary
The `ssdiff` top-level package re-exports the main objects so you can write:

```python
from ssdiff import (
  SSD,                       # continuous outcome analysis
  SSDGroup,                  # cross-group comparison
  SSDContrast,               # pairwise contrast (returned by SSDGroup.get_contrast)
  load_embeddings, normalize_kv,
  load_spacy, load_stopwords, preprocess_texts, build_docs_from_preprocessed,
  suggest_lexicon, token_presence_stats, coverage_by_lexicon,
)
```

### `SSD` (class)

- `__init__(kv, docs, y, lexicon, *, l2_normalize_docs=True,  N_PCA=20, use_unit_beta=True)`
- Attributes after fit: `beta`, `beta_unit`, `r2`, `f_stat`, `f_pvalue`, `beta_norm_stdCN`,
`delta_per_0p10_raw`, `iqr_effect_raw`, `y_corr_pred`, `n_kept`, etc.
- Methods:
  - `nbrs(sign=+1, n=20)` → list[(word, cosine)]
  - `cluster_neighbors_sign(side="pos", topn=100, k=None, k_min=2, k_max=10, restrict_vocab=50000, random_state=13, min_cluster_size=2, top_words=10, verbose=False)` → `(df_clusters, df_members)` and stores raw clusters in `pos_clusters_raw`/`neg_clusters_raw`
  - `cluster_snippets(pre_docs, side="both", top_per_cluster=100)` → dict with `"pos"`/`"neg"` DataFrames
  - `beta_snippets(pre_docs, top_per_side=200)` → dict with `"beta_pos"`/`"beta_neg"` DataFrames
  - `ssd_scores(include_all=True)` → DataFrame of per-essay scores

### `SSDGroup` (class)

- `__init__(kv, docs, groups, lexicon, *, n_perm=5000, random_state=42, l2_normalize_docs=True, window=3, sif_a=1e-3, use_full_doc=False)`
- Attributes after fit: `omnibus_T`, `omnibus_p`, `pairwise`, `centroids`, `group_labels`, `G`, `n_kept`, `n_dropped`
- Methods:
  - `print_results()` — pretty-print omnibus + pairwise results
  - `results_table()` → DataFrame of pairwise results
  - `get_contrast(group_a, group_b)` → `SSDContrast` (auto-flips if needed)
  - `contrast_scores(group_a, group_b)` → DataFrame with `group` and `cos_to_contrast` columns

### `SSDContrast` (class)

Returned by `SSDGroup.get_contrast()`. Duck-types with `SSD` for interpretation:
- `nbrs(sign=+1, n=16)` → nearest neighbors to the contrast direction
- `top_words(n=10, verbose=False)` → DataFrame of top words on both poles
- `cluster_neighbors(topn=100, verbose=False)` → `(df_clusters, df_members)`
- `beta_snippets(pre_docs, top_per_side=200)` → dict with `"beta_pos"`/`"beta_neg"` DataFrames
- `cluster_snippets(pre_docs, side="both", top_per_cluster=100)` → dict with `"pos"`/`"neg"` DataFrames

### Embeddings
- `load_embeddings(path)` → `gensim.models.KeyedVectors`
- `normalize_kv(kv, l2=True, abtt_m=0)` → new KeyedVectors with L2 + optional ABTT (“all-but-the-top”, top-m PCs removed)

### Preprocessing
- `load_spacy(model_name="pl_core_news_lg")` → spaCy nlp
- `load_stopwords(lang="pl")` → list of stopwords (remote Polish list with sensible fallback)
- `preprocess_texts(texts, nlp, stopwords)` → list of PreprocessedDoc
- `build_docs_from_preprocessed(pre_docs)` → list[list[str]] (lemmas for modeling)

### Lexicon
- `suggest_lexicon(df_or_tuple, text_col=None, score_col=None, top_k=150, min_docs=5, n_bins=4, corr_cap=0.30, var_type='continuous')` → DataFrame
- `token_presence_stats(texts, y, token, n_bins=4, corr_cap=0.30, verbose=False, var_type='continuous')` → dict
- `coverage_by_lexicon(df_or_tuple, text_col=None, score_col=None, lexicon=(), n_bins=4, verbose=False, var_type='continuous')` → `(summary, per_token_df)`
- `var_type`: `'continuous'` (numeric outcome, default) or `'categorical'` (group labels). When categorical, `corr` is Cramér's V, `cov_bal` is balanced across groups, and `q1`/`q4` are min/max group coverage.

--- 
## Citing & License

- License: MIT (see LICENSE).
- If you use SSD in published work, please cite the associated paper.
- A suggested citation:

Plisiecki, H., Lenartowicz, P., Pokropek, A., Małyska, K., & Flakus, M. (2025). Measuring Individual Differences in Meaning: The Supervised Semantic Differential. PsyArXiv. https://doi.org/10.31234/osf.io/gvrsb_v1

---
## Questions / Contributions
- File issues and feature requests on the repo’s Issues page.
- Pull requests welcome — especially for:
  - Robustness diagnostics and visualization helpers
  - Documentation improvements

Contact: hplisiecki@gmail.com

Project was funded by the National Science Centre, Poland (grant no. 2020/38/E/HS6/00302).
