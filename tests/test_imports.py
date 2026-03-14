def test_public_api_imports():
    from ssdiff import (
        SSD,
        SSDGroup,
        SSDContrast,
        Embeddings,
        load_embeddings,
        pca_sweep,
        PCAKSelectionResult,
        suggest_lexicon,
        coverage_by_lexicon,
        token_presence_stats,
        load_spacy,
        load_stopwords,
        preprocess_texts,
        build_docs_from_preprocessed,
    )

    for obj in [
        SSD, SSDGroup, SSDContrast, Embeddings, load_embeddings,
        pca_sweep, PCAKSelectionResult,
        suggest_lexicon, coverage_by_lexicon, token_presence_stats,
        load_spacy, load_stopwords, preprocess_texts, build_docs_from_preprocessed,
    ]:
        assert obj is not None


def test_internal_helper_imports():
    from ssdiff import (
        cluster_top_neighbors,
        normalize_kv,
        compute_global_sif,
        build_doc_vectors,
        filtered_neighbors,
    )

    for obj in [
        cluster_top_neighbors, normalize_kv, compute_global_sif,
        build_doc_vectors, filtered_neighbors,
    ]:
        assert obj is not None
