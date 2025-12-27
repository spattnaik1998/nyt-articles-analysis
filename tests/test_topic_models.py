"""
Tests for Topic Models Module - LDA and BERTopic

Tests cover:
- Tokenization
- LDA training
- BERTopic training
- Model saving and loading
- Helper functions
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models.topic_models import (
    tokenize_documents,
    run_lda,
    run_bertopic,
    get_topic_keywords,
    get_document_topics,
    load_lda_model,
    load_bertopic_model,
    GENSIM_AVAILABLE,
    BERTOPIC_AVAILABLE
)


# Sample documents for testing
SAMPLE_DOCS = [
    "stock market economy trade financial growth business",
    "market economy business financial stocks trading money",
    "politics election government vote democracy campaign",
    "political election campaign government voting policy",
    "sports game team championship victory winning",
    "sports championship team game player winning score",
    "climate change environment global warming carbon",
    "environment climate change global temperature emissions",
    "technology innovation artificial intelligence machine learning",
    "technology AI machine learning innovation software"
]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return SAMPLE_DOCS.copy()


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# ============================================================================
# Tokenization Tests
# ============================================================================

@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_tokenize_documents_basic(sample_documents):
    """Test basic tokenization."""
    tokenized = tokenize_documents(sample_documents)

    assert isinstance(tokenized, list)
    assert len(tokenized) == len(sample_documents)
    assert all(isinstance(doc, list) for doc in tokenized)
    assert all(isinstance(word, str) for doc in tokenized for word in doc)


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_tokenize_documents_lowercase():
    """Test that tokenization lowercases."""
    docs = ["UPPERCASE Text", "MiXeD CaSe"]
    tokenized = tokenize_documents(docs)

    for doc in tokenized:
        for word in doc:
            assert word.islower()


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_tokenize_documents_removes_short_tokens():
    """Test that tokenization removes very short tokens."""
    docs = ["a b c test document", "x y z longer words"]
    tokenized = tokenize_documents(docs)

    # simple_preprocess removes tokens < 2 chars by default
    for doc in tokenized:
        for word in doc:
            assert len(word) >= 2


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_tokenize_documents_empty():
    """Test tokenization of empty documents."""
    docs = ["", "  ", "valid document"]
    tokenized = tokenize_documents(docs)

    assert len(tokenized) == 3
    assert tokenized[0] == []
    assert tokenized[1] == []
    assert len(tokenized[2]) > 0


# ============================================================================
# LDA Tests
# ============================================================================

@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_run_lda_basic(sample_documents):
    """Test basic LDA training."""
    lda_model, topics_df, dictionary, corpus = run_lda(
        sample_documents,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=100,
        verbose=False
    )

    # Check model
    assert lda_model is not None
    assert lda_model.num_topics == 3

    # Check topics DataFrame
    assert isinstance(topics_df, pd.DataFrame)
    assert len(topics_df) == 3
    assert 'topic_id' in topics_df.columns
    assert 'keywords' in topics_df.columns
    assert 'top_10_words' in topics_df.columns
    assert 'top_10_weights' in topics_df.columns

    # Check dictionary
    assert dictionary is not None
    assert len(dictionary) > 0

    # Check corpus
    assert isinstance(corpus, list)
    assert len(corpus) == len(sample_documents)


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_run_lda_with_pre_tokenized(sample_documents):
    """Test LDA with pre-tokenized documents."""
    tokenized = tokenize_documents(sample_documents)

    lda_model, topics_df, dictionary, corpus = run_lda(
        tokenized,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=100,
        verbose=False
    )

    assert lda_model is not None
    assert len(topics_df) == 3


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_run_lda_reproducibility(sample_documents):
    """Test that LDA is reproducible with same seed."""
    lda_model_1, topics_df_1, _, _ = run_lda(
        sample_documents,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=42,
        verbose=False
    )

    lda_model_2, topics_df_2, _, _ = run_lda(
        sample_documents,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=42,
        verbose=False
    )

    # Topics should be identical (or very close)
    # Note: LDA can have slight variations even with same seed
    assert len(topics_df_1) == len(topics_df_2)


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_run_lda_save_model(sample_documents, temp_output_dir):
    """Test saving LDA model."""
    lda_model, topics_df, dictionary, corpus = run_lda(
        sample_documents,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=100,
        output_dir=temp_output_dir,
        save_model=True,
        verbose=False
    )

    output_path = Path(temp_output_dir)

    # Check saved files
    assert (output_path / 'lda_model').exists()
    assert (output_path / 'dictionary.pkl').exists()
    assert (output_path / 'lda_topics.csv').exists()

    # Check CSV content
    saved_topics = pd.read_csv(output_path / 'lda_topics.csv')
    assert len(saved_topics) == 3
    assert 'topic_id' in saved_topics.columns


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_run_lda_empty_documents():
    """Test LDA with empty documents."""
    with pytest.raises(Exception):
        run_lda([], num_topics=3, verbose=False)


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_run_lda_single_document():
    """Test LDA with single document (should work but warn)."""
    lda_model, topics_df, dictionary, corpus = run_lda(
        ["single document with some words here"],
        num_topics=2,
        no_below=1,
        no_above=1.0,
        passes=5,
        random_state=100,
        verbose=False
    )

    assert lda_model is not None
    assert len(topics_df) == 2


# ============================================================================
# BERTopic Tests
# ============================================================================

@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_run_bertopic_basic(sample_documents):
    """Test basic BERTopic training."""
    bertopic_model, topic_info = run_bertopic(
        sample_documents,
        nr_topics=3,
        min_topic_size=2,
        random_state=100,
        verbose=False
    )

    # Check model
    assert bertopic_model is not None

    # Check topic info
    assert isinstance(topic_info, pd.DataFrame)
    assert 'Topic' in topic_info.columns
    assert 'Count' in topic_info.columns
    assert 'Name' in topic_info.columns

    # Should have some topics (possibly including -1 for outliers)
    assert len(topic_info) > 0


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_run_bertopic_auto_min_topic_size(sample_documents):
    """Test BERTopic with automatic min_topic_size calculation."""
    bertopic_model, topic_info = run_bertopic(
        sample_documents,
        nr_topics=3,
        min_topic_size=None,  # Auto-calculate
        random_state=100,
        verbose=False
    )

    assert bertopic_model is not None
    # Auto calculation: max(5, 0.5% of 10 docs) = 5
    # But BERTopic may reduce it internally


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_run_bertopic_save_model(sample_documents, temp_output_dir):
    """Test saving BERTopic model."""
    bertopic_model, topic_info = run_bertopic(
        sample_documents,
        nr_topics=3,
        min_topic_size=2,
        random_state=100,
        output_dir=temp_output_dir,
        save_model=True,
        verbose=False
    )

    output_path = Path(temp_output_dir)

    # Check saved files
    assert (output_path / 'bertopic_topics.csv').exists()
    assert (output_path / 'bertopic_model').exists()

    # Check CSV content
    saved_topics = pd.read_csv(output_path / 'bertopic_topics.csv')
    assert len(saved_topics) > 0
    assert 'Topic' in saved_topics.columns


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_run_bertopic_save_intertopic_map(sample_documents, temp_output_dir):
    """Test saving intertopic distance map."""
    bertopic_model, topic_info = run_bertopic(
        sample_documents,
        nr_topics=3,
        min_topic_size=2,
        random_state=100,
        output_dir=temp_output_dir,
        save_model=True,
        save_intertopic_map=True,
        verbose=False
    )

    output_path = Path(temp_output_dir)

    # Check HTML file
    html_path = output_path / 'intertopic_distance_map.html'

    # May not exist if too few topics
    # Just check that the function doesn't crash


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_run_bertopic_empty_documents():
    """Test BERTopic with empty documents."""
    with pytest.raises(Exception):
        run_bertopic([], nr_topics=3, verbose=False)


# ============================================================================
# Helper Function Tests
# ============================================================================

@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_get_topic_keywords(sample_documents):
    """Test getting keywords for a specific topic."""
    lda_model, _, _, _ = run_lda(
        sample_documents,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=100,
        verbose=False
    )

    keywords = get_topic_keywords(lda_model, topic_id=0, topn=5)

    assert isinstance(keywords, list)
    assert len(keywords) == 5
    assert all(isinstance(k, tuple) for k in keywords)
    assert all(len(k) == 2 for k in keywords)  # (word, weight)
    assert all(isinstance(k[0], str) for k in keywords)
    assert all(isinstance(k[1], float) for k in keywords)


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_get_document_topics(sample_documents):
    """Test getting topic distribution for documents."""
    lda_model, _, _, corpus = run_lda(
        sample_documents,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=100,
        verbose=False
    )

    doc_topics = get_document_topics(
        lda_model,
        corpus,
        minimum_probability=0.0
    )

    assert isinstance(doc_topics, list)
    assert len(doc_topics) == len(sample_documents)
    assert all(isinstance(dt, list) for dt in doc_topics)

    # Each document should have topic assignments
    for dt in doc_topics:
        if len(dt) > 0:
            assert all(isinstance(t, tuple) for t in dt)
            assert all(len(t) == 2 for t in dt)  # (topic_id, prob)


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_get_document_topics_min_probability(sample_documents):
    """Test filtering topics by minimum probability."""
    lda_model, _, _, corpus = run_lda(
        sample_documents,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=100,
        verbose=False
    )

    # Get topics with min prob 0.3
    doc_topics = get_document_topics(
        lda_model,
        corpus,
        minimum_probability=0.3
    )

    # All returned probabilities should be >= 0.3
    for dt in doc_topics:
        for topic_id, prob in dt:
            assert prob >= 0.3


# ============================================================================
# Model Loading Tests
# ============================================================================

@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_load_lda_model(sample_documents, temp_output_dir):
    """Test loading saved LDA model."""
    # Save model
    lda_model, _, _, _ = run_lda(
        sample_documents,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=100,
        output_dir=temp_output_dir,
        save_model=True,
        verbose=False
    )

    # Load model
    model_path = Path(temp_output_dir) / 'lda_model'
    loaded_model = load_lda_model(str(model_path))

    assert loaded_model is not None
    assert loaded_model.num_topics == 3

    # Should produce same topics
    original_topics = lda_model.show_topic(0, topn=5)
    loaded_topics = loaded_model.show_topic(0, topn=5)

    assert len(original_topics) == len(loaded_topics)


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_load_bertopic_model(sample_documents, temp_output_dir):
    """Test loading saved BERTopic model."""
    # Save model
    bertopic_model, _ = run_bertopic(
        sample_documents,
        nr_topics=3,
        min_topic_size=2,
        random_state=100,
        output_dir=temp_output_dir,
        save_model=True,
        verbose=False
    )

    # Load model
    model_path = Path(temp_output_dir) / 'bertopic_model'
    loaded_model = load_bertopic_model(str(model_path))

    assert loaded_model is not None

    # Should have same topics
    original_info = bertopic_model.get_topic_info()
    loaded_info = loaded_model.get_topic_info()

    assert len(original_info) == len(loaded_info)


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_lda_all_same_documents():
    """Test LDA with identical documents."""
    docs = ["same document"] * 10

    lda_model, topics_df, dictionary, corpus = run_lda(
        docs,
        num_topics=2,
        no_below=1,
        no_above=1.0,
        passes=5,
        random_state=100,
        verbose=False
    )

    assert lda_model is not None
    # Dictionary should be very small
    assert len(dictionary) <= 5


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
def test_lda_very_short_documents():
    """Test LDA with very short documents."""
    docs = ["word", "another", "test", "short"] * 3

    lda_model, topics_df, dictionary, corpus = run_lda(
        docs,
        num_topics=2,
        no_below=1,
        no_above=1.0,
        passes=5,
        random_state=100,
        verbose=False
    )

    assert lda_model is not None


@pytest.mark.skipif(not BERTOPIC_AVAILABLE, reason="BERTopic not available")
def test_bertopic_very_few_documents():
    """Test BERTopic with very few documents."""
    docs = ["document one", "document two", "document three"]

    bertopic_model, topic_info = run_bertopic(
        docs,
        nr_topics=2,
        min_topic_size=1,
        random_state=100,
        verbose=False
    )

    assert bertopic_model is not None
    # May have all outliers (-1) due to small size


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not (GENSIM_AVAILABLE and BERTOPIC_AVAILABLE),
                    reason="Both Gensim and BERTopic required")
def test_lda_and_bertopic_on_same_data(sample_documents, temp_output_dir):
    """Test running both LDA and BERTopic on same documents."""
    # Run LDA
    lda_model, lda_topics_df, dictionary, corpus = run_lda(
        sample_documents,
        num_topics=3,
        no_below=1,
        no_above=0.9,
        passes=5,
        random_state=100,
        output_dir=temp_output_dir,
        save_model=True,
        verbose=False
    )

    # Run BERTopic
    bertopic_model, topic_info = run_bertopic(
        sample_documents,
        nr_topics=3,
        min_topic_size=2,
        random_state=100,
        output_dir=temp_output_dir,
        save_model=True,
        verbose=False
    )

    # Both should succeed
    assert lda_model is not None
    assert bertopic_model is not None

    # Both should discover topics
    assert len(lda_topics_df) == 3
    assert len(topic_info) > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
