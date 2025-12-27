"""
Tests for Sentiment Analysis Module

Tests cover:
- Model registry
- Device detection
- Single text classification
- Batch classification
- Multi-model inference
- Comparison report generation
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models.sentiment import (
    get_device,
    load_sentiment_model,
    classify_sentiment,
    batch_infer,
    model_comparison_report,
    get_model_info,
    list_available_models,
    MODEL_REGISTRY,
    TRANSFORMERS_AVAILABLE
)


# Sample texts for testing
SAMPLE_TEXTS = [
    "The stock market rallied today with strong gains across all sectors.",
    "Economic recession fears mount as markets tumble and investors flee.",
    "The economy shows moderate growth with stable employment figures.",
]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return SAMPLE_TEXTS.copy()


@pytest.fixture
def sample_dataframe():
    """Provide sample DataFrame for testing."""
    return pd.DataFrame({
        'text': SAMPLE_TEXTS,
        'section': ['Business', 'Business', 'Business']
    })


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# ============================================================================
# Model Registry Tests
# ============================================================================

def test_model_registry_structure():
    """Test that MODEL_REGISTRY has correct structure."""
    assert isinstance(MODEL_REGISTRY, dict)
    assert len(MODEL_REGISTRY) > 0

    for model_key, model_info in MODEL_REGISTRY.items():
        assert 'model_name' in model_info
        assert 'labels' in model_info
        assert 'description' in model_info
        assert isinstance(model_info['labels'], list)


def test_list_available_models():
    """Test listing available models."""
    models = list_available_models()

    assert isinstance(models, list)
    assert len(models) > 0
    assert 'finbert' in models
    assert 'distilroberta' in models


def test_get_model_info_valid():
    """Test getting info for valid model."""
    info = get_model_info('finbert')

    assert isinstance(info, dict)
    assert 'model_name' in info
    assert 'labels' in info
    assert 'description' in info
    assert info['model_name'] == 'ProsusAI/finbert'


def test_get_model_info_invalid():
    """Test getting info for invalid model raises error."""
    with pytest.raises(ValueError, match="Unknown model_key"):
        get_model_info('invalid_model')


# ============================================================================
# Device Detection Tests
# ============================================================================

def test_get_device():
    """Test device detection."""
    device = get_device()

    assert isinstance(device, str)
    assert device in ['cuda', 'mps', 'cpu']


# ============================================================================
# Model Loading Tests
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_load_sentiment_model_valid():
    """Test loading a valid sentiment model."""
    model, tokenizer, device = load_sentiment_model('finbert')

    assert model is not None
    assert tokenizer is not None
    assert device in ['cuda', 'mps', 'cpu']


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_load_sentiment_model_invalid():
    """Test loading invalid model raises error."""
    with pytest.raises(ValueError, match="Unknown model_key"):
        load_sentiment_model('invalid_model')


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_load_sentiment_model_custom_device():
    """Test loading model with custom device."""
    model, tokenizer, device = load_sentiment_model('finbert', device='cpu')

    assert device == 'cpu'


# ============================================================================
# Classification Tests
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_classify_sentiment_single_text():
    """Test classifying single text."""
    result = classify_sentiment(
        "The economy is booming!",
        model_key='finbert',
        return_scores=True
    )

    assert isinstance(result, dict)
    assert 'label' in result
    assert 'score' in result
    assert isinstance(result['label'], str)
    assert isinstance(result['score'], float)
    assert 0 <= result['score'] <= 1


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_classify_sentiment_batch():
    """Test classifying batch of texts."""
    texts = [
        "Market rallies on strong earnings",
        "Economy struggles amid recession fears"
    ]

    results = classify_sentiment(texts, model_key='finbert')

    assert isinstance(results, list)
    assert len(results) == len(texts)
    assert all('label' in r for r in results)
    assert all('score' in r for r in results)


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_classify_sentiment_without_scores():
    """Test classification without returning scores."""
    result = classify_sentiment(
        "Test text",
        model_key='finbert',
        return_scores=False
    )

    assert isinstance(result, dict)
    assert 'label' in result
    assert 'score' not in result


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_classify_sentiment_empty_text():
    """Test classification of empty text."""
    result = classify_sentiment("", model_key='finbert')

    assert isinstance(result, dict)
    assert 'label' in result
    # Should still produce some result even for empty text


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_classify_sentiment_different_models():
    """Test classification with different models."""
    text = "The market is doing well"

    for model_key in ['finbert', 'distilroberta']:
        result = classify_sentiment(text, model_key=model_key)

        assert isinstance(result, dict)
        assert 'label' in result
        assert 'score' in result


# ============================================================================
# Batch Inference Tests
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_batch_infer_single_model(sample_dataframe):
    """Test batch inference with single model."""
    result_df = batch_infer(
        sample_dataframe,
        text_col='text',
        models=['finbert'],
        verbose=False
    )

    # Check columns added
    assert 'finbert_label' in result_df.columns
    assert 'finbert_score' in result_df.columns

    # Check values
    assert len(result_df) == len(sample_dataframe)
    assert result_df['finbert_label'].notna().all()
    assert result_df['finbert_score'].notna().all()

    # Check score range
    assert (result_df['finbert_score'] >= 0).all()
    assert (result_df['finbert_score'] <= 1).all()


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_batch_infer_multiple_models(sample_dataframe):
    """Test batch inference with multiple models."""
    result_df = batch_infer(
        sample_dataframe,
        text_col='text',
        models=['finbert', 'distilroberta'],
        verbose=False
    )

    # Check columns for both models
    assert 'finbert_label' in result_df.columns
    assert 'finbert_score' in result_df.columns
    assert 'distilroberta_label' in result_df.columns
    assert 'distilroberta_score' in result_df.columns

    # Check values
    assert len(result_df) == len(sample_dataframe)


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_batch_infer_preserves_original_columns(sample_dataframe):
    """Test that batch_infer preserves original DataFrame columns."""
    original_cols = set(sample_dataframe.columns)

    result_df = batch_infer(
        sample_dataframe,
        text_col='text',
        models=['finbert'],
        verbose=False
    )

    # Original columns should still be present
    for col in original_cols:
        assert col in result_df.columns


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_batch_infer_invalid_text_column(sample_dataframe):
    """Test batch_infer with invalid text column raises error."""
    with pytest.raises(ValueError, match="Column .* not found"):
        batch_infer(
            sample_dataframe,
            text_col='nonexistent_column',
            models=['finbert'],
            verbose=False
        )


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_batch_infer_custom_batch_size(sample_dataframe):
    """Test batch inference with custom batch size."""
    result_df = batch_infer(
        sample_dataframe,
        text_col='text',
        models=['finbert'],
        batch_size=1,  # Process one at a time
        verbose=False
    )

    assert len(result_df) == len(sample_dataframe)
    assert 'finbert_label' in result_df.columns


# ============================================================================
# Comparison Report Tests
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_model_comparison_report_basic(sample_dataframe, temp_output_dir):
    """Test generating comparison report."""
    # First run batch inference
    result_df = batch_infer(
        sample_dataframe,
        text_col='text',
        models=['finbert', 'distilroberta'],
        verbose=False
    )

    # Generate report
    report_path = Path(temp_output_dir) / 'report.txt'

    report = model_comparison_report(
        result_df,
        models=['finbert', 'distilroberta'],
        output_path=str(report_path),
        verbose=False
    )

    # Check report structure
    assert isinstance(report, dict)
    assert 'total_documents' in report
    assert 'models_compared' in report
    assert 'label_distributions' in report
    assert 'agreement_analysis' in report

    # Check values
    assert report['total_documents'] == len(sample_dataframe)
    assert report['models_compared'] == ['finbert', 'distilroberta']

    # Check file created
    assert report_path.exists()

    # Check file content
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert 'SENTIMENT MODEL COMPARISON REPORT' in content
        assert 'finbert' in content.lower()
        assert 'distilroberta' in content.lower()


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_model_comparison_report_auto_detect_models(sample_dataframe):
    """Test auto-detecting models from DataFrame."""
    # Run batch inference
    result_df = batch_infer(
        sample_dataframe,
        text_col='text',
        models=['finbert'],
        verbose=False
    )

    # Generate report without specifying models (auto-detect)
    report = model_comparison_report(
        result_df,
        models=None,  # Auto-detect
        verbose=False
    )

    assert 'finbert' in report['models_compared']


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_model_comparison_report_agreement_stats(sample_dataframe):
    """Test agreement statistics in comparison report."""
    # Run batch inference with two models
    result_df = batch_infer(
        sample_dataframe,
        text_col='text',
        models=['finbert', 'distilroberta'],
        verbose=False
    )

    report = model_comparison_report(
        result_df,
        models=['finbert', 'distilroberta'],
        verbose=False
    )

    # Check agreement analysis exists
    assert 'agreement_analysis' in report
    aa = report['agreement_analysis']

    assert 'perfect_agreement_count' in aa
    assert 'perfect_agreement_rate' in aa
    assert 'disagreement_count' in aa
    assert 'disagreement_rate' in aa

    # Rates should sum to 1
    assert abs(aa['perfect_agreement_rate'] + aa['disagreement_rate'] - 1.0) < 0.01

    # Counts should sum to total
    assert aa['perfect_agreement_count'] + aa['disagreement_count'] == report['total_documents']


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_model_comparison_report_label_distributions(sample_dataframe):
    """Test label distributions in comparison report."""
    # Run batch inference
    result_df = batch_infer(
        sample_dataframe,
        text_col='text',
        models=['finbert'],
        verbose=False
    )

    report = model_comparison_report(
        result_df,
        models=['finbert'],
        verbose=False
    )

    # Check label distribution
    assert 'label_distributions' in report
    assert 'finbert' in report['label_distributions']

    label_dist = report['label_distributions']['finbert']
    assert isinstance(label_dist, dict)

    # Counts should sum to total documents
    total_count = sum(label_dist.values())
    assert total_count == report['total_documents']


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_model_comparison_report_missing_columns(sample_dataframe):
    """Test report with missing sentiment columns raises error."""
    # Don't run batch_infer, so columns don't exist
    with pytest.raises(ValueError, match="Column .* not found"):
        model_comparison_report(
            sample_dataframe,
            models=['finbert'],
            verbose=False
        )


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_classify_sentiment_very_long_text():
    """Test classification of very long text (will be truncated)."""
    long_text = "economy " * 1000  # Very long text

    result = classify_sentiment(long_text, model_key='finbert')

    assert isinstance(result, dict)
    assert 'label' in result
    # Should handle truncation gracefully


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_batch_infer_with_nan_values():
    """Test batch inference with NaN values."""
    df = pd.DataFrame({
        'text': ['Good news', None, 'Bad news', '']
    })

    result_df = batch_infer(
        df,
        text_col='text',
        models=['finbert'],
        verbose=False
    )

    # Should handle NaN/empty gracefully
    assert len(result_df) == len(df)
    assert 'finbert_label' in result_df.columns


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_batch_infer_single_row():
    """Test batch inference with single row."""
    df = pd.DataFrame({'text': ['Single text']})

    result_df = batch_infer(
        df,
        text_col='text',
        models=['finbert'],
        verbose=False
    )

    assert len(result_df) == 1
    assert 'finbert_label' in result_df.columns


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
def test_full_workflow(sample_dataframe, temp_output_dir):
    """Test complete workflow: batch_infer -> comparison_report."""
    # Step 1: Run batch inference with multiple models
    result_df = batch_infer(
        sample_dataframe,
        text_col='text',
        models=['finbert', 'distilroberta'],
        verbose=False
    )

    # Step 2: Save results
    output_path = Path(temp_output_dir) / 'results.csv'
    result_df.to_csv(output_path, index=False)
    assert output_path.exists()

    # Step 3: Generate comparison report
    report_path = Path(temp_output_dir) / 'report.txt'
    report = model_comparison_report(
        result_df,
        models=['finbert', 'distilroberta'],
        output_path=str(report_path),
        verbose=False
    )

    assert report_path.exists()
    assert report['total_documents'] == len(sample_dataframe)

    # Step 4: Verify we can reload results
    loaded_df = pd.read_csv(output_path)
    assert len(loaded_df) == len(result_df)
    assert 'finbert_label' in loaded_df.columns


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
