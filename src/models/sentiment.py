"""
Sentiment Analysis Module - Multi-Model Unified Interface

This module provides sentiment classification using multiple transformer models:
- FinBERT: Financial sentiment (positive, negative, neutral)
- FinBERT-Tone: Financial tone analysis
- DistilRoBERTa: General sentiment (positive, negative, neutral)
- PoliBERT: Political sentiment analysis

Based on patterns from the NYT analysis notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
import time
from tqdm import tqdm

# Transformers imports
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

logger = logging.getLogger(__name__)


# Model Registry - Maps model_key to HuggingFace model name
MODEL_REGISTRY = {
    'finbert': {
        'model_name': 'ProsusAI/finbert',
        'labels': ['positive', 'negative', 'neutral'],
        'description': 'Financial sentiment analysis',
        'use_for_sections': ['Business', 'Business Day', 'DealBook', 'Economy', 'Markets', 'Your Money']
    },
    'finbert_tone': {
        'model_name': 'yiyanghkust/finbert-tone',
        'labels': ['positive', 'negative', 'neutral'],
        'description': 'Financial tone analysis',
        'use_for_sections': ['Business', 'Business Day', 'DealBook', 'Economy', 'Markets', 'Your Money']
    },
    'distilroberta': {
        'model_name': 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
        'labels': ['positive', 'negative', 'neutral'],
        'description': 'Financial news sentiment',
        'use_for_sections': ['Business', 'Business Day', 'DealBook', 'Economy', 'Markets', 'Your Money']
    },
    'roberta_general': {
        'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'labels': ['negative', 'neutral', 'positive'],
        'description': 'General sentiment analysis (non-financial)',
        'use_for_sections': ['World', 'U.S.', 'Politics', 'Opinion', 'Technology', 'Science',
                            'Health', 'Sports', 'Arts', 'Books', 'Style', 'Food', 'Travel',
                            'Magazine', 'T Magazine', 'Real Estate', 'Education', 'Obituaries']
    },
    'polibert': {
        'model_name': 'bucketresearch/politicalBiasBERT',
        'labels': ['left', 'center', 'right'],  # Political bias labels
        'description': 'Political bias analysis',
        'use_for_sections': ['Politics', 'Opinion', 'Washington']
    }
}


def get_device() -> str:
    """
    Auto-detect available device (CUDA, MPS, or CPU).

    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
    """
    if not TRANSFORMERS_AVAILABLE:
        return 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Using Apple MPS GPU")
    else:
        device = 'cpu'
        logger.info("Using CPU")

    return device


def select_model_for_section(section: Optional[str] = None) -> str:
    """
    Intelligently select the appropriate sentiment model based on article section.

    FinBERT models are only used for business/financial content.
    General RoBERTa is used for all other content.

    Args:
        section (str, optional): Article section name (e.g., 'Business', 'World', 'Politics')

    Returns:
        str: Model key to use for sentiment analysis

    Example:
        >>> model_key = select_model_for_section('Business')
        >>> print(model_key)
        'finbert'

        >>> model_key = select_model_for_section('World')
        >>> print(model_key)
        'roberta_general'
    """
    if section is None:
        # Default to general model
        return 'roberta_general'

    # Check if section matches any financial model
    for model_key, model_info in MODEL_REGISTRY.items():
        if 'use_for_sections' in model_info:
            if section in model_info['use_for_sections']:
                # For financial sections, prefer finbert
                if section in ['Business', 'Business Day', 'DealBook', 'Economy', 'Markets', 'Your Money']:
                    return 'finbert'
                # For political sections, use polibert
                elif section in ['Politics', 'Opinion', 'Washington']:
                    return 'polibert'

    # Default to general sentiment model for all other sections
    return 'roberta_general'


def get_recommended_models_for_section(section: Optional[str] = None) -> List[str]:
    """
    Get list of recommended models for a given section.

    Returns primary model plus any additional relevant models for comparison.

    Args:
        section (str, optional): Article section name

    Returns:
        List[str]: List of model keys, with primary model first

    Example:
        >>> models = get_recommended_models_for_section('Business')
        >>> print(models)
        ['finbert', 'finbert_tone', 'distilroberta']
    """
    if section is None:
        return ['roberta_general']

    # Financial sections: use financial models
    if section in ['Business', 'Business Day', 'DealBook', 'Economy', 'Markets', 'Your Money']:
        return ['finbert', 'finbert_tone', 'distilroberta']

    # Political sections: use political + general
    elif section in ['Politics', 'Opinion', 'Washington']:
        return ['polibert', 'roberta_general']

    # All other sections: use general model
    else:
        return ['roberta_general']


def load_sentiment_model(
    model_key: str,
    device: Optional[str] = None
) -> Tuple:
    """
    Load sentiment model and tokenizer.

    Args:
        model_key (str): Model key from MODEL_REGISTRY
        device (str, optional): Device to use ('cuda', 'mps', 'cpu'). Auto-detect if None.

    Returns:
        Tuple: (model, tokenizer, device)

    Raises:
        ValueError: If model_key not in registry
        ImportError: If transformers not available

    Example:
        >>> model, tokenizer, device = load_sentiment_model('finbert')
        >>> print(f"Loaded {model_key} on {device}")
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers required. Install with: pip install transformers torch")

    if model_key not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_key '{model_key}'. Available: {available}")

    model_name = MODEL_REGISTRY[model_key]['model_name']

    if device is None:
        device = get_device()

    logger.info(f"Loading {model_key} ({model_name}) on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    logger.info(f"✓ Loaded {model_key}")

    return model, tokenizer, device


def classify_sentiment(
    texts: Union[str, List[str]],
    model_key: str = 'roberta_general',
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[str] = None,
    return_scores: bool = True
) -> Union[Dict, List[Dict]]:
    """
    Classify sentiment using specified model.

    This is the unified interface for sentiment classification across all models.

    Args:
        texts (str or List[str]): Text(s) to classify
        model_key (str): Model to use ('finbert', 'finbert_tone', 'distilroberta', 'polibert')
        batch_size (int): Batch size for processing (default: 32)
        max_length (int): Maximum sequence length (default: 512)
        device (str, optional): Device to use. Auto-detect if None.
        return_scores (bool): Return confidence scores (default: True)

    Returns:
        Dict or List[Dict]: Sentiment results with 'label' and 'score'

    Example:
        >>> result = classify_sentiment("The market is booming!", model_key='finbert')
        >>> print(result)
        {'label': 'positive', 'score': 0.95}

        >>> results = classify_sentiment(
        ...     ["Market crashes", "Economy grows"],
        ...     model_key='finbert'
        ... )
        >>> print(results[0])
        {'label': 'negative', 'score': 0.89}
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers required. Install with: pip install transformers torch")

    # Handle single string
    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    # Load model
    model, tokenizer, device_used = load_sentiment_model(model_key, device)

    results = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(device_used) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get predictions
        probs = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)

        # Extract results
        for idx in range(len(batch)):
            pred_class = predicted_classes[idx].item()
            pred_score = probs[idx][pred_class].item()

            # Get label name
            label = model.config.id2label.get(pred_class, f"label_{pred_class}")

            result = {
                'label': label,
                'score': pred_score if return_scores else None
            }

            if not return_scores:
                del result['score']

            results.append(result)

    # Return single result if single input
    if single_input:
        return results[0]

    return results


def batch_infer(
    df: pd.DataFrame,
    text_col: str = 'combined_text',
    models: Optional[List[str]] = None,
    section_col: str = 'section_name',
    auto_select_models: bool = True,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[str] = None,
    verbose: bool = True,
    parallelize: bool = False,
    execution_strategy: str = 'auto',
    max_workers: Optional[int] = None,
    measure_performance: bool = False
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Run batch sentiment inference with multiple models.

    Intelligently selects appropriate models based on article sections:
    - FinBERT models are only used for Business/Financial sections
    - General RoBERTa is used for all other sections
    - PoliBERT is used for Politics sections

    Adds columns to DataFrame: {model}_label, {model}_score for each model.

    Args:
        df (pd.DataFrame): Input DataFrame
        text_col (str): Column containing text to classify (default: 'combined_text')
        models (List[str], optional): List of model keys to use. If None and auto_select_models=True,
                                     models will be selected based on article sections.
        section_col (str): Column containing section names (default: 'section_name')
        auto_select_models (bool): Automatically select models based on sections (default: True)
        batch_size (int): Batch size for processing (default: 32)
        max_length (int): Maximum sequence length (default: 512)
        device (str, optional): Device to use. Auto-detect if None. (Ignored if parallelize=True)
        verbose (bool): Show progress bars (default: True)
        parallelize (bool): Use parallel execution (ProcessPoolExecutor/ThreadPoolExecutor) (default: False)
        execution_strategy (str): 'auto', 'process', 'thread', or 'sequential' (default: 'auto')
        max_workers (int, optional): Max workers for parallel execution. Auto-detect if None.
        measure_performance (bool): Track and return performance metrics (default: False)

    Returns:
        Tuple[pd.DataFrame, Optional[Dict]]: (result_df, performance_metrics)
            - If measure_performance=False: (result_df, None)
            - If measure_performance=True: (result_df, {timing, speedup, device_info, ...})

    Example:
        >>> # Auto-select models based on sections
        >>> df = pd.DataFrame({
        ...     'text': ['Market booms', 'Politics news', 'Sports update'],
        ...     'section_name': ['Business', 'Politics', 'Sports']
        ... })
        >>> df, perf = batch_infer(df, text_col='text', parallelize=True, measure_performance=True)
        >>> print(f"Speedup: {perf['speedup']:.2f}x")

        >>> # Use specific models with parallelization
        >>> df, _ = batch_infer(df, text_col='text', models=['finbert', 'roberta_general'],
        ...                     auto_select_models=False, parallelize=True, execution_strategy='process')
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers required. Install with: pip install transformers torch")

    # Validate
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    # Auto-select models based on sections if requested
    if models is None and auto_select_models:
        if section_col in df.columns:
            # Get unique sections
            unique_sections = df[section_col].dropna().unique()

            # Collect all recommended models for these sections
            models_set = set()
            for section in unique_sections:
                section_models = get_recommended_models_for_section(section)
                models_set.update(section_models)

            models = list(models_set)

            if verbose:
                logger.info(f"Auto-selected models based on sections: {models}")
                logger.info(f"Sections found: {list(unique_sections)}")
        else:
            # No section column, use general model
            models = ['roberta_general']
            if verbose:
                logger.info(f"No '{section_col}' column found, using general model")
    elif models is None:
        # No auto-select and no models specified, use general model
        models = ['roberta_general']

    # Prepare texts
    texts = df[text_col].fillna('').astype(str).tolist()

    if verbose:
        logger.info(f"Running batch inference on {len(texts):,} texts with {len(models)} model(s)")

    result_df = df.copy()

    # Parallel execution path
    if parallelize:
        from src.models.sentiment_parallel_executor import create_executor

        t_parallel_start = time.time()

        executor = create_executor(
            prefer_gpu=(device != 'cpu'),
            max_workers=max_workers,
            verbose=verbose
        )

        try:
            # Execute all models in parallel
            parallel_results, strategy_used = executor.execute_with_fallback(
                texts,
                models,
                batch_size,
                max_length,
                execution_strategy=execution_strategy
            )

            t_parallel_total = time.time() - t_parallel_start

            # Collect performance metrics if requested
            performance_metrics = {
                'execution_strategy': strategy_used,
                'total_time_ms': t_parallel_total * 1000,
                'texts_count': len(texts),
                'models_count': len(models),
                'model_timings': {}
            }

            # Add results to DataFrame
            for model_key, (labels, scores, timing) in parallel_results.items():
                result_df[f'{model_key}_label'] = labels
                result_df[f'{model_key}_score'] = scores

                performance_metrics['model_timings'][model_key] = {
                    'inference_ms': timing.get('inference_ms', 0),
                    'total_ms': timing.get('total_ms', 0)
                }

                if verbose:
                    # Show label distribution
                    label_counts = pd.Series(labels).value_counts()
                    logger.info(f"\n{model_key} Label Distribution:")
                    for label, count in label_counts.items():
                        pct = count / len(labels) * 100
                        logger.info(f"  {label}: {count:,} ({pct:.1f}%)")

            if verbose:
                logger.info(f"\n{'='*70}")
                logger.info(f"Parallel batch inference complete!")
                logger.info(f"Strategy: {strategy_used}, Total time: {t_parallel_total*1000:.1f}ms")
                logger.info(f"Added columns: {[f'{m}_label' for m in models] + [f'{m}_score' for m in models]}")
                logger.info(f"{'='*70}")

            if measure_performance:
                return result_df, performance_metrics
            else:
                return result_df, None

        except Exception as e:
            logger.error(f"Parallel execution failed: {e}. Falling back to sequential.")
            # Fall through to sequential execution
            parallelize = False

    # Auto-detect device once (sequential path only)
    if device is None:
        device = get_device()

    # Sequential execution path (original implementation)
    t_sequential_start = time.time()
    performance_metrics = {
        'execution_strategy': 'sequential',
        'total_time_ms': 0,
        'texts_count': len(texts),
        'models_count': len(models),
        'model_timings': {}
    } if measure_performance else None

    # Run each model
    for model_key in models:
        if verbose:
            logger.info(f"\n{'='*70}")
            logger.info(f"Running {model_key} ({MODEL_REGISTRY[model_key]['description']})")
            logger.info(f"{'='*70}")

        # Load model
        model, tokenizer, device_used = load_sentiment_model(model_key, device)

        labels = []
        scores = []

        # Process in batches with progress bar
        iterator = range(0, len(texts), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc=f"{model_key} inference")

        for i in iterator:
            batch = texts[i:i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            # Move to device
            inputs = {k: v.to(device_used) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Get predictions
            probs = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1)

            # Extract results
            for idx in range(len(batch)):
                pred_class = predicted_classes[idx].item()
                pred_score = probs[idx][pred_class].item()

                # Get label name
                label = model.config.id2label.get(pred_class, f"label_{pred_class}")

                labels.append(label)
                scores.append(pred_score)

        # Add columns to DataFrame
        result_df[f'{model_key}_label'] = labels
        result_df[f'{model_key}_score'] = scores

        if verbose:
            # Show label distribution
            label_counts = pd.Series(labels).value_counts()
            logger.info(f"\n{model_key} Label Distribution:")
            for label, count in label_counts.items():
                pct = count / len(labels) * 100
                logger.info(f"  {label}: {count:,} ({pct:.1f}%)")

        # Track timing if measuring performance
        if measure_performance:
            performance_metrics['model_timings'][model_key] = {
                'inference_ms': 0,  # Not tracked in sequential mode for simplicity
                'total_ms': 0
            }

        # Clean up to free memory
        # Clear CUDA cache first if using CUDA
        if device_used == 'cuda':
            torch.cuda.empty_cache()
        del model
        # Clear cache again after model deletion
        if device_used == 'cuda':
            torch.cuda.empty_cache()

    if verbose:
        logger.info(f"\n{'='*70}")
        logger.info(f"Batch inference complete!")
        logger.info(f"Added columns: {[f'{m}_label' for m in models] + [f'{m}_score' for m in models]}")
        logger.info(f"{'='*70}")

    if measure_performance:
        t_sequential_total = time.time() - t_sequential_start
        performance_metrics['total_time_ms'] = t_sequential_total * 1000
        return result_df, performance_metrics
    else:
        return result_df, None


def batch_infer_parallel(
    df: pd.DataFrame,
    text_col: str = 'combined_text',
    models: Optional[List[str]] = None,
    section_col: str = 'section_name',
    auto_select_models: bool = True,
    batch_size: int = 32,
    max_workers: int = 4,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    High-level parallel sentiment inference (recommended for large batches).

    Convenience function that calls batch_infer with sensible defaults for
    parallel execution:
    - Uses intelligent execution strategy selection (process → thread → sequential)
    - Automatically detects device (GPU preferred)
    - Always measures performance
    - Maximum 4 workers by default

    Args:
        df (pd.DataFrame): Input DataFrame
        text_col (str): Column containing text to classify (default: 'combined_text')
        models (List[str], optional): List of model keys to use. Auto-selects if None.
        section_col (str): Column containing section names (default: 'section_name')
        auto_select_models (bool): Automatically select models based on sections (default: True)
        batch_size (int): Batch size for processing (default: 32)
        max_workers (int): Max workers for parallelization (default: 4)
        verbose (bool): Show progress info (default: True)

    Returns:
        Tuple[pd.DataFrame, Dict]: (result_df, performance_metrics)
            - result_df: DataFrame with added sentiment columns
            - performance_metrics: Timing, strategy, speedup information

    Example:
        >>> df, metrics = batch_infer_parallel(df, text_col='combined_text')
        >>> print(f"Strategy: {metrics['execution_strategy']}")
        >>> print(f"Total time: {metrics['total_time_ms']:.1f}ms")
    """
    result_df, performance_metrics = batch_infer(
        df,
        text_col=text_col,
        models=models,
        section_col=section_col,
        auto_select_models=auto_select_models,
        batch_size=batch_size,
        max_length=512,
        device=None,  # Auto-detect
        verbose=verbose,
        parallelize=True,
        execution_strategy='auto',
        max_workers=max_workers,
        measure_performance=True
    )

    if performance_metrics is None:
        performance_metrics = {}

    return result_df, performance_metrics


def model_comparison_report(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    min_score_threshold: float = 0.5,
    verbose: bool = True
) -> Dict:
    """
    Generate sentiment model comparison report.

    Analyzes agreement/disagreement between models and generates statistics.

    Args:
        df (pd.DataFrame): DataFrame with sentiment results (from batch_infer)
        models (List[str], optional): Models to compare. Auto-detect if None.
        output_path (str, optional): Path to save report text file
        min_score_threshold (float): Minimum score to consider prediction confident (default: 0.5)
        verbose (bool): Print report to console (default: True)

    Returns:
        Dict: Report statistics including:
            - total_documents: Total number of documents
            - models_compared: List of models
            - agreement_rate: Overall agreement rate
            - disagreement_count: Number of disagreements
            - label_distributions: Per-model label counts
            - confusion_matrices: Cross-model confusion data

    Example:
        >>> df = batch_infer(df, models=['finbert', 'distilroberta'])
        >>> report = model_comparison_report(df, output_path='data/sentiment_report.txt')
        >>> print(f"Agreement rate: {report['agreement_rate']:.1%}")
    """
    # Auto-detect models if not specified
    if models is None:
        models = []
        for col in df.columns:
            if col.endswith('_label'):
                model_name = col.replace('_label', '')
                if model_name in MODEL_REGISTRY:
                    models.append(model_name)

        if not models:
            raise ValueError("No sentiment model columns found in DataFrame")

    # Validate columns exist
    for model in models:
        label_col = f'{model}_label'
        score_col = f'{model}_score'

        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found in DataFrame")
        if score_col not in df.columns:
            raise ValueError(f"Column '{score_col}' not found in DataFrame")

    logger.info(f"Generating comparison report for models: {models}")

    # Initialize report
    report = {
        'total_documents': len(df),
        'models_compared': models,
        'label_distributions': {},
        'score_stats': {},
        'agreement_analysis': {},
        'disagreements': []
    }

    # 1. Label distributions per model
    for model in models:
        label_col = f'{model}_label'
        score_col = f'{model}_score'

        label_counts = df[label_col].value_counts().to_dict()
        score_mean = df[score_col].mean()
        score_median = df[score_col].median()

        report['label_distributions'][model] = label_counts
        report['score_stats'][model] = {
            'mean': score_mean,
            'median': score_median,
            'min': df[score_col].min(),
            'max': df[score_col].max()
        }

    # 2. Agreement analysis
    if len(models) >= 2:
        # Check agreement across all models
        label_cols = [f'{m}_label' for m in models]

        # Perfect agreement: all models agree
        agreement_mask = df[label_cols].nunique(axis=1) == 1
        agreement_count = agreement_mask.sum()
        agreement_rate = agreement_count / len(df)

        report['agreement_analysis'] = {
            'perfect_agreement_count': int(agreement_count),
            'perfect_agreement_rate': float(agreement_rate),
            'disagreement_count': int(len(df) - agreement_count),
            'disagreement_rate': float(1 - agreement_rate)
        }

        # Find disagreement rows
        disagreement_df = df[~agreement_mask].copy()

        # Add to report
        if len(disagreement_df) > 0:
            # Get top disagreements (by lowest average score - indicates uncertainty)
            score_cols = [f'{m}_score' for m in models]
            disagreement_df['avg_score'] = disagreement_df[score_cols].mean(axis=1)
            disagreement_df = disagreement_df.sort_values('avg_score')

            # Store disagreement info
            for idx, row in disagreement_df.head(100).iterrows():  # Top 100 disagreements
                disagreement_info = {
                    'index': int(idx),
                    'avg_score': float(row['avg_score'])
                }

                # Add labels and scores for each model
                for model in models:
                    disagreement_info[f'{model}_label'] = row[f'{model}_label']
                    disagreement_info[f'{model}_score'] = float(row[f'{model}_score'])

                # Add text if available
                if 'combined_text' in row:
                    disagreement_info['text'] = str(row['combined_text'])[:200]  # First 200 chars

                report['disagreements'].append(disagreement_info)

        # Pairwise agreement
        if len(models) == 2:
            model_a, model_b = models
            label_a_col = f'{model_a}_label'
            label_b_col = f'{model_b}_label'

            pairwise_agreement = (df[label_a_col] == df[label_b_col]).sum()
            pairwise_rate = pairwise_agreement / len(df)

            report['agreement_analysis']['pairwise'] = {
                'models': [model_a, model_b],
                'agreement_count': int(pairwise_agreement),
                'agreement_rate': float(pairwise_rate)
            }

    # 3. Generate text report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("SENTIMENT MODEL COMPARISON REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"\nTotal Documents: {report['total_documents']:,}")
    report_lines.append(f"Models Compared: {', '.join(models)}")
    report_lines.append(f"\n{'-'*70}")
    report_lines.append("LABEL DISTRIBUTIONS")
    report_lines.append(f"{'-'*70}")

    for model in models:
        report_lines.append(f"\n{model.upper()} ({MODEL_REGISTRY[model]['description']}):")
        label_counts = report['label_distributions'][model]

        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            pct = count / report['total_documents'] * 100
            report_lines.append(f"  {label:15s}: {count:8,} ({pct:5.1f}%)")

        score_stats = report['score_stats'][model]
        report_lines.append(f"  Score stats: mean={score_stats['mean']:.3f}, median={score_stats['median']:.3f}")

    # Agreement analysis
    if 'agreement_analysis' in report and report['agreement_analysis']:
        report_lines.append(f"\n{'-'*70}")
        report_lines.append("AGREEMENT ANALYSIS")
        report_lines.append(f"{'-'*70}")

        aa = report['agreement_analysis']
        report_lines.append(f"\nPerfect Agreement: {aa['perfect_agreement_count']:,} / {report['total_documents']:,} ({aa['perfect_agreement_rate']:.1%})")
        report_lines.append(f"Disagreements: {aa['disagreement_count']:,} ({aa['disagreement_rate']:.1%})")

        if 'pairwise' in aa:
            pw = aa['pairwise']
            report_lines.append(f"\nPairwise Agreement ({pw['models'][0]} vs {pw['models'][1]}):")
            report_lines.append(f"  {pw['agreement_count']:,} / {report['total_documents']:,} ({pw['agreement_rate']:.1%})")

    # Disagreement examples
    if report['disagreements']:
        report_lines.append(f"\n{'-'*70}")
        report_lines.append(f"TOP DISAGREEMENTS (showing up to 20)")
        report_lines.append(f"{'-'*70}")

        for i, dis in enumerate(report['disagreements'][:20], 1):
            report_lines.append(f"\n{i}. Row {dis['index']} (avg_score={dis['avg_score']:.3f}):")

            for model in models:
                label = dis[f'{model}_label']
                score = dis[f'{model}_score']
                report_lines.append(f"   {model:15s}: {label:10s} (score={score:.3f})")

            if 'text' in dis:
                report_lines.append(f"   Text: {dis['text']}...")

    report_lines.append(f"\n{'='*70}")
    report_lines.append("END OF REPORT")
    report_lines.append(f"{'='*70}")

    report_text = '\n'.join(report_lines)

    # Print to console if verbose
    if verbose:
        print(report_text)

    # Save to file if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"\n✓ Report saved to {output_path}")

    return report


def get_model_info(model_key: str) -> Dict:
    """
    Get information about a sentiment model.

    Args:
        model_key (str): Model key

    Returns:
        Dict: Model information

    Example:
        >>> info = get_model_info('finbert')
        >>> print(info['description'])
        'Financial sentiment analysis'
    """
    if model_key not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_key '{model_key}'. Available: {available}")

    return MODEL_REGISTRY[model_key].copy()


def list_available_models() -> List[str]:
    """
    List all available sentiment models.

    Returns:
        List[str]: List of model keys

    Example:
        >>> models = list_available_models()
        >>> print(models)
        ['finbert', 'finbert_tone', 'distilroberta', 'polibert']
    """
    return list(MODEL_REGISTRY.keys())


def generate_sentiment_pie_chart(
    label_distribution: Dict[str, int],
    model_name: str,
    total_count: int,
    return_base64: bool = True
) -> Union[str, None]:
    """
    Generate a pie chart for sentiment label distribution.

    Args:
        label_distribution (Dict[str, int]): Dictionary with label counts
        model_name (str): Name of the sentiment model
        total_count (int): Total number of classified documents
        return_base64 (bool): If True, return base64-encoded PNG. If False, return matplotlib figure.

    Returns:
        str or None: Base64-encoded PNG image string if return_base64=True, else None

    Example:
        >>> label_dist = {'positive': 150, 'negative': 80, 'neutral': 70}
        >>> chart = generate_sentiment_pie_chart(label_dist, 'finbert', 300)
        >>> print(chart[:50])  # First 50 chars of base64 string
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import io
        import base64
    except ImportError:
        logger.warning("Matplotlib not available. Install with: pip install matplotlib")
        return None

    # Prepare data
    labels = list(label_distribution.keys())
    sizes = list(label_distribution.values())

    if not labels or sum(sizes) == 0:
        logger.warning("No data to plot for pie chart")
        return None

    # Calculate percentages
    percentages = [(size / total_count * 100) for size in sizes]

    # Color scheme based on sentiment labels
    color_map = {
        'positive': '#28a745',  # Green
        'negative': '#dc3545',  # Red
        'neutral': '#6c757d',   # Gray
        'left': '#0066cc',      # Blue
        'center': '#6c757d',    # Gray
        'right': '#cc0000',     # Red
    }

    colors = [color_map.get(label.lower(), '#326891') for label in labels]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create pie chart with improved styling
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    # Improve text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_weight('bold')

    for text in texts:
        text.set_fontsize(12)
        text.set_weight('bold')

    # Add title
    title = f'{model_name.upper()} Sentiment Distribution\n({total_count:,} articles)'
    ax.set_title(title, fontsize=14, weight='bold', pad=20)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    # Add legend with counts
    legend_labels = [f'{label}: {count:,} ({pct:.1f}%)'
                    for label, count, pct in zip(labels, sizes, percentages)]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    plt.tight_layout()

    if return_base64:
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='PNG', dpi=100, bbox_inches='tight')
        buffer.seek(0)

        # Encode to base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)

        return f'data:image/png;base64,{image_base64}'
    else:
        return None


if __name__ == "__main__":
    # Example usage
    print("Sentiment Analysis Module")
    print("=" * 70)

    # List available models
    print("\nAvailable Models:")
    for model_key in list_available_models():
        info = get_model_info(model_key)
        print(f"  {model_key:20s}: {info['description']}")
        print(f"    Model: {info['model_name']}")
        print(f"    Labels: {', '.join(info['labels'])}")
        print()

    # Example with sample texts
    if TRANSFORMERS_AVAILABLE:
        print("\n" + "=" * 70)
        print("Example: Single Text Classification")
        print("=" * 70)

        sample_text = "The stock market rallied today as economic indicators showed strong growth."

        print(f"\nText: {sample_text}")
        print(f"\nResults:")

        for model_key in ['finbert', 'distilroberta']:
            try:
                result = classify_sentiment(sample_text, model_key=model_key)
                print(f"  {model_key:15s}: {result['label']:10s} (score={result['score']:.3f})")
            except Exception as e:
                print(f"  {model_key:15s}: Error - {e}")

        print("\n" + "=" * 70)
        print("Example: Batch Classification")
        print("=" * 70)

        sample_df = pd.DataFrame({
            'text': [
                "The economy is booming with strong growth",
                "Market crashes amid recession fears",
                "Stable economic conditions continue"
            ]
        })

        print(f"\nProcessing {len(sample_df)} texts...")

        try:
            result_df = batch_infer(
                sample_df,
                text_col='text',
                models=['finbert'],
                verbose=False
            )

            print("\nResults:")
            print(result_df[['text', 'finbert_label', 'finbert_score']])
        except Exception as e:
            print(f"Error: {e}")

    else:
        print("\nTransformers not available. Install with:")
        print("  pip install transformers torch")

    print("\n" + "=" * 70)
    print("For full examples, see:")
    print("  - examples/sentiment_report.py")
    print("  - docs/sentiment_analysis.md")
    print("=" * 70)
