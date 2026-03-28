#!/usr/bin/env python3
"""
Sentiment Analysis Parallelization Demo

Demonstrates parallel model execution vs sequential execution.
Shows performance improvements with different execution strategies.

Usage:
    python sentiment_parallel_demo.py
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.sentiment import batch_infer, batch_infer_parallel
from src.models.sentiment_performance import (
    benchmark_sentiment_models,
    compare_execution_strategies,
    SentimentPerformanceMetrics
)

# Sample data for testing
SAMPLE_TEXTS = [
    "The market surged today with strong economic indicators pointing to growth.",
    "Stock prices plummeted amid recession fears and unemployment concerns.",
    "Technology companies showed resilience despite challenging market conditions.",
    "Federal Reserve signals potential interest rate increases in coming months.",
    "Consumer spending remains robust despite inflationary pressures.",
    "Banking sector faces headwinds from tightening credit conditions.",
    "Energy markets rally on supply constraints and geopolitical tensions.",
    "Retail sales exceeded expectations, surprising analysts and investors.",
    "Manufacturing output declined, raising concerns about economic slowdown.",
    "Real estate market shows signs of cooling as mortgage rates rise higher."
]


def demo_single_inference():
    """Demonstrate single inference with both sequential and parallel."""
    print("\n" + "=" * 70)
    print("DEMO 1: SINGLE INFERENCE - SEQUENTIAL VS PARALLEL")
    print("=" * 70)

    # Create sample dataframe
    df = pd.DataFrame({
        'text': SAMPLE_TEXTS * 5,  # 50 texts
        'section_name': ['Business'] * len(SAMPLE_TEXTS * 5)
    })

    print(f"\nDataset: {len(df)} articles")
    print("Models: Auto-select based on section (Business -> FinBERT)")

    # Sequential execution
    print("\n--- Sequential Execution ---")
    result_seq, perf_seq = batch_infer(
        df,
        text_col='text',
        auto_select_models=True,
        batch_size=32,
        verbose=False,
        parallelize=False,
        measure_performance=True
    )

    if perf_seq:
        print(f"Strategy: {perf_seq['execution_strategy']}")
        print(f"Total time: {perf_seq['total_time_ms']:.1f}ms")
        print(f"Models: {perf_seq['models_count']}")

    # Parallel execution
    print("\n--- Parallel Execution ---")
    result_par, perf_par = batch_infer(
        df,
        text_col='text',
        auto_select_models=True,
        batch_size=32,
        verbose=False,
        parallelize=True,
        execution_strategy='auto',
        measure_performance=True
    )

    if perf_par:
        print(f"Strategy: {perf_par['execution_strategy']}")
        print(f"Total time: {perf_par['total_time_ms']:.1f}ms")
        print(f"Models: {perf_par['models_count']}")

        if perf_seq and perf_seq['total_time_ms'] > 0:
            speedup = perf_seq['total_time_ms'] / perf_par['total_time_ms']
            print(f"\nSpeedup: {speedup:.2f}x")
            print(f"Time saved: {perf_seq['total_time_ms'] - perf_par['total_time_ms']:.1f}ms")


def demo_multiple_models():
    """Demonstrate parallelization with multiple models."""
    print("\n" + "=" * 70)
    print("DEMO 2: MULTIPLE MODELS - PARALLELIZATION BENEFIT")
    print("=" * 70)

    # Create diverse section dataset
    df = pd.DataFrame({
        'text': SAMPLE_TEXTS * 3,  # 30 texts
        'section_name': (
            ['Business'] * len(SAMPLE_TEXTS) +
            ['Politics'] * len(SAMPLE_TEXTS) +
            ['Sports'] * len(SAMPLE_TEXTS)
        )
    })

    print(f"\nDataset: {len(df)} articles")
    print("Sections: Business, Politics, Sports (different models needed)")

    # Sequential
    print("\n--- Sequential Execution ---")
    result_seq, perf_seq = batch_infer(
        df,
        text_col='text',
        auto_select_models=True,
        batch_size=32,
        verbose=False,
        parallelize=False,
        measure_performance=True
    )

    if perf_seq:
        print(f"Models used: {perf_seq['models_count']}")
        print(f"Total time: {perf_seq['total_time_ms']:.1f}ms")
        print("Model breakdown:")
        for model_key, timing in perf_seq['model_timings'].items():
            print(f"  {model_key:20s}: {timing.get('total_ms', 0):8.1f}ms")

    # Parallel
    print("\n--- Parallel Execution ---")
    result_par, perf_par = batch_infer(
        df,
        text_col='text',
        auto_select_models=True,
        batch_size=32,
        verbose=False,
        parallelize=True,
        execution_strategy='auto',
        measure_performance=True
    )

    if perf_par:
        print(f"Strategy: {perf_par['execution_strategy']}")
        print(f"Models used: {perf_par['models_count']}")
        print(f"Total time: {perf_par['total_time_ms']:.1f}ms")

        if perf_seq and perf_seq['total_time_ms'] > 0:
            speedup = perf_seq['total_time_ms'] / perf_par['total_time_ms']
            saved = perf_seq['total_time_ms'] - perf_par['total_time_ms']
            print(f"\n-> Speedup: {speedup:.2f}x faster")
            print(f"-> Time saved: {saved:.1f}ms ({saved/perf_seq['total_time_ms']*100:.1f}%)")


def demo_convenience_function():
    """Demonstrate the high-level parallel interface."""
    print("\n" + "=" * 70)
    print("DEMO 3: CONVENIENCE FUNCTION - batch_infer_parallel()")
    print("=" * 70)

    df = pd.DataFrame({
        'text': SAMPLE_TEXTS * 4,
        'section_name': ['Business', 'Politics', 'World', 'Sports'] * len(SAMPLE_TEXTS)
    })

    print(f"\nDataset: {len(df)} articles")
    print("Using batch_infer_parallel() - high-level interface")

    result_df, metrics = batch_infer_parallel(
        df,
        text_col='text',
        verbose=False
    )

    print(f"\nResults:")
    print(f"  Strategy: {metrics['execution_strategy']}")
    print(f"  Total time: {metrics['total_time_ms']:.1f}ms")
    print(f"  Throughput: {metrics['texts_count'] / (metrics['total_time_ms'] / 1000):.1f} texts/sec")

    # Show sample results
    print(f"\nSample results (first 3 rows):")
    label_cols = [col for col in result_df.columns if col.endswith('_label')]
    print(result_df[['text'] + label_cols].head(3))


def demo_device_allocation():
    """Demonstrate device allocation across workers."""
    print("\n" + "=" * 70)
    print("DEMO 4: DEVICE ALLOCATION")
    print("=" * 70)

    from src.models.device_manager import DeviceManager

    manager = DeviceManager(prefer_gpu=True)

    print("\nDevice Manager Info:")
    info = manager.get_device_info()
    print(f"  CUDA available: {info['cuda_available']}")
    print(f"  Number of GPUs: {info['num_gpus']}")

    models = ['finbert', 'finbert_tone', 'distilroberta', 'roberta_general', 'polibert']
    allocation = manager.allocate_devices_by_name(models)

    print(f"\nDevice allocation for {len(models)} models:")
    for model_key, device in allocation.items():
        print(f"  {model_key:20s} -> {device}")

    # Check if parallelization is viable
    can_run, reason = manager.can_parallelize(len(models))
    print(f"\nCan parallelize: {can_run}")
    print(f"Reason: {reason}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "Sentiment Analysis Parallelization - Demonstration".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        # Demo 1: Single inference
        demo_single_inference()

        # Demo 2: Multiple models
        demo_multiple_models()

        # Demo 3: Convenience function
        demo_convenience_function()

        # Demo 4: Device allocation
        demo_device_allocation()

        # Summary
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETE")
        print("=" * 70)

        print("\nKey Takeaways:")
        print("  1. Parallelization improves performance on multi-model tasks")
        print("  2. Strategy auto-selection handles GPU/CPU environments")
        print("  3. batch_infer_parallel() provides easy-to-use interface")
        print("  4. Device allocation optimizes multi-GPU setups")

        print("\nNext Steps:")
        print("  - Run benchmark_sentiment_models() for detailed timing analysis")
        print("  - Use compare_execution_strategies() for quick performance comparison")
        print("  - Set parallelize=True in batch_infer() for production use")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
