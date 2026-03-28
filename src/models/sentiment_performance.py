"""
Sentiment Analysis Performance Measurement and Reporting

Utilities for benchmarking sentiment analysis execution strategies
and generating performance reports.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.sentiment import batch_infer

logger = logging.getLogger(__name__)


class SentimentPerformanceMetrics:
    """Track and analyze sentiment analysis performance."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.executions = []

    def record_execution(
        self,
        strategy: str,
        total_time_ms: float,
        texts_count: int,
        models_count: int,
        model_timings: Optional[Dict] = None
    ) -> None:
        """
        Record an execution run.

        Args:
            strategy: Execution strategy ('sequential', 'thread', 'process')
            total_time_ms: Total execution time in milliseconds
            texts_count: Number of texts classified
            models_count: Number of models used
            model_timings: Per-model timing breakdowns
        """
        execution = {
            'strategy': strategy,
            'total_time_ms': total_time_ms,
            'texts_count': texts_count,
            'models_count': models_count,
            'model_timings': model_timings or {},
            'throughput_texts_per_sec': (texts_count / (total_time_ms / 1000))
                if total_time_ms > 0 else 0,
            'throughput_models_per_sec': (models_count / (total_time_ms / 1000))
                if total_time_ms > 0 else 0
        }
        self.executions.append(execution)

    def get_speedup(self, baseline_strategy: str = 'sequential') -> Dict[str, float]:
        """
        Calculate speedup compared to baseline strategy.

        Args:
            baseline_strategy: Strategy to use as baseline (default: 'sequential')

        Returns:
            Dict: {strategy: speedup_factor}
        """
        baseline = None

        for exec in self.executions:
            if exec['strategy'] == baseline_strategy:
                baseline = exec
                break

        if baseline is None:
            logger.warning(f"Baseline strategy '{baseline_strategy}' not found")
            return {}

        speedups = {}
        for exec in self.executions:
            if exec['strategy'] != baseline_strategy:
                speedup = baseline['total_time_ms'] / exec['total_time_ms']
                speedups[exec['strategy']] = speedup

        return speedups

    def generate_report(self) -> str:
        """
        Generate formatted performance report.

        Returns:
            str: Formatted report text
        """
        if not self.executions:
            return "No execution data available"

        lines = []
        lines.append("=" * 70)
        lines.append("SENTIMENT ANALYSIS PERFORMANCE REPORT")
        lines.append("=" * 70)

        # Summary of executions
        lines.append("\nExecution Summary:")
        lines.append("-" * 70)

        for exec in self.executions:
            lines.append(f"\nStrategy: {exec['strategy'].upper()}")
            lines.append(f"  Total time: {exec['total_time_ms']:.1f}ms")
            lines.append(f"  Texts: {exec['texts_count']:,}")
            lines.append(f"  Models: {exec['models_count']}")
            lines.append(f"  Throughput: {exec['throughput_texts_per_sec']:.1f} texts/sec")

        # Speedup analysis
        speedups = self.get_speedup('sequential')

        if speedups:
            lines.append("\n" + "-" * 70)
            lines.append("Speedup vs Sequential:")
            lines.append("-" * 70)

            for strategy, speedup in sorted(speedups.items(), key=lambda x: -x[1]):
                lines.append(f"  {strategy:15s}: {speedup:6.2f}x faster")

        # Per-model breakdown (if available)
        first_exec = self.executions[0]
        if first_exec['model_timings']:
            lines.append("\n" + "-" * 70)
            lines.append("Per-Model Timing (first execution):")
            lines.append("-" * 70)

            for model_key, timings in first_exec['model_timings'].items():
                if isinstance(timings, dict):
                    total = timings.get('total_ms', 0)
                    lines.append(f"  {model_key:20s}: {total:8.1f}ms")
                else:
                    lines.append(f"  {model_key:20s}: {timings}")

        lines.append("\n" + "=" * 70)

        return '\n'.join(lines)

    def summary(self) -> Dict:
        """
        Get summary statistics.

        Returns:
            Dict: Summary statistics
        """
        if not self.executions:
            return {}

        times = [e['total_time_ms'] for e in self.executions]
        throughputs = [e['throughput_texts_per_sec'] for e in self.executions]

        return {
            'total_executions': len(self.executions),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'avg_time_ms': np.mean(times),
            'max_throughput_texts_per_sec': max(throughputs),
            'fastest_strategy': self.executions[np.argmin(times)]['strategy']
        }


def benchmark_sentiment_models(
    df: pd.DataFrame,
    text_col: str = 'combined_text',
    models: Optional[List[str]] = None,
    batch_size: int = 32,
    runs: int = 1,
    strategies: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[Dict, str]:
    """
    Benchmark sentiment analysis with different execution strategies.

    Runs sequential and parallel execution on the same data and compares performance.

    Args:
        df: Input DataFrame
        text_col: Column containing text to classify
        models: List of model keys. If None, auto-selects.
        batch_size: Batch size for inference
        runs: Number of runs per strategy
        strategies: List of strategies to benchmark ('sequential', 'process', 'thread')
        verbose: Show progress info

    Returns:
        Tuple[Dict, str]: (results_dict, formatted_report_string)

    Example:
        >>> df = pd.DataFrame({'text': ['positive text', 'negative text', ...]})
        >>> results, report = benchmark_sentiment_models(df, runs=2)
        >>> print(report)
    """
    if strategies is None:
        strategies = ['sequential', 'thread']  # Skip process to avoid GPU issues in tests

    metrics = SentimentPerformanceMetrics()

    if verbose:
        logger.info(f"Benchmarking sentiment analysis")
        logger.info(f"  Dataset: {len(df):,} rows")
        logger.info(f"  Models: {models or 'auto-select'}")
        logger.info(f"  Strategies: {strategies}")
        logger.info(f"  Runs: {runs} per strategy")

    for strategy in strategies:
        times = []

        for run_idx in range(runs):
            try:
                logger.info(f"\nRunning {strategy} (run {run_idx+1}/{runs})...")

                t_start = time.time()

                result_df, perf = batch_infer(
                    df,
                    text_col=text_col,
                    models=models,
                    batch_size=batch_size,
                    verbose=False,
                    parallelize=(strategy != 'sequential'),
                    execution_strategy=strategy,
                    measure_performance=True
                )

                t_elapsed = (time.time() - t_start) * 1000
                times.append(t_elapsed)

                if perf:
                    logger.info(f"  Time: {t_elapsed:.1f}ms (models: {perf['models_count']})")

            except Exception as e:
                logger.warning(f"  Strategy {strategy} failed: {e}")

        if times:
            avg_time = np.mean(times)
            metrics.record_execution(
                strategy=strategy,
                total_time_ms=avg_time,
                texts_count=len(df),
                models_count=len(models) if models else 1
            )

    # Generate report
    report = metrics.generate_report()

    # Return structured results
    results = {
        'metrics': metrics,
        'summary': metrics.summary(),
        'speedups': metrics.get_speedup('sequential')
    }

    return results, report


def compare_execution_strategies(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    output_file: Optional[str] = None
) -> Dict:
    """
    Quick comparison of sequential vs parallel execution.

    Args:
        df: Input DataFrame
        models: List of model keys
        output_file: If provided, save report to file

    Returns:
        Dict: Comparison results with speedup and timing
    """
    logger.info("Quick performance comparison...")

    # Run sequential
    t_seq_start = time.time()
    seq_df, seq_perf = batch_infer(
        df,
        models=models,
        verbose=False,
        parallelize=False,
        measure_performance=True
    )
    t_seq = (time.time() - t_seq_start) * 1000

    # Run parallel
    t_par_start = time.time()
    par_df, par_perf = batch_infer(
        df,
        models=models,
        verbose=False,
        parallelize=True,
        execution_strategy='auto',
        measure_performance=True
    )
    t_par = (time.time() - t_par_start) * 1000

    # Calculate speedup
    speedup = t_seq / t_par if t_par > 0 else 0

    comparison = {
        'sequential_time_ms': t_seq,
        'parallel_time_ms': t_par,
        'speedup_factor': speedup,
        'time_saved_ms': t_seq - t_par,
        'execution_strategy': par_perf.get('execution_strategy', 'unknown')
        if par_perf else 'unknown',
        'texts_count': len(df),
        'models_used': models or 'auto-select'
    }

    # Format output
    output_lines = [
        "=" * 70,
        "SENTIMENT ANALYSIS PERFORMANCE COMPARISON",
        "=" * 70,
        f"",
        f"Dataset:     {len(df):,} texts",
        f"Models:      {models or 'auto-selected'}",
        f"",
        f"Sequential:  {t_seq:.1f}ms",
        f"Parallel:    {t_par:.1f}ms (strategy: {comparison['execution_strategy']})",
        f"",
        f"Speedup:     {speedup:.2f}x",
        f"Time saved:  {t_seq - t_par:.1f}ms ({(t_seq - t_par) / t_seq * 100:.1f}%)",
        f"",
        "=" * 70
    ]

    output_text = '\n'.join(output_lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)
        logger.info(f"Report saved to {output_file}")

    logger.info(output_text)

    return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Sentiment Performance Analysis Module")
    print("Use benchmark_sentiment_models() or compare_execution_strategies()")
    print("See docstrings for usage examples")
