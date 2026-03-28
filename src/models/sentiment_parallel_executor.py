"""
Sentiment Model Parallel Executor

Implements three execution strategies:
1. ProcessPoolExecutor - True parallelism with isolated CUDA contexts (GPU-optimized)
2. ThreadPoolExecutor - Lightweight concurrency with shared device (CPU fallback)
3. Sequential - Single-threaded baseline (debugging/reference)

Includes automatic fallback strategy and comprehensive error handling.
"""

import logging
import time
import asyncio
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

from src.models.sentiment_parallel_workers import (
    worker_infer_single_model,
    _init_worker_process,
    validate_worker_environment
)
from src.models.device_manager import DeviceManager, get_default_device

logger = logging.getLogger(__name__)


class SentimentModelParallelExecutor:
    """
    Orchestrate parallel execution of sentiment models.

    Provides three execution strategies with intelligent fallback:
    - ProcessPoolExecutor: Best for GPU (true parallelism)
    - ThreadPoolExecutor: Better for CPU (shared device)
    - Sequential: Fallback (debugging)
    """

    def __init__(
        self,
        prefer_gpu: bool = True,
        max_workers: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize executor.

        Args:
            prefer_gpu: If True, try to use GPU for parallelization
            max_workers: Max workers for ProcessPoolExecutor. If None, auto-detect.
            verbose: Enable detailed logging
        """
        self.prefer_gpu = prefer_gpu
        self.device_manager = DeviceManager(prefer_gpu=prefer_gpu)
        self.verbose = verbose
        self.max_workers = max_workers or self._auto_detect_max_workers()

        if verbose:
            info = self.device_manager.get_device_info()
            logger.info(f"Executor initialized: {info['num_gpus']} GPU(s), max_workers={self.max_workers}")

    def _auto_detect_max_workers(self) -> int:
        """
        Auto-detect optimal number of workers based on available resources.

        For GPU: number of GPUs (or 4 if more GPUs)
        For CPU: min(4, cpu_count // 2)

        Returns:
            int: Recommended max_workers
        """
        if self.device_manager.has_cuda and self.prefer_gpu:
            # Use one worker per GPU, but cap at 4
            return min(self.device_manager.num_gpus or 4, 4)
        else:
            # CPU: use fewer workers to avoid contention
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            return min(max(2, cpu_count // 2), 4)

    def execute_parallel_process(
        self,
        texts: List[str],
        models: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        timeout_per_model: int = 600
    ) -> Dict[str, Tuple]:
        """
        Execute models in parallel using ProcessPoolExecutor (GPU-optimized).

        Each model runs in separate process with isolated CUDA context.

        Args:
            texts: List of texts to classify
            models: List of model keys to run
            batch_size: Batch size for inference
            max_length: Max token length
            timeout_per_model: Timeout in seconds per model

        Returns:
            Dict: {model_key: (labels, scores, timing_dict)}

        Raises:
            TimeoutError: If any model exceeds timeout
            RuntimeError: If all models fail and no fallback available
        """
        logger.info(f"Executing {len(models)} models in parallel using ProcessPoolExecutor")
        logger.debug(f"  Texts: {len(texts)}, Batch size: {batch_size}, Timeout: {timeout_per_model}s")

        t_start = time.time()

        # Allocate devices to models
        device_allocation = self.device_manager.allocate_devices_by_name(models)
        worker_init_args = self.device_manager.get_worker_init_args(models)

        results = {}
        failed_models = []

        try:
            with ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_init_worker_process,
                initargs=None  # Can add cuda_device_id if needed
            ) as executor:
                # Submit all jobs
                future_to_model = {}
                for model_key in models:
                    device = device_allocation[model_key]

                    future = executor.submit(
                        worker_infer_single_model,
                        texts,
                        model_key,
                        batch_size,
                        device,
                        max_length
                    )
                    future_to_model[future] = model_key

                    if self.verbose:
                        logger.debug(f"Submitted {model_key} to {device}")

                # Collect results as they complete
                for future in as_completed(future_to_model, timeout=timeout_per_model):
                    model_key = future_to_model[future]

                    try:
                        model_key_ret, labels, scores, timing = future.result()
                        results[model_key_ret] = (labels, scores, timing)

                        if self.verbose:
                            logger.info(f"[PARALLEL] {model_key} completed: {timing['total_ms']:.1f}ms")

                    except FuturesTimeoutError:
                        logger.error(f"Model {model_key} timed out after {timeout_per_model}s")
                        failed_models.append(model_key)
                    except Exception as e:
                        logger.error(f"Model {model_key} failed: {e}")
                        failed_models.append(model_key)

        except Exception as e:
            logger.error(f"ProcessPoolExecutor failed: {e}")
            raise

        # Log results
        t_total = time.time() - t_start
        completed = len(results)
        failed = len(failed_models)

        logger.info(f"Parallel execution complete: {completed}/{len(models)} models succeeded in {t_total*1000:.1f}ms")
        if failed_models:
            logger.warning(f"Failed models: {failed_models}")

        if not results:
            raise RuntimeError(f"All {len(models)} models failed in parallel execution")

        return results

    def execute_parallel_thread(
        self,
        texts: List[str],
        models: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        timeout_per_model: int = 600
    ) -> Dict[str, Tuple]:
        """
        Execute models in parallel using ThreadPoolExecutor (CPU fallback).

        All threads share the same device, coordination through locking.

        Args:
            texts: List of texts to classify
            models: List of model keys to run
            batch_size: Batch size for inference
            max_length: Max token length
            timeout_per_model: Timeout in seconds per model

        Returns:
            Dict: {model_key: (labels, scores, timing_dict)}
        """
        logger.info(f"Executing {len(models)} models in parallel using ThreadPoolExecutor")

        t_start = time.time()

        results = {}
        failed_models = []

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs (all using CPU or same GPU)
                device = 'cpu' if not self.prefer_gpu else get_default_device()
                future_to_model = {}

                for model_key in models:
                    future = executor.submit(
                        worker_infer_single_model,
                        texts,
                        model_key,
                        batch_size,
                        device,
                        max_length
                    )
                    future_to_model[future] = model_key

                    if self.verbose:
                        logger.debug(f"Submitted {model_key} to {device}")

                # Collect results
                for future in as_completed(future_to_model, timeout=timeout_per_model):
                    model_key = future_to_model[future]

                    try:
                        model_key_ret, labels, scores, timing = future.result()
                        results[model_key_ret] = (labels, scores, timing)

                        if self.verbose:
                            logger.info(f"[THREAD] {model_key} completed: {timing['total_ms']:.1f}ms")

                    except FuturesTimeoutError:
                        logger.error(f"Model {model_key} timed out after {timeout_per_model}s")
                        failed_models.append(model_key)
                    except Exception as e:
                        logger.error(f"Model {model_key} failed: {e}")
                        failed_models.append(model_key)

        except Exception as e:
            logger.error(f"ThreadPoolExecutor failed: {e}")
            raise

        t_total = time.time() - t_start
        logger.info(f"Thread execution complete: {len(results)}/{len(models)} models in {t_total*1000:.1f}ms")

        if not results:
            raise RuntimeError(f"All {len(models)} models failed in thread execution")

        return results

    def execute_sequential(
        self,
        texts: List[str],
        models: List[str],
        batch_size: int = 32,
        max_length: int = 512
    ) -> Dict[str, Tuple]:
        """
        Execute models sequentially (single-threaded, for baseline/debugging).

        This is the current bottleneck - included for performance comparison.

        Args:
            texts: List of texts to classify
            models: List of model keys to run
            batch_size: Batch size for inference
            max_length: Max token length

        Returns:
            Dict: {model_key: (labels, scores, timing_dict)}
        """
        logger.info(f"Executing {len(models)} models sequentially (baseline)")

        t_start = time.time()
        results = {}
        device = get_default_device()

        for model_key in models:
            try:
                t_model_start = time.time()

                model_key_ret, labels, scores, timing = worker_infer_single_model(
                    texts,
                    model_key,
                    batch_size,
                    device,
                    max_length
                )

                results[model_key_ret] = (labels, scores, timing)

                logger.info(f"[SEQUENTIAL] {model_key} completed: {timing['total_ms']:.1f}ms")

            except Exception as e:
                logger.error(f"Model {model_key} failed: {e}")

        t_total = time.time() - t_start
        logger.info(f"Sequential execution complete: {len(results)}/{len(models)} models in {t_total*1000:.1f}ms")

        return results

    def execute_with_fallback(
        self,
        texts: List[str],
        models: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        execution_strategy: str = 'auto'
    ) -> Tuple[Dict[str, Tuple], str]:
        """
        Execute with intelligent fallback strategy.

        Execution paths:
        - 'process' -> 'thread' -> 'sequential'
        - 'thread' -> 'sequential'
        - 'sequential' (no fallback)
        - 'auto' (auto-detect strategy)

        Args:
            texts: List of texts to classify
            models: List of model keys to run
            batch_size: Batch size for inference
            max_length: Max token length
            execution_strategy: 'process', 'thread', 'sequential', or 'auto'

        Returns:
            Tuple: (results_dict, execution_strategy_used)
        """
        # Auto-detect strategy if requested
        if execution_strategy == 'auto':
            execution_strategy = self._auto_detect_strategy(len(models))
            if self.verbose:
                logger.info(f"Auto-detected strategy: {execution_strategy}")

        strategies = {
            'process': self.execute_parallel_process,
            'thread': self.execute_parallel_thread,
            'sequential': self.execute_sequential
        }

        fallback_order = {
            'process': ['thread', 'sequential'],
            'thread': ['sequential'],
            'sequential': []
        }

        # Try primary strategy
        try:
            logger.info(f"Attempting execution strategy: {execution_strategy}")
            results = strategies[execution_strategy](texts, models, batch_size, max_length)
            return results, execution_strategy

        except Exception as e:
            logger.warning(f"Strategy {execution_strategy} failed: {e}")

            # Try fallback strategies
            for fallback_strategy in fallback_order.get(execution_strategy, []):
                try:
                    logger.info(f"Falling back to strategy: {fallback_strategy}")
                    results = strategies[fallback_strategy](texts, models, batch_size, max_length)
                    return results, fallback_strategy

                except Exception as e2:
                    logger.warning(f"Strategy {fallback_strategy} also failed: {e2}")

        # All strategies failed
        raise RuntimeError(f"All execution strategies failed for {len(models)} models")

    def _auto_detect_strategy(self, model_count: int) -> str:
        """
        Auto-detect best execution strategy based on system resources.

        Returns:
            str: 'process', 'thread', or 'sequential'
        """
        # Check if we have requirements
        env = validate_worker_environment()

        if not env['torch'] or not env['transformers']:
            logger.warning("Missing torch/transformers, using sequential")
            return 'sequential'

        # Prefer process for GPU, thread for CPU
        if env['cuda_available'] and self.prefer_gpu:
            return 'process'
        elif model_count <= 2:
            # For very few models, sequential is fine
            return 'sequential'
        else:
            # CPU with multiple models: use thread
            return 'thread'

    def benchmark(
        self,
        texts: List[str],
        models: List[str],
        batch_size: int = 32,
        runs: int = 3
    ) -> Dict[str, float]:
        """
        Benchmark execution strategies.

        Args:
            texts: List of texts to classify
            models: List of model keys to run
            batch_size: Batch size for inference
            runs: Number of runs per strategy

        Returns:
            Dict: {strategy: avg_time_ms}
        """
        logger.info(f"Benchmarking {len(models)} models with {len(texts)} texts, {runs} runs each")

        results = {}

        for strategy in ['sequential', 'thread', 'process']:
            times = []

            for run_idx in range(runs):
                try:
                    t_start = time.time()
                    self.execute_with_fallback(
                        texts, models, batch_size,
                        execution_strategy=strategy
                    )
                    t_elapsed = (time.time() - t_start) * 1000
                    times.append(t_elapsed)

                    logger.info(f"  {strategy:12s} run {run_idx+1}/{runs}: {t_elapsed:8.1f}ms")

                except Exception as e:
                    logger.warning(f"  {strategy:12s} run {run_idx+1}/{runs}: FAILED - {e}")

            if times:
                avg_time = np.mean(times)
                results[strategy] = avg_time

                speedup = results.get('sequential', 0) / avg_time if avg_time > 0 else 0
                logger.info(f"  {strategy:12s} average: {avg_time:8.1f}ms (speedup: {speedup:5.2f}x)")

        return results


def create_executor(
    prefer_gpu: bool = True,
    max_workers: Optional[int] = None,
    verbose: bool = True
) -> SentimentModelParallelExecutor:
    """
    Factory function to create executor with appropriate settings.

    Args:
        prefer_gpu: If True, try to use GPU
        max_workers: Max workers for parallelization
        verbose: Enable detailed logging

    Returns:
        SentimentModelParallelExecutor: Configured executor
    """
    return SentimentModelParallelExecutor(
        prefer_gpu=prefer_gpu,
        max_workers=max_workers,
        verbose=verbose
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    executor = create_executor(prefer_gpu=True, verbose=True)

    print("\nExecutor Info:")
    print(f"  Max workers: {executor.max_workers}")
    print(f"  Device info: {executor.device_manager.get_device_info()}")
