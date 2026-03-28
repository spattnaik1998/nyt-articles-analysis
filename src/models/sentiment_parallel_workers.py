"""
Sentiment Analysis Worker Functions for Parallel Execution

These worker functions are designed to run in separate processes (ProcessPoolExecutor)
or threads (ThreadPoolExecutor). Each worker loads a single model and processes
all texts, then cleans up resources.

Key Design:
- Pickleable: No closures or non-pickleable objects
- Idempotent: Can be called multiple times independently
- Resource cleanup: Explicit cleanup of model and device memory
- Error handling: Graceful error messages for debugging
"""

import logging
import time
from typing import List, Tuple, Dict, Optional
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Module-level cache for models within this process
_MODEL_CACHE = {}
_DEVICE_INITIALIZED = False


def _init_worker_process(cuda_device_id: Optional[int] = None) -> None:
    """
    Initialize worker process (called once per worker in ProcessPoolExecutor).

    Sets up CUDA context and disables gradients globally for faster inference.

    Args:
        cuda_device_id: GPU device ID (0, 1, etc.) or None for CPU
    """
    global _DEVICE_INITIALIZED

    try:
        # Disable gradient computation globally (faster inference)
        torch.set_grad_enabled(False)

        # Set CUDA device if available and specified
        if cuda_device_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(cuda_device_id)
            torch.cuda.empty_cache()
            logger.debug(f"Worker process initialized on CUDA device {cuda_device_id}")

        _DEVICE_INITIALIZED = True
    except Exception as e:
        logger.warning(f"Worker initialization failed: {e}")
        _DEVICE_INITIALIZED = False


def _load_model_once(model_key: str, device: str) -> Tuple:
    """
    Load model once per process and cache it.

    Uses module-level cache to avoid reloading same model multiple times
    within the same worker process.

    Args:
        model_key: Model key (e.g., 'finbert', 'roberta_general')
        device: Device ('cuda', 'cuda:0', 'cpu')

    Returns:
        Tuple: (model, tokenizer, device_used)

    Raises:
        ImportError: If transformers not available
        ValueError: If model_key not recognized
    """
    global _MODEL_CACHE

    # Check cache first
    cache_key = f"{model_key}_{device}"
    if cache_key in _MODEL_CACHE:
        logger.debug(f"Using cached model: {model_key} on {device}")
        return _MODEL_CACHE[cache_key]

    # Load model
    from src.models.sentiment import load_sentiment_model

    logger.debug(f"Loading model: {model_key} on {device}")
    model, tokenizer, device_used = load_sentiment_model(model_key, device)

    # Cache it
    _MODEL_CACHE[cache_key] = (model, tokenizer, device_used)

    return model, tokenizer, device_used


def worker_infer_single_model(
    texts: List[str],
    model_key: str,
    batch_size: int = 32,
    device: str = 'cpu',
    max_length: int = 512
) -> Tuple[str, List[str], List[float], Dict]:
    """
    Worker function: Load single model and infer on all texts.

    This is designed to run in a separate process/thread. It:
    1. Loads the specified model once
    2. Processes all texts in batches
    3. Cleans up resources
    4. Returns results and timing

    Args:
        texts: List of text strings to classify
        model_key: Model to use ('finbert', 'roberta_general', etc.)
        batch_size: Batch size for processing
        device: Device to use ('cpu', 'cuda', 'cuda:0', etc.)
        max_length: Max sequence length for tokenizer

    Returns:
        Tuple: (model_key, labels_list, scores_list, timing_dict)
            - model_key: The model that was used
            - labels_list: List of label strings
            - scores_list: List of confidence scores
            - timing_dict: {'inference_ms': float, 'total_ms': float}

    Raises:
        ValueError: If model_key not found
        RuntimeError: If inference fails
    """
    t_start = time.time()

    try:
        # Load model (cached if already loaded in this process)
        model, tokenizer, device_used = _load_model_once(model_key, device)

        t_model_load = time.time()

        labels = []
        scores = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Clear GPU cache before each batch
            if device_used == 'cuda' or 'cuda' in device_used:
                torch.cuda.empty_cache()

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

                # Get label name from model config
                label = model.config.id2label.get(pred_class, f"label_{pred_class}")

                labels.append(label)
                scores.append(float(pred_score))

        # Cleanup
        t_inference = time.time() - t_model_load

        if device_used == 'cuda' or 'cuda' in device_used:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        del model

        if device_used == 'cuda' or 'cuda' in device_used:
            torch.cuda.empty_cache()

        # Return results with timing
        t_total = time.time() - t_start
        timing_dict = {
            'inference_ms': t_inference * 1000,
            'total_ms': t_total * 1000
        }

        logger.debug(f"Model {model_key} completed: {len(texts)} texts in {t_total*1000:.1f}ms")

        return (model_key, labels, scores, timing_dict)

    except Exception as e:
        logger.error(f"Worker failed for model {model_key}: {e}")
        raise


def worker_infer_single_model_with_memory_tracking(
    texts: List[str],
    model_key: str,
    batch_size: int = 32,
    device: str = 'cpu',
    max_length: int = 512
) -> Tuple[str, List[str], List[float], Dict]:
    """
    Enhanced worker function with memory tracking.

    Same as worker_infer_single_model but includes peak memory usage in timing.

    Args:
        texts: List of text strings to classify
        model_key: Model to use
        batch_size: Batch size for processing
        device: Device to use
        max_length: Max sequence length

    Returns:
        Tuple: (model_key, labels, scores, timing_with_memory)
            - timing_with_memory includes: {'inference_ms', 'total_ms', 'peak_memory_mb'}
    """
    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False

    if has_psutil:
        process = psutil.Process()
        mem_start = process.memory_info().rss / 1024 / 1024

    model_key, labels, scores, timing_dict = worker_infer_single_model(
        texts, model_key, batch_size, device, max_length
    )

    if has_psutil:
        mem_peak = process.memory_info().rss / 1024 / 1024
        mem_used = mem_peak - mem_start
        timing_dict['peak_memory_mb'] = mem_peak
        timing_dict['memory_used_mb'] = mem_used

    return (model_key, labels, scores, timing_dict)


def validate_worker_environment() -> Dict[str, bool]:
    """
    Validate that worker environment has required dependencies.

    Returns:
        Dict: {'torch': bool, 'transformers': bool, 'cuda_available': bool}
    """
    try:
        import torch
        torch_available = True
    except ImportError:
        torch_available = False

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        transformers_available = True
    except ImportError:
        transformers_available = False

    cuda_available = torch.cuda.is_available() if torch_available else False

    return {
        'torch': torch_available,
        'transformers': transformers_available,
        'cuda_available': cuda_available
    }


if __name__ == "__main__":
    # Test that worker functions are pickleable
    import pickle

    print("Testing worker function pickling...")

    try:
        pickle.dumps(worker_infer_single_model)
        print("[OK] worker_infer_single_model is pickleable")
    except Exception as e:
        print(f"[ERROR] worker_infer_single_model not pickleable: {e}")

    # Test environment
    print("\nEnvironment validation:")
    env = validate_worker_environment()
    for key, value in env.items():
        status = "[OK]" if value else "[MISSING]"
        print(f"  {status} {key}: {value}")
