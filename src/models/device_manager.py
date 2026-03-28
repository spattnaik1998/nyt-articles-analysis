"""
Device Management for Parallel Sentiment Analysis

Handles allocation of CUDA devices across worker processes and threads.
Ensures safe multi-GPU execution without conflicts.
"""

import logging
from typing import List, Optional, Dict
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage GPU device allocation across worker processes."""

    def __init__(self, prefer_gpu: bool = True):
        """
        Initialize device manager.

        Args:
            prefer_gpu: If True, try to use GPU. If False, force CPU.
        """
        self.prefer_gpu = prefer_gpu
        self.has_cuda = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.has_cuda else 0

        if self.has_cuda and prefer_gpu:
            logger.info(f"DeviceManager initialized: {self.num_gpus} GPU(s) available")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f}GB)")
        else:
            logger.info("DeviceManager initialized: Using CPU")

    def allocate_devices(
        self,
        model_count: int,
        use_processes: bool = True
    ) -> Dict[int, str]:
        """
        Allocate devices to models.

        Returns mapping of model index to device string.

        Args:
            model_count: Number of models to run
            use_processes: If True, distribute across GPUs. If False, use single device.

        Returns:
            Dict: {model_index -> device_string}
                Example: {0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:0', ...}
        """
        allocation = {}

        if not self.has_cuda or not self.prefer_gpu:
            # CPU-only: all models use CPU
            for i in range(model_count):
                allocation[i] = 'cpu'
            logger.info(f"Allocated {model_count} models to CPU")
            return allocation

        if use_processes:
            # Distribute models across GPUs (round-robin)
            for i in range(model_count):
                gpu_id = i % self.num_gpus
                allocation[i] = f'cuda:{gpu_id}'
                logger.info(f"  Model {i} -> cuda:{gpu_id}")
        else:
            # All models share single best GPU
            best_gpu = self._find_best_gpu()
            for i in range(model_count):
                allocation[i] = f'cuda:{best_gpu}'
            logger.info(f"Allocated {model_count} models to cuda:{best_gpu}")

        return allocation

    def allocate_devices_by_name(
        self,
        model_keys: List[str],
        use_processes: bool = True
    ) -> Dict[str, str]:
        """
        Allocate devices to models, indexed by model key instead of index.

        Args:
            model_keys: List of model keys (e.g., ['finbert', 'roberta_general'])
            use_processes: If True, distribute across GPUs

        Returns:
            Dict: {model_key -> device_string}
                Example: {'finbert': 'cuda:0', 'roberta_general': 'cuda:1'}
        """
        index_allocation = self.allocate_devices(len(model_keys), use_processes)

        allocation = {}
        for idx, model_key in enumerate(model_keys):
            allocation[model_key] = index_allocation[idx]

        return allocation

    def _find_best_gpu(self) -> int:
        """
        Find GPU with most available memory.

        Returns:
            int: GPU index with most free memory
        """
        if not self.has_cuda:
            return 0

        max_free = 0
        best_gpu = 0

        for i in range(self.num_gpus):
            try:
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory
                reserved_mem = torch.cuda.memory_reserved(i)
                free_mem = total_mem - reserved_mem

                if free_mem > max_free:
                    max_free = free_mem
                    best_gpu = i
            except Exception as e:
                logger.warning(f"Could not check memory for GPU {i}: {e}")

        return best_gpu

    def cleanup_device(self, device: str) -> None:
        """
        Clean up device memory (primarily for CUDA).

        Args:
            device: Device string ('cuda:0', 'cuda', 'cpu', etc.)
        """
        if 'cuda' in device.lower():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug(f"Cleaned up {device}")
            except Exception as e:
                logger.warning(f"Could not cleanup {device}: {e}")

    def get_device_info(self) -> Dict:
        """
        Get information about available devices.

        Returns:
            Dict: Device configuration and availability
        """
        info = {
            'cuda_available': self.has_cuda,
            'prefer_gpu': self.prefer_gpu,
            'num_gpus': self.num_gpus,
            'gpus': []
        }

        if self.has_cuda:
            for i in range(self.num_gpus):
                try:
                    props = torch.cuda.get_device_properties(i)
                    total_mem = props.total_memory / 1024 / 1024 / 1024

                    info['gpus'].append({
                        'index': i,
                        'name': torch.cuda.get_device_name(i),
                        'total_memory_gb': total_mem,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })
                except Exception as e:
                    logger.warning(f"Could not get info for GPU {i}: {e}")

        return info

    def estimate_memory_per_worker(self, model_size_mb: float = 400) -> float:
        """
        Estimate memory needed per worker.

        Includes model, tokenizer, batch tensors, and buffers.

        Args:
            model_size_mb: Size of single model in MB (default ~400MB for BERT)

        Returns:
            float: Estimated memory in MB per worker
        """
        # Model: 400MB (avg BERT)
        # Tokenizer: 10MB
        # Batch tensors (32 samples): 50MB
        # PyTorch overhead: 50MB
        # Total: ~500MB per worker (conservative)

        return model_size_mb + 100

    def can_parallelize(
        self,
        model_count: int,
        model_size_mb: float = 400,
        min_free_memory_gb: float = 3.0
    ) -> tuple[bool, str]:
        """
        Check if system has enough resources for parallelization.

        Args:
            model_count: Number of models to run in parallel
            model_size_mb: Size of each model in MB
            min_free_memory_gb: Minimum free memory needed in GB

        Returns:
            Tuple: (can_parallelize, reason_string)
        """
        if not self.has_cuda or not self.prefer_gpu:
            return (True, "Using CPU, memory constraints less critical")

        try:
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = props.total_memory / 1024 / 1024 / 1024

                if total_mem_gb < 4:
                    return (False, f"GPU {i} has only {total_mem_gb:.1f}GB (need >=4GB)")

            return (True, f"Sufficient GPU memory for {model_count} models")

        except Exception as e:
            logger.warning(f"Could not check memory: {e}")
            return (True, "Could not verify memory, attempting anyway")

    def get_worker_init_args(self, model_keys: List[str]) -> List[tuple]:
        """
        Get arguments to pass to worker initialization.

        For use with ProcessPoolExecutor's initializer parameter.

        Args:
            model_keys: List of model keys to allocate

        Returns:
            List[tuple]: List of (cuda_device_id,) tuples for each worker
        """
        if not self.has_cuda or not self.prefer_gpu:
            return [(None,)] * len(model_keys)

        # Round-robin allocation
        init_args = []
        for i in range(len(model_keys)):
            gpu_id = i % self.num_gpus
            init_args.append((gpu_id,))

        return init_args


def get_default_device() -> str:
    """
    Get recommended device for single-model inference.

    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


if __name__ == "__main__":
    # Test DeviceManager
    print("Testing DeviceManager...\n")

    manager = DeviceManager(prefer_gpu=True)

    # Get info
    print("Device Info:")
    info = manager.get_device_info()
    print(f"  CUDA available: {info['cuda_available']}")
    print(f"  Number of GPUs: {info['num_gpus']}")

    # Test allocation
    models = ['finbert', 'finbert_tone', 'distilroberta', 'roberta_general', 'polibert']
    allocation = manager.allocate_devices_by_name(models)

    print(f"\nDevice Allocation for {len(models)} models:")
    for model_key, device in allocation.items():
        print(f"  {model_key:20s} -> {device}")

    # Test can_parallelize
    can_run, reason = manager.can_parallelize(len(models))
    print(f"\nCan parallelize {len(models)} models: {can_run}")
    print(f"  Reason: {reason}")

    # Test memory estimation
    mem_per_worker = manager.estimate_memory_per_worker()
    print(f"\nEstimated memory per worker: {mem_per_worker:.0f}MB")
