"""
BERTweet model singleton for the embedding service.

Loads vinai/bertweet-base exactly once, thread-safely, on first request.
All subsequent calls reuse the same model instance — no per-request loading.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "vinai/bertweet-base"

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_tokenizer = None
_model     = None
_device: Optional[torch.device] = None
_load_lock = threading.Lock()
_load_error: Optional[str] = None


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("[MODEL] Using CUDA GPU: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("[MODEL] Using Apple MPS GPU")
    else:
        dev = torch.device("cpu")
        logger.info("[MODEL] Using CPU")
    return dev


def get_model() -> Tuple[AutoTokenizer, AutoModel]:
    """
    Return (tokenizer, model) for inference, loading them on first call.
    Thread-safe via double-checked locking.

    Raises RuntimeError if loading failed previously (cached error).
    """
    global _tokenizer, _model, _device, _load_error

    if _model is not None:
        return _tokenizer, _model

    if _load_error:
        raise RuntimeError(_load_error)

    with _load_lock:
        if _model is not None:
            return _tokenizer, _model
        if _load_error:
            raise RuntimeError(_load_error)

        try:
            t0 = time.time()
            logger.info("[MODEL] Loading %s …", MODEL_NAME)
            _device    = _detect_device()
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model     = AutoModel.from_pretrained(MODEL_NAME)
            _model.to(_device)
            _model.eval()
            logger.info("[MODEL] Loaded in %.1fs on %s", time.time() - t0, _device)
        except Exception as exc:
            _load_error = f"Model load failed: {exc}"
            raise RuntimeError(_load_error) from exc

    return _tokenizer, _model


def is_loaded() -> bool:
    return _model is not None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def encode(
    query: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    max_length: int = 128,
    pooling: str = "cls",
) -> np.ndarray:
    """
    Encode *query* into a float32 embedding vector.

    Parameters
    ----------
    query:      Text to encode.
    tokenizer:  HuggingFace tokenizer.
    model:      HuggingFace model (eval mode, on correct device).
    max_length: Tokeniser truncation limit.
    pooling:    ``"cls"`` (first token) or ``"mean"`` (average over tokens).

    Returns
    -------
    np.ndarray  shape (hidden_size,), dtype float32
    """
    if pooling not in ("cls", "mean"):
        raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")

    device = next(model.parameters()).device
    encoded = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    hs = outputs.last_hidden_state  # (1, seq_len, hidden)

    if pooling == "cls":
        emb = hs[:, 0, :].cpu().numpy()[0]
    else:  # mean
        mask = encoded["attention_mask"].unsqueeze(-1).expand(hs.size()).float()
        emb  = (torch.sum(hs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)).cpu().numpy()[0]

    return emb.astype(np.float32)


def warmup() -> None:
    """Pre-load model so the first real request is not penalised."""
    try:
        tok, mdl = get_model()
        # Single forward pass to initialise CUDA kernels etc.
        encode("warmup", tok, mdl)
        logger.info("[MODEL] Warmup complete")
    except Exception as exc:
        logger.warning("[MODEL] Warmup failed: %s", exc)
