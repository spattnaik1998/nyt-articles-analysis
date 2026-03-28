"""
BERTweet Embeddings Generation

This module provides functions for generating sentence embeddings from NYT articles
using BERTweet or other transformer models.
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Tuple, Union
import logging
from tqdm import tqdm

# Transformers imports with error handling
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers library not available. Install with: pip install transformers")

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: Best available device

    Example:
        >>> device = get_device()
        >>> print(device)
        device(type='cuda')  # or 'cpu' if no GPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS GPU")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (consider using GPU for faster processing)")

    return device


def extract_embeddings_batch(
    texts: list,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = 128,
    pooling: str = 'cls'
) -> np.ndarray:
    """
    Extract embeddings for a batch of texts.

    Args:
        texts (list): List of text strings
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device (torch.device): Device to run inference on
        max_length (int): Maximum sequence length (default: 128)
        pooling (str): Pooling strategy - 'cls' or 'mean' (default: 'cls')

    Returns:
        np.ndarray: Embeddings array of shape (batch_size, embedding_dim)

    Example:
        >>> embeddings = extract_embeddings_batch(
        ...     ['text 1', 'text 2'],
        ...     tokenizer, model, device
        ... )
        >>> embeddings.shape
        (2, 768)
    """
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    # Move to device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**encoded)

    # Extract embeddings based on pooling strategy
    if pooling == 'cls':
        # Use [CLS] token embedding (first token)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    elif pooling == 'mean':
        # Mean pooling over all tokens
        attention_mask = encoded['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}. Use 'cls' or 'mean'.")

    return embeddings


def build_bertweet_embeddings(
    df: pd.DataFrame,
    text_col: str = 'combined_text',
    model_name: str = 'vinai/bertweet-base',
    sample_limit: Optional[int] = None,
    batch_size: int = 32,
    max_length: int = 128,
    pooling: str = 'cls',
    output_dir: str = 'data',
    save_embeddings: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build BERTweet embeddings for a DataFrame of articles.

    This function:
    1. Loads the BERTweet model and tokenizer
    2. Extracts embeddings from the specified text column
    3. Saves embeddings to data/embeddings.npy
    4. Saves ID-to-index mapping to data/embeddings_mapping.csv

    Args:
        df (pd.DataFrame): Input DataFrame with text data
        text_col (str): Column name containing text (default: 'combined_text')
        model_name (str): HuggingFace model name (default: 'vinai/bertweet-base')
        sample_limit (int, optional): Limit processing to N rows (default: None = all)
        batch_size (int): Batch size for processing (default: 32)
        max_length (int): Maximum sequence length (default: 128)
        pooling (str): Pooling strategy - 'cls' or 'mean' (default: 'cls')
        output_dir (str): Directory to save outputs (default: 'data')
        save_embeddings (bool): Whether to save to disk (default: True)
        device (torch.device, optional): Device to use (default: auto-detect)
        verbose (bool): Show progress bar (default: True)

    Returns:
        Tuple[np.ndarray, pd.DataFrame]:
            - embeddings: NumPy array of shape (n_samples, embedding_dim)
            - mapping_df: DataFrame with columns ['_id', 'index']

    Raises:
        ValueError: If text_col not in DataFrame
        ImportError: If transformers not installed

    Example:
        >>> df = pd.read_parquet('data/preprocessed.parquet')
        >>> embeddings, mapping = build_bertweet_embeddings(df, sample_limit=100)
        >>> embeddings.shape
        (100, 768)
        >>> mapping.head()
             _id  index
        0  nyt://article/001      0
        1  nyt://article/002      1
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers library is required. Install with: pip install transformers torch"
        )

    # Validate inputs
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame. Available: {list(df.columns)}")

    # Sample if requested
    if sample_limit is not None:
        df = df.head(sample_limit).copy()
        if verbose:
            logger.info(f"Processing sample of {len(df)} articles")
    else:
        if verbose:
            logger.info(f"Processing all {len(df)} articles")

    # Get device
    if device is None:
        device = get_device()

    # Load model and tokenizer
    if verbose:
        logger.info(f"Loading model: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()  # Set to evaluation mode
        if verbose:
            logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Prepare texts
    texts = df[text_col].fillna('').astype(str).tolist()

    # Get embedding dimension from model config
    try:
        embedding_dim = model.config.hidden_size
    except AttributeError:
        embedding_dim = 768  # Default fallback for BERTweet
        logger.warning(f"Could not get embedding dimension from model, using default: {embedding_dim}")

    # Extract embeddings in batches
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    if verbose:
        logger.info(f"Extracting embeddings (batch_size={batch_size}, pooling={pooling}, dim={embedding_dim})...")
        pbar = tqdm(total=len(texts), desc="Generating embeddings")

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        try:
            batch_embeddings = extract_embeddings_batch(
                batch_texts,
                tokenizer,
                model,
                device,
                max_length=max_length,
                pooling=pooling
            )
            all_embeddings.append(batch_embeddings)

            if verbose:
                pbar.update(len(batch_texts))

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            # Create zero embeddings for failed batch using actual model dimension
            batch_embeddings = np.zeros((len(batch_texts), embedding_dim))
            all_embeddings.append(batch_embeddings)

    if verbose:
        pbar.close()

    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)

    if verbose:
        logger.info(f"Embeddings shape: {embeddings.shape}")

    # Create ID-to-index mapping
    if '_id' in df.columns:
        mapping_df = pd.DataFrame({
            '_id': df['_id'].values,
            'index': np.arange(len(df))
        })
    else:
        logger.warning("Column '_id' not found. Using DataFrame index.")
        mapping_df = pd.DataFrame({
            '_id': df.index.values,
            'index': np.arange(len(df))
        })

    # Save to disk if requested
    if save_embeddings:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        embeddings_file = output_path / 'embeddings.npy'
        np.save(embeddings_file, embeddings)
        if verbose:
            file_size = embeddings_file.stat().st_size / (1024 * 1024)
            logger.info(f"Saved embeddings to {embeddings_file} ({file_size:.2f} MB)")

        # Save mapping
        mapping_file = output_path / 'embeddings_mapping.csv'
        mapping_df.to_csv(mapping_file, index=False)
        if verbose:
            logger.info(f"Saved ID mapping to {mapping_file}")

    return embeddings, mapping_df


def load_embeddings(
    embeddings_path: str = 'data/embeddings.npy',
    mapping_path: str = 'data/embeddings_mapping.csv'
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load pre-computed embeddings and their ID mapping.

    Args:
        embeddings_path (str): Path to embeddings .npy file
        mapping_path (str): Path to mapping CSV file

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: Embeddings array and mapping DataFrame

    Example:
        >>> embeddings, mapping = load_embeddings()
        >>> embeddings.shape
        (10000, 768)
        >>> mapping.head()
    """
    embeddings = np.load(embeddings_path)
    mapping = pd.read_csv(mapping_path)

    logger.info(f"Loaded embeddings: {embeddings.shape}")
    logger.info(f"Loaded mapping: {len(mapping)} entries")

    return embeddings, mapping


def get_embedding_by_id(
    article_id: str,
    embeddings: np.ndarray,
    mapping: pd.DataFrame
) -> Optional[np.ndarray]:
    """
    Retrieve embedding for a specific article ID.

    Args:
        article_id (str): Article ID
        embeddings (np.ndarray): Embeddings array
        mapping (pd.DataFrame): ID-to-index mapping

    Returns:
        np.ndarray or None: Embedding vector or None if not found

    Example:
        >>> embedding = get_embedding_by_id('nyt://article/001', embeddings, mapping)
        >>> embedding.shape
        (768,)
    """
    match = mapping[mapping['_id'] == article_id]

    if match.empty:
        logger.warning(f"Article ID '{article_id}' not found in mapping")
        return None

    idx = match.iloc[0]['index']
    return embeddings[idx]


if __name__ == "__main__":
    # Example usage
    print("BERTweet Embeddings Module")
    print("=" * 60)

    # Create sample data
    sample_df = pd.DataFrame({
        '_id': ['nyt://article/001', 'nyt://article/002', 'nyt://article/003'],
        'combined_text': [
            'Breaking news from New York',
            'Stock market sees major gains',
            'World leaders meet for climate summit'
        ]
    })

    print("\nSample DataFrame:")
    print(sample_df)

    print("\n\nGenerating embeddings...")
    embeddings, mapping = build_bertweet_embeddings(
        sample_df,
        sample_limit=3,
        batch_size=2,
        save_embeddings=False,
        verbose=True
    )

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Mapping:\n{mapping}")

    # Test retrieval
    print("\n\nRetrieving embedding for 'nyt://article/001':")
    emb = get_embedding_by_id('nyt://article/001', embeddings, mapping)
    print(f"Embedding shape: {emb.shape}")
    print(f"First 10 values: {emb[:10]}")
