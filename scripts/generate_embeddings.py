"""
Generate Embeddings for NYT Dataset

This script generates BERTweet embeddings for the preprocessed NYT dataset.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embeddings import build_bertweet_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_embeddings(
    input_file: str = "data/preprocessed.parquet",
    output_dir: str = "data",
    sample_size: int = None,
    batch_size: int = 32,
    model_name: str = "vinai/bertweet-base"
):
    """
    Generate embeddings for the preprocessed dataset.

    Args:
        input_file (str): Path to preprocessed parquet file
        output_dir (str): Directory to save embeddings
        sample_size (int, optional): Process only N articles
        batch_size (int): Batch size for embedding generation
        model_name (str): HuggingFace model name
    """
    logger.info("="*80)
    logger.info("EMBEDDING GENERATION")
    logger.info("="*80)

    # Load preprocessed data
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run: python scripts/preprocess_data.py")
        sys.exit(1)

    logger.info(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} articles")

    # Check for required column
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'combined_text'

    if text_col not in df.columns:
        logger.error(f"Required column '{text_col}' not found!")
        logger.error(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    logger.info(f"Using text column: {text_col}")

    # Generate embeddings
    logger.info(f"\nGenerating embeddings with {model_name}...")
    logger.info(f"Batch size: {batch_size}")

    if sample_size:
        logger.info(f"Sample size: {sample_size:,}")

    embeddings, mapping = build_bertweet_embeddings(
        df=df,
        text_col=text_col,
        model_name=model_name,
        sample_limit=sample_size,
        batch_size=batch_size,
        output_dir=output_dir,
        save_embeddings=True,
        verbose=True
    )

    logger.info("\n" + "="*80)
    logger.info("EMBEDDING GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput files:")
    logger.info(f"  - Embeddings: {output_dir}/embeddings.npy")
    logger.info(f"  - Mapping: {output_dir}/embeddings_mapping.csv")
    logger.info(f"\nEmbeddings shape: {embeddings.shape}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Test search: python scripts/test_search.py")
    logger.info(f"  2. Run API: uvicorn src.api.app:app --reload")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for NYT dataset")
    parser.add_argument(
        '--input',
        type=str,
        default='data/preprocessed.parquet',
        help='Input preprocessed file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Process only N articles (for testing)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='vinai/bertweet-base',
        help='HuggingFace model name'
    )

    args = parser.parse_args()

    generate_embeddings(
        input_file=args.input,
        output_dir=args.output_dir,
        sample_size=args.sample,
        batch_size=args.batch_size,
        model_name=args.model
    )


if __name__ == "__main__":
    main()
