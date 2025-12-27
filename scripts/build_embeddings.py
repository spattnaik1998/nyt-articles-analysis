#!/usr/bin/env python
"""
Build BERTweet Embeddings for NYT Articles

This script reads preprocessed articles and generates BERTweet embeddings,
saving them for downstream similarity search and recommendation tasks.

Usage:
    python scripts/build_embeddings.py --input data/preprocessed.parquet --sample 1000
    python scripts/build_embeddings.py --input data/preprocessed.parquet --all --gpu
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embeddings import build_bertweet_embeddings, get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for building embeddings from preprocessed articles."""
    parser = argparse.ArgumentParser(
        description='Build BERTweet embeddings for NYT articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings for 1000 sample articles
  python scripts/build_embeddings.py --input data/preprocessed.parquet --sample 1000

  # Generate embeddings for all articles with GPU
  python scripts/build_embeddings.py --input data/preprocessed.parquet --all --gpu

  # Use custom model
  python scripts/build_embeddings.py --input data/preprocessed.parquet --model sentence-transformers/all-MiniLM-L6-v2

  # Use mean pooling instead of CLS
  python scripts/build_embeddings.py --input data/preprocessed.parquet --pooling mean

  # Custom batch size for memory constraints
  python scripts/build_embeddings.py --input data/preprocessed.parquet --batch-size 16
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/preprocessed.parquet',
        help='Path to preprocessed Parquet file (default: data/preprocessed.parquet)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data',
        help='Output directory for embeddings (default: data)'
    )

    parser.add_argument(
        '--sample', '-s',
        type=int,
        help='Number of articles to process (default: all)'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Process all articles (ignore --sample)'
    )

    parser.add_argument(
        '--text-col',
        type=str,
        default='cleaned_text',
        help='Text column to use (default: cleaned_text)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='vinai/bertweet-base',
        help='HuggingFace model name (default: vinai/bertweet-base)'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for processing (default: 32, reduce if OOM)'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Maximum sequence length (default: 128)'
    )

    parser.add_argument(
        '--pooling',
        type=str,
        choices=['cls', 'mean'],
        default='cls',
        help='Pooling strategy: cls or mean (default: cls)'
    )

    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Force GPU usage (fail if not available)'
    )

    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage even if GPU available'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with progress bars'
    )

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.cpu:
        device = torch.device('cpu')
        logger.info("Forcing CPU usage")
    elif args.gpu:
        if not torch.cuda.is_available():
            logger.error("GPU requested but CUDA not available!")
            sys.exit(1)
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = get_device()

    logger.info("="*60)
    logger.info("BERTweet Embeddings Generation")
    logger.info("="*60)

    # Step 1: Load preprocessed data
    logger.info(f"\n[1/3] Loading preprocessed data from: {input_path}")
    try:
        df = pd.read_parquet(input_path)
        logger.info(f"✓ Loaded {len(df):,} articles")
        logger.info(f"  Columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load Parquet file: {e}")
        sys.exit(1)

    # Validate text column
    if args.text_col not in df.columns:
        logger.error(f"Text column '{args.text_col}' not found!")
        logger.info(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Check for empty text
    empty_count = df[args.text_col].isna().sum() + (df[args.text_col] == '').sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count:,} articles with empty text in '{args.text_col}'")

    # Determine sample size
    sample_limit = None if args.all else args.sample
    if sample_limit is None and len(df) > 100000:
        logger.warning(f"Processing {len(df):,} articles. This may take a while.")
        logger.warning("Consider using --sample to test on a smaller dataset first.")

    # Step 2: Generate embeddings
    logger.info(f"\n[2/3] Generating embeddings...")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Text column: {args.text_col}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Pooling: {args.pooling}")
    logger.info(f"  Device: {device}")

    if sample_limit:
        logger.info(f"  Processing: {sample_limit:,} articles (sample)")
    else:
        logger.info(f"  Processing: {len(df):,} articles (all)")

    try:
        embeddings, mapping = build_bertweet_embeddings(
            df,
            text_col=args.text_col,
            model_name=args.model,
            sample_limit=sample_limit,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
            output_dir=str(output_dir),
            save_embeddings=True,
            device=device,
            verbose=args.verbose or True
        )

        logger.info(f"✓ Generated embeddings: {embeddings.shape}")

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Summary and verification
    logger.info(f"\n[3/3] Summary")
    logger.info("="*60)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"  - embeddings.npy ({embeddings.shape})")
    logger.info(f"  - embeddings_mapping.csv ({len(mapping)} rows)")

    # File sizes
    embeddings_file = output_dir / 'embeddings.npy'
    mapping_file = output_dir / 'embeddings_mapping.csv'

    if embeddings_file.exists():
        size_mb = embeddings_file.stat().st_size / (1024 * 1024)
        logger.info(f"\nEmbeddings file: {size_mb:.2f} MB")

    if mapping_file.exists():
        size_kb = mapping_file.stat().st_size / 1024
        logger.info(f"Mapping file: {size_kb:.2f} KB")

    logger.info("\n✓ Embeddings generation complete!")

    # Usage instructions
    logger.info("\n" + "="*60)
    logger.info("Next Steps")
    logger.info("="*60)
    logger.info("\nLoad embeddings in Python:")
    logger.info(f"""
    from src.models.embeddings import load_embeddings
    embeddings, mapping = load_embeddings(
        '{embeddings_file}',
        '{mapping_file}'
    )
    print(embeddings.shape)
    """)

    logger.info("\nUse for similarity search:")
    logger.info("""
    from sklearn.metrics.pairwise import cosine_similarity
    # Find similar articles
    query_idx = 0
    similarities = cosine_similarity([embeddings[query_idx]], embeddings)[0]
    top_k = similarities.argsort()[-10:][::-1]
    print(f"Top similar articles: {top_k}")
    """)


if __name__ == "__main__":
    main()
