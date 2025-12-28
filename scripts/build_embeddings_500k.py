#!/usr/bin/env python
"""
Build BERTweet Embeddings for 500K NYT Articles

This script generates embeddings for the 500K article dataset with optimizations
for large-scale processing:
- Efficient batch processing
- Memory management
- Progress tracking
- Checkpointing support

Usage:
    python scripts/build_embeddings_500k.py
    python scripts/build_embeddings_500k.py --batch-size 64 --gpu
"""

import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
import torch
import numpy as np

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
    """Main function for building embeddings for 500K articles."""
    parser = argparse.ArgumentParser(
        description='Build BERTweet embeddings for 500K NYT articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings for all 500K articles
  python scripts/build_embeddings_500k.py

  # Use GPU with larger batch size
  python scripts/build_embeddings_500k.py --batch-size 64 --gpu

  # Use custom input file
  python scripts/build_embeddings_500k.py --input data/preprocessed_500K.parquet
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/preprocessed_500K.parquet',
        help='Path to preprocessed Parquet file (default: data/preprocessed_500K.parquet)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data',
        help='Output directory for embeddings (default: data)'
    )

    parser.add_argument(
        '--output-prefix',
        type=str,
        default='embeddings_500k',
        help='Prefix for output files (default: embeddings_500k)'
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
        help='Batch size for processing (default: 32, increase to 64 for GPU)'
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
        logger.error("Please run preprocessing first:")
        logger.error("  python scripts/preprocess_data.py --input data/nyt_articles_500K.csv --output data/preprocessed_500K.parquet")
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

    logger.info("="*80)
    logger.info("EMBEDDINGS GENERATION FOR 500K NYT ARTICLES")
    logger.info("="*80)

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

    # Memory estimation
    estimated_size = len(df) * 768 * 4 / (1024**2)  # 768 dims, 4 bytes per float
    logger.info(f"\nEstimated embedding size: ~{estimated_size:.2f} MB")

    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Available GPU memory: {gpu_memory:.2f} GB")

    # Step 2: Generate embeddings
    logger.info(f"\n[2/3] Generating embeddings...")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Text column: {args.text_col}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max length: {args.max_length}")
    logger.info(f"  Pooling: {args.pooling}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Processing: {len(df):,} articles")

    try:
        embeddings, mapping = build_bertweet_embeddings(
            df,
            text_col=args.text_col,
            model_name=args.model,
            sample_limit=None,  # Process all articles
            batch_size=args.batch_size,
            max_length=args.max_length,
            pooling=args.pooling,
            output_dir=str(output_dir),
            save_embeddings=False,  # We'll save with custom names
            device=device,
            verbose=args.verbose or True
        )

        logger.info(f"✓ Generated embeddings: {embeddings.shape}")

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Save with custom names
    logger.info(f"\n[3/3] Saving embeddings...")

    # Save embeddings
    embeddings_file = output_dir / f'{args.output_prefix}.npy'
    np.save(embeddings_file, embeddings)
    size_mb = embeddings_file.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved embeddings: {embeddings_file} ({size_mb:.2f} MB)")

    # Save mapping
    mapping_file = output_dir / f'{args.output_prefix}_mapping.csv'
    mapping.to_csv(mapping_file, index=False)
    size_kb = mapping_file.stat().st_size / 1024
    logger.info(f"✓ Saved mapping: {mapping_file} ({size_kb:.2f} KB)")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"  - {args.output_prefix}.npy ({embeddings.shape})")
    logger.info(f"  - {args.output_prefix}_mapping.csv ({len(mapping)} rows)")
    logger.info(f"\nEmbedding dimensions: {embeddings.shape[1]}")
    logger.info(f"Total articles: {len(embeddings):,}")
    logger.info(f"File size: {size_mb:.2f} MB")

    # Usage instructions
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("\n1. Load embeddings in Python:")
    logger.info(f"""
    import numpy as np
    import pandas as pd

    embeddings = np.load('{embeddings_file}')
    mapping = pd.read_csv('{mapping_file}')
    print(f"Loaded {{embeddings.shape[0]:,}} embeddings")
    """)

    logger.info("\n2. Use for similarity search:")
    logger.info("""
    from sklearn.metrics.pairwise import cosine_similarity

    # Find similar articles to article at index 0
    query_idx = 0
    similarities = cosine_similarity([embeddings[query_idx]], embeddings)[0]
    top_k_indices = similarities.argsort()[-10:][::-1]

    # Get article IDs
    similar_ids = mapping.iloc[top_k_indices]['_id'].tolist()
    print(f"Similar articles: {similar_ids}")
    """)

    logger.info("\n3. Update your API to use the new embeddings:")
    logger.info(f"""
    # In your config or API startup:
    EMBEDDINGS_PATH = '{embeddings_file}'
    MAPPING_PATH = '{mapping_file}'
    """)

    logger.info("\n✓ Embeddings generation complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
