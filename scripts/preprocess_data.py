"""
Preprocess NYT Dataset for Analysis

This script preprocesses the raw NYT dataset by:
1. Cleaning text fields
2. Extracting features
3. Handling missing values
4. Creating combined text fields for embeddings/topic modeling
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess.text import clean_text, compute_text_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_nyt_dataset(
    input_file: str = "data/nyt_raw.parquet",
    output_file: str = "data/preprocessed.parquet",
    sample_size: int = None,
    use_sample_file: bool = False
):
    """
    Preprocess the NYT dataset.

    Args:
        input_file (str): Path to raw data file
        output_file (str): Path to save preprocessed data
        sample_size (int, optional): Process only N articles
        use_sample_file (bool): Use the 10k sample file for testing
    """
    logger.info("="*80)
    logger.info("NYT DATASET PREPROCESSING")
    logger.info("="*80)

    # Load data
    if use_sample_file:
        input_file = "data/nyt_sample_10k.parquet"
        logger.info(f"Using sample file: {input_file}")

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run: python scripts/download_kaggle_dataset.py")
        sys.exit(1)

    logger.info(f"Loading data from: {input_file}")

    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_file)
    elif input_path.suffix == '.csv':
        df = pd.read_csv(input_file)
    else:
        logger.error(f"Unsupported file format: {input_path.suffix}")
        sys.exit(1)

    logger.info(f"Loaded {len(df):,} articles")

    # Sample if requested
    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size:,} articles")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Display column info
    logger.info(f"\nColumns: {list(df.columns)}")

    # Standardize column names (common variations in NYT datasets)
    column_mapping = {}

    # Map common column name variations
    for col in df.columns:
        col_lower = col.lower()

        # Article ID
        if col_lower in ['_id', 'id', 'article_id', 'uri']:
            column_mapping[col] = '_id'

        # Headline
        elif col_lower in ['headline', 'title', 'main']:
            column_mapping[col] = 'headline'

        # Abstract/snippet
        elif col_lower in ['abstract', 'snippet', 'description']:
            column_mapping[col] = 'abstract'

        # Full text/body
        elif col_lower in ['body', 'content', 'text', 'lead_paragraph']:
            column_mapping[col] = 'body'

        # Publication date
        elif col_lower in ['pub_date', 'date', 'published', 'pub_time', 'publication_date']:
            column_mapping[col] = 'pub_date'

        # Section
        elif col_lower in ['section', 'section_name', 'desk']:
            column_mapping[col] = 'section_name'

        # Type
        elif col_lower in ['type', 'type_of_material', 'document_type']:
            column_mapping[col] = 'type_of_material'

        # Keywords
        elif col_lower in ['keywords', 'tags']:
            column_mapping[col] = 'keywords'

    if column_mapping:
        logger.info(f"\nStandardizing column names:")
        for old, new in column_mapping.items():
            logger.info(f"  {old} -> {new}")
        df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required_columns = ['_id', 'headline', 'pub_date']
    missing_required = [col for col in required_columns if col not in df.columns]

    if missing_required:
        logger.error(f"Missing required columns: {missing_required}")
        logger.error("Available columns: " + ", ".join(df.columns))
        sys.exit(1)

    # Convert pub_date to datetime
    logger.info("\nProcessing publication dates...")
    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

    # Show date range
    min_date = df['pub_date'].min()
    max_date = df['pub_date'].max()
    logger.info(f"Date range: {min_date} to {max_date}")

    # Add year, month columns
    df['year'] = df['pub_date'].dt.year
    df['month'] = df['pub_date'].dt.month

    # Show year distribution
    year_dist = df['year'].value_counts().sort_index()
    logger.info(f"\nArticles by year:")
    logger.info(year_dist.to_string())

    # Fill missing text fields
    logger.info("\nHandling missing values...")
    text_fields = ['headline', 'abstract', 'body']

    for field in text_fields:
        if field in df.columns:
            missing_count = df[field].isnull().sum()
            if missing_count > 0:
                logger.info(f"  {field}: {missing_count:,} missing ({missing_count/len(df)*100:.2f}%)")
                df[field] = df[field].fillna('')
        else:
            logger.info(f"  {field}: column not found, creating empty")
            df[field] = ''

    # Ensure section_name exists
    if 'section_name' not in df.columns:
        logger.warning("section_name not found, creating placeholder")
        df['section_name'] = 'Unknown'
    else:
        df['section_name'] = df['section_name'].fillna('Unknown')

    # Show section distribution
    logger.info("\nTop 15 sections:")
    logger.info(df['section_name'].value_counts().head(15).to_string())

    # Clean text fields
    logger.info("\nCleaning text fields...")

    for field in ['headline', 'abstract', 'body']:
        if field in df.columns:
            logger.info(f"  Cleaning {field}...")
            tqdm.pandas(desc=f"Cleaning {field}")
            df[f'{field}_cleaned'] = df[field].progress_apply(
                lambda x: clean_text(x) if isinstance(x, str) else ''
            )

    # Create combined text field for embeddings
    logger.info("\nCreating combined text field...")

    def combine_text(row):
        """Combine headline, abstract, and body with weights."""
        parts = []

        # Headline (most important, repeat 2x)
        if row.get('headline'):
            parts.append(row['headline'])
            parts.append(row['headline'])

        # Abstract
        if row.get('abstract'):
            parts.append(row['abstract'])

        # Body (truncate to first 500 chars to avoid overwhelming)
        if row.get('body'):
            body = str(row['body'])[:500]
            parts.append(body)

        return ' '.join(parts)

    tqdm.pandas(desc="Combining text")
    df['combined_text'] = df.progress_apply(combine_text, axis=1)

    # Clean combined text
    logger.info("Cleaning combined text...")
    tqdm.pandas(desc="Cleaning combined")
    df['cleaned_text'] = df['combined_text'].progress_apply(clean_text)

    # Compute text statistics
    logger.info("\nComputing text statistics...")
    stats_df = compute_text_stats(df, text_col='cleaned_text')

    # Show statistics
    logger.info("\nText Statistics:")
    logger.info(f"  Avg word count: {stats_df['word_count'].mean():.2f}")
    logger.info(f"  Avg char count: {stats_df['char_count'].mean():.2f}")
    logger.info(f"  Avg sentence count: {stats_df['sentence_count'].mean():.2f}")

    # Filter out very short articles (less than 10 words)
    min_words = 10
    before_filter = len(df)
    df = df[stats_df['word_count'] >= min_words].reset_index(drop=True)
    after_filter = len(df)
    removed = before_filter - after_filter

    logger.info(f"\nFiltered out {removed:,} articles with < {min_words} words")
    logger.info(f"Remaining: {after_filter:,} articles")

    # Save preprocessed data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving preprocessed data to: {output_file}")
    df.to_parquet(output_file, index=False)

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(df):,} articles ({file_size:.2f} MB)")

    # Display final column list
    logger.info(f"\nFinal columns ({len(df.columns)}):")
    for col in df.columns:
        logger.info(f"  - {col}")

    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput: {output_file}")
    logger.info(f"Articles: {len(df):,}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Generate embeddings: python scripts/generate_embeddings.py")
    logger.info(f"  2. Run topic modeling: python scripts/run_topic_modeling.py")
    logger.info(f"  3. Run sentiment analysis: python scripts/run_sentiment.py")


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess NYT dataset")
    parser.add_argument(
        '--input',
        type=str,
        default='data/nyt_raw.parquet',
        help='Input file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/preprocessed.parquet',
        help='Output file path'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Process only N articles (for testing)'
    )
    parser.add_argument(
        '--use-sample-file',
        action='store_true',
        help='Use the 10k sample file for quick testing'
    )

    args = parser.parse_args()

    preprocess_nyt_dataset(
        input_file=args.input,
        output_file=args.output,
        sample_size=args.sample,
        use_sample_file=args.use_sample_file
    )


if __name__ == "__main__":
    main()
