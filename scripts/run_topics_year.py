#!/usr/bin/env python
"""
Topic Modeling by Year and Section

This script runs LDA and BERTopic topic modeling for a specified year and section,
saving outputs to organized directories.

Usage:
    python scripts/run_topics_year.py --year 2001 --section World
    python scripts/run_topics_year.py --year 2020 --section "Business Day" --num-topics 15
    python scripts/run_topics_year.py --year 2024 --section U.S. --bertopic-only
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.load_nyt import load_nyt_csv
from src.preprocess.text import preprocess_dataframe
from src.models.topic_models import run_lda, run_bertopic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for running topic models by year and section."""
    parser = argparse.ArgumentParser(
        description='Run LDA and BERTopic topic modeling for a specific year and section',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both LDA and BERTopic for 2001 World articles
  python scripts/run_topics_year.py --year 2001 --section World

  # Run with custom number of topics
  python scripts/run_topics_year.py --year 2020 --section "Business Day" --num-topics 15

  # Run only BERTopic
  python scripts/run_topics_year.py --year 2024 --section U.S. --bertopic-only

  # Run only LDA
  python scripts/run_topics_year.py --year 2001 --section Opinion --lda-only

  # Use preprocessed data
  python scripts/run_topics_year.py --input data/preprocessed.parquet --year 2001 --section World

  # Save intertopic distance map
  python scripts/run_topics_year.py --year 2001 --section World --save-map
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV or Parquet file (will load and filter if provided)'
    )

    parser.add_argument(
        '--year', '-y',
        type=int,
        required=True,
        help='Year to analyze (required)'
    )

    parser.add_argument(
        '--section', '-s',
        type=str,
        required=True,
        help='Section to analyze (required, e.g., "World", "Business Day")'
    )

    parser.add_argument(
        '--num-topics', '-n',
        type=int,
        default=10,
        help='Number of topics for LDA (default: 10)'
    )

    parser.add_argument(
        '--bertopic-topics',
        type=int,
        default=10,
        help='Number of topics for BERTopic (default: 10)'
    )

    parser.add_argument(
        '--lda-only',
        action='store_true',
        help='Run only LDA (skip BERTopic)'
    )

    parser.add_argument(
        '--bertopic-only',
        action='store_true',
        help='Run only BERTopic (skip LDA)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/topics',
        help='Base output directory (default: data/topics)'
    )

    parser.add_argument(
        '--save-map',
        action='store_true',
        help='Save BERTopic intertopic distance map to HTML'
    )

    parser.add_argument(
        '--no-below',
        type=int,
        default=15,
        help='LDA: Minimum document frequency (default: 15)'
    )

    parser.add_argument(
        '--no-above',
        type=float,
        default=0.5,
        help='LDA: Maximum document proportion (default: 0.5)'
    )

    parser.add_argument(
        '--passes',
        type=int,
        default=10,
        help='LDA: Number of training passes (default: 10)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=100,
        help='Random seed for reproducibility (default: 100)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Validate
    if args.lda_only and args.bertopic_only:
        logger.error("Cannot specify both --lda-only and --bertopic-only")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Topic Modeling by Year and Section")
    logger.info("=" * 70)
    logger.info(f"Year: {args.year}")
    logger.info(f"Section: {args.section}")

    # Load and filter data
    if args.input:
        logger.info(f"\n[1/5] Loading data from: {args.input}")

        try:
            input_path = Path(args.input)

            if input_path.suffix == '.parquet':
                df = pd.read_parquet(args.input)
            elif input_path.suffix == '.csv':
                df = pd.read_csv(args.input)
            else:
                logger.error(f"Unsupported file format: {input_path.suffix}")
                sys.exit(1)

            logger.info(f"✓ Loaded {len(df):,} articles")

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            sys.exit(1)

    else:
        logger.error("No input file specified. Use --input to provide data.")
        sys.exit(1)

    # Filter by year and section
    logger.info(f"\n[2/5] Filtering by year={args.year} and section={args.section}")

    # Ensure pub_date is datetime
    if 'pub_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['pub_date']):
        df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

    # Filter
    filtered_df = df[
        (df['pub_date'].dt.year == args.year) &
        (df['section_name'] == args.section)
    ].copy()

    if len(filtered_df) == 0:
        logger.error(f"No articles found for year={args.year}, section={args.section}")
        logger.info(f"\nAvailable sections for {args.year}:")
        year_df = df[df['pub_date'].dt.year == args.year]
        if len(year_df) > 0:
            print(year_df['section_name'].value_counts().head(20))
        sys.exit(1)

    logger.info(f"✓ Found {len(filtered_df):,} articles")

    # Prepare text
    logger.info(f"\n[3/5] Preparing text for topic modeling")

    # Use cleaned_text if available, otherwise combined_text, otherwise combine
    if 'cleaned_text' in filtered_df.columns:
        text_col = 'cleaned_text'
        logger.info(f"Using '{text_col}' column")
    elif 'combined_text' in filtered_df.columns:
        text_col = 'combined_text'
        logger.info(f"Using '{text_col}' column")
    else:
        logger.info("Combining headline + abstract + lead_paragraph...")
        from src.preprocess.text import combine_text
        filtered_df = combine_text(filtered_df)
        text_col = 'combined_text'

    # Get documents
    documents = filtered_df[text_col].fillna('').astype(str).tolist()

    # Remove empty documents
    documents = [doc for doc in documents if doc.strip()]

    logger.info(f"✓ Prepared {len(documents):,} documents")

    if len(documents) < 10:
        logger.warning(f"Very few documents ({len(documents)}). Results may be poor.")

    # Create output directory
    section_safe = args.section.replace(' ', '_').replace('/', '_')
    output_dir = Path(args.output_dir) / f"{args.year}_{section_safe}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nOutput directory: {output_dir}")

    # Run LDA
    if not args.bertopic_only:
        logger.info(f"\n[4/5] Running LDA Topic Modeling")
        logger.info("-" * 70)
        logger.info(f"Parameters:")
        logger.info(f"  num_topics: {args.num_topics}")
        logger.info(f"  no_below: {args.no_below}")
        logger.info(f"  no_above: {args.no_above}")
        logger.info(f"  passes: {args.passes}")
        logger.info(f"  random_state: {args.seed}")

        try:
            lda_model, lda_topics_df, dictionary, corpus = run_lda(
                documents,
                num_topics=args.num_topics,
                no_below=args.no_below,
                no_above=args.no_above,
                passes=args.passes,
                random_state=args.seed,
                output_dir=str(output_dir),
                save_model=True,
                verbose=args.verbose
            )

            logger.info(f"\n✓ LDA complete")
            logger.info(f"  Dictionary size: {len(dictionary)} tokens")
            logger.info(f"  Topics discovered: {args.num_topics}")

            # Display topics
            logger.info(f"\nLDA Topics:")
            for idx, row in lda_topics_df.iterrows():
                logger.info(f"  Topic {row['topic_id']}: {row['keywords']}")

            # Save summary
            summary_path = output_dir / 'lda_summary.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"LDA Topic Modeling Results\n")
                f.write(f"=" * 70 + "\n")
                f.write(f"Year: {args.year}\n")
                f.write(f"Section: {args.section}\n")
                f.write(f"Documents: {len(documents):,}\n")
                f.write(f"Topics: {args.num_topics}\n")
                f.write(f"Dictionary size: {len(dictionary)}\n")
                f.write(f"\nParameters:\n")
                f.write(f"  no_below: {args.no_below}\n")
                f.write(f"  no_above: {args.no_above}\n")
                f.write(f"  passes: {args.passes}\n")
                f.write(f"  random_state: {args.seed}\n")
                f.write(f"\nTopics:\n")
                for idx, row in lda_topics_df.iterrows():
                    f.write(f"\nTopic {row['topic_id']}:\n")
                    f.write(f"  Keywords: {row['keywords']}\n")

            logger.info(f"Saved summary to {summary_path}")

        except Exception as e:
            logger.error(f"LDA failed: {e}")
            import traceback
            traceback.print_exc()

    # Run BERTopic
    if not args.lda_only:
        logger.info(f"\n[5/5] Running BERTopic Topic Modeling")
        logger.info("-" * 70)

        # Calculate min_topic_size
        min_topic_size = max(5, int(len(documents) * 0.005))

        logger.info(f"Parameters:")
        logger.info(f"  nr_topics: {args.bertopic_topics}")
        logger.info(f"  min_topic_size: {min_topic_size}")
        logger.info(f"  random_state: {args.seed}")

        try:
            bertopic_model, topic_info = run_bertopic(
                documents,
                nr_topics=args.bertopic_topics,
                min_topic_size=min_topic_size,
                random_state=args.seed,
                output_dir=str(output_dir),
                save_model=True,
                save_intertopic_map=args.save_map,
                verbose=args.verbose
            )

            logger.info(f"\n✓ BERTopic complete")
            logger.info(f"  Topics discovered: {len(topic_info) - 1} (excluding outliers)")

            # Display top topics
            logger.info(f"\nTop 10 BERTopic Topics:")
            top_topics = topic_info[topic_info['Topic'] != -1].head(10)
            for idx, row in top_topics.iterrows():
                logger.info(f"  Topic {row['Topic']} ({row['Count']} docs): {row['Name']}")

            # Save summary
            summary_path = output_dir / 'bertopic_summary.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"BERTopic Topic Modeling Results\n")
                f.write(f"=" * 70 + "\n")
                f.write(f"Year: {args.year}\n")
                f.write(f"Section: {args.section}\n")
                f.write(f"Documents: {len(documents):,}\n")
                f.write(f"Topics: {len(topic_info) - 1} (excluding outliers)\n")
                f.write(f"\nParameters:\n")
                f.write(f"  nr_topics: {args.bertopic_topics}\n")
                f.write(f"  min_topic_size: {min_topic_size}\n")
                f.write(f"  random_state: {args.seed}\n")
                f.write(f"\nTop 10 Topics:\n")
                for idx, row in top_topics.iterrows():
                    f.write(f"\nTopic {row['Topic']} ({row['Count']} documents):\n")
                    f.write(f"  {row['Name']}\n")

            logger.info(f"Saved summary to {summary_path}")

        except Exception as e:
            logger.error(f"BERTopic failed: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Topic Modeling Complete!")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nFiles created:")

    for file in sorted(output_dir.glob('*')):
        size = file.stat().st_size / 1024  # KB
        logger.info(f"  {file.name} ({size:.1f} KB)")

    logger.info(f"\nNext steps:")
    logger.info(f"  - Review topic summaries in {output_dir}")
    logger.info(f"  - Analyze topics with:")
    logger.info(f"      cd {output_dir}")
    logger.info(f"      cat lda_summary.txt")
    logger.info(f"      cat bertopic_summary.txt")

    if args.save_map:
        logger.info(f"  - View intertopic distance map:")
        logger.info(f"      open {output_dir}/intertopic_distance_map.html")


if __name__ == "__main__":
    main()
