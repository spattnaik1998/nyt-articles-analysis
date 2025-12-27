#!/usr/bin/env python
"""
Sentiment Analysis Report Generator

This script runs multi-model sentiment analysis on articles and generates
a comprehensive comparison report highlighting model agreements/disagreements.

Usage:
    # Basic usage with preprocessed data
    python examples/sentiment_report.py

    # Custom input file
    python examples/sentiment_report.py --input data/preprocessed.parquet

    # Specific models
    python examples/sentiment_report.py --models finbert distilroberta

    # Sample limit
    python examples/sentiment_report.py --sample 100

    # Save results
    python examples/sentiment_report.py --output data/sentiment_results.csv
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sentiment import (
    batch_infer,
    model_comparison_report,
    list_available_models,
    get_model_info
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run sentiment analysis and generate comparison report."""
    parser = argparse.ArgumentParser(
        description='Multi-Model Sentiment Analysis Report Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on sample articles with default models
  python examples/sentiment_report.py --sample 50

  # Analyze specific section
  python examples/sentiment_report.py --input data/preprocessed.parquet --section Business

  # Compare all available models
  python examples/sentiment_report.py --models finbert finbert_tone distilroberta

  # Save results and report
  python examples/sentiment_report.py --output data/sentiment_results.csv --report data/report.txt
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV or Parquet file with articles'
    )

    parser.add_argument(
        '--text-col',
        type=str,
        default='combined_text',
        help='Column containing text to analyze (default: combined_text)'
    )

    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['finbert', 'distilroberta', 'finbert_tone'],
        choices=list_available_models(),
        help='Models to use for sentiment analysis (default: finbert distilroberta finbert_tone)'
    )

    parser.add_argument(
        '--sample', '-s',
        type=int,
        help='Limit to N random articles (default: use all)'
    )

    parser.add_argument(
        '--section',
        type=str,
        help='Filter to specific section (e.g., Business, World)'
    )

    parser.add_argument(
        '--year',
        type=int,
        help='Filter to specific year'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save results CSV (default: data/sentiment_results.csv)'
    )

    parser.add_argument(
        '--report', '-r',
        type=str,
        help='Path to save comparison report (default: data/sentiment_report.txt)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip generating comparison report'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Sentiment Analysis Report Generator")
    logger.info("=" * 70)

    # Show selected models
    logger.info(f"\nModels to use: {', '.join(args.models)}")
    for model_key in args.models:
        info = get_model_info(model_key)
        logger.info(f"  {model_key}: {info['description']}")

    # Load data
    logger.info(f"\n[1/4] Loading data...")

    if args.input:
        input_path = Path(args.input)

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            logger.info("\nTo generate preprocessed data, run:")
            logger.info("  python scripts/preprocess_sample.py --output data/preprocessed.parquet")
            return 1

        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            return 1

        logger.info(f"✓ Loaded {len(df):,} articles from {input_path}")

    else:
        # Use sample data for demonstration
        logger.info("No input file specified. Creating sample dataset...")

        sample_data = {
            'combined_text': [
                "Stock market rallies as economy shows strong growth and investor confidence rises",
                "Market crashes amid fears of recession and economic downturn worries investors",
                "Economy remains stable with moderate growth in key sectors",
                "Financial markets tumble as trade tensions escalate and uncertainty grows",
                "Corporate earnings exceed expectations driving market gains",
                "Government announces new economic stimulus package to boost growth",
                "Central bank raises interest rates to combat inflation concerns",
                "Unemployment drops to lowest level in decades boosting economy",
                "Trade deficit widens as imports surge and exports decline",
                "Consumer spending increases signaling economic confidence and strength"
            ],
            'section_name': ['Business'] * 10,
            'headline': [f"Sample Article {i+1}" for i in range(10)]
        }

        df = pd.DataFrame(sample_data)
        logger.info(f"✓ Created sample dataset with {len(df)} articles")

    # Validate text column exists
    if args.text_col not in df.columns:
        logger.error(f"Text column '{args.text_col}' not found in DataFrame")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        return 1

    # Filter by section if requested
    if args.section:
        logger.info(f"\n[2/4] Filtering to section: {args.section}")

        if 'section_name' not in df.columns:
            logger.warning("section_name column not found, skipping filter")
        else:
            before_count = len(df)
            df = df[df['section_name'] == args.section].copy()
            logger.info(f"✓ Filtered to {len(df):,} articles (from {before_count:,})")

            if len(df) == 0:
                logger.error(f"No articles found for section '{args.section}'")
                logger.info("\nAvailable sections:")
                print(df['section_name'].value_counts().head(20))
                return 1

    # Filter by year if requested
    if args.year:
        logger.info(f"\n[2/4] Filtering to year: {args.year}")

        if 'pub_date' not in df.columns:
            logger.warning("pub_date column not found, skipping filter")
        else:
            # Ensure pub_date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['pub_date']):
                df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

            before_count = len(df)
            df = df[df['pub_date'].dt.year == args.year].copy()
            logger.info(f"✓ Filtered to {len(df):,} articles (from {before_count:,})")

    # Sample if requested
    if args.sample and len(df) > args.sample:
        logger.info(f"\n[2/4] Sampling {args.sample:,} random articles...")
        df = df.sample(n=args.sample, random_state=42)
        logger.info(f"✓ Sampled {len(df):,} articles")

    if len(df) == 0:
        logger.error("No articles to process after filtering")
        return 1

    # Run sentiment analysis
    logger.info(f"\n[3/4] Running sentiment analysis...")
    logger.info(f"Processing {len(df):,} articles with {len(args.models)} model(s)")

    try:
        result_df = batch_infer(
            df,
            text_col=args.text_col,
            models=args.models,
            batch_size=args.batch_size,
            verbose=True
        )

        logger.info(f"\n✓ Sentiment analysis complete!")

        # Show sample results
        logger.info(f"\nSample Results (first 5 articles):")
        logger.info("=" * 70)

        display_cols = [args.text_col] if args.text_col in result_df.columns else []
        for model in args.models:
            display_cols.extend([f'{model}_label', f'{model}_score'])

        # Limit text column length for display
        display_df = result_df.head(5)[display_cols].copy()

        if args.text_col in display_df.columns:
            display_df[args.text_col] = display_df[args.text_col].str[:80] + '...'

        print(display_df.to_string(index=False))

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save results
    output_path = args.output if args.output else 'data/sentiment_results.csv'
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Results saved to {output_file}")

    # Generate comparison report
    if not args.no_report and len(args.models) >= 2:
        logger.info(f"\n[4/4] Generating model comparison report...")

        report_path = args.report if args.report else 'data/sentiment_report.txt'

        try:
            report = model_comparison_report(
                result_df,
                models=args.models,
                output_path=report_path,
                verbose=True
            )

            logger.info(f"\n✓ Report saved to {report_path}")

            # Show key stats
            logger.info(f"\nKey Statistics:")
            logger.info(f"  Total documents: {report['total_documents']:,}")

            if 'agreement_analysis' in report:
                aa = report['agreement_analysis']
                logger.info(f"  Perfect agreement: {aa['perfect_agreement_count']:,} ({aa['perfect_agreement_rate']:.1%})")
                logger.info(f"  Disagreements: {aa['disagreement_count']:,} ({aa['disagreement_rate']:.1%})")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Sentiment Analysis Complete!")
    logger.info("=" * 70)

    logger.info(f"\nFiles created:")
    logger.info(f"  Results CSV: {output_path}")

    if not args.no_report and len(args.models) >= 2:
        logger.info(f"  Comparison Report: {report_path if args.report else 'data/sentiment_report.txt'}")

    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review results:")
    logger.info(f"     head {output_path}")
    logger.info(f"\n  2. View comparison report:")

    if not args.no_report:
        logger.info(f"     cat {report_path if args.report else 'data/sentiment_report.txt'}")

    logger.info(f"\n  3. Analyze in Python:")
    logger.info(f"     import pandas as pd")
    logger.info(f"     df = pd.read_csv('{output_path}')")
    logger.info(f"     print(df[['finbert_label', 'distilroberta_label']].value_counts())")

    logger.info(f"\n  4. Run on full dataset:")
    logger.info(f"     python examples/sentiment_report.py --input data/preprocessed.parquet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
