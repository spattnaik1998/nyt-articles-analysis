"""
Full Pipeline - NYT Data Journalism Platform

This script runs the complete pipeline:
1. Download Kaggle dataset
2. Preprocess data
3. Generate embeddings
4. (Optional) Run topic modeling
5. (Optional) Run sentiment analysis

Usage:
    # Quick test with sample
    python scripts/run_full_pipeline.py --quick-test

    # Full pipeline (all 21M articles - WARNING: Takes hours!)
    python scripts/run_full_pipeline.py --full

    # Custom sample size
    python scripts/run_full_pipeline.py --sample 50000
"""

import os
import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_step(step_name: str, func, *args, **kwargs):
    """Run a pipeline step with timing."""
    logger.info("\n" + "="*80)
    logger.info(f"STEP: {step_name}")
    logger.info("="*80)

    start_time = datetime.now()

    try:
        result = func(*args, **kwargs)
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✅ {step_name} completed in {elapsed:.2f} seconds")
        return result
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"\n❌ {step_name} failed after {elapsed:.2f} seconds")
        logger.error(f"Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run NYT data pipeline")

    # Pipeline options
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test with 10k sample (recommended for first run)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full pipeline on all 21M articles (WARNING: Takes hours!)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Custom sample size (e.g., 50000)'
    )

    # Step control
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download (use existing data)'
    )
    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip preprocessing (use existing preprocessed.parquet)'
    )
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Skip embedding generation (use existing embeddings)'
    )
    parser.add_argument(
        '--run-topic-modeling',
        action='store_true',
        help='Run topic modeling after embeddings'
    )
    parser.add_argument(
        '--run-sentiment',
        action='store_true',
        help='Run sentiment analysis after embeddings'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.quick_test and args.full:
        logger.error("Cannot specify both --quick-test and --full")
        sys.exit(1)

    if args.quick_test and args.sample:
        logger.error("Cannot specify both --quick-test and --sample")
        sys.exit(1)

    # Determine sample size
    if args.quick_test:
        use_sample_file = True
        sample_size = None
        logger.info("Mode: Quick test with 10k sample")
    elif args.full:
        use_sample_file = False
        sample_size = None
        logger.info("Mode: Full pipeline with all articles (21M+)")
        logger.warning("This will take several hours and significant disk space!")
        confirm = input("Are you sure you want to continue? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Aborted by user")
            sys.exit(0)
    elif args.sample:
        use_sample_file = False
        sample_size = args.sample
        logger.info(f"Mode: Custom sample with {sample_size:,} articles")
    else:
        # Default to quick test
        use_sample_file = True
        sample_size = None
        logger.info("Mode: Default quick test with 10k sample")
        logger.info("Use --help to see other options")

    logger.info("\n" + "="*80)
    logger.info("NYT DATA JOURNALISM PLATFORM - FULL PIPELINE")
    logger.info("="*80)

    pipeline_start = datetime.now()

    # Step 1: Download dataset
    if not args.skip_download:
        from download_kaggle_dataset import download_kaggle_dataset, prepare_dataset

        dataset_path = run_step(
            "Download Kaggle Dataset",
            download_kaggle_dataset
        )

        run_step(
            "Prepare Dataset",
            prepare_dataset,
            dataset_path
        )
    else:
        logger.info("Skipping dataset download (--skip-download)")

    # Step 2: Preprocess
    if not args.skip_preprocess:
        from preprocess_data import preprocess_nyt_dataset

        preprocess_args = {
            'use_sample_file': use_sample_file
        }

        if not use_sample_file and sample_size:
            preprocess_args['sample_size'] = sample_size

        run_step(
            "Preprocess Data",
            preprocess_nyt_dataset,
            **preprocess_args
        )
    else:
        logger.info("Skipping preprocessing (--skip-preprocess)")

    # Step 3: Generate embeddings
    if not args.skip_embeddings:
        from generate_embeddings import generate_embeddings

        embedding_args = {
            'batch_size': 32
        }

        if not use_sample_file and sample_size:
            embedding_args['sample_size'] = sample_size

        run_step(
            "Generate Embeddings",
            generate_embeddings,
            **embedding_args
        )
    else:
        logger.info("Skipping embedding generation (--skip-embeddings)")

    # Optional: Topic modeling
    if args.run_topic_modeling:
        logger.info("\nTopic modeling not included in quick pipeline")
        logger.info("Run manually: python scripts/run_topic_modeling.py")

    # Optional: Sentiment analysis
    if args.run_sentiment:
        logger.info("\nSentiment analysis not included in quick pipeline")
        logger.info("Run manually: python scripts/run_sentiment.py")

    # Pipeline complete
    total_elapsed = (datetime.now() - pipeline_start).total_seconds()
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    logger.info("\n🎉 Success! Your NYT data platform is ready!")
    logger.info("\nNext steps:")
    logger.info("  1. Start API: uvicorn src.api.app:app --reload")
    logger.info("  2. Open browser: http://localhost:8000/static/index.html")
    logger.info("  3. Try search, topic modeling, and sentiment analysis")


if __name__ == "__main__":
    main()
