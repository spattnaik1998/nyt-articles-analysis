"""
Download and Prepare NYT Dataset from Kaggle

This script downloads the NYT Articles dataset (21M articles, 2000-present)
from Kaggle and prepares it for the data journalism platform.

Dataset: aryansingh0909/nyt-articles-21m-2000-present
"""

import os
import sys
from pathlib import Path
import pandas as pd
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_kaggle_dataset():
    """
    Download the NYT dataset from Kaggle using kagglehub.

    Returns:
        str: Path to the downloaded dataset directory
    """
    try:
        import kagglehub

        logger.info("Downloading NYT dataset from Kaggle...")
        logger.info("Dataset: aryansingh0909/nyt-articles-21m-2000-present")

        # Download latest version
        path = kagglehub.dataset_download("aryansingh0909/nyt-articles-21m-2000-present")

        logger.info(f"Dataset downloaded to: {path}")
        return path

    except ImportError:
        logger.error("kagglehub is not installed. Install with: pip install kagglehub")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        sys.exit(1)


def explore_dataset_structure(dataset_path: str):
    """
    Explore the structure of the downloaded dataset.

    Args:
        dataset_path (str): Path to the dataset directory
    """
    logger.info("\n" + "="*80)
    logger.info("DATASET STRUCTURE")
    logger.info("="*80)

    dataset_dir = Path(dataset_path)

    # List all files
    logger.info("\nFiles in dataset:")
    for file_path in sorted(dataset_dir.rglob("*")):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {file_path.name} ({size_mb:.2f} MB)")

    # Try to load and inspect CSV/Parquet files
    logger.info("\nInspecting data files:")

    for file_path in dataset_dir.rglob("*"):
        if file_path.suffix in ['.csv', '.parquet', '.json']:
            try:
                logger.info(f"\n--- {file_path.name} ---")

                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path, nrows=5)
                elif file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                    df = df.head(5)

                logger.info(f"Shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")
                logger.info(f"\nFirst row sample:")
                logger.info(df.head(1).to_dict('records'))

            except Exception as e:
                logger.warning(f"Could not load {file_path.name}: {e}")


def prepare_dataset(dataset_path: str, output_dir: str = "data"):
    """
    Prepare the Kaggle dataset for use in the platform.

    Args:
        dataset_path (str): Path to the downloaded dataset
        output_dir (str): Output directory for prepared data
    """
    logger.info("\n" + "="*80)
    logger.info("PREPARING DATASET")
    logger.info("="*80)

    dataset_dir = Path(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find the main data file
    csv_files = list(dataset_dir.rglob("*.csv"))
    parquet_files = list(dataset_dir.rglob("*.parquet"))

    data_file = None

    if parquet_files:
        data_file = parquet_files[0]
        logger.info(f"Found parquet file: {data_file}")
        df = pd.read_parquet(data_file)
    elif csv_files:
        data_file = csv_files[0]
        logger.info(f"Found CSV file: {data_file}")
        df = pd.read_csv(data_file)
    else:
        logger.error("No CSV or Parquet files found in dataset!")
        sys.exit(1)

    logger.info(f"\nDataset loaded: {len(df):,} articles")
    logger.info(f"Columns: {list(df.columns)}")

    # Show column info
    logger.info("\nColumn Information:")
    logger.info(df.dtypes.to_string())

    # Show missing values
    logger.info("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percent': missing_pct
    })
    logger.info(missing_df[missing_df['Missing'] > 0].to_string())

    # Save raw data to project data directory
    raw_output = output_path / "nyt_raw.parquet"
    logger.info(f"\nSaving raw data to: {raw_output}")
    df.to_parquet(raw_output, index=False)
    logger.info(f"Saved {len(df):,} articles")

    # Create a sample for quick testing
    sample_size = min(10000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    sample_output = output_path / "nyt_sample_10k.parquet"
    logger.info(f"\nCreating sample dataset: {sample_output}")
    sample_df.to_parquet(sample_output, index=False)
    logger.info(f"Saved {len(sample_df):,} articles for testing")

    # Show date range if available
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        logger.info(f"\nDate Range:")
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                min_date = df[col].min()
                max_date = df[col].max()
                logger.info(f"  {col}: {min_date} to {max_date}")
            except:
                pass

    # Show section distribution
    section_cols = [col for col in df.columns if 'section' in col.lower()]
    if section_cols:
        logger.info(f"\nTop Sections:")
        section_col = section_cols[0]
        top_sections = df[section_col].value_counts().head(10)
        logger.info(top_sections.to_string())

    logger.info("\n" + "="*80)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput files:")
    logger.info(f"  - Raw data: {raw_output}")
    logger.info(f"  - Sample (10k): {sample_output}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review the column structure above")
    logger.info(f"  2. Run preprocessing: python scripts/preprocess_data.py")
    logger.info(f"  3. Generate embeddings: python scripts/generate_embeddings.py")
    logger.info(f"  4. Run topic modeling: python scripts/run_topic_modeling.py")


def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("NYT DATASET DOWNLOADER")
    logger.info("="*80)

    # Download dataset
    dataset_path = download_kaggle_dataset()

    # Explore structure
    explore_dataset_structure(dataset_path)

    # Prepare for platform
    prepare_dataset(dataset_path)

    logger.info("\n✅ All done! Dataset is ready to use.")


if __name__ == "__main__":
    main()
