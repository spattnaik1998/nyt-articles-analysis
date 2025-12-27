# Text Preprocessing Module

## Overview

The text preprocessing module (`src/preprocess/text.py`) provides robust functions for combining and cleaning NYT article text, preparing it for downstream NLP tasks like topic modeling and sentiment analysis.

## Quick Start

### Using the CLI Script

```bash
# Activate virtual environment
venv\Scripts\activate

# Process 1000 random articles (default)
python scripts/preprocess_sample.py --input data/nyt-metadata.csv

# Process 5000 articles
python scripts/preprocess_sample.py --input data/nyt-metadata.csv --sample 5000

# Process all articles from a specific year and section
python scripts/preprocess_sample.py --input data/nyt-metadata.csv --year 2001 --section World

# Process all articles (warning: may take time with large datasets)
python scripts/preprocess_sample.py --input data/nyt-metadata.csv --all
```

### Using in Python

```python
from src.ingest.load_nyt import load_nyt_csv
from src.preprocess.text import preprocess_dataframe

# Load data
df = load_nyt_csv('data/nyt-metadata.csv')

# Preprocess (combine + clean)
df_processed = preprocess_dataframe(df)

# Access results
print(df_processed['combined_text'].iloc[0])  # Original combined text
print(df_processed['cleaned_text'].iloc[0])   # Cleaned text
```

## Core Functions

### 1. `combine_text(df) -> DataFrame`

Combines headline, abstract, and lead_paragraph into a single 'combined_text' column.

**Special Features:**
- Extracts 'main' field from dict-like headlines: `"{'main': 'Actual Headline'}"`
- Handles NaN values gracefully
- Removes 'nan' string artifacts
- Normalizes multiple spaces

**Example:**
```python
from src.preprocess.text import combine_text

df_combined = combine_text(df)
print(df_combined['combined_text'].head())
```

### 2. `clean_text(s) -> str`

Cleans a single text string for NLP processing.

**Cleaning Operations:**
1. Converts to lowercase
2. Removes punctuation
3. Removes numbers
4. Removes 'nan' artifacts
5. Removes English stopwords (via NLTK)
6. Normalizes whitespace

**Example:**
```python
from src.preprocess.text import clean_text

text = "The Stock Market fell 5% today!"
cleaned = clean_text(text)
print(cleaned)  # Output: "stock market fell today"

# Customize cleaning
cleaned = clean_text(text, remove_numbers=False, remove_stopwords=False)
print(cleaned)  # Output: "stock market fell 5 today"
```

### 3. `preprocess_dataframe(df) -> DataFrame`

Full preprocessing pipeline: combine + clean in one step.

**Example:**
```python
from src.preprocess.text import preprocess_dataframe, get_preprocessing_stats

# Full pipeline
df_processed = preprocess_dataframe(df)

# Get statistics
stats = get_preprocessing_stats(df_processed)
print(f"Avg word reduction: {stats['avg_word_reduction']:.1f} words")
print(f"Reduction %: {stats['avg_reduction_pct']:.1f}%")
```

## CLI Script Options

### Basic Usage

```bash
python scripts/preprocess_sample.py [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | `data/nyt-metadata.csv` | Input CSV file path |
| `--output` | `-o` | `data/preprocessed.parquet` | Output Parquet file path |
| `--sample` | `-s` | `1000` | Number of articles to sample |
| `--all` | `-a` | - | Process all articles |
| `--year` | `-y` | - | Filter by publication year |
| `--section` | - | - | Filter by section name |
| `--random-seed` | - | `42` | Random seed for sampling |
| `--no-clean` | - | - | Skip cleaning (only combine) |
| `--verbose` | `-v` | - | Verbose output |

### Example Commands

```bash
# Sample 10,000 articles and save to custom location
python scripts/preprocess_sample.py \
    --input data/nyt-metadata.csv \
    --output data/sample_10k.parquet \
    --sample 10000

# Process all World section articles from 2001
python scripts/preprocess_sample.py \
    --input data/nyt-metadata.csv \
    --output data/world_2001.parquet \
    --year 2001 \
    --section World \
    --all

# Process sample with verbose logging
python scripts/preprocess_sample.py \
    --input data/nyt-metadata.csv \
    --sample 5000 \
    --verbose

# Only combine text without cleaning
python scripts/preprocess_sample.py \
    --input data/nyt-metadata.csv \
    --no-clean
```

## Output Format

The CLI script saves results to Parquet format with these columns:

- **Original columns**: `_id`, `pub_date`, `headline`, `web_url`, etc.
- **New columns**:
  - `combined_text`: Headline + abstract + lead_paragraph
  - `cleaned_text`: Cleaned version ready for NLP

**Load preprocessed data:**
```python
import pandas as pd

df = pd.read_parquet('data/preprocessed.parquet')
print(df.columns)
print(df[['headline', 'combined_text', 'cleaned_text']].head())
```

## Advanced Usage

### Custom Cleaning Options

```python
from src.preprocess.text import clean_text

# Keep numbers
clean_text("Test 123", remove_numbers=False)

# Keep case
clean_text("UPPERCASE Text", lowercase=False)

# Minimum word length
clean_text("a big cat is here", min_word_length=3)

# Disable stopword removal
clean_text("the quick fox", remove_stopwords=False)
```

### Safe Dict String Extraction

```python
from src.preprocess.text import safe_extract_from_dict_string

# Extract from dict-like string
headline = "{'main': 'Breaking News', 'kicker': None}"
extracted = safe_extract_from_dict_string(headline)
print(extracted)  # Output: "Breaking News"

# Regular strings pass through
headline = "Regular Headline"
extracted = safe_extract_from_dict_string(headline)
print(extracted)  # Output: "Regular Headline"
```

## Testing

Run the comprehensive test suite:

```bash
# Run all preprocessing tests
pytest tests/test_preprocess_text.py -v

# Run with coverage
pytest tests/test_preprocess_text.py -v --cov=src.preprocess --cov-report=term-missing

# Run specific test class
pytest tests/test_preprocess_text.py::TestCleanText -v
```

**Test Coverage:**
- 39 unit tests
- 100% coverage of core functions
- Tests for: dict extraction, text combination, cleaning, edge cases

## Performance

### Benchmark (on typical hardware)

| Dataset Size | Processing Time | Memory Usage |
|--------------|----------------|--------------|
| 1,000 articles | ~2 seconds | ~50 MB |
| 10,000 articles | ~15 seconds | ~200 MB |
| 100,000 articles | ~2 minutes | ~1.5 GB |
| 1,000,000 articles | ~20 minutes | ~12 GB |

**Tips for large datasets:**
- Use `--sample` for testing
- Process in batches by year/section
- Use Parquet format for efficient storage

## Integration with Pipeline

```
Data Ingestion (load_nyt.py)
    ↓
Text Preprocessing (text.py)  ← You are here
    ↓
Topic Modeling (models/)
    ↓
Sentiment Analysis (models/)
    ↓
API/Analysis (api/)
```

## Troubleshooting

### Issue: NLTK stopwords not found

```bash
# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

### Issue: File not found

```bash
# Verify file exists
ls data/nyt-metadata.csv

# Use absolute path
python scripts/preprocess_sample.py --input C:\full\path\to\data.csv
```

### Issue: Memory error with large datasets

```bash
# Process in smaller batches
python scripts/preprocess_sample.py --sample 10000

# Or filter by section
python scripts/preprocess_sample.py --section "U.S." --all
```

## Next Steps

After preprocessing:
1. **Topic Modeling**: Use `src/models/bertopic_model.py` for topic discovery
2. **Sentiment Analysis**: Use `src/models/sentiment.py` for tone detection
3. **Embeddings**: Generate BERTweet embeddings for similarity search
4. **API**: Expose via `src/api/` endpoints

## Example Workflow

```bash
# 1. Download or prepare your data
# data/nyt-metadata.csv should be ready

# 2. Preprocess a sample
python scripts/preprocess_sample.py \
    --input data/nyt-metadata.csv \
    --output data/preprocessed.parquet \
    --sample 10000

# 3. Verify output
python -c "import pandas as pd; df = pd.read_parquet('data/preprocessed.parquet'); print(df.info()); print(df[['headline', 'cleaned_text']].head())"

# 4. Use in analysis
# ... continue with topic modeling or sentiment analysis
```
