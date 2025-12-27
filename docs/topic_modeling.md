# Topic Modeling - LDA and BERTopic

## Overview

The topic modeling module (`src/models/topic_models.py`) provides reproducible topic modeling using:

- **LDA (Latent Dirichlet Allocation)**: Classic probabilistic topic modeling via Gensim
- **BERTopic**: Neural topic modeling with transformer embeddings
- **CLI Tool**: Run topic modeling by year and section (`scripts/run_topics_year.py`)

All implementations follow the patterns from the original NYT analysis notebook with reproducible parameters.

## Quick Start

### 1. Run LDA Topic Modeling

```python
from src.models.topic_models import run_lda

documents = [
    "stock market economy trade financial growth",
    "politics election government vote democracy",
    "climate change environment global warming"
]

lda_model, topics_df, dictionary, corpus = run_lda(
    documents,
    num_topics=5,
    no_below=15,
    no_above=0.5,
    passes=10,
    random_state=100
)

# Display topics
for idx, row in topics_df.iterrows():
    print(f"Topic {row['topic_id']}: {row['keywords']}")
```

### 2. Run BERTopic Modeling

```python
from src.models.topic_models import run_bertopic

bertopic_model, topic_info = run_bertopic(
    documents,
    nr_topics=5,
    min_topic_size=10,
    random_state=100,
    save_intertopic_map=True,
    output_dir='data/topics'
)

# Display topics
print(topic_info[['Topic', 'Count', 'Name']])
```

### 3. Use CLI Tool

```bash
# Run both LDA and BERTopic for 2001 World articles
python scripts/run_topics_year.py --year 2001 --section World

# Custom number of topics
python scripts/run_topics_year.py --year 2020 --section "Business Day" --num-topics 15

# Run only BERTopic with intertopic map
python scripts/run_topics_year.py --year 2024 --section U.S. --bertopic-only --save-map

# Run only LDA
python scripts/run_topics_year.py --year 2001 --section Opinion --lda-only
```

## Core Functions

### `run_lda(...)`

Run LDA topic modeling using Gensim, following the notebook pattern.

**Parameters:**
- `documents` (List[str] or List[List[str]]): Documents or pre-tokenized documents
- `num_topics` (int): Number of topics (default: 10)
- `no_below` (int): Minimum document frequency (default: 15)
- `no_above` (float): Maximum document proportion (default: 0.5)
- `keep_n` (int): Keep only top N tokens (default: 100000)
- `passes` (int): Number of training passes (default: 10)
- `random_state` (int): Random seed for reproducibility (default: 100)
- `chunksize` (int): Chunk size for training (default: 100)
- `workers` (int): Number of worker threads (default: 4)
- `per_word_topics` (bool): Compute per-word topic distribution (default: True)
- `alpha` (str): Document-topic density ('auto' or float)
- `eta` (str): Topic-word density ('auto' or float)
- `output_dir` (str, optional): Directory to save outputs
- `save_model` (bool): Save model to disk (default: False)
- `verbose` (bool): Print progress (default: True)

**Returns:**
- `lda_model`: Trained LDA model (Gensim LdaMulticore)
- `topics_df`: DataFrame with topic keywords and weights
- `dictionary`: Gensim Dictionary object
- `corpus`: Bag-of-words corpus

**Example:**
```python
from src.models.topic_models import run_lda

# Load preprocessed data
import pandas as pd
df = pd.read_parquet('data/preprocessed.parquet')

# Filter to specific year and section
df_subset = df[
    (df['pub_date'].dt.year == 2001) &
    (df['section_name'] == 'World')
]

documents = df_subset['cleaned_text'].tolist()

# Run LDA
lda_model, topics_df, dictionary, corpus = run_lda(
    documents,
    num_topics=10,
    no_below=15,
    no_above=0.5,
    passes=10,
    random_state=100,
    output_dir='data/topics/2001_world',
    save_model=True
)

# Display topics
print("\nLDA Topics:")
for idx, row in topics_df.iterrows():
    print(f"Topic {row['topic_id']}: {row['keywords']}")
```

### `run_bertopic(...)`

Run BERTopic neural topic modeling.

**Parameters:**
- `documents` (List[str]): List of text documents
- `nr_topics` (int, optional): Number of topics (None = auto, int = reduce to N)
- `min_topic_size` (int, optional): Minimum cluster size (default: auto-calculated as max(5, 0.5% of documents))
- `language` (str): Language for models (default: 'english')
- `seed_topic_list` (List[List[str]], optional): Seed topics for guided modeling
- `random_state` (int): Random seed (default: 100)
- `verbose` (bool): Verbose output (default: False)
- `output_dir` (str, optional): Directory to save outputs
- `save_model` (bool): Save BERTopic model (default: False)
- `save_intertopic_map` (bool): Save intertopic distance map HTML (default: False)
- `intertopic_map_width` (int): Width of intertopic map (default: 1200)
- `intertopic_map_height` (int): Height of intertopic map (default: 800)

**Returns:**
- `bertopic_model`: Trained BERTopic model
- `topic_info`: DataFrame with topic information (Topic, Count, Name, Representation)

**Example:**
```python
from src.models.topic_models import run_bertopic

# Load documents
df = pd.read_parquet('data/preprocessed.parquet')
df_subset = df[df['section_name'] == 'Technology']
documents = df_subset['cleaned_text'].tolist()

# Run BERTopic
bertopic_model, topic_info = run_bertopic(
    documents,
    nr_topics=10,
    min_topic_size=20,
    random_state=100,
    output_dir='data/topics/tech',
    save_model=True,
    save_intertopic_map=True
)

# Display top topics
print("\nTop 10 Topics:")
top_topics = topic_info[topic_info['Topic'] != -1].head(10)
for idx, row in top_topics.iterrows():
    print(f"Topic {row['Topic']} ({row['Count']} docs): {row['Name']}")
```

## Reproducible Parameters

Following the notebook patterns for reproducibility:

### LDA Default Parameters
```python
{
    'num_topics': 10,
    'no_below': 15,           # Filter tokens in < 15 documents
    'no_above': 0.5,          # Filter tokens in > 50% of documents
    'keep_n': 100000,         # Keep top 100K tokens
    'passes': 10,             # Training iterations
    'random_state': 100,      # Reproducible seed
    'chunksize': 100,         # Documents per update
    'workers': 4,             # Parallel workers
    'per_word_topics': True,  # Compute per-word distributions
    'alpha': 'auto',          # Document-topic density
    'eta': 'auto'             # Topic-word density
}
```

### BERTopic Default Parameters
```python
{
    'nr_topics': 10,          # Reduce to 10 topics (None = auto)
    'min_topic_size': None,   # Auto: max(5, 0.5% of docs)
    'language': 'english',    # Language for stop words
    'random_state': 100,      # Reproducible seed
    'verbose': False,         # Suppress output
    'calculate_probabilities': True  # For better topic assignments
}
```

## CLI Tool: `run_topics_year.py`

### Usage

```bash
python scripts/run_topics_year.py --year YEAR --section SECTION [OPTIONS]
```

### Required Arguments

- `--year, -y`: Year to analyze (required)
- `--section, -s`: Section to analyze (required, e.g., "World", "Business Day")

### Optional Arguments

- `--input, -i`: Path to input CSV or Parquet file
- `--num-topics, -n`: Number of topics for LDA (default: 10)
- `--bertopic-topics`: Number of topics for BERTopic (default: 10)
- `--lda-only`: Run only LDA (skip BERTopic)
- `--bertopic-only`: Run only BERTopic (skip LDA)
- `--output-dir, -o`: Base output directory (default: data/topics)
- `--save-map`: Save BERTopic intertopic distance map to HTML
- `--no-below`: LDA minimum document frequency (default: 15)
- `--no-above`: LDA maximum document proportion (default: 0.5)
- `--passes`: LDA training passes (default: 10)
- `--seed`: Random seed for reproducibility (default: 100)
- `--verbose, -v`: Verbose output

### Examples

#### Example 1: Basic Usage
```bash
python scripts/run_topics_year.py \
    --input data/preprocessed.parquet \
    --year 2001 \
    --section World
```

**Output Directory:** `data/topics/2001_World/`

**Files Created:**
- `lda_model` - Saved Gensim LDA model
- `lda_topics.csv` - LDA topic keywords and weights
- `lda_summary.txt` - Human-readable LDA summary
- `dictionary.pkl` - Gensim dictionary
- `bertopic_model` - Saved BERTopic model
- `bertopic_topics.csv` - BERTopic topic info
- `bertopic_summary.txt` - Human-readable BERTopic summary

#### Example 2: Custom Topics with Map
```bash
python scripts/run_topics_year.py \
    --input data/preprocessed.parquet \
    --year 2020 \
    --section "Business Day" \
    --num-topics 15 \
    --save-map
```

**Additional File:**
- `intertopic_distance_map.html` - Interactive visualization

#### Example 3: BERTopic Only
```bash
python scripts/run_topics_year.py \
    --input data/preprocessed.parquet \
    --year 2024 \
    --section U.S. \
    --bertopic-only \
    --bertopic-topics 20 \
    --save-map
```

#### Example 4: LDA Only with Custom Parameters
```bash
python scripts/run_topics_year.py \
    --input data/preprocessed.parquet \
    --year 2001 \
    --section Opinion \
    --lda-only \
    --num-topics 20 \
    --no-below 10 \
    --no-above 0.7 \
    --passes 15
```

## Complete Workflow Example

```python
import pandas as pd
from src.ingest.load_nyt import load_nyt_csv
from src.preprocess.text import preprocess_dataframe
from src.models.topic_models import run_lda, run_bertopic

# 1. Load and preprocess data
df = load_nyt_csv('data/nyt-metadata.csv')
df = preprocess_dataframe(df)

# 2. Filter to specific year and section
df_2001_world = df[
    (df['pub_date'].dt.year == 2001) &
    (df['section_name'] == 'World')
].copy()

print(f"Found {len(df_2001_world):,} articles")

# 3. Prepare documents
documents = df_2001_world['cleaned_text'].fillna('').astype(str).tolist()
documents = [doc for doc in documents if doc.strip()]  # Remove empty

# 4. Run LDA
print("\n=== Running LDA ===")
lda_model, lda_topics_df, dictionary, corpus = run_lda(
    documents,
    num_topics=10,
    no_below=15,
    no_above=0.5,
    passes=10,
    random_state=100,
    output_dir='data/topics/2001_world',
    save_model=True
)

print("\nLDA Topics:")
for idx, row in lda_topics_df.iterrows():
    print(f"Topic {row['topic_id']}: {row['keywords']}")

# 5. Run BERTopic
print("\n=== Running BERTopic ===")
bertopic_model, topic_info = run_bertopic(
    documents,
    nr_topics=10,
    min_topic_size=max(5, int(len(documents) * 0.005)),
    random_state=100,
    output_dir='data/topics/2001_world',
    save_model=True,
    save_intertopic_map=True
)

print("\nTop BERTopic Topics:")
top_topics = topic_info[topic_info['Topic'] != -1].head(10)
for idx, row in top_topics.iterrows():
    print(f"Topic {row['Topic']} ({row['Count']} docs): {row['Name']}")

# 6. View intertopic map
print("\nOpen data/topics/2001_world/intertopic_distance_map.html in browser")
```

## Helper Functions

### `tokenize_documents(...)`

Tokenize documents using Gensim's simple_preprocess.

```python
from src.models.topic_models import tokenize_documents

docs = ["This is a test document", "Another test"]
tokenized = tokenize_documents(docs, deacc=True)
print(tokenized[0])
# Output: ['this', 'is', 'test', 'document']
```

### `get_topic_keywords(...)`

Get keywords for a specific LDA topic.

```python
from src.models.topic_models import get_topic_keywords

# After training LDA
keywords = get_topic_keywords(lda_model, topic_id=0, topn=10)
print(keywords)
# Output: [('economy', 0.045), ('market', 0.038), ...]
```

### `get_document_topics(...)`

Get topic distribution for each document.

```python
from src.models.topic_models import get_document_topics

# After training LDA
doc_topics = get_document_topics(lda_model, corpus, minimum_probability=0.1)

# Topic distribution for first document
print(doc_topics[0])
# Output: [(2, 0.65), (5, 0.25)]  # 65% topic 2, 25% topic 5
```

### `load_lda_model(...)` / `load_bertopic_model(...)`

Load saved models.

```python
from src.models.topic_models import load_lda_model, load_bertopic_model

# Load LDA
lda_model = load_lda_model('data/topics/2001_world/lda_model')

# Load BERTopic
bertopic_model = load_bertopic_model('data/topics/2001_world/bertopic_model')
```

## Understanding Topic Model Outputs

### LDA Topics DataFrame

Columns:
- `topic_id` (int): Topic number (0 to num_topics-1)
- `keywords` (str): Comma-separated top keywords
- `top_10_words` (list): List of top 10 words
- `top_10_weights` (list): Corresponding weights (probabilities)

Example:
```
topic_id  keywords                                    top_10_words                  top_10_weights
0         war, iraq, military, troops, baghdad, ...   ['war', 'iraq', 'military']   [0.045, 0.038, 0.032]
1         economy, market, growth, financial, ...     ['economy', 'market']         [0.052, 0.041]
```

### BERTopic Info DataFrame

Columns:
- `Topic` (int): Topic ID (-1 = outliers)
- `Count` (int): Number of documents in topic
- `Name` (str): Topic label with top words
- `Representation` (list): Top words as list

Example:
```
Topic  Count  Name                                    Representation
-1     245    -1_the_to_and_of                       ['the', 'to', 'and']
0      892    0_war_iraq_military_troops            ['war', 'iraq', 'military']
1      654    1_economy_market_growth_financial     ['economy', 'market', 'growth']
```

## Troubleshooting

### Issue 1: Gensim Not Available

**Error:**
```
ImportError: Gensim required. Install with: pip install gensim
```

**Solution:**
```bash
pip install gensim
```

### Issue 2: BERTopic Not Available

**Error:**
```
ImportError: BERTopic required. Install with: pip install bertopic
```

**Solution:**
```bash
pip install bertopic
```

### Issue 3: No Articles Found for Year/Section

**Error:**
```
No articles found for year=2001, section=WorldNews
```

**Solution:**
Check available sections for that year:
```python
df = pd.read_parquet('data/preprocessed.parquet')
year_df = df[df['pub_date'].dt.year == 2001]
print(year_df['section_name'].value_counts().head(20))
```

Use the exact section name from the output.

### Issue 4: Very Few Documents Warning

**Warning:**
```
Very few documents (8). Results may be poor.
```

**Solution:**
- Use broader filter criteria (multiple years, broader section)
- Reduce min_topic_size for BERTopic
- Use fewer topics (num_topics=3)

### Issue 5: Out of Memory

**Problem:** Training crashes with large dataset

**Solution:**
```python
# Sample documents before training
import random
random.seed(100)
sampled_docs = random.sample(documents, min(10000, len(documents)))

lda_model, topics_df, dictionary, corpus = run_lda(
    sampled_docs,
    num_topics=10
)
```

## Performance Benchmarks

### LDA Training Time

| Documents | Topics | Passes | Workers | Time    |
|-----------|--------|--------|---------|---------|
| 1,000     | 10     | 10     | 4       | ~10 sec |
| 10,000    | 10     | 10     | 4       | ~2 min  |
| 100,000   | 10     | 10     | 4       | ~20 min |
| 1,000,000 | 10     | 10     | 4       | ~3 hrs  |

### BERTopic Training Time

| Documents | min_topic_size | Time     |
|-----------|----------------|----------|
| 1,000     | 10             | ~30 sec  |
| 10,000    | 50             | ~3 min   |
| 100,000   | 100            | ~30 min  |
| 1,000,000 | 500            | ~5 hrs   |

**Note:** BERTopic is slower but produces more coherent topics for short texts.

## Integration with Other Modules

### Combine with Embeddings

```python
from src.models.embeddings import build_bertweet_embeddings
from src.models.topic_models import run_lda, run_bertopic

# Generate embeddings
embeddings, mapping = build_bertweet_embeddings(
    df,
    text_col='cleaned_text',
    output_dir='data'
)

# Run topic models on same documents
documents = df['cleaned_text'].tolist()
lda_model, topics_df, dictionary, corpus = run_lda(documents)
bertopic_model, topic_info = run_bertopic(documents)

# Compare: Which topics align with embedding clusters?
```

### Combine with Similarity Search

```python
from src.models.similarity import recommend_by_embedding
from src.models.topic_models import run_lda

# Run LDA
lda_model, topics_df, dictionary, corpus = run_lda(documents)

# Get dominant topic for each document
doc_topics = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]

# Find articles similar to those in topic 3
topic_3_docs = [i for i, t in enumerate(doc_topics) if t == 3]
query_text = documents[topic_3_docs[0]]

results = recommend_by_embedding(query_text, top_k=10)
print("Articles similar to topic 3:")
print(results[['headline', 'similarity']])
```

## Next Steps

After implementing topic modeling:

1. **Analyze topic evolution over time**
   - Run topic models for each year
   - Track keyword changes
   - Visualize topic trends

2. **Build topic-based search**
   - Assign topics to all articles
   - Allow filtering by topic
   - Combine with similarity search

3. **Create topic summaries**
   - Extract representative documents per topic
   - Generate topic descriptions
   - Build topic browser UI

4. **Deploy topic modeling API**
   - FastAPI endpoint for on-demand topic modeling
   - Cache results for common queries
   - Real-time topic assignment for new articles

## Summary

**Features Implemented:**
- ✅ LDA topic modeling with Gensim (reproducible parameters)
- ✅ BERTopic neural topic modeling
- ✅ Intertopic distance map visualization (HTML)
- ✅ CLI tool for year/section analysis
- ✅ Model saving and loading
- ✅ Complete documentation and examples

**Key Parameters:**
- LDA: `no_below=15`, `no_above=0.5`, `passes=10`, `random_state=100`
- BERTopic: `nr_topics=10`, `min_topic_size=auto`, `random_state=100`

**Ready for:**
- Large-scale topic analysis across years and sections
- Topic-based article recommendations
- Temporal topic evolution studies
- Integration with embeddings and similarity search
