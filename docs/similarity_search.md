## Similarity Search & Recommendations

## Overview

The similarity module (`src/models/similarity.py`) provides functions for finding similar articles and making recommendations based on semantic embeddings. It supports:

- **Article recommendations**: Find articles similar to a query text
- **Book recommendations**: Recommend books with extracted metadata
- **FAISS integration**: Fast similarity search for large datasets
- **NumPy fallback**: Works without FAISS (slower but always available)

## Quick Start

### 1. Recommend Articles

```python
from src.models.similarity import recommend_by_embedding

results = recommend_by_embedding(
    query_text="stock market crash economic recession",
    embeddings_path='data/embeddings.npy',
    id_map_csv='data/embeddings_mapping.csv',
    articles_df_path='data/preprocessed.parquet',
    top_k=5
)

print(results[['_id', 'similarity', 'headline']])
```

### 2. Recommend Books

```python
from src.models.similarity import recommend_books

results = recommend_books(
    query_text="historical fiction world war",
    books_df_path='data/books_with_metadata.parquet',
    book_embeddings_path='data/book_embeddings.npy',
    top_k=5
)

print(results[['book_title', 'author_name', 'similarity']])
```

### 3. Run Examples

```bash
# Recommend articles
python examples/recommend_articles.py "climate change global warming"

# Recommend books
python examples/recommend_books.py "science fiction space"
```

## Core Functions

### `recommend_by_embedding(...)`

Find articles similar to a query text.

**Parameters:**
- `query_text` (str): Query text
- `embeddings_path` (str): Path to embeddings .npy
- `id_map_csv` (str): Path to ID mapping CSV
- `articles_df_path` (str, optional): Path to articles DataFrame (for metadata)
- `top_k` (int): Number of recommendations (default: 5)
- `embed_model` (str): Model name (default: 'vinai/bertweet-base')
- `use_faiss` (bool): Use FAISS if available (default: True)

**Returns:**
- DataFrame with columns: `_id`, `similarity`, `headline`, `abstract`, etc.

**Example:**
```python
results = recommend_by_embedding(
    "technology artificial intelligence",
    top_k=10,
    use_faiss=True
)

# Top result
top = results.iloc[0]
print(f"Article: {top['headline']}")
print(f"Similarity: {top['similarity']:.4f}")
```

### `recommend_books(...)`

Recommend books based on query text.

**Parameters:**
- `query_text` (str): Query describing desired book
- `books_df_path` (str): Path to books DataFrame
- `book_embeddings_path` (str): Path to book embeddings
- `book_mapping_path` (str): Path to book ID mapping
- `top_k` (int): Number of recommendations (default: 5)
- `embed_model` (str): Model name (default: 'vinai/bertweet-base')
- `use_faiss` (bool): Use FAISS (default: True)

**Returns:**
- DataFrame with columns: `_id`, `similarity`, `book_title`, `author_name`, etc.

**Example:**
```python
results = recommend_books(
    "mystery thriller detective",
    books_df_path='data/books.parquet',
    book_embeddings_path='data/book_embeddings.npy',
    top_k=5
)

for _, book in results.iterrows():
    print(f"{book['book_title']} by {book['author_name']}")
    print(f"  Similarity: {book['similarity']:.4f}")
```

## FAISS vs NumPy

### FAISS (Recommended for Large Datasets)

**Pros:**
- 10-100x faster than NumPy
- Scales to millions of vectors
- Supports GPU acceleration
- Approximate search options

**Cons:**
- Requires installation (`pip install faiss-cpu`)
- Additional dependency

**When to use:**
- >10,000 articles
- Real-time search required
- Building production API

### NumPy (Fallback)

**Pros:**
- Always available (no extra install)
- Exact similarity scores
- Simple and reliable

**Cons:**
- Slower for large datasets
- Memory intensive

**When to use:**
- Small datasets (<10,000)
- Prototyping
- FAISS not available

### Installation

```bash
# CPU version (recommended)
pip install faiss-cpu

# GPU version (if CUDA available)
pip install faiss-gpu
```

## Advanced Usage

### Custom Similarity Computation

```python
from src.models.similarity import (
    compute_cosine_similarity_numpy,
    compute_cosine_similarity_faiss
)

# Load embeddings
embeddings = np.load('data/embeddings.npy')

# Query embedding (assume we have it)
query_emb = embeddings[0]

# NumPy version
indices_np, scores_np = compute_cosine_similarity_numpy(
    query_emb, embeddings, top_k=5
)

# FAISS version (if available)
indices_faiss, scores_faiss = compute_cosine_similarity_faiss(
    query_emb, embeddings, top_k=5
)

print(f"Top 5 indices: {indices_np}")
print(f"Top 5 scores: {scores_np}")
```

### Build and Save FAISS Index

```python
from src.models.similarity import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index
)

# Load embeddings
embeddings = np.load('data/embeddings.npy')

# Build index
index = build_faiss_index(embeddings, index_type='flat')

# Save to disk
save_faiss_index(index, 'data/faiss_index.bin')

# Later, load from disk
loaded_index = load_faiss_index('data/faiss_index.bin')

# Search
query = embeddings[0:1]  # First embedding as query
distances, indices = loaded_index.search(query, k=5)
```

### Batch Recommendations

```python
queries = [
    "stock market economy",
    "climate change environment",
    "technology innovation"
]

all_results = []

for query in queries:
    results = recommend_by_embedding(query, top_k=5)
    all_results.append(results)

# Combine results
combined = pd.concat(all_results, ignore_index=True)
print(combined[['_id', 'similarity', 'headline']])
```

## Performance Comparison

### Similarity Search Time (1,000,000 embeddings)

| Method | Search Time (1 query) | Search Time (100 queries) |
|--------|----------------------|--------------------------|
| NumPy | ~2.5 seconds | ~4 minutes |
| FAISS Flat | ~0.1 seconds | ~10 seconds |
| FAISS IVF | ~0.01 seconds | ~1 second |

### Memory Usage

| Dataset Size | Embeddings Size | Index Size (FAISS) |
|--------------|----------------|-------------------|
| 1,000 | 3 MB | 3 MB |
| 10,000 | 30 MB | 30 MB |
| 100,000 | 300 MB | 300 MB |
| 1,000,000 | 3 GB | 3 GB |

## Complete Workflow Example

```python
import pandas as pd
from src.ingest.load_nyt import load_nyt_csv
from src.preprocess.text import preprocess_dataframe
from src.models.embeddings import build_bertweet_embeddings
from src.models.similarity import recommend_by_embedding

# 1. Load and preprocess data
df = load_nyt_csv('data/nyt-metadata.csv')
df = preprocess_dataframe(df)

# 2. Generate embeddings
embeddings, mapping = build_bertweet_embeddings(
    df,
    text_col='cleaned_text',
    sample_limit=10000,
    output_dir='data'
)

# 3. Find similar articles
results = recommend_by_embedding(
    "economic crisis financial market",
    articles_df_path='data/preprocessed.parquet',
    top_k=10
)

# 4. Display results
for idx, row in results.iterrows():
    print(f"{idx+1}. {row['headline']}")
    print(f"   Similarity: {row['similarity']:.4f}")
    print()
```

## Book Recommendation Workflow

```python
from src.models.similarity import recommend_books
from src.models.embeddings import build_bertweet_embeddings

# 1. Filter books articles
books_df = df[df['section_name'] == 'Books'].copy()

# Note: You need to extract book_title and author_name
# This can be done using LLM extraction (instructor + OpenAI)
# See the original notebook for extraction examples

# 2. Generate book embeddings
book_embeddings, book_mapping = build_bertweet_embeddings(
    books_df,
    output_dir='data'
)

# Rename for clarity
# mv data/embeddings.npy data/book_embeddings.npy
# mv data/embeddings_mapping.csv data/book_embeddings_mapping.csv

# 3. Get book recommendations
results = recommend_books(
    "historical fiction novel",
    books_df_path='data/books_with_metadata.parquet',
    book_embeddings_path='data/book_embeddings.npy',
    top_k=5
)

print(results[['book_title', 'author_name', 'similarity']])
```

## Error Handling

### Embeddings Not Found

```python
from pathlib import Path

embeddings_path = Path('data/embeddings.npy')

if not embeddings_path.exists():
    print("Embeddings not found!")
    print("Generate them with:")
    print("  python scripts/build_embeddings.py --sample 1000")
```

### FAISS Not Available

```python
try:
    from src.models.similarity import recommend_by_embedding
    results = recommend_by_embedding("query", use_faiss=True)
except ImportError:
    print("FAISS not available, falling back to NumPy")
    results = recommend_by_embedding("query", use_faiss=False)
```

### Model Download Issues

```python
# Set HuggingFace cache directory
import os
os.environ['TRANSFORMERS_CACHE'] = '/path/to/cache'

results = recommend_by_embedding("query text")
```

## Integration with API

```python
from fastapi import FastAPI, Query
from src.models.similarity import recommend_by_embedding

app = FastAPI()

@app.get("/recommend")
async def recommend(
    q: str = Query(..., description="Query text"),
    top_k: int = Query(5, ge=1, le=100)
):
    results = recommend_by_embedding(q, top_k=top_k)
    return results.to_dict('records')

# Run with: uvicorn app:app --reload
```

## Troubleshooting

### Issue: Slow search with large dataset

```python
# Solution: Use FAISS with IVF index
from src.models.similarity import build_faiss_index

embeddings = np.load('data/embeddings.npy')
index = build_faiss_index(embeddings, index_type='ivf', nlist=100)
save_faiss_index(index, 'data/faiss_ivf_index.bin')
```

### Issue: Out of memory during search

```python
# Solution: Process in batches
def batch_search(queries, batch_size=10):
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        for q in batch:
            r = recommend_by_embedding(q, top_k=5)
            results.append(r)
    return pd.concat(results)
```

### Issue: Different results between NumPy and FAISS

```python
# This is expected! FAISS uses approximate search for speed.
# For exact results, use:
results = recommend_by_embedding(query, use_faiss=False)
```

## Next Steps

After implementing similarity search:
1. **Build REST API** (src/api/search.py)
2. **Add caching** (Redis, Memcached)
3. **Implement hybrid search** (combine with topic models)
4. **Add user feedback** (improve recommendations)
5. **Deploy to production** (Kubernetes, Docker)
