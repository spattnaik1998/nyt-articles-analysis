# BERTweet Embeddings Module

## Overview

The embeddings module (`src/models/embeddings.py`) generates semantic vector representations of NYT articles using BERTweet or other transformer models. These embeddings enable:

- **Similarity search**: Find articles similar to a query article
- **Recommendation systems**: Recommend related articles to users
- **Clustering**: Group articles by semantic similarity
- **Downstream ML tasks**: Use as features for classification, etc.

## Quick Start

### Generate Embeddings for 10 Test Articles

```bash
# Run the test script
python examples/test_embeddings_10_rows.py
```

This will:
1. Create 10 sample articles
2. Generate BERTweet embeddings
3. Save to `data/embeddings.npy` and `data/embeddings_mapping.csv`
4. Display similarity analysis

### Generate Embeddings from Preprocessed Data

```bash
# Activate environment
venv\Scripts\activate

# Generate for 1000 articles (recommended for testing)
python scripts/build_embeddings.py --input data/preprocessed.parquet --sample 1000

# Generate for all articles (may take hours)
python scripts/build_embeddings.py --input data/preprocessed.parquet --all --gpu
```

## Core Functions

### 1. `build_bertweet_embeddings(df, ...)`

Main function to generate embeddings from a DataFrame.

**Parameters:**
- `df` (DataFrame): Input data
- `text_col` (str): Column with text (default: 'combined_text')
- `model_name` (str): HuggingFace model (default: 'vinai/bertweet-base')
- `sample_limit` (int): Limit to N rows (default: None = all)
- `batch_size` (int): Batch size (default: 32, reduce if OOM)
- `pooling` (str): 'cls' or 'mean' (default: 'cls')
- `output_dir` (str): Save directory (default: 'data')

**Returns:**
- `embeddings` (ndarray): Shape (n_articles, embedding_dim)
- `mapping` (DataFrame): ID-to-index mapping

**Example:**
```python
from src.models.embeddings import build_bertweet_embeddings

df = pd.read_parquet('data/preprocessed.parquet')
embeddings, mapping = build_bertweet_embeddings(df, sample_limit=1000)

print(embeddings.shape)  # (1000, 768)
```

### 2. `load_embeddings(...)`

Load previously saved embeddings.

**Example:**
```python
from src.models.embeddings import load_embeddings

embeddings, mapping = load_embeddings(
    'data/embeddings.npy',
    'data/embeddings_mapping.csv'
)
```

### 3. `get_embedding_by_id(article_id, embeddings, mapping)`

Retrieve embedding for a specific article.

**Example:**
```python
from src.models.embeddings import get_embedding_by_id

emb = get_embedding_by_id('nyt://article/001', embeddings, mapping)
print(emb.shape)  # (768,)
```

## CLI Script Options

```bash
python scripts/build_embeddings.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `data/preprocessed.parquet` | Input Parquet file |
| `--output-dir` | `data` | Output directory |
| `--sample` | None | Number of articles to process |
| `--all` | False | Process all articles |
| `--text-col` | `cleaned_text` | Text column name |
| `--model` | `vinai/bertweet-base` | HuggingFace model |
| `--batch-size` | `32` | Batch size (reduce if OOM) |
| `--pooling` | `cls` | 'cls' or 'mean' pooling |
| `--gpu` | Auto | Force GPU usage |
| `--cpu` | Auto | Force CPU usage |

### Example Commands

```bash
# Small sample with GPU
python scripts/build_embeddings.py --sample 1000 --gpu

# Large dataset with CPU (slower but more memory)
python scripts/build_embeddings.py --all --cpu --batch-size 16

# Use different model
python scripts/build_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2

# Mean pooling instead of CLS
python scripts/build_embeddings.py --pooling mean
```

## Device Handling (CPU/GPU)

### Automatic Detection

The module automatically detects the best device:
- **CUDA GPU** (NVIDIA): Fastest, recommended for large datasets
- **MPS GPU** (Apple Silicon): Fast on M1/M2/M3 Macs
- **CPU**: Works everywhere, slower but no memory limits

### Manual Control

```bash
# Force GPU (fail if not available)
python scripts/build_embeddings.py --gpu

# Force CPU (even if GPU available)
python scripts/build_embeddings.py --cpu
```

### Memory Considerations

**GPU Memory Issues (OOM):**
```bash
# Reduce batch size
python scripts/build_embeddings.py --batch-size 8

# Use CPU instead
python scripts/build_embeddings.py --cpu
```

**CPU Memory Issues:**
```bash
# Process in smaller batches by section
python scripts/preprocess_sample.py --section "World" --all
python scripts/build_embeddings.py --input data/preprocessed.parquet --all
```

## Pooling Strategies

### CLS Token (default)

Uses the [CLS] token embedding from BERT-style models.

```python
embeddings, _ = build_bertweet_embeddings(df, pooling='cls')
```

**Pros:** Fast, standard approach
**Cons:** May lose some information from other tokens

### Mean Pooling

Averages all token embeddings (attention-weighted).

```python
embeddings, _ = build_bertweet_embeddings(df, pooling='mean')
```

**Pros:** Captures more context
**Cons:** Slightly slower

## Output Format

### Embeddings File (`embeddings.npy`)

NumPy array of shape `(n_articles, embedding_dim)`:
- BERTweet: 768 dimensions
- Sentence-Transformers: 384-768 dimensions

### Mapping File (`embeddings_mapping.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `_id` | str | Article ID (e.g., nyt://article/001) |
| `index` | int | Index in embeddings array |

## Usage Examples

### Similarity Search

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.models.embeddings import load_embeddings

# Load embeddings
embeddings, mapping = load_embeddings()

# Find articles similar to article at index 0
query_idx = 0
similarities = cosine_similarity([embeddings[query_idx]], embeddings)[0]

# Get top 10 most similar (excluding self)
top_10_indices = similarities.argsort()[-11:-1][::-1]

print("Top 10 similar articles:")
for idx in top_10_indices:
    article_id = mapping.iloc[idx]['_id']
    similarity = similarities[idx]
    print(f"  {article_id}: {similarity:.4f}")
```

### Article Recommendation

```python
def recommend_articles(article_id, embeddings, mapping, top_k=5):
    """Recommend similar articles"""
    from sklearn.metrics.pairwise import cosine_similarity

    # Get embedding for query article
    query_emb = get_embedding_by_id(article_id, embeddings, mapping)

    if query_emb is None:
        return []

    # Compute similarities
    similarities = cosine_similarity([query_emb], embeddings)[0]

    # Get top-k (excluding the query itself)
    top_indices = similarities.argsort()[-top_k-1:-1][::-1]

    # Return recommendations
    recommendations = []
    for idx in top_indices:
        rec_id = mapping.iloc[idx]['_id']
        score = similarities[idx]
        recommendations.append({'id': rec_id, 'score': score})

    return recommendations

# Use it
recs = recommend_articles('nyt://article/001', embeddings, mapping, top_k=5)
for rec in recs:
    print(f"{rec['id']}: {rec['score']:.4f}")
```

### Clustering

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load embeddings
embeddings, mapping = load_embeddings()

# Cluster into 8 groups
kmeans = KMeans(n_clusters=8, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Add cluster labels to mapping
mapping['cluster'] = clusters

# Analyze clusters
print(mapping['cluster'].value_counts())

# Visualize with PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='tab10', alpha=0.6)
plt.title('Article Clusters (PCA visualization)')
plt.show()
```

## Performance Benchmarks

### Processing Time (vinai/bertweet-base)

| Articles | GPU (RTX 3090) | CPU (16 cores) | Notes |
|----------|---------------|----------------|-------|
| 100 | ~5 seconds | ~30 seconds | |
| 1,000 | ~40 seconds | ~5 minutes | |
| 10,000 | ~6 minutes | ~45 minutes | |
| 100,000 | ~60 minutes | ~7.5 hours | Batch processing recommended |

### Memory Usage

| Articles | Embeddings Size | GPU Memory | System RAM |
|----------|----------------|------------|------------|
| 1,000 | ~3 MB | ~1 GB | ~2 GB |
| 10,000 | ~30 MB | ~1.5 GB | ~4 GB |
| 100,000 | ~300 MB | ~2 GB | ~8 GB |
| 1,000,000 | ~3 GB | ~3 GB | ~16 GB |

## Supported Models

### BERTweet (Recommended for Social/News Text)

```bash
python scripts/build_embeddings.py --model vinai/bertweet-base
```
- Optimized for Twitter/informal text
- 768 dimensions
- ~500 MB model size

### Sentence Transformers (General Purpose)

```bash
python scripts/build_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
```
- Fast and efficient
- 384 dimensions
- ~80 MB model size

### Other HuggingFace Models

Any `AutoModel` compatible model:
```bash
python scripts/build_embeddings.py --model bert-base-uncased
python scripts/build_embeddings.py --model roberta-base
```

## Troubleshooting

### Issue: CUDA out of memory

```bash
# Reduce batch size
python scripts/build_embeddings.py --batch-size 8

# Use CPU
python scripts/build_embeddings.py --cpu
```

### Issue: Model download fails

```bash
# Set HuggingFace cache directory
export TRANSFORMERS_CACHE=/path/to/cache
python scripts/build_embeddings.py
```

### Issue: Process killed (OOM)

```bash
# Process in smaller chunks
python scripts/build_embeddings.py --sample 10000
# Then process more batches separately
```

### Issue: Slow on CPU

```bash
# Reduce batch size for better memory usage
python scripts/build_embeddings.py --cpu --batch-size 16

# Or use a smaller model
python scripts/build_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
```

## Integration with Pipeline

```
Data Ingestion → Preprocessing → Embeddings ← You are here
                                      ↓
                            Similarity Search / Recommendations
```

## Next Steps

After generating embeddings:
1. **Build similarity search API** (src/api/search.py)
2. **Create recommendation engine** (src/api/recommend.py)
3. **Add to vector database** (Faiss, Milvus, etc.)
4. **Combine with topic models** for hybrid search
