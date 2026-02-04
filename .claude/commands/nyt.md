# NYT Article Analytics Platform

## Project Overview
A full-stack analytics platform for analyzing New York Times articles. The platform processes up to 21M articles from the Kaggle corpus, generating semantic embeddings and enabling multi-modal search, topic discovery, sentiment analysis, and book extraction.

## Core Features
1. **Semantic + Keyword Hybrid Search** - Combine meaning-based and keyword matching for relevant results
2. **Topic Modeling** - LDA-based topic discovery for year+section combinations
3. **Sentiment Analysis** - Multi-model sentiment classification with confidence scores
4. **Wordcloud Visualization** - Visual representation of topic keywords
5. **Book Extraction** - Extract and verify book titles/authors from articles using LLM + web search

## Layered Architecture

```
Data Layer (Kaggle 21M CSV)
        ↓
Preprocessing (parquet + cleaned_text + word_count filter)
        ↓
Embeddings (FAISS IVF index + memmap vectors)
        ↓
API Layer (FastAPI with hybrid search, topic modeling, sentiment)
        ↓
Frontend (SPA with React/Vue - static files)
```

## Canonical File Map

### Core API
- `src/api/main.py` - FastAPI app, startup data loading, all endpoints

### Models & NLP
- `src/models/embeddings.py` - BERTweet embedding generation (`extract_embeddings_batch`, `build_bertweet_embeddings`)
- `src/models/similarity.py` - Cosine similarity + hybrid scoring
- `src/models/sentiment.py` - Multi-model sentiment inference (DistilBERT, RoBERTa)
- `src/models/topic_models.py` - Topic description generation
- `src/models/extraction.py` - LLM-based book extraction + Gemini verification

### Text Processing
- `src/preprocess/text.py` - Core cleaning (`clean_text`, `combine_text`)

### Scripts
- `scripts/preprocess_21m.py` - Chunked CSV→parquet preprocessing (new)
- `scripts/build_embeddings_21m.py` - Chunked embeddings + FAISS IVF index (new)
- `scripts/gcp_pipeline.sh` - GCE instance setup + 21M pipeline runner (new)
- `scripts/gcp_deploy.sh` - Docker → Cloud Run deployment (new)

### Deployment
- `Dockerfile` - Python 3.10 + deps, runs `uvicorn src.api.main:app` (new)

## Data Layout

### Source
- **Kaggle NYT Dataset**: 21M articles as CSV (~40 GB uncompressed)

### Processed Files
| File | Purpose | Size (21M) |
|---|---|---|
| `data/preprocessed_21m.parquet` | Cleaned articles with word_count filter | ~8 GB |
| `data/preprocessed_500K.parquet` | Existing 500K subset (fallback) | ~150 MB |
| `data/embeddings_21m.memmap` | 21M × 768 float32 vectors (no-copy access) | ~62 GB |
| `data/faiss_index_21m.bin` | FAISS IVF index with 200K training sample | ~2 GB |
| `data/embeddings_21m_mapping.csv` | _id → row index lookup | ~500 MB |

## Running the Platform

### Start API
```bash
make run-api
```
- Frontend: http://localhost:8000/app
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Processing 21M Dataset
```bash
# 1. Preprocess (chunked read + parquet write)
python scripts/preprocess_21m.py \
  --input /path/to/nyt_articles.csv \
  --output data/preprocessed_21m.parquet \
  --chunk-size 500000

# 2. Generate embeddings + FAISS index
python scripts/build_embeddings_21m.py \
  --input data/preprocessed_21m.parquet \
  --output-dir data \
  --batch-size 128 \
  --chunk-size 100000 \
  --gpu
```

### GCP Deployment
```bash
# Pipeline on GPU instance
bash scripts/gcp_pipeline.sh --create --gpu t4

# Deploy to Cloud Run
bash scripts/gcp_deploy.sh --project my-gcp-project --region us-central1
```

## Technical Conventions

### Embeddings
- **Model**: `vinai/bertweet-base` (BERTweet)
- **Pooling**: CLS token (first [CLS] hidden state)
- **Dimension**: 768
- **Max Length**: 128 tokens
- **Batch Size**: 128 (GPU) or 32 (CPU)

### Search
- **Query Embedding**: Same BERTweet model as corpus
- **Semantic Similarity**: Cosine similarity from embeddings
- **Keyword Matching**: BM25Okapi (disabled at 21M articles, falls back to semantic-only)
- **Hybrid Combination**: `score = alpha * semantic + (1-alpha) * keyword`
- **FAISS Search**: IVFFlat index with `nprobe=32` for 21M

### API Responses
- **Images**: Base64-encoded PNG with `data:image/png;base64,` prefix
- **Year Filter**: Applied post-load (all data loaded, filtered in-memory)
- **Active App**: `main.py` (not `app.py`)

### Environment (.env)
```
OPENAI_API_KEY=...        # GPT extraction
GEMINI_API_KEY=...        # Book verification
TAVILY_API_KEY=...        # Web search verification
KAGGLE_API_KEY=...        # Kaggle dataset download
```

## Current Status & Scaling

### What Works Now (500K)
- ✓ Full embeddings in RAM
- ✓ Flat-index cosine search
- ✓ BM25 keyword search
- ✓ All NLP pipelines

### What Changes at 21M
| Constraint | Bottleneck | Solution |
|---|---|---|
| RAM (40 GB CSV) | `pd.read_csv()` loads all | Chunked iteration with `chunksize` |
| RAM (62 GB embeddings) | `np.vstack()` accumulates | Pre-allocated `np.memmap` + direct writes |
| Startup (62 GB load) | `np.load(embeddings.npy)` | Persistent FAISS IVF index (~2 GB) |
| BM25 (30 GB tokenized) | `BM25Okapi` builds full index | Skip BM25, use semantic-only fallback |
| Search latency | Flat-index cosine over 21M | Pre-built FAISS IVF with `nprobe=32` |
| Model reloading | 2-3s load on every search call | Load once at startup, cache in globals |

## Key Dependencies
- **fastapi** - Web framework
- **transformers + torch** - BERTweet embeddings
- **faiss-cpu** - Vector index (use `faiss-gpu` on GPU instances)
- **rank-bm25** - Keyword search
- **sklearn, gensim, bertopic** - Topic modeling
- **pyarrow** - Efficient parquet I/O

---

**Last Updated**: 2026-02-04
