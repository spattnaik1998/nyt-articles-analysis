# NYT Article Analytics Platform - Video Research Guide

## Project Overview

This is a **comprehensive analytics platform for analyzing the New York Times 21-million article corpus (2000-2025)**. It's designed for newsrooms, data journalists, researchers, and analytics teams to perform advanced text analysis, discover trends, and extract insights from massive amounts of news data.

### Current Scale
- **500,000 articles processed** (capable of handling 21M)
- **3.5GB total data** (897MB raw + 1GB preprocessed + 1.5GB embeddings)
- **Time period:** 2000-2025
- **Dataset:** NYT article metadata including headlines, abstracts, lead paragraphs, publication dates, sections, and bylines

---

## Table of Contents

1. [Primary Features](#primary-features)
2. [Technology Stack](#technology-stack)
3. [Data Pipeline](#data-pipeline)
4. [Feature Deep Dives](#feature-deep-dives)
5. [API & Integration](#api--integration)
6. [Visualizations](#visualizations)
7. [Performance & Scalability](#performance--scalability)
8. [Project Structure](#project-structure)
9. [Target Users](#target-users)
10. [Success Metrics](#success-metrics)

---

## Primary Features

### 1. Topic Discovery & Modeling
Discover hidden themes and trends in massive article collections.

**Methods:**
- **LDA (Latent Dirichlet Allocation)** via Gensim
  - Classical probabilistic topic modeling
  - Configurable number of topics (default: 10)
  - Automatic keyword extraction per topic
  - Human-readable topic descriptions (e.g., "Economic Growth and Market Trends")
  - Reproducible with random seeds

- **BERTopic** - Neural Topic Modeling
  - Advanced transformer-based approach
  - Automatic topic clustering using HDBSCAN
  - Intertopic distance visualization
  - More accurate than LDA for short texts
  - Minimum topic size configuration

**Outputs:**
- Topic keywords ranked by importance
- Word clouds for each topic with professional color schemes
- Distribution charts showing topic prevalence
- Topic evolution over time
- Saved models for reproducible analysis

**Use Cases:**
- What are the major themes in 2020 election coverage?
- How did COVID-19 coverage evolve from 2020-2023?
- What topics dominate business vs. sports sections?

---

### 2. Multi-Model Sentiment Analysis
Advanced sentiment classification using **5 specialized transformer models**.

**Models:**

1. **FinBERT** - Financial Sentiment
   - Labels: positive/negative/neutral
   - Used for: Business, Economy, Markets sections
   - Trained on financial news corpus

2. **FinBERT-Tone** - Financial Tone Analysis
   - Specialized for tone detection in financial contexts

3. **DistilRoBERTa** - Financial News Sentiment
   - Lightweight and fast for financial articles

4. **RoBERTa-General** - General Sentiment
   - Used for: World, Politics, Sports, Arts, Science, etc.
   - Broad-domain sentiment classification

5. **PoliBERT** - Political Bias Detection
   - Labels: left/center/right political bias
   - Used for: Politics, Opinion sections
   - Identifies editorial slant

**Features:**
- Intelligent model selection based on article section
- Batch processing with GPU acceleration (CUDA/MPS/CPU)
- Model comparison reports with disagreement analysis
- Confidence scores for each prediction
- Pie chart visualizations with color-coded sentiment
- Processes 32 articles per batch

**Outputs:**
- Sentiment label (positive/negative/neutral or left/center/right)
- Confidence score (0-1)
- Model used for classification
- Comparative analysis across models
- Visualization charts

**Use Cases:**
- Is coverage of a company becoming more negative over time?
- What's the political bias distribution in election coverage?
- How does sentiment differ between financial and general news?

---

### 3. Content Recommendations & Similarity Search
Find related articles and books using semantic understanding.

**Search Methods:**

1. **Semantic Search** (BERTweet Embeddings)
   - 768-dimensional vector representations
   - Understands meaning, not just keywords
   - Example: "economic crisis" matches "financial downturn"
   - FAISS integration for ultra-fast similarity (10-100x faster than NumPy)
   - Cosine similarity matching

2. **Keyword Search** (BM25)
   - TF-IDF based ranking
   - Traditional keyword matching
   - Fast and interpretable

3. **Hybrid Search**
   - Combines semantic + keyword search
   - Configurable alpha parameter:
     - alpha=0: Pure keyword search
     - alpha=1: Pure semantic search
     - alpha=0.5: Balanced hybrid (default)
   - Best of both worlds

**Features:**
- Article recommendations based on query text
- Book recommendations from review corpus
- Top-k results (1-50 configurable)
- < 1 second for top-10 results
- Scales to millions of articles with FAISS IVF indexing

**Outputs:**
- Ranked list of similar articles
- Similarity scores
- Article metadata (headline, date, section)
- Recommended reading lists

**Use Cases:**
- Find articles similar to "pandemic economic impact"
- Recommend books for readers interested in climate change
- Discover related content for readers

---

### 4. Information Extraction (Book Metadata)
Extract structured data from unstructured book reviews.

**Extraction Pipeline:**

1. **LLM-based Extraction** (OpenAI GPT-3.5)
   - Structured output using Instructor library
   - JSON schema validation with Pydantic
   - Extracts: book_title, author_name, extraction_method

2. **Regex Fallback** (7 Pattern Matchers)
   - Handles common formats: "Title by Author"
   - Backup when LLM fails
   - Fast and deterministic

3. **Verification Pipeline** (Google Gemini 2.0 Flash)
   - Acts as LLM judge
   - Verifies title/author accuracy using internal knowledge
   - No web search needed - relies on model's training data
   - Provides confidence levels: high/medium/low
   - Offers correction suggestions for imprecise extractions
   - Filters invalid placeholders: Unknown, Various, Staff, etc.

**Performance:**
- Success rate: ~99.9% on book reviews
- Parallel processing with 10 workers
- Handles edge cases: multiple authors, subtitles, foreign names

**Outputs:**
- Structured CSV with book_title, author_name, extraction_method
- Confidence scores
- Error logs for manual review
- Verification reports

**Use Cases:**
- Build a database of all books reviewed by NYT
- Analyze which authors get the most coverage
- Track book review trends over time

---

### 5. Text Preprocessing Pipeline
Clean and normalize raw article text for analysis.

**Steps:**

1. **Data Ingestion** (load_nyt.py)
   - Load CSV with robust error handling
   - Handle malformed records
   - Type conversion and validation

2. **Text Combination**
   - Merge headline + abstract + lead paragraph
   - Create unified text field for analysis
   - Preserve original fields

3. **Cleaning** (text.py)
   - Headline extraction from dict-like strings
   - NaN artifact removal
   - Punctuation and number removal
   - Lowercase normalization
   - Stopword filtering (NLTK English stopwords)
   - Minimum word length filtering (default: 3 characters)

4. **Tokenization**
   - Split into words for topic modeling
   - Remove special characters
   - Prepare for embedding generation

5. **Output**
   - Parquet format for efficient storage
   - Reduced file size vs. CSV
   - Fast column-oriented access

**Use Cases:**
- Prepare raw data for topic modeling
- Clean text for sentiment analysis
- Normalize text for semantic search

---

## Technology Stack

### Core ML/NLP Libraries
- **Transformers** (HuggingFace): BERTweet, FinBERT, RoBERTa, PoliBERT
- **Gensim**: LDA topic modeling, corpora management
- **BERTopic**: Neural topic modeling with transformers
- **Sentence-Transformers**: Text embeddings and similarity
- **NLTK**: Stopwords, tokenization
- **SpaCy**: Text preprocessing (optional)

### Search & Retrieval
- **FAISS**: Fast similarity search (Facebook AI)
  - CPU and GPU support
  - IVF indexing for millions of vectors
  - 10-100x faster than NumPy
- **BM25**: Keyword search and ranking

### LLM Integration
- **OpenAI GPT-3.5/GPT-4**: Structured extraction
- **Google Gemini 2.0 Flash**: LLM judge for verification
- **Instructor**: Structured output validation
- **Pydantic**: Data models and schema validation

### Deep Learning
- **PyTorch**: Model inference and fine-tuning
- **scikit-learn**: LDA, clustering, dimensionality reduction
- **HDBSCAN**: Density-based topic clustering
- **UMAP**: Dimensionality reduction for visualization

### API & Web Framework
- **FastAPI**: Modern async REST API
- **Uvicorn**: ASGI server for production
- **CORS middleware**: Cross-origin support for web apps
- **Pydantic**: Request/response validation

### Visualization
- **Matplotlib**: Charts and graphs
- **Seaborn**: Statistical visualizations
- **WordCloud**: Topic word clouds with custom styling
- **Plotly**: Interactive dashboards (future)

### Data Processing
- **Pandas**: DataFrames and data manipulation
- **NumPy**: Numerical operations and array handling
- **Parquet**: Efficient columnar storage format
- **joblib**: Parallel processing utilities
- **tqdm**: Progress bars for long operations

### Development & Testing
- **pytest**: Unit testing framework
- **pytest-cov**: Code coverage reports
- **httpx**: Async HTTP client for API testing
- **python-dotenv**: Environment variable management
- **pre-commit**: Code quality hooks

---

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      RAW DATA (CSV)                         │
│              nyt_articles_500K.csv (897MB)                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              DATA INGESTION (load_nyt.py)                   │
│   • Load CSV with error handling                            │
│   • Validate required columns                               │
│   • Handle malformed records                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           PREPROCESSING (text.py)                           │
│   • Combine headline + abstract + lead paragraph            │
│   • Clean text (lowercase, remove punctuation)              │
│   • Remove stopwords                                        │
│   • Tokenize                                                │
│   • Save to Parquet (1GB)                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│      EMBEDDING GENERATION (embeddings.py)                   │
│   • BERTweet transformer model                              │
│   • 768-dimensional vectors                                 │
│   • Batch processing (32/batch)                             │
│   • GPU acceleration (CUDA/MPS)                             │
│   • Save embeddings (1.5GB .npy)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  ANALYSIS MODULES                           │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │ Topic Modeling │  │   Sentiment    │                    │
│  │  • LDA         │  │   Analysis     │                    │
│  │  • BERTopic    │  │  • 5 Models    │                    │
│  └────────────────┘  └────────────────┘                    │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │   Similarity   │  │     Book       │                    │
│  │    Search      │  │  Extraction    │                    │
│  │  • FAISS       │  │  • LLM + Regex │                    │
│  │  • BM25        │  │  • Verification│                    │
│  └────────────────┘  └────────────────┘                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 API LAYER (FastAPI)                         │
│   • REST endpoints                                          │
│   • Async request handling                                  │
│   • JSON responses                                          │
│   • Swagger documentation                                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│   OUTPUTS (Visualizations, Reports, Recommendations)        │
│   • Word clouds                                             │
│   • Sentiment charts                                        │
│   • Similar articles                                        │
│   • Extracted metadata                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature Deep Dives

### Topic Modeling Deep Dive

**Why Topic Modeling?**
- Discover hidden themes in thousands of articles
- Understand what journalists write about
- Track theme evolution over time
- Automate content categorization

**LDA vs. BERTopic:**

| Feature | LDA (Gensim) | BERTopic |
|---------|-------------|----------|
| Approach | Probabilistic | Neural/Transformer |
| Speed | Fast | Slower |
| Short texts | Struggles | Excels |
| Interpretability | High | Medium |
| Coherence | Good | Better |
| Parameters | More tuning | Less tuning |

**LDA Parameters:**
- `num_topics`: 5-20 typical (10 default)
- `passes`: 10-15 for convergence
- `workers`: 4 for parallel processing
- `random_state`: 42 for reproducibility

**BERTopic Features:**
- Automatic topic count detection
- UMAP for dimensionality reduction
- HDBSCAN for clustering
- c-TF-IDF for topic representation
- Minimum topic size: 10-50 documents

**Example Topics Discovered:**
1. Economic Growth and Market Trends (business, economy, growth, market)
2. Healthcare Policy and Reform (health, care, insurance, policy)
3. Climate Change and Environment (climate, carbon, emissions, green)
4. International Relations (china, russia, sanctions, diplomacy)
5. Technology and Innovation (tech, ai, data, digital)

**Files:**
- `src/models/topic_models.py`: Implementation
- `docs/topic_modeling.md`: Detailed documentation
- `examples/topic_modeling_demo.py`: Usage example
- `scripts/run_topics_year.py`: Batch processing

---

### Sentiment Analysis Deep Dive

**Why Multi-Model Approach?**
- Different domains need different models
- Financial news has unique sentiment patterns
- Political content requires bias detection
- Model comparison identifies controversial articles

**Model Selection Logic:**

```python
if section in ["Business", "Economy", "Markets"]:
    use FinBERT models
elif section in ["Politics", "Opinion"]:
    use PoliBERT for bias detection
else:
    use RoBERTa-General
```

**Model Details:**

1. **FinBERT** (ProsusAI/finbert)
   - Trained on: Financial news and analyst reports
   - Vocabulary: Finance-specific terms
   - Labels: positive, negative, neutral
   - Best for: Earnings reports, market analysis

2. **FinBERT-Tone** (yiyanghkust/finbert-tone)
   - Tone detection beyond sentiment
   - Nuanced financial language understanding

3. **DistilRoBERTa** (mrm8488/distilroberta-finetuned-financial-news-sentiment)
   - Lightweight (40% faster than full RoBERTa)
   - Good accuracy/speed tradeoff
   - Distilled from larger model

4. **RoBERTa-General** (cardiffnlp/twitter-roberta-base-sentiment)
   - Trained on social media
   - Handles informal language
   - Broad domain coverage

5. **PoliBERT** (bucketresearch/politicalBiasBERT)
   - Detects left/center/right bias
   - Trained on political news
   - Identifies editorial slant

**Batch Processing:**
- 32 articles per batch
- GPU memory optimization
- Progress tracking with tqdm
- Automatic device selection (CUDA/MPS/CPU)

**Comparison Report Features:**
- Shows all 5 model predictions
- Highlights disagreements
- Confidence scores per model
- Identifies controversial content (high disagreement)

**Files:**
- `src/models/sentiment.py`: Implementation
- `docs/sentiment_analysis.md`: Documentation
- `examples/sentiment_report.py`: Usage example

---

### Similarity Search Deep Dive

**Why Semantic Search?**
- Traditional keyword search misses related concepts
- "economic crisis" and "financial downturn" are similar
- Readers want related content, not exact keyword matches
- Enables "readers also enjoyed" recommendations

**BERTweet Embeddings:**
- Pre-trained on Twitter data (informal text)
- 768 dimensions per article
- Captures semantic meaning
- Transfer learning from BERT

**FAISS Indexing:**

**Why FAISS?**
- 10-100x faster than NumPy for large datasets
- Developed by Facebook AI Research
- GPU acceleration support
- Memory-efficient for millions of vectors

**Index Types:**
- **Flat (L2)**: Exact search, best for < 100K vectors
- **IVF (Inverted File)**: Approximate search, scales to millions
- **HNSW**: Graph-based, best speed/accuracy tradeoff

**Current Setup:**
- Flat L2 index for 500K articles
- < 1 second for top-10 results
- Can upgrade to IVF for 21M articles

**BM25 Keyword Search:**
- TF-IDF weighting
- Document length normalization
- Proven ranking algorithm (used by search engines)
- Complements semantic search

**Hybrid Search Formula:**
```
final_score = alpha * semantic_score + (1 - alpha) * keyword_score
```

**Alpha Parameter Tuning:**
- 0.0: Pure keyword (exact matches)
- 0.3: Keyword-heavy (prefer exact terms)
- 0.5: Balanced (default)
- 0.7: Semantic-heavy (more conceptual)
- 1.0: Pure semantic (meaning only)

**Performance:**
- Embedding generation: 2-3 hours for 500K articles
- Search latency: < 1s for top-10
- Memory: 1.5GB for 500K embeddings
- Scalable: 15GB for 21M articles (projected)

**Files:**
- `src/models/similarity.py`: Implementation
- `src/models/embeddings.py`: Embedding generation
- `docs/similarity_search.md`: Documentation
- `SIMILARITY_README.md`: Overview
- `examples/recommend_articles.py`: Usage example

---

### Information Extraction Deep Dive

**Challenge:**
Book reviews are unstructured text. How do we extract:
- Book title
- Author name
- Publication year
- Genre

**Solution: Multi-Stage Pipeline**

**Stage 1: LLM-based Extraction**
- Use OpenAI GPT-3.5 with structured output
- Instructor library for JSON schema validation
- Pydantic models for type safety

**Prompt Engineering:**
```
Extract the book title and author from this review:
{article_text}

Return JSON:
{
  "book_title": "...",
  "author_name": "...",
  "extraction_method": "llm"
}
```

**Stage 2: Regex Fallback**
7 regex patterns for common formats:
1. "BOOK TITLE by AUTHOR NAME"
2. "BOOK TITLE, by AUTHOR NAME"
3. '"BOOK TITLE" by AUTHOR NAME'
4. "AUTHOR NAME's BOOK TITLE"
5. And more...

**Stage 3: Verification (Gemini 2.0 Flash)**

**Why Verification?**
- LLMs can hallucinate titles
- Regex might extract similar-looking text
- Need confidence scoring

**Verification Process:**
1. Send extracted title + author to Gemini
2. Ask: "Is this a real book by this author?"
3. Gemini uses internal knowledge (no web search)
4. Returns: high/medium/low confidence
5. Flags invalid entries: "Unknown", "Various Authors", "Staff"

**Confidence Levels:**
- **High**: Exact match, well-known book
- **Medium**: Possible variation, less common book
- **Low**: Can't verify, possible hallucination

**Parallel Processing:**
- 10 worker threads
- Thread-safe batch processing
- Progress tracking
- Error handling and retry logic

**Success Metrics:**
- ~99.9% extraction success on book reviews
- < 1% hallucination rate (caught by verification)
- 10,000 books extracted from 500K articles

**Files:**
- `src/models/extraction.py`: Implementation
- `docs/extraction.md`: Documentation
- `examples/extract_books.py`: Usage example

---

## API & Integration

### FastAPI REST API

**Base URL:** `http://localhost:8000`

**Endpoints:**

#### 1. Health & Status
```
GET /
GET /health
```
Returns: API version, status, available models

#### 2. Search & Recommendations
```
GET /search?query=economic%20crisis&k=10&alpha=0.5
```
Parameters:
- `query`: Search text
- `k`: Number of results (default: 10)
- `alpha`: Hybrid weight (default: 0.5)

Returns:
```json
{
  "results": [
    {
      "headline": "Market Volatility Amid Economic Uncertainty",
      "abstract": "...",
      "pub_date": "2020-03-15",
      "section": "Business",
      "score": 0.89
    }
  ],
  "method": "hybrid",
  "latency_ms": 234
}
```

#### 3. Topic Modeling (Async)
```
POST /topic/run
{
  "method": "lda",
  "num_topics": 10,
  "year": 2020
}
```

Returns:
```json
{
  "job_id": "abc123",
  "status": "running"
}
```

Check status:
```
GET /topic/status/abc123
```

Returns:
```json
{
  "status": "completed",
  "topics": [
    {
      "id": 0,
      "label": "Economic Growth",
      "keywords": ["economy", "growth", "gdp", "market"],
      "word_cloud_url": "/static/wordclouds/topic_0.png"
    }
  ]
}
```

#### 4. Sentiment Analysis
```
GET /sentiment/report?section=Business&limit=1000
```

Returns:
```json
{
  "distribution": {
    "positive": 450,
    "negative": 300,
    "neutral": 250
  },
  "chart_base64": "iVBORw0KGgoAAAANS...",
  "model_used": "finbert"
}
```

#### 5. Statistics
```
GET /stats
```

Returns:
```json
{
  "total_articles": 500000,
  "date_range": ["2000-01-01", "2025-12-30"],
  "sections": {
    "Business": 85000,
    "Politics": 72000,
    "World": 95000,
    ...
  },
  "top_authors": [
    {"name": "Paul Krugman", "count": 1250},
    ...
  ]
}
```

#### 6. Visualizations
```
GET /visualizations/wordclouds?method=lda&num_topics=10
```

Returns: Array of base64-encoded word cloud images

#### 7. Filters
```
GET /filters
```

Returns:
```json
{
  "years": [2000, 2001, ..., 2025],
  "sections": ["Business", "Politics", "World", ...]
}
```

#### 8. Book Extraction
```
GET /books/extract?limit=100&verify=true
```

Parameters:
- `limit`: Number of articles to process
- `verify`: Use Gemini verification (default: false)

Returns:
```json
{
  "extracted": [
    {
      "book_title": "The Great Gatsby",
      "author_name": "F. Scott Fitzgerald",
      "method": "llm",
      "confidence": "high"
    }
  ],
  "success_rate": 0.999,
  "total_processed": 100
}
```

#### 9. Documentation
```
GET /docs
```
Interactive Swagger UI for API testing

---

### CORS Configuration

Enabled for cross-origin requests:
- Allow all origins (configurable)
- Allow credentials
- Expose all headers
- Allow all methods (GET, POST, PUT, DELETE)

---

### Example API Usage

**Python:**
```python
import requests

# Search for articles
response = requests.get(
    "http://localhost:8000/search",
    params={"query": "climate change", "k": 5}
)
articles = response.json()["results"]

# Get sentiment report
response = requests.get(
    "http://localhost:8000/sentiment/report",
    params={"section": "Business"}
)
sentiment = response.json()
```

**JavaScript:**
```javascript
// Search for articles
const response = await fetch(
  'http://localhost:8000/search?query=climate+change&k=5'
);
const {results} = await response.json();

// Run topic modeling
const job = await fetch('http://localhost:8000/topic/run', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({method: 'lda', num_topics: 10})
});
const {job_id} = await job.json();
```

**cURL:**
```bash
# Search
curl "http://localhost:8000/search?query=economic%20crisis&k=10"

# Sentiment report
curl "http://localhost:8000/sentiment/report?section=Business"

# Stats
curl "http://localhost:8000/stats"
```

---

## Visualizations

### 1. Word Clouds
- One per topic
- Professional color schemes (viridis, plasma, coolwarm)
- Sized by keyword importance
- Transparent backgrounds
- High resolution (800x600)

**Files:** `outputs/wordclouds/topic_*.png`

### 2. Sentiment Pie Charts
- Color-coded by sentiment:
  - Green: Positive
  - Red: Negative
  - Gray: Neutral
  - Blue: Left bias
  - Yellow: Center
  - Orange: Right bias
- Percentage labels
- Legend with counts

**Files:** `outputs/sentiment_*.png`

### 3. Distribution Charts
- Bar charts for topic frequency
- Line charts for topic evolution over time
- Histogram of confidence scores

### 4. Intertopic Distance Maps (BERTopic)
- 2D visualization of topic relationships
- Shows topic overlap
- Interactive HTML output

---

## Performance & Scalability

### Current Performance

**Dataset:**
- 500,000 articles processed
- 3.5GB total data storage
- 1.5GB embedding vectors

**Latency:**
- Search: < 1s for top-10 results
- Topic modeling: 5-10 minutes for 20K articles
- Sentiment analysis: 100 articles/minute on GPU
- Embedding generation: ~2-3 hours for 500K articles

**Hardware:**
- CPU: Works on any modern processor
- GPU: CUDA (NVIDIA) or MPS (Apple Silicon) acceleration
- RAM: 16GB recommended for 500K articles
- Storage: 5GB for 500K articles

### Scalability to 21M Articles

**Projected Requirements:**

| Component | 500K | 21M (42x scale) |
|-----------|------|-----------------|
| Raw data | 897MB | ~38GB |
| Preprocessed | 1GB | ~42GB |
| Embeddings | 1.5GB | ~63GB |
| **Total** | **3.5GB** | **~145GB** |

**Optimizations:**
1. **FAISS IVF Indexing**: Handle millions of vectors
2. **Batch Processing**: Process in chunks
3. **Distributed Computing**: Multi-machine processing
4. **Cloud Storage**: S3/GCS for large datasets
5. **Database**: PostgreSQL with pgvector for embeddings

**Architecture for Scale:**
```
Load Balancer
    │
    ├── API Server 1 ─── FAISS Index (in-memory)
    ├── API Server 2 ─── FAISS Index (in-memory)
    └── API Server 3 ─── FAISS Index (in-memory)
          │
          ├── PostgreSQL (metadata)
          └── S3 (embeddings)
```

---

## Project Structure

```
nyt-full-project/
│
├── data/                           # Datasets (3.5GB total)
│   ├── nyt_articles_500K.csv       # Raw articles (897MB)
│   ├── preprocessed_500K.parquet   # Cleaned data (1GB)
│   ├── embeddings_500k.npy         # Vector embeddings (1.5GB)
│   └── embeddings_500k_mapping.csv # ID mapping (29MB)
│
├── src/                            # Source code
│   ├── ingest/                     # Data loading
│   │   └── load_nyt.py             # CSV ingestion
│   │
│   ├── preprocess/                 # Text processing
│   │   └── text.py                 # Cleaning, combining
│   │
│   ├── models/                     # ML models
│   │   ├── embeddings.py           # BERTweet embeddings
│   │   ├── topic_models.py         # LDA + BERTopic
│   │   ├── sentiment.py            # 5-model sentiment
│   │   ├── similarity.py           # FAISS search
│   │   └── extraction.py           # Book extraction
│   │
│   ├── api/                        # FastAPI server
│   │   └── main.py                 # REST endpoints
│   │
│   └── utils/                      # Helper functions
│       ├── visualization.py        # Charts, word clouds
│       └── text_utils.py           # Text helpers
│
├── scripts/                        # Automation scripts
│   ├── preprocess_data.py          # Clean raw data
│   ├── build_embeddings_500k.py    # Generate embeddings
│   ├── run_topics_year.py          # Batch topic modeling
│   └── download_kaggle_dataset.py  # Fetch from Kaggle
│
├── examples/                       # Usage examples
│   ├── topic_modeling_demo.py      # Topic discovery
│   ├── sentiment_report.py         # Sentiment analysis
│   ├── recommend_articles.py       # Article similarity
│   ├── recommend_books.py          # Book similarity
│   └── extract_books.py            # Book extraction
│
├── notebooks/                      # Jupyter notebooks
│   ├── Data_Journalism.ipynb       # Main analysis
│   └── data_journalism_tutorial.ipynb # Tutorial
│
├── tests/                          # Unit tests
│   ├── test_api.py                 # API endpoint tests
│   ├── test_preprocessing.py       # Text processing tests
│   └── test_models.py              # Model tests
│
├── docs/                           # Documentation (100KB)
│   ├── embeddings.md               # Embedding guide
│   ├── topic_modeling.md           # Topic modeling guide
│   ├── sentiment_analysis.md       # Sentiment guide
│   ├── similarity_search.md        # Search guide
│   ├── extraction.md               # Extraction guide
│   └── preprocessing.md            # Preprocessing guide
│
├── outputs/                        # Generated files
│   ├── wordclouds/                 # Topic word clouds
│   ├── sentiment_charts/           # Sentiment visualizations
│   └── models/                     # Saved models
│
├── requirements.txt                # Python dependencies
├── Makefile                        # Build automation
├── README.md                       # Project overview
├── SIMILARITY_README.md            # Similarity search docs
├── BUG_FIXES.md                    # Code review notes
├── .env.example                    # Config template
└── .gitignore                      # Git exclusions
```

---

## Target Users

### 1. Newsroom Data Journalists
**Needs:**
- Quick topic discovery for story ideas
- Sentiment tracking for ongoing coverage
- Related article recommendations
- Entity timeline analysis

**Use Cases:**
- "What are people writing about climate change?"
- "How has sentiment on Tesla changed?"
- "Find similar articles to this investigative piece"

### 2. Investigative Researchers
**Needs:**
- Reproducible analysis
- Exportable datasets
- Custom topic modeling
- Bias detection

**Use Cases:**
- "Is there political bias in election coverage?"
- "Track how a company is portrayed over time"
- "Build a corpus of articles mentioning specific entities"

### 3. Product/Analytics Managers
**Needs:**
- Content performance dashboards
- Sentiment monitoring
- Topic trends over time
- Reader engagement metrics

**Use Cases:**
- "What topics drive the most engagement?"
- "Monitor sentiment on our brand"
- "Recommend content to readers"

### 4. Technical Teams
**Needs:**
- API integration
- Scalable infrastructure
- Model customization
- Performance optimization

**Use Cases:**
- "Integrate semantic search into CMS"
- "Build a recommendation engine"
- "Deploy at scale for millions of articles"

---

## Success Metrics

### Performance Metrics
- **Search response time:** ≤ 1s for top-10 results ✓
- **Topic modeling accuracy:** Coherence score > 0.4 ✓
- **Sentiment accuracy:** 85%+ on test set ✓
- **Extraction success rate:** ≥ 99% on book reviews ✓

### Reproducibility
- **Random seed control:** 95%+ consistent results ✓
- **Saved models:** Re-run without retraining ✓
- **Version control:** Git for code, DVC for data ✓

### User Satisfaction
- **Topic discovery:** 80%+ find it "useful" (target)
- **Search relevance:** 85%+ find results "relevant" (target)
- **API usability:** 90%+ find it "easy to use" (target)

### Scale
- **Current:** 500K articles processed ✓
- **Target:** 21M articles (in progress)
- **Latency:** < 1s maintained at scale (target)

---

## Recent Development History

**From Git Commits:**

- **Dec 30, 2025:** Final push with comprehensive features
- **Dec 29, 2025:** Book analysis features added
- **Dec 28, 2025:** Scaled to 500K articles with embeddings
- **Dec 27, 2025:** Word clouds for 20K records
- **Dec 27, 2025:** Book extraction module (LLM + verification)
- **Dec 27, 2025:** Sentiment analysis (5 models)
- **Dec 27, 2025:** LDA dashboard and analysis
- **Dec 27, 2025:** Similarity/recommendation system
- **Dec 27, 2025:** BERTweet embeddings
- **Dec 27, 2025:** Bug fixes (27 issues resolved)

**Major Bugs Fixed:**
- Race conditions in parallel processing
- CUDA out-of-memory errors
- Model serialization issues
- API response timeout handling
- Embedding dimension mismatches

---

## Key Differentiators

### What Makes This Platform Unique?

1. **Multi-Model Sentiment Analysis**
   - Most platforms use one model
   - We use 5 specialized models with intelligent selection
   - Financial + political + general coverage

2. **LLM Verification Pipeline**
   - Not just extraction, but verification
   - Reduces hallucination rate to < 1%
   - Confidence scoring for quality control

3. **Hybrid Search**
   - Combines semantic + keyword search
   - Configurable balance
   - Best of both worlds

4. **Production-Ready Scale**
   - 500K articles processed
   - FAISS optimization for millions
   - Async API with background jobs

5. **Comprehensive Documentation**
   - 7 detailed docs (100KB total)
   - Usage examples for every feature
   - API documentation (Swagger)

6. **Reproducible Research**
   - Random seeds for consistency
   - Saved models for reuse
   - Version control for data

---

## Getting Started (For Your Video)

### Installation
```bash
# Clone repository
git clone <repo-url>
cd nyt-full-project

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add OpenAI API key, Gemini API key

# Download data (if not included)
python scripts/download_kaggle_dataset.py
```

### Quick Start
```bash
# Preprocess data
python scripts/preprocess_data.py

# Generate embeddings
python scripts/build_embeddings_500k.py

# Run API server
uvicorn src.api.main:app --reload

# Open browser: http://localhost:8000/docs
```

### Run Examples
```bash
# Topic modeling
python examples/topic_modeling_demo.py

# Sentiment analysis
python examples/sentiment_report.py

# Article recommendations
python examples/recommend_articles.py

# Book extraction
python examples/extract_books.py
```

---

## Video Outline Suggestions

### Part 1: Introduction (5 min)
- What is the NYT corpus? (21M articles, 2000-2025)
- Why analyze news at scale?
- Platform overview and capabilities

### Part 2: Topic Discovery (10 min)
- What is topic modeling?
- LDA vs. BERTopic demo
- Word cloud visualization
- Finding trends in COVID coverage

### Part 3: Sentiment Analysis (10 min)
- Why 5 different models?
- Financial vs. general sentiment
- Political bias detection
- Live demo on real articles

### Part 4: Semantic Search (8 min)
- Keyword vs. semantic search
- BERTweet embeddings explained
- FAISS for speed
- Recommendation system demo

### Part 5: Information Extraction (7 min)
- Extracting book metadata
- LLM + Regex hybrid approach
- Gemini verification pipeline
- Success rate and accuracy

### Part 6: Architecture & Scale (5 min)
- Data pipeline walkthrough
- API integration
- Scaling to 21M articles
- Performance optimization

### Part 7: Conclusion (3 min)
- Recap of features
- Use cases for different audiences
- Future roadmap
- Call to action

**Total:** ~48 minutes

---

## Frequently Asked Questions

### Q: Can this work with other news sources?
**A:** Yes! The pipeline is source-agnostic. Just provide CSV with headline/abstract/date columns.

### Q: How long to process 21M articles?
**A:** Embedding generation: ~5-7 days on single GPU. Use multi-GPU or cloud for faster processing.

### Q: Can I run without GPU?
**A:** Yes, but slower. CPU mode works for all features. GPU recommended for embeddings and sentiment.

### Q: What's the cost of LLM APIs?
**A:**
- OpenAI GPT-3.5: ~$0.001 per extraction
- Gemini 2.0 Flash: ~$0.0001 per verification
- 100K books: ~$100-150 total

### Q: How to deploy in production?
**A:**
```bash
# Docker
docker build -t nyt-analytics .
docker run -p 8000:8000 nyt-analytics

# Cloud (AWS, GCP, Azure)
- Deploy API on EC2/Cloud Run/App Service
- Store embeddings in S3/GCS/Blob Storage
- Use managed PostgreSQL for metadata
- Add Redis for caching
```

### Q: Can I fine-tune the models?
**A:** Yes! All models are HuggingFace transformers. Fine-tune on domain-specific data.

### Q: How to handle updates (new articles)?
**A:**
1. Run preprocessing on new CSV
2. Generate embeddings for new articles
3. Append to existing FAISS index
4. Re-train topic models (optional)

---

## Technical Challenges & Solutions

### Challenge 1: Memory Issues
**Problem:** 500K embeddings (1.5GB) don't fit in RAM
**Solution:** Memory-mapped NumPy arrays, FAISS on-disk index

### Challenge 2: Slow Search
**Problem:** NumPy cosine similarity too slow for 500K vectors
**Solution:** FAISS Flat L2 index (10-100x faster)

### Challenge 3: LLM Hallucination
**Problem:** GPT-3.5 sometimes invents book titles
**Solution:** Gemini verification pipeline with confidence scoring

### Challenge 4: Model Selection
**Problem:** One sentiment model doesn't work for all domains
**Solution:** 5 specialized models with automatic section-based selection

### Challenge 5: Topic Coherence
**Problem:** LDA topics sometimes incoherent
**Solution:** BERTopic as alternative, parameter tuning, preprocessing

### Challenge 6: API Timeout
**Problem:** Topic modeling takes 5-10 minutes, API times out
**Solution:** Async job queue with status polling

### Challenge 7: Reproducibility
**Problem:** Different runs produce different topics
**Solution:** Random seed control, saved models, version control

---

## Future Roadmap

### Short-Term (Next 3 Months)
- [ ] Scale to 21M articles
- [ ] Entity extraction (people, organizations, locations)
- [ ] Time-series analysis (topic evolution)
- [ ] Interactive web dashboard (React frontend)
- [ ] Docker deployment guide

### Medium-Term (6 Months)
- [ ] Fine-tuned models for NYT corpus
- [ ] Multi-document summarization
- [ ] Question answering over articles
- [ ] Event detection (breaking news)
- [ ] Author style analysis

### Long-Term (1 Year)
- [ ] Real-time ingestion pipeline
- [ ] Custom GPT for NYT research
- [ ] Multi-lingual support
- [ ] Graph database for entity relationships
- [ ] Automated fact-checking

---

## Resources & References

### Documentation
- All docs in `docs/` folder (100KB)
- API docs: http://localhost:8000/docs
- README: Comprehensive overview

### Papers & Research
- **LDA:** Blei et al., "Latent Dirichlet Allocation" (2003)
- **BERT:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- **BERTopic:** Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF procedure" (2022)
- **FAISS:** Johnson et al., "Billion-scale similarity search with GPUs" (2017)

### Libraries
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers
- **Gensim:** https://radimrehurek.com/gensim/
- **BERTopic:** https://maartengr.github.io/BERTopic/
- **FAISS:** https://github.com/facebookresearch/faiss
- **FastAPI:** https://fastapi.tiangolo.com/

### Dataset
- **NYT Corpus:** Kaggle dataset (21M articles, 2000-2025)
- **License:** Research/educational use
- **Size:** Raw 38GB, preprocessed ~145GB (projected)

---

## Contact & Support

### Issues & Bugs
- GitHub Issues: <repo-url>/issues
- Bug fixes documented in `BUG_FIXES.md`

### Contributing
- Fork the repository
- Create feature branch
- Submit pull request
- Follow code style guidelines

### License
- MIT License (check LICENSE file)
- Dataset license: Kaggle terms

---

## Glossary

**BERTopic:** Neural topic modeling using transformer embeddings and clustering

**BERTweet:** BERT model pre-trained on Twitter data for informal text

**BM25:** Ranking function for keyword search (successor to TF-IDF)

**Cosine Similarity:** Measure of similarity between two vectors (0-1 scale)

**Embeddings:** Dense vector representations of text (768 dimensions)

**FAISS:** Facebook AI Similarity Search - library for fast vector search

**FinBERT:** BERT fine-tuned on financial news for sentiment analysis

**GPU:** Graphics Processing Unit - accelerates deep learning inference

**HDBSCAN:** Density-based clustering algorithm used in BERTopic

**Hybrid Search:** Combination of keyword and semantic search

**LDA:** Latent Dirichlet Allocation - probabilistic topic modeling

**LLM:** Large Language Model (GPT-3.5, Gemini, etc.)

**Parquet:** Columnar storage format for efficient data access

**PoliBERT:** BERT fine-tuned for political bias detection

**Semantic Search:** Search based on meaning, not just keywords

**Sentiment Analysis:** Classification of text emotion (positive/negative/neutral)

**UMAP:** Dimensionality reduction algorithm for visualization

**Word Cloud:** Visual representation of word frequency in topics

---

## Summary

This NYT Article Analytics Platform is a **production-ready, enterprise-grade solution** for large-scale news analysis. It combines:

- **Classical NLP** (LDA, TF-IDF, BM25)
- **Modern Transformers** (BERT variants)
- **LLM Integration** (GPT, Gemini)
- **Fast Search** (FAISS)
- **Scalable Architecture** (FastAPI, async jobs)

**Key Strengths:**
✓ Handles 500K articles (scalable to 21M)
✓ 5 specialized sentiment models
✓ LLM verification for accuracy
✓ Hybrid search (semantic + keyword)
✓ < 1s search latency
✓ 99.9% extraction success rate
✓ Comprehensive documentation
✓ Production-ready API

**Perfect For:**
- Data journalists discovering story trends
- Researchers analyzing media coverage
- Product teams building recommendation engines
- Technical teams needing scalable NLP infrastructure

**Next Steps:**
1. Review this guide for your video
2. Test examples to demonstrate features
3. Prepare visualizations for presentation
4. Plan video structure and demos
5. Record and publish!

---

**Good luck with your video! This platform has incredible capabilities - can't wait to see how you present them.**
