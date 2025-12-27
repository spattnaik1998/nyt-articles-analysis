# NYT Data Journalism Platform

A comprehensive data journalism analytics platform for analyzing the NYT 21M article corpus (2000-2025) with advanced NLP pipelines.

## Project Goals

Build a Data-Journalism Analytics Platform for newsrooms and researchers that provides:

- **Topic Modeling**: LDA and BERTopic pipelines for discovering themes and trends
- **Multi-Model Sentiment Analysis**: FinBERT, FinBERT-Tone, DistilRoBERTa, and PoliBERT for domain-specific sentiment detection
- **Embedding-Based Recommendations**: BERTweet embeddings with cosine similarity search for article discovery
- **Entity & Structured Extraction**: LLM-powered extraction with Pydantic validation
- **Classification & Clustering**: Advanced clustering (KMeans, DBSCAN, hierarchical) and temporal analytics
- **Visualization**: Word clouds, PCA plots, intertopic distance maps, and interactive dashboards

## Target Users

- **Newsroom Data Journalists**: Quick topic discovery, entity timelines, similarity search
- **Investigative Researchers**: Reproducible pipelines, model comparisons, exportable datasets
- **Product/Analytics Managers**: Content performance dashboards, reputation monitoring
- **ML Ops Engineers**: Model deployment, monitoring, and retraining workflows

## Project Structure

```
nyt-data-journalism-platform/
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for exploration
├── src/
│   ├── ingest/           # Data ingestion modules
│   ├── preprocess/       # Text preprocessing utilities
│   ├── models/           # ML models (LDA, BERTopic, sentiment, embeddings)
│   ├── api/              # FastAPI REST API
│   └── utils/            # Shared utilities and helpers
├── tests/                # Unit and integration tests
├── requirements.txt      # Python dependencies
├── Makefile             # Build automation
└── README.md            # This file
```

## Quick Start

### 1. Setup Virtual Environment

```bash
# Create virtual environment
make venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
make install
```

### 2. Run the API Server

```bash
# Start FastAPI development server
make run-api
```

The API will be available at `http://localhost:8000`

### 3. Run Tests

```bash
# Run all tests with coverage
make test
```

## Manual Setup (Alternative)

If you prefer not to use the Makefile:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v --cov=src
```

## Development Workflow

1. **Data Ingestion**: Load and preprocess NYT article corpus
2. **Model Training**: Train topic models, sentiment analyzers, and embeddings
3. **API Development**: Build REST endpoints for search, analysis, and recommendations
4. **Testing**: Write unit and integration tests
5. **Deployment**: Deploy to cloud infrastructure

## Key Technologies

- **NLP Models**: Gensim (LDA), BERTopic, BERT variants (FinBERT, PoliBERT, BERTweet)
- **ML Libraries**: scikit-learn, transformers, sentence-transformers
- **Vector Search**: Faiss for efficient similarity search
- **API Framework**: FastAPI with async support
- **Visualization**: matplotlib, seaborn, wordcloud

## Success Metrics

- Query latency (embedding search): ≤ 1s for top-10 results
- Topic pipeline reproducibility: 95%+ identical results with same seed
- Extraction accuracy: ≥ 99% on sample data
- User satisfaction: 80%+ editors find topic discovery "useful"

## Contributing

1. Create a feature branch from `main`
2. Implement changes with tests
3. Run `make test` to ensure all tests pass
4. Submit pull request for review

## License

MIT License

## Repository

GitHub: https://github.com/spattnaik1998/nyt-articles-analysis.git

## Contact

For questions or support, please open an issue on GitHub.
