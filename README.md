# NYT Article Analytics Platform

A comprehensive analytics platform for analyzing the NYT 21M article corpus (2000-2025) with advanced text analysis capabilities.

## Project Goals

Build an analytics platform for newsrooms and researchers that provides:

- **Topic Discovery**: Automatically identify themes and trends across large article collections
- **Sentiment Analysis**: Detect emotional tone and perspective across different content types
- **Content Recommendations**: Intelligent article discovery based on meaning and relevance
- **Information Extraction**: Automatically extract key entities and structured data from articles
- **Content Organization**: Advanced grouping and categorization with temporal analysis
- **Visualization**: Word clouds, distribution charts, and interactive dashboards

## Target Users

- **Newsroom Data Journalists**: Quick topic discovery, entity timelines, related content finding
- **Investigative Researchers**: Reproducible analysis workflows, exportable datasets
- **Product/Analytics Managers**: Content performance dashboards, reputation monitoring
- **Technical Teams**: System deployment, monitoring, and maintenance workflows

## Project Structure

```
nyt-article-analytics-platform/
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for exploration
├── src/
│   ├── ingest/           # Data ingestion modules
│   ├── preprocess/       # Text preprocessing utilities
│   ├── models/           # Analysis models (topic discovery, sentiment, recommendations)
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
2. **Analysis Setup**: Configure topic discovery, sentiment analysis, and content recommendation systems
3. **API Development**: Build REST endpoints for search, analysis, and recommendations
4. **Testing**: Write unit and integration tests
5. **Deployment**: Deploy to cloud infrastructure

## Key Technologies

- **Text Analysis**: Advanced natural language processing and machine learning
- **Data Processing**: Python-based analytics and data science libraries
- **Search**: Efficient content discovery and retrieval systems
- **API Framework**: FastAPI with async support
- **Visualization**: matplotlib, seaborn, wordcloud

## Success Metrics

- Search response time: ≤ 1s for top-10 results
- Analysis reproducibility: 95%+ consistent results
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
