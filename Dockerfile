FROM python:3.11-slim

# Install system dependencies
# libgomp1: OpenMP for faiss-cpu
# gcc/g++: compile extensions (hdbscan, umap-learn)
# curl: health-check probe
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and services
COPY src/ ./src/
COPY services/ ./services/

# Create data directory — mount a Railway Volume here at runtime
RUN mkdir -p data

ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Default: run the FastAPI API server
# Override CMD in Railway service settings for workers:
#   celery -A src.worker.celery_app worker -Q topic    -c 2
#   celery -A src.worker.celery_app worker -Q sentiment -c 2
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
