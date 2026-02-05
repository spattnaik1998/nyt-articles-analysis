FROM python:3.10-slim

# Install system dependencies (libgomp for OpenMP in faiss-cpu)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY Makefile .

# Create data directory for volume mounting
RUN mkdir -p data

# Expose API port
EXPOSE 8000

# Run FastAPI with uvicorn
# data/ is expected to be mounted as a volume or fetched from GCS at startup
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
