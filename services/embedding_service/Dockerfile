FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY services/embedding_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the service code
COPY services/ /app/services/

# Pre-download the model at build time so the container starts fast
# (remove this RUN if you prefer lazy loading at runtime)
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('vinai/bertweet-base'); \
    AutoModel.from_pretrained('vinai/bertweet-base')" || true

EXPOSE 8001

ENV PORT=8001

CMD ["sh", "-c", "uvicorn services.embedding_service.main:app --host 0.0.0.0 --port ${PORT}"]
