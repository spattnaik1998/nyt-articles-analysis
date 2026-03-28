# Railway Deployment Guide

## Service Topology

```
Railway Project
├── redis             (Railway Redis plugin — managed)
├── nyt-embedding     (services/embedding_service/Dockerfile)
├── nyt-api           (root Dockerfile, default CMD)
├── nyt-worker-topic  (root Dockerfile, custom start command)
└── nyt-worker-sentiment (root Dockerfile, custom start command)
```

---

## Step 1 — Create a Railway project

1. Go to railway.app → New Project → Empty Project.
2. Name it `nyt-analytics`.

---

## Step 2 — Add Redis

1. In the project, click **+ New** → **Database** → **Redis**.
2. Railway provisions a managed Redis instance and exposes `REDIS_URL` as a shared variable.

---

## Step 3 — Deploy the Embedding Service

1. Click **+ New** → **GitHub Repo** → select your repo.
2. Rename the service to `nyt-embedding`.
3. In **Settings → Build**:
   - **Dockerfile path**: `services/embedding_service/Dockerfile`
4. In **Settings → Deploy**:
   - **Start command**: *(leave blank — Dockerfile CMD handles it)*
   - **Health check path**: `/health`
5. Add **Variables**:
   | Variable | Value |
   |---|---|
   | `REDIS_URL` | `${{Redis.REDIS_URL}}` |
   | `PORT` | `8001` |

---

## Step 4 — Deploy the Main API

1. Click **+ New** → **GitHub Repo** → same repo.
2. Rename to `nyt-api`.
3. In **Settings → Build**:
   - **Dockerfile path**: `Dockerfile`
4. In **Settings → Deploy**:
   - **Start command**: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
   - **Health check path**: `/health`
5. Add **Variables**:
   | Variable | Value |
   |---|---|
   | `REDIS_URL` | `${{Redis.REDIS_URL}}` |
   | `EMBEDDING_SERVICE_URL` | `http://${{nyt-embedding.RAILWAY_PRIVATE_DOMAIN}}:8001` |
   | `OPENAI_API_KEY` | *(your key)* |
   | `LOG_LEVEL` | `INFO` |
   | `LOAD_LOCAL_EMBEDDING_FALLBACK` | `false` |
6. Add a **Volume**:
   - Mount path: `/app/data`
   - Upload your parquet files via Railway's volume CLI or dashboard after deploy.

---

## Step 5 — Deploy the Topic Worker

1. Click **+ New** → **GitHub Repo** → same repo.
2. Rename to `nyt-worker-topic`.
3. In **Settings → Build**:
   - **Dockerfile path**: `Dockerfile`
4. In **Settings → Deploy**:
   - **Start command**:
     ```
     celery -A src.worker.celery_app worker --queues topic --concurrency 2 --loglevel info --hostname topic@%h
     ```
   - *No health check needed for workers.*
5. Add **Variables** (same as API):
   | Variable | Value |
   |---|---|
   | `REDIS_URL` | `${{Redis.REDIS_URL}}` |
   | `OPENAI_API_KEY` | *(your key)* |
6. Add the **same Volume** mounted at `/app/data` (Railway lets multiple services share a volume).

---

## Step 6 — Deploy the Sentiment Worker

1. Same as Step 5, but rename to `nyt-worker-sentiment`.
2. Start command:
   ```
   celery -A src.worker.celery_app worker --queues sentiment --concurrency 2 --loglevel info --hostname sentiment@%h
   ```
3. Same Variables and Volume as the topic worker.

---

## Step 7 — Upload data files

Railway Volumes can be populated via the CLI:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Upload parquet files to the volume
railway volume cp data/preprocessed_21m.parquet /app/data/preprocessed_21m.parquet --service nyt-api
```

Or use the Railway dashboard: **Service → Volume → Browse Files → Upload**.

---

## Environment Variables Quick Reference

### All services that load data or call Redis

| Variable | Description | Example |
|---|---|---|
| `REDIS_URL` | Redis connection string | `redis://default:pass@host:6379/0` |

### API + Workers

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI key for LLM extraction | — |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

### API only

| Variable | Description | Default |
|---|---|---|
| `EMBEDDING_SERVICE_URL` | Internal URL of embedding service | *(falls back to local model)* |
| `EMBEDDING_SERVICE_TIMEOUT` | HTTP timeout (seconds) | `5.0` |
| `LOAD_LOCAL_EMBEDDING_FALLBACK` | Keep local BERTweet as fallback | `false` |

### Embedding service only

| Variable | Description | Default |
|---|---|---|
| `PORT` | Port the service listens on | `8001` |

---

## Local Testing with Docker Compose

```bash
# Copy and fill in your keys
cp .env.example .env

# Build and start all services
docker compose up --build

# API is available at http://localhost:8000
# Embedding service at http://localhost:8001
```

---

## Verifying Deployment

```bash
# Health check
curl https://<your-railway-domain>/health

# Cache stats
curl https://<your-railway-domain>/cache/stats

# Embedding service health (proxied through API)
curl https://<your-railway-domain>/embedding/health

# Submit a topic modeling job
curl -X POST https://<your-railway-domain>/topic/run \
  -H "Content-Type: application/json" \
  -d '{"year": 2020, "section": "Politics", "num_topics": 5}'

# Poll job status
curl https://<your-railway-domain>/jobs/<job_id>
```
