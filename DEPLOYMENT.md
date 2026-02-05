# NYT Article Analytics Platform - Deployment Guide

This guide provides step-by-step instructions for deploying the NYT Article Analytics Platform to Google Cloud Platform.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Security Setup](#security-setup)
3. [GCP Project Setup](#gcp-project-setup)
4. [Data Preparation](#data-preparation)
5. [Deployment](#deployment)
6. [Verification](#verification)
7. [Monitoring & Logs](#monitoring--logs)

---

## Prerequisites

### Required Tools

- **gcloud CLI**: [Install Cloud SDK](https://cloud.google.com/sdk/docs/install)
- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **gsutil**: Included with Cloud SDK

### GCP Account

- Active GCP account with billing enabled
- Project created: `Sarthak-ai-aura`

### Verify Installation

```bash
gcloud --version
docker --version
gsutil --version
```

---

## Security Setup

### ⚠️ CRITICAL: API Key Rotation

**Your API keys were exposed in the repository and MUST be rotated immediately:**

1. **OpenAI**: Generate new key at https://platform.openai.com/api-keys
2. **Gemini/Google**: Generate new key at https://makersuite.google.com/app/apikey
3. **Tavily**: Generate new key at https://tavily.com/
4. **Serper** (if used): Generate new key at https://serper.dev/

### Secure Credential Storage

The application uses **GCP Secret Manager** for production credentials:

- Secrets are never stored in code or `.env` files
- Docker container doesn't include `.env` file
- Cloud Run retrieves secrets from Secret Manager at runtime
- Each secret is encrypted and access-controlled via IAM

**Local Development**: Use `.env` file (it's in `.gitignore`)

```bash
# Copy the template
cp .env.example .env

# Edit with your local API keys (NEVER commit this)
vim .env
```

---

## GCP Project Setup

### Step 1: Initialize GCP Project

```bash
# Set your project as default
gcloud config set project Sarthak-ai-aura

# Verify
gcloud config get-value project
```

### Step 2: Run Setup Script

This script enables required APIs, creates storage buckets, and configures IAM:

```bash
bash scripts/setup_gcp_project.sh --project Sarthak-ai-aura --region us-central1
```

**What this does:**
- ✓ Enables Cloud Run, Cloud Build, Container Registry, Secret Manager, Cloud Storage APIs
- ✓ Creates Cloud Storage bucket for data (`Sarthak-ai-aura-nyt-data`)
- ✓ Creates Secret Manager placeholders
- ✓ Configures IAM permissions for Cloud Run service account

### Step 3: Create/Update Secrets in Secret Manager

Replace with your actual API keys:

```bash
bash scripts/gcp_deploy_secure.sh \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --openai-key "sk-proj-YOUR-ACTUAL-KEY-HERE" \
  --gemini-key "YOUR-ACTUAL-GEMINI-KEY-HERE" \
  --tavily-key "YOUR-ACTUAL-TAVILY-KEY-HERE" \
  --setup-secrets
```

**Verify secrets were created:**

```bash
gcloud secrets list --project Sarthak-ai-aura
```

---

## Data Preparation

### For 500K Scale (Recommended for Testing)

The repository includes preprocessed 500K embeddings:

```bash
# Verify local data exists
ls -lh data/

# Expected files:
# - data/preprocessed_500K.parquet (~150 MB)
# - data/embeddings_500k.npy (~1.5 GB)
# - data/embeddings_500k_mapping.csv
```

### Upload Data to Cloud Storage

```bash
# Create data directory in GCS bucket
gsutil -m cp -r data/* gs://Sarthak-ai-aura-nyt-data/data/

# Verify upload
gsutil ls -r gs://Sarthak-ai-aura-nyt-data/data/
```

### For 21M Scale (Full Dataset)

Generate embeddings using GPU instances:

```bash
# 1. Create GPU instance and generate embeddings
bash scripts/gcp_pipeline.sh \
  --create \
  --gpu t4 \
  --project Sarthak-ai-aura \
  --region us-central1

# 2. Run pipeline (generates preprocessed_21m.parquet and FAISS index)
bash scripts/gcp_pipeline.sh \
  --run \
  --instance nyt-pipeline-01 \
  --project Sarthak-ai-aura

# 3. Download results to local machine
bash scripts/gcp_pipeline.sh \
  --download \
  --instance nyt-pipeline-01 \
  --project Sarthak-ai-aura

# 4. Upload to GCS
gsutil -m cp -r data/preprocessed_21m.parquet gs://Sarthak-ai-aura-nyt-data/data/
gsutil -m cp -r data/faiss_index_21m.bin gs://Sarthak-ai-aura-nyt-data/data/
```

---

## Deployment

### Quick Deploy (Recommended)

Deploy using the secure deployment script with all credentials:

```bash
bash scripts/gcp_deploy_secure.sh \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --deploy
```

This script:
1. Builds the Docker image
2. Pushes to Google Container Registry
3. Deploys to Cloud Run
4. Configures Secret Manager access
5. Sets up data fetching from Cloud Storage

### What Gets Deployed

- **Service**: `nyt-api` on Cloud Run
- **Memory**: 8 GB per instance
- **CPU**: 2 vCPU
- **Scaling**: 1-10 instances (auto-scales based on traffic)
- **Timeout**: 3600 seconds (1 hour, for heavy operations)

### Deployment Output

After deployment completes, you'll see:

```
========================================
Deployment Complete
==========================================
  Service URL: https://nyt-api-xxx.run.app
  API Docs:    https://nyt-api-xxx.run.app/docs
  Frontend:    https://nyt-api-xxx.run.app/app
  Health:      https://nyt-api-xxx.run.app/health
```

---

## Verification

### 1. Check Service Status

```bash
gcloud run services describe nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1
```

### 2. Test Health Endpoint

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --format='value(status.url)')

# Test health
curl ${SERVICE_URL}/health

# Should return:
# {"status": "healthy", "datasets": {...}}
```

### 3. Test Search Endpoint

```bash
curl -X POST ${SERVICE_URL}/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "climate change",
    "limit": 10,
    "year": null
  }'
```

### 4. Access Frontend

Open in browser: `https://nyt-api-xxx.run.app/app`

### 5. View API Documentation

Open in browser: `https://nyt-api-xxx.run.app/docs`

---

## Monitoring & Logs

### View Real-Time Logs

```bash
gcloud run logs read nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --follow
```

### View in Cloud Console

1. Go to [Cloud Run console](https://console.cloud.google.com/run)
2. Select project `Sarthak-ai-aura`
3. Click on `nyt-api` service
4. View **Logs** tab

### Key Metrics to Monitor

- **Request Count**: Number of API requests per minute
- **Error Rate**: Percentage of failed requests
- **Latency**: Response time in milliseconds
- **Memory Usage**: RAM usage per instance
- **CPU Usage**: CPU utilization percentage

### Set Up Alerts

Create an alert for high error rates:

```bash
# In Cloud Console: Monitoring > Alert Policies > Create Policy
# Condition: Cloud Run > nyt-api > Error Rate > 5%
# Notification: Email or Slack
```

---

## Troubleshooting

### Issue: Deployment Fails - Docker Build Error

```bash
# Solution: Check for missing dependencies
cat requirements.txt

# Rebuild locally first
docker build -t test-image .
```

### Issue: Secrets Not Found

```bash
# Verify secrets exist
gcloud secrets list --project Sarthak-ai-aura

# Verify service account has access
gcloud secrets get-iam-policy nyt-openai-api-key --project Sarthak-ai-aura
```

### Issue: Data Not Loading

```bash
# Check GCS bucket
gsutil ls gs://Sarthak-ai-aura-nyt-data/data/

# Check Cloud Run logs
gcloud run logs read nyt-api --project Sarthak-ai-aura --region us-central1
```

### Issue: High Latency on First Request

**Expected**: First request loads embeddings/FAISS index (can take 30-60 seconds)
**Solution**: Increase `--min-instances` to pre-warm containers

```bash
gcloud run deploy nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --min-instances 2
```

---

## Cost Optimization

### Reduce Costs

1. **Scale down min instances** (if low traffic):
   ```bash
   gcloud run deploy nyt-api --min-instances 0 --project Sarthak-ai-aura
   ```

2. **Use 500K dataset** instead of 21M for testing:
   - Reduces memory to 4GB
   - Reduces startup time from ~60s to ~10s

3. **Enable Cloud CDN** for static files:
   ```bash
   gcloud compute backend-services update nyt-api-backend --cache-mode CACHE_ALL
   ```

### Cost Estimates (Monthly, us-central1)

| Scale | Memory | Requests/mo | Est. Cost |
|-------|--------|-------------|-----------|
| 500K (testing) | 4GB | 10,000 | $5-10 |
| 500K (production) | 8GB | 100,000 | $50-100 |
| 21M (production) | 16GB | 100,000 | $200-300 |

---

## Next Steps

1. ✓ Run setup script
2. ✓ Update secrets in Secret Manager
3. ✓ Upload data to Cloud Storage
4. ✓ Deploy application
5. ✓ Verify deployment works
6. ✓ Set up monitoring and alerts
7. Monitor costs and performance

---

## Support & Documentation

- **API Documentation**: `/docs` endpoint (Swagger UI)
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Architecture**: See `docs/` directory
- **GCP Documentation**: https://cloud.google.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com/

