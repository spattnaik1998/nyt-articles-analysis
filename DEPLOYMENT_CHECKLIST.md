# NYT Article Analytics Platform - Deployment Checklist

This checklist will guide you through deploying to Google Cloud Platform with your local machine.

**Status**: ✓ Codebase prepared for deployment

---

## Prerequisites Verification

### 1. Install Required Tools

- [ ] **Google Cloud SDK** - https://cloud.google.com/sdk/docs/install
  ```bash
  # After installation, verify
  gcloud --version
  ```

- [ ] **Docker** - https://docs.docker.com/get-docker/
  ```bash
  # Verify
  docker --version
  ```

### 2. Verify GCP Project

- [ ] GCP Project created: `Sarthak-ai-aura`
- [ ] Billing enabled on the project
- [ ] You have Owner or Editor permissions

---

## Step 1: Configure Local Environment

Run these commands on your local machine (where gcloud is installed):

```bash
# Clone or navigate to the repository
cd /path/to/nyt-full-project

# Set default project
gcloud config set project Sarthak-ai-aura

# Verify project is set
gcloud config get-value project
# Should output: Sarthak-ai-aura

# Authenticate with Google Cloud
gcloud auth login

# Set default region
gcloud config set compute/region us-central1
```

---

## Step 2: Initialize GCP Project

Run the setup script to enable APIs, create buckets, and configure permissions:

```bash
bash scripts/setup_gcp_project.sh --project Sarthak-ai-aura --region us-central1
```

**What this does:**
- ✓ Enables required Google Cloud APIs
- ✓ Creates Cloud Storage bucket for data
- ✓ Creates Secret Manager placeholders
- ✓ Configures IAM permissions

**Expected output:**
```
==========================================
GCP Project Setup Complete!
==========================================

Next steps:
1. Update your API keys in Secret Manager
2. Upload preprocessed data to Cloud Storage
3. Deploy the application
```

---

## Step 3: Update Secrets in GCP Secret Manager

**IMPORTANT**: Use your NEW rotated API keys (not the old exposed ones)

```bash
bash scripts/gcp_deploy_secure.sh \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --openai-key "sk-proj-YOUR-NEW-OPENAI-KEY" \
  --gemini-key "YOUR-NEW-GEMINI-KEY" \
  --tavily-key "YOUR-NEW-TAVILY-KEY" \
  --setup-secrets
```

**Verify secrets were created:**
```bash
gcloud secrets list --project Sarthak-ai-aura
```

You should see:
- nyt-openai-api-key
- nyt-gemini-api-key
- nyt-tavily-api-key

---

## Step 4: Prepare Data

### Option A: Use 500K Dataset (Recommended for Testing)

The repository includes preprocessed 500K embeddings. Just upload to Google Cloud Storage:

```bash
# Upload data to GCS
gsutil -m cp -r data/* gs://Sarthak-ai-aura-nyt-data/data/

# Verify upload
gsutil ls -r gs://Sarthak-ai-aura-nyt-data/data/

# Should show:
# gs://Sarthak-ai-aura-nyt-data/data/embeddings_500k.npy
# gs://Sarthak-ai-aura-nyt-data/data/preprocessed_500K.parquet
# gs://Sarthak-ai-aura-nyt-data/data/embeddings_500k_mapping.csv
```

### Option B: Generate 21M Dataset (Advanced)

For full Kaggle corpus (requires GPU and more time):

```bash
# Create GPU instance and run preprocessing
bash scripts/gcp_pipeline.sh \
  --create \
  --gpu t4 \
  --project Sarthak-ai-aura \
  --region us-central1

# Then follow prompts in that script
```

---

## Step 5: Deploy Application to Cloud Run

Run the secure deployment script:

```bash
bash scripts/gcp_deploy_secure.sh \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --deploy
```

**What this does:**
1. Builds Docker image
2. Pushes to Google Container Registry
3. Deploys to Cloud Run
4. Configures Secret Manager access
5. Sets up data loading from Cloud Storage

**Expected output:**
```
==========================================
Deployment Complete
==========================================
  Service URL: https://nyt-api-xxx.run.app
  API Docs:    https://nyt-api-xxx.run.app/docs
  Frontend:    https://nyt-api-xxx.run.app/app
  Health:      https://nyt-api-xxx.run.app/health
```

---

## Step 6: Verify Deployment

### Test Health Endpoint

```bash
SERVICE_URL="https://nyt-api-xxx.run.app"  # Replace with actual URL

# Test health check
curl ${SERVICE_URL}/health

# Expected response:
# {"status": "healthy", "datasets": {"embeddings_loaded": true, ...}}
```

### Test Search

```bash
curl -X POST ${SERVICE_URL}/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "climate change",
    "limit": 5,
    "year": null
  }'
```

### Access Frontend

Open in browser: `https://nyt-api-xxx.run.app/app`

---

## Step 7: Monitor Deployment

### View Logs

```bash
# Real-time logs
gcloud run logs read nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --follow

# Last 50 lines
gcloud run logs read nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --limit 50
```

### Check Service Status

```bash
gcloud run services describe nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1
```

---

## Troubleshooting

### Issue: "gcloud: command not found"

**Solution**: Install Google Cloud SDK
- https://cloud.google.com/sdk/docs/install
- After installation, restart terminal and verify: `gcloud --version`

### Issue: "Docker daemon is not running"

**Solution**: Start Docker
- Windows: Open Docker Desktop
- Linux: `sudo systemctl start docker`
- Mac: Open Docker.app

### Issue: "Permission denied" on script

**Solution**: Make script executable
```bash
chmod +x scripts/gcp_deploy_secure.sh
chmod +x scripts/setup_gcp_project.sh
chmod +x scripts/gcp_pipeline.sh
```

### Issue: Secrets not found during deployment

**Solution**: Verify secrets exist
```bash
gcloud secrets list --project Sarthak-ai-aura

# If missing, run Step 3 again
bash scripts/gcp_deploy_secure.sh \
  --project Sarthak-ai-aura \
  --setup-secrets \
  --openai-key "YOUR-KEY" \
  --gemini-key "YOUR-KEY" \
  --tavily-key "YOUR-KEY"
```

### Issue: High latency on first request

**Expected behavior**: First request loads embeddings/FAISS (30-60 seconds)

**Optimization**: Increase min instances to pre-warm
```bash
gcloud run deploy nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --min-instances 2
```

---

## Security Checklist

- [ ] Old API keys have been rotated and invalidated
- [ ] New secrets created in GCP Secret Manager
- [ ] `.env` file not committed to git (verify in `.gitignore`)
- [ ] Cloud Run service has limited IAM permissions
- [ ] Secrets not visible in deployment logs

---

## Files Modified for Deployment

These files have been updated to prepare for secure deployment:

1. **Dockerfile** - Removed `.env` copy (uses Secret Manager instead)
2. **scripts/gcp_deploy_secure.sh** - NEW: Secure deployment with Secret Manager
3. **scripts/setup_gcp_project.sh** - NEW: GCP project initialization
4. **.env.example** - Updated with security notes
5. **DEPLOYMENT.md** - NEW: Comprehensive deployment guide
6. **DEPLOYMENT_CHECKLIST.md** - This file

---

## Next Steps

1. **Install gcloud SDK** on your local machine
2. **Run Step 1-3** above to configure GCP
3. **Run Step 4** to upload data
4. **Run Step 5** to deploy application
5. **Run Step 6** to verify deployment works
6. **Monitor** with Step 7 commands

---

## Support

- **API Documentation**: `/docs` endpoint after deployment
- **Deployment Guide**: See `DEPLOYMENT.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **GCP Documentation**: https://cloud.google.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com/

---

**Happy Deploying!** 🚀
