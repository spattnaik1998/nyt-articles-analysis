# 🚀 NYT Platform Deployment Instructions - Tomorrow's Guide

**Status**: ✅ All preparation complete and committed to git
**Project**: `Sarthak-ai-aura`
**Region**: `us-central1`
**Date**: 2026-02-05

---

## ⚡ One-Command Deployment

Everything is prepared. Run this ONE command tomorrow (replace with your actual API keys):

```bash
cd /path/to/nyt-full-project

bash scripts/deploy_complete.sh \
  --openai-key "sk-proj-YOUR-OPENAI-KEY-HERE" \
  --gemini-key "YOUR-GEMINI-KEY-HERE" \
  --tavily-key "YOUR-TAVILY-KEY-HERE"
```

**That's it!** The script handles everything automatically:
- ✓ Configures GCP
- ✓ Initializes resources (APIs, storage, IAM)
- ✓ Stores secrets in Secret Manager (encrypted)
- ✓ Uploads data to Cloud Storage
- ✓ Builds Docker image
- ✓ Deploys to Cloud Run
- ✓ Displays your live API URL

**Total time**: ~15-25 minutes

---

## 📋 Pre-Deployment Checklist

Before running the command, ensure you have:

- [ ] **Google Cloud SDK installed** - https://cloud.google.com/sdk/docs/install
  ```bash
  gcloud --version
  ```

- [ ] **Docker installed** - https://docs.docker.com/get-docker/
  ```bash
  docker --version
  ```

- [ ] **New API Keys ready** (NOT the old exposed ones!)
  - [ ] **OpenAI**: https://platform.openai.com/api-keys (format: `sk-proj-...`)
  - [ ] **Gemini**: https://makersuite.google.com/app/apikey
  - [ ] **Tavily**: https://tavily.com/ (sign up if needed)

- [ ] **GCP Project Access**
  - Project: `Sarthak-ai-aura`
  - Billing enabled
  - You have Owner/Editor permissions

---

## 🔑 API Key Notes

**IMPORTANT**: Use NEW keys from the providers, NOT the old exposed ones!

1. **OpenAI API Key**
   - Go to: https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy it (starts with `sk-proj-`)

2. **Gemini API Key**
   - Go to: https://makersuite.google.com/app/apikey
   - Click "Create API Key"
   - Copy it

3. **Tavily API Key**
   - Go to: https://tavily.com/
   - Sign up or login
   - Get your API key
   - Copy it

---

## 🎯 Step-by-Step Deployment Process

### What the deploy_complete.sh script does:

**Step 1: Configure GCP** (1 minute)
```
Sets default project and region
```

**Step 2: Initialize GCP Resources** (3-5 minutes)
```
✓ Enables Cloud Run, Secret Manager, Container Registry APIs
✓ Creates Cloud Storage bucket (Sarthak-ai-aura-nyt-data)
✓ Configures IAM permissions
✓ Creates Secret Manager placeholders
```

**Step 3: Store Secrets** (1 minute)
```
✓ Stores OpenAI API key in Secret Manager (encrypted)
✓ Stores Gemini API key in Secret Manager (encrypted)
✓ Stores Tavily API key in Secret Manager (encrypted)
✓ Grants Cloud Run service account access
```

**Step 4: Upload Data** (2-5 minutes)
```
✓ Uploads 500K preprocessed embeddings (~1.5 GB)
✓ Uploads metadata and mappings (~150 MB)
✓ Data stored in Cloud Storage bucket
```

**Step 5: Build & Deploy** (5-10 minutes)
```
✓ Builds Docker image locally
✓ Pushes to Google Container Registry
✓ Deploys to Cloud Run
✓ Configures auto-scaling (1-10 instances)
✓ Sets up Secret Manager access
```

**Step 6: Display Results**
```
Shows your live API URL:
  Frontend:     https://nyt-api-xxx.run.app/app
  API Docs:     https://nyt-api-xxx.run.app/docs
  Health Check: https://nyt-api-xxx.run.app/health
```

---

## 🌐 After Deployment - Access Your App

Once the script finishes, you'll have a live API. Access it:

**Frontend Interface** (Search, Sentiment, Topics)
```
https://nyt-api-YOUR-ID.run.app/app
```

**API Documentation** (Interactive Swagger UI)
```
https://nyt-api-YOUR-ID.run.app/docs
```

**Health Check**
```
curl https://nyt-api-YOUR-ID.run.app/health
```

**Search Articles** (Programmatic)
```bash
curl -X POST https://nyt-api-YOUR-ID.run.app/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "climate change",
    "limit": 10,
    "year": null
  }'
```

---

## ✨ Expected Output

When deployment succeeds, you'll see:

```
==========================================
🎉 DEPLOYMENT SUCCESSFUL!
==========================================

Your API is live at:
  https://nyt-api-abc123xyz.run.app

Access your application:
  Frontend:     https://nyt-api-abc123xyz.run.app/app
  API Docs:     https://nyt-api-abc123xyz.run.app/docs
  Health Check: https://nyt-api-abc123xyz.run.app/health

Next steps:
1. Open https://nyt-api-abc123xyz.run.app/app in your browser
2. Try searching for articles
3. Test API endpoints at https://nyt-api-abc123xyz.run.app/docs
4. Check logs for any issues
```

---

## 🆘 Troubleshooting

### "gcloud: command not found"
**Solution**: Install Google Cloud SDK
```bash
# https://cloud.google.com/sdk/docs/install
# Then verify:
gcloud --version
```

### "Docker daemon is not running"
**Solution**:
- Windows/Mac: Open Docker Desktop
- Linux: `sudo systemctl start docker`

### "Permission denied" on script
**Solution**: Make scripts executable
```bash
chmod +x scripts/deploy_complete.sh
chmod +x scripts/gcp_deploy_secure.sh
chmod +x scripts/setup_gcp_project.sh
```

### Deployment fails mid-way
**Solution**: Just run the script again. It will skip completed steps and continue.

### "Secrets not found" error
**Solution**: Verify secrets were created
```bash
gcloud secrets list --project Sarthak-ai-aura
```

Should show: `nyt-openai-api-key`, `nyt-gemini-api-key`, `nyt-tavily-api-key`

### High latency on first request
**Expected behavior**: First request takes 30-60 seconds to load embeddings
**This is normal** - subsequent requests are fast (~500ms)

---

## 📊 Cost Estimate

**Monthly costs for 500K dataset** (approximate):
- Compute: $30-50
- Storage: $5-10
- Data transfer: $0-5
- **Total**: ~$40-60/month

---

## 📞 If Something Goes Wrong

**1. Check real-time logs:**
```bash
gcloud run logs read nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1 \
  --follow
```

**2. Check service status:**
```bash
gcloud run services describe nyt-api \
  --project Sarthak-ai-aura \
  --region us-central1
```

**3. Read detailed documentation:**
- `DEPLOYMENT.md` - Comprehensive reference
- `DEPLOYMENT_CHECKLIST.md` - Step-by-step guide
- `DEPLOYMENT_CONTEXT.md` - Full context (in memory folder)

---

## 🔒 Security Summary

✅ **What's been fixed:**
- API keys removed from Dockerfile
- `.env` file not copied to Docker image
- Secrets stored in GCP Secret Manager (encrypted)
- No credentials in git repository
- Access controlled via IAM

⚠️ **What you need to do:**
- Use ONLY new API keys (not the old exposed ones)
- Keep new keys secure
- Monitor logs for unauthorized access

---

## ✅ Deployment Success Criteria

Your deployment is successful when:

✅ Script completes without errors
✅ Health endpoint returns 200 status
✅ Frontend loads and responds
✅ Search endpoint returns results
✅ API docs page loads at `/docs`
✅ No errors in logs after 5 minutes

---

## 📚 Documentation Files

If you need more details tomorrow:

1. **DEPLOY_TOMORROW.md** (this file) - Quick start guide
2. **DEPLOYMENT.md** - Full reference with all details
3. **DEPLOYMENT_CHECKLIST.md** - Step-by-step checklist
4. **DEPLOYMENT_CONTEXT.md** - Full context (memory folder)
5. **IMPLEMENTATION_SUMMARY.md** - Technical implementation
6. **README.md** - Project overview

---

## 🎉 You're All Set!

Everything is prepared and committed to git. Tomorrow just:

1. **Get your 3 API keys**
2. **Run the one-command deployment**
3. **Enjoy your live analytics platform!**

---

## 💡 Quick Reference Command

**Save this for tomorrow:**

```bash
bash scripts/deploy_complete.sh \
  --openai-key "sk-proj-YOUR-KEY" \
  --gemini-key "YOUR-GEMINI-KEY" \
  --tavily-key "YOUR-TAVILY-KEY"
```

**That's all you need!** 🚀
