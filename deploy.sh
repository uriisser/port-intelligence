#!/bin/bash
# Port Intelligence Platform — Firebase + Cloud Run Deployment
# Run this after: firebase login && gcloud auth login

PROJECT_ID="port-analysis"
REGION="us-central1"

echo "=== Setting GCP project ==="
gcloud config set project $PROJECT_ID

echo "=== Enabling required APIs ==="
gcloud services enable run.googleapis.com containerregistry.googleapis.com

echo "=== Authenticating Docker with GCR ==="
gcloud auth configure-docker

echo "=== Building and pushing API image ==="
docker build -t gcr.io/$PROJECT_ID/port-api:latest -f Dockerfile .
docker push gcr.io/$PROJECT_ID/port-api:latest

echo "=== Building and pushing Dashboard image ==="
docker build -t gcr.io/$PROJECT_ID/port-dashboard:latest -f Dockerfile.dashboard .
docker push gcr.io/$PROJECT_ID/port-dashboard:latest

echo "=== Deploying API to Cloud Run ==="
gcloud run deploy port-api \
  --image gcr.io/$PROJECT_ID/port-api:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8000 \
  --memory 1Gi \
  --cpu 1

echo "=== Deploying Dashboard to Cloud Run ==="
gcloud run deploy port-dashboard \
  --image gcr.io/$PROJECT_ID/port-dashboard:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1

echo "=== Deploying Firebase Hosting ==="
firebase deploy --only hosting

echo ""
echo "=== DONE ==="
echo "Dashboard: https://$PROJECT_ID.web.app"
echo "API:       https://$PROJECT_ID.web.app/api"
echo "API Docs:  https://$PROJECT_ID.web.app/api/docs"
