# Proposal: FBI UCR Crime Prediction Service Deployment

## Overview

Deploy the FBI UCR crime prediction models as a real-time inference API on OpenShift AI, accessible to applications via REST endpoints.

## Model Serving Decision

### Why NOT OpenVINO

OpenVINO is optimized for neural network inference (CNNs, transformers, etc.). Our winning models are:
- **ARIMA** (3 datasets) - statsmodels time series
- **SARIMA** (1 dataset) - statsmodels seasonal time series
- **Prophet** (1 dataset) - Facebook's probabilistic forecasting

These are statistical/probabilistic models that don't benefit from OpenVINO's hardware acceleration.

### Recommended: Custom FastAPI Inference Service

A lightweight FastAPI service that:
1. Loads pre-trained model artifacts at startup
2. Exposes REST endpoints for predictions
3. Handles model versioning and health checks
4. Integrates with OpenShift AI monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenShift AI Cluster                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              fbi-ucr namespace                          ││
│  │  ┌─────────────────┐    ┌─────────────────────────────┐││
│  │  │  PVC: models    │    │  fbi-ucr-inference          │││
│  │  │  ├─ violent-crime│───▶│  FastAPI Service            │││
│  │  │  ├─ property-crime    │  - /predict/{offense}       │││
│  │  │  ├─ homicide    │    │  - /health                  │││
│  │  │  ├─ burglary    │    │  - /models                  │││
│  │  │  └─ motor-vehicle│    │  Port: 8080                 │││
│  │  └─────────────────┘    └──────────┬──────────────────┘││
│  │                                    │                    ││
│  │  ┌─────────────────────────────────▼──────────────────┐││
│  │  │              OpenShift Route                        │││
│  │  │         fbi-ucr-api.apps.cluster.example.com       │││
│  │  └─────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## API Design

### Endpoints

```
POST /predict/{offense}
  Request:  { "steps": 6, "include_history": false }
  Response: {
    "offense": "violent-crime",
    "model": "ARIMA(1,1,1)",
    "predictions": [
      { "date": "2025-10", "predicted": 78500, "lower": 72000, "upper": 85000 },
      ...
    ],
    "metadata": {
      "training_end": "2025-09",
      "mape": 9.0,
      "generated_at": "2025-11-30T12:00:00Z"
    }
  }

GET /models
  Response: {
    "models": [
      { "offense": "violent-crime", "type": "ARIMA", "mape": 9.0, "last_trained": "..." },
      ...
    ]
  }

GET /health
  Response: { "status": "healthy", "models_loaded": 5 }

GET /history/{offense}
  Request params: ?months=24
  Response: { "offense": "violent-crime", "data": [...] }
```

## Project Structure

```
fbi-ucr/
├── Containerfile
├── pyproject.toml
├── src/
│   └── fbi_ucr/
│       ├── __init__.py
│       ├── main.py              # FastAPI app
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py        # API endpoints
│       │   └── schemas.py       # Pydantic models
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── loader.py        # Model loading
│       │   └── predictor.py     # Prediction logic
│       └── models/              # Model class definitions
│           ├── __init__.py
│           ├── arima.py
│           ├── sarima.py
│           └── prophet.py
├── models/                      # Serialized model artifacts
│   ├── violent-crime.joblib
│   ├── property-crime.joblib
│   ├── homicide.joblib
│   ├── burglary.joblib
│   └── motor-vehicle-theft.joblib
├── manifests/
│   ├── base/
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── route.yaml
│   │   ├── pvc.yaml
│   │   └── configmap.yaml
│   └── overlays/
│       ├── dev/
│       │   └── kustomization.yaml
│       └── prod/
│           └── kustomization.yaml
└── tests/
    ├── test_api.py
    └── test_inference.py
```

## Model Serialization Strategy

### ARIMA/SARIMA Models
```python
import joblib
# Save fitted model
joblib.dump(fitted_model, "models/violent-crime.joblib")
# Load for inference
model = joblib.load("models/violent-crime.joblib")
```

### Prophet Models
```python
from prophet.serialize import model_to_json, model_from_json
# Save
with open("models/homicide.json", "w") as f:
    f.write(model_to_json(model))
# Load
with open("models/homicide.json", "r") as f:
    model = model_from_json(f.read())
```

## Deployment Configuration

### Resource Requirements

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

### Scaling

- **Initial**: 1 replica (low traffic expected)
- **HPA**: Scale 1-3 replicas based on CPU utilization
- **Model loading**: Models loaded once at startup, stored in memory

### Health Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
```

## Container Image

### Base Image
```dockerfile
FROM registry.redhat.io/ubi9/python-311:latest
```

### Key Dependencies
- fastapi
- uvicorn
- statsmodels (ARIMA/SARIMA)
- prophet
- joblib
- pandas
- numpy

## Model Refresh Strategy

### Option A: Rebuild and Redeploy (Recommended for MVP)
1. Retrain models locally or in a notebook
2. Export model artifacts
3. Rebuild container with new models baked in
4. Rolling deployment via ArgoCD

### Option B: External Model Storage (Future)
1. Store models in S3/MinIO
2. Service fetches models at startup
3. Endpoint to trigger model reload without restart

## Security Considerations

1. **No PII**: Crime statistics are aggregate, no personal data
2. **Rate limiting**: Implement via OpenShift Route annotations
3. **Authentication**: Optional OAuth proxy for restricted access
4. **Network policy**: Limit ingress to known consumers

## Integration with MCP Server

The inference service will be the backend for the MCP server (future work):

```
LibreChat Agent → MCP Server → FBI UCR Inference API
                     ↓
              Tool: predict_crime_trend
              Tool: compare_states
              Tool: get_historical_trend
```

## Implementation Phases

### Phase 1: Core Service (This Proposal)
- [ ] Create new fbi-ucr project
- [ ] Implement FastAPI inference service
- [ ] Serialize and package trained models
- [ ] Create OpenShift manifests
- [ ] Deploy to dev namespace
- [ ] Test endpoints

### Phase 2: Production Hardening
- [ ] Add monitoring (ServiceMonitor)
- [ ] Implement caching layer
- [ ] Add state-level predictions
- [ ] Set up ArgoCD deployment

### Phase 3: MCP Integration
- [ ] Build MCP server consuming inference API
- [ ] Integrate with LibreChat
- [ ] Add TrustyAI fairness monitoring

## Open Questions

1. **Model storage**: Bake models into container or use external PVC/S3?
2. **State-level models**: Train separate models per state or use national + adjustment?
3. **Retraining frequency**: Monthly? Quarterly? On-demand?
4. **Authentication**: Open API or OAuth-protected?

## Next Steps

1. Get proposal approval
2. Create fbi-ucr project directory
3. Implement inference service
4. Train and serialize production models
5. Build container image
6. Deploy to OpenShift AI
