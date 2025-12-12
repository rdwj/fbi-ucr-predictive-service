# FBI UCR Crime Prediction Service

A FastAPI-based prediction service for U.S. crime statistics using FBI Uniform Crime Reporting (UCR) data. The service provides time series forecasts for various crime types at both national and state levels.

> **New to Machine Learning?** [Click here for a primer on time series forecasting](docs/ML_PRIMER.md) covering Prophet, ARIMA, SARIMA, and how we selected our models.

## Overview

This service:
- Loads pre-trained time series models (Prophet, ARIMA) at startup
- Exposes a REST API for crime predictions with confidence intervals
- Provides explainability through Prophet component decomposition (trend + seasonality)
- Fetches historical data directly from the FBI Crime Data Explorer API
- Deploys to Red Hat OpenShift with health checks and resource management

## Supported Predictions

**Offense Types:**
- `violent-crime` - Violent crime aggregate
- `property-crime` - Property crime aggregate
- `homicide` - Murder and non-negligent manslaughter
- `burglary` - Burglary offenses
- `motor-vehicle-theft` - Motor vehicle theft

**Locations:**
- National (default)
- State-level: CA, TX, FL, NY, IL

## API Endpoints

All endpoints are prefixed with `/api/v1`.

### Health Check
```
GET /api/v1/health
```
Returns service health status and model count.

### List Models
```
GET /api/v1/models
GET /api/v1/models?state=CA
```
Lists all loaded models with their accuracy metrics (MAPE).

### Generate Predictions
```
POST /api/v1/predict/{offense}
POST /api/v1/predict/{offense}?state=CA
```
**Request Body:**
```json
{
  "steps": 6,
  "include_history": false,
  "history_months": 12
}
```

**Response includes:**
- Predictions with confidence intervals (95%)
- Model metadata (type, MAPE, training period)
- Data freshness information
- Explanation with trend/seasonality decomposition and narrative

### Get Historical Data
```
GET /api/v1/history/{offense}?from_year=2020&to_year=2024
GET /api/v1/history/{offense}?from_year=2020&state=CA
```
Fetches real-time historical data from the FBI Crime Data Explorer API.

## Project Structure

```
fbi-ucr/
├── src/fbi_ucr/           # Application source code
│   ├── api/               # FastAPI routes and schemas
│   ├── data/              # FBI API client
│   ├── inference/         # Model loading and prediction service
│   ├── models/            # Model class definitions (Prophet, ARIMA, SARIMA)
│   └── main.py            # FastAPI application entry point
├── models/                # Trained model artifacts (.joblib + metadata)
├── data/                  # Training data (CSV files from FBI API)
├── scripts/               # Training and data fetching scripts
├── manifests/             # OpenShift/Kubernetes deployment manifests
├── Containerfile          # Container build definition
└── pyproject.toml         # Python project configuration
```

## Trained Models

The `models/` directory contains 30 pre-trained models:
- 5 offense types × 6 locations (national + 5 states)
- Each model has a `.joblib` file (serialized model) and `_meta.json` (metadata)

**Model Selection (from model rodeo with grid search):**

| Offense | Model Type | Key Parameters |
|---------|------------|----------------|
| violent-crime | Prophet | yearly_seasonality=True, changepoint_prior_scale=0.5 |
| property-crime | Prophet | yearly_seasonality=False, changepoint_prior_scale=0.5 |
| homicide | Prophet | yearly_seasonality=True, seasonality_prior_scale=0.01 |
| burglary | Prophet | yearly_seasonality=True, changepoint_prior_scale=0.5 |
| motor-vehicle-theft | ARIMA | order=(0,1,1) |

## Local Development

### Prerequisites
- Python 3.11+
- Virtual environment

### Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run the service
MODELS_DIR=models python -m uvicorn fbi_ucr.main:app --reload --port 8080
```

### Testing
```bash
# Health check
curl http://localhost:8080/api/v1/health

# List models
curl http://localhost:8080/api/v1/models

# Get prediction
curl -X POST http://localhost:8080/api/v1/predict/violent-crime \
  -H "Content-Type: application/json" \
  -d '{"steps": 6}'

# State-level prediction
curl -X POST "http://localhost:8080/api/v1/predict/homicide?state=CA" \
  -H "Content-Type: application/json" \
  -d '{"steps": 3}'
```

## Training New Models

### 1. Fetch Data from FBI API
```bash
python scripts/fetch_fbi_data.py
```
Downloads monthly crime data for all offense types and locations to `data/`.

### 2. Run Model Rodeo (Optional)
```bash
python scripts/run_optimized_rodeo.py
```
Compares Prophet, ARIMA, and SARIMA models with grid search to find optimal configurations.

### 3. Train and Export Models
```bash
python scripts/train_models.py
```
Trains models using configurations from the rodeo results and exports to `models/`.

## Container Build

```bash
# Build for OpenShift (from Mac)
podman build --platform linux/amd64 -t quay.io/your-repo/crime-stats-api:latest -f Containerfile .

# Push to registry
podman push quay.io/your-repo/crime-stats-api:latest
```

## OpenShift Deployment

The service deploys to OpenShift with:
- 512Mi-1Gi memory, 250m-500m CPU
- Liveness and readiness probes on `/api/v1/health`
- Models baked into the container image

```bash
# Apply manifests
oc apply -f manifests/base/ -n your-namespace

# Trigger rollout after image update
oc rollout restart deployment/fbi-ucr -n your-namespace
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `FBI_API_KEY` | *(required for data fetch)* | FBI Crime Data Explorer API key ([get free key](https://api.data.gov/signup/)) |
| `MODELS_DIR` | `models` | Directory containing trained model files |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8080` | Server port |
| `CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `RELOAD` | `false` | Enable auto-reload for development |

> **Note:** The `FBI_API_KEY` is only required when fetching new data from the FBI API (training scripts). The prediction API itself does not require an API key since trained models are loaded from disk.

## Explainability

The prediction response includes an `explanation` field with:

- **Method**: `prophet_decomposition` or `arima_trend`
- **Components**:
  - `trend`: Direction (increasing/decreasing/stable) and change percentage
  - `yearly_seasonality`: Current effect, peak months, trough months (Prophet only)
- **Narrative**: Human-readable explanation of the forecast

Example:
```json
{
  "explanation": {
    "method": "prophet_decomposition",
    "components": {
      "trend": {"direction": "stable", "change_pct": -0.8},
      "yearly_seasonality": {
        "current_effect_pct": 3.87,
        "peak_months": ["Jul", "Aug"],
        "trough_months": ["Feb", "Mar"]
      }
    },
    "narrative": "Violent Crime is predicted to remain relatively stable over the forecast period (+0.8%), with seasonal factors adding 3.9% (crime typically peaks in Jul, Aug)."
  }
}
```

## Data Source

Historical crime data is sourced from the [FBI Crime Data Explorer API](https://crime-data-explorer.fr.cloud.gov/). Data typically has a ~2 month reporting lag.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
