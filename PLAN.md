# FBI UCR Real Data + State-Level Support Plan

## Overview

Upgrade the FBI UCR crime prediction system from sample data to real FBI API data, and add state-level prediction support.

## Phase 1: Data Fetching Infrastructure

**Location:** `/Users/wjackson/Developer/PoC/fbi-ucr/scripts/fetch_fbi_data.py`

**Tasks:**
- Create FBI API client with rate limiting and error handling
- Fetch national data for all 5 offenses (2020-2024, ~60 months)
- Fetch state data for a representative set of states (CA, TX, NY, FL, IL - top 5 by population)
- Save raw data to `/Users/wjackson/Developer/PoC/fbi-ucr/data/` as CSV files
- Data structure: `date, offense, location, actual_count, rate`

**API Details:**
- Base URL: `https://api.usa.gov/crime/fbi/cde`
- Endpoints:
  - `/summarized/national/{offense}?from=MM-YYYY&to=MM-YYYY`
  - `/summarized/state/{state}/{offense}?from=MM-YYYY&to=MM-YYYY`
- Data includes both `rates` (per 100K) and `actuals` (raw counts)

## Phase 2: Model Training with Real Data

**Location:** `/Users/wjackson/Developer/PoC/fbi-ucr/scripts/train_models.py`

**Tasks:**
- Load fetched FBI data from CSV files
- Train national models for each offense using same model types (ARIMA, SARIMA, Prophet based on rodeo results)
- Train state-level models for selected states
- Calculate actual MAPE on held-out test data (not hardcoded)
- Export models to `/Users/wjackson/Developer/PoC/fbi-ucr/models/`

**Model Naming Convention:**
- National: `{offense}.joblib` or `{offense}.json`
- State: `{offense}_{state}.joblib` or `{offense}_{state}.json`
- Metadata: `{offense}_meta.json` and `{offense}_{state}_meta.json`

## Phase 3: Update Backend API

**Location:** `/Users/wjackson/Developer/PoC/fbi-ucr/src/fbi_ucr/`

**Tasks:**
- Update `inference/loader.py` to load state-level models
- Update `inference/predictor.py` to handle state parameter
- Update `api/routes.py` to accept optional `state` query parameter
- Update `api/schemas.py` with state field

**API Changes:**
- `POST /api/v1/predict/{offense}?state=CA` - State-level prediction
- `GET /api/v1/history/{offense}?state=CA&months=12` - State-level history
- `GET /api/v1/models?state=CA` - List models available for a state

## Phase 4: Update MCP Tools

**Location:** `/Users/wjackson/Developer/PoC/crime_stats/fbi-crime-stats-mcp/src/tools/`

**Tasks:**
- Add `state` parameter to `ucr_forecast` (optional, default=None for national)
- Add `state` parameter to `ucr_compare` (compare same offense across states OR different offenses in same state)
- Update `ucr_info` to list available states and their models
- Add supportability statement to all tool descriptions

## Phase 5: Supportability Statement

**Add to each tool description:**

```
SUPPORTED SCOPE:
- Geographic: National level + 5 states (CA, TX, NY, FL, IL)
- Offenses: violent-crime, property-crime, homicide, burglary, motor-vehicle-theft
- Time horizon: 1-12 months ahead
- Data source: FBI Uniform Crime Reporting (UCR) Program
- Training data: January 2020 - October 2024

NOT SUPPORTED:
- Other states (coming soon)
- City/county level predictions
- Other offense types (robbery, assault, larceny, arson)
- Real-time data (data has ~2 month lag)
- Demographic breakdowns
```

## Phase 6: Rebuild and Deploy

**Tasks:**
- Rebuild fbi-ucr container with new models
- Push to quay.io
- Redeploy to OpenShift
- Rebuild fbi-crime-stats-mcp with updated tools
- Redeploy MCP server
- Test all endpoints

## File Changes Summary

| File | Action |
|------|--------|
| `fbi-ucr/scripts/fetch_fbi_data.py` | Create - FBI API data fetcher |
| `fbi-ucr/scripts/train_models.py` | Create - Model training with real data |
| `fbi-ucr/data/*.csv` | Create - Raw FBI data files |
| `fbi-ucr/models/*` | Replace - Real trained models |
| `fbi-ucr/src/fbi_ucr/inference/loader.py` | Update - State-level model loading |
| `fbi-ucr/src/fbi_ucr/inference/predictor.py` | Update - State parameter handling |
| `fbi-ucr/src/fbi_ucr/api/routes.py` | Update - State query parameter |
| `fbi-ucr/src/fbi_ucr/api/schemas.py` | Update - State field |
| `fbi-crime-stats-mcp/src/tools/ucr_forecast.py` | Update - State parameter + supportability |
| `fbi-crime-stats-mcp/src/tools/ucr_compare.py` | Update - State parameter + supportability |
| `fbi-crime-stats-mcp/src/tools/ucr_info.py` | Update - State listing + supportability |
| `fbi-crime-stats-mcp/TOOLS_PLAN.md` | Update - Document state support |

## States to Support Initially

Top 5 US states by population:
1. California (CA)
2. Texas (TX)
3. Florida (FL)
4. New York (NY)
5. Illinois (IL)

This gives good geographic diversity and covers ~38% of US population.

## Offenses Supported

| Offense | Model Type | Notes |
|---------|------------|-------|
| violent-crime | ARIMA | Aggregate violent crimes |
| property-crime | ARIMA | Aggregate property crimes |
| homicide | Prophet | Best for seasonal patterns |
| burglary | ARIMA | Property crime subset |
| motor-vehicle-theft | SARIMA | Strong seasonality |

## Future Enhancements (Out of Scope)

- Additional states (all 50 + DC)
- Additional offenses: robbery, aggravated-assault, larceny, arson
- Agency-level predictions
- Demographic breakdowns (not available via API)

## Risks

1. **FBI API rate limits** - Need to add delays between requests
2. **Missing data** - Some state/offense combinations may have gaps
3. **Model accuracy** - Real data may produce different accuracy than sample data
4. **Container size** - State models will increase container size (5 offenses × 6 locations = 30 models)

---

**Created:** 2024-12-01
**Status:** Phases 1-5 Complete, Ready for Deploy
