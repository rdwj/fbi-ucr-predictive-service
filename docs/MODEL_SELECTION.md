# Model Selection Results (Optimized)

This document tracks the winning model for each crime data series based on optimized hyperparameter search.

**Generated:** 2025-12-01 02:08
**Data Source:** Real FBI UCR data (fetched from Crime Data Explorer API)
**Test Period:** 6 months held out for evaluation
**Optimization:** auto_arima (ARIMA/SARIMA) + grid search (Prophet)

## Summary Table - National

| Offense | Winning Model | MAPE | MAE | Training Period |
|---------|---------------|------|-----|-----------------|
| violent-crime | Prophet(yearly, cp=0.5, sp=0.1) | 2.1% | 2,212 | 52 months |
| property-crime | Prophet(no-season, cp=0.5, sp=0.01) | 1.8% | 8,695 | 52 months |
| homicide | Prophet(yearly, cp=0.5, sp=0.01) | 1.9% | 28 | 52 months |
| burglary | Prophet(yearly, cp=0.5, sp=0.1) | 1.3% | 832 | 52 months |
| motor-vehicle-theft | ARIMA(0, 1, 1) | 4.0% | 2,847 | 52 months |

## Summary Table - By State

| State | Offense | Winning Model | MAPE | MAE |
|-------|---------|---------------|------|-----|
| CA | violent-crime | Prophet(no-season, cp=0.5, sp=0.01) | 1.9% | 305 |
| CA | property-crime | SARIMA(1, 0, 0)(0, 0, 1, 12) | 5.3% | 3,357 |
| CA | homicide | ARIMA(1, 0, 0) | 9.9% | 16 |
| CA | burglary | ARIMA(1, 0, 0) | 3.9% | 394 |
| CA | motor-vehicle-theft | SARIMA(1, 0, 0)(0, 0, 1, 12) | 7.3% | 872 |
| TX | violent-crime | Prophet(yearly, cp=0.001, sp=0.1) | 2.1% | 228 |
| TX | property-crime | Prophet(no-season, cp=0.5, sp=0.01) | 2.2% | 1,147 |
| TX | homicide | Prophet(yearly, cp=0.1, sp=0.1) | 6.4% | 9 |
| TX | burglary | SARIMA(1, 1, 0)(1, 0, 0, 12) | 3.4% | 262 |
| TX | motor-vehicle-theft | ARIMA(1, 1, 0) | 7.4% | 591 |
| FL | violent-crime | Prophet(yearly, cp=0.5, sp=0.01) | 10.1% | 336 |
| FL | property-crime | Prophet(yearly, cp=0.5, sp=0.01) | 10.3% | 1,673 |
| FL | homicide | Prophet(yearly, cp=0.5, sp=0.1) | 21.0% | 10 |
| FL | burglary | Prophet(no-season, cp=0.5, sp=0.01) | 14.5% | 248 |
| FL | motor-vehicle-theft | Prophet(yearly, cp=0.5, sp=0.1) | 20.3% | 274 |
| NY | violent-crime | Prophet(yearly, cp=0.5, sp=1.0) | 3.3% | 227 |
| NY | property-crime | Prophet(yearly, cp=0.5, sp=1.0) | 3.3% | 970 |
| NY | homicide | Prophet(no-season, cp=0.001, sp=0.01) | 9.3% | 6 |
| NY | burglary | Prophet(yearly, cp=0.5, sp=0.1) | 6.6% | 152 |
| NY | motor-vehicle-theft | ARIMA(0, 1, 0) | 10.8% | 310 |
| IL | violent-crime | Prophet(yearly, cp=0.5, sp=0.01) | 6.4% | 201 |
| IL | property-crime | Prophet(yearly, cp=0.5, sp=1.0) | 4.8% | 919 |
| IL | homicide | Prophet(yearly, cp=0.001, sp=0.01) | 11.4% | 8 |
| IL | burglary | Prophet(yearly, cp=0.5, sp=0.01) | 7.0% | 226 |
| IL | motor-vehicle-theft | Prophet(no-season, cp=0.5, sp=0.01) | 4.5% | 149 |

## Key Findings

- **Prophet** wins 22/30 datasets (73%)
- **ARIMA** wins 5/30 datasets (17%)
- **SARIMA** wins 3/30 datasets (10%)

## Production Model Configuration

```python
# Model assignments for deployment (optimized hyperparameters)
MODEL_CONFIG = {
    "violent-crime": {"type": "Prophet", "params": {"changepoint_prior_scale": 0.5, "seasonality_prior_scale": 0.1, "yearly_seasonality": True}},
    "property-crime": {"type": "Prophet", "params": {"changepoint_prior_scale": 0.5, "seasonality_prior_scale": 0.01, "yearly_seasonality": False}},
    "homicide": {"type": "Prophet", "params": {"changepoint_prior_scale": 0.5, "seasonality_prior_scale": 0.01, "yearly_seasonality": True}},
    "burglary": {"type": "Prophet", "params": {"changepoint_prior_scale": 0.5, "seasonality_prior_scale": 0.1, "yearly_seasonality": True}},
    "motor-vehicle-theft": {"type": "ARIMA", "params": {"order": (0, 1, 1)}},
}
```

## Detailed Results by Location

### national

**violent-crime**
- Winner: Prophet(yearly, cp=0.5, sp=0.1)
- MAPE: 2.10%
- MAE: 2,212
- RMSE: 2,443
- Training: 52 months
- Optimization: grid search (32 combinations)

**property-crime**
- Winner: Prophet(no-season, cp=0.5, sp=0.01)
- MAPE: 1.77%
- MAE: 8,695
- RMSE: 9,930
- Training: 52 months
- Optimization: grid search (32 combinations)

**homicide**
- Winner: Prophet(yearly, cp=0.5, sp=0.01)
- MAPE: 1.92%
- MAE: 28
- RMSE: 31
- Training: 52 months
- Optimization: grid search (32 combinations)

**burglary**
- Winner: Prophet(yearly, cp=0.5, sp=0.1)
- MAPE: 1.26%
- MAE: 832
- RMSE: 1,087
- Training: 52 months
- Optimization: grid search (32 combinations)

**motor-vehicle-theft**
- Winner: ARIMA(0, 1, 1)
- MAPE: 3.99%
- MAE: 2,847
- RMSE: 3,106
- Training: 52 months
- Optimization: auto_arima (stepwise AIC)

### State: CA

**violent-crime**
- Winner: Prophet(no-season, cp=0.5, sp=0.01)
- MAPE: 1.87%
- MAE: 305
- RMSE: 385
- Training: 52 months
- Optimization: grid search (32 combinations)

**property-crime**
- Winner: SARIMA(1, 0, 0)(0, 0, 1, 12)
- MAPE: 5.35%
- MAE: 3,357
- RMSE: 4,117
- Training: 52 months
- Optimization: auto_arima (stepwise AIC)

**homicide**
- Winner: ARIMA(1, 0, 0)
- MAPE: 9.93%
- MAE: 16
- RMSE: 23
- Training: 52 months
- Optimization: auto_arima (stepwise AIC)

**burglary**
- Winner: ARIMA(1, 0, 0)
- MAPE: 3.85%
- MAE: 394
- RMSE: 516
- Training: 52 months
- Optimization: auto_arima (stepwise AIC)

**motor-vehicle-theft**
- Winner: SARIMA(1, 0, 0)(0, 0, 1, 12)
- MAPE: 7.30%
- MAE: 872
- RMSE: 1,218
- Training: 52 months
- Optimization: auto_arima (stepwise AIC)

### State: TX

**violent-crime**
- Winner: Prophet(yearly, cp=0.001, sp=0.1)
- MAPE: 2.11%
- MAE: 228
- RMSE: 297
- Training: 52 months
- Optimization: grid search (32 combinations)

**property-crime**
- Winner: Prophet(no-season, cp=0.5, sp=0.01)
- MAPE: 2.17%
- MAE: 1,147
- RMSE: 1,488
- Training: 52 months
- Optimization: grid search (32 combinations)

**homicide**
- Winner: Prophet(yearly, cp=0.1, sp=0.1)
- MAPE: 6.42%
- MAE: 9
- RMSE: 13
- Training: 52 months
- Optimization: grid search (32 combinations)

**burglary**
- Winner: SARIMA(1, 1, 0)(1, 0, 0, 12)
- MAPE: 3.35%
- MAE: 262
- RMSE: 351
- Training: 52 months
- Optimization: auto_arima (stepwise AIC)

**motor-vehicle-theft**
- Winner: ARIMA(1, 1, 0)
- MAPE: 7.39%
- MAE: 591
- RMSE: 773
- Training: 52 months
- Optimization: auto_arima (stepwise AIC)

### State: FL

**violent-crime**
- Winner: Prophet(yearly, cp=0.5, sp=0.01)
- MAPE: 10.13%
- MAE: 336
- RMSE: 393
- Training: 52 months
- Optimization: grid search (32 combinations)

**property-crime**
- Winner: Prophet(yearly, cp=0.5, sp=0.01)
- MAPE: 10.31%
- MAE: 1,673
- RMSE: 1,857
- Training: 52 months
- Optimization: grid search (32 combinations)

**homicide**
- Winner: Prophet(yearly, cp=0.5, sp=0.1)
- MAPE: 21.04%
- MAE: 10
- RMSE: 13
- Training: 52 months
- Optimization: grid search (32 combinations)

**burglary**
- Winner: Prophet(no-season, cp=0.5, sp=0.01)
- MAPE: 14.47%
- MAE: 248
- RMSE: 273
- Training: 52 months
- Optimization: grid search (32 combinations)

**motor-vehicle-theft**
- Winner: Prophet(yearly, cp=0.5, sp=0.1)
- MAPE: 20.29%
- MAE: 274
- RMSE: 346
- Training: 52 months
- Optimization: grid search (32 combinations)

### State: NY

**violent-crime**
- Winner: Prophet(yearly, cp=0.5, sp=1.0)
- MAPE: 3.33%
- MAE: 227
- RMSE: 305
- Training: 52 months
- Optimization: grid search (32 combinations)

**property-crime**
- Winner: Prophet(yearly, cp=0.5, sp=1.0)
- MAPE: 3.34%
- MAE: 970
- RMSE: 1,093
- Training: 52 months
- Optimization: grid search (32 combinations)

**homicide**
- Winner: Prophet(no-season, cp=0.001, sp=0.01)
- MAPE: 9.28%
- MAE: 6
- RMSE: 9
- Training: 52 months
- Optimization: grid search (32 combinations)

**burglary**
- Winner: Prophet(yearly, cp=0.5, sp=0.1)
- MAPE: 6.60%
- MAE: 152
- RMSE: 183
- Training: 52 months
- Optimization: grid search (32 combinations)

**motor-vehicle-theft**
- Winner: ARIMA(0, 1, 0)
- MAPE: 10.83%
- MAE: 310
- RMSE: 392
- Training: 52 months
- Optimization: auto_arima (stepwise AIC)

### State: IL

**violent-crime**
- Winner: Prophet(yearly, cp=0.5, sp=0.01)
- MAPE: 6.37%
- MAE: 201
- RMSE: 233
- Training: 52 months
- Optimization: grid search (32 combinations)

**property-crime**
- Winner: Prophet(yearly, cp=0.5, sp=1.0)
- MAPE: 4.85%
- MAE: 919
- RMSE: 1,177
- Training: 52 months
- Optimization: grid search (32 combinations)

**homicide**
- Winner: Prophet(yearly, cp=0.001, sp=0.01)
- MAPE: 11.40%
- MAE: 8
- RMSE: 9
- Training: 52 months
- Optimization: grid search (32 combinations)

**burglary**
- Winner: Prophet(yearly, cp=0.5, sp=0.01)
- MAPE: 6.95%
- MAE: 226
- RMSE: 244
- Training: 52 months
- Optimization: grid search (32 combinations)

**motor-vehicle-theft**
- Winner: Prophet(no-season, cp=0.5, sp=0.01)
- MAPE: 4.51%
- MAE: 149
- RMSE: 192
- Training: 52 months
- Optimization: grid search (32 combinations)

## Optimization Methods

### ARIMA/SARIMA
- **Method:** pmdarima auto_arima with stepwise AIC selection
- **Search space:** p,q ∈ [0,3], d auto-detected, P,Q ∈ [0,2], D auto-detected, m=12

### Prophet
- **Method:** Grid search over key hyperparameters
- **Parameters searched:**
  - changepoint_prior_scale: [0.001, 0.01, 0.1, 0.5]
  - seasonality_prior_scale: [0.01, 0.1, 1.0, 10.0]
  - yearly_seasonality: [True, False]
- **Total combinations:** 32
