# Machine Learning Primer: Time Series Forecasting

This document provides an introduction to time series forecasting and the models used in this project. If you're new to machine learning or time series analysis, start here before diving into the codebase.

## What is Time Series Prediction?

Time series prediction (or forecasting) is the task of predicting future values based on historical observations ordered by time. Unlike other machine learning problems where data points are independent, time series data has a temporal structure—what happened yesterday influences what happens today, and seasonal patterns (like crime rates rising in summer) repeat predictably. The goal is to learn these patterns from historical data and project them forward. Key concepts include **trend** (the long-term direction), **seasonality** (repeating patterns at fixed intervals like yearly or weekly), and **noise** (random variation). For crime statistics, we're looking at monthly data points spanning several years, trying to forecast the next 3-12 months while accounting for both the overall trend in crime rates and seasonal patterns like summer increases in violent crime.

## Prophet

Prophet is a forecasting model developed by Facebook (now Meta) designed to handle business time series with strong seasonal effects and missing data. It works by decomposing the time series into three additive components: trend (piecewise linear or logistic growth), seasonality (Fourier series for yearly/weekly/daily patterns), and holidays/events. Prophet is particularly well-suited for our crime data because it handles the yearly seasonality in crime rates elegantly, is robust to missing data and outliers, and requires minimal hyperparameter tuning to get good results. The key parameters we tune are `changepoint_prior_scale` (how flexible the trend is—higher values allow more trend changes) and `seasonality_prior_scale` (how strongly to fit seasonal patterns). Prophet excels when you have at least a year of data and clear seasonal patterns.

**Learn more:** [Prophet Documentation](https://facebook.github.io/prophet/) | [Prophet Paper](https://peerj.com/preprints/3190/)

## ARIMA

ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical model for time series forecasting that has been a workhorse in econometrics and forecasting for decades. The model is defined by three parameters: **p** (autoregressive order—how many past values to use), **d** (differencing order—how many times to difference the data to make it stationary), and **q** (moving average order—how many past forecast errors to use). An ARIMA(1,1,1) model, for example, uses one lagged value, differences the data once, and uses one lagged error term. ARIMA works well for data without strong seasonality or when the seasonal pattern has been removed. It's mathematically elegant and interpretable, but requires the data to be stationary (constant mean and variance over time), which is why the "I" (integrated/differencing) component exists. We use the `pmdarima` library's `auto_arima` function to automatically search for optimal (p,d,q) values using AIC (Akaike Information Criterion).

**Learn more:** [ARIMA Explained](https://otexts.com/fpp3/arima.html) | [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)

## SARIMA

SARIMA (Seasonal ARIMA) extends ARIMA to explicitly model seasonal patterns. It adds three seasonal parameters—**P**, **D**, **Q**—plus a seasonal period **m** (12 for monthly data with yearly seasonality). A SARIMA(1,1,1)(1,0,1,12) model combines the non-seasonal ARIMA(1,1,1) with seasonal autoregressive and moving average terms at lag 12 (one year). This allows the model to capture patterns like "January is typically lower than July" directly in its structure. SARIMA is powerful when seasonality is strong and consistent, but it adds complexity and requires more data to estimate reliably. In our rodeo testing, SARIMA sometimes performed well but often the seasonal component added more noise than signal, especially for state-level data with fewer observations. When SARIMA works, it works well; when it doesn't, Prophet's more flexible seasonality handling tends to win.

**Learn more:** [Seasonal ARIMA](https://otexts.com/fpp3/seasonal-arima.html) | [pmdarima auto_arima](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)

## Running a Model Rodeo

A "model rodeo" is the process of systematically comparing multiple models on the same data to determine which performs best. Rather than assuming one model is always superior, we let the data decide. The basic approach is:

1. **Split the data** into training and test sets (we use the last 6 months as test data)
2. **Train each candidate model** on the training data
3. **Generate forecasts** for the test period
4. **Calculate error metrics** comparing predictions to actual values
5. **Select the winner** based on the metric that matters most (we use MAPE)

**MAPE (Mean Absolute Percentage Error)** is our primary metric because it's interpretable—a MAPE of 5% means predictions are off by 5% on average—and it's scale-independent, allowing comparison across different offense types with vastly different counts.

### How We Did Our Rodeo

For this project, we ran an optimized rodeo with the following approach:

**Models Tested:**

- **ARIMA**: Used `auto_arima` with stepwise AIC optimization to find the best (p,d,q) order automatically
- **SARIMA**: Used `auto_arima` with seasonal=True and m=12 for yearly seasonality
- **Prophet**: Grid search over 32 parameter combinations:
  - `changepoint_prior_scale`: [0.001, 0.01, 0.1, 0.5]
  - `seasonality_prior_scale`: [0.01, 0.1, 1.0, 10.0]
  - `yearly_seasonality`: [True, False]

**Data:**

- 52 months of training data (roughly 4.3 years)
- 6 months held out for testing
- Tested on 5 offense types × 6 locations = 30 combinations

**Results Summary:**

| Offense             | Winner       | MAPE | Why It Won                                                                             |
| ------------------- | ------------ | ---- | -------------------------------------------------------------------------------------- |
| violent-crime       | Prophet      | 2.1% | Strong yearly seasonality captured well                                                |
| property-crime      | Prophet      | 1.8% | Flexible trend handling beat ARIMA                                                     |
| homicide            | Prophet      | 1.9% | ARIMA/SARIMA struggled (14%+ MAPE)                                                     |
| burglary            | Prophet      | 1.3% | Clear seasonal pattern, Prophet excelled                                               |
| motor-vehicle-theft | ARIMA(0,1,1) | 4.0% | Simple differencing worked; Prophet [overfit](https://en.wikipedia.org/wiki/Overfitting) |

**Key Findings:**

- Prophet won 4 out of 5 offense types at the national level
- ARIMA won for motor-vehicle-theft, which has a strong trend but weak seasonality
- SARIMA rarely outperformed both ARIMA and Prophet—the added complexity didn't pay off
- State-level models showed more variation, but Prophet remained dominant
- Grid search was essential for Prophet—default parameters often lost to tuned ARIMA

**Running Your Own Rodeo:**

```bash
# Fetch fresh data from FBI API
python scripts/fetch_fbi_data.py

# Run the optimized rodeo (takes ~10-15 minutes)
python scripts/run_optimized_rodeo.py

# Results saved to outputs/optimized_rodeo_results.csv
```

The rodeo results CSV contains every model tested with its parameters and metrics, making it easy to analyze which configurations work best for your specific use case.

---

**Back to:** [Main README](../README.md)
