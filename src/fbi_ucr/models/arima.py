"""ARIMA model implementation for inference."""

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .base import BasePredictor, ComponentBreakdown, PredictionResult, TrendComponent


class ARIMAPredictor(BasePredictor):
    """ARIMA time series predictor."""

    name = "ARIMA"

    def __init__(self, order: tuple = (1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "ARIMAPredictor":
        """Fit ARIMA model."""
        self.model = ARIMA(y, order=self.order)
        self.fitted = self.model.fit()
        return self

    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> PredictionResult:
        """Generate predictions with confidence intervals."""
        if self.fitted is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.fitted.get_forecast(steps=steps)
        predicted = forecast.predicted_mean.values
        conf_int = forecast.conf_int(alpha=0.05)

        # Extract trend component
        components = self._extract_components(predicted)

        return PredictionResult(
            predicted=predicted,
            lower_bound=conf_int.iloc[:, 0].values,
            upper_bound=conf_int.iloc[:, 1].values,
            confidence_level=0.95,
            components=components,
        )

    def _extract_components(self, predicted: np.ndarray) -> ComponentBreakdown:
        """Extract trend component from ARIMA predictions.

        ARIMA doesn't decompose into trend/seasonality like Prophet,
        so we calculate trend from the predicted values themselves.

        Args:
            predicted: Array of predicted values

        Returns:
            ComponentBreakdown with trend info only
        """
        if len(predicted) < 2:
            trend_change_pct = 0.0
            trend_direction = "stable"
        else:
            first_val = predicted[0]
            last_val = predicted[-1]

            if first_val != 0:
                trend_change_pct = ((last_val - first_val) / abs(first_val)) * 100
            else:
                trend_change_pct = 0.0

            if trend_change_pct > 1.0:
                trend_direction = "increasing"
            elif trend_change_pct < -1.0:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"

        return ComponentBreakdown(
            method="arima_trend",
            trend=TrendComponent(
                direction=trend_direction,
                change_pct=round(trend_change_pct, 2),
            ),
            yearly_seasonality=None,
        )

    def get_params(self) -> dict:
        """Return model parameters."""
        return {"order": self.order}
