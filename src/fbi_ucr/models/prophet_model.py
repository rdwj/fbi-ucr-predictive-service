"""Prophet model implementation for inference."""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .base import (
    BasePredictor,
    ComponentBreakdown,
    PredictionResult,
    SeasonalityComponent,
    TrendComponent,
)

warnings.filterwarnings("ignore")

# Month abbreviations for seasonality reporting
MONTH_ABBREVS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class ProphetPredictor(BasePredictor):
    """Facebook Prophet time series predictor."""

    name = "Prophet"

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.model = None
        self.train_df = None

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "ProphetPredictor":
        """Fit Prophet model."""
        from prophet import Prophet

        # Prophet requires specific column names
        self.train_df = pd.DataFrame({
            "ds": y.index if isinstance(y.index, pd.DatetimeIndex) else pd.to_datetime(y.index),
            "y": y.values,
        })

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )

        self.model.fit(self.train_df)
        return self

    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> PredictionResult:
        """Generate predictions with confidence intervals."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps, freq="MS")
        forecast = self.model.predict(future)

        # Get only the forecast period
        forecast_period = forecast.tail(steps)

        # Extract component breakdown
        components = self._extract_components(forecast, steps)

        return PredictionResult(
            predicted=forecast_period["yhat"].values,
            lower_bound=forecast_period["yhat_lower"].values,
            upper_bound=forecast_period["yhat_upper"].values,
            confidence_level=0.95,
            components=components,
        )

    def _extract_components(
        self, forecast: pd.DataFrame, steps: int
    ) -> ComponentBreakdown:
        """Extract trend and seasonality components from Prophet forecast.

        Args:
            forecast: Full Prophet forecast DataFrame
            steps: Number of forecast steps

        Returns:
            ComponentBreakdown with trend and seasonality info
        """
        forecast_period = forecast.tail(steps)

        # Calculate trend direction and change
        trend_start = forecast_period["trend"].iloc[0]
        trend_end = forecast_period["trend"].iloc[-1]

        if trend_start != 0:
            trend_change_pct = ((trend_end - trend_start) / abs(trend_start)) * 100
        else:
            trend_change_pct = 0.0

        if trend_change_pct > 1.0:
            trend_direction = "increasing"
        elif trend_change_pct < -1.0:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        trend = TrendComponent(
            direction=trend_direction,
            change_pct=round(trend_change_pct, 2),
        )

        # Extract yearly seasonality if present
        yearly_seasonality = None
        if self.yearly_seasonality and "yearly" in forecast.columns:
            yearly_seasonality = self._extract_yearly_seasonality(forecast, steps)

        return ComponentBreakdown(
            method="prophet_decomposition",
            trend=trend,
            yearly_seasonality=yearly_seasonality,
        )

    def _extract_yearly_seasonality(
        self, forecast: pd.DataFrame, steps: int
    ) -> SeasonalityComponent:
        """Extract yearly seasonality component details.

        Args:
            forecast: Full Prophet forecast DataFrame
            steps: Number of forecast steps

        Returns:
            SeasonalityComponent with peak/trough months and current effect
        """
        forecast_period = forecast.tail(steps)

        # Get the average yearly effect across forecast period
        current_effect = forecast_period["yearly"].mean()

        # Calculate effect as percentage of typical prediction
        avg_prediction = forecast_period["yhat"].mean()
        if avg_prediction != 0:
            current_effect_pct = (current_effect / abs(avg_prediction)) * 100
        else:
            current_effect_pct = 0.0

        # Find peak and trough months from historical pattern
        # Use the full forecast to get a full year of seasonality
        peak_months, trough_months = self._find_seasonal_extremes(forecast)

        return SeasonalityComponent(
            current_effect_pct=round(current_effect_pct, 2),
            peak_months=peak_months,
            trough_months=trough_months,
        )

    def _find_seasonal_extremes(
        self, forecast: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        """Find months with highest and lowest seasonal effects.

        Args:
            forecast: Full Prophet forecast DataFrame

        Returns:
            Tuple of (peak_months, trough_months) as month abbreviations
        """
        if "yearly" not in forecast.columns:
            return [], []

        # Group by month and get average seasonal effect
        forecast_copy = forecast.copy()
        forecast_copy["month"] = forecast_copy["ds"].dt.month

        monthly_effects = forecast_copy.groupby("month")["yearly"].mean()

        if len(monthly_effects) == 0:
            return [], []

        # Find top 2 and bottom 2 months
        sorted_effects = monthly_effects.sort_values(ascending=False)

        peak_indices = sorted_effects.head(2).index.tolist()
        trough_indices = sorted_effects.tail(2).index.tolist()

        # Convert to month abbreviations (1-indexed to 0-indexed)
        peak_months = [MONTH_ABBREVS[m - 1] for m in peak_indices]
        trough_months = [MONTH_ABBREVS[m - 1] for m in trough_indices]

        return peak_months, trough_months

    def get_params(self) -> dict:
        """Return model parameters."""
        return {
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
        }
