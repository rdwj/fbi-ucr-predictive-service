"""Base model interface for crime prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TrendComponent:
    """Trend component of the prediction explanation."""

    direction: str  # "increasing", "decreasing", "stable"
    change_pct: float  # Percent change over forecast period


@dataclass
class SeasonalityComponent:
    """Seasonality component of the prediction explanation."""

    current_effect_pct: float  # Effect on current forecast period
    peak_months: list[str]  # Months with highest seasonal effect
    trough_months: list[str]  # Months with lowest seasonal effect


@dataclass
class ComponentBreakdown:
    """Breakdown of prediction into interpretable components."""

    method: str  # "prophet_decomposition" or "arima_trend"
    trend: TrendComponent
    yearly_seasonality: Optional[SeasonalityComponent] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "method": self.method,
            "components": {
                "trend": {
                    "direction": self.trend.direction,
                    "change_pct": self.trend.change_pct,
                }
            },
        }
        if self.yearly_seasonality:
            result["components"]["yearly_seasonality"] = {
                "current_effect_pct": self.yearly_seasonality.current_effect_pct,
                "peak_months": self.yearly_seasonality.peak_months,
                "trough_months": self.yearly_seasonality.trough_months,
            }
        return result


@dataclass
class PredictionResult:
    """Container for prediction results with confidence intervals."""

    predicted: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float = 0.95
    components: Optional[ComponentBreakdown] = None


class BasePredictor(ABC):
    """Abstract base class for all prediction models."""

    name: str = "BasePredictor"

    @abstractmethod
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "BasePredictor":
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, steps: int, X: Optional[pd.DataFrame] = None) -> PredictionResult:
        """Generate predictions for future time steps."""
        pass

    def get_params(self) -> dict:
        """Return model parameters."""
        return {}
