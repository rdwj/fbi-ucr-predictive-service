"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# Supported states for state-level predictions
VALID_STATES = frozenset(["CA", "TX", "FL", "NY", "IL"])


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""

    steps: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Number of months to predict (1-12)",
    )
    include_history: bool = Field(
        default=False,
        description="Include historical data in response",
    )
    history_months: int = Field(
        default=12,
        ge=1,
        le=36,
        description="Months of history to include if include_history=True",
    )


class PredictionPoint(BaseModel):
    """Single prediction data point."""

    date: str = Field(description="Month in YYYY-MM format")
    predicted: float = Field(description="Predicted incident count")
    lower: float = Field(description="Lower bound of 95% confidence interval")
    upper: float = Field(description="Upper bound of 95% confidence interval")


class HistoryPoint(BaseModel):
    """Single historical data point."""

    date: str = Field(description="Month in YYYY-MM format")
    actual: float = Field(description="Actual incident count")


class DataFreshness(BaseModel):
    """Information about data freshness and recency."""

    training_end: str = Field(description="Last month of training data (YYYY-MM)")
    months_since_training: int = Field(
        description="Number of months since training data ended"
    )
    note: str = Field(
        default="FBI UCR data has ~2 month reporting lag",
        description="Note about data freshness",
    )


class TrendComponentSchema(BaseModel):
    """Trend component of prediction explanation."""

    direction: str = Field(description="Trend direction: increasing, decreasing, stable")
    change_pct: float = Field(description="Percent change over forecast period")


class SeasonalityComponentSchema(BaseModel):
    """Seasonality component of prediction explanation."""

    current_effect_pct: float = Field(
        description="Effect of seasonality on current forecast period as percentage"
    )
    peak_months: list[str] = Field(description="Months with highest seasonal effect")
    trough_months: list[str] = Field(description="Months with lowest seasonal effect")


class ExplanationComponents(BaseModel):
    """Components that explain the prediction."""

    trend: TrendComponentSchema = Field(description="Trend component")
    yearly_seasonality: Optional[SeasonalityComponentSchema] = Field(
        default=None,
        description="Yearly seasonality component (Prophet models only)",
    )


class Explanation(BaseModel):
    """Explanation of prediction decomposition."""

    method: str = Field(
        description="Decomposition method: prophet_decomposition, arima_trend, sarima_trend"
    )
    components: ExplanationComponents = Field(
        description="Breakdown of prediction components"
    )
    narrative: str = Field(description="Human-readable explanation of the prediction")


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""

    model_type: str = Field(description="Type of model used (ARIMA, SARIMA, Prophet)")
    model_params: dict = Field(description="Model parameters")
    training_end: str = Field(description="Last month of training data")
    mape: float = Field(description="Model MAPE from validation")
    generated_at: datetime = Field(description="Timestamp of prediction generation")
    data_freshness: DataFreshness = Field(description="Information about data recency")


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""

    offense: str = Field(description="Crime type predicted")
    location: str = Field(
        default="national",
        description="Geographic location: 'national' or state code (CA, TX, FL, NY, IL)",
    )
    predictions: list[PredictionPoint] = Field(description="Predicted values")
    history: Optional[list[HistoryPoint]] = Field(
        default=None,
        description="Historical data if requested",
    )
    metadata: PredictionMetadata = Field(description="Prediction metadata")
    explanation: Explanation = Field(description="Explanation of prediction components")


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    offense: str = Field(description="Crime type this model predicts")
    location: str = Field(
        default="national",
        description="Geographic location: 'national' or state code",
    )
    model_type: str = Field(description="Type of model (ARIMA, SARIMA, Prophet)")
    model_params: dict = Field(description="Model configuration parameters")
    mape: float = Field(description="Validation MAPE percentage")
    training_end: str = Field(description="Last month of training data")
    last_loaded: datetime = Field(description="When model was loaded")


class ModelsResponse(BaseModel):
    """Response listing all available models."""

    models: list[ModelInfo] = Field(description="List of loaded models")
    total: int = Field(description="Total number of models")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    models_loaded: int = Field(description="Number of models loaded")
    uptime_seconds: float = Field(description="Service uptime in seconds")


class HistoryRequest(BaseModel):
    """Request for historical data."""

    months: int = Field(
        default=24,
        ge=1,
        le=60,
        description="Number of months of history to return",
    )


class HistoryResponse(BaseModel):
    """Response with historical crime data."""

    offense: str = Field(description="Crime type")
    location: str = Field(
        default="national",
        description="Geographic location: 'national' or state code",
    )
    data: list[HistoryPoint] = Field(description="Historical data points")
    from_date: str = Field(description="Start date (YYYY-MM)")
    to_date: str = Field(description="End date (YYYY-MM)")
    total_months: int = Field(description="Number of months returned")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error info")
