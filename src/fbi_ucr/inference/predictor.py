"""Prediction service using loaded models."""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from ..api.schemas import (
    DataFreshness,
    Explanation,
    ExplanationComponents,
    HistoryPoint,
    PredictionMetadata,
    PredictionPoint,
    PredictionResponse,
    SeasonalityComponentSchema,
    TrendComponentSchema,
)
from ..models.base import ComponentBreakdown
from .loader import ModelManager

logger = logging.getLogger(__name__)


# Narrative templates for explanation generation
NARRATIVE_TEMPLATES = {
    "increasing_with_seasonality": (
        "{offense_display} is predicted to {change_verb} {change_pct:.1f}% over the "
        "forecast period. This is driven by an ongoing upward trend ({trend_pct:+.1f}%), "
        "{seasonality_clause}."
    ),
    "decreasing_with_seasonality": (
        "{offense_display} is predicted to {change_verb} {change_pct:.1f}% over the "
        "forecast period. This reflects a downward trend ({trend_pct:+.1f}%), "
        "{seasonality_clause}."
    ),
    "stable_with_seasonality": (
        "{offense_display} is predicted to remain relatively stable over the forecast "
        "period ({change_pct:+.1f}%). {seasonality_clause}."
    ),
    "increasing_no_seasonality": (
        "{offense_display} is predicted to {change_verb} {change_pct:.1f}% over the "
        "forecast period based on recent historical patterns."
    ),
    "decreasing_no_seasonality": (
        "{offense_display} is predicted to {change_verb} {change_pct:.1f}% over the "
        "forecast period based on recent historical patterns."
    ),
    "stable_no_seasonality": (
        "{offense_display} is predicted to remain relatively stable over the forecast "
        "period ({change_pct:+.1f}%) based on recent historical patterns."
    ),
}


class PredictionService:
    """Service for generating crime predictions."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def predict(
        self,
        offense: str,
        steps: int = 6,
        include_history: bool = False,
        history_months: int = 12,
        state: Optional[str] = None,
    ) -> PredictionResponse:
        """Generate predictions for a crime type.

        Args:
            offense: Crime type to predict
            steps: Number of months to forecast
            include_history: Whether to include historical data
            history_months: Number of history months to include
            state: Optional state code (CA, TX, FL, NY, IL) for state-level prediction.
                   If None, returns national-level prediction.
        """
        # Get model and metadata
        model = self.model_manager.get_model(offense, state=state)
        meta = self.model_manager.get_metadata(offense, state=state)

        # Generate prediction
        result = model.predict(steps=steps)

        # Build prediction dates
        training_end = meta.get("training_end", "unknown")
        if training_end != "unknown":
            try:
                last_date = pd.to_datetime(training_end)
            except Exception:
                last_date = pd.Timestamp.now().replace(day=1)
        else:
            last_date = pd.Timestamp.now().replace(day=1)

        # Generate future dates
        prediction_points = []
        for i in range(steps):
            pred_date = last_date + pd.DateOffset(months=i + 1)
            prediction_points.append(
                PredictionPoint(
                    date=pred_date.strftime("%Y-%m"),
                    predicted=float(result.predicted[i]),
                    lower=float(result.lower_bound[i]),
                    upper=float(result.upper_bound[i]),
                )
            )

        # Get history if requested
        history = None
        if include_history:
            history = self._get_history(model, history_months, last_date)

        # Calculate data freshness
        data_freshness = self._calculate_data_freshness(training_end)

        # Build metadata
        metadata = PredictionMetadata(
            model_type=meta["model_type"],
            model_params=meta["params"],
            training_end=training_end,
            mape=meta["mape"],
            generated_at=datetime.now(),
            data_freshness=data_freshness,
        )

        # Build explanation from model components
        explanation = self._build_explanation(offense, result.components)

        # Determine location string
        location = state if state else "national"

        return PredictionResponse(
            offense=offense,
            location=location,
            predictions=prediction_points,
            history=history,
            metadata=metadata,
            explanation=explanation,
        )

    def _calculate_data_freshness(self, training_end: str) -> DataFreshness:
        """Calculate data freshness information.

        Args:
            training_end: Training end date in YYYY-MM format

        Returns:
            DataFreshness schema with recency info
        """
        if training_end == "unknown":
            return DataFreshness(
                training_end="unknown",
                months_since_training=0,
                note="Training end date unknown",
            )

        try:
            training_date = pd.to_datetime(training_end)
            now = pd.Timestamp.now()
            months_diff = (now.year - training_date.year) * 12 + (
                now.month - training_date.month
            )
            return DataFreshness(
                training_end=training_end,
                months_since_training=max(0, months_diff),
                note="FBI UCR data has ~2 month reporting lag",
            )
        except Exception:
            return DataFreshness(
                training_end=training_end,
                months_since_training=0,
                note="Could not calculate data freshness",
            )

    def _build_explanation(
        self, offense: str, components: Optional[ComponentBreakdown]
    ) -> Explanation:
        """Build human-readable explanation from model components.

        Args:
            offense: Crime type being predicted
            components: Component breakdown from model

        Returns:
            Explanation schema with narrative
        """
        offense_display = offense.replace("-", " ").title()

        # Handle case where components are not available
        if components is None:
            return Explanation(
                method="unknown",
                components=ExplanationComponents(
                    trend=TrendComponentSchema(direction="unknown", change_pct=0.0),
                    yearly_seasonality=None,
                ),
                narrative=f"{offense_display} prediction based on historical patterns.",
            )

        # Build trend component schema
        trend_schema = TrendComponentSchema(
            direction=components.trend.direction,
            change_pct=components.trend.change_pct,
        )

        # Build seasonality component schema if present
        seasonality_schema = None
        if components.yearly_seasonality:
            seasonality_schema = SeasonalityComponentSchema(
                current_effect_pct=components.yearly_seasonality.current_effect_pct,
                peak_months=components.yearly_seasonality.peak_months,
                trough_months=components.yearly_seasonality.trough_months,
            )

        # Generate narrative
        narrative = self._generate_narrative(
            offense_display, components.trend, components.yearly_seasonality
        )

        return Explanation(
            method=components.method,
            components=ExplanationComponents(
                trend=trend_schema,
                yearly_seasonality=seasonality_schema,
            ),
            narrative=narrative,
        )

    def _generate_narrative(
        self,
        offense_display: str,
        trend,
        seasonality,
    ) -> str:
        """Generate template-based narrative explanation.

        Args:
            offense_display: Human-readable offense name
            trend: TrendComponent from model
            seasonality: SeasonalityComponent from model (or None)

        Returns:
            Narrative string
        """
        has_seasonality = seasonality is not None
        trend_direction = trend.direction
        trend_pct = trend.change_pct

        # Determine change verb
        if trend_direction == "increasing":
            change_verb = "increase"
        elif trend_direction == "decreasing":
            change_verb = "decrease"
        else:
            change_verb = "change"

        # Select template
        template_key = f"{trend_direction}_{'with' if has_seasonality else 'no'}_seasonality"
        template = NARRATIVE_TEMPLATES.get(
            template_key, NARRATIVE_TEMPLATES["stable_no_seasonality"]
        )

        # Build seasonality clause if applicable
        seasonality_clause = ""
        if has_seasonality:
            effect_pct = seasonality.current_effect_pct
            if effect_pct > 0:
                seasonality_clause = (
                    f"with seasonal factors adding {effect_pct:.1f}% "
                    f"(crime typically peaks in {', '.join(seasonality.peak_months)})"
                )
            elif effect_pct < 0:
                seasonality_clause = (
                    f"partially offset by typical seasonal decline of {abs(effect_pct):.1f}% "
                    f"(crime typically lowest in {', '.join(seasonality.trough_months)})"
                )
            else:
                seasonality_clause = "with minimal seasonal effect"

        return template.format(
            offense_display=offense_display,
            change_verb=change_verb,
            change_pct=abs(trend_pct),
            trend_pct=trend_pct,
            seasonality_clause=seasonality_clause,
        )

    def _get_history(
        self,
        model,
        months: int,
        end_date: pd.Timestamp,
    ) -> Optional[list[HistoryPoint]]:
        """Extract historical data from model if available."""
        try:
            # Try to get training data from different model types
            if hasattr(model, "fitted") and model.fitted is not None:
                # ARIMA/SARIMA models
                if hasattr(model.fitted, "data"):
                    data = model.fitted.data.endog
                    index = model.fitted.data.dates

                    if index is not None:
                        history = []
                        for i in range(max(0, len(data) - months), len(data)):
                            history.append(
                                HistoryPoint(
                                    date=index[i].strftime("%Y-%m"),
                                    actual=float(data[i]),
                                )
                            )
                        return history

            elif hasattr(model, "model") and hasattr(model.model, "history"):
                # Prophet models
                history_df = model.model.history
                if history_df is not None and len(history_df) > 0:
                    history = []
                    start_idx = max(0, len(history_df) - months)
                    for _, row in history_df.iloc[start_idx:].iterrows():
                        history.append(
                            HistoryPoint(
                                date=row["ds"].strftime("%Y-%m"),
                                actual=float(row["y"]),
                            )
                        )
                    return history

        except Exception as e:
            logger.warning(f"Could not extract history: {e}")

        return None
