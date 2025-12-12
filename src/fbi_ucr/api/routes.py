"""API routes for the FBI UCR inference service."""

import logging
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from ..data import FBICrimeDataAPI, get_fbi_api
from ..inference import VALID_OFFENSES, VALID_STATES, ModelManager
from ..inference.predictor import PredictionService
from .schemas import (
    ErrorResponse,
    HealthResponse,
    HistoryPoint,
    HistoryResponse,
    ModelInfo,
    ModelsResponse,
    PredictionRequest,
    PredictionResponse,
    VALID_STATES as SCHEMA_VALID_STATES,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances (set during app startup)
_model_manager: ModelManager | None = None
_prediction_service: PredictionService | None = None
_start_time: datetime | None = None


def init_services(model_manager: ModelManager, start_time: datetime):
    """Initialize the router with service instances."""
    global _model_manager, _prediction_service, _start_time
    _model_manager = model_manager
    _prediction_service = PredictionService(model_manager)
    _start_time = start_time


def get_model_manager() -> ModelManager:
    """Dependency to get model manager."""
    if _model_manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _model_manager


def get_prediction_service() -> PredictionService:
    """Dependency to get prediction service."""
    if _prediction_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _prediction_service


@router.get("/health", response_model=HealthResponse)
async def health_check(
    model_manager: Annotated[ModelManager, Depends(get_model_manager)],
) -> HealthResponse:
    """Check service health and model status."""
    uptime = (datetime.now() - _start_time).total_seconds() if _start_time else 0

    return HealthResponse(
        status="healthy" if model_manager.model_count > 0 else "degraded",
        models_loaded=model_manager.model_count,
        uptime_seconds=uptime,
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    model_manager: Annotated[ModelManager, Depends(get_model_manager)],
    state: Annotated[
        Optional[str],
        Query(
            description="Filter by state code (CA, TX, FL, NY, IL). "
            "If omitted, returns all models (national + state)."
        ),
    ] = None,
) -> ModelsResponse:
    """List all available prediction models.

    Optionally filter by state to see only models available for a specific location.
    """
    # Validate state if provided
    if state is not None:
        state = state.upper()
        if state not in VALID_STATES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid state: {state}. Valid options: {VALID_STATES}",
            )

    models = []
    for model_info in model_manager.list_models(state=state):
        models.append(
            ModelInfo(
                offense=model_info["offense"],
                location=model_info["location"],
                model_type=model_info["model_type"],
                model_params=model_info["params"],
                mape=model_info["mape"],
                training_end=model_info["training_end"],
                last_loaded=model_info["loaded_at"],
            )
        )

    return ModelsResponse(models=models, total=len(models))


@router.post(
    "/predict/{offense}",
    response_model=PredictionResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Offense not found"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict(
    offense: Annotated[str, Path(description="Crime type to predict")],
    request: PredictionRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
    state: Annotated[
        Optional[str],
        Query(
            description="State code for state-level prediction (CA, TX, FL, NY, IL). "
            "If omitted, returns national-level prediction."
        ),
    ] = None,
) -> PredictionResponse:
    """Generate crime predictions for the specified offense type.

    Supports both national-level (default) and state-level predictions.
    For state-level, pass the state query parameter (e.g., ?state=CA).
    """
    if offense not in VALID_OFFENSES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown offense: {offense}. Valid options: {list(VALID_OFFENSES)}",
        )

    # Validate state if provided
    if state is not None:
        state = state.upper()
        if state not in VALID_STATES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid state: {state}. Valid options: {VALID_STATES}",
            )

    try:
        return service.predict(
            offense=offense,
            steps=request.steps,
            include_history=request.include_history,
            history_months=request.history_months,
            state=state,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Prediction failed for {offense} (state={state})")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get(
    "/history/{offense}",
    response_model=HistoryResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Offense not found"},
        502: {"model": ErrorResponse, "description": "FBI API error"},
    },
)
async def get_history(
    offense: Annotated[str, Path(description="Crime type")],
    from_year: Annotated[
        int,
        Query(ge=2015, le=2030, description="Start year for historical data"),
    ] = 2020,
    to_year: Annotated[
        Optional[int],
        Query(ge=2015, le=2030, description="End year (default: current year)"),
    ] = None,
    state: Annotated[
        Optional[str],
        Query(
            description="State code for state-level history (CA, TX, FL, NY, IL). "
            "If omitted, returns national-level history."
        ),
    ] = None,
) -> HistoryResponse:
    """Get historical crime data from the FBI Crime Data Explorer API.

    Fetches real-time data from the FBI API for the specified date range.
    Supports both national-level (default) and state-level history.

    **Date Range:**
    - from_year: Start year (default: 2020)
    - to_year: End year (default: current year)

    **Example:** `/history/violent-crime?from_year=2020&to_year=2024&state=CA`
    """
    if offense not in VALID_OFFENSES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown offense: {offense}. Valid options: {list(VALID_OFFENSES)}",
        )

    # Validate state if provided
    if state is not None:
        state = state.upper()
        if state not in VALID_STATES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid state: {state}. Valid options: {VALID_STATES}",
            )

    # Default to current year if not specified
    if to_year is None:
        to_year = datetime.now().year

    # Validate year range
    if from_year > to_year:
        raise HTTPException(
            status_code=400,
            detail=f"from_year ({from_year}) must be <= to_year ({to_year})",
        )

    try:
        fbi_api = get_fbi_api()

        if state:
            result = await fbi_api.get_state_history(
                state=state,
                offense=offense,
                from_year=from_year,
                to_year=to_year,
            )
        else:
            result = await fbi_api.get_national_history(
                offense=offense,
                from_year=from_year,
                to_year=to_year,
            )

        # Convert to response format
        history_points = [
            HistoryPoint(date=dp.date, actual=dp.actual)
            for dp in result.data
        ]

        return HistoryResponse(
            offense=offense,
            location=result.location,
            data=history_points,
            from_date=result.from_date,
            to_date=result.to_date,
            total_months=result.total_months,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"FBI API error for {offense} (state={state})")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch data from FBI API: {str(e)}",
        )
