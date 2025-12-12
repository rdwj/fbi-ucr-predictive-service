"""API module for FastAPI routes and schemas."""

from .routes import init_services, router
from .schemas import (
    ErrorResponse,
    HealthResponse,
    HistoryResponse,
    ModelInfo,
    ModelsResponse,
    PredictionRequest,
    PredictionResponse,
)

__all__ = [
    "router",
    "init_services",
    "PredictionRequest",
    "PredictionResponse",
    "HealthResponse",
    "ModelsResponse",
    "ModelInfo",
    "HistoryResponse",
    "ErrorResponse",
]
