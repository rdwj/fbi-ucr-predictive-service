"""FBI UCR Crime Prediction Service - FastAPI Application."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import init_services, router
from .inference import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model manager
model_manager: ModelManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load models on startup."""
    global model_manager

    # Get models directory from environment or default
    models_dir = os.getenv("MODELS_DIR", "models")
    logger.info(f"Loading models from: {models_dir}")

    # Initialize model manager and load models
    model_manager = ModelManager(models_dir=models_dir)

    try:
        loaded = model_manager.load_all()
        logger.info(f"Successfully loaded {loaded} models")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

    # Initialize API routes with services
    init_services(model_manager, datetime.now())

    yield

    # Cleanup on shutdown
    logger.info("Shutting down FBI UCR service")


# Create FastAPI app
app = FastAPI(
    title="FBI UCR Crime Prediction API",
    description="Real-time crime prediction service using FBI Uniform Crime Reporting data",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix="/api/v1", tags=["predictions"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "FBI UCR Crime Prediction API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


def main():
    """Run the application with uvicorn."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(
        "fbi_ucr.main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
