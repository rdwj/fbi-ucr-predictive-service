"""Model loading and management."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

# Note: Model classes (ARIMAPredictor, ProphetPredictor, SARIMAPredictor) are loaded
# via joblib and don't need to be imported here explicitly

logger = logging.getLogger(__name__)


# Model configuration: maps offense to model type and parameters
# Updated based on optimized rodeo with grid search (2025-12-01)
# See crime_stats/docs/MODEL_SELECTION.md for detailed rodeo results
MODEL_CONFIG = {
    "violent-crime": {
        "type": "Prophet",
        "params": {
            "yearly_seasonality": True,
            "changepoint_prior_scale": 0.5,
            "seasonality_prior_scale": 0.1,
        },
        "mape": 2.1,
    },
    "property-crime": {
        "type": "Prophet",
        "params": {
            "yearly_seasonality": False,
            "changepoint_prior_scale": 0.5,
            "seasonality_prior_scale": 0.01,
        },
        "mape": 1.8,
    },
    "homicide": {
        "type": "Prophet",
        "params": {
            "yearly_seasonality": True,
            "changepoint_prior_scale": 0.5,
            "seasonality_prior_scale": 0.01,
        },
        "mape": 1.9,
    },
    "burglary": {
        "type": "Prophet",
        "params": {
            "yearly_seasonality": True,
            "changepoint_prior_scale": 0.5,
            "seasonality_prior_scale": 0.1,
        },
        "mape": 1.3,
    },
    "motor-vehicle-theft": {
        "type": "ARIMA",
        "params": {"order": (0, 1, 1)},
        "mape": 4.0,
    },
}

VALID_OFFENSES = list(MODEL_CONFIG.keys())

# Supported states for state-level predictions
VALID_STATES = ["CA", "TX", "FL", "NY", "IL"]


def _make_model_key(offense: str, state: Optional[str] = None) -> str:
    """Create a unique key for a model (offense or offense+state combination)."""
    if state:
        return f"{offense}_{state}"
    return offense


class ModelManager:
    """Manages loading and access to prediction models."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models: dict = {}
        self.metadata: dict = {}
        self.load_time: Optional[datetime] = None

    def load_all(self) -> int:
        """Load all available models (national + state-level). Returns count of models loaded."""
        loaded = 0

        # Load national models
        for offense in VALID_OFFENSES:
            try:
                self._load_model(offense, state=None)
                loaded += 1
                logger.info(f"Loaded national model for {offense}")
            except Exception as e:
                logger.error(f"Failed to load national model for {offense}: {e}")

        # Load state-level models
        for state in VALID_STATES:
            for offense in VALID_OFFENSES:
                try:
                    self._load_model(offense, state=state)
                    loaded += 1
                    logger.info(f"Loaded model for {offense} ({state})")
                except Exception as e:
                    logger.warning(f"No model for {offense} ({state}): {e}")

        self.load_time = datetime.now()
        return loaded

    def _load_model(self, offense: str, state: Optional[str] = None) -> None:
        """Load a single model from disk."""
        config = MODEL_CONFIG.get(offense)
        if not config:
            raise ValueError(f"Unknown offense: {offense}")

        model_type = config["type"]
        model_key = _make_model_key(offense, state)

        # Build the filename based on whether it's state-level
        if state:
            base_name = f"{offense}_{state}"
        else:
            base_name = offense

        # Try different file extensions
        model_path = None
        for ext in [".joblib", ".json", ".pkl"]:
            path = self.models_dir / f"{base_name}{ext}"
            if path.exists():
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(f"No model file found for {model_key}")

        # Load model (all types are now saved as joblib with the wrapper class)
        model = joblib.load(model_path)

        # Load metadata if available
        meta_path = self.models_dir / f"{base_name}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {
                "training_end": "unknown",
                "mape": config["mape"],
            }

        self.models[model_key] = model
        self.metadata[model_key] = {
            "model_type": model_type,
            "params": config["params"],
            "mape": meta.get("mape", config["mape"]),
            "training_end": meta.get("training_end", "unknown"),
            "loaded_at": datetime.now(),
            "offense": offense,
            "location": state if state else "national",
        }

    def get_model(self, offense: str, state: Optional[str] = None):
        """Get a loaded model by offense type and optional state."""
        model_key = _make_model_key(offense, state)
        if model_key not in self.models:
            location = state if state else "national"
            raise KeyError(f"Model not loaded for offense: {offense} ({location})")
        return self.models[model_key]

    def get_metadata(self, offense: str, state: Optional[str] = None) -> dict:
        """Get metadata for a model."""
        model_key = _make_model_key(offense, state)
        if model_key not in self.metadata:
            location = state if state else "national"
            raise KeyError(f"No metadata for offense: {offense} ({location})")
        return self.metadata[model_key]

    def list_models(self, state: Optional[str] = None) -> list[dict]:
        """List loaded models with their metadata.

        Args:
            state: If provided, filter to only models for this state.
                   If None, return all models (national + all states).
        """
        result = []
        for model_key, meta in self.metadata.items():
            # Filter by state if requested
            if state is not None:
                if meta["location"] != state:
                    continue

            result.append({
                "offense": meta["offense"],
                "location": meta["location"],
                "model_type": meta["model_type"],
                "params": meta["params"],
                "mape": meta["mape"],
                "training_end": meta["training_end"],
                "loaded_at": meta["loaded_at"],
            })
        return result

    def is_loaded(self, offense: str, state: Optional[str] = None) -> bool:
        """Check if a model is loaded."""
        model_key = _make_model_key(offense, state)
        return model_key in self.models

    def get_available_states(self, offense: Optional[str] = None) -> list[str]:
        """Get list of states with available models.

        Args:
            offense: If provided, filter to states with models for this offense.
                     If None, return all states with any models.
        """
        states = set()
        for meta in self.metadata.values():
            if meta["location"] != "national":
                if offense is None or meta["offense"] == offense:
                    states.add(meta["location"])
        return sorted(states)

    @property
    def model_count(self) -> int:
        """Number of loaded models."""
        return len(self.models)
