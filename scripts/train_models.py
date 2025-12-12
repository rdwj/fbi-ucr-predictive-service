#!/usr/bin/env python3
"""Train crime prediction models using FBI Crime Data.

This script loads fetched FBI data, trains models based on the optimized rodeo results,
calculates MAPE on held-out test data, and exports models with metadata.

Model assignments (from optimized rodeo with grid search, 2025-12-01):
- violent-crime: Prophet (yearly_seasonality=True, changepoint_prior_scale=0.5, seasonality_prior_scale=0.1)
- property-crime: Prophet (yearly_seasonality=False, changepoint_prior_scale=0.5, seasonality_prior_scale=0.01)
- homicide: Prophet (yearly_seasonality=True, changepoint_prior_scale=0.5, seasonality_prior_scale=0.01)
- burglary: Prophet (yearly_seasonality=True, changepoint_prior_scale=0.5, seasonality_prior_scale=0.1)
- motor-vehicle-theft: ARIMA(0,1,1)
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fbi_ucr.models import ARIMAPredictor, ProphetPredictor, SARIMAPredictor

# Suppress warnings
warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

# Test set size (last N months)
TEST_MONTHS = 6

# Model configurations based on optimized rodeo results (grid search, 2025-12-01)
# See crime_stats/docs/MODEL_SELECTION.md for detailed rodeo results
MODEL_CONFIG = {
    "violent-crime": {
        "model_class": ProphetPredictor,
        "params": {
            "yearly_seasonality": True,
            "changepoint_prior_scale": 0.5,
            "seasonality_prior_scale": 0.1,
        },
        "model_type": "Prophet",
    },
    "property-crime": {
        "model_class": ProphetPredictor,
        "params": {
            "yearly_seasonality": False,
            "changepoint_prior_scale": 0.5,
            "seasonality_prior_scale": 0.01,
        },
        "model_type": "Prophet",
    },
    "homicide": {
        "model_class": ProphetPredictor,
        "params": {
            "yearly_seasonality": True,
            "changepoint_prior_scale": 0.5,
            "seasonality_prior_scale": 0.01,
        },
        "model_type": "Prophet",
    },
    "burglary": {
        "model_class": ProphetPredictor,
        "params": {
            "yearly_seasonality": True,
            "changepoint_prior_scale": 0.5,
            "seasonality_prior_scale": 0.1,
        },
        "model_type": "Prophet",
    },
    "motor-vehicle-theft": {
        "model_class": ARIMAPredictor,
        "params": {"order": (0, 1, 1)},
        "model_type": "ARIMA",
    },
}

# Locations to process
LOCATIONS = ["national", "CA", "TX", "FL", "NY", "IL"]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(location: str, offense: str) -> Optional[pd.DataFrame]:
    """Load CSV data for a location and offense.

    Args:
        location: 'national' or state abbreviation (e.g., 'CA')
        offense: Offense type (e.g., 'violent-crime')

    Returns:
        DataFrame with date index and 'actual' column, or None if not found
    """
    if location == "national":
        filename = f"national_{offense}.csv"
    else:
        filename = f"{location}_{offense}.csv"

    filepath = DATA_DIR / filename

    if not filepath.exists():
        logger.warning(f"Data file not found: {filepath}")
        return None

    df = pd.read_csv(filepath)

    # Convert date to datetime index
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df.set_index("date", inplace=True)
    df = df.sort_index()

    return df


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        MAPE as percentage (e.g., 8.5 means 8.5%)
    """
    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        return float("nan")

    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return round(mape, 2)


def train_and_evaluate(
    data: pd.DataFrame,
    offense: str,
    location: str,
) -> Optional[dict]:
    """Train model and evaluate on held-out test set.

    Args:
        data: DataFrame with 'actual' column and datetime index
        offense: Offense type for model selection
        location: Location identifier for logging

    Returns:
        Dictionary with trained model, metrics, and metadata, or None if failed
    """
    config = MODEL_CONFIG.get(offense)
    if not config:
        logger.error(f"No model config for offense: {offense}")
        return None

    # Ensure we have enough data
    if len(data) < TEST_MONTHS + 12:  # Need at least 12 months for training
        logger.warning(f"Insufficient data for {location}/{offense}: {len(data)} months")
        return None

    # Split data
    train_data = data.iloc[:-TEST_MONTHS]
    test_data = data.iloc[-TEST_MONTHS:]

    train_series = train_data["actual"]
    test_actual = test_data["actual"].values

    # Create and train model
    model_class = config["model_class"]
    params = config["params"]

    logger.info(f"Training {config['model_type']} for {location}/{offense}...")

    try:
        model = model_class(**params)
        model.fit(train_series)

        # Generate predictions for test period
        result = model.predict(steps=TEST_MONTHS)
        test_predicted = result.predicted

        # Calculate MAPE
        mape = calculate_mape(test_actual, test_predicted)

        logger.info(f"  MAPE: {mape}%")

        return {
            "model": model,
            "model_type": config["model_type"],
            "params": params,
            "mape": mape,
            "training_start": train_data.index[0].strftime("%Y-%m"),
            "training_end": train_data.index[-1].strftime("%Y-%m"),
            "test_start": test_data.index[0].strftime("%Y-%m"),
            "test_end": test_data.index[-1].strftime("%Y-%m"),
            "test_actual": test_actual.tolist(),
            "test_predicted": test_predicted.tolist(),
        }

    except Exception as e:
        logger.error(f"Training failed for {location}/{offense}: {e}")
        return None


def export_model(model, model_type: str, offense: str, location: str) -> str:
    """Export trained model to file.

    Args:
        model: Trained model instance
        model_type: Type of model (ARIMA, SARIMA, Prophet)
        offense: Offense type
        location: Location identifier

    Returns:
        Filename of exported model
    """
    if location == "national":
        base_name = offense
    else:
        base_name = f"{offense}_{location}"

    if model_type == "Prophet":
        # Prophet models exported as joblib (same as ARIMA/SARIMA)
        # This preserves the entire ProphetPredictor wrapper
        filename = f"{base_name}.joblib"
        filepath = MODELS_DIR / filename
        joblib.dump(model, filepath)
    else:
        # ARIMA/SARIMA models exported as joblib
        filename = f"{base_name}.joblib"
        filepath = MODELS_DIR / filename

        # Export the entire model wrapper (not just the fitted result)
        # This preserves the predict() method that returns PredictionResult
        joblib.dump(model, filepath)

    logger.info(f"  Exported to {filename}")
    return filename


def export_metadata(
    result: dict,
    offense: str,
    location: str,
) -> str:
    """Export model metadata to JSON file.

    Args:
        result: Training result dictionary
        offense: Offense type
        location: Location identifier

    Returns:
        Filename of exported metadata
    """
    if location == "national":
        filename = f"{offense}_meta.json"
    else:
        filename = f"{offense}_{location}_meta.json"

    filepath = MODELS_DIR / filename

    metadata = {
        "offense": offense,
        "location": location,
        "model_type": result["model_type"],
        "params": result["params"],
        "mape": result["mape"],
        "training_start": result["training_start"],
        "training_end": result["training_end"],
        "test_start": result["test_start"],
        "test_end": result["test_end"],
        "exported_at": datetime.now().isoformat(),
    }

    # Convert tuple params to lists for JSON serialization
    if "order" in metadata["params"]:
        metadata["params"]["order"] = list(metadata["params"]["order"])
    if "seasonal_order" in metadata["params"]:
        metadata["params"]["seasonal_order"] = list(metadata["params"]["seasonal_order"])

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    return filename


def main():
    """Main function to train all models."""
    logger.info("Starting model training...")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Test set size: {TEST_MONTHS} months")

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Track results
    results_summary = []
    total_models = 0
    successful_models = 0
    failed_models = 0

    # Train models for each location and offense
    offenses = list(MODEL_CONFIG.keys())

    for location in LOCATIONS:
        for offense in offenses:
            total_models += 1

            # Load data
            data = load_data(location, offense)
            if data is None:
                failed_models += 1
                results_summary.append({
                    "location": location,
                    "offense": offense,
                    "status": "FAILED",
                    "reason": "No data file",
                })
                continue

            # Train and evaluate
            result = train_and_evaluate(data, offense, location)
            if result is None:
                failed_models += 1
                results_summary.append({
                    "location": location,
                    "offense": offense,
                    "status": "FAILED",
                    "reason": "Training failed",
                })
                continue

            # Export model
            export_model(
                result["model"],
                result["model_type"],
                offense,
                location,
            )

            # Export metadata
            export_metadata(result, offense, location)

            successful_models += 1
            results_summary.append({
                "location": location,
                "offense": offense,
                "status": "SUCCESS",
                "model_type": result["model_type"],
                "mape": result["mape"],
            })

    # Print summary
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total models: {total_models}")
    logger.info(f"Successful: {successful_models}")
    logger.info(f"Failed: {failed_models}")
    logger.info("")
    logger.info("MODEL PERFORMANCE (MAPE on held-out test set):")
    logger.info("-" * 70)
    logger.info(f"{'Location':<12} {'Offense':<25} {'Model':<10} {'MAPE %':<10}")
    logger.info("-" * 70)

    for r in sorted(results_summary, key=lambda x: (x["location"], x["offense"])):
        if r["status"] == "SUCCESS":
            logger.info(
                f"{r['location']:<12} {r['offense']:<25} {r['model_type']:<10} {r['mape']:<10}"
            )
        else:
            logger.info(
                f"{r['location']:<12} {r['offense']:<25} {'FAILED':<10} {r['reason']}"
            )

    logger.info("-" * 70)

    # List exported files
    model_files = list(MODELS_DIR.glob("*.joblib")) + list(MODELS_DIR.glob("*.json"))
    logger.info(f"\nExported {len(model_files)} files to {MODELS_DIR}")

    return successful_models, failed_models


if __name__ == "__main__":
    successful, failed = main()
    sys.exit(0 if failed == 0 else 1)
