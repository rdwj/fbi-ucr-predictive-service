#!/usr/bin/env python3
"""
Model Rodeo: Compare ARIMA, SARIMA, and Prophet on real FBI crime data.

This script trains multiple model types on each location/offense combination
and selects the best performer based on MAPE on held-out test data.

Note: Bi-LSTM and XGBoost are excluded due to limited training data (< 60 months).
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fbi_ucr.models import ARIMAPredictor, ProphetPredictor, SARIMAPredictor

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Locations and offenses to test
LOCATIONS = ["national", "CA", "TX", "FL", "NY", "IL"]
OFFENSES = ["violent-crime", "property-crime", "homicide", "burglary", "motor-vehicle-theft"]

# Test set size
TEST_MONTHS = 6

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RodeoResult:
    """Results from a model rodeo run."""
    location: str
    offense: str
    model_name: str
    model_type: str
    params: dict
    mape: float
    mae: float
    rmse: float
    training_months: int
    test_months: int


def load_data(location: str, offense: str) -> Optional[pd.DataFrame]:
    """Load CSV data for a location and offense."""
    if location == "national":
        filename = f"national_{offense}.csv"
    else:
        filename = f"{location}_{offense}.csv"

    filepath = DATA_DIR / filename

    if not filepath.exists():
        logger.warning(f"Data file not found: {filepath}")
        return None

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df.set_index("date", inplace=True)
    df = df.sort_index()

    return df


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    """Calculate MAPE, MAE, and RMSE."""
    # Avoid division by zero for MAPE
    mask = actual != 0
    if not mask.any():
        mape = float("nan")
    else:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return round(mape, 2), round(mae, 2), round(rmse, 2)


def run_model_rodeo(
    data: pd.DataFrame,
    location: str,
    offense: str,
) -> list[RodeoResult]:
    """Run all models on a single location/offense combination."""

    # Ensure we have enough data
    if len(data) < TEST_MONTHS + 12:
        logger.warning(f"Insufficient data for {location}/{offense}: {len(data)} months")
        return []

    # Split data
    train_data = data.iloc[:-TEST_MONTHS]
    test_data = data.iloc[-TEST_MONTHS:]

    train_series = train_data["actual"]
    test_actual = test_data["actual"].values

    results = []

    # Define models to test
    models = [
        ("ARIMA(1,1,1)", "ARIMA", ARIMAPredictor, {"order": (1, 1, 1)}),
        ("ARIMA(2,1,2)", "ARIMA", ARIMAPredictor, {"order": (2, 1, 2)}),
        ("SARIMA(1,1,1)(1,1,1,12)", "SARIMA", SARIMAPredictor, {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)}),
        ("SARIMA(1,1,1)(0,1,1,12)", "SARIMA", SARIMAPredictor, {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 12)}),
        ("Prophet", "Prophet", ProphetPredictor, {"yearly_seasonality": True}),
        ("Prophet-no-season", "Prophet", ProphetPredictor, {"yearly_seasonality": False}),
    ]

    for model_name, model_type, model_class, params in models:
        try:
            # Create and train model
            model = model_class(**params)
            model.fit(train_series)

            # Predict
            pred_result = model.predict(steps=TEST_MONTHS)
            test_predicted = pred_result.predicted

            # Calculate metrics
            mape, mae, rmse = calculate_metrics(test_actual, test_predicted)

            results.append(RodeoResult(
                location=location,
                offense=offense,
                model_name=model_name,
                model_type=model_type,
                params=params,
                mape=mape,
                mae=mae,
                rmse=rmse,
                training_months=len(train_data),
                test_months=TEST_MONTHS,
            ))

            logger.info(f"  {model_name}: MAPE={mape:.2f}%, MAE={mae:,.0f}")

        except Exception as e:
            logger.warning(f"  {model_name}: FAILED - {e}")

    return results


def select_best_model(results: list[RodeoResult]) -> Optional[RodeoResult]:
    """Select the best model based on MAPE."""
    if not results:
        return None

    # Filter out NaN MAPE results
    valid_results = [r for r in results if not np.isnan(r.mape) and r.mape < 500]

    if not valid_results:
        return None

    # Sort by MAPE (lower is better)
    return min(valid_results, key=lambda r: r.mape)


def generate_model_selection_document(all_results: dict[str, RodeoResult], output_path: Path):
    """Generate the MODEL_SELECTION.md document."""

    lines = [
        "# Model Selection Results",
        "",
        "This document tracks the winning model for each crime data series based on the model rodeo results.",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Data Source:** Real FBI UCR data (fetched from Crime Data Explorer API)",
        f"**Test Period:** {TEST_MONTHS} months held out for evaluation",
        "",
        "## Summary Table - National",
        "",
        "| Offense | Winning Model | MAPE | MAE | Training Period |",
        "|---------|---------------|------|-----|-----------------|",
    ]

    # National summary
    for offense in OFFENSES:
        key = f"national/{offense}"
        if key in all_results:
            r = all_results[key]
            lines.append(f"| {offense} | {r.model_name} | {r.mape:.1f}% | {r.mae:,.0f} | {r.training_months} months |")
        else:
            lines.append(f"| {offense} | N/A | - | - | - |")

    lines.extend([
        "",
        "## Summary Table - By State",
        "",
        "| State | Offense | Winning Model | MAPE | MAE |",
        "|-------|---------|---------------|------|-----|",
    ])

    # State summaries
    for state in ["CA", "TX", "FL", "NY", "IL"]:
        for offense in OFFENSES:
            key = f"{state}/{offense}"
            if key in all_results:
                r = all_results[key]
                lines.append(f"| {state} | {offense} | {r.model_name} | {r.mape:.1f}% | {r.mae:,.0f} |")

    # Key findings
    lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    # Analyze model type distribution
    model_counts = {}
    for r in all_results.values():
        model_counts[r.model_type] = model_counts.get(r.model_type, 0) + 1

    total = sum(model_counts.values())
    for model_type, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        lines.append(f"- **{model_type}** wins {count}/{total} datasets ({pct:.0f}%)")

    # Identify problem cases (high MAPE)
    problem_cases = [(k, r) for k, r in all_results.items() if r.mape > 25]
    if problem_cases:
        lines.extend([
            "",
            "## Data Quality Concerns",
            "",
            "The following location/offense combinations have high MAPE (>25%), suggesting data quality issues:",
            "",
        ])
        for key, r in sorted(problem_cases, key=lambda x: -x[1].mape):
            lines.append(f"- **{key}**: MAPE={r.mape:.1f}% - may have incomplete or anomalous data")

    # Production config
    lines.extend([
        "",
        "## Production Model Configuration",
        "",
        "```python",
        "# Model assignments for deployment",
        "MODEL_CONFIG = {",
    ])

    # Generate config for national models
    for offense in OFFENSES:
        key = f"national/{offense}"
        if key in all_results:
            r = all_results[key]
            if r.model_type == "ARIMA":
                order = r.params.get("order", (1, 1, 1))
                lines.append(f'    "{offense}": {{"type": "ARIMA", "params": {{"order": {order}}}}},')
            elif r.model_type == "SARIMA":
                order = r.params.get("order", (1, 1, 1))
                seasonal = r.params.get("seasonal_order", (1, 1, 1, 12))
                lines.append(f'    "{offense}": {{"type": "SARIMA", "params": {{"order": {order}, "seasonal_order": {seasonal}}}}},')
            elif r.model_type == "Prophet":
                yearly = r.params.get("yearly_seasonality", True)
                lines.append(f'    "{offense}": {{"type": "Prophet", "params": {{"yearly_seasonality": {yearly}}}}},')

    lines.extend([
        "}",
        "```",
        "",
        "## Detailed Results by Location",
        "",
    ])

    # Detailed results for each location
    for location in LOCATIONS:
        location_display = location if location == "national" else f"State: {location}"
        lines.extend([
            f"### {location_display}",
            "",
        ])

        for offense in OFFENSES:
            key = f"{location}/{offense}"
            if key in all_results:
                r = all_results[key]
                lines.extend([
                    f"**{offense}**",
                    f"- Winner: {r.model_name}",
                    f"- MAPE: {r.mape:.2f}%",
                    f"- MAE: {r.mae:,.0f}",
                    f"- RMSE: {r.rmse:,.0f}",
                    f"- Training: {r.training_months} months",
                    "",
                ])

    lines.extend([
        "## Rodeo Configuration",
        "",
        "- **Models tested:** ARIMA(1,1,1), ARIMA(2,1,2), SARIMA(1,1,1)(1,1,1,12), SARIMA(1,1,1)(0,1,1,12), Prophet, Prophet-no-season",
        f"- **Test period:** {TEST_MONTHS} months",
        "- **Primary metric:** MAPE (Mean Absolute Percentage Error)",
        "- **Data source:** FBI Crime Data Explorer API (real data)",
        "",
    ])

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Model selection document saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run model rodeo on real FBI crime data")
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Specific location to test (default: all locations)",
    )
    parser.add_argument(
        "--offense",
        type=str,
        default=None,
        help="Specific offense to test (default: all offenses)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Determine what to run
    locations = [args.location] if args.location else LOCATIONS
    offenses = [args.offense] if args.offense else OFFENSES

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MODEL RODEO - FBI UCR Real Data")
    logger.info("=" * 70)
    logger.info(f"Locations: {locations}")
    logger.info(f"Offenses: {offenses}")
    logger.info(f"Test months: {TEST_MONTHS}")
    logger.info("")

    # Store all results
    all_detailed_results = []
    best_results = {}

    # Run rodeo for each combination
    for location in locations:
        for offense in offenses:
            logger.info(f"\n{'='*60}")
            logger.info(f"RODEO: {location} / {offense}")
            logger.info(f"{'='*60}")

            # Load data
            data = load_data(location, offense)
            if data is None:
                continue

            logger.info(f"Data: {len(data)} months ({data.index[0].strftime('%Y-%m')} to {data.index[-1].strftime('%Y-%m')})")

            # Run rodeo
            results = run_model_rodeo(data, location, offense)
            all_detailed_results.extend(results)

            # Select best
            best = select_best_model(results)
            if best:
                key = f"{location}/{offense}"
                best_results[key] = best
                logger.info(f"\n  WINNER: {best.model_name} (MAPE: {best.mape:.2f}%)")

    # Save detailed results to CSV
    if all_detailed_results:
        results_df = pd.DataFrame([
            {
                "location": r.location,
                "offense": r.offense,
                "model_name": r.model_name,
                "model_type": r.model_type,
                "mape": r.mape,
                "mae": r.mae,
                "rmse": r.rmse,
                "training_months": r.training_months,
            }
            for r in all_detailed_results
        ])

        csv_path = output_dir / "rodeo_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"\nDetailed results saved to: {csv_path}")

    # Generate model selection document
    doc_path = Path(__file__).parent.parent / "docs" / "MODEL_SELECTION.md"
    generate_model_selection_document(best_results, doc_path)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("RODEO COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total combinations tested: {len(best_results)}")

    # Count wins by model type
    model_wins = {}
    for r in best_results.values():
        model_wins[r.model_type] = model_wins.get(r.model_type, 0) + 1

    logger.info("\nWins by model type:")
    for model_type, count in sorted(model_wins.items(), key=lambda x: -x[1]):
        logger.info(f"  {model_type}: {count}")

    # Identify national model selections for updating train_models.py
    logger.info("\nNATIONAL MODEL SELECTIONS (for train_models.py):")
    for offense in OFFENSES:
        key = f"national/{offense}"
        if key in best_results:
            r = best_results[key]
            logger.info(f"  {offense}: {r.model_name} (MAPE: {r.mape:.2f}%)")

    return best_results


if __name__ == "__main__":
    main()
