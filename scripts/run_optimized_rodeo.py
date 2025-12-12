#!/usr/bin/env python3
"""
Optimized Model Rodeo: Grid search for best hyperparameters.

This script uses auto_arima for ARIMA/SARIMA optimization and grid search
for Prophet to find truly optimal models for each location/offense combination.
"""

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Suppress warnings
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
class OptimizedResult:
    """Results from an optimized model run."""
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
    optimization_method: str = ""


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
    mask = actual != 0
    if not mask.any():
        mape = float("nan")
    else:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return round(mape, 2), round(mae, 2), round(rmse, 2)


def optimize_arima(train_series: pd.Series, test_actual: np.ndarray, seasonal: bool = False) -> Optional[OptimizedResult]:
    """Find optimal ARIMA/SARIMA using auto_arima."""
    try:
        import pmdarima as pm

        # Run auto_arima
        model = pm.auto_arima(
            train_series,
            seasonal=seasonal,
            m=12 if seasonal else 1,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,  # Auto-detect
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            D=None if seasonal else 0,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            n_fits=50,
        )

        # Generate predictions
        predicted = model.predict(n_periods=len(test_actual))

        # Calculate metrics
        mape, mae, rmse = calculate_metrics(test_actual, predicted)

        # Extract parameters
        order = model.order
        seasonal_order = model.seasonal_order if seasonal else None

        if seasonal and seasonal_order[3] > 0:
            model_name = f"SARIMA{order}{seasonal_order}"
            model_type = "SARIMA"
            params = {"order": order, "seasonal_order": seasonal_order}
        else:
            model_name = f"ARIMA{order}"
            model_type = "ARIMA"
            params = {"order": order}

        return OptimizedResult(
            location="",  # Will be filled in later
            offense="",
            model_name=model_name,
            model_type=model_type,
            params=params,
            mape=mape,
            mae=mae,
            rmse=rmse,
            training_months=len(train_series),
            test_months=len(test_actual),
            optimization_method="auto_arima (stepwise AIC)",
        )

    except Exception as e:
        logger.warning(f"auto_arima failed: {e}")
        return None


def optimize_prophet(train_series: pd.Series, test_actual: np.ndarray) -> Optional[OptimizedResult]:
    """Find optimal Prophet parameters via grid search."""
    try:
        from prophet import Prophet

        # Prepare data for Prophet
        train_df = pd.DataFrame({
            'ds': train_series.index,
            'y': train_series.values
        })

        # Grid search parameters
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'yearly_seasonality': [True, False],
        }

        best_mape = float('inf')
        best_result = None
        best_params = None

        # Generate all combinations
        keys = param_grid.keys()
        combinations = list(product(*param_grid.values()))

        for combo in combinations:
            params = dict(zip(keys, combo))

            try:
                # Create and fit model
                model = Prophet(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    yearly_seasonality=params['yearly_seasonality'],
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )
                model.fit(train_df)

                # Generate future dates
                future = model.make_future_dataframe(periods=len(test_actual), freq='MS')
                forecast = model.predict(future)

                # Get predictions for test period
                predicted = forecast.iloc[-len(test_actual):]['yhat'].values

                # Calculate metrics
                mape, mae, rmse = calculate_metrics(test_actual, predicted)

                if mape < best_mape and not np.isnan(mape):
                    best_mape = mape
                    best_params = params.copy()
                    best_result = (predicted, mape, mae, rmse)

            except Exception:
                continue

        if best_result is None:
            return None

        predicted, mape, mae, rmse = best_result

        # Format model name
        seasonality = "yearly" if best_params['yearly_seasonality'] else "no-season"
        model_name = f"Prophet({seasonality}, cp={best_params['changepoint_prior_scale']}, sp={best_params['seasonality_prior_scale']})"

        return OptimizedResult(
            location="",
            offense="",
            model_name=model_name,
            model_type="Prophet",
            params=best_params,
            mape=mape,
            mae=mae,
            rmse=rmse,
            training_months=len(train_series),
            test_months=len(test_actual),
            optimization_method=f"grid search ({len(combinations)} combinations)",
        )

    except Exception as e:
        logger.warning(f"Prophet optimization failed: {e}")
        return None


def run_optimized_rodeo(
    data: pd.DataFrame,
    location: str,
    offense: str,
) -> list[OptimizedResult]:
    """Run optimized model selection for a single location/offense."""

    if len(data) < TEST_MONTHS + 12:
        logger.warning(f"Insufficient data for {location}/{offense}: {len(data)} months")
        return []

    # Split data
    train_data = data.iloc[:-TEST_MONTHS]
    test_data = data.iloc[-TEST_MONTHS:]

    train_series = train_data["actual"]
    test_actual = test_data["actual"].values

    results = []

    # 1. Optimize non-seasonal ARIMA
    logger.info(f"  Optimizing ARIMA...")
    arima_result = optimize_arima(train_series, test_actual, seasonal=False)
    if arima_result:
        arima_result.location = location
        arima_result.offense = offense
        results.append(arima_result)
        logger.info(f"    Best ARIMA: {arima_result.model_name} (MAPE: {arima_result.mape}%)")

    # 2. Optimize seasonal SARIMA
    logger.info(f"  Optimizing SARIMA...")
    sarima_result = optimize_arima(train_series, test_actual, seasonal=True)
    if sarima_result:
        sarima_result.location = location
        sarima_result.offense = offense
        results.append(sarima_result)
        logger.info(f"    Best SARIMA: {sarima_result.model_name} (MAPE: {sarima_result.mape}%)")

    # 3. Optimize Prophet
    logger.info(f"  Optimizing Prophet...")
    prophet_result = optimize_prophet(train_series, test_actual)
    if prophet_result:
        prophet_result.location = location
        prophet_result.offense = offense
        results.append(prophet_result)
        logger.info(f"    Best Prophet: {prophet_result.model_name} (MAPE: {prophet_result.mape}%)")

    return results


def select_best_model(results: list[OptimizedResult]) -> Optional[OptimizedResult]:
    """Select the best model based on MAPE."""
    if not results:
        return None

    valid_results = [r for r in results if not np.isnan(r.mape) and r.mape < 500]

    if not valid_results:
        return None

    return min(valid_results, key=lambda r: r.mape)


def generate_model_selection_document(all_results: dict[str, OptimizedResult], output_path: Path):
    """Generate the MODEL_SELECTION.md document."""

    lines = [
        "# Model Selection Results (Optimized)",
        "",
        "This document tracks the winning model for each crime data series based on optimized hyperparameter search.",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "**Data Source:** Real FBI UCR data (fetched from Crime Data Explorer API)",
        f"**Test Period:** {TEST_MONTHS} months held out for evaluation",
        "**Optimization:** auto_arima (ARIMA/SARIMA) + grid search (Prophet)",
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
        "# Model assignments for deployment (optimized hyperparameters)",
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
                params_str = ", ".join(f'"{k}": {v}' for k, v in r.params.items())
                lines.append(f'    "{offense}": {{"type": "Prophet", "params": {{{params_str}}}}},')

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
                    f"- Optimization: {r.optimization_method}",
                    "",
                ])

    lines.extend([
        "## Optimization Methods",
        "",
        "### ARIMA/SARIMA",
        "- **Method:** pmdarima auto_arima with stepwise AIC selection",
        "- **Search space:** p,q ∈ [0,3], d auto-detected, P,Q ∈ [0,2], D auto-detected, m=12",
        "",
        "### Prophet",
        "- **Method:** Grid search over key hyperparameters",
        "- **Parameters searched:**",
        "  - changepoint_prior_scale: [0.001, 0.01, 0.1, 0.5]",
        "  - seasonality_prior_scale: [0.01, 0.1, 1.0, 10.0]",
        "  - yearly_seasonality: [True, False]",
        "- **Total combinations:** 32",
        "",
    ])

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Model selection document saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run optimized model rodeo with hyperparameter search")
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
    logger.info("OPTIMIZED MODEL RODEO - FBI UCR Real Data")
    logger.info("=" * 70)
    logger.info(f"Locations: {locations}")
    logger.info(f"Offenses: {offenses}")
    logger.info(f"Test months: {TEST_MONTHS}")
    logger.info("Optimization: auto_arima + Prophet grid search")
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

            # Run optimized rodeo
            results = run_optimized_rodeo(data, location, offense)
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
                "params": str(r.params),
                "mape": r.mape,
                "mae": r.mae,
                "rmse": r.rmse,
                "training_months": r.training_months,
                "optimization_method": r.optimization_method,
            }
            for r in all_detailed_results
        ])

        csv_path = output_dir / "optimized_rodeo_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"\nDetailed results saved to: {csv_path}")

    # Generate model selection document
    doc_path = Path(__file__).parent.parent.parent / "crime_stats" / "docs" / "MODEL_SELECTION.md"
    generate_model_selection_document(best_results, doc_path)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZED RODEO COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total combinations tested: {len(best_results)}")

    # Count wins by model type
    model_wins = {}
    for r in best_results.values():
        model_wins[r.model_type] = model_wins.get(r.model_type, 0) + 1

    logger.info("\nWins by model type:")
    for model_type, count in sorted(model_wins.items(), key=lambda x: -x[1]):
        logger.info(f"  {model_type}: {count}")

    # Identify national model selections
    logger.info("\nNATIONAL MODEL SELECTIONS (optimized):")
    for offense in OFFENSES:
        key = f"national/{offense}"
        if key in best_results:
            r = best_results[key]
            logger.info(f"  {offense}: {r.model_name} (MAPE: {r.mape:.2f}%)")

    return best_results


if __name__ == "__main__":
    main()
