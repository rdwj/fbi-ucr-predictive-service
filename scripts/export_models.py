"""
Export trained models for deployment.

This script trains and exports models using the fbi-ucr module classes
so they can be loaded correctly by the inference service.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fbi_ucr.models import ARIMAPredictor, ProphetPredictor, SARIMAPredictor

warnings.filterwarnings("ignore")

# Model configuration based on rodeo results
MODEL_CONFIG = {
    "violent-crime": {
        "class": ARIMAPredictor,
        "params": {"order": (1, 1, 1)},
        "mape": 9.0,
    },
    "property-crime": {
        "class": ARIMAPredictor,
        "params": {"order": (1, 1, 1)},
        "mape": 7.9,
    },
    "homicide": {
        "class": ProphetPredictor,
        "params": {"yearly_seasonality": True},
        "mape": 8.2,
    },
    "burglary": {
        "class": ARIMAPredictor,
        "params": {"order": (1, 1, 1)},
        "mape": 7.7,
    },
    "motor-vehicle-theft": {
        "class": SARIMAPredictor,
        "params": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
        "mape": 5.4,
    },
}

# Sample data for demonstration
SAMPLE_DATA = {
    "violent-crime": {
        "dates": pd.date_range("2022-01", periods=36, freq="MS"),
        "values": [
            105000, 98000, 102000, 108000, 112000, 118000,
            115000, 110000, 105000, 98000, 95000, 92000,
            100000, 95000, 98000, 105000, 110000, 115000,
            112000, 108000, 102000, 95000, 92000, 90000,
            97000, 93000, 96000, 102000, 107000, 112000,
            108000, 104000, 98000, 92000, 88000, 85000,
        ],
    },
    "property-crime": {
        "dates": pd.date_range("2022-01", periods=36, freq="MS"),
        "values": [
            520000, 490000, 510000, 540000, 560000, 580000,
            570000, 550000, 520000, 490000, 470000, 460000,
            500000, 480000, 495000, 525000, 550000, 570000,
            560000, 540000, 510000, 480000, 460000, 450000,
            485000, 465000, 480000, 510000, 535000, 555000,
            545000, 525000, 495000, 465000, 445000, 430000,
        ],
    },
    "homicide": {
        "dates": pd.date_range("2022-01", periods=36, freq="MS"),
        "values": [
            1600, 1500, 1550, 1650, 1700, 1750,
            1720, 1680, 1620, 1550, 1500, 1480,
            1550, 1480, 1520, 1600, 1650, 1700,
            1670, 1630, 1580, 1510, 1470, 1450,
            1500, 1450, 1480, 1550, 1600, 1640,
            1610, 1570, 1520, 1460, 1420, 1400,
        ],
    },
    "burglary": {
        "dates": pd.date_range("2022-01", periods=36, freq="MS"),
        "values": [
            68000, 64000, 66000, 70000, 73000, 76000,
            74000, 71000, 67000, 63000, 61000, 59000,
            65000, 62000, 64000, 68000, 72000, 75000,
            73000, 70000, 66000, 62000, 60000, 58000,
            63000, 60000, 62000, 66000, 70000, 73000,
            71000, 68000, 64000, 60000, 58000, 55000,
        ],
    },
    "motor-vehicle-theft": {
        "dates": pd.date_range("2022-01", periods=36, freq="MS"),
        "values": [
            78000, 73000, 76000, 81000, 85000, 89000,
            87000, 83000, 79000, 74000, 71000, 69000,
            76000, 72000, 75000, 80000, 84000, 88000,
            86000, 82000, 77000, 73000, 70000, 68000,
            74000, 70000, 73000, 78000, 82000, 86000,
            84000, 80000, 75000, 71000, 68000, 65000,
        ],
    },
}


def get_training_data(offense: str) -> pd.Series:
    """Get training data."""
    data = SAMPLE_DATA[offense]
    ts = pd.Series(data["values"], index=pd.DatetimeIndex(data["dates"]))
    return ts


def export_model(offense: str, output_dir: Path) -> dict:
    """Train and export a model for an offense type."""
    print(f"\n{'='*60}")
    print(f"Exporting {offense}")
    print(f"{'='*60}")

    config = MODEL_CONFIG[offense]
    model_class = config["class"]
    params = config["params"]

    # Get training data
    ts = get_training_data(offense)
    training_end = ts.index[-1].strftime("%Y-%m")
    print(f"  Training data: {ts.index[0].strftime('%Y-%m')} to {training_end} ({len(ts)} months)")

    # Create and train model
    print(f"  Training {model_class.name} model...")
    model = model_class(**params)
    model.fit(ts)

    # Export based on model type
    if model_class == ProphetPredictor:
        # Export Prophet as JSON
        from prophet.serialize import model_to_json

        model_path = output_dir / f"{offense}.json"
        with open(model_path, "w") as f:
            f.write(model_to_json(model.model))
        print(f"  Exported Prophet model to {model_path}")
    else:
        # Export ARIMA/SARIMA as joblib
        model_path = output_dir / f"{offense}.joblib"
        joblib.dump(model, model_path)
        print(f"  Exported {model_class.name} model to {model_path}")

    # Export metadata
    meta = {
        "offense": offense,
        "model_type": model_class.name,
        "params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
        "mape": config["mape"],
        "training_end": training_end,
        "exported_at": datetime.now().isoformat(),
    }

    meta_path = output_dir / f"{offense}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Exported metadata to {meta_path}")

    return meta


def main():
    """Export all models."""
    output_dir = Path(__file__).parent.parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Export all models
    results = []
    for offense in MODEL_CONFIG:
        try:
            meta = export_model(offense, output_dir)
            results.append(meta)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Models exported: {len(results)}/{len(MODEL_CONFIG)}")
    print(f"Output directory: {output_dir}")
    print("\nExported models:")
    for meta in results:
        print(f"  - {meta['offense']}: {meta['model_type']} (MAPE: {meta['mape']}%)")


if __name__ == "__main__":
    main()
