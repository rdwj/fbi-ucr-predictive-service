#!/usr/bin/env python3
"""Fetch FBI Crime Data from the FBI Crime Data Explorer API.

This script fetches national and state-level crime data for multiple offenses
and saves them as CSV files for model training.

API Documentation: https://crime-data-explorer.fr.cloud.gov/api
"""

import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

# Configuration
BASE_URL = "https://api.usa.gov/crime/fbi/cde"
API_KEY = os.environ.get("FBI_API_KEY")
if not API_KEY:
    raise ValueError(
        "FBI_API_KEY environment variable is required. "
        "Get a free key at https://api.data.gov/signup/"
    )

# Date range: January 2020 to October 2024
DATE_FROM = "01-2020"
DATE_TO = "10-2024"

# Offenses to fetch
OFFENSES = [
    "violent-crime",
    "property-crime",
    "homicide",
    "burglary",
    "motor-vehicle-theft",
]

# States to fetch (top 5 by population)
STATES = ["CA", "TX", "FL", "NY", "IL"]

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between API calls

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_national_data(offense: str) -> Optional[dict]:
    """Fetch national-level crime data for a specific offense.

    Args:
        offense: Crime offense type (e.g., 'violent-crime')

    Returns:
        Dictionary with offense data or None if request fails
    """
    url = f"{BASE_URL}/summarized/national/{offense}"
    params = {
        "from": DATE_FROM,
        "to": DATE_TO,
        "API_KEY": API_KEY,
    }

    logger.info(f"Fetching national data for {offense}...")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch national {offense}: {e}")
        return None


def fetch_state_data(state: str, offense: str) -> Optional[dict]:
    """Fetch state-level crime data for a specific offense.

    Args:
        state: State abbreviation (e.g., 'CA')
        offense: Crime offense type (e.g., 'violent-crime')

    Returns:
        Dictionary with offense data or None if request fails
    """
    url = f"{BASE_URL}/summarized/state/{state}/{offense}"
    params = {
        "from": DATE_FROM,
        "to": DATE_TO,
        "API_KEY": API_KEY,
    }

    logger.info(f"Fetching {state} data for {offense}...")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {state} {offense}: {e}")
        return None


def parse_api_response(data: dict, location: str) -> list[dict]:
    """Parse API response into a list of records.

    The API returns data in this format:
    {
        "offenses": {
            "rates": {"United States": {"01-2020": 29.01, ...}},
            "actuals": {"United States": {"01-2020": 95000, ...}}
        },
        "populations": {"United States": {"2020": 331000000, ...}}
    }

    Args:
        data: Raw API response
        location: Location key in the response (e.g., 'United States' or 'California')

    Returns:
        List of dictionaries with date, actual, rate columns
    """
    records = []

    if not data or "offenses" not in data:
        logger.warning(f"No offense data found for {location}")
        return records

    offenses = data.get("offenses", {})
    actuals = offenses.get("actuals", {}).get(location, {})
    rates = offenses.get("rates", {}).get(location, {})

    # Get all dates from actuals
    for date_str, actual in actuals.items():
        rate = rates.get(date_str)

        # Convert MM-YYYY to YYYY-MM for proper sorting
        if "-" in date_str:
            month, year = date_str.split("-")
            date_normalized = f"{year}-{month}"
        else:
            date_normalized = date_str

        records.append({
            "date": date_normalized,
            "actual": actual,
            "rate": rate,
        })

    # Sort by date
    records.sort(key=lambda x: x["date"])

    return records


def save_to_csv(records: list[dict], filename: str) -> None:
    """Save records to CSV file.

    Args:
        records: List of dictionaries with date, actual, rate
        filename: Output filename (without path)
    """
    filepath = DATA_DIR / filename

    if not records:
        logger.warning(f"No records to save for {filename}")
        return

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "actual", "rate"])
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Saved {len(records)} records to {filepath}")


def get_state_full_name(state_abbrev: str) -> str:
    """Convert state abbreviation to full name for API response parsing.

    Args:
        state_abbrev: State abbreviation (e.g., 'CA')

    Returns:
        Full state name (e.g., 'California')
    """
    state_names = {
        "CA": "California",
        "TX": "Texas",
        "FL": "Florida",
        "NY": "New York",
        "IL": "Illinois",
    }
    return state_names.get(state_abbrev, state_abbrev)


def main():
    """Main function to fetch all FBI crime data."""
    logger.info("Starting FBI data fetch...")
    logger.info(f"Date range: {DATE_FROM} to {DATE_TO}")
    logger.info(f"Offenses: {OFFENSES}")
    logger.info(f"States: {STATES}")

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_requests = len(OFFENSES) * (1 + len(STATES))  # national + states
    completed = 0
    failed = 0

    # Fetch national data for each offense
    for offense in OFFENSES:
        data = fetch_national_data(offense)
        if data:
            records = parse_api_response(data, "United States")
            save_to_csv(records, f"national_{offense}.csv")
            completed += 1
        else:
            failed += 1

        time.sleep(REQUEST_DELAY)

    # Fetch state data for each state and offense
    for state in STATES:
        state_full_name = get_state_full_name(state)

        for offense in OFFENSES:
            data = fetch_state_data(state, offense)
            if data:
                records = parse_api_response(data, state_full_name)
                save_to_csv(records, f"{state}_{offense}.csv")
                completed += 1
            else:
                failed += 1

            time.sleep(REQUEST_DELAY)

    # Summary
    logger.info("=" * 50)
    logger.info("Fetch complete!")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Successful: {completed}")
    logger.info(f"Failed: {failed}")

    # List generated files
    csv_files = list(DATA_DIR.glob("*.csv"))
    logger.info(f"Generated {len(csv_files)} CSV files:")
    for f in sorted(csv_files):
        logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
