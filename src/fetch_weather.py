import logging
import sys
import time
import pandas as pd
import requests
from config import (
    WEATHER_ARCHIVE_DELAY_DAYS, WEATHER_DATA_DIR, WEATHER_LOG_PATH, WEATHER_START_MONTH, WEATHER_STATION_ID,
)
from log_setup import setup_logging

BASE_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
REQUEST_DELAY = 1 # Seconds between requests, to be polite to the server
REQUEST_TIMEOUT = 60 # Seconds to wait for a response

logger = logging.getLogger(__name__)

# A saved month is complete only if it was downloaded after its last hours reached the archive
def is_complete(path, month_start):
    if not path.exists():
        return False

    downloaded = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")
    archived_by = month_start + pd.offsets.MonthBegin(1) + pd.Timedelta(days=WEATHER_ARCHIVE_DELAY_DAYS)

    return downloaded > archived_by

# Retrieves one month of hourly observations as a CSV file
def get_month(session, month_start):
    params = {
        "format": "csv",
        "stationID": WEATHER_STATION_ID,
        "Year": month_start.year,
        "Month": month_start.month,
        "Day": 1,
        "timeframe": 1, # Hourly
        "submit": "Download Data",
    }

    response = session.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    # An error page can still come back with status 200, so check that it is the expected CSV
    if b"Date/Time (LST)" not in response.content[:2000]:
        raise ValueError("Response is not an hourly observations CSV")

    return response.content

# Saves to a temporary file first so an interrupted run cannot leave a partial file behind
def save_month(path, content):
    temp_path = path.with_suffix(".tmp")
    temp_path.write_bytes(content)
    temp_path.replace(path)

def run_fetch_weather():
    setup_logging(WEATHER_LOG_PATH)
    WEATHER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    months = pd.date_range(WEATHER_START_MONTH, pd.Timestamp.now(tz="UTC").tz_localize(None), freq="MS")

    with requests.Session() as session:
        session.headers.update({"User-Agent": "AQ Forecaster"})

        for month in months:
            month_start = month.tz_localize("UTC")
            path = WEATHER_DATA_DIR / f"pearson_{month_start:%Y_%m}.csv"

            if is_complete(path, month_start):
                logger.info(f"{month_start:%Y-%m}: already downloaded, skipping")
                continue

            # Finished files are kept, so running this again resumes after the failure
            try:
                content = get_month(session, month_start)
            except (requests.exceptions.RequestException, ValueError) as error:
                logger.error(f"{month_start:%Y-%m}: download failed, stopping ({error})")
                sys.exit(1)

            save_month(path, content)
            logger.info(f"{month_start:%Y-%m}: saved {len(content)} bytes")
            time.sleep(REQUEST_DELAY)

    logger.info("Weather download finished")

if __name__ == "__main__":
    run_fetch_weather()
