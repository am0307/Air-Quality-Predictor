import json
import logging
import sys
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from config import ENV_PATH, FETCH_LOG_PATH, RAW_DATA_DIR, SENSORS, START_YEAR
from openaq_api import create_session, get_sensor_year

logger = logging.getLogger(__name__)

# Sends log messages to both the terminal and the log file
def setup_logging():
    FETCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(FETCH_LOG_PATH, encoding="utf-8"),
        ],
    )

# A saved year is complete only if it was downloaded after that year ended
def is_complete(path, year):
    if not path.exists():
        return False

    with open(path, encoding="utf-8") as f:
        saved = json.load(f)

    return int(saved["fetched_at"][:4]) > year

# Saves to a temporary file first so an interrupted run cannot leave a partial file behind
def save_rows(path, station, sensor_id, year, fetched_at, rows):
    data = {
        "station": station,
        "sensor_id": sensor_id,
        "year": year,
        "fetched_at": fetched_at,
        "rows": rows,
    }

    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    temp_path.replace(path)

def run_fetch():
    setup_logging()
    load_dotenv(dotenv_path=ENV_PATH)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    current_year = datetime.now(timezone.utc).year

    with create_session() as session:

        # Loop through each station and year so that files can be downloaded one at a time
        for station, sensor_id in SENSORS.items():
            for year in range(START_YEAR, current_year + 1):
                path = RAW_DATA_DIR / f"{station}_{year}.json"

                if is_complete(path, year):
                    logger.info(f"{station} {year}: already downloaded, skipping")
                    continue

                # Finished files are kept, so running this again resumes after the failure
                try:
                    fetched_at = datetime.now(timezone.utc).isoformat()
                    rows = get_sensor_year(session, sensor_id, year)
                except requests.exceptions.RequestException as error:
                    logger.error(f"{station} {year}: download failed, stopping ({error})")
                    sys.exit(1)

                save_rows(path, station, sensor_id, year, fetched_at, rows)
                logger.info(f"{station} {year}: saved {len(rows)} rows")

    logger.info("Download finished")

if __name__ == "__main__":
    run_fetch()