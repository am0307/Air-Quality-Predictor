from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Locations of the API key, downloaded data, and download log
ENV_PATH = ROOT_DIR / ".env"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
FETCH_LOG_PATH = ROOT_DIR / "data" / "logs" / "fetch_data.log"

# Local time zone
TIMEZONE = "America/Toronto"

# First year with PM2.5 data from any of the stations
START_YEAR = 2022

# OpenAQ PM2.5 sensor IDs for Toronto's four long-running stations
SENSORS = {
    "downtown": 5077866,
    "north": 6553645,
    "west": 6553681,
    "east": 6554969,
}
