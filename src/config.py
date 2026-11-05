from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Locations of the API key, downloaded data, and download log
ENV_PATH = ROOT_DIR / ".env"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
FETCH_LOG_PATH = ROOT_DIR / "data" / "logs" / "fetch_data.log"

# Locations of the cleaned data, removed rows, cleaning report, and cleaning log
CLEAN_DATA_PATH = ROOT_DIR / "data" / "clean" / "pm25_hourly.csv"
REMOVED_ROWS_PATH = ROOT_DIR / "data" / "clean" / "removed_rows.csv"
CLEANING_REPORT_PATH = ROOT_DIR / "data" / "clean" / "cleaning_report.json"
CLEAN_LOG_PATH = ROOT_DIR / "data" / "logs" / "clean_data.log"

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

# Readings before this date are ignored because steady reporting begins here
CLEAN_START_DATE = "2023-05-01"

# PM2.5 readings outside of this range (µg/m³) are treated as invalid
PM25_MIN = 0
PM25_MAX = 1000

# This many hours of an identical reading in a row suggests that a sensor is stuck
STUCK_MIN_HOURS = 48

# Variables for evaluating whether a reading is a spike compared to other stations' readings
SPIKE_MIN_VALUE = 50 # Minimum reading to be considered a spike (µg/m³)
SPIKE_RATIO = 3 # Ratio of the spike value to the maximum value from other stations
SPIKE_WINDOW_HOURS = 3 # Hours checked on each side of the reading
SPIKE_MIN_OTHER_STATIONS = 2 # Stations that must be reporting to make a fair comparison
