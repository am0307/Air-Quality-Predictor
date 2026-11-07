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

# Locations of the features, normalisation values, and feature log
FEATURES_DIR = ROOT_DIR / "data" / "features"
NORMALISATION_PATH = FEATURES_DIR / "normalisation.json"
FEATURES_LOG_PATH = ROOT_DIR / "data" / "logs" / "build_features.log"

# Locations of each run's saved results and the baseline logs
RUNS_DIR = ROOT_DIR / "runs"
PERSISTENCE_LOG_PATH = ROOT_DIR / "data" / "logs" / "baseline_persistence.log"
LINEAR_LOG_PATH = ROOT_DIR / "data" / "logs" / "baseline_linear.log"
PLOT_LOG_PATH = ROOT_DIR / "data" / "logs" / "plot_history.log"
DENSE_LOG_PATH = ROOT_DIR / "data" / "logs" / "train_dense.log"
DIAGNOSIS_LOG_PATH = ROOT_DIR / "data" / "logs" / "diagnose_train_dev.log"

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

# Hours of past readings given to the model
NUM_LAG_HOURS = 24

# Longest stretch of missing hours filled in with the last known reading
MAX_FILL_HOURS = 3

# Exponentially weighted average betas, which average over roughly 1 / (1 - beta) hours
EMA_BETAS = [0.9, 0.98]

# Data is split by time into roughly 60% train, 20% dev, and 20% test, with smoke events in each
DEV_START_DATE = "2025-07-01"
TEST_START_DATE = "2026-03-01"

# Hours with a reading at or above this (µg/m³) are reported as smoke, as the data has no smoke label
SMOKE_MIN_PM25 = 35

# Regularisation strengths compared on the dev set, which are large because the penalty is divided by 2m
# Zero is left out because the one-hot groups and the bias are redundant, so it has no unique solution
LINEAR_LAMBDAS = [1, 10, 100, 1000, 10000]

# Batch gradient descent settings, where the learning rate must stay below about 0.1 to converge
LINEAR_LEARNING_RATE = 0.05
LINEAR_ITERATIONS = 10000

# Iterations between saved costs
COST_RECORD_INTERVAL = 100

# Seed for weight initialisation and shuffling, so that a run can be repeated
RANDOM_SEED = 1

# Dense network starting point, to be tuned on the dev set
DENSE_HIDDEN_UNITS = [64, 32] # Units in each hidden layer
DENSE_LEARNING_RATE = 0.0003 # Below Adam's usual 0.001, which made the dev cost bounce more
DENSE_BATCH_SIZE = 1024
DENSE_EPOCHS = 100
DENSE_LAMBDA = 1000 # The best of those tried, as the network otherwise starts to overfit

# Whole weeks of training rows held out as the train-dev set, as neighbouring hours are too alike to sample randomly
TRAIN_DEV_WEEK_INTERVAL = 10 # Every 10th week, which is about 10% of the rows
TRAIN_DEV_WEEK_OFFSET = 4 # Holds out smoke from several months, while keeping June 2025 for training
TRAIN_DEV_GAP_HOURS = 24 # Training rows this close to a held-out week would share its lag values
TRAIN_DEV_MIN_SMOKE_ROWS = 50 # Fewer than this would make the smoke results meaningless

# Seeds for the diagnosis, since one dense network can be lucky or unlucky
DIAGNOSIS_SEEDS = [1, 2, 3]

# The lambda chosen by the linear baseline
DIAGNOSIS_LINEAR_LAMBDA = 1
