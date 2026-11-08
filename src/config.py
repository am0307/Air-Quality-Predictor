from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Locations of the API key, downloaded data, and download log
ENV_PATH = ROOT_DIR / ".env"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
FETCH_LOG_PATH = ROOT_DIR / "data" / "logs" / "fetch_data.log"

# Locations of the downloaded weather data and its download log
WEATHER_DATA_DIR = RAW_DATA_DIR / "weather"
WEATHER_LOG_PATH = ROOT_DIR / "data" / "logs" / "fetch_weather.log"

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
SEARCH_LOG_PATH = ROOT_DIR / "data" / "logs" / "tune_dense.log"
DIAGNOSIS_LOG_PATH = ROOT_DIR / "data" / "logs" / "diagnose_train_dev.log"
ANALYSIS_LOG_PATH = ROOT_DIR / "data" / "logs" / "analyse_errors.log"
SEQUENCE_LOG_PATH = ROOT_DIR / "data" / "logs" / "train_sequence.log"
COMPARISON_LOG_PATH = ROOT_DIR / "data" / "logs" / "compare_dense_gru_lstm.log"
FINAL_LOG_PATH = ROOT_DIR / "data" / "logs" / "train_final.log"
TEST_LOG_PATH = ROOT_DIR / "data" / "logs" / "evaluate_test.log"
PREDICT_LOG_PATH = ROOT_DIR / "data" / "logs" / "predict.log"

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

# Hours of past readings given to the model from each of the other stations
NEARBY_LAG_HOURS = 3

# Environment Canada station ID for Toronto Pearson (TORONTO INTL A), which provides the hourly wind
WEATHER_STATION_ID = 51459

# First month of weather downloaded, a month before CLEAN_START_DATE so that the first rows have past wind
WEATHER_START_MONTH = "2023-04"

# The archive lags by about a day, so a month counts as complete once downloaded this many days after it ends
WEATHER_ARCHIVE_DELAY_DAYS = 2

# Weather times are in local standard time, which is UTC-5 all year as it ignores daylight saving
LST_UTC_OFFSET_HOURS = 5

# Hours of past wind given to the model
WIND_LAG_HOURS = 3

# Environment Canada's real-time observations for Pearson, which appear about a minute after each hour
LIVE_WEATHER_URL = "https://api.weather.gc.ca/collections/swob-realtime/items"
LIVE_WEATHER_STATION_ID = "6158731" # Pearson's climate ID, which the real-time data uses
LIVE_WEATHER_HOURS = 12 # Hours of live wind retrieved, enough for the lags plus filling short gaps

# Days of recent readings retrieved for a live prediction, where the slower average gives older readings under 1% weight
LIVE_LOOKBACK_DAYS = 10

# A warning is given if the newest reading ended longer ago than this
LIVE_MAX_DATA_AGE_HOURS = 3

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

# Dense network settings, which had the lowest mean dev RMSE after the random search and seed check
DENSE_SETTINGS = {
    "hidden_units": [256, 128, 128], # Units in each hidden layer
    "learning_rate": 0.0025872174983709267,
    "decay_rate": 0.1, # Settles training so that the final weights can be used without choosing an epoch
    "batch_size": 2048,
    "epochs": 300,
    "lambda": 2526.931078696125,
}

# Random search over the dense network's settings, with a fixed seed so the same trials can be repeated
SEARCH_SEED = 2
SEARCH_TRIALS = 20
SEARCH_EPOCHS = 300

# These are sampled on a log scale, so that each power of 10 gets the same share of trials
SEARCH_LEARNING_RATE_RANGE = (0.0003, 0.003)
SEARCH_LAMBDA_RANGE = (5, 5000)

# These are sampled uniformly
SEARCH_LAYER_OPTIONS = [1, 2, 3]
SEARCH_UNIT_OPTIONS = [32, 64, 128, 256]
SEARCH_BATCH_OPTIONS = [1024, 2048]

# Whole weeks of training rows held out as the train-dev set, as neighbouring hours are too alike to sample randomly
TRAIN_DEV_WEEK_INTERVAL = 10 # Every 10th week, which is about 10% of the rows
TRAIN_DEV_WEEK_OFFSET = 4 # Holds out smoke from several months, while keeping June 2025 for training
TRAIN_DEV_GAP_HOURS = 24 # Training rows this close to a held-out week would share its lag values
TRAIN_DEV_MIN_SMOKE_ROWS = 50 # Fewer than this would make the smoke results meaningless

# Seeds for the diagnosis, since one dense network can be lucky or unlucky
DIAGNOSIS_SEEDS = [1, 2, 3]

# The lambda chosen by the linear baseline
DIAGNOSIS_LINEAR_LAMBDA = 1

# Readings at or above this (µg/m³) are reported separately in the final evaluation, as test has far more than train
EXTREME_MIN_PM25 = 100

# Reading levels (µg/m³) used to group errors in the error analysis
ERROR_LEVEL_EDGES = [0, 5, 10, 20, 35, 50, 100, 1000]

# A station has an onset when it rises this much in one hour (µg/m³)
ONSET_MIN_RISE = 10

# Another station gives early warning of an onset if it rose this much in the hour before (µg/m³)
EARLY_WARNING_MIN_RISE = 5

# GRU or LSTM settings, which train the same way as the dense network apart from gradient clipping
SEQUENCE_SETTINGS = {
    **DENSE_SETTINGS,
    "cell": "gru", # "gru" or "lstm"
    "hidden_size": 64, # Chosen on the train-dev weeks by compare_dense_gru_lstm.py
    "head_units": 64, # Units in the dense layer after the sequence
    "max_gradient_norm": 10,
}

# Options compared on the train-dev weeks before the single comparison on dev
COMPARISON_CELLS = ["gru", "lstm"]
COMPARISON_HIDDEN_SIZES = [16, 32, 64]
COMPARISON_SELECTION_SEEDS = [1, 2, 3]
COMPARISON_DEV_SEEDS = [1, 2, 3, 11, 12, 13]

# A sequence model replaces dense only if its mean dev RMSE is this much lower, with at least this many seeds better
COMPARISON_MIN_IMPROVEMENT = 0.01
COMPARISON_MIN_SEEDS_BETTER = 5
