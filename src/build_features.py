import json
import logging
import pandas as pd
from config import (
    CLEAN_DATA_PATH, DEV_START_DATE, EMA_BETAS, FEATURES_DIR, FEATURES_LOG_PATH, MAX_FILL_HOURS,
    NEARBY_LAG_HOURS, NORMALISATION_PATH, NUM_LAG_HOURS, SENSORS, TEST_START_DATE, TIMEZONE,
)
from log_setup import setup_logging

logger = logging.getLogger(__name__)

# Loads the cleaned data into a table with one row per hour and one column per station
def load_hourly_table():
    clean = pd.read_csv(CLEAN_DATA_PATH)
    clean["datetime_utc"] = pd.to_datetime(clean["datetime_utc"], utc=True)
    table = clean.pivot(index="datetime_utc", columns="station", values="pm25")

    all_hours = pd.date_range(table.index.min(), table.index.max(), freq="h")

    return table.reindex(all_hours)

# Creates one row per hour to predict, where every input comes from an earlier hour
def build_station_features(table, station):
    readings = table[station]

    # Only the inputs are filled in, so the model is never scored against a made-up reading
    all_filled = table.ffill(limit=MAX_FILL_HOURS)
    filled = all_filled[station]
    features = {"target": readings}

    # Readings from the last NUM_LAG_HOURS hours
    for hours_ago in range(1, NUM_LAG_HOURS + 1):
        features[f"pm25_lag_{hours_ago}"] = filled.shift(hours_ago)

    # Bias-corrected exponentially weighted averages, up to and including the previous hour
    for beta in EMA_BETAS:
        features[f"ema_{beta}"] = filled.ewm(alpha=1 - beta, adjust=True).mean().shift(1)

    # Each station keeps its own columns, and a row's own station counts as missing since its lags are above
    for name in SENSORS:
        nearby = all_filled[name] if name != station else pd.Series(float("nan"), index=readings.index)
        is_missing = pd.Series(False, index=readings.index)

        for hours_ago in range(1, NEARBY_LAG_HOURS + 1):
            lag = nearby.shift(hours_ago)
            features[f"nearby_lag_{name}_{hours_ago}"] = lag
            is_missing = is_missing | lag.isna()

        features[f"nearby_missing_{name}"] = is_missing.astype(int)

    # One-hot hour of day and day of week in local time
    local_times = readings.index.tz_convert(TIMEZONE)

    for hour in range(24):
        features[f"hour_{hour}"] = (local_times.hour == hour).astype(int)

    for day in range(7): # Monday is 0
        features[f"day_{day}"] = (local_times.day_of_week == day).astype(int)

    # One-hot station, since a single model predicts for all stations
    for name in SENSORS:
        features[f"station_{name}"] = int(name == station)

    rows = pd.DataFrame(features, index=readings.index)
    rows.insert(0, "station", station)

    # Hours without a real target or a station's own inputs cannot be used, but another station's gap is allowed
    required = [c for c in rows.columns if not c.startswith("nearby_lag_")]

    return rows.dropna(subset=required)

# Splits rows by the hour being predicted
def split_rows(rows):
    dev_start = pd.Timestamp(DEV_START_DATE, tz="UTC")
    test_start = pd.Timestamp(TEST_START_DATE, tz="UTC")

    train = rows[rows.index < dev_start]
    dev = rows[(rows.index >= dev_start) & (rows.index < test_start)]
    test = rows[rows.index >= test_start]

    return train, dev, test

# Normalises every split using only the training set's mean and standard deviation
def normalise_features(train, dev, test):
    columns = [c for c in train.columns if c.startswith(("pm25_lag_", "ema_", "nearby_lag_"))]

    means = train[columns].mean()
    stds = train[columns].std()

    splits = [split.copy() for split in (train, dev, test)]
    for split in splits:
        split[columns] = (split[columns] - means) / stds

        # A missing nearby reading becomes the training mean, and its missing column tells the model
        split[columns] = split[columns].fillna(0)

    values = {"means": means.to_dict(), "stds": stds.to_dict()}

    return *splits, values

# Stops the pipeline if the features break a rule that they should always meet
def check_features(train, dev, test):
    for name, split in (("train", train), ("dev", dev), ("test", test)):
        if split.empty:
            raise ValueError(f"The {name} split is empty")
        if split.isna().any(axis=None):
            raise ValueError(f"The {name} split contains empty values")

    if train.index.max() >= dev.index.min() or dev.index.max() >= test.index.min():
        raise ValueError("Splits overlap in time")

    # Each row must have exactly one column set in each one-hot group
    for prefix in ("hour_", "day_", "station_"):
        for name, split in (("train", train), ("dev", dev), ("test", test)):
            columns = [c for c in split.columns if c.startswith(prefix)]
            if not (split[columns].sum(axis=1) == 1).all():
                raise ValueError(f"The {prefix} columns in {name} are not one-hot")

def run_build_features():
    setup_logging(FEATURES_LOG_PATH)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    table = load_hourly_table()

    # Stack each station's rows into one table
    station_rows = []
    for station in SENSORS:
        rows = build_station_features(table, station)
        real_hours = table[station].notna().sum()
        logger.info(f"{station}: usable rows {len(rows)} of {real_hours} cleaned readings")
        station_rows.append(rows)

    rows = pd.concat(station_rows).rename_axis("datetime_utc").sort_index()

    train, dev, test, normalisation_values = normalise_features(*split_rows(rows))
    check_features(train, dev, test)

    # Save the results
    for name, split in (("train", train), ("dev", dev), ("test", test)):
        split.to_csv(FEATURES_DIR / f"{name}.csv", float_format="%.6f")
        logger.info(f"{name}: {len(split)} rows from {split.index.min()} to {split.index.max()}")

    with open(NORMALISATION_PATH, "w", encoding="utf-8") as f:
        json.dump(normalisation_values, f, indent=4)

    logger.info("Feature building finished")

if __name__ == "__main__":
    run_build_features()
