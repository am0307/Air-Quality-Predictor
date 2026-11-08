import json
import logging
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from build_features import (
    add_wind_features, apply_normalisation, build_station_features, create_hourly_table, create_wind_table,
)
from clean_data import apply_cleaning_rules, create_rows_table
from config import (
    ENV_PATH, FEATURES_DIR, LIVE_LOOKBACK_DAYS, LIVE_MAX_DATA_AGE_HOURS, LIVE_WEATHER_HOURS, LIVE_WEATHER_STATION_ID,
    LIVE_WEATHER_URL, NORMALISATION_PATH, PREDICT_LOG_PATH, SENSORS, TIMEZONE,
)
from evaluate_test import find_final_run
from log_setup import setup_logging
from openaq_api import create_session, get_sensor_range
from train_dense import build_model
from training import get_feature_columns

REQUEST_TIMEOUT = 60 # Seconds to wait for the weather service

logger = logging.getLogger(__name__)

# Formats a time the way the OpenAQ API expects
def format_utc(time):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")

# Retrieves the last LIVE_LOOKBACK_DAYS of hourly readings for every station, in the same form as the raw files
def get_recent_readings(now):
    start = now - pd.Timedelta(days=LIVE_LOOKBACK_DAYS)
    saved_responses = []

    with create_session() as session:
        for station, sensor_id in SENSORS.items():
            rows = get_sensor_range(session, sensor_id, format_utc(start), format_utc(now))
            saved_responses.append({"station": station, "rows": rows})
            logger.info(f"{station}: retrieved {len(rows)} hourly readings")

    return create_rows_table(saved_responses)

# Cleans recent readings with every rule that does not need future readings
def clean_recent_readings(df):
    apply_cleaning_rules(df, include_spikes=False)
    removed = df["removal_reason"].value_counts().drop("", errors="ignore").to_dict()
    if removed:
        logger.info(f"Removed while cleaning: {removed}")

    return df[df["removal_reason"] == ""][["datetime_utc", "station", "pm25"]]

# Retrieves Pearson's recent on-the-hour wind, rounded the same way as the archive used in training
def get_recent_wind(now):
    start = now - pd.Timedelta(hours=LIVE_WEATHER_HOURS)
    params = {
        "f": "json",
        "limit": 500,
        "msc_id-value": LIVE_WEATHER_STATION_ID,
        "datetime": f"{format_utc(start)}/{format_utc(now)}",
    }

    response = requests.get(LIVE_WEATHER_URL, params=params, headers={"User-Agent": "AQ Forecaster"}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    records = []
    for feature in response.json().get("features", []):
        properties = feature["properties"]
        time = pd.Timestamp(properties["date_tm-value"])

        # Only the reading on the hour matches the archive's hourly value, which is a 2-minute average
        speed = properties.get("avg_wnd_spd_10m_pst2mts")
        if time.minute != 0 or speed is None:
            continue

        # The archive rounds speed to whole km/h and direction to tens of degrees, so the live values must match
        direction = properties.get("avg_wnd_dir_10m_pst2mts")
        records.append({
            "time": time,
            "speed": round(speed),
            "direction_tens": round(direction / 10) if direction is not None else None,
        })

    if not records:
        return None

    wind = pd.DataFrame(records).drop_duplicates("time").sort_values("time")
    times = pd.DatetimeIndex(wind["time"])
    speed = pd.to_numeric(wind["speed"]).set_axis(times)
    direction_tens = pd.to_numeric(wind["direction_tens"]).set_axis(times)

    return create_wind_table(times, speed, direction_tens)

# Builds the normalised input row for each station for the hour after the newest reading
def build_prediction_rows(clean, wind, normalisation_values):
    table = create_hourly_table(clean)
    predict_hour = table.index.max() + pd.Timedelta(hours=1)

    # The hour being predicted gets an empty row, so the lags and averages reach up to the newest reading
    table = table.reindex(pd.date_range(table.index.min(), predict_hour, freq="h"))

    station_rows = []
    for station in SENSORS:
        rows = build_station_features(table, station, keep_unlabelled=True)

        if predict_hour in rows.index:
            station_rows.append(rows.loc[[predict_hour]])
        else:
            logger.warning(f"{station}: no prediction, as its own recent readings are missing")

    if not station_rows:
        raise ValueError("No station has enough recent readings to predict")

    rows = pd.concat(station_rows).rename_axis("datetime_utc")

    if wind is None:
        wind = pd.DataFrame({"u": [], "v": []}, index=pd.DatetimeIndex([], tz="UTC"))
    rows = add_wind_features(rows, wind)

    if rows["wind_missing"].any():
        logger.warning("Recent wind is missing, and the model never saw missing wind while training")

    return apply_normalisation(rows, normalisation_values), predict_hour

# Loads the final dense network, which learned from train and dev together
def load_final_dense(num_inputs):
    run_dir = find_final_run()

    with open(run_dir / "results.json", encoding="utf-8") as f:
        hidden_units = json.load(f)["dense_settings"]["hidden_units"]

    model = build_model(num_inputs, hidden_units)
    model.load_state_dict(torch.load(run_dir / "dense.pt", map_location="cpu"))
    model.eval()

    return model, run_dir

# Predicts each station's reading from its normalised input row
def predict_rows(rows, normalisation_values):
    feature_columns = get_feature_columns(rows)

    # The columns must be in exactly the order the network was trained on
    training_columns = get_feature_columns(pd.read_csv(FEATURES_DIR / "train.csv", nrows=0, index_col=0))
    if feature_columns != training_columns:
        raise ValueError("The live features do not match the training features")

    model, run_dir = load_final_dense(len(feature_columns))
    inputs = torch.tensor(rows[feature_columns].to_numpy(dtype="float32"))

    with torch.no_grad():
        predictions = model(inputs).squeeze(1).numpy()

    # The newest reading, converted back to µg/m³ for comparison
    mean = normalisation_values["means"]["pm25_lag_1"]
    std = normalisation_values["stds"]["pm25_lag_1"]
    newest = rows["pm25_lag_1"] * std + mean

    return pd.DataFrame({
        "station": rows["station"].to_numpy(),
        "newest_reading": newest.round(1).to_numpy(),
        "prediction": predictions.clip(min=0).round(1), # PM2.5 cannot be negative
    }), run_dir

def run_predict():
    setup_logging(PREDICT_LOG_PATH)
    load_dotenv(dotenv_path=ENV_PATH)
    now = pd.Timestamp.now(tz="UTC").floor("min")

    with open(NORMALISATION_PATH, encoding="utf-8") as f:
        normalisation_values = json.load(f)

    clean = clean_recent_readings(get_recent_readings(now))
    wind = get_recent_wind(now)
    rows, predict_hour = build_prediction_rows(clean, wind, normalisation_values)

    # Each reading covers the hour after its time, so the newest one ended when the predicted hour began
    data_age = (now - predict_hour) / pd.Timedelta(hours=1)
    if data_age > LIVE_MAX_DATA_AGE_HOURS:
        logger.warning(f"The newest reading ended {data_age:.1f} hours ago, so this prediction is for an hour already passed")

    predictions, run_dir = predict_rows(rows, normalisation_values)

    local_hour = predict_hour.tz_convert(TIMEZONE)
    logger.info(f"Model: {run_dir.name}")
    logger.info(f"Predicted PM2.5 (µg/m³) for {local_hour:%Y-%m-%d %H:%M} to {local_hour + pd.Timedelta(hours=1):%H:%M} Toronto time:\n"
                f"{predictions.to_string(index=False)}")
    logger.info("Live readings cannot be checked for single-station spikes, because that rule needs readings from later hours")

if __name__ == "__main__":
    run_predict()
