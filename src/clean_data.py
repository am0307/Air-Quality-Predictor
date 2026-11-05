import json
import logging
import pandas as pd
from config import (
    CLEAN_DATA_PATH, CLEAN_LOG_PATH, CLEAN_START_DATE, CLEANING_REPORT_PATH, PM25_MAX, PM25_MIN,
    RAW_DATA_DIR, REMOVED_ROWS_PATH, SPIKE_MIN_OTHER_STATIONS, SPIKE_MIN_VALUE, SPIKE_RATIO,
    SPIKE_WINDOW_HOURS, STUCK_MIN_HOURS,
)
from log_setup import setup_logging

logger = logging.getLogger(__name__)

# Reads every raw file into one table with a row for each reading
def load_raw_rows():
    records = []

    for path in sorted(RAW_DATA_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)

        for row in saved["rows"]:
            records.append({
                "station": saved["station"],
                "datetime_utc": row["period"]["datetimeFrom"]["utc"],
                "pm25": row["value"],
                "parameter": row["parameter"]["name"],
                "units": row["parameter"]["units"],
                "interval": row["period"]["interval"],
                "has_flags": row["flagInfo"]["hasFlags"],
            })

    if not records:
        raise FileNotFoundError(f"No raw data found in {RAW_DATA_DIR}, run fetch_data.py")

    df = pd.DataFrame(records)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["removal_reason"] = "" # Stays empty for rows that are kept

    return df

# Records why rows were removed, keeping the first reason if several rules apply
def mark_removed(df, mask, reason):
    df.loc[mask & (df["removal_reason"] == ""), "removal_reason"] = reason

# Creates a table with one row per hour and one column per station, where missing hours are empty
def create_hourly_table(df):
    kept = df[df["removal_reason"] == ""]
    table = kept.pivot(index="datetime_utc", columns="station", values="pm25")

    all_hours = pd.date_range(table.index.min(), table.index.max(), freq="h")

    return table.reindex(all_hours)

# Marks long runs of an identical reading, which suggest that a sensor is stuck
def mark_stuck_readings(df):
    table = create_hourly_table(df)

    for station in table.columns:
        readings = table[station]

        # A new run starts whenever the reading changes, and empty hours always end a run
        run_ids = (readings != readings.shift()).cumsum()
        run_lengths = readings.groupby(run_ids).transform("size")
        stuck_hours = readings[run_lengths >= STUCK_MIN_HOURS].index

        mask = (df["station"] == station) & df["datetime_utc"].isin(stuck_hours)
        mark_removed(df, mask, "stuck_sensor")

# Marks high readings that no other station saw anything close to around the same time
def mark_spikes(df):
    table = create_hourly_table(df)
    window = 2 * SPIKE_WINDOW_HOURS + 1

    for station in table.columns:
        others = table.drop(columns=station)

        # Highest reading from any other station within the window around each hour
        others_max = others.rolling(window, center=True, min_periods=1).max().max(axis=1)

        # Number of other stations with at least one reading within the window
        others_reporting = others.notna().astype(int).rolling(window, center=True, min_periods=1).sum()
        others_reporting = (others_reporting > 0).sum(axis=1)

        is_spike = (
            (table[station] >= SPIKE_MIN_VALUE)
            & (table[station] > SPIKE_RATIO * others_max)
            & (others_reporting >= SPIKE_MIN_OTHER_STATIONS)
        )
        spike_hours = table.index[is_spike]

        mask = (df["station"] == station) & df["datetime_utc"].isin(spike_hours)
        mark_removed(df, mask, "single_station_spike")

# Applies each cleaning rule in order
def apply_cleaning_rules(df):
    bad_format = (
        df["pm25"].isna()
        | (df["parameter"] != "pm25")
        | (df["units"] != "µg/m³")
        | (df["interval"] != "01:00:00")
    )
    mark_removed(df, bad_format, "bad_format")
    mark_removed(df, df["has_flags"], "provider_flagged")
    mark_removed(df, df["datetime_utc"] < pd.Timestamp(CLEAN_START_DATE, tz="UTC"), "before_start_date")

    # Only compare rows that are still kept, so a removed row cannot make a valid one a duplicate
    kept = df["removal_reason"] == ""
    duplicates = df[kept].duplicated(subset=["station", "datetime_utc"]).reindex(df.index, fill_value=False)
    mark_removed(df, duplicates, "duplicate")
    mark_removed(df, (df["pm25"] < PM25_MIN) | (df["pm25"] > PM25_MAX), "out_of_range")

    mark_stuck_readings(df)
    mark_spikes(df)

# Stops the pipeline if the cleaned data breaks a rule that it should always meet
def check_clean_data(clean):
    if clean["pm25"].isna().any():
        raise ValueError("Cleaned data contains empty readings")
    if clean.duplicated(subset=["station", "datetime_utc"]).any():
        raise ValueError("Cleaned data contains duplicate hours")
    if not clean["pm25"].between(PM25_MIN, PM25_MAX).all():
        raise ValueError("Cleaned data contains readings outside of the valid range")

# Counts the gaps of missing hours for one station, grouped by length
def count_gaps(hours):
    missing = hours.sort_values().diff().dropna() / pd.Timedelta(hours=1) - 1
    gaps = missing[missing > 0]

    return {
        "missing_hours": int(gaps.sum()),
        "gaps_1_hour": int((gaps == 1).sum()),
        "gaps_2_to_3_hours": int(gaps.between(2, 3).sum()),
        "gaps_4_to_24_hours": int(gaps.between(4, 24).sum()),
        "gaps_over_24_hours": int((gaps > 24).sum()),
    }

# Summarises what happened to each station's rows
def create_report(df):
    report = {}

    for station, rows in df.groupby("station"):
        kept = rows[rows["removal_reason"] == ""]
        removed = rows[rows["removal_reason"] != ""]

        report[station] = {
            "raw_rows": len(rows),
            "kept_rows": len(kept),
            "removed_rows": removed["removal_reason"].value_counts().to_dict(),
            "first_hour": kept["datetime_utc"].min().isoformat(),
            "last_hour": kept["datetime_utc"].max().isoformat(),
            **count_gaps(kept["datetime_utc"]),
        }

        logger.info(f"{station}: kept {len(kept)} of {len(rows)} rows, removed {report[station]['removed_rows']}")

    return report

def run_clean():
    setup_logging(CLEAN_LOG_PATH)
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_rows()
    apply_cleaning_rules(df)

    # Sort so that each station's readings are in time order
    df = df.sort_values(["station", "datetime_utc"])

    clean = df[df["removal_reason"] == ""][["datetime_utc", "station", "pm25"]]
    removed = df[df["removal_reason"] != ""][["datetime_utc", "station", "pm25", "removal_reason"]]

    check_clean_data(clean)
    report = create_report(df)

    # Save the results
    clean.to_csv(CLEAN_DATA_PATH, index=False)
    removed.to_csv(REMOVED_ROWS_PATH, index=False)
    with open(CLEANING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    logger.info(f"Cleaning finished: kept {len(clean)} rows, removed {len(removed)} rows")

if __name__ == "__main__":
    run_clean()
