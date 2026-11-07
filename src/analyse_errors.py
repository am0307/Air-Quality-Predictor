import json
import logging
import pandas as pd
import torch
from baseline_persistence import predict_persistence
from build_features import load_hourly_table
from config import (
    ANALYSIS_LOG_PATH, DEV_START_DATE, EARLY_WARNING_MIN_RISE, ERROR_LEVEL_EDGES, NORMALISATION_PATH, ONSET_MIN_RISE,
    RUNS_DIR, SMOKE_MIN_PM25, TEST_START_DATE, TIMEZONE,
)
from evaluation import create_run_dir, load_split
from log_setup import setup_logging
from train_dense import build_model
from training import create_tensors

logger = logging.getLogger(__name__)

# Loads the most recently saved dense network
def load_latest_dense_model(num_inputs, device):
    model_paths = list(RUNS_DIR.glob("dense_*/model.pt"))

    if not model_paths:
        raise FileNotFoundError(f"No dense network found in {RUNS_DIR}, run train_dense.py")

    model_path = max(model_paths, key=lambda path: path.stat().st_mtime)
    logger.info(f"Analysing {model_path.parent.name}")

    model = build_model(num_inputs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

# Compares the models for each group of hours
def summarise_errors(rows, groups):
    summary = rows.groupby(groups, observed=True).agg(
        rows=("target", "size"),
        mean_reading=("target", "mean"),
        persistence_rmse=("persistence_error", lambda errors: (errors ** 2).mean() ** 0.5),
        dense_rmse=("dense_error", lambda errors: (errors ** 2).mean() ** 0.5),
        dense_mean_error=("dense_error", "mean"), # Negative means that the network predicts too low
    )

    # Each group's share of the total squared error, which shows where the overall RMSE comes from
    squared_errors = rows["dense_error"] ** 2
    summary["share_of_dense_error"] = squared_errors.groupby(groups, observed=True).sum() / squared_errors.sum()

    return summary

# Tracks how hard non-smoke hours are by month, using persistence since it has nothing to overfit
def summarise_months(rows):
    calm = rows[rows["target"] < SMOKE_MIN_PM25]

    return calm.groupby(calm.index.tz_convert(TIMEZONE).strftime("%Y-%m").rename("local_month")).agg(
        rows=("target", "size"),
        mean_reading=("target", "mean"),
        persistence_rmse=("persistence_error", lambda errors: (errors ** 2).mean() ** 0.5),
        mean_hourly_change=("persistence_error", lambda errors: errors.abs().mean()),
        no_change_share=("persistence_error", lambda errors: (errors.abs() < 0.01).mean()),
    )

# Lists the days with high readings, and how many stations saw them, to tell city-wide events from local ones
def summarise_high_days(rows):
    high = rows[rows["target"] >= SMOKE_MIN_PM25]

    return high.groupby(high.index.tz_convert(TIMEZONE).strftime("%Y-%m-%d").rename("local_day")).agg(
        high_rows=("target", "size"),
        stations=("station", "nunique"),
        highest_reading=("target", "max"),
    )

# Checks whether a rise at one station tends to follow a rise at another station an hour earlier
def summarise_early_warning(table):
    changes = table.diff()
    summary = []

    for station in table.columns:
        own = changes[station]
        others = changes.drop(columns=station)

        # Whether any other station rose in the hour before, compared to how often that happens anyway
        warned = others.shift(1).max(axis=1) >= EARLY_WARNING_MIN_RISE
        known = own.notna()
        onsets = own >= ONSET_MIN_RISE

        summary.append({
            "station": station,
            "onsets": int(onsets.sum()),
            "warned_before_onset": warned[onsets].mean(),
            "warned_in_any_hour": warned[known].mean(),
            "correlation_with_own_last_change": own.corr(own.shift(1)),
            "correlation_with_others_last_change": others.shift(1).corrwith(own).mean(),
            "correlation_with_others_same_hour": others.corrwith(own).mean(),
        })

    return pd.DataFrame(summary).set_index("station")

def run_analysis():
    setup_logging(ANALYSIS_LOG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(NORMALISATION_PATH, encoding="utf-8") as f:
        normalisation_values = json.load(f)

    # The test set is left out until the final evaluation
    train = load_split("train")
    dev = load_split("dev")
    inputs, _ = create_tensors(dev, device)

    model = load_latest_dense_model(inputs.shape[1], device)
    with torch.no_grad():
        dense_predictions = model(inputs).squeeze(1).cpu().numpy()

    # Keep only the columns needed, with each error as the prediction minus the reading
    dev_rows = dev[["station", "target"]].copy()
    dev_rows["persistence_error"] = predict_persistence(dev, normalisation_values) - dev["target"]
    dev_rows["dense_error"] = dense_predictions - dev["target"]

    train_rows = train[["station", "target"]].copy()
    train_rows["persistence_error"] = predict_persistence(train, normalisation_values) - train["target"]
    both_rows = pd.concat([train_rows, dev_rows[["station", "target", "persistence_error"]]])

    months = dev_rows.index.tz_convert(TIMEZONE).strftime("%Y-%m").rename("local_month")
    levels = pd.cut(dev_rows["target"], ERROR_LEVEL_EDGES, right=False)

    # Only the cleaned readings from before the test set are compared
    table = load_hourly_table()
    table = table[table.index < pd.Timestamp(TEST_START_DATE, tz="UTC")]
    dev_start = pd.Timestamp(DEV_START_DATE, tz="UTC")

    tables = {
        "dev_by_month": summarise_errors(dev_rows, months),
        "dev_by_station": summarise_errors(dev_rows, dev_rows["station"]),
        "dev_by_level": summarise_errors(dev_rows, levels),
        "non_smoke_by_month": summarise_months(both_rows),
        "dev_high_days": summarise_high_days(dev_rows),
        "early_warning_train": summarise_early_warning(table[table.index < dev_start]),
        "early_warning_dev": summarise_early_warning(table[table.index >= dev_start]),
    }

    # Save the results
    run_dir = create_run_dir("analysis")
    for name, summary in tables.items():
        summary.to_csv(run_dir / f"{name}.csv")
        logger.info(f"{name}:\n{summary.round(3).to_string()}")

    logger.info(f"Results saved to {run_dir}")

if __name__ == "__main__":
    run_analysis()
