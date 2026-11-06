import json
from datetime import datetime
import pandas as pd
from config import FEATURES_DIR, RUNS_DIR, SMOKE_MIN_PM25

# Loads one split of the features
def load_split(name):
    return pd.read_csv(FEATURES_DIR / f"{name}.csv", index_col="datetime_utc", parse_dates=True)

# Calculates the errors in µg/m³ for one group of hours
def calculate_errors(predictions, targets):
    errors = predictions - targets

    return {
        "rows": len(errors),
        "rmse": float((errors ** 2).mean() ** 0.5),
        "mae": float(errors.abs().mean()),
    }

# Reports smoke and non-smoke hours separately, since a few smoke hours can dominate the overall error
def evaluate_predictions(predictions, targets):
    is_smoke = targets >= SMOKE_MIN_PM25

    return {
        "all_hours": calculate_errors(predictions, targets),
        "smoke_hours": calculate_errors(predictions[is_smoke], targets[is_smoke]),
        "non_smoke_hours": calculate_errors(predictions[~is_smoke], targets[~is_smoke]),
    }

# Creates a new folder for a run, so that earlier runs are never overwritten
def create_run_dir(model_name):
    run_dir = RUNS_DIR / f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True)

    return run_dir

# Saves a run's settings and final results
def save_results(run_dir, results):
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
