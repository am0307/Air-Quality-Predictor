import json
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from baseline_persistence import predict_persistence
from config import EXTREME_MIN_PM25, NORMALISATION_PATH, ONSET_MIN_RISE, RUNS_DIR, SMOKE_MIN_PM25, TEST_LOG_PATH
from evaluation import create_run_dir, load_split, save_results
from log_setup import setup_logging
from train_dense import build_model
from training import create_tensors

logger = logging.getLogger(__name__)

# Finds the most recent final run, which holds the models trained on train and dev
def find_final_run():
    run_dirs = [path.parent for path in RUNS_DIR.glob("final_*/dense.pt")]

    if not run_dirs:
        raise FileNotFoundError(f"No final models found in {RUNS_DIR}, run train_final.py")

    return max(run_dirs, key=lambda path: path.stat().st_mtime)

# Loads the final dense network and linear model with the shapes they were trained with
def load_final_models(run_dir, num_inputs, device):
    with open(run_dir / "results.json", encoding="utf-8") as f:
        settings = json.load(f)["dense_settings"]

    dense = build_model(num_inputs, settings["hidden_units"]).to(device)
    dense.load_state_dict(torch.load(run_dir / "dense.pt", map_location=device))

    linear = nn.Linear(num_inputs, 1).to(device)
    linear.load_state_dict(torch.load(run_dir / "linear.pt", map_location=device))

    return {"dense": dense, "linear": linear}

# Predicts every row of a split with persistence and both final models
def predict_split(rows, models, device):
    with open(NORMALISATION_PATH, encoding="utf-8") as f:
        normalisation_values = json.load(f)

    inputs, _ = create_tensors(rows, device)
    predictions = {"persistence": predict_persistence(rows, normalisation_values).to_numpy()}

    with torch.no_grad():
        for name, model in models.items():
            predictions[name] = model(inputs).squeeze(1).cpu().numpy()

    return predictions

# Scores each model on each group of hours, including its gain over persistence
def score_predictions(rows, predictions):
    target = rows["target"].to_numpy()
    last_reading = predictions["persistence"]

    groups = {
        "all_hours": np.ones(len(target), dtype=bool),
        "smoke_hours": target >= SMOKE_MIN_PM25,
        "non_smoke_hours": target < SMOKE_MIN_PM25,
        "onset_hours": target - last_reading >= ONSET_MIN_RISE,
        "extreme_hours": target >= EXTREME_MIN_PM25,
    }

    scores = []
    for group, mask in groups.items():
        persistence_error = last_reading[mask] - target[mask]
        persistence_rmse = float((persistence_error ** 2).mean() ** 0.5)
        persistence_mae = float(np.abs(persistence_error).mean())

        for name, prediction in predictions.items():
            error = prediction[mask] - target[mask]
            rmse = float((error ** 2).mean() ** 0.5)
            mae = float(np.abs(error).mean())

            scores.append({
                "group": group,
                "model": name,
                "rows": int(mask.sum()),
                "rmse": rmse,
                "mae": mae,
                "mean_error": float(error.mean()), # Negative means predicting too low
                "rmse_gain_vs_persistence_%": 100 * (1 - rmse / persistence_rmse),
                "mae_gain_vs_persistence_%": 100 * (1 - mae / persistence_mae),
            })

    return pd.DataFrame(scores)

def run_test_evaluation():
    setup_logging(TEST_LOG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The one evaluation on the test set, made after every choice about the models was fixed
    run_dir = find_final_run()
    test_rows = load_split("test")
    logger.info(f"Evaluating {run_dir.name} on {len(test_rows)} test rows from {test_rows.index.min()} to {test_rows.index.max()}")

    num_inputs = create_tensors(test_rows.iloc[:1], device)[0].shape[1]
    models = load_final_models(run_dir, num_inputs, device)
    scores = score_predictions(test_rows, predict_split(test_rows, models, device))

    for group, group_scores in scores.groupby("group", sort=False):
        logger.info(f"{group} ({group_scores['rows'].iloc[0]} rows):\n{group_scores.drop(columns=['group', 'rows']).round(3).to_string(index=False)}")

    # Save the results
    output_dir = create_run_dir("test")
    scores.to_csv(output_dir / "test_scores.csv", index=False)
    save_results(output_dir, {"model": "test_evaluation", "final_run": run_dir.name, "rows": len(test_rows)})
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    run_test_evaluation()
