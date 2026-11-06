import json
import logging
from config import NORMALISATION_PATH, PERSISTENCE_LOG_PATH
from evaluation import create_run_dir, evaluate_predictions, load_split, save_results
from log_setup import setup_logging

logger = logging.getLogger(__name__)

# Predicts that the next hour's reading will equal the most recent one
def predict_persistence(rows, normalisation_values):

    # The lag columns are normalised, so convert back to µg/m³ to compare with the target
    mean = normalisation_values["means"]["pm25_lag_1"]
    std = normalisation_values["stds"]["pm25_lag_1"]

    return rows["pm25_lag_1"] * std + mean

def run_persistence():
    setup_logging(PERSISTENCE_LOG_PATH)

    with open(NORMALISATION_PATH, encoding="utf-8") as f:
        normalisation_values = json.load(f)

    # The test set is left out until the final evaluation
    results = {"model": "persistence"}
    for name in ("train", "dev"):
        rows = load_split(name)
        predictions = predict_persistence(rows, normalisation_values)
        results[name] = evaluate_predictions(predictions, rows["target"])

        for group, errors in results[name].items():
            logger.info(f"{name} {group}: RMSE {errors['rmse']:.3f}, MAE {errors['mae']:.3f}, {errors['rows']} rows")

    run_dir = create_run_dir("persistence")
    save_results(run_dir, results)
    logger.info(f"Results saved to {run_dir}")

if __name__ == "__main__":
    run_persistence()
