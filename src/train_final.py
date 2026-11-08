import logging
import pandas as pd
import torch
from baseline_linear import train_model as train_linear
from config import DENSE_SETTINGS, DIAGNOSIS_LINEAR_LAMBDA, FINAL_LOG_PATH, RANDOM_SEED
from evaluation import create_run_dir, load_split, save_results
from log_setup import setup_logging
from train_dense import train_model as train_dense
from training import create_tensors, evaluate_model

logger = logging.getLogger(__name__)

def run_final():
    setup_logging(FINAL_LOG_PATH)
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    # The final models learn from train and dev together, with every setting already fixed, and the test set is not loaded
    rows = pd.concat([load_split("train"), load_split("dev")]).sort_index()
    x_rows, y_rows = create_tensors(rows, device)
    logger.info(f"Training on {len(rows)} rows from {rows.index.min()} to {rows.index.max()}")

    # With no data held out, the training rows fill the place of dev in the cost records
    dense, history = train_dense(DENSE_SETTINGS, x_rows, y_rows, x_rows, y_rows)
    linear, _ = train_linear(DIAGNOSIS_LINEAR_LAMBDA, x_rows, y_rows, x_rows, y_rows)

    # Errors on the rows each model learned from, which show how well it fits but are not a fair score
    results = {
        "model": "final",
        "rows": len(rows),
        "first_hour": str(rows.index.min()),
        "last_hour": str(rows.index.max()),
        "seed": RANDOM_SEED,
        "dense_settings": DENSE_SETTINGS,
        "linear_lambda": DIAGNOSIS_LINEAR_LAMBDA,
        "dense_training_fit": evaluate_model(dense, x_rows, rows),
        "linear_training_fit": evaluate_model(linear, x_rows, rows),
    }

    for name in ("dense", "linear"):
        errors = results[f"{name}_training_fit"]["all_hours"]
        logger.info(f"{name} fit on its own training rows: RMSE {errors['rmse']:.3f}, MAE {errors['mae']:.3f}")

    # Saved under its own name, since there is no dev cost to plot
    run_dir = create_run_dir("final")
    pd.DataFrame(history).drop(columns="dev_cost").to_csv(run_dir / "final_history.csv", index=False)
    torch.save(dense.state_dict(), run_dir / "dense.pt")
    torch.save(linear.state_dict(), run_dir / "linear.pt")
    save_results(run_dir, results)
    logger.info(f"Final models saved to {run_dir}")

if __name__ == "__main__":
    run_final()
