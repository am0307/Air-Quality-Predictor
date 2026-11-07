import logging
import pandas as pd
import torch
from baseline_linear import train_model as train_linear
from config import (
    DENSE_SETTINGS, DIAGNOSIS_LINEAR_LAMBDA, DIAGNOSIS_LOG_PATH, DIAGNOSIS_SEEDS, RANDOM_SEED, SMOKE_MIN_PM25,
    TRAIN_DEV_GAP_HOURS, TRAIN_DEV_MIN_SMOKE_ROWS, TRAIN_DEV_WEEK_INTERVAL, TRAIN_DEV_WEEK_OFFSET,
)
from evaluation import create_run_dir, load_split, save_results
from log_setup import setup_logging
from train_dense import train_model as train_dense
from training import create_tensors, evaluate_model

logger = logging.getLogger(__name__)

# Marks the hours that fall in a held-out week
def is_held_out(hours, first_hour):
    week_numbers = (hours - first_hour) // pd.Timedelta(weeks=1)

    return week_numbers % TRAIN_DEV_WEEK_INTERVAL == TRAIN_DEV_WEEK_OFFSET

# Separates the training rows into a smaller training set and a train-dev set
def split_train_dev(rows):
    first_hour = rows.index.min()
    gap = pd.Timedelta(hours=TRAIN_DEV_GAP_HOURS)

    train_dev_mask = is_held_out(rows.index, first_hour)

    # An hour within the gap of a held-out week is held out once shifted by the gap in either direction
    near_mask = is_held_out(rows.index - gap, first_hour) | is_held_out(rows.index + gap, first_hour)

    return rows[~train_dev_mask & ~near_mask], rows[train_dev_mask]

# Stops the diagnosis if the sets overlap or the train-dev set has too little smoke to judge
def check_train_dev(train, train_dev):
    if train.index.intersection(train_dev.index).size > 0:
        raise ValueError("The train and train-dev sets share hours")

    smoke_rows = (train_dev["target"] >= SMOKE_MIN_PM25).sum()
    if smoke_rows < TRAIN_DEV_MIN_SMOKE_ROWS:
        raise ValueError(f"The train-dev set has only {smoke_rows} smoke rows")

# Creates one results row for each split and group of hours
def create_rows(model_name, seed, model, splits):
    rows = []

    for split_name, (inputs, split_rows) in splits.items():
        for group, errors in evaluate_model(model, inputs, split_rows).items():
            rows.append({"model": model_name, "seed": seed, "split": split_name, "group": group, **errors})

    return rows

def run_diagnosis():
    setup_logging(DIAGNOSIS_LOG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    # The test set is left out until the final evaluation
    train_rows, train_dev_rows = split_train_dev(load_split("train"))
    dev_rows = load_split("dev")
    check_train_dev(train_rows, train_dev_rows)
    logger.info(f"Training on {len(train_rows)} rows, holding out {len(train_dev_rows)} rows")

    x_train, y_train = create_tensors(train_rows, device)
    x_train_dev, _ = create_tensors(train_dev_rows, device)
    x_dev, y_dev = create_tensors(dev_rows, device)

    splits = {
        "train": (x_train, train_rows),
        "train_dev": (x_train_dev, train_dev_rows),
        "dev": (x_dev, dev_rows),
    }

    # Linear regression starts from zero weights without shuffling, so one seed is enough
    model, _ = train_linear(DIAGNOSIS_LINEAR_LAMBDA, x_train, y_train, x_dev, y_dev)
    results = create_rows("linear", RANDOM_SEED, model, splits)

    for seed in DIAGNOSIS_SEEDS:
        torch.manual_seed(seed)
        model, _, _ = train_dense(DENSE_SETTINGS, x_train, y_train, x_dev, y_dev)
        results.extend(create_rows("dense", seed, model, splits))

    # Average over the seeds and put the splits in the order that they are compared
    results = pd.DataFrame(results)
    summary = results.groupby(["model", "group", "split"])["rmse"].mean().unstack()[["train", "train_dev", "dev"]]
    summary["train_to_train_dev"] = summary["train_dev"] - summary["train"]
    summary["train_dev_to_dev"] = summary["dev"] - summary["train_dev"]
    logger.info(f"Mean RMSE by split:\n{summary.round(3).to_string()}")

    # Save the results
    run_dir = create_run_dir("diagnosis")
    results.to_csv(run_dir / "results.csv", index=False)
    summary.to_csv(run_dir / "summary.csv")
    save_results(run_dir, {
        "model": "diagnosis",
        "week_interval": TRAIN_DEV_WEEK_INTERVAL,
        "week_offset": TRAIN_DEV_WEEK_OFFSET,
        "gap_hours": TRAIN_DEV_GAP_HOURS,
        "seeds": DIAGNOSIS_SEEDS,
        "train_rows": len(train_rows),
        "train_dev_rows": len(train_dev_rows),
    })
    logger.info(f"Results saved to {run_dir}")

if __name__ == "__main__":
    run_diagnosis()
