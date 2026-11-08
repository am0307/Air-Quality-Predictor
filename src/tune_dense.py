import logging
import math
import random
import time
import pandas as pd
import torch
from config import (
    DENSE_SETTINGS, RANDOM_SEED, SEARCH_BATCH_OPTIONS, SEARCH_EPOCHS, SEARCH_LAMBDA_RANGE, SEARCH_LAYER_OPTIONS,
    SEARCH_LEARNING_RATE_RANGE, SEARCH_LOG_PATH, SEARCH_SEED, SEARCH_TRIALS, SEARCH_UNIT_OPTIONS,
)
from evaluation import create_run_dir, load_split, save_results
from log_setup import setup_logging
from train_dense import train_model
from training import create_tensors, evaluate_model

logger = logging.getLogger(__name__)

# Samples evenly across powers of 10, so a small range like 0.0001 to 0.001 is searched as much as 0.001 to 0.01
def sample_log_scale(generator, value_range):
    low, high = value_range

    return 10 ** generator.uniform(math.log10(low), math.log10(high))

# Samples the units in each hidden layer, where a layer is never wider than the one before it
def sample_hidden_units(generator):
    hidden_units = []
    options = SEARCH_UNIT_OPTIONS

    for _ in range(generator.choice(SEARCH_LAYER_OPTIONS)):
        hidden_units.append(generator.choice(options))
        options = [units for units in options if units <= hidden_units[-1]]

    return hidden_units

# Samples one set of random settings
def sample_settings(generator):
    return {
        "hidden_units": sample_hidden_units(generator),
        "learning_rate": sample_log_scale(generator, SEARCH_LEARNING_RATE_RANGE),
        "decay_rate": DENSE_SETTINGS["decay_rate"], # Not searched
        "batch_size": generator.choice(SEARCH_BATCH_OPTIONS),
        "epochs": SEARCH_EPOCHS,
        "lambda": sample_log_scale(generator, SEARCH_LAMBDA_RANGE),
    }

# Trains one trial and summarises it as a row of the trials table
def run_trial(trial, settings, splits, history_dir):
    x_train, y_train, train_rows, x_dev, y_dev, dev_rows = splits

    # Every trial uses the same seed, so differences come from the settings and not the starting weights
    torch.manual_seed(RANDOM_SEED)
    start_time = time.time()

    model, history = train_model(settings, x_train, y_train, x_dev, y_dev)
    pd.DataFrame(history).to_csv(history_dir / f"trial_{trial}.csv", index=False)

    train_errors = evaluate_model(model, x_train, train_rows)
    dev_errors = evaluate_model(model, x_dev, dev_rows)

    return {
        "trial": trial,
        "hidden_units": "-".join(str(units) for units in settings["hidden_units"]),
        "learning_rate": settings["learning_rate"],
        "decay_rate": settings["decay_rate"],
        "lambda": settings["lambda"],
        "batch_size": settings["batch_size"],
        "seed": RANDOM_SEED,
        "train_rmse": train_errors["all_hours"]["rmse"],
        "dev_rmse": dev_errors["all_hours"]["rmse"],
        "dev_smoke_rmse": dev_errors["smoke_hours"]["rmse"],
        "dev_non_smoke_rmse": dev_errors["non_smoke_hours"]["rmse"],
        "seconds": round(time.time() - start_time),
    }

def run_search():
    setup_logging(SEARCH_LOG_PATH)

    # Each trial's progress is saved to a file, so the per-epoch messages are not needed here
    logging.getLogger("training").setLevel(logging.WARNING)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    # The test set is left out until the final evaluation
    train_rows = load_split("train")
    dev_rows = load_split("dev")
    x_train, y_train = create_tensors(train_rows, device)
    x_dev, y_dev = create_tensors(dev_rows, device)
    splits = (x_train, y_train, train_rows, x_dev, y_dev, dev_rows)

    run_dir = create_run_dir("search")
    history_dir = run_dir / "histories"
    history_dir.mkdir()

    save_results(run_dir, {
        "model": "dense_search",
        "search_seed": SEARCH_SEED,
        "trials": SEARCH_TRIALS,
        "epochs": SEARCH_EPOCHS,
        "learning_rate_range": SEARCH_LEARNING_RATE_RANGE,
        "lambda_range": SEARCH_LAMBDA_RANGE,
        "layer_options": SEARCH_LAYER_OPTIONS,
        "unit_options": SEARCH_UNIT_OPTIONS,
        "batch_options": SEARCH_BATCH_OPTIONS,
    })

    # A separate generator keeps the sampled settings the same whatever happens during training
    generator = random.Random(SEARCH_SEED)
    trials = []

    for trial in range(1, SEARCH_TRIALS + 1):
        settings = sample_settings(generator)

        # One learning rate that is too high should not end the whole search
        try:
            row = run_trial(trial, settings, splits, history_dir)
        except ValueError as error:
            logger.warning(f"Trial {trial} failed with settings {settings}: {error}")
            continue

        trials.append(row)

        # Saved after every trial, so that a crash does not lose finished work
        pd.DataFrame(trials).to_csv(run_dir / "trials.csv", index=False)
        logger.info(f"Trial {trial}/{SEARCH_TRIALS}: dev RMSE {row['dev_rmse']:.3f}, settings {settings}")

    best_trials = pd.DataFrame(trials).sort_values("dev_rmse").head(5)
    logger.info(f"Best trials:\n{best_trials.round(4).to_string(index=False)}")
    logger.info(f"Results saved to {run_dir}")

if __name__ == "__main__":
    run_search()
