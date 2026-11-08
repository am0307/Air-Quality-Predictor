import json
import logging
import pandas as pd
import torch
from baseline_linear import train_model as train_linear
from baseline_persistence import predict_persistence
from config import (
    COMPARISON_CELLS, COMPARISON_DEV_SEEDS, COMPARISON_HIDDEN_SIZES, COMPARISON_LOG_PATH, COMPARISON_MIN_IMPROVEMENT,
    COMPARISON_MIN_SEEDS_BETTER, COMPARISON_SELECTION_SEEDS, DENSE_SETTINGS, DIAGNOSIS_LINEAR_LAMBDA,
    NORMALISATION_PATH, SEQUENCE_SETTINGS,
)
from diagnose_train_dev import split_train_dev
from evaluation import create_run_dir, evaluate_predictions, load_split, save_results
from log_setup import setup_logging
from train_dense import train_model as train_dense
from train_sequence import train_model as train_sequence
from training import create_tensors, evaluate_model, get_feature_columns

logger = logging.getLogger(__name__)

# Flattens the errors for one model and split into a single row
def create_row(errors, **labels):
    return {
        **labels,
        "rmse": errors["all_hours"]["rmse"],
        "mae": errors["all_hours"]["mae"],
        "smoke_rmse": errors["smoke_hours"]["rmse"],
        "non_smoke_rmse": errors["non_smoke_hours"]["rmse"],
    }

# Chooses each cell's hidden size by mean RMSE on the train-dev weeks, so dev plays no part in the choice
def choose_hidden_sizes(data):
    feature_columns, x_train, y_train, x_train_dev, y_train_dev, train_dev_rows = data
    rows = []

    for cell in COMPARISON_CELLS:
        for hidden_size in COMPARISON_HIDDEN_SIZES:
            settings = {**SEQUENCE_SETTINGS, "cell": cell, "hidden_size": hidden_size}

            for seed in COMPARISON_SELECTION_SEEDS:
                torch.manual_seed(seed)
                model, _ = train_sequence(settings, feature_columns, x_train, y_train, x_train_dev, y_train_dev)
                errors = evaluate_model(model, x_train_dev, train_dev_rows)
                rows.append(create_row(errors, cell=cell, hidden_size=hidden_size, seed=seed))
                logger.info(f"{cell} {hidden_size} seed {seed}: train-dev RMSE {rows[-1]['rmse']:.3f}")

    selection = pd.DataFrame(rows)
    mean_rmse = selection.groupby(["cell", "hidden_size"])["rmse"].mean()
    chosen = {cell: int(mean_rmse[cell].idxmin()) for cell in COMPARISON_CELLS}

    return selection, chosen

# Applies the rule fixed before the comparison: a sequence model must clearly beat dense on most seeds
def decide(comparison, name):
    per_seed = comparison.pivot(index="seed", columns="model", values="rmse")
    improvement = per_seed["dense"] - per_seed[name]
    replaces = improvement.mean() >= COMPARISON_MIN_IMPROVEMENT and (improvement > 0).sum() >= COMPARISON_MIN_SEEDS_BETTER

    return {
        "mean_improvement": float(improvement.mean()),
        "seeds_better": int((improvement > 0).sum()),
        "replaces_dense": bool(replaces),
    }

def run_comparison():
    setup_logging(COMPARISON_LOG_PATH)

    # Each epoch's progress is not needed for the many models trained here
    logging.getLogger("training").setLevel(logging.WARNING)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    # Every model trains on the same rows, leaving out the train-dev weeks, and the test set is not used
    train_rows, train_dev_rows = split_train_dev(load_split("train"))
    dev_rows = load_split("dev")
    feature_columns = get_feature_columns(train_rows)
    x_train, y_train = create_tensors(train_rows, device)
    x_train_dev, y_train_dev = create_tensors(train_dev_rows, device)
    x_dev, y_dev = create_tensors(dev_rows, device)

    selection, chosen = choose_hidden_sizes((feature_columns, x_train, y_train, x_train_dev, y_train_dev, train_dev_rows))
    logger.info(f"Hidden sizes chosen on the train-dev weeks: {chosen}")

    # The baselines need no seeds
    with open(NORMALISATION_PATH, encoding="utf-8") as f:
        normalisation_values = json.load(f)

    persistence = evaluate_predictions(predict_persistence(dev_rows, normalisation_values), dev_rows["target"])
    rows = [create_row(persistence, model="persistence", seed=0)]

    linear, _ = train_linear(DIAGNOSIS_LINEAR_LAMBDA, x_train, y_train, x_dev, y_dev)
    rows.append(create_row(evaluate_model(linear, x_dev, dev_rows), model="linear", seed=0))

    # One comparison on dev, with every network trained from the same seeds
    for seed in COMPARISON_DEV_SEEDS:
        torch.manual_seed(seed)
        model, _ = train_dense(DENSE_SETTINGS, x_train, y_train, x_dev, y_dev)
        rows.append(create_row(evaluate_model(model, x_dev, dev_rows), model="dense", seed=seed))

        for cell, hidden_size in chosen.items():
            settings = {**SEQUENCE_SETTINGS, "cell": cell, "hidden_size": hidden_size}
            torch.manual_seed(seed)
            model, _ = train_sequence(settings, feature_columns, x_train, y_train, x_dev, y_dev)
            rows.append(create_row(evaluate_model(model, x_dev, dev_rows), model=cell, seed=seed))

        logger.info(f"Seed {seed} finished")

    comparison = pd.DataFrame(rows)
    summary = comparison.groupby("model", sort=False)[["rmse", "mae", "smoke_rmse", "non_smoke_rmse"]].mean()
    logger.info(f"Mean dev errors:\n{summary.round(3).to_string()}")

    decisions = {cell: decide(comparison, cell) for cell in chosen}
    for cell, decision in decisions.items():
        verdict = "replaces dense" if decision["replaces_dense"] else "does not replace dense"
        logger.info(f"{cell}: mean improvement {decision['mean_improvement']:.4f}, "
                    f"{decision['seeds_better']} of {len(COMPARISON_DEV_SEEDS)} seeds better, {verdict}")

    # Save the results
    run_dir = create_run_dir("comparison")
    selection.to_csv(run_dir / "selection.csv", index=False)
    comparison.to_csv(run_dir / "comparison.csv", index=False)
    summary.to_csv(run_dir / "summary.csv")
    save_results(run_dir, {"model": "comparison", "chosen_hidden_sizes": chosen, "decisions": decisions})
    logger.info(f"Results saved to {run_dir}")

if __name__ == "__main__":
    run_comparison()
