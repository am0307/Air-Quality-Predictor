import logging
import math
import pandas as pd
import torch
from torch import nn
from config import (
    COST_RECORD_INTERVAL, LINEAR_ITERATIONS, LINEAR_LAMBDAS, LINEAR_LEARNING_RATE, LINEAR_LOG_PATH,
)
from evaluation import create_run_dir, load_split, save_results
from log_setup import setup_logging
from training import calculate_cost, create_tensors, evaluate_model

logger = logging.getLogger(__name__)

# The L2 penalty, lambda/(2m) * sum(w^2), which leaves the bias out as in the course
def calculate_penalty(model, lambd, m):
    return lambd / (2 * m) * (model.weight ** 2).sum()

# Records the costs at one iteration, where train and dev costs leave out the penalty so they can be compared
def record_costs(model, lambd, iteration, x_train, y_train, x_dev, y_dev):
    with torch.no_grad():
        train_cost = calculate_cost(model, x_train, y_train)
        dev_cost = calculate_cost(model, x_dev, y_dev)
        penalty = calculate_penalty(model, lambd, len(y_train))

    return {
        "lambda": lambd,
        "iteration": iteration,
        "regularised_cost": (train_cost + penalty).item(),
        "train_cost": train_cost.item(),
        "dev_cost": dev_cost.item(),
    }

# Trains one model with batch gradient descent, starting from zero weights
def train_model(lambd, x_train, y_train, x_dev, y_dev):
    model = nn.Linear(x_train.shape[1], 1).to(x_train.device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    optimiser = torch.optim.SGD(model.parameters(), lr=LINEAR_LEARNING_RATE)
    history = [record_costs(model, lambd, 0, x_train, y_train, x_dev, y_dev)]

    for iteration in range(1, LINEAR_ITERATIONS + 1):
        cost = calculate_cost(model, x_train, y_train) + calculate_penalty(model, lambd, len(y_train))

        optimiser.zero_grad()
        cost.backward()
        optimiser.step()

        if iteration % COST_RECORD_INTERVAL == 0:
            history.append(record_costs(model, lambd, iteration, x_train, y_train, x_dev, y_dev))

    # A learning rate that is too high makes the costs blow up instead of falling
    if not math.isfinite(history[-1]["regularised_cost"]):
        raise ValueError(f"Gradient descent diverged with lambda {lambd}")

    return model, history

def run_linear():
    setup_logging(LINEAR_LOG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    # The test set is left out until the final evaluation
    train_rows = load_split("train")
    dev_rows = load_split("dev")
    x_train, y_train = create_tensors(train_rows, device)
    x_dev, y_dev = create_tensors(dev_rows, device)

    # Train one model for each lambda and keep the one with the lowest dev cost
    history = []
    best_model = None
    best_costs = None

    for lambd in LINEAR_LAMBDAS:
        model, model_history = train_model(lambd, x_train, y_train, x_dev, y_dev)
        history.extend(model_history)

        final_costs = model_history[-1]
        logger.info(f"Lambda {lambd}: train cost {final_costs['train_cost']:.4f}, dev cost {final_costs['dev_cost']:.4f}")

        if best_costs is None or final_costs["dev_cost"] < best_costs["dev_cost"]:
            best_model = model
            best_costs = final_costs

    logger.info(f"Best lambda: {best_costs['lambda']}")

    results = {
        "model": "linear",
        "lambdas": LINEAR_LAMBDAS,
        "best_lambda": best_costs["lambda"],
        "learning_rate": LINEAR_LEARNING_RATE,
        "iterations": LINEAR_ITERATIONS,
        "train": evaluate_model(best_model, x_train, train_rows),
        "dev": evaluate_model(best_model, x_dev, dev_rows),
    }

    for name in ("train", "dev"):
        for group, errors in results[name].items():
            logger.info(f"{name} {group}: RMSE {errors['rmse']:.3f}, MAE {errors['mae']:.3f}, {errors['rows']} rows")

    # Save the results
    run_dir = create_run_dir("linear")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    save_results(run_dir, results)
    logger.info(f"Results saved to {run_dir}")

if __name__ == "__main__":
    run_linear()
