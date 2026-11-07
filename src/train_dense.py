import copy
import logging
import math
import pandas as pd
import torch
from torch import nn
from config import (
    DENSE_BATCH_SIZE, DENSE_EPOCHS, DENSE_HIDDEN_UNITS, DENSE_LAMBDA, DENSE_LEARNING_RATE, DENSE_LOG_PATH,
    RANDOM_SEED,
)
from evaluation import create_run_dir, load_split, save_results
from log_setup import setup_logging
from training import calculate_cost, create_tensors, evaluate_model

logger = logging.getLogger(__name__)

# Creates a layer with He initialisation, which suits ReLU activations
def create_layer(num_inputs, num_units):
    layer = nn.Linear(num_inputs, num_units)
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)

    return layer

# Builds ReLU hidden layers and one output unit, which has no activation so it can predict any value
def build_model(num_inputs):
    layers = []

    for num_units in DENSE_HIDDEN_UNITS:
        layers.append(create_layer(num_inputs, num_units))
        layers.append(nn.ReLU())
        num_inputs = num_units

    layers.append(create_layer(num_inputs, 1))

    return nn.Sequential(*layers)

# The L2 penalty, lambda/(2m) * sum(w^2), over each layer's weights but not its bias
def calculate_penalty(model, lambd, m):
    total = sum((layer.weight ** 2).sum() for layer in model if isinstance(layer, nn.Linear))

    return lambd / (2 * m) * total

# Records the costs after one epoch, where train and dev costs leave out the penalty so they can be compared
def record_costs(model, lambd, epoch, x_train, y_train, x_dev, y_dev):
    with torch.no_grad():
        train_cost = calculate_cost(model, x_train, y_train)
        dev_cost = calculate_cost(model, x_dev, y_dev)
        penalty = calculate_penalty(model, lambd, len(y_train))

    # Each epoch takes one gradient step per mini-batch
    batches_per_epoch = math.ceil(len(y_train) / DENSE_BATCH_SIZE)

    return {
        "lambda": lambd,
        "epoch": epoch,
        "iteration": epoch * batches_per_epoch,
        "regularised_cost": (train_cost + penalty).item(),
        "train_cost": train_cost.item(),
        "dev_cost": dev_cost.item(),
    }

# Trains with mini-batch gradient descent and Adam, keeping the weights from the epoch with the lowest dev cost
def train_model(lambd, x_train, y_train, x_dev, y_dev):
    model = build_model(x_train.shape[1]).to(x_train.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=DENSE_LEARNING_RATE)

    history = [record_costs(model, lambd, 0, x_train, y_train, x_dev, y_dev)]
    best_costs = history[0]
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, DENSE_EPOCHS + 1):

        # Every row carries its own past readings, so rows can be shuffled without mixing up time
        shuffled = torch.randperm(len(y_train), device=x_train.device)

        for batch in torch.split(shuffled, DENSE_BATCH_SIZE):
            cost = calculate_cost(model, x_train[batch], y_train[batch])
            cost = cost + calculate_penalty(model, lambd, len(y_train))

            optimiser.zero_grad()
            cost.backward()
            optimiser.step()

        costs = record_costs(model, lambd, epoch, x_train, y_train, x_dev, y_dev)
        history.append(costs)

        # A learning rate that is too high makes the costs blow up instead of falling
        if not math.isfinite(costs["regularised_cost"]):
            raise ValueError(f"Training diverged at epoch {epoch}")

        if costs["dev_cost"] < best_costs["dev_cost"]:
            best_costs = costs
            best_weights = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train cost {costs['train_cost']:.4f}, dev cost {costs['dev_cost']:.4f}")

    model.load_state_dict(best_weights)

    return model, history, best_costs

def run_dense():
    setup_logging(DENSE_LOG_PATH)
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    # The test set is left out until the final evaluation
    train_rows = load_split("train")
    dev_rows = load_split("dev")
    x_train, y_train = create_tensors(train_rows, device)
    x_dev, y_dev = create_tensors(dev_rows, device)

    model, history, best_costs = train_model(DENSE_LAMBDA, x_train, y_train, x_dev, y_dev)
    logger.info(f"Best epoch: {best_costs['epoch']}, dev cost {best_costs['dev_cost']:.4f}")

    results = {
        "model": "dense",
        "hidden_units": DENSE_HIDDEN_UNITS,
        "best_lambda": DENSE_LAMBDA,
        "learning_rate": DENSE_LEARNING_RATE,
        "batch_size": DENSE_BATCH_SIZE,
        "epochs": DENSE_EPOCHS,
        "seed": RANDOM_SEED,
        "best_epoch": best_costs["epoch"],
        "train": evaluate_model(model, x_train, train_rows),
        "dev": evaluate_model(model, x_dev, dev_rows),
    }

    for name in ("train", "dev"):
        for group, errors in results[name].items():
            logger.info(f"{name} {group}: RMSE {errors['rmse']:.3f}, MAE {errors['mae']:.3f}, {errors['rows']} rows")

    # Save the results
    run_dir = create_run_dir("dense")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    save_results(run_dir, results)
    logger.info(f"Results saved to {run_dir}")

if __name__ == "__main__":
    run_dense()
