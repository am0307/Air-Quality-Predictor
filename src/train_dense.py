import logging
import math
import pandas as pd
import torch
from torch import nn
from config import DENSE_LOG_PATH, DENSE_SETTINGS, RANDOM_SEED
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
def build_model(num_inputs, hidden_units):
    layers = []

    for num_units in hidden_units:
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
def record_costs(model, settings, epoch, x_train, y_train, x_dev, y_dev):
    with torch.no_grad():
        train_cost = calculate_cost(model, x_train, y_train)
        dev_cost = calculate_cost(model, x_dev, y_dev)
        penalty = calculate_penalty(model, settings["lambda"], len(y_train))

    # Each epoch takes one gradient step per mini-batch
    batches_per_epoch = math.ceil(len(y_train) / settings["batch_size"])

    return {
        "lambda": settings["lambda"],
        "epoch": epoch,
        "iteration": epoch * batches_per_epoch,
        "regularised_cost": (train_cost + penalty).item(),
        "train_cost": train_cost.item(),
        "dev_cost": dev_cost.item(),
    }

# Trains with mini-batch gradient descent, Adam, and learning rate decay, keeping the final weights
def train_model(settings, x_train, y_train, x_dev, y_dev):
    model = build_model(x_train.shape[1], settings["hidden_units"]).to(x_train.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])

    # The learning rate after each epoch is alpha = alpha_0 / (1 + decay_rate * epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda epoch: 1 / (1 + settings["decay_rate"] * epoch))

    history = [record_costs(model, settings, 0, x_train, y_train, x_dev, y_dev)]

    for epoch in range(1, settings["epochs"] + 1):

        # Every row carries its own past readings, so rows can be shuffled without mixing up time
        shuffled = torch.randperm(len(y_train), device=x_train.device)

        for batch in torch.split(shuffled, settings["batch_size"]):
            cost = calculate_cost(model, x_train[batch], y_train[batch])
            cost = cost + calculate_penalty(model, settings["lambda"], len(y_train))

            optimiser.zero_grad()
            cost.backward()
            optimiser.step()

        scheduler.step()

        costs = record_costs(model, settings, epoch, x_train, y_train, x_dev, y_dev)
        history.append(costs)

        # A learning rate that is too high makes the costs blow up instead of falling
        if not math.isfinite(costs["regularised_cost"]):
            raise ValueError(f"Training diverged at epoch {epoch}")

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train cost {costs['train_cost']:.4f}, dev cost {costs['dev_cost']:.4f}")

    return model, history

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

    model, history = train_model(DENSE_SETTINGS, x_train, y_train, x_dev, y_dev)

    results = {
        "model": "dense",
        "hidden_units": DENSE_SETTINGS["hidden_units"],
        "best_lambda": DENSE_SETTINGS["lambda"],
        "learning_rate": DENSE_SETTINGS["learning_rate"],
        "decay_rate": DENSE_SETTINGS["decay_rate"],
        "batch_size": DENSE_SETTINGS["batch_size"],
        "epochs": DENSE_SETTINGS["epochs"],
        "seed": RANDOM_SEED,
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
