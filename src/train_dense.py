import logging
import pandas as pd
import torch
from torch import nn
from config import DENSE_LOG_PATH, DENSE_SETTINGS, RANDOM_SEED
from evaluation import create_run_dir, load_split, save_results
from log_setup import setup_logging
from training import create_tensors, evaluate_model, fit_model

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

# Builds and trains a dense network with the given settings
def train_model(settings, x_train, y_train, x_dev, y_dev):
    model = build_model(x_train.shape[1], settings["hidden_units"]).to(x_train.device)

    return fit_model(model, settings, x_train, y_train, x_dev, y_dev)

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
