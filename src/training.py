import logging
import math
import pandas as pd
import torch
from evaluation import evaluate_predictions

logger = logging.getLogger(__name__)

# Lists the input columns of a split, which are every column except the station name and target
def get_feature_columns(rows):
    return list(rows.drop(columns=["station", "target"]).columns)

# Separates a split into input and target tensors
def create_tensors(rows, device):
    inputs = rows[get_feature_columns(rows)].to_numpy(dtype="float32")
    targets = rows["target"].to_numpy(dtype="float32")

    return torch.tensor(inputs, device=device), torch.tensor(targets, device=device)

# Squared error cost, J = 1/(2m) * sum((prediction - target)^2)
def calculate_cost(model, inputs, targets):
    predictions = model(inputs).squeeze(1)

    return ((predictions - targets) ** 2).mean() / 2

# The L2 penalty, lambda/(2m) * sum(w^2), over every weight matrix but not the biases
def calculate_penalty(model, lambd, m):
    total = sum((parameter ** 2).sum() for name, parameter in model.named_parameters() if "weight" in name)

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
def fit_model(model, settings, x_train, y_train, x_dev, y_dev):
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

            # Gradient clipping limits the rare very large steps that recurrent networks can take
            if settings.get("max_gradient_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_gradient_norm"])

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

# Evaluates a trained model in µg/m³
def evaluate_model(model, inputs, rows):
    with torch.no_grad():
        predictions = model(inputs).squeeze(1).cpu().numpy()

    return evaluate_predictions(pd.Series(predictions, index=rows.index), rows["target"])
