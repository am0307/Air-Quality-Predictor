import pandas as pd
import torch
from evaluation import evaluate_predictions

# Separates a split into input and target tensors
def create_tensors(rows, device):
    inputs = rows.drop(columns=["station", "target"]).to_numpy(dtype="float32")
    targets = rows["target"].to_numpy(dtype="float32")

    return torch.tensor(inputs, device=device), torch.tensor(targets, device=device)

# Squared error cost, J = 1/(2m) * sum((prediction - target)^2)
def calculate_cost(model, inputs, targets):
    predictions = model(inputs).squeeze(1)

    return ((predictions - targets) ** 2).mean() / 2

# Evaluates a trained model in µg/m³
def evaluate_model(model, inputs, rows):
    with torch.no_grad():
        predictions = model(inputs).squeeze(1).cpu().numpy()

    return evaluate_predictions(pd.Series(predictions, index=rows.index), rows["target"])
