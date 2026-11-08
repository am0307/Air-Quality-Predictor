import logging
import pandas as pd
import torch
from torch import nn
from config import NUM_LAG_HOURS, RANDOM_SEED, SEQUENCE_LOG_PATH, SEQUENCE_SETTINGS
from evaluation import create_run_dir, load_split, save_results
from log_setup import setup_logging
from training import create_tensors, evaluate_model, fit_model, get_feature_columns

logger = logging.getLogger(__name__)

# Reads the station's lags as a sequence, then joins the result with every other input before a ReLU layer
class SequenceModel(nn.Module):
    def __init__(self, feature_columns, cell, hidden_size, head_units):
        super().__init__()

        # The lags are read oldest to newest, one hour per step
        lag_indices = [feature_columns.index(f"pm25_lag_{hours}") for hours in range(NUM_LAG_HOURS, 0, -1)]

        # The latest reading also skips the recurrent layer, because tanh and sigmoid squash the very high
        # readings seen during smoke, which left the first version unable to even match persistence
        joined_indices = [i for i in range(len(feature_columns)) if i not in lag_indices]
        joined_indices.append(feature_columns.index("pm25_lag_1"))

        self.register_buffer("lag_indices", torch.tensor(lag_indices))
        self.register_buffer("joined_indices", torch.tensor(joined_indices))

        recurrent_layer = nn.GRU if cell == "gru" else nn.LSTM
        self.recurrent_layer = recurrent_layer(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size + len(joined_indices), head_units)
        self.output_layer = nn.Linear(head_units, 1)

        # He initialisation for the ReLU layers, as in the dense network
        for layer in (self.hidden_layer, self.output_layer):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        sequence = inputs[:, self.lag_indices].unsqueeze(-1)
        _, state = self.recurrent_layer(sequence)

        # An LSTM also returns its memory cell, but only the final hidden state is used
        final_hidden = state[0][-1] if isinstance(state, tuple) else state[-1]

        joined = torch.cat([final_hidden, inputs[:, self.joined_indices]], dim=1)

        return self.output_layer(torch.relu(self.hidden_layer(joined)))

# Builds and trains a sequence model with the given settings
def train_model(settings, feature_columns, x_train, y_train, x_dev, y_dev):
    model = SequenceModel(feature_columns, settings["cell"], settings["hidden_size"], settings["head_units"])
    model = model.to(x_train.device)

    return fit_model(model, settings, x_train, y_train, x_dev, y_dev)

def run_sequence():
    setup_logging(SEQUENCE_LOG_PATH)
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training a {SEQUENCE_SETTINGS['cell'].upper()} on {device}")

    # The test set is left out until the final evaluation
    train_rows = load_split("train")
    dev_rows = load_split("dev")
    x_train, y_train = create_tensors(train_rows, device)
    x_dev, y_dev = create_tensors(dev_rows, device)

    model, history = train_model(SEQUENCE_SETTINGS, get_feature_columns(train_rows), x_train, y_train, x_dev, y_dev)

    results = {
        "model": SEQUENCE_SETTINGS["cell"],
        "hidden_size": SEQUENCE_SETTINGS["hidden_size"],
        "head_units": SEQUENCE_SETTINGS["head_units"],
        "best_lambda": SEQUENCE_SETTINGS["lambda"],
        "learning_rate": SEQUENCE_SETTINGS["learning_rate"],
        "decay_rate": SEQUENCE_SETTINGS["decay_rate"],
        "batch_size": SEQUENCE_SETTINGS["batch_size"],
        "epochs": SEQUENCE_SETTINGS["epochs"],
        "max_gradient_norm": SEQUENCE_SETTINGS["max_gradient_norm"],
        "seed": RANDOM_SEED,
        "train": evaluate_model(model, x_train, train_rows),
        "dev": evaluate_model(model, x_dev, dev_rows),
    }

    for name in ("train", "dev"):
        for group, errors in results[name].items():
            logger.info(f"{name} {group}: RMSE {errors['rmse']:.3f}, MAE {errors['mae']:.3f}, {errors['rows']} rows")

    # Save the results
    run_dir = create_run_dir(SEQUENCE_SETTINGS["cell"])
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    save_results(run_dir, results)
    logger.info(f"Results saved to {run_dir}")

if __name__ == "__main__":
    run_sequence()
