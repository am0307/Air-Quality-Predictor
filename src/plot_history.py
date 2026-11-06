import json
import logging
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
from config import PLOT_LOG_PATH, RUNS_DIR
from log_setup import setup_logging

# Colours of the lines, labels, and axes
TRAIN_COLOUR = "#2a78d6"
DEV_COLOUR = "#eb6834"
BACKGROUND_COLOUR = "#fcfcfb"
TEXT_COLOUR = "#0b0b0b"
LABEL_COLOUR = "#52514e"
GRID_COLOUR = "#e1e0d9"
AXIS_COLOUR = "#c3c2b7"

logger = logging.getLogger(__name__)

# Finds the most recent run that saved a cost history
def find_latest_run():
    history_paths = list(RUNS_DIR.glob("*/history.csv"))

    if not history_paths:
        raise FileNotFoundError(f"No cost history found in {RUNS_DIR}, run a training script")

    return max(history_paths, key=lambda path: path.stat().st_mtime).parent

# Draws the train and dev cost curves for one lambda, labelling where each one finishes
def plot_costs(axis, history, title):
    axis.plot(history["iteration"], history["train_cost"], color=TRAIN_COLOUR, linewidth=1.5, label="Train cost")
    axis.plot(history["iteration"], history["dev_cost"], color=DEV_COLOUR, linewidth=1.5, label="Dev cost")

    # Each label sits just above the end of its line
    final = history.iloc[-1]
    axis.annotate(f"Dev {final['dev_cost']:.2f}", (final["iteration"], final["dev_cost"]), xytext=(0, 4),
                  textcoords="offset points", ha="right", va="bottom", color=LABEL_COLOUR, fontsize=9)
    axis.annotate(f"Train {final['train_cost']:.2f}", (final["iteration"], final["train_cost"]), xytext=(0, 4),
                  textcoords="offset points", ha="right", va="bottom", color=LABEL_COLOUR, fontsize=9)

    axis.set_title(title, color=TEXT_COLOUR, fontsize=11, loc="left")
    axis.set_xlabel("Iteration", color=LABEL_COLOUR, fontsize=9)
    axis.set_facecolor(BACKGROUND_COLOUR)
    axis.grid(axis="y", color=GRID_COLOUR, linewidth=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(colors=LABEL_COLOUR, labelsize=9, length=0)
    axis.set_xticks([0, final["iteration"] / 2, final["iteration"]]) # Few enough that panels don't crowd
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(AXIS_COLOUR)

# Plots one panel per lambda on a shared axis, so that the panels can be compared directly
def plot_run(history, best_lambda, run_name):
    lambdas = history["lambda"].unique()
    figure, axes = plt.subplots(1, len(lambdas), figsize=(3 * len(lambdas), 3.8), sharey=True, squeeze=False)
    figure.set_facecolor(BACKGROUND_COLOUR)

    for axis, lambd in zip(axes[0], lambdas):
        title = f"λ = {lambd:g} (chosen)" if lambd == best_lambda else f"λ = {lambd:g}"
        plot_costs(axis, history[history["lambda"] == lambd], title)

    # The cost starts far above where it ends, so a log scale keeps the final gap visible
    first_axis = axes[0][0]
    first_axis.set_yscale("log")
    first_axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    first_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    first_axis.yaxis.set_minor_locator(NullLocator())
    first_axis.set_ylabel("Cost J (log scale)", color=LABEL_COLOUR, fontsize=9)

    handles, labels = first_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", ncols=2, frameon=False, fontsize=9, labelcolor=LABEL_COLOUR)
    figure.suptitle(f"Cost by iteration: {run_name}", x=0.01, ha="left", color=TEXT_COLOUR, fontsize=12)
    figure.tight_layout()

    return figure

def run_plot():
    setup_logging(PLOT_LOG_PATH)

    # A run folder can be given, otherwise the most recent one is used
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_run()

    history = pd.read_csv(run_dir / "history.csv")
    with open(run_dir / "results.json", encoding="utf-8") as f:
        results = json.load(f)

    figure = plot_run(history, results["best_lambda"], run_dir.name)

    plot_path = run_dir / "cost_curves.png"
    figure.savefig(plot_path, dpi=150, facecolor=BACKGROUND_COLOUR)
    plt.close(figure)
    logger.info(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    run_plot()
