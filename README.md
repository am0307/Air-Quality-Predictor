# AQ Forecaster

Forecasts fine particulate matter (PM2.5) one hour ahead at four Toronto air quality stations, using hourly readings from OpenAQ and wind from Toronto Pearson airport. The final model is a dense neural network built with PyTorch. On held-out data from March to November 2026, its root mean squared error (RMSE) was 14.6% lower than persistence, which predicts that the next hour will equal the latest reading.

## Results

These results are for the test period, March to November 2026, which no modelling decision used. Errors are in µg/m³. Each gain is the percentage reduction in RMSE compared with persistence.

| Hours | Rows | Persistence RMSE | Dense RMSE (gain) | Linear RMSE (gain) |
|---|---|---|---|---|
| All | 23,340 | 3.64 | 3.11 (14.6%) | 3.04 (16.7%) |
| Smoke (35 µg/m³ or more) | 372 | 22.92 | 18.98 (17.2%) | 18.23 (20.5%) |
| Not smoke | 22,968 | 2.23 | 2.00 (10.4%) | 2.00 (10.5%) |
| Sudden rises (10 µg/m³ or more in an hour) | 164 | 27.04 | 19.57 (27.6%) | 22.65 (16.2%) |
| Extreme (100 µg/m³ or more) | 109 | 32.29 | 24.55 (24.0%) | 24.61 (23.8%) |

The test period had far more extreme wildfire smoke than earlier years, with 109 hours at or above 100 µg/m³. Persistence does worst in those hours, so gains are larger here than in the earlier development data, where dense improved on persistence by about 7%.

Linear regression scored similarly overall. The dense network was chosen before the test evaluation, and it handled sudden rises much better than linear regression.

In typical hours, both models are often within 1–2 µg/m³ of the reading. Their mean absolute error (MAE) is slightly worse than persistence in non-smoke hours, because readings are whole numbers and often stay the same from one hour to the next.

## Quick start

You need:

- Python 3.13
- An NVIDIA GPU with CUDA. `requirements.txt` installs the CUDA 13.0 build of PyTorch from PyTorch's own package index. For CPU only, install the standard `torch` package instead.
- A free API key from [OpenAQ](https://openaq.org/)

Set up the project from its root folder on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a `.env` file in the project root containing your OpenAQ API key:

```
OPENAQ_API_KEY=your_key_here
```

Run the pipeline below once to download the data and train the final model. After that, predict the coming hour with:

```powershell
.\.venv\Scripts\python.exe src\predict.py
```

```
Predicted PM2.5 (µg/m³) for 2026-11-07 19:00 to 20:00 Toronto time:
 station  newest_reading  prediction
downtown             8.0         7.5
   north             2.0         2.2
    west             2.0         2.5
    east             5.0         4.7
```

## Running the pipeline

Run these scripts from the project root, in order. Each stage reads the files written by the previous one, so if you change a stage, rerun every stage after it. Run times are for an RTX 3080.

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `src\fetch_data.py` | Downloads hourly PM2.5 for each station and year | Several minutes |
| 2 | `src\fetch_weather.py` | Downloads Pearson's hourly weather for each month | About 1 minute |
| 3 | `src\clean_data.py` | Removes invalid readings and records why each was removed | Seconds |
| 4 | `src\build_features.py` | Builds the training, development and test sets | Seconds |
| 5 | `src\baseline_persistence.py` and `src\baseline_linear.py` | Scores the two baselines on development data | About 1 minute |
| 6 | `src\train_dense.py` | Trains the dense network and scores it on development data | About 30 seconds |
| 7 | `src\train_final.py` | Trains the final models on the training and development data together | About 1 minute |
| 8 | `src\evaluate_test.py` | Scores the final models on the test set | Seconds |
| 9 | `src\predict.py` | Predicts the coming hour from live data | Seconds |

Run step 8 only once, after every modelling choice is final. Changing the model after seeing test results makes those results unreliable.

Each command runs with the project's Python, for example `.\.venv\Scripts\python.exe src\fetch_data.py`.

These optional scripts were used to make modelling decisions:

| Command | What it does |
|---|---|
| `src\tune_dense.py` | Runs a random search over the dense network's settings |
| `src\diagnose_train_dev.py` | Holds out some training weeks to separate overfitting from differences between periods |
| `src\analyse_errors.py` | Breaks development errors down by month, station and reading level |
| `src\compare_dense_gru_lstm.py` | Compares dense, GRU and LSTM networks on equal terms (about 30 minutes) |
| `src\plot_history.py` | Plots the cost curves for the latest run, or for a run folder given as an argument |

## How it works

**Data.** The model uses hourly PM2.5 from four long-running Toronto stations (downtown, north, west and east) from May 2023 onwards, and hourly wind from Toronto Pearson.

**Cleaning.** Readings are removed if they are badly formatted, flagged by the provider, duplicated, or outside 0–1,000 µg/m³. A sensor reporting the same value for 48 hours or more counts as stuck. A reading counts as a spike if it is at least 50 µg/m³, more than 3 times what any other station saw within 3 hours, and at least two other stations were reporting. Removed readings are kept in `data/clean/removed_rows.csv` along with the reason for removal.

**Features.** Each row predicts one station's next hour and has 84 inputs:

- That station's readings from the last 24 hours
- Two exponentially weighted averages of its readings, covering roughly the last 10 and 50 hours
- Each other station's readings from the last 3 hours
- Pearson's wind from the last 3 hours, split into components so that directions such as 350° and 10° stay close together
- The hour, day of the week and station

Short gaps of up to 3 hours are filled with the last reading, but only in the inputs. The model is always scored against real readings.

**Split.** Rows are split by time into three sets. Each set contains wildfire smoke.

| Set | Period |
|---|---|
| Training | May 2023 to June 2025 |
| Development | July 2025 to February 2026 |
| Test | March to November 2026 |

**Model.** The final model is a dense network with ReLU hidden layers of 256, 128 and 128 units. It is trained with Adam, mini-batches of 2,048, L2 regularisation and learning rate decay over 300 epochs. Its settings came from a random search on the development data. The final version is trained on the training and development data together.

## Outputs

Everything the pipeline creates goes in `data/` or `runs/`. Both folders are excluded from Git and can be rebuilt by rerunning the pipeline.

### Data

| Path | Created by | What it contains |
|---|---|---|
| `data/raw/{station}_{year}.json` | `fetch_data.py` | Hourly PM2.5 readings for one station and year, exactly as the OpenAQ API returned them. A year is downloaded again until a download made after that year ended has been saved. |
| `data/raw/weather/pearson_{year}_{month}.csv` | `fetch_weather.py` | Pearson's hourly weather observations for one month. A month is downloaded again until about two days after it ends, once the archive is complete. |
| `data/clean/pm25_hourly.csv` | `clean_data.py` | The readings that passed every cleaning rule, with one row per station and hour |
| `data/clean/removed_rows.csv` | `clean_data.py` | Every removed reading, with the rule that removed it. |
| `data/clean/cleaning_report.json` | `clean_data.py` | For each station, how many readings were kept and removed, for which reasons, and how many gaps of each length there are. |
| `data/features/train.csv`, `dev.csv`, `test.csv` | `build_features.py` | The model's inputs and target for each set, with one row per station and predicted hour. |
| `data/features/normalisation.json` | `build_features.py` | The training-set mean and standard deviation of each normalised input. Prediction reuses these values, so live data is scaled the same way as training data. |
| `data/logs/*.log` | Every script | Each script's log messages, added to on every run. `predict.log` also keeps every prediction made. |

### Run folders

Every script that trains or evaluates something saves its results to a new folder in `runs/`. Each folder is named by its run type plus the date and time it was saved, such as `dense_20261107_164530`, so a new run never overwrites an earlier one. Scripts that use another run's results, such as `plot_history.py` and `predict.py`, pick the most recent folder of the right type.

| Folder | Created by | What it contains |
|---|---|---|
| `persistence_*/` | `baseline_persistence.py` | `results.json`: the persistence baseline's RMSE and MAE on the training and development data, for all hours, smoke hours and non-smoke hours |
| `linear_*/` | `baseline_linear.py` | `results.json`: the regularisation strength (λ) chosen and the errors. `history.csv`: the training and development cost every 100 iterations for each λ tried |
| `dense_*/` | `train_dense.py` | `results.json`: the network's settings and errors. `history.csv`: the training and development cost after each epoch. `model.pt`: the trained weights. `cost_curves.png`: the cost chart, once `plot_history.py` has been run |
| `gru_*/`, `lstm_*/` | `train_sequence.py` | The same files as `dense_*/`, for a GRU or LSTM network |
| `search_*/` | `tune_dense.py` | `trials.csv`: one row per random trial, with its settings and errors. `histories/`: each trial's cost history. `results.json`: the ranges searched |
| `diagnosis_*/` | `diagnose_train_dev.py` | `summary.csv`: the mean RMSE on training data, held-out training weeks and development data, with the gaps between them. `results.csv`: the result of every individual run |
| `analysis_*/` | `analyse_errors.py` | Development errors broken down by month, station and reading level, with days of high readings and how often one station's rise follows another's |
| `comparison_*/` | `compare_dense_gru_lstm.py` | `selection.csv`: the GRU and LSTM sizes tried on held-out training weeks. `comparison.csv` and `summary.csv`: every model's development errors. `results.json`: the sizes chosen and whether either replaced dense |
| `final_*/` | `train_final.py` | `dense.pt` and `linear.pt`: the final models, trained on the training and development data together. `results.json`: their settings and how closely they fit that data. `final_history.csv`: the training cost after each epoch |
| `test_*/` | `evaluate_test.py` | `test_scores.csv`: the test results described below. `results.json`: which final run was evaluated and on how many rows |

### Reading `test_scores.csv`

Each row gives one model's score on one group of test hours.

| Column | Meaning |
|---|---|
| `group` | The hours being scored: `all_hours`, `smoke_hours` (35 µg/m³ or more), `non_smoke_hours`, `onset_hours` (a rise of 10 µg/m³ or more within an hour) or `extreme_hours` (100 µg/m³ or more). |
| `model` | `persistence`, `linear` or `dense`. |
| `rows` | How many station-hours are in the group. |
| `rmse`, `mae` | Root mean squared error and mean absolute error, in µg/m³. RMSE weighs large misses more heavily. |
| `mean_error` | The average of the prediction minus the target. A negative value means the model predicts too low. |
| `rmse_gain_vs_persistence_%`, `mae_gain_vs_persistence_%` | How much lower the error is than persistence's for the same hours. A negative value means the model did worse than persistence. |

## Limitations

- **Data delay.** OpenAQ publishes each hour's reading about an hour later, so a prediction usually covers the hour already in progress rather than a full hour ahead.
- **Spike removal.** The spike rule needs readings from up to 3 hours later, so it cannot clean live data. A live prediction may use a spike that training would have removed.
- **Smoke onsets.** Sudden smoke arrivals remain the largest source of error. Even the best model is off by about 20 µg/m³ in these hours.
- **One weather station.** Wind comes only from Pearson, which may not match conditions at each station.
- **Repeated use of development data.** The development data guided many decisions, so its results are somewhat optimistic. The test results above are the reliable ones. Statistics from the test set were viewed once early in the project, but no decision used them.

## What didn't help

The following were tested and rejected. Each result is the dense network's root mean squared error (RMSE) on the development data, in µg/m³ and averaged over the same 6 training runs. Each row compares the network without the change to the network with it. The starting value differs between rows because wind was added to the inputs partway through testing, which lowered it from 2.513 to 2.502.

| Idea | Development RMSE (µg/m³), without → with | Why it was rejected |
|---|---|---|
| Giving high readings more weight during training. | 2.513 → 2.564 to 2.629, depending on how strongly they were weighted. | The network memorised the few smoke hours in its training data. Its smoke RMSE on those hours fell from 9.6 to 6.8, but stayed about 9.7 on held-out training weeks. |
| GRU and LSTM networks reading the 24 hours as a sequence. | Dense 2.502 → LSTM 2.591 and GRU 2.672. | Their tanh and sigmoid activations compress the very high readings during smoke. Before the latest reading was also given to them directly, bypassing those activations, both were far worse, at about 3.43. |
| NASA FIRMS wildfire hotspots upwind of Toronto. | 2.502 → 2.509, with only 1 of 6 runs improving. | Upwind fires make smoke more likely in the coming days but do not predict which hour it arrives. |
| Pearson visibility and its smoke reports. | Not trained, since a check first showed no warning. | Visibility drops during smoke but not before it: its change 1–3 hours earlier is uncorrelated with the next change in PM2.5. |

## Data sources

- PM2.5 readings from the [OpenAQ API](https://openaq.org/), which needs an API key.
- Weather data from Environment and Climate Change Canada (Data Source: Environment and Climate Change Canada): historical observations from the [Historical Climate Data archive](https://climate.weather.gc.ca/) and live observations from the [MSC GeoMet API](https://api.weather.gc.ca/), under [ECCC's data licence terms](https://eccc-msc.github.io/open-data/licence/readme_en).
