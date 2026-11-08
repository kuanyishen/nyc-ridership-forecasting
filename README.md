# NYC ridership forecasting

[![ci](https://github.com/kuanyishen/nyc-ridership-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/kuanyishen/nyc-ridership-forecasting/actions/workflows/ci.yml)

**Can we forecast NYC transit ridership two weeks out, and know when to stop trusting the forecast?**
A production-style pipeline on the MTA's public daily ridership data: five series (subway, bus, LIRR, Metro-North,
bridge and tunnel traffic), six models from a naive baseline to Prophet and XGBoost, a rolling backtest that never peeks
at the future, prediction intervals checked for coverage, MLflow tracking, a drift monitor, and a weekly retrain on
GitHub Actions.

Dashboard: `docs/index.html` (publish with GitHub Pages: Settings, Pages, branch `main`, folder `/docs`).

## Results (48 forecast origins, 2023-01 to 2026-10, 14-day horizon)
- **Ensemble (Prophet + XGBoost): 4.3% average error (WAPE)** vs 7.1% for "same weekday last week" and 22.5% for plain naive. Better than the benchmark on every series, significantly (origin bootstrap).
- XGBoost beats Prophet significantly on subway and bus; the ensemble's edge over XGBoost alone is within noise on every series. The ensemble is best on weekdays, XGBoost alone on weekends and holidays.
- Holidays and weekends are the hard days; error is nearly flat from 1 to 14 days ahead.
- Prediction intervals built only from earlier errors cover ~84% of actuals for an 80% target.
- Drift monitor: catches ~20% level shifts within about a week and often 10% ones, mostly misses 5%; congestion pricing moved bridge-and-tunnel traffic by only ~2 points, below its sensitivity.
- **Weather (NOAA, Central Park):** the late-January and late-February 2026 alarm clusters were the two biggest snowstorms of the winter (a Sunday with 11.4 in of snow cut ridership 56-68%, a Monday with 10.9 in cut it 74-100%). 16 of 26 out-of-sample alarms fall on snow or heavy-rain days; the monitor now labels alarms as weather events or not. A regression weather adjustment of the monitor did not help (it under-predicts the day after a storm), reported as a negative result.
- **Does weather help the forecast?** Oracle experiment (actual weather as input, which a 14-day forecast cannot have): bad-weather-day error falls from 10.0% to 6.9% but overall only from 4.3% to 4.1%; with perfect weather for just the next 3 days, 4.3% to 4.2%. Realistic use needs archived weather forecasts (not done).

## Method highlights
- **Direct multi-horizon XGBoost:** one model for all 14 horizons predicting log(actual / base), base = median of the last four same weekdays; features use only information available at the forecast origin (unit-tested by corrupting all later data).
- **Honest backtest:** origins spaced 29 days apart so every weekday is an origin (28 days made every origin a Sunday and confounded horizon with weekday; caught and fixed). Uncertainty comes from bootstrapping whole origins, not single days.
- **Intervals:** empirical relative-error quantiles from earlier origins only; coverage measured on later ones.
- **Drift monitor:** CUSUM on a year-over-year-adjusted "surprise" series; false-alarm rate and detection delay measured by injecting known shifts into real history.

## Layout
```
config/pipeline.yaml   series, horizon, backtest and interval settings
src/ridership/         data, calendar, models, backtest, intervals, metrics, drift, tracking, run (CLI)
analysis/              build_report.py (figures + dashboard)
data/raw/              MTA daily ridership CSV and NOAA Central Park weather (both refreshed weekly)
data/processed/        backtest.csv (every backtest forecast vs actual)
forecasts/             one file per weekly forecast (the live track record)
docs/                  dashboard, figures, monitor.json, mlflow_runs.json
```

## Run it
```bash
pip install -e ".[dev]"
python -m ridership.run backtest    # ~4 min: 5 series x 6 models x 48 origins
python -m ridership.run forecast    # next 14 days with intervals
python -m ridership.run monitor     # drift detectors + live forecast quality
python -m ridership.run track       # log backtest runs to MLflow (SQLite)
python -m ridership.run weather-backtest   # oracle experiment: forecast with actual weather as input
python analysis/weather_analysis.py        # storm effects, alarm attribution, oracle results
python analysis/build_report.py
pytest
```
`.github/workflows/weekly.yml` does fetch, forecast, monitor, track and report every Monday (and on demand).

## Limitations
Systemwide daily totals only (no weather, events or fares); horizon and retraining cadence are my choices; the live
forecast track record starts empty and fills in as forecasts mature; the weekly fetch assumes the data.ny.gov API schema
matches the CSV export.

Data: [MTA Daily Ridership and Traffic: Beginning 2020](https://catalog.data.gov/dataset/mta-daily-ridership-and-traffic-beginning-2020) (data.ny.gov). MIT licensed code.
