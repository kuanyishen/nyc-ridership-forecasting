"""Build the dashboard (docs/index.html) and figures from the backtest, live forecast, monitor and MLflow exports.

    python analysis/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ridership import data, metrics
from ridership.run import load_config

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
REPO = "https://github.com/kuanyishen/nyc-ridership-forecasting"
BLUE, SKY, ORANGE, GREEN, RED, GREY, INK, PINK = "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00", "#8a8f98", "#1f2328", "#CC79A7"
MODEL_COLOR = {"naive": "#c3c7cd", "seasonal_naive": GREY, "seasonal_median4": "#5d6673", "prophet": ORANGE, "xgboost": BLUE, "ensemble": GREEN}
MODEL_LABEL = {"naive": "Naive (last value)", "seasonal_naive": "Seasonal naive (same weekday last week)", "seasonal_median4": "Median of last 4 same weekdays",
               "prophet": "Prophet", "xgboost": "XGBoost", "ensemble": "Ensemble (Prophet + XGBoost)"}
SERIES_LABEL = {"Subway": "Subway", "Bus": "Bus", "LIRR": "Long Island Rail Road", "MNR": "Metro-North", "BT": "Bridges & tunnels (vehicles)"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8ebef", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
})
pc = lambda x, d=1: f"{x * 100:.{d}f}%"


def save(fig, name):
    fig.savefig(ASSETS / name, facecolor="white")
    plt.close(fig)


def fig_forecast(F, cfg):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, name in zip(axes.ravel(), cfg["series"]):
        s = F["series"][name]
        hist = pd.Series(s["history"]); hist.index = pd.to_datetime(hist.index)
        fc = pd.DataFrame(s["forecast"]); fc["date"] = pd.to_datetime(fc["date"])
        scale, unit = (1e6, "M") if hist.max() > 1.5e6 else (1e3, "k")
        ax.plot(hist.index, hist / scale, color=INK, lw=1.3)
        ax.fill_between(fc["date"], fc["lo"] / scale, fc["hi"] / scale, color=GREEN, alpha=0.25, lw=0)
        ax.plot(fc["date"], fc["forecast"] / scale, color=GREEN, lw=2)
        ax.set_title(SERIES_LABEL[name], fontsize=10)
        ax.set_ylabel(f"daily, {unit}")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", labelsize=8)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0, 0.6, "black: last 90 published days\ngreen: next 14 days\nband: 80% interval", fontsize=9.5, va="top", color=INK)
    fig.tight_layout()
    save(fig, "fig_forecast.png")


def fig_wape(tbl):
    order = ["Subway", "Bus", "LIRR", "MNR", "BT"]
    models = ["naive", "seasonal_naive", "seasonal_median4", "prophet", "xgboost", "ensemble"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for j, m in enumerate(models):
        for i, s in enumerate(order[::-1]):
            r = tbl[(s, m)]
            y = i + (j - 2.5) * 0.12
            ax.errorbar(r[0] * 100, y, xerr=[[(r[0] - r[1]) * 100], [(r[2] - r[0]) * 100]], fmt="o", color=MODEL_COLOR[m], ms=5, capsize=2, lw=1.4)
    ax.set_yticks(range(len(order)), [SERIES_LABEL[s] for s in order[::-1]])
    ax.set_xlabel("14-day-ahead error, WAPE % (95% CI over forecast origins). Lower is better")
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 8, 12, 20, 35], ["2", "3", "5", "8", "12", "20", "35"])
    ax.minorticks_off()
    handles = [plt.Line2D([0], [0], marker="o", color=MODEL_COLOR[m], lw=0, ms=6) for m in models]
    ax.legend(handles, [MODEL_LABEL[m] for m in models], frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Accuracy by series and model")
    save(fig, "fig_wape.png")


def fig_breakdown(bt):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    models = ["seasonal_naive", "prophet", "xgboost", "ensemble"]
    for ax, col, title, order in ((axes[0], "h_bucket", "By how far ahead", ["1-3 days", "4-7 days", "8-14 days"]),
                                  (axes[1], "day_type", "By type of day", ["weekday", "weekend", "holiday period"])):
        w = 0.8 / len(models)
        for k, m in enumerate(models):
            vals = [metrics.wape(bt[(bt["model"] == m) & (bt[col] == o)]) * 100 for o in order]
            xs = np.arange(len(order)) + (k - 1.5) * w
            ax.bar(xs, vals, w, color=MODEL_COLOR[m], label=MODEL_LABEL[m] if col == "h_bucket" else None)
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.15, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(order)), order)
        ax.set_title(title)
        ax.set_ylabel("WAPE %")
        ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fig_breakdown.png")


def fig_coverage(cov):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    models = ["naive", "seasonal_naive", "seasonal_median4", "prophet", "xgboost", "ensemble"]
    v = [cov.loc[m, "coverage"] * 100 for m in models]
    ax.barh(range(len(models))[::-1], v, color=[MODEL_COLOR[m] for m in models], height=0.65)
    ax.axvline(80, color=RED, ls="--", lw=1.2)
    ax.text(80.4, len(models) - 0.55, "target 80%", color=RED, fontsize=8.5)
    for y, x in zip(range(len(models))[::-1], v):
        ax.text(x + 0.3, y, f"{x:.0f}%", va="center", fontsize=8.5)
    ax.set_yticks(range(len(models))[::-1], [MODEL_LABEL[m] for m in models], fontsize=8.5)
    ax.set_xlim(60, 92)
    ax.set_xlabel("share of actual days inside the 80% interval")
    ax.grid(axis="y", visible=False)
    ax.set_title("Do the intervals deliver what they promise?")
    save(fig, "fig_coverage.png")


def fig_drift(M, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    deltas = sorted({i["delta"] for i in M["series"]["Subway"]["injection"]})
    for name in cfg["series"]:
        ds = {i["delta"]: i["detected_share"] for i in M["series"][name]["injection"]}
        ax.plot([d * 100 for d in sorted(ds)], [ds[d] * 100 for d in sorted(ds)], "o-", lw=1.8, label=SERIES_LABEL[name].split(" (")[0])
    ax.set_xlabel("size of an injected level shift (%)")
    ax.set_ylabel("detected within 4 weeks (%)")
    ax.set_title("How big a shift can it catch?")
    ax.set_ylim(-3, 103)
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    for i, name in enumerate(cfg["series"][::-1]):
        al = M["series"][name]["alarms_out_of_sample"]
        ax.scatter(pd.to_datetime([a["date"] for a in al]), [i] * len(al), color=[RED if a["direction"] == "down" else BLUE for a in al], s=38, zorder=3)
    ax.set_yticks(range(len(cfg["series"])), [SERIES_LABEL[n].split(" (")[0] for n in cfg["series"][::-1]])
    ax.set_title("Alarms on history the thresholds were not tuned on")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.text(0.01, 0.02, "red = ridership below normal, blue = above", transform=ax.transAxes, fontsize=8, color=GREY)
    fig.tight_layout()
    save(fig, "fig_drift.png")


def table(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def main() -> None:
    cfg = load_config()
    bt = metrics.annotate(pd.read_csv(ROOT / "data" / "processed" / "backtest.csv", parse_dates=["origin", "date"]))
    F = json.loads((ASSETS / "latest_forecast.json").read_text())
    M = json.loads((ASSETS / "monitor.json").read_text())
    runs = json.loads((ASSETS / "mlflow_runs.json").read_text()) if (ASSETS / "mlflow_runs.json").exists() else []
    abl = json.loads((ASSETS / "ablation.json").read_text()) if (ASSETS / "ablation.json").exists() else []
    models_all = cfg["models"]
    ref = cfg["reference_model"]
    ASSETS.mkdir(parents=True, exist_ok=True)

    tbl = {(s, m): metrics.origin_ci(bt, s, m) for s in cfg["series"] for m in models_all}
    pooled = bt.groupby("model").apply(metrics.wape)
    cov = metrics.interval_metrics(bt, cfg["interval_level"]).groupby("model")[["coverage", "rel_width"]].mean()
    paired = {(s, a, b): metrics.origin_bootstrap(bt, s, a, b) for s in cfg["series"] for a, b in (("ensemble", ref), ("xgboost", "prophet"), ("ensemble", "xgboost"))}
    fig_forecast(F, cfg); fig_wape(tbl); fig_breakdown(bt); fig_coverage(cov); fig_drift(M, cfg)

    n_orig, o0, o1 = bt["origin"].nunique(), bt["origin"].min().date(), bt["origin"].max().date()
    wide = data.to_wide(data.load_raw())
    raw_end = wide.index.max().date()
    best = pooled.drop(["naive"]).idxmin()
    skill_cut = 1 - pooled["ensemble"] / pooled[ref]

    # congestion-pricing check on bridge and tunnel traffic
    y = data.get_series(wide, "BT")
    yoy = lambda a, b: y.loc[a:b].sum() / y.shift(364).loc[a:b].sum() - 1
    cp_before, cp_after = yoy("2024-11-10", "2025-01-04"), yoy("2025-01-12", "2025-03-08")

    t_wape = table(["Series", *[MODEL_LABEL[m].split(" (")[0] for m in models_all]],
                   [[SERIES_LABEL[s], *[(f"<b>{pc(tbl[(s, m)][0])}</b>" if m == min(models_all, key=lambda mm: tbl[(s, mm)][0]) else pc(tbl[(s, m)][0])) + f" <span class='sub'>({pc(tbl[(s, m)][1])}-{pc(tbl[(s, m)][2])})</span>" for m in models_all]] for s in cfg["series"]]
                   + [["<i>All series pooled</i>", *[pc(pooled[m]) for m in models_all]]])
    t_pair = table(["Series", "Ensemble vs seasonal naive", "XGBoost vs Prophet", "Ensemble vs XGBoost"],
                   [[SERIES_LABEL[s], *[f"{paired[(s, a, b)]['wape_diff'] * 100:+.2f} pp [{paired[(s, a, b)]['lo'] * 100:+.2f}, {paired[(s, a, b)]['hi'] * 100:+.2f}], p {('< 0.001' if paired[(s, a, b)]['p_value'] < 0.001 else '= ' + format(paired[(s, a, b)]['p_value'], '.3f'))}"
                                         for a, b in (("ensemble", ref), ("xgboost", "prophet"), ("ensemble", "xgboost"))]] for s in cfg["series"]])
    t_cov = table(["Model", "Coverage of the 80% interval", "Typical interval width (relative to the forecast)"],
                  [[MODEL_LABEL[m], pc(cov.loc[m, "coverage"]), pc(cov.loc[m, "rel_width"])] for m in models_all])
    sub = F["series"]["Subway"]["forecast"]
    t_fc = table(["Date", "Forecast", "80% interval"], [[r["date"], f"{r['forecast'] / 1e6:.2f} M", f"{r['lo'] / 1e6:.2f} - {r['hi'] / 1e6:.2f} M"] for r in sub])
    t_inj = table(["Series", "Alarm threshold (CUSUM h)", "False alarms per year (out of sample)", "Detects a 20% drop", "Detects a 10% drop", "Detects a 5% drop"],
                  [[SERIES_LABEL[s].split(" (")[0], f"{M['series'][s]['h']:.1f}", f"{M['series'][s]['false_alarm_rate_per_year']:.1f}",
                    *[(lambda i: f"{i['detected_share'] * 100:.0f}% (median {i['median_delay_days']:.0f} days)" if i["median_delay_days"] is not None else f"{i['detected_share'] * 100:.0f}%")(next(x for x in M['series'][s]['injection'] if abs(x['delta'] - d) < 1e-9)) for d in (-0.2, -0.1, -0.05)]]
                   for s in cfg["series"]])
    t_abl, abl_text = "", ""
    if abl:
        A = pd.DataFrame(abl).pivot_table(index=["series", "model"], columns="train_start", values="wape")
        allh, mid, late = "all history (2020-03)", "2021-06-01", "2022-06-01"
        abl_text = (f"Starting in 2021-06 helps Prophet (error {pc(A.loc[('Bus', 'prophet'), mid])} vs {pc(A.loc[('Bus', 'prophet'), allh])} on bus, {pc(A.loc[('Subway', 'prophet'), mid])} vs {pc(A.loc[('Subway', 'prophet'), allh])} on subway, with all history) "
                    f"and is neutral for XGBoost (within {max(abs(A.loc[(sr, 'xgboost'), mid] - A.loc[(sr, 'xgboost'), allh]) for sr in ('Bus', 'Subway')) * 100:.1f} points either way, no consistent winner). "
                    f"Starting later hurts Prophet badly ({pc(A.loc[('Subway', 'prophet'), late])} on subway) because it loses the year-over-year seasonality it needs.")
        df = pd.DataFrame(abl).pivot_table(index=["series", "model"], columns="train_start", values="wape").reset_index()
        cols = [c for c in df.columns if c not in ("series", "model")]
        t_abl = table(["Series", "Model", *cols], [[r["series"], r["model"], *[pc(r[c]) for c in cols]] for _, r in df.iterrows()])
    dt = lambda m, d: metrics.wape(bt[(bt["model"] == m) & (bt["day_type"] == d)])
    hb = lambda m, b: metrics.wape(bt[(bt["model"] == m) & (bt["h_bucket"] == b)])
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NYC Ridership Forecasting</title>
<meta name="description" content="Two-week-ahead forecasts of daily NYC transit ridership with honest rolling backtests, calibrated intervals, MLflow tracking and a drift monitor, retrained weekly.">
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav class="top"><div class="wrap">
  <a class="brand" href="index.html">NYC ridership forecasting</a>
  <a href="#forecast">Forecast</a><a href="#results">Results</a><a href="#monitor">Drift monitor</a><a href="#pipeline">Pipeline</a><a href="{REPO}">GitHub</a>
</div></nav>
<main class="wrap">
<header class="hero">
  <div class="eyebrow">Time series · MLOps · public MTA data</div>
  <h1>Can we forecast NYC ridership two weeks out, and know when to stop trusting the forecast?</h1>
  <p class="lead">A production-style pipeline on the MTA's daily ridership data: five series (subway, bus, LIRR, Metro-North, bridge and tunnel traffic), six models from a naive baseline to Prophet and XGBoost, a rolling backtest that never peeks at the future, prediction intervals checked for coverage, MLflow tracking, and a drift monitor. It retrains every week on GitHub Actions.</p>
  <div class="row"><a class="btn" href="#forecast">See the live forecast</a><a class="btn ghost" href="{REPO}">View code</a></div>
</header>

<div class="kpis">
  <div class="kpi"><b>{pc(pooled['ensemble'])}</b><span>average 14-day-ahead error (WAPE) of the best model, the Prophet + XGBoost ensemble</span></div>
  <div class="kpi"><b>{skill_cut * 100:.0f}% lower</b><span>error than "same weekday last week" ({pc(pooled[ref])}); better on every series, significantly</span></div>
  <div class="kpi"><b>{pc(cov.loc['ensemble', 'coverage'], 0)}</b><span>of actual days land inside the 80% interval (the intervals use only past errors)</span></div>
  <div class="kpi"><b>&ge; 20%</b><span>level shifts the drift monitor catches within about a week; shifts under ~5% it mostly misses</span></div>
</div>

<div class="callout"><p><b>TL;DR.</b> Ridership has strong weekly and holiday structure, so the bar is "beat last week's value", not "beat a naive guess" (plain naive is {pc(pooled['naive'], 0)} off). XGBoost and Prophet clear that bar on all five series, and averaging them is the most accurate overall ({pc(pooled['ensemble'])} vs {pc(pooled[ref])}), though the ensemble's edge over XGBoost alone is within noise on every series. XGBoost beats Prophet significantly on subway and bus. Holidays and weekends are hardest ({pc(bt[bt.day_type == 'holiday period'].pipe(lambda d: metrics.wape(d[d.model == 'ensemble'])))} and {pc(bt[bt.day_type == 'weekend'].pipe(lambda d: metrics.wape(d[d.model == 'ensemble'])))} error vs {pc(bt[bt.day_type == 'weekday'].pipe(lambda d: metrics.wape(d[d.model == 'ensemble'])))} on weekdays). The drift monitor is reliable for big shifts and honest about being blind to small ones: congestion pricing moved bridge-and-tunnel traffic by only about {abs(cp_after - cp_before) * 100:.1f} points, below what it can detect.</p></div>

<h2 id="data">The data</h2>
<p>MTA Daily Ridership and Traffic ({"2020-03-01"} to {raw_end}, from <a href="https://catalog.data.gov/dataset/mta-daily-ridership-and-traffic-beginning-2020">data.ny.gov</a>): one row per day and mode. Practical issues I handled rather than assumed away: counts arrive as text with thousands separators; the newest days are missing for some modes while the MTA catches up (publication lag), so each series is cut at its last published day; the Staten Island Railway has gaps in 2020-21; and the 2020 collapse and recovery is a different regime from today. I model five modes with complete histories, and I start training in 2021-06 (see the ablation below).</p>

<h2 id="forecast">The forecast: next 14 days</h2>
<figure><img src="assets/fig_forecast.png" alt="14-day forecasts for five series"><figcaption>Ensemble forecast with an 80% interval, made from data through {F['data_end']}. This figure and table are regenerated every week.</figcaption></figure>
<h3>Subway, day by day</h3>
{t_fc}
<p class="sub">Day-of-week and holidays drive the pattern (for example the weekend dips). The forecast starts at each series' last published day, so subway and bus forecasts begin a day or two before the others.</p>

<h2 id="method">How the models were built and tested</h2>
{table(['Model', 'What it does'], [
    ['Naive', 'Repeats the last value. Ignores the weekly cycle, so it is a floor.'],
    ['Seasonal naive', 'Same weekday last week (or the most recent same weekday for horizons beyond 7 days). The real benchmark.'],
    ['Median of 4 weeks', 'Median of the last four same weekdays; robust to one odd week.'],
    ['Prophet', 'Trend + weekly and yearly seasonality + New York holidays.'],
    ['XGBoost (direct multi-horizon)', 'One model for all 14 horizons. Predicts log(actual / base), where base is the median of the last four same weekdays, from lags, recent level and trend, and calendar features of the target day. Features use only information available at the forecast origin (a unit test corrupts all later data and checks nothing changes).'],
    ['Ensemble', 'Average of Prophet and XGBoost.'],
])}
<ul class="tight">
<li><b>Rolling-origin backtest:</b> {n_orig} forecast origins from {o0} to {o1}, spaced 29 days apart so every weekday is an origin (my first version used 28-day spacing, which made every origin a Sunday and confounded forecast horizon with day of week; I caught it because error did not rise with horizon, then fixed and re-ran). At each origin every model is refit on the past only and scored on the next 14 days.</li>
<li><b>Metric:</b> WAPE = total absolute error divided by total actual ridership (it is not thrown off by small days like MAPE).</li>
<li><b>Uncertainty:</b> the 14 forecasts from one origin share the same shocks, so confidence intervals resample <em>whole origins</em>, never single days.</li>
<li><b>Prediction intervals:</b> for each model and horizon band, the relative errors of <em>earlier</em> origins give the interval around a new forecast, and coverage is then measured on later origins.</li>
</ul>

<h2 id="results">Results</h2>
<figure><img src="assets/fig_wape.png" alt="WAPE by series and model"><figcaption>14-day-ahead WAPE with 95% confidence intervals over forecast origins (log scale).</figcaption></figure>
{t_wape}
<p>Pooled over all series: ensemble {pc(pooled['ensemble'])}, XGBoost {pc(pooled['xgboost'])}, Prophet {pc(pooled['prophet'])}, median of 4 weeks {pc(pooled['seasonal_median4'])}, seasonal naive {pc(pooled['seasonal_naive'])}, naive {pc(pooled['naive'])}. Paired differences, with confidence intervals over origins (negative = the first model is better):</p>
{t_pair}
<ul class="tight">
<li><b>Every model class beats the benchmark</b> on every series; the ensemble's gain over seasonal naive is {abs(max(paired[(s, 'ensemble', ref)]['wape_diff'] for s in cfg['series'])) * 100:.1f}-{abs(min(paired[(s, 'ensemble', ref)]['wape_diff'] for s in cfg['series'])) * 100:.1f} percentage points and significant everywhere.</li>
<li><b>XGBoost vs Prophet:</b> XGBoost is better on every series, but the gap is only statistically clear for subway and bus.</li>
<li><b>Ensemble vs XGBoost:</b> the difference is within noise on all five series. I would ship the ensemble for robustness, not because it is proven better.</li>
</ul>
<figure><img src="assets/fig_breakdown.png" alt="Error by horizon and day type"><figcaption>Pooled WAPE by forecast horizon and by type of day (holiday period = a holiday and the days next to it).</figcaption></figure>
<p>Holiday periods and weekends are where forecasts are weakest, which is also where staffing decisions are hardest. Models that know the calendar cut holiday-period error to about {pc(dt('xgboost', 'holiday period'), 0)} (XGBoost) from {pc(dt(ref, 'holiday period'), 0)} for seasonal naive. One caution: Prophet is {'worse' if dt('prophet', 'weekend') > dt(ref, 'weekend') else 'better'} than seasonal naive on weekends ({pc(dt('prophet', 'weekend'))} vs {pc(dt(ref, 'weekend'))}), so its gains come from weekdays and holidays. Error is also nearly flat from 1 to 14 days ahead ({pc(hb('ensemble', '1-3 days'))} to {pc(hb('ensemble', '8-14 days'))} for the ensemble), which says the signal is mostly the weekly and calendar pattern rather than short-term momentum. The ensemble is best on weekdays ({pc(dt('ensemble', 'weekday'))}), but XGBoost alone is better on weekends ({pc(dt('xgboost', 'weekend'))} vs {pc(dt('ensemble', 'weekend'))}) and holiday periods ({pc(dt('xgboost', 'holiday period'))} vs {pc(dt('ensemble', 'holiday period'))}), so which to deploy depends on which days matter most.</p>
<h3>Are the intervals honest?</h3>
<figure><img src="assets/fig_coverage.png" alt="Interval coverage by model" style="max-width:640px"><figcaption>An 80% interval should contain 80% of actuals.</figcaption></figure>
{t_cov}
<p>The model intervals are slightly conservative (about {pc(cov.loc['ensemble', 'coverage'], 0)} vs the 80% target); the baselines' intervals are narrower but come with much larger point errors.</p>
{('<h3>Why train from 2021-06?</h3><p>Same backtest, three training start dates (WAPE), on the two series with the most data:</p>' + t_abl + f'<p>{abl_text}</p>') if t_abl else ''}

<h2 id="monitor">Drift monitor: knowing when the forecast stops being trustworthy</h2>
<p>Two checks run every week:</p>
<ol class="tight">
<li><b>Level-shift detector.</b> Each day's "surprise" is log(actual / median of the last four same weekdays), minus the same quantity one and two years earlier so recurring calendar transitions (December holidays, summer) cancel. A two-sided CUSUM on that, scaled by its own recent spread, alarms when ridership drifts away from its pattern. Thresholds were tuned on {M['tuning_window'][0]} to {M['tuning_window'][1]} and evaluated on history from {M['evaluation_start']} onward.</li>
<li><b>Forecast quality.</b> The live forecasts are scored against actuals as they arrive and compared with the error distribution the backtest saw. Status now: {M['live']['status']}. {"(It starts producing a score after the first forecasts mature, about two weeks.)" if M['live']['status'] != 'scored' else ''}</li>
</ol>
<figure><img src="assets/fig_drift.png" alt="Detection power and alarm timeline"><figcaption>Left: how often a level shift injected into real history is detected within four weeks. Right: alarms on out-of-sample history.</figcaption></figure>
{t_inj}
<ul class="tight">
<li><b>Measured, not assumed:</b> detection power comes from injecting known shifts into the real series at 24 start dates; false-alarm rates come from history the thresholds never saw.</li>
<li><b>It reliably catches big shifts</b> (about 20% within a week) and often 10% ones, but mostly misses 5%. It is a smoke alarm, not a microscope.</li>
<li><b>Alarms cluster on the same days across series</b> (late January and late February 2026 in particular), which points to region-wide disruptions rather than random noise; I did not match them to weather or news.</li>
<li><b>A real test, honestly reported:</b> congestion pricing started 2025-01-05. Year-over-year, bridge-and-tunnel traffic changed from {cp_before * 100:+.1f}% in the eight weeks before to {cp_after * 100:+.1f}% in the eight weeks after, a {abs(cp_after - cp_before) * 100:.1f}-point move, far below the monitor's sensitivity, so it (correctly, by design) did not alarm.</li>
</ul>

<h2 id="pipeline">The production pipeline</h2>
<ul class="tight">
<li><b>Weekly run on GitHub Actions</b> (<code>.github/workflows/weekly.yml</code>): pull the latest published days from the data.ny.gov API, refit every model on all data, write the next 14 days with intervals, score earlier forecasts against actuals, run the drift monitor, log to MLflow, rebuild this page.</li>
<li><b>MLflow tracking:</b> {len(runs)} runs logged (one per series and model for the backtest, one per live forecast) with parameters, metrics, data end date and git commit. The run table is exported to <code>docs/assets/mlflow_runs.json</code>.</li>
<li><b>Failure handling:</b> if the API is unreachable the run keeps the existing data and says so; every forecast file is kept in <code>forecasts/</code>, so the live track record builds up automatically.</li>
<li><b>Tests:</b> 24 tests, including no-leakage, interval no-peeking, origin spacing and drift-detector calibration.</li>
</ul>

{('<figure><img src="assets/mlflow_ui.png" alt="MLflow runs table" style="max-width:760px"><figcaption>The MLflow UI for this project: one run per series and model, each with parameters, metrics, the data end date and the git commit.</figcaption></figure>') if (ASSETS / 'mlflow_ui.png').exists() else ''}

<h2>Limitations</h2>
<ul class="tight">
<li>Systemwide daily totals only: no weather, events, service changes or fares as predictors, and no station-level detail.</li>
<li>The 14-day horizon and weekly retraining are my choices; shorter-horizon or intraday forecasting would need different data.</li>
<li>Most backtest origins fall after the pandemic recovery, so a new regime shock would test the models differently than {n_orig} calm-ish origins do.</li>
<li>Intervals assume future errors resemble past errors; they widen only through the retraining cycle.</li>
</ul>

<h2>Reproduce</h2>
<pre><code>pip install -e ".[dev]"
python -m ridership.run backtest    # ~4 min: 5 series x 6 models x {n_orig} origins
python -m ridership.run forecast
python -m ridership.run monitor
python -m ridership.run track
python analysis/build_report.py
pytest</code></pre>
<footer>Built by Sky Shen · <a href="{REPO}">source and data</a></footer>
</main>
</body>
</html>
"""
    (ROOT / "docs" / "index.html").write_text(page)
    print("wrote docs/index.html")


if __name__ == "__main__":
    main()
