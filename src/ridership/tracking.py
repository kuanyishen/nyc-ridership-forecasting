"""MLflow experiment tracking: one run per (series, model) with parameters, metrics and a tag for the data it saw."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

EXPERIMENT = "nyc-ridership-forecasting"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def log_runs(table: pd.DataFrame, params: dict, *, kind: str, data_end: str, tracking_uri: str, artifacts: dict[str, Path] | None = None) -> int:
    """``table`` has one row per (series, model) with numeric metric columns; every other column is ignored."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    metric_cols = [c for c in table.columns if c not in ("series", "model") and pd.api.types.is_numeric_dtype(table[c])]
    for _, r in table.iterrows():
        with mlflow.start_run(run_name=f"{kind}-{r['series']}-{r['model']}"):
            mlflow.log_params({**params, "series": r["series"], "model": r["model"], "kind": kind})
            mlflow.log_metrics({c: float(r[c]) for c in metric_cols if pd.notna(r[c])})
            mlflow.set_tags({"data_end": data_end, "git_sha": _git_sha(), "kind": kind})
            for name, path in (artifacts or {}).items():
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
    return len(table)


def export_runs(tracking_uri: str, out: Path) -> int:
    """Dump the run table (params, metrics, tags) as JSON for the dashboard."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        return 0
    runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"])
    keep = [c for c in runs.columns if c.startswith(("params.", "metrics.", "tags.kind", "tags.data_end", "tags.git_sha")) or c in ("run_id", "start_time", "tags.mlflow.runName")]
    out.write_text(json.dumps(json.loads(runs[keep].to_json(orient="records", date_format="iso")), indent=1))
    return len(runs)
