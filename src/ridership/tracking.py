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


def merge_run_tables(old: list[dict], new: list[dict]) -> list[dict]:
    """Union of two run lists keyed by run_id (new wins), newest first. Lets a CI job with a fresh MLflow database keep history."""
    by_id = {r["run_id"]: r for r in old}
    by_id.update({r["run_id"]: r for r in new})
    return sorted(by_id.values(), key=lambda r: str(r.get("start_time", "")), reverse=True)


def export_runs(tracking_uri: str, out: Path) -> int:
    """Dump the run table (params, metrics, tags) as JSON for the dashboard, merged with what is already in ``out``."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    new: list[dict] = []
    if exp is not None:
        runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"])
        keep = [c for c in runs.columns if c.startswith(("params.", "metrics.", "tags.kind", "tags.data_end", "tags.git_sha")) or c in ("run_id", "start_time", "tags.mlflow.runName")]
        new = json.loads(runs[keep].to_json(orient="records", date_format="iso"))
    old = json.loads(out.read_text()) if out.exists() else []
    merged = merge_run_tables(old, new)
    out.write_text(json.dumps(merged, indent=1))
    return len(merged)
