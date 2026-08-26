"""Archive the completed D8 execution into MLflow without rerunning D8.

This script is archival only. It reads the already-completed D8 artifacts,
records their result/provenance in MLflow, and does not fit or refit any model.

It exists because the original live D8 MLflow write failed on Windows:
FileStore treats parameter filenames case-insensitively, so
`filter_d_test` and `filter_D_test` collided. This archive uses unambiguous
parameter names instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT = "medical-assistance-demand-forecasting"
EXPECTED_ROW = "D8"
EXPECTED_PROTOCOL = "protocol-v1.5"


# ----------------------------- utilities -----------------------------


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def as_param(value: Any) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def log_params_safe(values: dict[str, Any]) -> None:
    mlflow.log_params({k: as_param(v) for k, v in values.items()})


def log_metrics_safe(values: dict[str, Any]) -> None:
    cleaned = {}
    for key, value in values.items():
        if value is None or isinstance(value, bool):
            continue
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if x == x and x not in (float("inf"), float("-inf")):
            cleaned[key] = x
    if cleaned:
        mlflow.log_metrics(cleaned)


def git_last_commit_for(path: Path, project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty_for(path: Path, project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return str(bool(result.stdout.strip()))
    except Exception:
        return "unknown"


# ------------------------------ main ------------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    tracking_uri = (project_root / "mlruns").as_uri()

    result_dir = project_root / "results" / "d8"
    selection_path = result_dir / "d8_lag_selection.json"
    screen_path = result_dir / "d8_ccf_screen.csv"
    grid_path = result_dir / "d8_lag_grid.csv"
    report_path = result_dir / "d8_report.md"

    addendum_path = project_root / "docs" / "addenda.md"
    source_path = project_root / "src" / "d8_fx_lag.py"

    required = [
        selection_path,
        screen_path,
        grid_path,
        report_path,
        addendum_path,
        source_path,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        sys.exit(
            "Missing required archival file(s):\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    rec = load_json(selection_path)

    if rec.get("row") != EXPECTED_ROW:
        sys.exit(
            "Expected row {}, found {}.".format(
                EXPECTED_ROW, rec.get("row")
            )
        )
    if rec.get("protocol_tag") != EXPECTED_PROTOCOL:
        sys.exit(
            "Expected protocol {}, found {}.".format(
                EXPECTED_PROTOCOL, rec.get("protocol_tag")
            )
        )

    selection_sha = sha256_of(selection_path)
    addendum_sha = sha256_of(addendum_path)
    source_sha = sha256_of(source_path)

    archive_key = "D8|{}|{}".format(
        rec.get("run_utc", "unknown"),
        selection_sha[:16],
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        sys.exit("MLflow experiment was not created.")

    client = MlflowClient(tracking_uri=tracking_uri)
    for run in client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=5000,
    ):
        if (
            run.data.tags.get("archive_key") == archive_key
            and run.data.tags.get("archive_complete") == "true"
        ):
            print("D8 is already archived in MLflow.")
            print("run_id:", run.info.run_id)
            return

    stage1 = rec["stage1_ccf"]
    filt = stage1["filter"]
    best = rec["selected_lag"]
    m1 = rec["operative_m1"]
    cand_lags = rec["candidate_lags"]
    opt = rec["optimizer"]

    source_git_commit = git_last_commit_for(
        Path("src") / "d8_fx_lag.py", project_root
    )
    source_git_dirty = git_dirty_for(
        Path("src") / "d8_fx_lag.py", project_root
    )

    run = mlflow.start_run(run_name="ARCHIVE D8 FX lag")
    try:
        mlflow.set_tags({
            "archive_backfill": "true",
            "archive_complete": "false",
            "execution_recomputed": "false",
            "archive_key": archive_key,
            "archive_reason":
                "live D8 MLflow write failed due Windows FileStore "
                "case-insensitive parameter-name collision",
            "protocol.row": rec["row"],
            "protocol.tag": rec["protocol_tag"],
            "protocol.freeze_tag": rec.get("protocol_freeze_tag", "unknown"),
            "protocol.run_utc": rec.get("run_utc", "unknown"),
            "protocol.addendum_sha256": addendum_sha,
            "data_sha256": rec["data"]["sha256"],
            "history_sha256": rec["history"]["sha256"],
            "selection_json_sha256": selection_sha,
            "source_sha256": source_sha,
            "source_git_commit": source_git_commit,
            "source_git_dirty": source_git_dirty,
            "smoke_mode": str(bool(rec.get("smoke_mode", False))),
        })

        log_params_safe({
            "scale": rec["scale"],
            "lag_min": cand_lags["min"],
            "lag_max": cand_lags["max"],
            "n_candidate_lags": cand_lags["n"],
            "n_screen": len(stage1["selected_lags"]),
            "ccf_method": stage1["method"],
            "screened_lags": stage1["selected_lags"],
            "raw_ccf_top5_diagnostic":
                stage1.get("raw_ccf_top5_diagnostic"),
            "filter_order": filt["order"],
            "filter_seasonal_order": filt["seasonal_order"],
            "filter_with_intercept": filt["with_intercept"],
            # Windows-safe names: do not use d/D-only case distinctions.
            "filter_nonseasonal_test": filt["d_test"],
            "filter_seasonal_test": filt["D_test"],
            "filter_max_nonseasonal_difference": filt["max_d"],
            "filter_max_seasonal_difference": filt["max_D"],
            "filter_estimation_start": filt["estimation_start"],
            "filter_estimation_end": filt["estimation_end"],
            "filter_n_estimation": filt["n_estimation"],
            "filter_burn_fx": filt["burn_in_fx"],
            "filter_burn_response": filt["burn_in_response"],
            "ccf_response_start": filt["ccf_response_start"],
            "ccf_response_end": filt["ccf_response_end"],
            "ccf_n_response": filt["n_ccf_response"],
            "m1_candidate": m1["candidate"],
            "m1_order": m1["order"],
            "m1_seasonal_order": m1["seasonal_order"],
            "m1_with_intercept": m1["with_intercept"],
            "stage2_method": opt["method"],
            "stage2_maxiter": opt["maxiter"],
            "tie_rule": rec["tie_rule"],
        })

        log_metrics_safe({
            "selected_lag": best["lag"],
            "selected_aicc": best["aicc"],
            "margin_over_runner_up": best["margin_over_runner_up"],
            "ccf_prewhitened_at_selected": best["ccf_prewhitened"],
            "screen_rank_selected": best["screen_rank"],
            "filter_aicc": filt["aicc"],
            "filter_iterations": filt["iterations"],
            "raw_ccf_agrees": int(bool(stage1.get("raw_ccf_agrees"))),
        })

        for path in (
            selection_path,
            screen_path,
            grid_path,
            report_path,
        ):
            mlflow.log_artifact(str(path), artifact_path="d8")

        mlflow.log_artifact(
            str(addendum_path),
            artifact_path="protocol",
        )
        mlflow.log_artifact(
            str(source_path),
            artifact_path="source",
        )

        mlflow.set_tag("archive_complete", "true")
        mlflow.end_run(status="FINISHED")

    except Exception:
        mlflow.set_tag("archive_complete", "false")
        mlflow.end_run(status="FAILED")
        raise

    print("D8 archived successfully.")
    print("tracking URI:", tracking_uri)
    print("experiment:", EXPERIMENT)
    print("selected FX lag:", best["lag"])
    print("selected AICc:", best["aicc"])
    print("original D8 run UTC:", rec.get("run_utc"))
    print("source git commit:", source_git_commit)
    print("source git dirty:", source_git_dirty)


if __name__ == "__main__":
    main()
