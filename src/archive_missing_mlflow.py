"""Backfill missing MLflow records from existing thesis result artifacts.

This script is ARCHIVAL ONLY:
    * it does not fit, refit, transform, select, or forecast;
    * it reads already-written JSON/report/CSV artifacts;
    * it creates fresh MLflow runs marked archive_backfill=true;
    * it preserves each original execution timestamp as a tag;
    * it skips a run if the same archive_key already exists.

Expected project layout (relative to the project root):
    results/d5/d5_selection.json
    results/d11/d11_scale_trigger.json
    results/d12_d2/d12_d2_differencing.json
    results/d12_d3/d12_d3_seasonal_differencing.json
    results/d12_d5/d5_selection.json
    results/d12_d5_diagnostic/d12_d5_convergence_diagnostic.json
    results/d7/d7_pad_selection.json

D12/D4 is archived as its own continuity run, derived mechanically from the
D4 outcome already recorded inside the transformed D3 artifact.

Proprietary residual-level CSV files are deliberately NOT logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow.tracking import MlflowClient


DEFAULT_EXPERIMENT = "medical-assistance-demand-forecasting"


# ----------------------------- small utilities -----------------------------


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_param(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "None"
    return str(value)


def log_params_safe(values: dict[str, Any]) -> None:
    mlflow.log_params({k: as_param(v) for k, v in values.items()})


def finite_metric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return x


def log_metrics_safe(values: dict[str, Any]) -> None:
    cleaned = {}
    for key, value in values.items():
        x = finite_metric(value)
        if x is not None:
            cleaned[key] = x
    if cleaned:
        mlflow.log_metrics(cleaned)


def log_artifact_if_exists(path: Path, artifact_path: str = "source_artifacts") -> None:
    if path.exists() and path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def tags_common(record: dict, source_json: Path, archive_key: str) -> dict[str, str]:
    return {
        "archive_backfill": "true",
        "execution_recomputed": "false",
        "archive_key": archive_key,
        "source_json": str(source_json.resolve()),
        "source_json_sha256": sha256(source_json),
        "original_run_utc": str(record.get("run_utc", "unknown")),
        "protocol_tag": str(record.get("protocol_tag", "unknown")),
        "protocol_freeze_tag": str(record.get("protocol_freeze_tag", "unknown")),
        "data_sha256": str(record.get("data", {}).get("sha256", "unknown")),
        "smoke_mode": str(bool(record.get("smoke_mode", False))),
    }


def experiment_id_for(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id


def existing_archive_keys(client: MlflowClient, experiment_id: str) -> set[str]:
    keys = set()
    for run in client.search_runs(
        experiment_ids=[experiment_id],
        max_results=5000,
    ):
        key = run.data.tags.get("archive_key")
        if key:
            keys.add(key)
    return keys


def archive_key(label: str, record: dict, source_json: Path) -> str:
    return "{}|{}|{}|{}".format(
        label,
        record.get("run_utc", "unknown"),
        record.get("data", {}).get("sha256", "unknown"),
        sha256(source_json)[:16],
    )


def start_archive_run(
    *,
    run_name: str,
    label: str,
    record: dict,
    source_json: Path,
    existing_keys: set[str],
):
    key = archive_key(label, record, source_json)

    if key in existing_keys:
        print("SKIP  {:<30} already archived".format(run_name))
        return None, key

    run = mlflow.start_run(run_name=run_name)
    mlflow.set_tags(tags_common(record, source_json, key))
    mlflow.set_tag("archive_label", label)
    return run, key


# ------------------------------ run archivers ------------------------------


def archive_d5(
    source_json: Path,
    *,
    run_name: str,
    expected_scale: str,
    existing_keys: set[str],
) -> bool:
    rec = load_json(source_json)

    if rec.get("row") != "D5":
        raise ValueError("{} is not a D5 record".format(source_json))
    if rec.get("scale") != expected_scale:
        raise ValueError(
            "{} has scale {}, expected {}".format(
                source_json, rec.get("scale"), expected_scale
            )
        )

    run, key = start_archive_run(
        run_name=run_name,
        label="D5_{}".format(expected_scale),
        record=rec,
        source_json=source_json,
        existing_keys=existing_keys,
    )
    if run is None:
        return False

    try:
        sel = rec["selected"]
        hk = rec["hk_settings"]
        counts = rec["counts"]
        est = rec["estimation_sample"]

        log_params_safe({
            "row": "D5",
            "scale": rec["scale"],
            "candidate": sel["candidate"],
            "order": sel["order"],
            "seasonal_order": sel["seasonal_order"],
            "with_intercept": sel["with_intercept"],
            "d": hk.get("d"),
            "D": hk.get("D"),
            "seasonal_m": hk.get("m"),
            "method": hk.get("method"),
            "maxiter": hk.get("maxiter"),
            "information_criterion": hk.get("information_criterion"),
            "stepwise": hk.get("stepwise"),
            "d6_triggered": rec.get("d6_triggered"),
            "n_train": est.get("n_train"),
            "nobs_effective": est.get("nobs_effective"),
        })

        log_metrics_safe({
            "selected_aicc": sel.get("aicc"),
            "selected_lb_pvalue": sel.get("lb_pvalue"),
            "selected_lb_lag": sel.get("lb_lag"),
            "selected_lb_df": sel.get("lb_df"),
            "selected_converged": int(sel.get("converged") is True),
            "n_candidates": counts.get("candidates"),
            "n_visited": counts.get("visited_valid_fits"),
            "n_lb_pass": counts.get("lb_pass"),
            "n_not_converged": counts.get("not_converged"),
            "n_intercept_fits_dropped_d4": counts.get(
                "intercept_fits_dropped_d4"
            ),
            "d6_triggered": int(bool(rec.get("d6_triggered"))),
        })

        outdir = source_json.parent
        for name in (
            "d5_selection.json",
            "d5_report.md",
            "d5_visited_fits.csv",
            "d5_candidate_winners.csv",
        ):
            log_artifact_if_exists(outdir / name)

        trace_files = sorted(outdir.glob("trace_*.txt"))
        for path in trace_files:
            log_artifact_if_exists(path, artifact_path="source_artifacts/traces")

        mlflow.set_tag("archival_note", "existing D5 outputs only; no refit performed")
    finally:
        mlflow.end_run()

    existing_keys.add(key)
    print("ADDED {:<30} {}".format(run_name, rec.get("run_utc")))
    return True


def archive_d11(
    source_json: Path,
    *,
    existing_keys: set[str],
) -> bool:
    rec = load_json(source_json)

    run, key = start_archive_run(
        run_name="ARCHIVE D11 scale trigger",
        label="D11_scale_trigger",
        record=rec,
        source_json=source_json,
        existing_keys=existing_keys,
    )
    if run is None:
        return False

    try:
        ev = rec["evaluated_on"]
        jb = rec["jarque_bera"]
        sp = rec["spearman_absresid_vs_fitted"]
        dec = rec["decision"]

        log_params_safe({
            "row": rec.get("row"),
            "evaluated_scale": ev.get("scale"),
            "candidate": ev.get("candidate"),
            "order": ev.get("order"),
            "seasonal_order": ev.get("seasonal_order"),
            "with_intercept": ev.get("with_intercept"),
            "jb_alpha": jb.get("alpha"),
            "spearman_threshold": sp.get("threshold"),
            "triggered": dec.get("triggered"),
            "burn_in_dropped": ev.get("burn_in_dropped"),
            "n_residuals": ev.get("n_residuals"),
        })

        log_metrics_safe({
            "refit_aicc": ev.get("refit_aicc"),
            "d5_recorded_aicc": ev.get("d5_recorded_aicc"),
            "jarque_bera_stat": jb.get("statistic"),
            "jarque_bera_p": jb.get("pvalue"),
            "residual_skew": jb.get("skew"),
            "residual_kurtosis": jb.get("kurtosis"),
            "spearman_rho": sp.get("rho"),
            "spearman_abs_rho": sp.get("abs_rho"),
            "spearman_p": sp.get("pvalue"),
            "triggered": int(bool(dec.get("triggered"))),
        })

        outdir = source_json.parent
        log_artifact_if_exists(outdir / "d11_scale_trigger.json")
        log_artifact_if_exists(outdir / "d11_report.md")

        # Deliberately do NOT log d11_residuals.csv.
        mlflow.set_tag(
            "excluded_artifact",
            "d11_residuals.csv (residual-level/proprietary; not archived to MLflow)",
        )
    finally:
        mlflow.end_run()

    existing_keys.add(key)
    print("ADDED {:<30} {}".format("ARCHIVE D11 scale trigger", rec.get("run_utc")))
    return True


def archive_d12_d2(
    source_json: Path,
    *,
    existing_keys: set[str],
) -> bool:
    rec = load_json(source_json)

    run, key = start_archive_run(
        run_name="ARCHIVE D12 D2 log1p",
        label="D12_D2_log1p",
        record=rec,
        source_json=source_json,
        existing_keys=existing_keys,
    )
    if run is None:
        return False

    try:
        adf = rec["levels"]["adf"]
        kpss = rec["levels"]["kpss"]
        dec = rec["decision"]

        log_params_safe({
            "row": rec.get("row"),
            "branch": rec.get("branch"),
            "scale": rec.get("scale"),
            "alpha": rec.get("alpha"),
            "candidate_set": rec.get("candidate_set"),
            "pilot_fourier_K": rec.get("pilot_form", {}).get("fourier_K"),
            "d_selected": dec.get("d"),
            "adf_rejects_unit_root": dec.get("adf_rejects_unit_root"),
            "kpss_rejects_stationarity": dec.get(
                "kpss_rejects_stationarity"
            ),
        })

        log_metrics_safe({
            "adf_stat": adf.get("statistic"),
            "adf_p": adf.get("pvalue"),
            "adf_crit_5pct": adf.get("crit_5pct"),
            "adf_used_lag": adf.get("used_lag"),
            "kpss_stat": kpss.get("statistic"),
            "kpss_p": kpss.get("pvalue"),
            "kpss_crit_5pct": kpss.get("crit_5pct"),
            "kpss_used_lags": kpss.get("used_lags"),
            "d_selected": dec.get("d"),
        })

        confirm = rec.get("confirmatory_check")
        if isinstance(confirm, dict):
            for test_name in ("adf", "kpss"):
                test = confirm.get(test_name)
                if isinstance(test, dict):
                    log_metrics_safe({
                        "confirm_{}_stat".format(test_name): test.get("statistic"),
                        "confirm_{}_p".format(test_name): test.get("pvalue"),
                        "confirm_{}_crit_5pct".format(test_name): test.get("crit_5pct"),
                    })

        outdir = source_json.parent
        log_artifact_if_exists(outdir / "d12_d2_differencing.json")
        log_artifact_if_exists(outdir / "d12_d2_report.md")

        # Deliberately do NOT log d12_d2_pilot_residuals.csv.
        mlflow.set_tag(
            "excluded_artifact",
            "d12_d2_pilot_residuals.csv (reconstructive/proprietary; not archived to MLflow)",
        )
    finally:
        mlflow.end_run()

    existing_keys.add(key)
    print("ADDED {:<30} {}".format("ARCHIVE D12 D2 log1p", rec.get("run_utc")))
    return True


def archive_d12_d3(
    source_json: Path,
    *,
    existing_keys: set[str],
) -> bool:
    rec = load_json(source_json)

    run, key = start_archive_run(
        run_name="ARCHIVE D12 D3 log1p",
        label="D12_D3_log1p",
        record=rec,
        source_json=source_json,
        existing_keys=existing_keys,
    )
    if run is None:
        return False

    try:
        ocsb = rec["ocsb"]
        dec = rec["decision"]
        d4 = rec["d4"]

        log_params_safe({
            "row": rec.get("row"),
            "branch": rec.get("branch"),
            "scale": rec.get("scale"),
            "s": ocsb.get("s"),
            "lag_method": ocsb.get("lag_method"),
            "D_selected": dec.get("D"),
            "ocsb_rejects": ocsb.get("rejects"),
            "d4_with_intercept": d4.get("with_intercept"),
        })

        log_metrics_safe({
            "ocsb_stat": ocsb.get("statistic"),
            "ocsb_crit_val": ocsb.get("critical_value"),
            "D_selected": dec.get("D"),
            "ocsb_rejects": int(bool(ocsb.get("rejects"))),
        })

        outdir = source_json.parent
        log_artifact_if_exists(outdir / "d12_d3_seasonal_differencing.json")
        log_artifact_if_exists(outdir / "d12_d3_report.md")
    finally:
        mlflow.end_run()

    existing_keys.add(key)
    print("ADDED {:<30} {}".format("ARCHIVE D12 D3 log1p", rec.get("run_utc")))
    return True


def archive_d12_d4(
    d3_source_json: Path,
    *,
    existing_keys: set[str],
) -> bool:
    """Create the continuity D4 record from the D4 outcome embedded in D12/D3."""
    rec = load_json(d3_source_json)
    d4 = rec["d4"]
    operative = rec["operative_differencing"]

    # Use the D3 artifact as the source record and distinguish this run by label.
    run, key = start_archive_run(
        run_name="ARCHIVE D12 D4 log1p",
        label="D12_D4_log1p",
        record=rec,
        source_json=d3_source_json,
        existing_keys=existing_keys,
    )
    if run is None:
        return False

    try:
        log_params_safe({
            "row": "D4",
            "branch": "D12",
            "scale": "log1p",
            "d": operative.get("d"),
            "D": operative.get("D"),
            "with_intercept": d4.get("with_intercept"),
            "rule": d4.get("rule"),
            "source": d4.get("source"),
        })
        log_metrics_safe({
            "d": operative.get("d"),
            "D": operative.get("D"),
            "with_intercept": int(bool(d4.get("with_intercept"))),
        })

        log_artifact_if_exists(d3_source_json)
        mlflow.set_tag(
            "archival_note",
            "D4 is mechanical and was recorded inside transformed D3; no separate estimation was run",
        )
    finally:
        mlflow.end_run()

    existing_keys.add(key)
    print("ADDED {:<30} {}".format("ARCHIVE D12 D4 log1p", rec.get("run_utc")))
    return True


def archive_convergence(
    source_json: Path,
    *,
    existing_keys: set[str],
) -> bool:
    rec = load_json(source_json)

    run, key = start_archive_run(
        run_name="ARCHIVE D12 D5 convergence",
        label="D12_D5_convergence_diagnostic",
        record=rec,
        source_json=source_json,
        existing_keys=existing_keys,
    )
    if run is None:
        return False

    try:
        src = rec["source_D5"]
        design = rec["diagnostic_design"]
        summary = rec["summary"]

        log_params_safe({
            "row": rec.get("row"),
            "status": rec.get("status"),
            "changes_D5_selection": rec.get("changes_D5_selection"),
            "scale": src.get("scale"),
            "candidate": src.get("candidate"),
            "order": src.get("order"),
            "seasonal_order": src.get("seasonal_order"),
            "with_intercept": src.get("with_intercept"),
            "method": src.get("method"),
            "recorded_maxiter": src.get("recorded_maxiter"),
            "budgets": design.get("budgets"),
            "fresh_fit_each_budget": design.get("fresh_fit_each_budget"),
            "warm_start": design.get("warm_start"),
        })

        metrics = {
            "d5_recorded_aicc": src.get("recorded_aicc"),
            "d5_recorded_converged": int(bool(src.get("recorded_converged"))),
            "first_converged_maxiter": summary.get("first_converged_maxiter"),
            "highest_budget": summary.get("highest_budget"),
            "highest_budget_converged": int(
                bool(summary.get("highest_budget_converged"))
            ),
            "highest_budget_aicc": summary.get("highest_budget_aicc"),
            "highest_budget_aicc_delta_vs_d5": summary.get(
                "highest_budget_aicc_delta_vs_d5"
            ),
        }

        for fit in rec.get("fits", []):
            budget = fit.get("maxiter")
            if budget is None:
                continue
            prefix = "maxiter_{}".format(int(budget))
            metrics.update({
                "{}_aicc".format(prefix): fit.get("aicc"),
                "{}_converged".format(prefix): int(bool(fit.get("converged"))),
                "{}_iterations".format(prefix): fit.get("iterations"),
                "{}_aicc_delta_vs_d5".format(prefix): fit.get(
                    "aicc_delta_vs_d5"
                ),
            })

        log_metrics_safe(metrics)

        outdir = source_json.parent
        log_artifact_if_exists(outdir / "d12_d5_convergence_diagnostic.json")
        log_artifact_if_exists(outdir / "d12_d5_convergence_report.md")
        mlflow.set_tag(
            "archival_note",
            "diagnostic only; source D5 selection unchanged",
        )
    finally:
        mlflow.end_run()

    existing_keys.add(key)
    print(
        "ADDED {:<30} {}".format(
            "ARCHIVE D12 D5 convergence", rec.get("run_utc")
        )
    )
    return True


def archive_d7(
    source_json: Path,
    *,
    existing_keys: set[str],
) -> bool:
    rec = load_json(source_json)

    run, key = start_archive_run(
        run_name="ARCHIVE D7 holiday pad",
        label="D7_holiday_pad",
        record=rec,
        source_json=source_json,
        existing_keys=existing_keys,
    )
    if run is None:
        return False

    try:
        m1 = rec["operative_m1"]
        best = rec["selected_pad"]
        opt = rec["optimizer"]
        diag = rec["diagnostic_m1_no_holidays"]

        log_params_safe({
            "row": rec.get("row"),
            "scale": rec.get("scale"),
            "source_d5_protocol_tag": rec.get("source_d5_protocol_tag"),
            "candidate": m1.get("candidate"),
            "order": m1.get("order"),
            "seasonal_order": m1.get("seasonal_order"),
            "with_intercept": m1.get("with_intercept"),
            "pad_grid": rec.get("pad_grid", {}).get("set"),
            "n_pads": rec.get("pad_grid", {}).get("n_pads"),
            "selected_b": best.get("b"),
            "selected_f": best.get("f"),
            "method": opt.get("method"),
            "maxiter": opt.get("maxiter"),
        })

        overlaps = best.get("overlap_days_absorbed_by_union", {})

        log_metrics_safe({
            "selected_aicc": best.get("aicc"),
            "selected_iterations": best.get("iterations"),
            "selected_converged": int(bool(best.get("converged"))),
            "margin_over_runner_up": best.get("margin_over_runner_up"),
            "selected_b": best.get("b"),
            "selected_f": best.get("f"),
            "days_H_NY": best.get("days_H_NY"),
            "days_H_OT": best.get("days_H_OT"),
            "overlap_days_H_NY": overlaps.get("H_NY"),
            "overlap_days_H_OT": overlaps.get("H_OT"),
            "diagnostic_m1_aicc": diag.get("aicc"),
            "diagnostic_m1_converged": int(bool(diag.get("converged"))),
        })

        outdir = source_json.parent
        for name in (
            "d7_pad_selection.json",
            "d7_report.md",
            "d7_pad_grid.csv",
        ):
            log_artifact_if_exists(outdir / name)
    finally:
        mlflow.end_run()

    existing_keys.add(key)
    print("ADDED {:<30} {}".format("ARCHIVE D7 holiday pad", rec.get("run_utc")))
    return True


# ----------------------------------- main -----------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Archive existing thesis result artifacts into MLflow; no modelling",
    )
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument(
        "--tracking-uri",
        default=None,
        help="default: <project-root>/mlruns",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate expected artifacts and show planned archive runs only",
    )
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    results = root / "results"

    paths = {
        "D5 count": results / "d5" / "d5_selection.json",
        "D11": results / "d11" / "d11_scale_trigger.json",
        "D12 D2": results / "d12_d2" / "d12_d2_differencing.json",
        "D12 D3": results / "d12_d3" / "d12_d3_seasonal_differencing.json",
        "D12 D5": results / "d12_d5" / "d5_selection.json",
        "D12 D5 diagnostic": (
            results
            / "d12_d5_diagnostic"
            / "d12_d5_convergence_diagnostic.json"
        ),
        "D7": results / "d7" / "d7_pad_selection.json",
    }

    print("project root : {}".format(root))
    print("results root : {}".format(results))
    print("")

    missing = []
    for label, path in paths.items():
        status = "FOUND" if path.exists() else "MISSING"
        print("{:<20} {}  {}".format(label, status, path))
        if not path.exists():
            missing.append((label, path))

    if missing:
        print("")
        print("Nothing was written to MLflow.")
        print("Resolve the missing path(s), then run again.")
        sys.exit(2)

    print("")
    print("D12 D4            DERIVED from {}".format(paths["D12 D3"]))

    if args.dry_run:
        print("")
        print("DRY RUN ONLY -- no MLflow runs created.")
        print("Planned new archival runs:")
        for name in (
            "ARCHIVE D5 count",
            "ARCHIVE D11 scale trigger",
            "ARCHIVE D12 D2 log1p",
            "ARCHIVE D12 D3 log1p",
            "ARCHIVE D12 D4 log1p",
            "ARCHIVE D12 D5 log1p",
            "ARCHIVE D12 D5 convergence",
            "ARCHIVE D7 holiday pad",
        ):
            print("  - {}".format(name))
        return

    tracking_uri = args.tracking_uri
    if tracking_uri is None:
        tracking_uri = (root / "mlruns").as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = experiment_id_for(args.experiment)
    mlflow.set_experiment(args.experiment)

    client = MlflowClient(tracking_uri=tracking_uri)
    existing_keys = existing_archive_keys(client, experiment_id)

    print("")
    print("tracking URI : {}".format(mlflow.get_tracking_uri()))
    print("experiment   : {} (id {})".format(args.experiment, experiment_id))
    print("archive keys already present: {}".format(len(existing_keys)))
    print("")

    added = 0

    added += archive_d5(
        paths["D5 count"],
        run_name="ARCHIVE D5 count",
        expected_scale="count",
        existing_keys=existing_keys,
    )
    added += archive_d11(
        paths["D11"],
        existing_keys=existing_keys,
    )
    added += archive_d12_d2(
        paths["D12 D2"],
        existing_keys=existing_keys,
    )
    added += archive_d12_d3(
        paths["D12 D3"],
        existing_keys=existing_keys,
    )
    added += archive_d12_d4(
        paths["D12 D3"],
        existing_keys=existing_keys,
    )
    added += archive_d5(
        paths["D12 D5"],
        run_name="ARCHIVE D12 D5 log1p",
        expected_scale="log1p",
        existing_keys=existing_keys,
    )
    added += archive_convergence(
        paths["D12 D5 diagnostic"],
        existing_keys=existing_keys,
    )
    added += archive_d7(
        paths["D7"],
        existing_keys=existing_keys,
    )

    print("")
    print("Archive backfill complete.")
    print("new MLflow runs added: {}".format(added))
    print("No modelling or selection step was rerun.")


if __name__ == "__main__":
    main()
