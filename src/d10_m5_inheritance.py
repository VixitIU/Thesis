"""D10 -- M5 inheritance.

Operative protocol state: protocol-v1.6.

Frozen D10:
    inherits (b, f) from M2, FX lag from M3, search lag from M4;
    no re-tuning.

D10 selects nothing. It is the bookkeeping row that assembles the three
already-selected indicator settings into one specification and verifies
that they are mutually consistent, so that Section E has a single
authoritative description of M5 to estimate at each origin.

What "no re-tuning" means here
------------------------------
No search of any kind is run. The ARIMA orders, seasonal orders, annual
form and constant decision come from the operative D5 selection; the pad
comes from D7; the FX lag from D8; the search lag from D9. Nothing is
re-selected, and no alternative to any of them is fitted.

M5 is fitted ONCE on the training sample. That fit is a feasibility and
record check -- it confirms the assembled specification estimates and
converges, and gives an AICc for the descriptive ladder table -- not a decision.
No frozen rule makes M5 conditional on that number, and no diagnostic
or later evaluation is allowed to change the inherited M5 design.

Consistency requirements checked, not assumed
---------------------------------------------
The nesting relied on at E8 (Clark-West for M2, M3, M4 and M5 against
M1) holds only if every augmented model differs from M1 in its
exogenous regressors alone. D7, D8 and D9 each declared that they held
M1's orders and annual form fixed; this script verifies that all three
artifacts in fact describe the SAME operative M1, that they ran on the
same training extraction and indicator history, and that none was a
smoke run. A mismatch aborts: assembling M5 from rows that were tuned
against different baselines would silently break the nesting claim.

Regressor construction is inherited rather than reimplemented: the
holiday windows are built by importing D7's own loader and window
builder, so the M5 columns cannot drift from the ones D7 selected on.

Per the addendum of 26 August 2026, the Stage-2 search-regressor scaling
travels with the inherited regressor: the search column enters M5 in
thousands of queries, exactly as at D9. FX enters unscaled, as at D8.

Per the addendum of 24 August 2026, the M5 fit uses method='lbfgs',
maxiter=500, convergence is recorded, and execution PAUSES rather than
substituting anything if it fails to converge.

Outputs (--outdir, default results/d10)
---------------------------------------
    d10_m5_specification.json  the authoritative M5 spec for Section E
    d10_ladder.csv             M1..M5 training AICc, assembled from the
                               row artifacts plus the M5 fit
    d10_report.md              prose record for the research log

No proprietary artifact is written. The live run is recorded in MLflow
together with the operative addendum and the source script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:
    import d5_baseline_order_selection as d5
except ImportError:
    sys.exit("d5_baseline_order_selection.py must be importable (same "
             "directory or on PYTHONPATH).")
try:
    import d7_holiday_pad as d7
except ImportError:
    sys.exit("d7_holiday_pad.py must be importable (same directory or on "
             "PYTHONPATH): D10 reuses its cluster loader and window "
             "builder so the M5 holiday columns cannot drift from the "
             "ones D7 selected the pad on.")

# ----------------------------- frozen constants ------------------------------
ROW = "D10"
PROTOCOL_TAG = "protocol-v1.6"

# Historical row provenance is intentional: each source artifact must be the
# operative artifact actually selected under the protocol state in force when
# that row was executed.
SOURCE_D5_PROTOCOL_TAG = "protocol-v1.3"
SOURCE_D7_PROTOCOL_TAG = "protocol-v1.4"
SOURCE_D8_PROTOCOL_TAG = "protocol-v1.5"
SOURCE_D9_PROTOCOL_TAG = "protocol-v1.6"

DOWNSTREAM_MAXITER = 500
DOWNSTREAM_METHOD = "lbfgs"

# 26 Aug 2026 addendum: the Stage-2 scaling travels with the inherited
# search regressor. FX is unscaled, as at D8.
INDICATOR_SCALE_DIVISOR = 1000.0

FX_COL = "fx_rub_per_thb"
SEARCH_COL = "search_index"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_history(path: Path, col: str, need_from: pd.Timestamp,
                 need_to: pd.Timestamp) -> pd.Series:
    h = pd.read_csv(path, parse_dates=["obs_date"]).set_index("obs_date")
    if col not in h.columns:
        sys.exit("{} has no column '{}'".format(path, col))
    s = h[col].astype(float).sort_index()
    gaps = s.index.to_series().diff().dropna().dt.days
    if not (gaps == 1).all():
        sys.exit("the indicator history is not daily-dense")
    if s.index.min() > need_from or s.index.max() < need_to:
        sys.exit("the indicator history does not cover {} -> {}".format(
            need_from.date(), need_to.date()))
    if s.isna().any():
        sys.exit("NaNs in the {} history".format(col))
    return s


def require(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit("D10 consistency check failed: " + msg)


def m1_signature(rec: dict) -> tuple:
    m = rec["operative_m1"]
    return (m["candidate"], tuple(m["order"]), tuple(m["seasonal_order"]),
            bool(m["with_intercept"]))


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False, description="D10: M5 inheritance")
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--history", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--selection", required=True,
                    help="operative d5_selection.json")
    ap.add_argument("--d7", required=True, help="d7_pad_selection.json")
    ap.add_argument("--d8", required=True, help="d8_lag_selection.json")
    ap.add_argument("--d9", required=True, help="d9_lag_selection.json")
    ap.add_argument("--outdir", default="results/d10")
    ap.add_argument("--experiment",
                    default="medical-assistance-demand-forecasting")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()
    args.d = None
    args.D_seasonal = None

    env_diffs = d5.check_environment(args.allow_env_mismatch)

    project_root = Path(__file__).resolve().parents[1]
    tracking_uri = (project_root / "mlruns").as_uri()
    addendum_path = project_root / "docs" / "addenda.md"
    if not addendum_path.exists():
        sys.exit("protocol addendum not found at {}; D10 requires the "
                 "operative addendum to be present before execution."
                 .format(addendum_path))
    addendum_digest = sha256_of(addendum_path)

    selection_path = Path(args.selection)
    d7_path = Path(args.d7)
    d8_path = Path(args.d8)
    d9_path = Path(args.d9)

    source_artifact_paths = {
        "d5_selection": selection_path,
        "d7_selection": d7_path,
        "d8_selection": d8_path,
        "d9_selection": d9_path,
    }
    for name, path in source_artifact_paths.items():
        require(path.exists(), "{} is missing: {}".format(name, path))

    source_artifact_hashes = {
        name: sha256_of(path)
        for name, path in source_artifact_paths.items()
    }

    sel = json.loads(selection_path.read_text())
    r7 = json.loads(d7_path.read_text())
    r8 = json.loads(d8_path.read_text())
    r9 = json.loads(d9_path.read_text())

    require(sel.get("row") == "D5", "--selection is not a D5 artifact.")
    require(r7.get("row") == "D7", "--d7 is not a D7 artifact.")
    require(r8.get("row") == "D8", "--d8 is not a D8 artifact.")
    require(r9.get("row") == "D9", "--d9 is not a D9 artifact.")
    require(not sel.get("smoke_mode", False),
            "the operative D5 selection was a smoke run.")
    require(sel.get("protocol_tag") == SOURCE_D5_PROTOCOL_TAG,
            "the operative D5 artifact is {}, expected {}.".format(
                sel.get("protocol_tag"), SOURCE_D5_PROTOCOL_TAG))
    require(r7.get("protocol_tag") == SOURCE_D7_PROTOCOL_TAG,
            "the operative D7 artifact is {}, expected {}.".format(
                r7.get("protocol_tag"), SOURCE_D7_PROTOCOL_TAG))
    require(r8.get("protocol_tag") == SOURCE_D8_PROTOCOL_TAG,
            "the operative D8 artifact is {}, expected {}.".format(
                r8.get("protocol_tag"), SOURCE_D8_PROTOCOL_TAG))
    require(r9.get("protocol_tag") == SOURCE_D9_PROTOCOL_TAG,
            "the operative D9 artifact is {}, expected {}.".format(
                r9.get("protocol_tag"), SOURCE_D9_PROTOCOL_TAG))

    # Same baseline for every augmented model -- the E8 nesting claim.
    ann = sel["annual_regressor"]
    cand_expected = (
        "monthly"
        if ann["kind"] == "monthly_dummies"
        else "fourier_K{}".format(ann["K"])
    )
    expected_m1 = (
        cand_expected,
        tuple(sel["selected"]["order"]),
        tuple(sel["selected"]["seasonal_order"]),
        bool(sel["selected"]["with_intercept"]),
    )

    for name, rec in (("D7", r7), ("D8", r8), ("D9", r9)):
        got = m1_signature(rec)
        require(
            got == expected_m1,
            "{} describes M1 as {}, but the operative D5 M1 is {}. "
            "M2-M5 must differ from M1 in regressors alone or the E8 "
            "nesting claim fails.".format(name, got, expected_m1)
        )
        require(rec.get("scale") == sel["scale"],
                "{} ran on the {} scale, D5 on {}.".format(
                    name, rec.get("scale"), sel["scale"]))
        require(not rec.get("smoke_mode", False),
                "{} was a smoke run.".format(name))

    # Every downstream row must have used the fixed-specification optimizer
    # policy in force from 24 August onward. D10 does not re-select anything,
    # but the descriptive AICc ladder is only like-for-like if the ceiling and
    # optimizer match.
    for name, rec in (("D7", r7), ("D8", r8), ("D9", r9)):
        opt = rec.get("optimizer") or {}
        require(
            opt.get("method") == DOWNSTREAM_METHOD
            and int(opt.get("maxiter", -1)) == DOWNSTREAM_MAXITER,
            "{} optimizer record is {}, expected method='{}', maxiter={}."
            .format(name, opt, DOWNSTREAM_METHOD, DOWNSTREAM_MAXITER)
        )

    # Same training extraction and indicator history throughout.
    data_path, hist_path = Path(args.data), Path(args.history)
    digest, hist_digest = sha256_of(data_path), sha256_of(hist_path)
    require(digest == sel["data"]["sha256"],
            "the training file does not match the one D5 ran on.")
    for name, rec in (("D7", r7), ("D8", r8), ("D9", r9)):
        require(rec["data"]["sha256"] == digest,
                "{} ran on a different training extraction.".format(name))
    for name, rec in (("D8", r8), ("D9", r9)):
        require(rec["history"]["sha256"] == hist_digest,
                "{} ran on a different indicator history.".format(name))
    cl_digest = sha256_of(Path(args.clusters))
    require(r7["clusters"]["sha256"] == cl_digest,
            "D7 selected the pad on a different holiday cluster file.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- inherited settings -------------------------------------------
    b_pad = int(r7["selected_pad"]["b"])
    f_pad = int(r7["selected_pad"]["f"])
    fx_lag = int(r8["selected_lag"]["lag"])
    search_lag = int(r9["selected_lag"]["lag"])
    print("inherited: pad (b, f) = ({}, {}) from M2; FX lag {} from M3; "
          "search lag {} from M4. Nothing is re-tuned.".format(
              b_pad, f_pad, fx_lag, search_lag))

    args.scale = sel["scale"]
    s, y, y_raw = d5.load_training_series(args)
    max_lag = max(fx_lag, search_lag)
    need_from = y.index.min() - pd.Timedelta(max_lag, "D")
    fx = load_history(hist_path, FX_COL, need_from, y.index.max())
    sr = load_history(hist_path, SEARCH_COL, need_from, y.index.max())
    fx = fx.loc[:y.index.max()]
    sr = sr.loc[:y.index.max()]

    d5.WITH_INTERCEPT = bool(sel["selected"]["with_intercept"])
    cand = cand_expected
    X_ann = d5.build_candidate(cand, y.index)
    require(list(X_ann.columns) == list(ann["columns"]),
            "the annual regressor rebuilt as {} but D5 recorded {}."
            .format(list(X_ann.columns), ann["columns"]))

    clusters = d7.load_clusters(Path(args.clusters))
    X_hol, hol_info = d7.holiday_regressors(clusters, y.index, b_pad, f_pad)
    require(int(X_hol.sum().sum()) > 0, "the holiday windows are empty.")

    fx_col = fx.shift(fx_lag).reindex(y.index)
    sr_col = sr.shift(search_lag).reindex(y.index) / INDICATOR_SCALE_DIVISOR
    require(not fx_col.isna().any() and not sr_col.isna().any(),
            "the inherited lags leave missing regressor values.")
    fx_name = "fx_lag{}".format(fx_lag)
    sr_name = "search_lag{}_per{}".format(
        search_lag, int(INDICATOR_SCALE_DIVISOR))
    X5 = pd.concat([X_ann, X_hol, fx_col.rename(fx_name),
                    sr_col.rename(sr_name)], axis=1)
    print("M5 exogenous block: {} columns ({} annual + 2 holiday + 1 FX "
          "+ 1 search)".format(X5.shape[1], len(ann["columns"])))

    # ---- single M5 fit: record and feasibility check, not a decision ---
    t0 = time.time()
    model = pm.ARIMA(order=tuple(sel["selected"]["order"]),
                     seasonal_order=tuple(sel["selected"]["seasonal_order"]),
                     with_intercept=sel["selected"]["with_intercept"],
                     maxiter=DOWNSTREAM_MAXITER, method=DOWNSTREAM_METHOD,
                     suppress_warnings=True)
    model.fit(y.to_numpy(dtype=float), X=X5.to_numpy(dtype=float))
    res = model.arima_res_
    mle = getattr(res, "mle_retvals", {}) or {}
    m5_aicc = float(res.info_criteria("aicc"))
    converged = bool(mle.get("converged", False))
    print("M5 fitted: AICc {:.4f} (converged {}, iters {}, {:.0f} s)".format(
        m5_aicc, converged, int(mle.get("iterations", -1)),
        time.time() - t0))
    if not converged:
        sys.exit("EXECUTION PAUSED -- the assembled M5 did not converge at "
                 "maxiter = {}. Per the addendum of 24 August 2026 nothing "
                 "is substituted or dropped: document and resolve before "
                 "finalizing D10.".format(DOWNSTREAM_MAXITER))

    # ---- ladder table, assembled from the row artifacts ---------------
    m1_aicc = (r7.get("diagnostic_m1_no_holidays") or {}).get("aicc")
    ladder = pd.DataFrame([
        {"model": "M1", "description": "baseline, annual form only",
         "aicc": m1_aicc, "source": "D7 diagnostic refit"},
        {"model": "M2", "description": "M1 + holidays at pad ({}, {})"
         .format(b_pad, f_pad), "aicc": r7["selected_pad"]["aicc"],
         "source": "D7"},
        {"model": "M3", "description": "M1 + FX at lag {}".format(fx_lag),
         "aicc": r8["selected_lag"]["aicc"], "source": "D8"},
        {"model": "M4", "description": "M1 + search at lag {}"
         .format(search_lag), "aicc": r9["selected_lag"]["aicc"],
         "source": "D9"},
        {"model": "M5", "description": "M1 + all three indicators",
         "aicc": m5_aicc, "source": "D10"},
    ])
    ladder.to_csv(outdir / "d10_ladder.csv", index=False)

    out = {
        "protocol_tag": PROTOCOL_TAG,
        "protocol_freeze_tag": d5.PROTOCOL_FREEZE_TAG,
        "source_d5_protocol_tag": sel.get("protocol_tag"),
        "row": ROW,
        "scale": sel["scale"],
        "selects_nothing": True,
        "rule": "M5 inherits (b, f) from M2, the FX lag from M3 and the "
                "search lag from M4; no re-tuning.",
        "m5_specification": {
            "order": list(sel["selected"]["order"]),
            "seasonal_order": list(sel["selected"]["seasonal_order"]),
            "with_intercept": sel["selected"]["with_intercept"],
            "annual_regressor": ann,
            "holiday": {
                "pad_b": b_pad, "pad_f": f_pad,
                "columns": ["hol_H_NY", "hol_H_OT"],
                "window": "[start - b, end + f] inclusive; union, binary "
                          "(C1-9, C1-10)",
                "days_H_NY": hol_info["H_NY"],
                "days_H_OT": hol_info["H_OT"],
                "clusters_sha256": cl_digest,
            },
            "fx": {"lag_days": fx_lag, "column": fx_name,
                   "scaling": "none (C2-3 units: RUB per 1 THB)"},
            "search": {"lag_days": search_lag, "column": sr_name,
                       "scaling_divisor": INDICATOR_SCALE_DIVISOR,
                       "scaling_source": "26 Aug 2026 addendum; travels "
                                         "with the inherited regressor",
                       "coefficient_interpretation": "per {} queries"
                       .format(int(INDICATOR_SCALE_DIVISOR))},
            "exog_columns": list(X5.columns),
            "n_exog": int(X5.shape[1]),
        },
        "source_artifacts": {
            "d5": {
                "path": str(selection_path.resolve()),
                "sha256": source_artifact_hashes["d5_selection"],
                "protocol_tag": sel.get("protocol_tag"),
            },
            "d7": {
                "path": str(d7_path.resolve()),
                "sha256": source_artifact_hashes["d7_selection"],
                "protocol_tag": r7.get("protocol_tag"),
            },
            "d8": {
                "path": str(d8_path.resolve()),
                "sha256": source_artifact_hashes["d8_selection"],
                "protocol_tag": r8.get("protocol_tag"),
            },
            "d9": {
                "path": str(d9_path.resolve()),
                "sha256": source_artifact_hashes["d9_selection"],
                "protocol_tag": r9.get("protocol_tag"),
            },
        },
        "inheritance": {
            "pad_from": {"row": "D7", "artifact": str(d7_path.resolve()),
                         "protocol_tag": r7.get("protocol_tag")},
            "fx_lag_from": {"row": "D8",
                            "artifact": str(d8_path.resolve()),
                            "protocol_tag": r8.get("protocol_tag")},
            "search_lag_from": {"row": "D9",
                                "artifact": str(d9_path.resolve()),
                                "protocol_tag": r9.get("protocol_tag")},
            "re_tuned": False,
        },
        "m5_training_fit": {
            "aicc": m5_aicc, "converged": converged,
            "iterations": int(mle.get("iterations", -1)),
            "nobs_effective": int(res.nobs_effective),
            "k_params_total": int(np.asarray(res.params).shape[0]),
            "status": "record and feasibility check only; D10 selects "
                      "nothing and no rule makes M5 conditional on this "
                      "value",
        },
        "ladder_training_aicc": {
            r["model"]: r["aicc"] for _, r in ladder.iterrows()},
        "optimizer": {"method": DOWNSTREAM_METHOD,
                      "maxiter": DOWNSTREAM_MAXITER,
                      "source": "24 Aug 2026 addendum"},
        "protocol_addendum": {"path": str(addendum_path),
                              "sha256": addendum_digest},
        "environment_mismatch": env_diffs,
        "environment": d5.observed_environment(),
        "d1_environment_frozen": d5.D1_ENVIRONMENT,
        "data": {"path": str(data_path), "sha256": digest},
        "history": {"path": str(hist_path), "sha256": hist_digest},
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (outdir / "d10_m5_specification.json").write_text(json.dumps(out, indent=2))

    L = []
    L.append("# D10 run report -- M5 inheritance")
    L.append("")
    stamp = ("  **NON-PROTOCOL: D1 environment not in force: "
             + "; ".join(env_diffs) + ".**") if env_diffs else ""
    L.append("Protocol state {} (freeze tag {}), row D10. Run (UTC): {}.{}"
             .format(PROTOCOL_TAG, d5.PROTOCOL_FREEZE_TAG, out["run_utc"],
                     stamp))
    L.append("")
    L.append("Operative addendum: `{}` (SHA-256 `{}`).".format(
        addendum_path.name, addendum_digest))
    L.append("")
    L.append("**D10 selects nothing.** It assembles the three "
             "already-selected indicator settings into one specification "
             "and verifies that they are mutually consistent. M5 inherits "
             "the pad ({}, {}) from M2, the FX lag of {} days from M3 and "
             "the search lag of {} days from M4, with no re-tuning: the "
             "ARIMA orders, seasonal orders, annual form and constant "
             "decision are those of the operative D5 selection, and no "
             "alternative to any inherited setting was fitted.".format(
                 b_pad, f_pad, fx_lag, search_lag))
    L.append("")
    L.append("## Consistency checks")
    L.append("")
    L.append("E8 assigns the Clark-West test to M2, M3, M4 and M5 against "
             "M1 on the grounds that each augmented model nests M1. That "
             "holds only if every augmented model differs from M1 in its "
             "exogenous regressors alone. Verified here rather than "
             "assumed: D7, D8 and D9 all describe the same operative M1 "
             "(**{}, ARIMA{}{}[{}]{}**, {} scale), all ran on training "
             "extraction `{}`, D8 and D9 on indicator history `{}`, D7 on "
             "cluster file `{}`, and none was a smoke run.".format(
                 cand, tuple(sel["selected"]["order"]),
                 tuple(sel["selected"]["seasonal_order"][:3]),
                 sel["selected"]["seasonal_order"][3],
                 "" if sel["selected"]["with_intercept"]
                 else ", no constant", sel["scale"], digest[:16],
                 hist_digest[:16], cl_digest[:16]))
    L.append("")
    L.append("## M5 specification")
    L.append("")
    L.append("| component | value |")
    L.append("|---|---|")
    L.append("| ARIMA order | {} |".format(tuple(sel["selected"]["order"])))
    L.append("| seasonal order | {}[{}] |".format(
        tuple(sel["selected"]["seasonal_order"][:3]),
        sel["selected"]["seasonal_order"][3]))
    L.append("| constant | {} |".format(
        "included" if sel["selected"]["with_intercept"] else "none (D4)"))
    L.append("| annual form | {} ({} columns) |".format(
        cand, len(ann["columns"])))
    L.append("| holiday pad (b, f) | ({}, {}) -- {} H_NY days, {} H_OT "
             "days |".format(b_pad, f_pad, hol_info["H_NY"],
                             hol_info["H_OT"]))
    L.append("| FX | lag {} days, unscaled |".format(fx_lag))
    L.append("| search | lag {} days, per {} queries |".format(
        search_lag, int(INDICATOR_SCALE_DIVISOR)))
    L.append("| exogenous columns | {} |".format(X5.shape[1]))
    L.append("")
    L.append("## Training ladder (AICc)")
    L.append("")
    L.append("| model | description | AICc | source |")
    L.append("|---|---|---|---|")
    for _, r in ladder.iterrows():
        L.append("| {} | {} | {} | {} |".format(
            r["model"], r["description"],
            "n/a" if r["aicc"] is None else "{:.4f}".format(r["aicc"]),
            r["source"]))
    L.append("")
    L.append("All five values come from fits of the same specification "
             "family on the same training sample under the same optimizer "
             "ceiling, so they are comparable with one another. They are "
             "descriptive only. No D10 result, training AICc, or later "
             "evaluation is allowed to alter the inherited M5 design; "
             "confirmatory model comparisons occur out of sample in "
             "Section E.")
    L.append("")
    L.append("## Next step")
    L.append("")
    L.append("Section D is complete. `d10_m5_specification.json` is the "
             "authoritative description of M5 for Section E, which "
             "re-estimates coefficients only (E3) and never re-selects "
             "orders, annual form, pad or lags (E4).")
    (outdir / "d10_report.md").write_text("\n".join(L) + "\n")

    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name="D10_{}".format(sel["scale"])):
            mlflow.set_tags({
                "protocol.row": ROW,
                "protocol.tag": PROTOCOL_TAG,
                "protocol.freeze_tag": d5.PROTOCOL_FREEZE_TAG,
                "protocol.addendum_sha256": addendum_digest,
                "archive_backfill": "false",
                "execution_recomputed": "true",
                "protocol.run_utc": out["run_utc"],
                "data_sha256": digest,
                "history_sha256": hist_digest,
                "clusters_sha256": cl_digest,
                "d5_selection_sha256":
                    source_artifact_hashes["d5_selection"],
                "d7_selection_sha256":
                    source_artifact_hashes["d7_selection"],
                "d8_selection_sha256":
                    source_artifact_hashes["d8_selection"],
                "d9_selection_sha256":
                    source_artifact_hashes["d9_selection"],
                "smoke_mode": "False",
            })
            mlflow.log_params({
                "scale": sel["scale"], "selects_nothing": "True",
                "pad_b": b_pad, "pad_f": f_pad,
                "fx_lag": fx_lag, "search_lag": search_lag,
                "search_scale_divisor": INDICATOR_SCALE_DIVISOR,
                "n_exog": int(X5.shape[1]),
                "m1_candidate": cand,
                "m1_order": str(sel["selected"]["order"]),
                "m1_seasonal_order": str(sel["selected"]["seasonal_order"]),
                "maxiter": DOWNSTREAM_MAXITER, "method": DOWNSTREAM_METHOD,
            })
            metrics = {"m5_aicc": m5_aicc,
                       "m5_iterations": float(mle.get("iterations", -1))}
            for m, v in out["ladder_training_aicc"].items():
                if v is not None:
                    metrics["ladder_aicc_" + m] = float(v)
            mlflow.log_metrics(metrics)
            for f in ("d10_m5_specification.json", "d10_ladder.csv",
                      "d10_report.md"):
                mlflow.log_artifact(str(outdir / f), artifact_path="d10")

            # D5/D7/D8/D9 result JSONs are not required to live in Git.
            # Archive the exact source artifacts consumed by D10 so the M5
            # inheritance record remains independently auditable.
            for source_artifact in (
                selection_path,
                d7_path,
                d8_path,
                d9_path,
            ):
                mlflow.log_artifact(
                    str(source_artifact),
                    artifact_path="inputs",
                )

            mlflow.log_artifact(str(addendum_path), artifact_path="protocol")
            mlflow.log_artifact(str(Path(__file__).resolve()),
                                artifact_path="source")
        print("logged to MLflow experiment '{}' at {}".format(
            args.experiment, tracking_uri))
    except Exception as exc:                               # noqa: BLE001
        sys.exit("MLflow logging FAILED: {}\nD10 artifacts were written to "
                 "disk, but this row is NOT considered complete until the "
                 "live run is present in MLflow.".format(exc))

    print("")
    print("D10 COMPLETE -- M5 = M1 + holidays({},{}) + FX(lag {}) + "
          "search(lag {}); nothing re-tuned.".format(
              b_pad, f_pad, fx_lag, search_lag))
    print("Section D is complete. M5 spec written for Section E.")
    print("outputs written to {}".format(outdir.resolve()))


if __name__ == "__main__":
    main()
