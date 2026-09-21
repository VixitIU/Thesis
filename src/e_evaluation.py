"""Section E -- frozen out-of-sample evaluation.

Operative protocol state: protocol-v1.8.

This script implements frozen rows E1-E15 after Section D model selection is
final. It deliberately does NOT perform any model selection. The operative
D10 artifact is treated as the authoritative description of M1-M5.

Frozen evaluation design
------------------------
E1   expanding estimation window.
E2   origins every 1 day from 2025-11-30 while origin + 28 days <=
     2026-07-31 (216 origins). This removes the restriction of
     evaluating only one weekday at horizons that are multiples of seven.
E3   coefficients are re-estimated at every origin.
E4   orders, annual form, holiday pad and indicator lags are never re-selected.
E5   reported horizons h = 7, 14, 28 days.
E6   holiday regressors are deterministic; indicator lags must use only values
     available by the forecast origin. The selected FX lag is 84 days and the
     selected search lag is 42 days. The search-vintage check retains the
     conservative 10-day all-weekday availability bound established in D9.
E7   MAE (primary), RMSE, MASE, MAPE and MdAPE are reported on the ORIGINAL
     count scale. The MASE denominator is the in-sample M0 same-calendar-date-
     previous-year MAE over 2024-07-01..2025-11-30 and is computed once.
E8   Clark-West, squared-error loss: M2, M3, M4 and M5 versus M1.
E9   modified Diebold-Mariano (HLN), squared-error loss: M1 versus M0.
E10  one-sided tests; t reference with n-1 degrees of freedom. The Harvey,
     Leybourne & Newbold small-sample factor is applied to the test statistic.
E11  Bartlett-kernel HAC truncation h-1, reflecting daily-spaced forecast
     origins. With daily origins, an h-day-ahead error sequence can overlap
     through h-1 daily lags. The 216 origins are a denser evaluation grid,
     not 216 independent replications; the unchanged test window limits the
     amount of independent information, especially at h=28. HAC addresses
     this dependence in the variance estimate, but finite-sample size
     distortion may remain at longer horizons, so confirmatory p-values are
     interpreted cautiously.
E12  Holm adjustment across H1a/H1b/H1c (M2/M3/M4 vs M1) separately at each
     horizon, FWER 5%; horizons are not mutually adjusted.
E13  H2 (M5 vs M1) is standalone at 5%, outside the Holm family.
E14  confirmatory decisions use squared-error Clark-West tests while MAE
     remains the primary descriptive accuracy measure.
E15  VIFs are computed on the frozen 884-day training design using a
     diagnostic-only constant for conventional centered VIF calculation.
     The constant is not included in any forecasting model, and VIF remains
     descriptive only.

Scale and inverse transformation
--------------------------------
All M1-M5 models are estimated on log(y + 1), as fixed by D12/D13. For a
forecast mean m_h and forecast variance s_h^2 on that transformed scale, the
point forecast returned to the count scale is the frozen bias-adjusted mean

    exp(m_h + s_h^2 / 2) - 1.

The unadjusted exp(m_h)-1 median is NOT used for any metric or confirmatory
loss. No clipping or rounding is applied to the point forecasts.

Benchmark M0
------------
M0 is deterministic seasonal naive: the forecast for a target date equals the
observed count on the same calendar date of the previous year. For 29 February,
where no same calendar date exists in the preceding non-leap year, the forecast
is defined by linear interpolation as the arithmetic mean of the observed counts
on 28 February and 1 March of the preceding year. M0 is already on the original
count scale and is not log-transformed.

Protocol-boundary safeguard
---------------------------
The default mode is --mode preflight. It validates every input, rebuilds all
regressors, verifies the 216-origin daily schedule and weekday coverage, checks
E6 availability, computes the frozen MASE denominator and E15 VIFs, checks that
no evaluation target has a zero actual (E7), and reproduces the frozen D10 M5
training fit on the original 884-day window as a hard AICc gate. It produces NO
test-window forecast.

--mode execute first repeats the same checks, then fits every M1-M5 origin
model and verifies optimizer convergence for ALL 1,080 fits BEFORE calling any
forecast method. Only after all fits pass is the first test-window forecast
produced. This reduces the risk of discovering a numerical convergence problem
after crossing the frozen protocol's first-forecast boundary.

Inputs
------
--data       full dense daily response spine, 2023-07-01..2026-07-31
--train-data exact 884-row training extraction used by the operative D10 run
--history    daily-dense extended indicator history with columns obs_date,
             fx_rub_per_thb and search_index
--clusters   holiday_clusters.csv used by D7/D10
--d10        operative d10_m5_specification.json produced after the v1.7 D5
             re-execution and the required D7-D10 re-execution

Outputs (--outdir, default results/e)
-------------------------------------
preflight mode:
    e_preflight.json
    e_vif.csv
    e_origin_schedule.csv
    e_preflight_report.md

execute mode additionally writes:
    e_origin_fits.csv
    e_forecasts.csv
    e_accuracy.csv
    e_tests.csv
    e_holm.csv
    e_results.json
    e_report.md

The result files are research artifacts and may contain proprietary aggregate
case information. They are not intended for the public repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pmdarima as pm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
             "PYTHONPATH): Section E reuses D7's cluster loader and holiday "
             "window builder so the forecast regressors cannot drift from "
             "the selected D7/D10 design.")


# ----------------------------- frozen constants ------------------------------
ROW = "E"
PROTOCOL_TAG = "protocol-v1.8"
PROTOCOL_FREEZE_TAG = "protocol-v1.0"
SOURCE_D10_PROTOCOL_TAG = "protocol-v1.7"
SOURCE_D5_PROTOCOL_TAG = "protocol-v1.7"

SPINE_START = pd.Timestamp("2023-07-01")
TRAIN_END = pd.Timestamp("2025-11-30")
SAMPLE_END = pd.Timestamp("2026-07-31")
N_TRAIN = 884
N_TOTAL = 1127

ORIGIN_START = TRAIN_END
ORIGIN_STEP_DAYS = 1
MAX_HORIZON = 28
HORIZONS = (7, 14, 28)
N_ORIGINS = 216

MASE_START = pd.Timestamp("2024-07-01")
MASE_END = TRAIN_END

DOWNSTREAM_MAXITER = 500
DOWNSTREAM_METHOD = "lbfgs"

FX_COL = "fx_rub_per_thb"
SEARCH_COL = "search_index"

# D9's conservative search-vintage availability requirement was 6 days for
# the represented week span plus a 4-day availability bound.
SEARCH_WEEK_SPAN_DAYS = 6
SEARCH_PUBLICATION_DELAY_BOUND_DAYS = 4
SEARCH_SAFE_AVAILABILITY_DAYS = (
    SEARCH_WEEK_SPAN_DAYS + SEARCH_PUBLICATION_DELAY_BOUND_DAYS
)

ALPHA = 0.05

# Final operative settings supplied by the completed v1.7 D5 -> D10 chain.
# These are consistency checks, not a second model-definition source.
EXPECTED_M1_CANDIDATE = "fourier_K1"
EXPECTED_ORDER = (0, 1, 2)
EXPECTED_SEASONAL_ORDER = (0, 0, 0, 7)
EXPECTED_WITH_INTERCEPT = False
EXPECTED_HOLIDAY_PAD = (3, 2)
EXPECTED_FX_LAG = 84
EXPECTED_SEARCH_LAG = 42
EXPECTED_N_EXOG_M5 = 6
# Origin 1 trains on exactly the frozen training window, so its M5 fit must
# reproduce the AICc recorded by D10.
D10_AICC_TOLERANCE = 1e-3


# ----------------------------- small utilities -------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit("Section E consistency check failed: " + msg)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_y_column(df: pd.DataFrame, date_col: str,
                   requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            sys.exit("y column '{}' not found; columns present: {}".format(
                requested, list(df.columns)))
        return requested
    others = [c for c in df.columns if c != date_col]
    if len(others) != 1:
        sys.exit("--y-col required (cannot infer); columns present: {}".format(
            list(df.columns)))
    return others[0]


def calendar_year_back(date: pd.Timestamp) -> pd.Timestamp:
    """Same calendar date in the previous year."""
    return pd.Timestamp(date) - pd.DateOffset(years=1)


def m0_forecast(y: pd.Series, target: pd.Timestamp) -> float:
    """Seasonal-naive M0 forecast with explicit leap-day interpolation."""
    target = pd.Timestamp(target)

    if target.month == 2 and target.day == 29:
        feb28 = pd.Timestamp(year=target.year - 1, month=2, day=28)
        mar1 = pd.Timestamp(year=target.year - 1, month=3, day=1)

        require(feb28 in y.index and mar1 in y.index,
                "M0 leap-day interpolation cannot find {} and {}."
                .format(feb28.date(), mar1.date()))

        return (
            float(y.loc[feb28]) +
            float(y.loc[mar1])
        ) / 2.0

    prev = calendar_year_back(target)
    require(prev in y.index,
            "M0 cannot find previous-year counterpart {} for target {}."
            .format(prev.date(), target.date()))

    return float(y.loc[prev])


def check_m0_leap_day_rule() -> None:
    """Verify the deterministic M0 leap-day convention."""
    idx = pd.to_datetime(["2023-02-28", "2023-03-01"])
    test_y = pd.Series([10.0, 14.0], index=idx)

    got = m0_forecast(test_y, pd.Timestamp("2024-02-29"))

    require(
        np.isclose(got, 12.0),
        "M0 leap-day interpolation check failed: expected 12.0, got {}."
        .format(got)
    )


def fmt_float(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return str(x)
    return ("{:.%df}" % digits).format(x)


# --------------------------------- loading -----------------------------------
def load_full_response(path: Path, date_col: str,
                       y_col: str | None) -> pd.Series:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        sys.exit("date column '{}' not found; columns present: {}".format(
            date_col, list(df.columns)))
    ycol = infer_y_column(df, date_col, y_col)

    idx = pd.DatetimeIndex(pd.to_datetime(df[date_col]).dt.normalize())
    s = pd.Series(pd.to_numeric(df[ycol]).to_numpy(), index=idx,
                  name=ycol).sort_index()

    problems = []
    if s.index.has_duplicates:
        problems.append("duplicate dates present")
    if len(s) != N_TOTAL:
        problems.append("expected {} rows, found {}".format(N_TOTAL, len(s)))
    if len(s) and (s.index[0] != SPINE_START or s.index[-1] != SAMPLE_END):
        problems.append("full data must run {}..{}; found {}..{}".format(
            SPINE_START.date(), SAMPLE_END.date(),
            s.index[0].date(), s.index[-1].date()))
    spine = pd.date_range(SPINE_START, SAMPLE_END, freq="D")
    if len(s) == N_TOTAL and not s.index.equals(spine):
        problems.append("full daily spine has gaps or extra dates")
    if s.isna().any():
        problems.append("NaNs in the response series")
    if (s < 0).any():
        problems.append("negative counts")
    vals = s.to_numpy(dtype=float)
    if not np.allclose(vals, np.round(vals)):
        problems.append("non-integer response values")
    if problems:
        sys.exit("full response validation failed: " + "; ".join(problems))
    return s.astype(float)


def load_exact_training(path: Path, date_col: str,
                        y_col: str | None) -> pd.Series:
    # Reuse D5's own training loader so the E input check follows the same
    # frozen validation logic used by model selection.
    a = SimpleNamespace(
        data=str(path),
        date_col=date_col,
        y_col=y_col,
        scale="log1p",
    )
    _, _, raw = d5.load_training_series(a)
    return raw.astype(float)


def load_history(path: Path, col: str, need_from: pd.Timestamp,
                 need_to: pd.Timestamp) -> pd.Series:
    h = pd.read_csv(path, parse_dates=["obs_date"]).set_index("obs_date")
    if col not in h.columns:
        sys.exit("{} has no column '{}'".format(path, col))
    s = h[col].astype(float).sort_index()
    if s.index.has_duplicates:
        sys.exit("duplicate dates in {} history".format(col))
    gaps = s.index.to_series().diff().dropna().dt.days
    if not (gaps == 1).all():
        sys.exit("the {} indicator history is not daily-dense".format(col))
    if s.index.min() > need_from or s.index.max() < need_to:
        sys.exit("the {} history does not cover {} -> {}; found {} -> {}"
                 .format(col, need_from.date(), need_to.date(),
                         s.index.min().date(), s.index.max().date()))
    if s.isna().any():
        sys.exit("NaNs in the {} history".format(col))
    return s


# ----------------------------- protocol parsing ------------------------------
def operative_spec(rec: dict) -> dict:
    require(rec.get("row") == "D10", "--d10 is not a D10 artifact.")
    require(rec.get("protocol_tag") == SOURCE_D10_PROTOCOL_TAG,
            "the D10 artifact is {}, expected {}.".format(
                rec.get("protocol_tag"), SOURCE_D10_PROTOCOL_TAG))
    require(rec.get("source_d5_protocol_tag") == SOURCE_D5_PROTOCOL_TAG,
            "D10 inherited D5 from {}, expected {}.".format(
                rec.get("source_d5_protocol_tag"), SOURCE_D5_PROTOCOL_TAG))
    require(rec.get("scale") == "log1p",
            "operative D10 scale is {}, expected log1p.".format(
                rec.get("scale")))

    spec = rec.get("m5_specification") or {}
    ann = spec.get("annual_regressor") or {}
    candidate = (
        "monthly"
        if ann.get("kind") == "monthly_dummies"
        else "fourier_K{}".format(ann.get("K"))
    )

    order = tuple(int(v) for v in spec.get("order", []))
    seasonal_order = tuple(int(v) for v in spec.get("seasonal_order", []))
    with_intercept = bool(spec.get("with_intercept"))
    holiday = spec.get("holiday") or {}
    fx = spec.get("fx") or {}
    search = spec.get("search") or {}

    b_pad = int(holiday.get("pad_b", -999))
    f_pad = int(holiday.get("pad_f", -999))
    fx_lag = int(fx.get("lag_days", -999))
    search_lag = int(search.get("lag_days", -999))
    search_divisor = float(search.get("scaling_divisor", 1000.0))

    require(candidate == EXPECTED_M1_CANDIDATE,
            "operative M1 candidate is {}, expected {} from the completed "
            "v1.7 D5 re-execution.".format(candidate,
                                           EXPECTED_M1_CANDIDATE))
    require(order == EXPECTED_ORDER,
            "operative M1 order is {}, expected {}.".format(
                order, EXPECTED_ORDER))
    require(seasonal_order == EXPECTED_SEASONAL_ORDER,
            "operative M1 seasonal order is {}, expected {}.".format(
                seasonal_order, EXPECTED_SEASONAL_ORDER))
    require(with_intercept == EXPECTED_WITH_INTERCEPT,
            "operative M1 intercept decision is {}, expected {}.".format(
                with_intercept, EXPECTED_WITH_INTERCEPT))
    require((b_pad, f_pad) == EXPECTED_HOLIDAY_PAD,
            "operative holiday pad is ({}, {}), expected {}.".format(
                b_pad, f_pad, EXPECTED_HOLIDAY_PAD))
    require(fx_lag == EXPECTED_FX_LAG,
            "operative FX lag is {}, expected {}.".format(
                fx_lag, EXPECTED_FX_LAG))
    require(search_lag == EXPECTED_SEARCH_LAG,
            "operative search lag is {}, expected {}.".format(
                search_lag, EXPECTED_SEARCH_LAG))
    require(search_divisor > 0,
            "search scaling divisor must be positive.")

    opt = rec.get("optimizer") or {}
    require(opt.get("method") == DOWNSTREAM_METHOD,
            "D10 optimizer method is {}, expected {}.".format(
                opt.get("method"), DOWNSTREAM_METHOD))
    require(int(opt.get("maxiter", -1)) == DOWNSTREAM_MAXITER,
            "D10 optimizer maxiter is {}, expected {}.".format(
                opt.get("maxiter"), DOWNSTREAM_MAXITER))

    if spec.get("n_exog") is not None:
        require(int(spec["n_exog"]) == EXPECTED_N_EXOG_M5,
                "D10 records {} M5 exogenous columns, expected {} for the "
                "final K1 + holidays + FX + search design.".format(
                    spec["n_exog"], EXPECTED_N_EXOG_M5))

    return {
        "candidate": candidate,
        "order": order,
        "seasonal_order": seasonal_order,
        "with_intercept": with_intercept,
        "annual_regressor": ann,
        "b_pad": b_pad,
        "f_pad": f_pad,
        "fx_lag": fx_lag,
        "search_lag": search_lag,
        "search_divisor": search_divisor,
        "holiday_clusters_sha256": holiday.get("clusters_sha256"),
    }


# ----------------------------- design builders -------------------------------
def build_origin_schedule() -> pd.DataFrame:
    rows = []
    origin = ORIGIN_START
    n = 0
    while origin + pd.Timedelta(MAX_HORIZON, "D") <= SAMPLE_END:
        n += 1
        row = {"origin_no": n, "origin": origin}
        for h in HORIZONS:
            row["target_h{}".format(h)] = origin + pd.Timedelta(h, "D")
        rows.append(row)
        origin += pd.Timedelta(ORIGIN_STEP_DAYS, "D")
    out = pd.DataFrame(rows)
    require(len(out) == N_ORIGINS,
            "origin rule produced {} origins, expected {}.".format(
                len(out), N_ORIGINS))
    require(pd.Timestamp(out.iloc[0]["origin"]) == ORIGIN_START,
            "first origin is not {}.".format(ORIGIN_START.date()))
    require(pd.Timestamp(out.iloc[-1]["origin"]) == pd.Timestamp("2026-07-03"),
            "last origin is not 2026-07-03.")
    require(pd.Timestamp(out.iloc[-1]["target_h28"]) == pd.Timestamp(
        "2026-07-31"), "last h=28 target is not 2026-07-31.")
    for h in HORIZONS:
        weekdays = pd.DatetimeIndex(out["target_h{}".format(h)]).dayofweek
        require(set(weekdays) == set(range(7)),
                "h={} targets do not cover all seven weekdays.".format(h))
    return out


def build_exog_blocks(spec: dict, full_idx: pd.DatetimeIndex,
                      fx: pd.Series, search: pd.Series,
                      clusters) -> tuple[dict[str, pd.DataFrame], dict]:
    d5.WITH_INTERCEPT = bool(spec["with_intercept"])
    X_ann = d5.build_candidate(spec["candidate"], full_idx)

    ann_record = spec["annual_regressor"]
    require(list(X_ann.columns) == list(ann_record.get("columns", [])),
            "annual regressors rebuilt as {}, D10 records {}.".format(
                list(X_ann.columns), ann_record.get("columns")))
    if ann_record.get("origin") is not None:
        require(str(ann_record["origin"]) == str(d5.FOURIER_ORIGIN),
                "Fourier origin in D10 is {}, D5 builder uses {}.".format(
                    ann_record["origin"], d5.FOURIER_ORIGIN))

    X_hol, hol_info = d7.holiday_regressors(
        clusters, full_idx, spec["b_pad"], spec["f_pad"]
    )
    require(list(X_hol.columns) == ["hol_H_NY", "hol_H_OT"],
            "unexpected holiday columns: {}".format(list(X_hol.columns)))

    hol_eval = X_hol.loc[X_hol.index > TRAIN_END]
    require(len(hol_eval) > 0
            and float(hol_eval.to_numpy(dtype=float).sum()) > 0.0,
            "the holiday regressors are identically zero after {}: the "
            "cluster file does not cover the evaluation window, so H1a "
            "and H2 would be tested on a design whose holiday columns "
            "carry no information.".format(TRAIN_END.date()))

    fx_lagged = fx.shift(spec["fx_lag"]).reindex(full_idx)
    search_lagged = (
        search.shift(spec["search_lag"]).reindex(full_idx)
        / spec["search_divisor"]
    )
    require(not fx_lagged.isna().any(),
            "FX lag leaves missing values on the response spine.")
    require(not search_lagged.isna().any(),
            "search lag leaves missing values on the response spine.")

    fx_name = "fx_lag{}".format(spec["fx_lag"])
    search_name = "search_lag{}_per{}".format(
        spec["search_lag"], int(spec["search_divisor"]))

    X_fx = fx_lagged.rename(fx_name).to_frame()
    X_search = search_lagged.rename(search_name).to_frame()

    blocks = {
        "M1": X_ann.copy(),
        "M2": pd.concat([X_ann, X_hol], axis=1),
        "M3": pd.concat([X_ann, X_fx], axis=1),
        "M4": pd.concat([X_ann, X_search], axis=1),
        "M5": pd.concat([X_ann, X_hol, X_fx, X_search], axis=1),
    }
    require(blocks["M5"].shape[1] == EXPECTED_N_EXOG_M5,
            "rebuilt M5 has {} columns, expected {}.".format(
                blocks["M5"].shape[1], EXPECTED_N_EXOG_M5))

    for model, X in blocks.items():
        require(X.index.equals(full_idx),
                "{} exogenous index does not match the response spine."
                .format(model))
        require(not X.isna().any().any(),
                "NaNs in {} exogenous matrix.".format(model))
        require(np.isfinite(X.to_numpy(dtype=float)).all(),
                "non-finite values in {} exogenous matrix.".format(model))

    info = {
        "annual_columns": list(X_ann.columns),
        "holiday_columns": list(X_hol.columns),
        "fx_column": fx_name,
        "search_column": search_name,
        "holiday_days": hol_info,
        "model_columns": {m: list(X.columns) for m, X in blocks.items()},
    }
    return blocks, info


def check_e6(schedule: pd.DataFrame, spec: dict) -> list[dict]:
    records = []
    for _, row in schedule.iterrows():
        origin = pd.Timestamp(row["origin"])
        for h in HORIZONS:
            target = origin + pd.Timedelta(h, "D")
            fx_source = target - pd.Timedelta(spec["fx_lag"], "D")
            sr_source = target - pd.Timedelta(spec["search_lag"], "D")
            sr_available = sr_source + pd.Timedelta(
                SEARCH_SAFE_AVAILABILITY_DAYS, "D")

            fx_ok = fx_source <= origin
            search_ok = sr_available <= origin
            require(fx_ok,
                    "E6 fails at origin {}, h={}: FX source {} is after "
                    "origin.".format(origin.date(), h, fx_source.date()))
            require(search_ok,
                    "E6 fails at origin {}, h={}: conservative search "
                    "availability date {} is after origin {}.".format(
                        origin.date(), h, sr_available.date(), origin.date()))
            records.append({
                "origin": origin,
                "horizon": h,
                "target": target,
                "fx_source_date": fx_source,
                "fx_available": fx_ok,
                "search_source_date": sr_source,
                "search_conservative_available_date": sr_available,
                "search_available": search_ok,
            })
    return records


# ------------------------------- E7 metrics ----------------------------------
def mase_denominator(y: pd.Series) -> float:
    errs = []

    for date in pd.date_range(MASE_START, MASE_END, freq="D"):
        require(date in y.index,
                "MASE denominator cannot find actual {}."
                .format(date.date()))

        naive = m0_forecast(y, date)
        errs.append(abs(float(y.loc[date]) - naive))

    scale = float(np.mean(errs))
    require(np.isfinite(scale) and scale > 0.0,
            "MASE denominator is {}, must be finite and > 0.".format(scale))
    return scale


def check_evaluation_actuals(y: pd.Series,
                             schedule: pd.DataFrame) -> list[pd.Timestamp]:
    targets = sorted({
        pd.Timestamp(row["origin"]) + pd.Timedelta(h, "D")
        for _, row in schedule.iterrows()
        for h in HORIZONS
    })
    zero_dates = [d for d in targets if float(y.loc[d]) == 0.0]
    if zero_dates:
        sys.exit(
            "E7 ZERO-ACTUAL CHECK FAILED. The frozen protocol treats a zero "
            "evaluation actual as a data error requiring re-verification "
            "against the source database. Zero target date(s): {}. No "
            "test-window forecast has been produced by this script."
            .format(", ".join(str(d.date()) for d in zero_dates))
        )
    return targets


def accuracy_table(forecasts: pd.DataFrame,
                   mase_scale: float) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        for model in ("M0", "M1", "M2", "M3", "M4", "M5"):
            g = forecasts[(forecasts["horizon"] == h)
                          & (forecasts["model"] == model)].copy()
            require(len(g) == N_ORIGINS,
                    "{} h={} has {} forecasts, expected {}.".format(
                        model, h, len(g), N_ORIGINS))
            actual = g["actual"].to_numpy(dtype=float)
            pred = g["forecast"].to_numpy(dtype=float)
            err = actual - pred
            ae = np.abs(err)
            ape = 100.0 * ae / actual
            rows.append({
                "horizon": h,
                "model": model,
                "n": len(g),
                "MAE": float(np.mean(ae)),
                "RMSE": float(np.sqrt(np.mean(err ** 2))),
                "MASE": float(np.mean(ae) / mase_scale),
                "MAPE": float(np.mean(ape)),
                "MdAPE": float(np.median(ape)),
            })
    return pd.DataFrame(rows)


# -------------------------- E8-E13 forecast tests -----------------------------
def hac_long_run_variance(x: np.ndarray, truncation: int) -> float:
    """Bartlett-kernel Newey-West long-run variance of a scalar series."""
    z = np.asarray(x, dtype=float)
    n = len(z)
    require(n >= 2, "HAC series needs at least two observations.")
    require(0 <= truncation < n,
            "invalid HAC truncation {} for n={}.".format(truncation, n))
    u = z - np.mean(z)
    gamma0 = float(np.dot(u, u) / n)
    lrv = gamma0
    for lag in range(1, truncation + 1):
        gamma = float(np.dot(u[lag:], u[:-lag]) / n)
        weight = 1.0 - lag / float(truncation + 1)
        lrv += 2.0 * weight * gamma
    # Tiny negative values can arise only through floating-point cancellation.
    require(lrv >= -1e-12,
            "HAC long-run variance is materially negative: {}.".format(lrv))
    return max(lrv, 0.0)


def hln_factor(n: int, horizon_days: int) -> tuple[float, int]:
    # The effective forecast horizon for the HLN correction is measured in
    # origin intervals. With daily origins ORIGIN_STEP_DAYS=1, k=h. Keeping
    # the expression generic ties the correction to the declared origin grid.
    k = int(math.ceil(horizon_days / ORIGIN_STEP_DAYS))
    inside = (n + 1.0 - 2.0 * k + (k * (k - 1.0) / n)) / n
    require(inside > 0.0,
            "HLN correction is undefined for n={}, k={}.".format(n, k))
    return float(math.sqrt(inside)), k


def one_sided_test_from_differential(differential: np.ndarray,
                                     horizon_days: int) -> dict:
    d = np.asarray(differential, dtype=float)
    n = len(d)
    q = int(math.ceil(horizon_days / ORIGIN_STEP_DAYS) - 1)
    lrv = hac_long_run_variance(d, q)
    var_mean = lrv / n
    mean_d = float(np.mean(d))

    if var_mean == 0.0:
        if mean_d > 0:
            raw_stat = float("inf")
        elif mean_d < 0:
            raw_stat = float("-inf")
        else:
            raw_stat = 0.0
    else:
        raw_stat = mean_d / math.sqrt(var_mean)

    factor, k = hln_factor(n, horizon_days)
    stat = raw_stat * factor
    pvalue = float(stats.t.sf(stat, df=n - 1))
    return {
        "n": n,
        "horizon": horizon_days,
        "effective_horizon_intervals": k,
        "hac_truncation": q,
        "mean_differential": mean_d,
        "hac_long_run_variance": lrv,
        "variance_of_mean": var_mean,
        "hln_factor": factor,
        "statistic_raw": raw_stat,
        "statistic_hln": stat,
        "df": n - 1,
        "pvalue_one_sided": pvalue,
    }


def pivot_forecasts(forecasts: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = forecasts[forecasts["horizon"] == horizon].copy()
    p = g.pivot(index="origin", columns="model", values="forecast")
    a = g.drop_duplicates("origin").set_index("origin")["actual"]
    p["actual"] = a
    p = p.sort_index()
    require(len(p) == N_ORIGINS,
            "h={} pivot has {} origins, expected {}.".format(
                horizon, len(p), N_ORIGINS))
    return p


def forecast_tests(forecasts: pd.DataFrame) -> tuple[pd.DataFrame,
                                                      pd.DataFrame]:
    rows = []
    holm_rows = []

    hypothesis_map = {
        "M2": "H1a_holidays",
        "M3": "H1b_fx",
        "M4": "H1c_search",
        "M5": "H2_combined",
    }

    for h in HORIZONS:
        p = pivot_forecasts(forecasts, h)
        actual = p["actual"].to_numpy(dtype=float)
        f1 = p["M1"].to_numpy(dtype=float)
        e1 = actual - f1

        # E8 Clark-West
        cw_pvals_h1 = []
        cw_row_indices_h1 = []
        for model in ("M2", "M3", "M4", "M5"):
            fa = p[model].to_numpy(dtype=float)
            ea = actual - fa
            d_cw = e1 ** 2 - ea ** 2 + (f1 - fa) ** 2
            r = one_sided_test_from_differential(d_cw, h)
            r.update({
                "test": "Clark-West",
                "comparison": "{} vs M1".format(model),
                "baseline": "M1",
                "alternative_model": model,
                "hypothesis": hypothesis_map[model],
                "positive_favours": model,
                "loss": "squared_error",
                "alpha": ALPHA,
                "reject_raw_5pct": bool(r["pvalue_one_sided"] < ALPHA),
                "multiplicity": (
                    "Holm within H1a-H1c at this horizon"
                    if model in ("M2", "M3", "M4")
                    else "standalone H2 at 5%"
                ),
            })
            rows.append(r)
            if model in ("M2", "M3", "M4"):
                cw_pvals_h1.append(r["pvalue_one_sided"])
                cw_row_indices_h1.append(len(rows) - 1)

        # E12 Holm H1a-H1c, separately within each horizon.
        reject, p_adj, _, _ = multipletests(
            cw_pvals_h1, alpha=ALPHA, method="holm"
        )
        for idx, rej, padj in zip(cw_row_indices_h1, reject, p_adj):
            rows[idx]["pvalue_holm"] = float(padj)
            rows[idx]["reject_holm_5pct"] = bool(rej)
            holm_rows.append({
                "horizon": h,
                "hypothesis": rows[idx]["hypothesis"],
                "comparison": rows[idx]["comparison"],
                "pvalue_raw": rows[idx]["pvalue_one_sided"],
                "pvalue_holm": float(padj),
                "reject_holm_5pct": bool(rej),
            })

        # E13 H2 is outside Holm.
        rows[-1]["pvalue_holm"] = None
        rows[-1]["reject_holm_5pct"] = None

        # E9 modified DM (HLN): positive differential favours M1 over M0.
        f0 = p["M0"].to_numpy(dtype=float)
        e0 = actual - f0
        d_dm = e0 ** 2 - e1 ** 2
        r = one_sided_test_from_differential(d_dm, h)
        r.update({
            "test": "modified Diebold-Mariano (HLN)",
            "comparison": "M1 vs M0",
            "baseline": "M0",
            "alternative_model": "M1",
            "hypothesis": "benchmark_M1_vs_M0",
            "positive_favours": "M1",
            "loss": "squared_error",
            "alpha": ALPHA,
            "reject_raw_5pct": bool(r["pvalue_one_sided"] < ALPHA),
            "multiplicity": "not in H1a-H1c Holm family",
            "pvalue_holm": None,
            "reject_holm_5pct": None,
        })
        rows.append(r)

    tests = pd.DataFrame(rows)
    holm = pd.DataFrame(holm_rows)
    return tests, holm


# -------------------------------- E15 VIF ------------------------------------
def vif_table(blocks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """E15 descriptive VIFs on the frozen training exogenous designs.

    E15 requires VIFs to be reported for M2-M5 but does not declare any
    threshold, exclusion rule or model change. A singular design therefore
    must not become an undeclared evaluation gate: where perfect
    multicollinearity makes a VIF infinite, it is reported as such.
    """
    rows = []
    train_idx = pd.date_range(SPINE_START, TRAIN_END, freq="D")
    for model in ("M2", "M3", "M4", "M5"):
        X = blocks[model].reindex(train_idx).astype(float)
        arr = X.to_numpy(dtype=float)
        require(np.isfinite(arr).all(),
                "{} training exogenous matrix contains non-finite values."
                .format(model))
        rank = int(np.linalg.matrix_rank(arr))
        rank_deficient = rank < arr.shape[1]
        # variance_inflation_factor regresses each column on the others
        # WITHOUT adding an intercept. On a design carrying no constant
        # (D4 fixed with_intercept=False) this returns uncentered VIFs,
        # which are not the conventional quantity: they are inflated by
        # the column means and are not comparable to published VIF rules
        # of thumb. A diagnostic-only constant is therefore prepended for
        # the VIF computation alone. It enters no estimated model: no fit
        # in this script uses it, and E15 declares no threshold, so this
        # affects reporting only and cannot change any model.
        arr_vif = np.column_stack([np.ones(arr.shape[0]), arr])
        for i, col in enumerate(X.columns):
            try:
                vif = float(variance_inflation_factor(arr_vif, i + 1))
            except Exception:
                vif = float("inf")
            rows.append({
                "model": model,
                "variable": col,
                "VIF": vif,
                "sample_start": SPINE_START.date().isoformat(),
                "sample_end": TRAIN_END.date().isoformat(),
                "n": len(X),
                "matrix_rank": rank,
                "n_columns": int(arr.shape[1]),
                "rank_deficient": rank_deficient,
                "diagnostic_constant_added": True,
                "role": "descriptive only; no threshold changes model",
            })
    return pd.DataFrame(rows)


def json_safe_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to strict-JSON-safe records."""
    records = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for key, value in rec.items():
            if value is None or (not isinstance(value, (list, dict)) and
                                 pd.isna(value)):
                clean[key] = None
            elif isinstance(value, np.generic):
                clean[key] = value.item()
            elif isinstance(value, (pd.Timestamp, datetime)):
                clean[key] = value.isoformat()
            else:
                clean[key] = value
        records.append(clean)
    return records


# ----------------------------- fitting / forecast ----------------------------
def fit_one(y_log: pd.Series, X: pd.DataFrame, spec: dict):
    model = pm.ARIMA(
        order=spec["order"],
        seasonal_order=spec["seasonal_order"],
        with_intercept=spec["with_intercept"],
        maxiter=DOWNSTREAM_MAXITER,
        method=DOWNSTREAM_METHOD,
        suppress_warnings=True,
    )
    model.fit(y_log.to_numpy(dtype=float), X=X.to_numpy(dtype=float))
    res = model.arima_res_
    mle = getattr(res, "mle_retvals", {}) or {}
    return model, {
        "aicc": float(res.aicc),
        "converged": bool(mle.get("converged", False)),
        "iterations": int(mle.get("iterations", -1)),
        "nobs_effective": int(res.nobs_effective),
        "k_params_total": int(np.asarray(res.params).shape[0]),
    }


def forecast_one(model, X_future: pd.DataFrame) -> tuple[np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray]:
    res = model.arima_res_
    pred = res.get_forecast(
        steps=len(X_future),
        exog=X_future.to_numpy(dtype=float),
    )
    mu = np.asarray(pred.predicted_mean, dtype=float)
    var = np.asarray(pred.var_pred_mean, dtype=float)
    require(len(mu) == len(X_future) and len(var) == len(X_future),
            "forecast output length does not match the future exogenous "
            "matrix.")
    require(np.isfinite(mu).all() and np.isfinite(var).all(),
            "non-finite transformed forecast mean or variance.")
    require((var >= 0.0).all(),
            "negative forecast variance returned by statsmodels.")
    count_mean = np.exp(mu + 0.5 * var) - 1.0
    require(np.isfinite(count_mean).all(),
            "non-finite count-scale forecast after bias adjustment.")
    return mu, var, count_mean


# ------------------------------ report writers -------------------------------
def write_preflight_report(path: Path, out: dict,
                           vif: pd.DataFrame) -> None:
    L = []
    L.append("# Section E preflight report")
    L.append("")
    L.append("Protocol state {} (freeze tag {}), row E. Run (UTC): {}."
             .format(PROTOCOL_TAG, PROTOCOL_FREEZE_TAG, out["run_utc"]))
    L.append("")
    L.append("**No test-window forecast was produced.** This preflight only "
             "validated the frozen evaluation design and inputs.")
    L.append("")
    L.append("## Operative ladder")
    L.append("")
    L.append("M1: {} + ARIMA{}{}[{}], no constant, log(y+1).".format(
        out["operative_spec"]["candidate"],
        tuple(out["operative_spec"]["order"]),
        tuple(out["operative_spec"]["seasonal_order"][:3]),
        out["operative_spec"]["seasonal_order"][3]))
    L.append("")
    L.append("M2 adds holidays pad ({}, {}); M3 adds FX lag {}; M4 adds "
             "search lag {}; M5 inherits all three. Nothing is re-selected."
             .format(out["operative_spec"]["b_pad"],
                     out["operative_spec"]["f_pad"],
                     out["operative_spec"]["fx_lag"],
                     out["operative_spec"]["search_lag"]))
    L.append("")
    L.append("## Evaluation schedule")
    L.append("")
    L.append("{} expanding-window origins, every {} days from {} through {}; "
             "reported horizons {}. The last h=28 target is {}.".format(
                 out["n_origins"], ORIGIN_STEP_DAYS,
                 out["first_origin"], out["last_origin"], list(HORIZONS),
                 out["last_target_h28"]))
    L.append("")
    L.append("E6 availability passed at every origin/horizon. The search "
             "check includes the conservative {}-day availability bound "
             "(6-day week span + 4-day publication-delay bound).".format(
                 SEARCH_SAFE_AVAILABILITY_DAYS))
    L.append("")
    L.append("The 216 daily origins are not treated as 216 independent "
             "replications. They sample the same fixed test window more "
             "densely, and adjacent multi-step forecast errors overlap. "
             "E11 therefore uses Bartlett/Newey-West truncation h-1. For "
             "intuition only, 216/h corresponds to about {:.1f}, {:.1f} "
             "and {:.1f} non-overlapping h-day spans at h=7,14,28; these "
             "figures are NOT used as degrees of freedom or as an effective "
             "sample size.".format(216/7, 216/14, 216/28))
    L.append("")
    L.append("## D10 replication gate")
    L.append("")
    gate = out["d10_replication_gate"]
    L.append("PASSED. Rebuilt M5 on the exact frozen 884-day training window: "
             "AICc {:.6f}; operative D10 AICc {:.6f}; absolute difference "
             "{:.6g} (tolerance {:g}). No rolling-origin estimation or "
             "test-window forecast is required for this check.".format(
                 gate["rebuilt_m5_aicc"], gate["d10_m5_aicc"],
                 gate["absolute_difference"], gate["tolerance"]))
    L.append("")
    L.append("## MASE scale")
    L.append("")
    L.append("M0 uses the same calendar date of the previous year; for 29 February, "
            "the benchmark uses the arithmetic mean of 28 February and 1 March of "
            "the previous year. In-sample M0 MAE over {}..{}: **{:.6f}**. "
            "This denominator is computed once and reused for all models and horizons."
            .format(MASE_START.date(), MASE_END.date(), out["mase_denominator"]))
    L.append("")
    L.append("## E15 VIF")
    L.append("")
    L.append("VIF is reported on the frozen 884-day training design only. "
             "A diagnostic-only constant is added when computing conventional "
             "centered VIFs; this constant is not included in any forecasting "
             "model. VIF is descriptive and has no selection threshold.")
    L.append("")
    L.append("| model | max VIF | variable |")
    L.append("|---|---:|---|")
    for model in ("M2", "M3", "M4", "M5"):
        g = vif[vif["model"] == model].sort_values("VIF", ascending=False)
        r = g.iloc[0]
        L.append("| {} | {:.4f} | {} |".format(
            model, r["VIF"], r["variable"]))
    L.append("")
    L.append("## Ready state")
    L.append("")
    L.append(("Preflight passed, including the D10 M5 AICc replication gate. "
              "Running the script with `--mode execute` will first fit and "
              "verify convergence for all {} M1-M5 origin models. Only after "
              "every fit converges will the script produce the first "
              "test-window forecast.")
             .format(N_ORIGINS * 5))
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_execution_report(path: Path, out: dict,
                           accuracy: pd.DataFrame,
                           tests: pd.DataFrame) -> None:
    L = []
    L.append("# Section E run report -- out-of-sample evaluation")
    L.append("")
    L.append("Protocol state {} (freeze tag {}), row E. Run (UTC): {}."
             .format(PROTOCOL_TAG, PROTOCOL_FREEZE_TAG, out["run_utc"]))
    L.append("")
    L.append("Model selection was final before this run. E1-E15 re-estimate "
             "coefficients only; no order, annual-form, holiday-pad or lag "
             "selection is performed here.")
    L.append("")
    L.append("All M1-M5 point forecasts were produced on log(y+1) and "
             "returned to the original count scale with the frozen "
             "bias-adjusted inverse exp(m_h + s_h^2/2) - 1. All descriptive "
             "metrics and confirmatory squared-error losses below therefore "
             "use count-scale forecasts.")
    L.append("")
    L.append("## Accuracy")
    L.append("")
    L.append("MAE is the primary descriptive metric (reported first).")
    L.append("")
    L.append("| h | model | MAE | RMSE | MASE | MAPE | MdAPE |")
    L.append("|---:|---|---:|---:|---:|---:|---:|")
    for _, r in accuracy.sort_values(["horizon", "model"]).iterrows():
        L.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |"
                 .format(int(r["horizon"]), r["model"], r["MAE"],
                         r["RMSE"], r["MASE"], r["MAPE"], r["MdAPE"]))
    L.append("")
    L.append("## Confirmatory tests")
    L.append("")
    L.append("Positive statistics favour the named alternative model. "
             "Clark-West H1a-H1c p-values are Holm-adjusted within each "
             "horizon; H2 is standalone at 5%. The M1-vs-M0 modified DM "
             "comparison is reported separately.")
    L.append("")
    L.append("| h | hypothesis | comparison | test | stat | p raw | p Holm | "
             "decision basis |")
    L.append("|---:|---|---|---|---:|---:|---:|---|")
    for _, r in tests.iterrows():
        if r["hypothesis"] in ("H1a_holidays", "H1b_fx", "H1c_search"):
            decision = "Holm 5%: {}".format(
                "reject" if bool(r["reject_holm_5pct"]) else "not reject")
        elif r["hypothesis"] == "H2_combined":
            decision = "standalone 5%: {}".format(
                "reject" if bool(r["reject_raw_5pct"]) else "not reject")
        else:
            decision = "standalone descriptive benchmark test: {}".format(
                "reject" if bool(r["reject_raw_5pct"]) else "not reject")
        ph = ("n/a" if pd.isna(r.get("pvalue_holm"))
              else "{:.6f}".format(float(r["pvalue_holm"])))
        L.append("| {} | {} | {} | {} | {:.4f} | {:.6f} | {} | {} |"
                 .format(int(r["horizon"]), r["hypothesis"],
                         r["comparison"], r["test"],
                         float(r["statistic_hln"]),
                         float(r["pvalue_one_sided"]), ph, decision))
    L.append("")
    L.append("## Dependence and finite-sample caveat")
    L.append("")
    L.append("The 216 daily origins do not represent 216 independent "
             "forecast experiments. The test window is unchanged and "
             "adjacent h-step forecast errors overlap. The declared "
             "Bartlett/Newey-West HAC truncation of h-1 adjusts the variance "
             "estimate for this serial dependence, and the HLN correction "
             "is applied in daily-origin units. However, at longer horizons "
             "-- especially h=28 -- the HAC bandwidth is large relative to "
             "the finite evaluation window, so finite-sample size distortion "
             "may remain. Confirmatory p-values are therefore interpreted "
             "cautiously and together with the descriptive accuracy measures; "
             "n=216 is not described as 216 independent replications.")
    L.append("")
    L.append("## Interpretation guard")
    L.append("")
    L.append("A non-significant holiday comparison means that the frozen "
             "holiday augmentation did not demonstrate incremental "
             "out-of-sample predictive improvement conditional on M1. It "
             "does not establish that holidays have no effect, because "
             "holiday timing can overlap with the deterministic annual "
             "seasonal component and the training sample spans fewer than "
             "two and a half annual cycles.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------- mlflow -----------------------------------
def log_mlflow(args, outdir: Path, out: dict,
               executed: bool) -> None:
    try:
        import mlflow

        project_root = Path(__file__).resolve().parents[1]
        tracking_uri = (project_root / "mlruns").as_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.experiment)
        run_name = "E_execute" if executed else "E_preflight"

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "protocol.row": ROW,
                "protocol.tag": PROTOCOL_TAG,
                "protocol.freeze_tag": PROTOCOL_FREEZE_TAG,
                "protocol.run_utc": out["run_utc"],
                "execution_mode": "execute" if executed else "preflight",
                "test_window_forecast_produced": str(bool(executed)),
                "data_sha256": out["hashes"]["full_data"],
                "train_data_sha256": out["hashes"]["train_data"],
                "history_sha256": out["hashes"]["history"],
                "clusters_sha256": out["hashes"]["clusters"],
                "d10_sha256": out["hashes"]["d10"],
            })
            mlflow.log_params({
                "window": "expanding",
                "n_origins": out["n_origins"],
                "origin_step_days": ORIGIN_STEP_DAYS,
                "horizons": str(list(HORIZONS)),
                "scale": "log1p",
                "inverse": "exp(m + variance/2) - 1",
                "mase_start": str(MASE_START.date()),
                "mase_end": str(MASE_END.date()),
                "maxiter": DOWNSTREAM_MAXITER,
                "method": DOWNSTREAM_METHOD,
                "m1_candidate": out["operative_spec"]["candidate"],
                "m1_order": str(out["operative_spec"]["order"]),
                "m1_seasonal_order":
                    str(out["operative_spec"]["seasonal_order"]),
                "holiday_pad": "({}, {})".format(
                    out["operative_spec"]["b_pad"],
                    out["operative_spec"]["f_pad"]),
                "fx_lag": out["operative_spec"]["fx_lag"],
                "search_lag": out["operative_spec"]["search_lag"],
                "search_safe_availability_days":
                    SEARCH_SAFE_AVAILABILITY_DAYS,
            })
            mlflow.log_metric("mase_denominator",
                              float(out["mase_denominator"]))

            if executed:
                acc = pd.read_csv(outdir / "e_accuracy.csv")
                tests = pd.read_csv(outdir / "e_tests.csv")
                for _, r in acc.iterrows():
                    prefix = "{}_h{}".format(r["model"], int(r["horizon"]))
                    mlflow.log_metric(prefix + "_mae", float(r["MAE"]))
                    mlflow.log_metric(prefix + "_rmse", float(r["RMSE"]))
                    mlflow.log_metric(prefix + "_mase", float(r["MASE"]))
                for _, r in tests.iterrows():
                    hyp = str(r["hypothesis"]).replace("_", "-")
                    key = "{}_h{}_p".format(hyp, int(r["horizon"]))
                    mlflow.log_metric(key, float(r["pvalue_one_sided"]))

            for path in sorted(outdir.glob("e_*")):
                if path.is_file():
                    mlflow.log_artifact(str(path), artifact_path="section_e")

            project_root = Path(__file__).resolve().parents[1]
            addendum_path = project_root / "docs" / "addenda.md"
            if addendum_path.exists():
                mlflow.log_artifact(str(addendum_path),
                                    artifact_path="protocol")
            mlflow.log_artifact(str(Path(__file__).resolve()),
                                artifact_path="source")

        print("logged to MLflow experiment '{}' at {}".format(
            args.experiment, tracking_uri))
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            "MLflow logging FAILED: {}\nSection E artifacts were written to "
            "disk, but this run is NOT considered complete until the live "
            "MLflow record is present.".format(exc)
        )


# ----------------------------------- main ------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Section E: frozen rolling-origin evaluation ({})"
                    .format(PROTOCOL_TAG),
    )
    ap.add_argument("--data", required=True,
                    help="full 1,127-row response spine through 2026-07-31")
    ap.add_argument("--train-data", required=True,
                    help="exact 884-row training CSV used by operative D10")
    ap.add_argument("--history", required=True,
                    help="daily-dense extended FX/search history")
    ap.add_argument("--clusters", required=True,
                    help="holiday_clusters.csv used by D7/D10")
    ap.add_argument("--d10", required=True,
                    help="operative d10_m5_specification.json")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--outdir", default="results/e")
    ap.add_argument("--mode", choices=["preflight", "execute"],
                    default="preflight")
    ap.add_argument("--experiment",
                    default="medical-assistance-demand-forecasting")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()

    env_diffs = d5.check_environment(args.allow_env_mismatch)

    data_path = Path(args.data)
    train_path = Path(args.train_data)
    history_path = Path(args.history)
    clusters_path = Path(args.clusters)
    d10_path = Path(args.d10)
    for label, path in (
        ("full response", data_path),
        ("training response", train_path),
        ("indicator history", history_path),
        ("holiday clusters", clusters_path),
        ("D10 specification", d10_path),
    ):
        require(path.exists(), "{} file is missing: {}".format(label, path))

    hashes = {
        "full_data": sha256_of(data_path),
        "train_data": sha256_of(train_path),
        "history": sha256_of(history_path),
        "clusters": sha256_of(clusters_path),
        "d10": sha256_of(d10_path),
    }

    d10_rec = load_json(d10_path)
    spec = operative_spec(d10_rec)
    m5_fit_rec = d10_rec.get("m5_training_fit") or {}
    require("aicc" in m5_fit_rec,
            "the D10 artifact has no m5_training_fit.aicc, so the D10 "
            "replication gate cannot be evaluated.")
    d10_m5_aicc = float(m5_fit_rec["aicc"])
    require(
        np.isfinite(d10_m5_aicc),
        "D10 does not contain a finite M5 training AICc."
    )

    # D10 provenance checks.
    require(hashes["train_data"] == d10_rec["data"]["sha256"],
            "--train-data does not match the training extraction recorded "
            "by D10.")
    require(hashes["history"] == d10_rec["history"]["sha256"],
            "--history does not match the indicator history recorded by D10.")
    if spec["holiday_clusters_sha256"]:
        require(hashes["clusters"] == spec["holiday_clusters_sha256"],
                "--clusters does not match the holiday file recorded by D10.")

    y_full = load_full_response(data_path, args.date_col, args.y_col)
    y_train_exact = load_exact_training(train_path, args.date_col, args.y_col)
    require(y_full.loc[SPINE_START:TRAIN_END].equals(y_train_exact),
            "the training segment of --data does not exactly match "
            "--train-data. Section E must begin from the same frozen response "
            "history used by Section D.")

    max_lag = max(spec["fx_lag"], spec["search_lag"])
    need_from = SPINE_START - pd.Timedelta(max_lag, "D")
    fx_need_to = SAMPLE_END - pd.Timedelta(spec["fx_lag"], "D")
    sr_need_to = SAMPLE_END - pd.Timedelta(spec["search_lag"], "D")
    fx = load_history(history_path, FX_COL, need_from, fx_need_to)
    sr = load_history(history_path, SEARCH_COL, need_from, sr_need_to)

    full_idx = pd.date_range(SPINE_START, SAMPLE_END, freq="D")
    clusters = d7.load_clusters(clusters_path)
    blocks, design_info = build_exog_blocks(spec, full_idx, fx, sr, clusters)

    schedule = build_origin_schedule()
    e6 = check_e6(schedule, spec)
    targets = check_evaluation_actuals(y_full, schedule)
    check_m0_leap_day_rule()
    mase_scale = mase_denominator(y_full)
    vif = vif_table(blocks)

    # ---- D10 replication gate: part of preflight -------------------------
    # Reproduce the frozen D10 M5 fit on exactly the original 884-day
    # training window before preflight can pass. This uses no test-window
    # forecast and performs no rolling-origin estimation. If the AICc does
    # not match the authoritative D10 artifact, stop immediately.
    print("")
    print("D10 REPLICATION GATE -- checking frozen M5 training fit.")

    gate_idx = pd.date_range(SPINE_START, TRAIN_END, freq="D")
    gate_y_log = np.log1p(y_full.reindex(gate_idx).astype(float))
    gate_X = blocks["M5"].reindex(gate_idx)

    _, gate_fit = fit_one(gate_y_log, gate_X, spec)

    require(
        gate_fit["converged"],
        "D10 replication gate failed: M5 did not converge on the frozen "
        "884-day training window."
    )

    d10_aicc_delta = abs(gate_fit["aicc"] - d10_m5_aicc)
    require(
        d10_aicc_delta <= D10_AICC_TOLERANCE,
        "D10 replication gate FAILED: rebuilt M5 AICc is {:.6f}, "
        "but operative D10 records {:.6f} "
        "(absolute difference {:.6f} exceeds tolerance {:g}). "
        "Section E preflight has failed; rolling-origin estimation and "
        "test-window forecasting have not begun.".format(
            gate_fit["aicc"],
            d10_m5_aicc,
            d10_aicc_delta,
            D10_AICC_TOLERANCE
        )
    )

    print(
        "D10 REPLICATION GATE PASSED -- M5 AICc {:.6f} "
        "matches operative D10 {:.6f}.".format(
            gate_fit["aicc"],
            d10_m5_aicc
        )
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(outdir / "e_origin_schedule.csv", index=False)
    vif.to_csv(outdir / "e_vif.csv", index=False)

    run_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base_out = {
        "protocol_tag": PROTOCOL_TAG,
        "protocol_freeze_tag": PROTOCOL_FREEZE_TAG,
        "row": ROW,
        "mode": args.mode,
        "run_utc": run_utc,
        "environment_mismatch": env_diffs,
        "environment": d5.observed_environment(),
        "d1_environment_frozen": d5.D1_ENVIRONMENT,
        "hashes": hashes,
        "operative_spec": {
            "candidate": spec["candidate"],
            "order": list(spec["order"]),
            "seasonal_order": list(spec["seasonal_order"]),
            "with_intercept": spec["with_intercept"],
            "b_pad": spec["b_pad"],
            "f_pad": spec["f_pad"],
            "fx_lag": spec["fx_lag"],
            "search_lag": spec["search_lag"],
            "search_divisor": spec["search_divisor"],
            "model_columns": design_info["model_columns"],
        },
        "evaluation": {
            "window": "expanding",
            "origin_step_days": ORIGIN_STEP_DAYS,
            "horizons": list(HORIZONS),
            "reestimate": "coefficients only",
            "reselection": False,
            "forecast_scale": "original count scale",
            "inverse": "exp(m_h + s_h^2 / 2) - 1",
            "m0": (
                "same calendar date of previous year; "
                "29 February = arithmetic mean of 28 February and "
                "1 March of previous year"
            ),
            "search_safe_availability_days":
                SEARCH_SAFE_AVAILABILITY_DAYS,
        },
        "n_origins": len(schedule),
        "first_origin": str(pd.Timestamp(schedule.iloc[0]["origin"]).date()),
        "last_origin": str(pd.Timestamp(schedule.iloc[-1]["origin"]).date()),
        "last_target_h28": str(pd.Timestamp(
            schedule.iloc[-1]["target_h28"]).date()),
        "n_unique_evaluation_targets": len(targets),
        "e6_checks": len(e6),
        "mase_denominator": mase_scale,
        "mase_window": [str(MASE_START.date()), str(MASE_END.date())],
        "d10_replication_gate": {
            "passed": True,
            "d10_m5_aicc": d10_m5_aicc,
            "rebuilt_m5_aicc": gate_fit["aicc"],
            "absolute_difference": d10_aicc_delta,
            "tolerance": D10_AICC_TOLERANCE,
            "training_window": [str(SPINE_START.date()), str(TRAIN_END.date())],
        },
        "vif_basis": "frozen training design only; exact exogenous matrix; "
                     "diagnostic-only constant added for conventional centered "
                     "VIF; descriptive only",
        "test_window_forecast_produced": False,
    }
    (outdir / "e_preflight.json").write_text(
        json.dumps(base_out, indent=2), encoding="utf-8"
    )
    write_preflight_report(outdir / "e_preflight_report.md", base_out, vif)

    print("")
    print("SECTION E PREFLIGHT PASSED")
    print("origins: {} ({} through {}), horizons {}".format(
        len(schedule), base_out["first_origin"], base_out["last_origin"],
        list(HORIZONS)))
    print("M1: {} ARIMA{}{}[{}], no constant; log(y+1)".format(
        spec["candidate"], spec["order"], spec["seasonal_order"][:3],
        spec["seasonal_order"][3]))
    print("M5: + holidays({},{}) + FX(lag {}) + search(lag {}) / {}".format(
        spec["b_pad"], spec["f_pad"], spec["fx_lag"],
        spec["search_lag"], int(spec["search_divisor"])))
    print("MASE denominator: {:.6f}".format(mase_scale))
    print("E6 availability checks passed: {}".format(len(e6)))
    print("zero evaluation actuals: none")
    print("D10 replication gate: PASSED (rebuilt M5 AICc {:.6f}; D10 {:.6f})"
          .format(gate_fit["aicc"], d10_m5_aicc))

    if args.mode == "preflight":
        log_mlflow(args, outdir, base_out, executed=False)
        print("NO TEST-WINDOW FORECAST PRODUCED.")
        print("Review e_preflight_report.md, then run with --mode execute "
              "when ready to cross the first-forecast boundary.")
        print("outputs written to {}".format(outdir.resolve()))
        return

    # ---- pass 1: fit ALL origin models, produce NO forecast yet ---------
    print("")
    total_origin_models = N_ORIGINS * len(("M1", "M2", "M3", "M4", "M5"))
    print("PASS 1 -- fitting all {} origin/model specifications. No forecast "
          "method is called until every fit has converged.".format(
              total_origin_models))
    fitted = {}
    fit_rows = []
    models = ("M1", "M2", "M3", "M4", "M5")

    for _, srw in schedule.iterrows():
        origin_no = int(srw["origin_no"])
        origin = pd.Timestamp(srw["origin"])
        train_idx = pd.date_range(SPINE_START, origin, freq="D")
        y_log = np.log1p(y_full.reindex(train_idx).astype(float))
        require(not y_log.isna().any(),
                "response missing in expanding window ending {}.".format(
                    origin.date()))

        for model_name in models:
            X_train = blocks[model_name].reindex(train_idx)
            t0 = time.time()
            model, frec = fit_one(y_log, X_train, spec)
            seconds = time.time() - t0
            fit_rows.append({
                "origin_no": origin_no,
                "origin": origin,
                "model": model_name,
                "n_train": len(train_idx),
                "aicc": frec["aicc"],
                "converged": frec["converged"],
                "iterations": frec["iterations"],
                "nobs_effective": frec["nobs_effective"],
                "k_params_total": frec["k_params_total"],
                "seconds": round(seconds, 3),
            })
            print("origin {:02d}/{} {} {}: converged {} in {} iterations "
                  "({:.1f}s)".format(
                      origin_no, N_ORIGINS, origin.date(), model_name,
                      frec["converged"], frec["iterations"], seconds),
                  flush=True)
            if not frec["converged"]:
                pd.DataFrame(fit_rows).to_csv(
                    outdir / "e_origin_fits.csv", index=False)
                sys.exit(
                    "EXECUTION PAUSED -- {} at origin {} did not converge "
                    "under method='{}', maxiter={}. No test-window forecast "
                    "has been produced by this script because forecasting "
                    "begins only after all {} fits pass.".format(
                        model_name, origin.date(), DOWNSTREAM_METHOD,
                        DOWNSTREAM_MAXITER, total_origin_models)
                )

            fitted[(origin, model_name)] = model

    fit_table = pd.DataFrame(fit_rows)
    fit_table.to_csv(outdir / "e_origin_fits.csv", index=False)
    require(len(fit_table) == N_ORIGINS * len(models),
            "fit table has {} rows, expected {}.".format(
                len(fit_table), N_ORIGINS * len(models)))
    require(fit_table["converged"].all(),
            "not every origin/model fit converged.")

    # ---- pass 2: first forecast boundary is crossed here ---------------
    print("")
    print("PASS 2 -- all {} fits converged. Producing frozen test-window "
          "forecasts now. Model selection is final and unchanged.".format(
              total_origin_models))
    forecast_rows = []

    for _, srw in schedule.iterrows():
        origin_no = int(srw["origin_no"])
        origin = pd.Timestamp(srw["origin"])
        future_idx = pd.date_range(
            origin + pd.Timedelta(1, "D"),
            origin + pd.Timedelta(MAX_HORIZON, "D"),
            freq="D",
        )

        # M0 is generated only after the convergence gate above, so even the
        # deterministic benchmark does not cross the first-forecast boundary
        # during pass 1.
        for h in HORIZONS:
            target = origin + pd.Timedelta(h, "D")
            m0 = m0_forecast(y_full, target)
            prev = calendar_year_back(target)
            forecast_rows.append({
                "origin_no": origin_no,
                "origin": origin,
                "horizon": h,
                "target": target,
                "model": "M0",
                "actual": float(y_full.loc[target]),
                "forecast": m0,
                "forecast_log_mean": np.nan,
                "forecast_log_variance": np.nan,
                "inverse": "not applicable -- original-scale benchmark",
                "m0_source_date": (
                    "interpolated: {} + {}".format(
                        pd.Timestamp(target.year - 1, 2, 28).date(),
                        pd.Timestamp(target.year - 1, 3, 1).date()
                    )
                    if target.month == 2 and target.day == 29
                    else str(prev.date())
                ),
            })

        for model_name in models:
            X_future = blocks[model_name].reindex(future_idx)
            mu, var, fc = forecast_one(
                fitted[(origin, model_name)], X_future
            )
            for h in HORIZONS:
                i = h - 1
                target = origin + pd.Timedelta(h, "D")
                forecast_rows.append({
                    "origin_no": origin_no,
                    "origin": origin,
                    "horizon": h,
                    "target": target,
                    "model": model_name,
                    "actual": float(y_full.loc[target]),
                    "forecast": float(fc[i]),
                    "forecast_log_mean": float(mu[i]),
                    "forecast_log_variance": float(var[i]),
                    "inverse": "exp(m + variance/2) - 1",
                    "m0_source_date": pd.NaT,
                })

    forecasts = pd.DataFrame(forecast_rows).sort_values(
        ["horizon", "origin", "model"], kind="mergesort"
    ).reset_index(drop=True)
    expected_forecasts = N_ORIGINS * len(HORIZONS) * 6
    require(len(forecasts) == expected_forecasts,
            "forecast table has {} rows, expected {}.".format(
                len(forecasts), expected_forecasts))
    require(np.isfinite(forecasts["forecast"].to_numpy(dtype=float)).all(),
            "non-finite count-scale point forecast in final table.")
    forecasts.to_csv(outdir / "e_forecasts.csv", index=False)

    accuracy = accuracy_table(forecasts, mase_scale)
    accuracy.to_csv(outdir / "e_accuracy.csv", index=False)

    tests, holm = forecast_tests(forecasts)
    tests.to_csv(outdir / "e_tests.csv", index=False)
    holm.to_csv(outdir / "e_holm.csv", index=False)

    final_out = dict(base_out)
    final_out["mode"] = "execute"
    final_out["test_window_forecast_produced"] = True
    final_out["first_forecast_boundary_crossed_in"] = (
        "pass 2, only after all {} M1-M5 expanding-window fits converged"
        .format(N_ORIGINS * 5)
    )
    final_out["counts"] = {
        "origin_model_fits": len(fit_table),
        "forecast_rows": len(forecasts),
        "accuracy_rows": len(accuracy),
        "test_rows": len(tests),
        "holm_rows": len(holm),
    }
    final_out["accuracy"] = json_safe_records(accuracy)
    final_out["tests"] = json_safe_records(tests)
    final_out["holm"] = json_safe_records(holm)

    (outdir / "e_results.json").write_text(
        json.dumps(final_out, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    write_execution_report(outdir / "e_report.md", final_out,
                           accuracy, tests)

    log_mlflow(args, outdir, final_out, executed=True)

    print("")
    print("SECTION E COMPLETE")
    print("all accuracy measures use original-scale, bias-adjusted point "
          "forecasts for M1-M5")
    print("MAE is primary descriptive metric; Clark-West uses squared-error "
          "loss for H1a-H1c and H2")
    print("outputs written to {}".format(outdir.resolve()))


if __name__ == "__main__":
    main()
