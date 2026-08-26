"""D9 -- search-interest lag selection.

Operative protocol state: protocol-v1.6.

Frozen D9:
    candidate lags 28, 35, ..., 119 (multiples of 7); same
    five-candidate CCF then AICc rule.

"Same rule" points back to D8: the five largest |CCF| on the training
sample proceed to AICc, and ties go to the shorter lag. The addendum of
25 August 2026 operationalises the CCF for both rows as Box-Jenkins
prewhitening, per indicator: an auxiliary ARIMA filter is selected on
the indicator values corresponding to the frozen 884-day training window
only, using the D5 order bounds, s = 7, stepwise AICc, d <= 1, D <= 1,
KPSS for non-seasonal differencing, OCSB for seasonal differencing and
automatic intercept handling; the selected fixed filter is then applied
unchanged to the extended indicator history and to the operative
transformed response, with no response-side ARIMA estimated separately.
Raw CCF values are diagnostic only. If the filter fails to converge,
execution pauses before screening.

The filter is fitted to THIS indicator, not reused from D8. The search
index is a different series with different dynamics: C3-3 step-expands
four summed weekly queries, so the daily series is a step function with
seven-day plateaus, and the candidate lags are multiples of 7 to match.

C3-4 vintage
------------
A week's value may be used only after its publication date. The
step-expanded daily series represents seven-day weekly values. A week
ending six days after its first day gives a conservative all-weekday
availability condition of lag >= 6 + observed publication delay. This
does not alter the frozen D9 lag grid: candidates remain exactly
28, 35, ..., 119. The observed delay is supplied on the command line,
verified against the full frozen candidate set before fitting, and
recorded in the artifact as required by C3-4.

Stage 2 -- AICc
---------------
Per the addendum of 20 August 2026, the operative M1 orders and annual
form are held fixed and only the candidate lag varies. M4 = M1 + search
index at lag L; neither the holiday regressors (M2) nor FX (M3) are
present. The original search-index units are used for Stage-1 CCF
identification; only the Stage-2 exogenous regressor is divided by 1000
for numerical conditioning, an exact coefficient reparameterisation.
Per the addendum of 24 August 2026, every candidate is fitted
under a common ceiling (maxiter = 500), convergence is recorded, and
execution PAUSES rather than excluding a fit that fails to converge.

Lagged values come from indicator_history.csv, which extends the series
back before the spine start; aligned_train.csv cannot supply a 119-day
lag for the first training day. E6 is satisfied by construction: every
candidate lag is at least 28 days, the longest horizon, so all values
predate the forecast origin.

Outputs (--outdir, default results/d9)
--------------------------------------
    d9_ccf_screen.csv     all 14 candidate lags, prewhitened and raw
    d9_lag_grid.csv       the five AICc candidates
    d9_lag_selection.json machine-readable lag -> M4 and D10
    d9_report.md          prose record for the research log

No proprietary artifact is written: the tables carry correlations, AICc
values and lags only. The live run is recorded in MLflow together with
the operative addendum and the source script.
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

# The project already has an existing local MLflow FileStore.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:
    import d5_baseline_order_selection as d5
except ImportError:
    sys.exit("d5_baseline_order_selection.py must be importable (same "
             "directory or on PYTHONPATH): D9 reuses its loader, its "
             "annual-form builders and its D1 environment guard.")

# ----------------------------- frozen constants ------------------------------
ROW = "D9"
# D9 candidate lags: 28, 35, ..., 119 (multiples of 7).
LAG_STEP = 7
LAG_VALUES = list(range(28, 119 + 1, LAG_STEP))
LAG_MIN, LAG_MAX = min(LAG_VALUES), max(LAG_VALUES)

# C3-4 vintage rule. A week's value may be used only after that week's
# publication date. The search series is step-expanded from weekly values.
# A seven-day week runs from its first day through first_day + 6, hence the
# maximum wait from the start of a represented week to its completion is
# six days. For a lagged weekly value to be known by forecast date t for
# every weekday represented in the daily spine, the conservative condition
# is lag >= 6 + observed publication delay. This availability check is
# separate from the frozen D9 lag grid, which remains 28, 35, ..., 119.
# The observed delay is supplied on the command line and recorded.
MAX_DAYS_WEEK_START_TO_END = 6

# Numerical conditioning of the Stage-2 search regressor only.
#
# Stage 1 (prewhitening and CCF screening) uses the original frozen
# search_index values without rescaling. After the five candidate lags
# have been selected, Stage 2 expresses the exogenous search regressor in
# thousands of queries to improve optimizer conditioning.
#
# For an exogenous regressor this is an exact reparameterisation:
# X* = X / c and beta* = c * beta, so beta* X* = beta X. At the same
# optimum, fitted values, likelihood, parameter count, AICc and forecasts
# are unchanged; only the numerical scale of the coefficient changes.
# No candidate lag, CCF value, threshold, or selection rule is altered.
#
# Downstream coefficients using this representation are per
# INDICATOR_SCALE_DIVISOR queries rather than per one query.
INDICATOR_SCALE_DIVISOR = 1000.0
N_SCREEN = 5                       # five largest |CCF| proceed to AICc
IND_COL = "search_index"

# 24 Aug 2026 addendum.
DOWNSTREAM_MAXITER = 500
DOWNSTREAM_METHOD = "lbfgs"

# Recorded implementation choice: the prewhitening filter (see above).
FILTER_MAX_D = 1
FILTER_MAX_BIG_D = 1
FILTER_SEASONAL = True
FILTER_TEST = "kpss"
FILTER_SEASONAL_TEST = "ocsb"
FILTER_WITH_INTERCEPT = "auto"

# Frozen public indicator-history artifact built immediately before D8/D9.
FROZEN_HISTORY_SHA256 = (
    "ae994353fa0310d28240fb3740c5f6cf61d11be3847293f1ca2ab84243323f99"
)

PROTOCOL_TAG = "protocol-v1.6"

# Lineage pin, as at D7: the operative M1 must be the artifact produced by
# the transformed-scale D5 execution under protocol-v1.3, not a later
# re-run. Any legitimate re-execution of D5 under a later state requires
# this constant to be updated deliberately.
SOURCE_D5_PROTOCOL_TAG = "protocol-v1.3"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_history(path: Path, need_from: pd.Timestamp,
                 need_to: pd.Timestamp) -> pd.Series:
    h = pd.read_csv(path, parse_dates=["obs_date"]).set_index("obs_date")
    if IND_COL not in h.columns:
        sys.exit("{} has no column '{}'; found {}".format(
            path, IND_COL, list(h.columns)))
    s = h[IND_COL].astype(float).sort_index()
    if not s.index.is_unique:
        sys.exit("duplicate dates in the indicator history")
    gaps = s.index.to_series().diff().dropna().dt.days
    if not (gaps == 1).all():
        sys.exit("the indicator history is not daily-dense")
    if s.index.min() > need_from:
        sys.exit("the indicator history starts {} but a {}-day lag on the "
                 "first training day needs {}. Rebuild it with a longer "
                 "lead-in.".format(s.index.min().date(), LAG_MAX,
                                   need_from.date()))
    if s.index.max() < need_to:
        sys.exit("the indicator history ends {} but the training window "
                 "runs to {}".format(s.index.max().date(), need_to.date()))
    if s.isna().any():
        sys.exit("NaNs in the indicator history")
    return s


def prewhiten(x_train: pd.Series, x_history: pd.Series,
              y: pd.Series, bounds: dict):
    """Box-Jenkins prewhitening with training-only filter estimation.

    The auxiliary ARIMA specification and parameters are estimated on indicator
    values aligned exactly to the frozen response training window. The
    fitted state-space model is then cloned and filtered, without
    re-estimation, over (a) the extended indicator history and (b) the response.
    """
    if not x_train.index.equals(y.index):
        sys.exit("indicator filter-estimation index must equal the response "
                 "training index exactly.")
    if x_train.isna().any() or x_history.isna().any() or y.isna().any():
        sys.exit("NaNs are not permitted in prewhitening inputs.")

    filt = pm.auto_arima(
        x_train.to_numpy(dtype=float),
        d=None, D=None,
        m=d5.SEASONAL_M,
        seasonal=FILTER_SEASONAL,
        max_d=FILTER_MAX_D,
        max_D=FILTER_MAX_BIG_D,
        test=FILTER_TEST,
        seasonal_test=FILTER_SEASONAL_TEST,
        stepwise=True,
        information_criterion=d5.IC,
        with_intercept=FILTER_WITH_INTERCEPT,
        method=DOWNSTREAM_METHOD,
        maxiter=DOWNSTREAM_MAXITER,
        error_action="ignore",
        suppress_warnings=True,
        **bounds,
    )

    source_res = filt.arima_res_
    mle = getattr(source_res, "mle_retvals", {}) or {}
    converged = bool(mle.get("converged", False))
    iterations = int(mle.get("iterations", -1))

    if not converged:
        print("")
        print("EXECUTION PAUSED -- the selected search prewhitening model did "
              "not converge at maxiter = {}. The filter is upstream of the "
              "five CCF candidates, so D9 is not screened or finalized."
              .format(DOWNSTREAM_MAXITER))
        print("selected filter: ARIMA{}{}[{}], with_intercept={}, "
              "iterations={}".format(
                  tuple(filt.order),
                  tuple(filt.seasonal_order[:3]),
                  filt.seasonal_order[3],
                  bool(filt.with_intercept),
                  iterations))
        sys.exit(2)

    # Clone the exact fitted SARIMAX specification. This preserves trend /
    # intercept handling and every state-space option; only endog changes.
    # No parameters are estimated in either application.
    x_mod = source_res.model.clone(x_history.to_numpy(dtype=float))
    y_mod = source_res.model.clone(y.to_numpy(dtype=float))

    if list(x_mod.param_names) != list(source_res.model.param_names):
        sys.exit("cloned indicator-history filter parameterization differs from "
                 "the fitted whitening model.")
    if list(y_mod.param_names) != list(source_res.model.param_names):
        sys.exit("cloned response filter parameterization differs from "
                 "the fitted whitening model.")

    x_res = x_mod.filter(source_res.params)
    y_res = y_mod.filter(source_res.params)

    alpha = pd.Series(np.asarray(x_res.resid, dtype=float),
                      index=x_history.index)
    beta = pd.Series(np.asarray(y_res.resid, dtype=float),
                     index=y.index)

    burn_ind = int(getattr(x_res, "loglikelihood_burn", 0) or 0)
    burn_y = int(getattr(y_res, "loglikelihood_burn", 0) or 0)
    alpha = alpha.iloc[burn_ind:]
    beta = beta.iloc[burn_y:]

    # All 14 frozen D9 lags must use the same response dates. The common
    # start also guarantees that alpha_{t-LAG_MAX} exists after its own
    # state-space burn.
    common_start = max(
        beta.index.min(),
        alpha.index.min() + pd.Timedelta(LAG_MAX, "D"),
    )
    beta = beta.loc[common_start:]
    if len(beta) < 30:
        sys.exit("prewhitening leaves only {} common response dates after "
                 "state-space burn and lag coverage.".format(len(beta)))

    info = {
        "order": list(filt.order),
        "seasonal_order": list(filt.seasonal_order),
        "with_intercept": bool(filt.with_intercept),
        "information_criterion": d5.IC,
        "aicc": float(source_res.info_criteria("aicc")),
        "converged": converged,
        "iterations": iterations,
        "method": DOWNSTREAM_METHOD,
        "maxiter": DOWNSTREAM_MAXITER,
        "d_test": FILTER_TEST,
        "D_test": FILTER_SEASONAL_TEST,
        "max_d": FILTER_MAX_D,
        "max_D": FILTER_MAX_BIG_D,
        "seasonal": FILTER_SEASONAL,
        "seasonal_period": d5.SEASONAL_M,
        "estimation_start": str(x_train.index.min().date()),
        "estimation_end": str(x_train.index.max().date()),
        "n_estimation": int(len(x_train)),
        "indicator_application_start": str(x_history.index.min().date()),
        "indicator_application_end": str(x_history.index.max().date()),
        "burn_in_indicator": burn_ind,
        "burn_in_response": burn_y,
        "ccf_response_start": str(beta.index.min().date()),
        "ccf_response_end": str(beta.index.max().date()),
        "n_ccf_response": int(len(beta)),
        "parameter_names": list(source_res.model.param_names),
    }
    return alpha, beta, info


def ccf_at(u: pd.Series, v: pd.Series, lag: int) -> tuple[float, int]:
    """corr(u_{t-lag}, v_t) over the dates where both are available."""
    shifted = u.shift(lag)
    joint = pd.concat([shifted.rename("u"), v.rename("v")], axis=1,
                      join="inner").dropna()
    if len(joint) < 30:
        sys.exit("only {} usable pairs at lag {}".format(len(joint), lag))
    return float(np.corrcoef(joint["u"], joint["v"])[0, 1]), int(len(joint))


def fit_lag(y: pd.Series, X_ann: pd.DataFrame, ind: pd.Series, lag: int,
            sel: dict) -> dict:
    # Stage 2 only: express the search regressor in thousands of queries.
    # Stage-1 prewhitening / CCF receives the original unscaled series.
    col_raw = ind.shift(lag).reindex(y.index)
    if col_raw.isna().any():
        sys.exit("lag {} leaves {} missing indicator values on the training "
                 "index".format(lag, int(col_raw.isna().sum())))
    col = col_raw / INDICATOR_SCALE_DIVISOR
    X = pd.concat([X_ann, col.rename("search_lag{}".format(lag))], axis=1)
    model = pm.ARIMA(order=tuple(sel["selected"]["order"]),
                     seasonal_order=tuple(sel["selected"]["seasonal_order"]),
                     with_intercept=sel["selected"]["with_intercept"],
                     maxiter=DOWNSTREAM_MAXITER, method=DOWNSTREAM_METHOD,
                     suppress_warnings=True)
    model.fit(y.to_numpy(dtype=float), X=X.to_numpy(dtype=float))
    res = model.arima_res_
    mle = getattr(res, "mle_retvals", {}) or {}
    return {"lag": lag, "aicc": float(res.info_criteria("aicc")),
            "converged": (None if "converged" not in mle
                          else bool(mle["converged"])),
            "iterations": int(mle.get("iterations", -1)),
            "nobs_effective": int(res.nobs_effective),
            "n_exog": int(getattr(res.model, "k_exog", 0))}


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False, description="D9: search lag")
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--history", required=True,
                    help="path to indicator_history.csv")
    ap.add_argument("--selection", required=True,
                    help="path to the operative d5_selection.json")
    ap.add_argument("--publication-delay-days", type=int, required=True,
                    dest="pub_delay",
                    help="observed Wordstat publication delay in days, "
                         "measured from the END of a week to the date its "
                         "value became available (C3-4 requires this to be "
                         "recorded). D9 verifies that every candidate lag "
                         "is usable under that delay; it is not assumed.")
    ap.add_argument("--outdir", default="results/d9")
    ap.add_argument("--experiment",
                    default="medical-assistance-demand-forecasting")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()
    args.d = None
    args.D_seasonal = None

    project_root = Path(__file__).resolve().parents[1]
    tracking_uri = (project_root / "mlruns").as_uri()
    addendum_path = project_root / "docs" / "addenda.md"
    if not addendum_path.exists():
        sys.exit("protocol addendum not found at {}; D9 requires the "
                 "operative addendum to be present before execution."
                 .format(addendum_path))
    addendum_digest = sha256_of(addendum_path)

    env_diffs = d5.check_environment(args.allow_env_mismatch)

    sel = json.loads(Path(args.selection).read_text())
    if sel.get("row") != "D5":
        sys.exit("--selection must point at a D5 selection artifact.")
    args.scale = sel["scale"]
    if sel.get("protocol_tag") != SOURCE_D5_PROTOCOL_TAG:
        sys.exit("expected the operative D5 source artifact to be {}, "
                 "found {}.".format(SOURCE_D5_PROTOCOL_TAG,
                                    sel.get("protocol_tag")))
    if sel["hk_settings"]["method"] != DOWNSTREAM_METHOD:
        sys.exit("the operative D5 selection used method='{}' but D9 fits "
                 "with '{}'. The addendum raises the ceiling only."
                 .format(sel["hk_settings"]["method"], DOWNSTREAM_METHOD))

    # C3-4 vintage rule, verified rather than assumed.
    if args.pub_delay < 0:
        sys.exit("--publication-delay-days cannot be negative.")
    min_usable_lag = MAX_DAYS_WEEK_START_TO_END + args.pub_delay
    unusable = [L for L in LAG_VALUES if L < min_usable_lag]
    if unusable:
        sys.exit(
            "C3-4 violated: with an observed publication delay of {} days, "
            "the conservative availability requirement is {} days "
            "from week start to forecast date, so candidate lag(s) {} could "
            "use a weekly value before publication. The frozen candidate "
            "set cannot be narrowed here; resolve before running D9."
            .format(args.pub_delay, min_usable_lag, unusable))
    print("Stage 1 uses the original search-index units; Stage 2 divides "
          "the exogenous regressor by {:.0f} for optimizer conditioning "
          "(exact coefficient reparameterisation).".format(
              INDICATOR_SCALE_DIVISOR))
    print("C3-4: publication delay {} days; using a maximum six-day span "
          "from week start to week end gives a minimum safe lag of {} days; "
          "the smallest frozen candidate lag is {}, so every candidate is "
          "usable.".format(args.pub_delay, min_usable_lag, LAG_MIN))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path, hist_path = Path(args.data), Path(args.history)
    digest = sha256_of(data_path)
    if digest != sel["data"]["sha256"]:
        sys.exit("input SHA-256 {} does not match the file D5 ran on ({})."
                 .format(digest[:16], sel["data"]["sha256"][:16]))
    hist_digest = sha256_of(hist_path)
    if hist_digest != FROZEN_HISTORY_SHA256:
        sys.exit("indicator_history.csv SHA-256 {} does not match the "
                 "frozen D8/D9 history artifact ({})."
                 .format(hist_digest[:16], FROZEN_HISTORY_SHA256[:16]))

    s, y, y_raw = d5.load_training_series(args)
    need_from = y.index.min() - pd.Timedelta(LAG_MAX, "D")
    ind = load_history(hist_path, need_from, y.index.max())
    print("loaded {} training days; scale in force: {}".format(
        len(y), sel["scale"]))
    print("search history {} -> {} ({} days), sha256 {}...".format(
        ind.index.min().date(), ind.index.max().date(), len(ind),
        hist_digest[:16]))

    d5.WITH_INTERCEPT = bool(sel["selected"]["with_intercept"])
    ann = sel["annual_regressor"]
    cand = ("monthly" if ann["kind"] == "monthly_dummies"
            else "fourier_K{}".format(ann["K"]))
    X_ann = d5.build_candidate(cand, y.index)
    if list(X_ann.columns) != list(ann["columns"]):
        sys.exit("annual regressor rebuilt as {} but D5 recorded {}"
                 .format(list(X_ann.columns), ann["columns"]))
    print("operative M1: {} ARIMA{}{}[{}], constant={}".format(
        cand, tuple(sel["selected"]["order"]),
        tuple(sel["selected"]["seasonal_order"][:3]),
        sel["selected"]["seasonal_order"][3],
        sel["selected"]["with_intercept"]))

    # ---- stage 1: prewhitened CCF -------------------------------------
    bounds = {k: sel["hk_settings"][k] for k in
              ("start_p", "start_q", "start_P", "start_Q",
               "max_p", "max_q", "max_P", "max_Q")}

    # Estimate the auxiliary whitening model on the frozen 884-day
    # training window only. The lead-in is used only when the resulting
    # fixed filter is applied.
    ind_through_train = ind.loc[:y.index.max()]
    ind_train = ind_through_train.reindex(y.index)
    if ind_train.isna().any() or len(ind_train) != len(y):
        sys.exit("indicator history does not provide a complete value for every "
                 "training date used to estimate the whitening filter.")

    t0 = time.time()
    alpha, beta, filt_info = prewhiten(
        ind_train, ind_through_train, y, bounds)
    print("prewhitening filter estimated on {} search training days: "
          "ARIMA{}{}[{}], intercept={}, converged=True, iters {} "
          "({:.0f} s)".format(
              filt_info["n_estimation"],
              tuple(filt_info["order"]),
              tuple(filt_info["seasonal_order"][:3]),
              filt_info["seasonal_order"][3],
              filt_info["with_intercept"],
              filt_info["iterations"],
              time.time() - t0))
    print("state-space burn: indicator {}, response {}; common CCF response "
          "sample {} -> {} ({} days)".format(
              filt_info["burn_in_indicator"],
              filt_info["burn_in_response"],
              filt_info["ccf_response_start"],
              filt_info["ccf_response_end"],
              filt_info["n_ccf_response"]))

    # Raw CCF is diagnostic only, but evaluate it on the same response
    # dates as the prewhitened CCF so the comparison is sample-matched.
    y_raw_ccf = y.reindex(beta.index)

    rows = []
    for lag in LAG_VALUES:
        pw, n_pw = ccf_at(alpha, beta, lag)
        raw, n_raw = ccf_at(ind_through_train, y_raw_ccf, lag)
        if n_pw != n_raw:
            sys.exit("prewhitened and raw diagnostic CCF use different "
                     "pair counts at lag {} ({} vs {}).".format(
                         lag, n_pw, n_raw))
        rows.append({"lag": lag, "ccf_prewhitened": pw,
                     "abs_ccf_prewhitened": abs(pw), "n_pairs": n_pw,
                     "ccf_raw_diagnostic": raw})
    screen = pd.DataFrame(rows)
    if screen["n_pairs"].nunique() != 1:
        sys.exit("candidate CCFs do not share one common response sample; "
                 "pair counts are {}.".format(
                     sorted(screen["n_pairs"].unique().tolist())))
    # tie to shorter lag: descending |CCF|, ascending lag
    screen = screen.sort_values(["abs_ccf_prewhitened", "lag"],
                               ascending=[False, True], kind="mergesort")
    screen["screen_rank"] = range(1, len(screen) + 1)
    screen = screen.sort_values("lag").reset_index(drop=True)
    screen.to_csv(outdir / "d9_ccf_screen.csv", index=False)

    chosen = (screen.sort_values("screen_rank").head(N_SCREEN)["lag"]
              .astype(int).tolist())
    print("stage 1: five largest |prewhitened CCF| -> lags {}".format(
        chosen))
    raw_top = (screen.reindex(screen["ccf_raw_diagnostic"].abs()
                              .sort_values(ascending=False).index)
               .head(N_SCREEN)["lag"].astype(int).tolist())
    print("           (raw CCF would have given {} -- diagnostic only)"
          .format(sorted(raw_top)))

    # ---- stage 2: AICc ------------------------------------------------
    grid = []
    for lag in chosen:
        t0 = time.time()
        r = fit_lag(y, X_ann, ind_through_train, lag, sel)
        r["seconds"] = round(time.time() - t0, 1)
        grid.append(r)
        print("[lag {}] AICc {:.4f} (converged {}, iters {}, {:.0f} s)"
              .format(lag, r["aicc"], r["converged"], r["iterations"],
                      r["seconds"]), flush=True)
    grid = pd.DataFrame(grid).sort_values(["aicc", "lag"],
                                          kind="mergesort").reset_index(
                                              drop=True)
    grid.to_csv(outdir / "d9_lag_grid.csv", index=False)

    if grid["nobs_effective"].nunique() != 1 or grid["n_exog"].nunique() != 1:
        sys.exit("the five fits do not share an estimation sample or "
                 "regressor count; their AICc values are not comparable.")
    stalled = grid[grid["converged"] != True]  # noqa: E712
    if len(stalled):
        print("")
        print("EXECUTION PAUSED -- {} of {} fits did not converge at "
              "maxiter = {}. Per the addendum of 24 August 2026 the fit is "
              "not excluded or replaced: document and resolve before "
              "finalizing D9. No lag selected.".format(
                  len(stalled), N_SCREEN, DOWNSTREAM_MAXITER))
        print(stalled[["lag", "aicc", "converged",
                       "iterations"]].to_string(index=False))
        sys.exit(2)

    best = grid.iloc[0]
    lag_sel = int(best["lag"])
    margin = float(grid.iloc[1]["aicc"] - best["aicc"])
    scr = screen.set_index("lag")

    out = {
        "protocol_tag": PROTOCOL_TAG,
        "protocol_freeze_tag": d5.PROTOCOL_FREEZE_TAG,
        "source_d5_protocol_tag": sel.get("protocol_tag"),
        "row": ROW,
        "scale": sel["scale"],
        "operative_m1": {
            "source_selection": str(Path(args.selection).resolve()),
            "candidate": cand,
            "order": list(sel["selected"]["order"]),
            "seasonal_order": list(sel["selected"]["seasonal_order"]),
            "with_intercept": sel["selected"]["with_intercept"],
        },
        "candidate_lags": {"min": LAG_MIN, "max": LAG_MAX,
                           "step": LAG_STEP, "values": LAG_VALUES,
                           "n": len(LAG_VALUES)},
        "stage1_ccf": {
            "method": "prewhitened cross-correlation (Box et al. 2016)",
            "filter": filt_info,
            "filter_choice_note": "not fixed by the frozen row; auxiliary "
                                  "search-index ARIMA estimated on the frozen training "
                                  "window only using D5 stepwise bounds, "
                                  "s = 7, d <= 1, D <= 1, KPSS/OCSB "
                                  "differencing tests and automatic "
                                  "intercept handling; the fixed fitted "
                                  "state-space filter is then applied to "
                                  "extended indicator history and response",
            "selected_lags": chosen,
            "raw_ccf_top5_diagnostic": sorted(raw_top),
            "raw_ccf_agrees": sorted(raw_top) == sorted(chosen),
        },
        "stage2_aicc": {
            "lags": [int(v) for v in grid["lag"]],
            "aicc": [float(v) for v in grid["aicc"]],
        },
        "selected_lag": {
            "lag": lag_sel, "aicc": float(best["aicc"]),
            "margin_over_runner_up": margin,
            "ccf_prewhitened": float(scr.loc[lag_sel, "ccf_prewhitened"]),
            "screen_rank": int(scr.loc[lag_sel, "screen_rank"]),
            "converged": bool(best["converged"]),
        },
        "tie_rule": "tie to shorter lag (frozen D8, inherited by D9), applied at both "
                    "stages",
        "optimizer": {"method": DOWNSTREAM_METHOD,
                      "maxiter": DOWNSTREAM_MAXITER,
                      "source": "24 Aug 2026 addendum"},
        "indicator_scaling": {
            "applies_to": "Stage-2 AICc fits only",
            "stage1_ccf_units": "original search_index units",
            "divisor": INDICATOR_SCALE_DIVISOR,
            "neutrality": "exact exogenous-regressor reparameterisation: "
                          "at the same optimum fitted values, likelihood, "
                          "parameter count, AICc and forecasts are unchanged; "
                          "only the coefficient scale differs",
            "reason": "optimizer conditioning in Stage-2 M4 fits",
            "source": "26 Aug 2026 addendum; operative protocol-v1.6",
            "coefficient_interpretation": "per {} queries".format(
                int(INDICATOR_SCALE_DIVISOR)),
        },
        "c3_4_vintage": {
            "publication_delay_days": int(args.pub_delay),
            "max_days_week_start_to_end": MAX_DAYS_WEEK_START_TO_END,
            "min_usable_lag": int(MAX_DAYS_WEEK_START_TO_END + args.pub_delay),
            "smallest_candidate_lag": LAG_MIN,
            "all_candidates_usable": True,
            "rule": "a week's value enters only after publication; "
                    "a seven-day week ends six days after its start, so "
                    "the conservative all-weekday requirement is "
                    "lag >= 6 + observed publication delay",
        },
        "e6_check": "every candidate lag >= 28 days = the longest "
                    "horizon, so all regressor values predate the origin",
        "next_step": "M4 = M1 + search index at lag {}; D10 transfers this lag to "
                     "M5 without re-tuning.".format(lag_sel),
        "smoke_mode": bool(sel.get("smoke_mode", False)),
        "environment_mismatch": env_diffs,
        "environment": d5.observed_environment(),
        "d1_environment_frozen": d5.D1_ENVIRONMENT,
        "data": {"path": str(data_path), "sha256": digest},
        "history": {"path": str(hist_path), "sha256": hist_digest},
        "protocol_addendum": {
            "path": str(addendum_path),
            "sha256": addendum_digest,
        },
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (outdir / "d9_lag_selection.json").write_text(json.dumps(out, indent=2))

    L = []
    L.append("# D9 run report -- search lag")
    L.append("")
    stamp = ""
    if env_diffs:
        stamp += ("  **NON-PROTOCOL: D1 environment not in force: "
                  + "; ".join(env_diffs) + ".**")
    if out["smoke_mode"]:
        stamp += "  **Inherited SMOKE MODE from the D5 selection.**"
    L.append("Protocol state {} (freeze tag {}), row D9. Run (UTC): {}.{}"
             .format(PROTOCOL_TAG, d5.PROTOCOL_FREEZE_TAG, out["run_utc"],
                     stamp))
    L.append("")
    L.append("Operative addendum: `{}` (SHA-256 `{}`).".format(
        addendum_path.name, addendum_digest))
    L.append("")
    L.append("Operative M1 read from `{}`: {}, ARIMA{}{}[{}]{}, on the {} "
             "scale, with orders and annual form held fixed throughout "
             "(20 August 2026 addendum). Lagged search interest comes from `{}` "
             "(SHA-256 `{}`), which extends the series back before the "
             "spine start; aligned_train.csv alone cannot supply a "
             "{}-day lag for the first training day.".format(
                 Path(args.selection).name, cand,
                 tuple(sel["selected"]["order"]),
                 tuple(sel["selected"]["seasonal_order"][:3]),
                 sel["selected"]["seasonal_order"][3],
                 "" if sel["selected"]["with_intercept"]
                 else ", no constant", sel["scale"], hist_path.name,
                 hist_digest, LAG_MAX))
    L.append("")
    L.append("## Stage 1 -- CCF identification")
    L.append("")
    L.append("Following Box-Jenkins transfer-function identification, "
             "an auxiliary ARIMA model is fitted to the search input and that "
             "same fixed state-space filter is applied to the search index and to the "
             "response before computing the CCF. Raw CCF is diagnostic "
             "only because serial dependence in the persistent indicator input "
             "can create misleading cross-correlation peaks. The auxiliary "
             "filter is estimated on the frozen {}-day training window "
             "only ({} through {}), not on the pre-spine lead-in. It is "
             "ARIMA{}{}[{}], intercept={}, AICc {:.2f}, selected with "
             "KPSS/OCSB differencing tests, d <= 1 and D <= 1; it "
             "converged in {} iterations under method='{}', maxiter={}. "
             "State-space initialization is removed using the filtered "
             "results' loglikelihood_burn (indicator {}, response {}). All lag "
             "CCFs use the same response sample, {} through {} ({} "
             "observations).".format(
                 filt_info["n_estimation"],
                 filt_info["estimation_start"],
                 filt_info["estimation_end"],
                 tuple(filt_info["order"]),
                 tuple(filt_info["seasonal_order"][:3]),
                 filt_info["seasonal_order"][3],
                 filt_info["with_intercept"],
                 filt_info["aicc"],
                 filt_info["iterations"],
                 filt_info["method"],
                 filt_info["maxiter"],
                 filt_info["burn_in_indicator"],
                 filt_info["burn_in_response"],
                 filt_info["ccf_response_start"],
                 filt_info["ccf_response_end"],
                 filt_info["n_ccf_response"]))
    L.append("")
    L.append("Candidate lags {}-{} ({} in total). The five largest "
             "|prewhitened CCF|, ties broken to the shorter lag, are "
             "**{}**.".format(LAG_MIN, LAG_MAX,
                              len(LAG_VALUES), chosen))
    L.append("")
    L.append("| lag | prewhitened CCF | rank | raw CCF (diagnostic) |")
    L.append("|---|---|---|---|")
    for lag in chosen:
        r = scr.loc[lag]
        L.append("| {} | {:+.4f} | {} | {:+.4f} |".format(
            lag, r["ccf_prewhitened"], int(r["screen_rank"]),
            r["ccf_raw_diagnostic"]))
    L.append("")
    L.append("Diagnostic: the raw CCF would have screened lags {} "
             "instead, {} the prewhitened set. The raw values take no "
             "part in the selection and are recorded only so the "
             "difference between the two readings is visible.".format(
                 sorted(raw_top),
                 "which agrees with" if sorted(raw_top) == sorted(chosen)
                 else "which differs from"))
    L.append("")
    L.append("Stage 1 uses the original search-index units throughout "
             "prewhitening and CCF screening. Only after the five candidate "
             "lags have been frozen does Stage 2 divide the exogenous search "
             "regressor by {:.0f} for numerical conditioning. For an "
             "exogenous regressor this is an exact coefficient "
             "reparameterisation: at the same optimum fitted values, "
             "likelihood, parameter count, AICc and forecasts are unchanged; "
             "only the coefficient scale differs. It therefore alters no "
             "candidate lag, CCF value, threshold or D9 selection rule. "
             "Downstream coefficients are interpreted per {:.0f} queries."
             .format(INDICATOR_SCALE_DIVISOR,
                     INDICATOR_SCALE_DIVISOR))
    L.append("")
    L.append("## Stage 2 -- AICc")
    L.append("")
    L.append("| lag | AICc | converged | iterations |")
    L.append("|---|---|---|---|")
    for _, r in grid.iterrows():
        L.append("| {} | {:.4f} | {} | {} |".format(
            int(r["lag"]), r["aicc"], "yes" if r["converged"] else "no",
            int(r["iterations"])))
    L.append("")
    L.append("All five fits share the estimation sample ({} effective "
             "observations) and regressor count ({}); both were checked. "
             "Per the addendum of 24 August 2026 they were fitted under a "
             "common ceiling (method='{}', maxiter={}) and all converged."
             .format(int(grid["nobs_effective"].iloc[0]),
                     int(grid["n_exog"].iloc[0]), DOWNSTREAM_METHOD,
                     DOWNSTREAM_MAXITER))
    L.append("")
    L.append("## Selection")
    L.append("")
    L.append("**search lag = {} days**, AICc {:.4f}, ahead of the runner-up "
             "by {:.4f}. Prewhitened CCF at that lag {:+.4f} (screen rank "
             "{}). Ties are broken to the shorter lag at both stages, as "
             "the frozen row declares.".format(
                 lag_sel, best["aicc"], margin,
                 scr.loc[lag_sel, "ccf_prewhitened"],
                 int(scr.loc[lag_sel, "screen_rank"])))
    L.append("")
    L.append("E6 holds by construction: every candidate lag is at least "
             "28 days, which is the longest forecast horizon, so all "
             "regressor values predate the origin.")
    L.append("")
    L.append("## Next step")
    L.append("")
    L.append(out["next_step"])
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("d9_ccf_screen.csv (all {} candidate lags, prewhitened and "
             "raw), d9_lag_grid.csv (the five AICc candidates), "
             "d9_lag_selection.json, this report. No proprietary artifact: "
             "the tables carry correlations, AICc values and lags only."
             .format(len(LAG_VALUES)))
    (outdir / "d9_report.md").write_text("\n".join(L) + "\n")

    # ---- mandatory MLflow record ---------------------------------------
    # D9 is not considered operationally complete unless its live run is
    # recorded in the existing project tracking store.
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.experiment)

        with mlflow.start_run(run_name="D9_{}".format(sel["scale"])):
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
                "smoke_mode": str(out["smoke_mode"]),
            })
            mlflow.log_params({
                "scale": sel["scale"],
                "lag_min": LAG_MIN,
                "lag_max": LAG_MAX,
                "n_screen": N_SCREEN,
                "ccf_method": "prewhitened",
                "indicator_scale_stage": "stage2_aicc_only",
                "indicator_scale_divisor": INDICATOR_SCALE_DIVISOR,
                "indicator_scale_source": "26 Aug 2026 addendum",
                "publication_delay_days": args.pub_delay,
                "filter_order": str(filt_info["order"]),
                "filter_seasonal_order": str(filt_info["seasonal_order"]),
                "filter_with_intercept": str(
                    filt_info["with_intercept"]),
                # Windows FileStore names one file per param key and
                # NTFS is case-insensitive, so d/D-only distinctions
                # collide. Use unambiguous names.
                "filter_test_nonseasonal": filt_info["d_test"],
                "filter_test_seasonal": filt_info["D_test"],
                "filter_estimation_start":
                    filt_info["estimation_start"],
                "filter_estimation_end":
                    filt_info["estimation_end"],
                "filter_n_estimation":
                    filt_info["n_estimation"],
                "filter_burn_indicator":
                    filt_info["burn_in_indicator"],
                "filter_burn_response":
                    filt_info["burn_in_response"],
                "ccf_response_start":
                    filt_info["ccf_response_start"],
                "ccf_n_response":
                    filt_info["n_ccf_response"],
                "screened_lags": str(chosen),
                "maxiter": DOWNSTREAM_MAXITER,
                "method": DOWNSTREAM_METHOD,
                "m1_candidate": cand,
                "m1_order": str(sel["selected"]["order"]),
                "m1_seasonal_order":
                    str(sel["selected"]["seasonal_order"]),
            })
            mlflow.log_metrics({
                "selected_lag": float(lag_sel),
                "selected_aicc": float(best["aicc"]),
                "margin_over_runner_up": margin,
                "ccf_prewhitened_at_selected": float(
                    scr.loc[lag_sel, "ccf_prewhitened"]),
                "raw_ccf_agrees": float(
                    sorted(raw_top) == sorted(chosen)),
                "filter_aicc": float(filt_info["aicc"]),
                "filter_iterations":
                    float(filt_info["iterations"]),
            })

            # D9 result artifacts.
            for f in (
                "d9_ccf_screen.csv",
                "d9_lag_grid.csv",
                "d9_lag_selection.json",
                "d9_report.md",
            ):
                mlflow.log_artifact(str(outdir / f),
                                    artifact_path="d9")

            # Freeze the exact protocol clarification used by this run.
            mlflow.log_artifact(
                str(addendum_path),
                artifact_path="protocol",
            )

            # Archive the exact source file that produced this D9 run.
            mlflow.log_artifact(
                str(Path(__file__).resolve()),
                artifact_path="source",
            )

        print("logged to MLflow experiment '{}' at {}".format(
            args.experiment, tracking_uri))

    except Exception as exc:  # noqa: BLE001
        sys.exit(
            "MLflow logging FAILED: {}\n"
            "D9 result artifacts were written to disk, but this row is NOT "
            "considered complete until the live run is present in MLflow."
            .format(exc)
        )

    print("")
    print("D9 SELECTION: search lag = {} days, AICc {:.4f} (runner-up +{:.4f})"
          .format(lag_sel, best["aicc"], margin))
    print("next: {}".format(out["next_step"]))
    print("outputs written to {}".format(outdir.resolve()))


if __name__ == "__main__":
    main()
