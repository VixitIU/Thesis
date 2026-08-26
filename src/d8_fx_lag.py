"""D8 -- FX lag selection.

Frozen D8:
    two-stage rule, declared: candidate lags 28-90 days; five largest
    |CCF| on training sample proceed to AICc; tie to shorter lag. CCF
    identification per Box et al. (2016).

Stage 1 -- CCF identification
-----------------------------
Box, Jenkins, Reinsel & Ljung (2016) identify a transfer function by
PREWHITENING: an ARIMA filter is fitted to the input series, that same
fixed filter is applied to both input and output, and the
cross-correlation is computed between the filtered series. Raw CCF can
be distorted by serial dependence in a persistent input; FX is strongly
persistent here, so raw CCF is diagnostic only.

The five candidates are therefore selected from the prewhitened |CCF|
alone. Raw CCF is also written to the artifact for auditability but has
no role in selection.

Not fixed by the frozen row -- the prewhitening filter
------------------------------------------------------
D8 cites the Box-Jenkins identification method but does not pin an exact
input filter. Recorded implementation choice:

* estimate an auxiliary seasonal ARIMA filter on FX observations aligned
  to the frozen 884-day training window only;
* use the D5 Hyndman-Khandakar stepwise order bounds, weekly period s=7,
  AICc, d <= 1 and D <= 1;
* make pmdarima's differencing tests explicit: KPSS for d and OCSB for D;
* use with_intercept="auto" for this auxiliary FX model rather than
  inheriting the case-model D4 constant decision;
* require the selected whitening model to converge at method="lbfgs",
  maxiter=500, otherwise pause D8 before screening;
* apply the fitted state-space specification and parameter vector,
  unchanged, to the extended pre-spine FX history and to the response;
* drop state-space initialization observations using each filtered
  result's model-reported loglikelihood_burn, not a hand-built formula;
* evaluate every candidate CCF on one common post-burn response sample.

The pre-spine FX history is therefore used to APPLY the fixed whitening
filter and to construct lagged regressors. It does not enlarge the
sample used to ESTIMATE the whitening model.

Stage 2 -- AICc
---------------
Per the addendum of 20 August 2026, the operative M1 orders and annual
form are held fixed and only the candidate lag varies. M3 = M1 + FX at
lag L; the holiday regressors belong to M2 and are not present here.
Per the addendum of 24 August 2026, every candidate is fitted under a
common ceiling (maxiter = 500), convergence is recorded, and execution
PAUSES rather than excluding a fit that fails to converge.

Ties: the frozen row declares "tie to shorter lag", which is applied at
both stages -- to |CCF| ties when choosing the five, and to AICc ties
when choosing the winner.

Lagged values come from indicator_history.csv, which extends the FX
series back before the spine start; aligned_train.csv cannot supply a
90-day lag for the first training day. E6 is satisfied by construction:
every candidate lag is at least 28 days, the longest horizon, so all
values predate the forecast origin.

Outputs (--outdir, default results/d8)
--------------------------------------
    d8_ccf_screen.csv     all 63 candidate lags, prewhitened and raw
    d8_lag_grid.csv       the five AICc candidates
    d8_lag_selection.json machine-readable lag -> M3 and D10
    d8_report.md          prose record for the research log

No proprietary artifact is written: the tables carry correlations,
AICc values and lags only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm

try:
    import d5_baseline_order_selection as d5
except ImportError:
    sys.exit("d5_baseline_order_selection.py must be importable:" 
             "D8 reuses its loader, its "
             "annual-form builders and its D1 environment guard.")

# ----------------------------- frozen constants ------------------------------
ROW = "D8"
LAG_MIN, LAG_MAX = 28, 90          # D8 candidate lags
N_SCREEN = 5                       # five largest |CCF| proceed to AICc
FX_COL = "fx_rub_per_thb"

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

PROTOCOL_TAG = "protocol-v1.5"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_history(path: Path, need_from: pd.Timestamp,
                 need_to: pd.Timestamp) -> pd.Series:
    h = pd.read_csv(path, parse_dates=["obs_date"]).set_index("obs_date")
    if FX_COL not in h.columns:
        sys.exit("{} has no column '{}'; found {}".format(
            path, FX_COL, list(h.columns)))
    s = h[FX_COL].astype(float).sort_index()
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
        sys.exit("NaNs in the FX history")
    return s


def prewhiten(x_train: pd.Series, x_history: pd.Series,
              y: pd.Series, bounds: dict):
    """Box-Jenkins prewhitening with training-only filter estimation.

    The auxiliary ARIMA specification and parameters are estimated on FX
    values aligned exactly to the frozen response training window. The
    fitted state-space model is then cloned and filtered, without
    re-estimation, over (a) the extended FX history and (b) the response.
    """
    if not x_train.index.equals(y.index):
        sys.exit("FX filter-estimation index must equal the response "
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
        print("EXECUTION PAUSED -- the selected FX prewhitening model did "
              "not converge at maxiter = {}. The filter is upstream of the "
              "five CCF candidates, so D8 is not screened or finalized."
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
        sys.exit("cloned FX-history filter parameterization differs from "
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

    burn_x = int(getattr(x_res, "loglikelihood_burn", 0) or 0)
    burn_y = int(getattr(y_res, "loglikelihood_burn", 0) or 0)
    alpha = alpha.iloc[burn_x:]
    beta = beta.iloc[burn_y:]

    # All 63 lags must be screened on the same response dates. The common
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
        "fx_application_start": str(x_history.index.min().date()),
        "fx_application_end": str(x_history.index.max().date()),
        "burn_in_fx": burn_x,
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


def fit_lag(y: pd.Series, X_ann: pd.DataFrame, fx: pd.Series, lag: int,
            sel: dict) -> dict:
    col = fx.shift(lag).reindex(y.index)
    if col.isna().any():
        sys.exit("lag {} leaves {} missing FX values on the training "
                 "index".format(lag, int(col.isna().sum())))
    X = pd.concat([X_ann, col.rename("fx_lag{}".format(lag))], axis=1)
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
        allow_abbrev=False, description="D8: FX lag")
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--history", required=True,
                    help="path to indicator_history.csv")
    ap.add_argument("--selection", required=True,
                    help="path to the operative d5_selection.json")
    ap.add_argument("--outdir", default="results/d8")
    ap.add_argument("--experiment", default="thesis-baselines")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()
    args.d = None
    args.D_seasonal = None

    env_diffs = d5.check_environment(args.allow_env_mismatch)

    sel = json.loads(Path(args.selection).read_text())
    if sel.get("row") != "D5":
        sys.exit("--selection must point at a D5 selection artifact.")
    args.scale = sel["scale"]
    if sel["hk_settings"]["method"] != DOWNSTREAM_METHOD:
        sys.exit("the operative D5 selection used method='{}' but D8 fits "
                 "with '{}'. The addendum raises the ceiling only."
                 .format(sel["hk_settings"]["method"], DOWNSTREAM_METHOD))

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
    fx = load_history(hist_path, need_from, y.index.max())
    print("loaded {} training days; scale in force: {}".format(
        len(y), sel["scale"]))
    print("FX history {} -> {} ({} days), sha256 {}...".format(
        fx.index.min().date(), fx.index.max().date(), len(fx),
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
    fx_through_train = fx.loc[:y.index.max()]
    fx_train = fx_through_train.reindex(y.index)
    if fx_train.isna().any() or len(fx_train) != len(y):
        sys.exit("FX history does not provide a complete value for every "
                 "training date used to estimate the whitening filter.")

    t0 = time.time()
    alpha, beta, filt_info = prewhiten(
        fx_train, fx_through_train, y, bounds)
    print("prewhitening filter estimated on {} FX training days: "
          "ARIMA{}{}[{}], intercept={}, converged=True, iters {} "
          "({:.0f} s)".format(
              filt_info["n_estimation"],
              tuple(filt_info["order"]),
              tuple(filt_info["seasonal_order"][:3]),
              filt_info["seasonal_order"][3],
              filt_info["with_intercept"],
              filt_info["iterations"],
              time.time() - t0))
    print("state-space burn: FX {}, response {}; common CCF response "
          "sample {} -> {} ({} days)".format(
              filt_info["burn_in_fx"],
              filt_info["burn_in_response"],
              filt_info["ccf_response_start"],
              filt_info["ccf_response_end"],
              filt_info["n_ccf_response"]))

    # Raw CCF is diagnostic only, but evaluate it on the same response
    # dates as the prewhitened CCF so the comparison is sample-matched.
    y_raw_ccf = y.reindex(beta.index)

    rows = []
    for lag in range(LAG_MIN, LAG_MAX + 1):
        pw, n_pw = ccf_at(alpha, beta, lag)
        raw, n_raw = ccf_at(fx_through_train, y_raw_ccf, lag)
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
    screen.to_csv(outdir / "d8_ccf_screen.csv", index=False)

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
        r = fit_lag(y, X_ann, fx_through_train, lag, sel)
        r["seconds"] = round(time.time() - t0, 1)
        grid.append(r)
        print("[lag {}] AICc {:.4f} (converged {}, iters {}, {:.0f} s)"
              .format(lag, r["aicc"], r["converged"], r["iterations"],
                      r["seconds"]), flush=True)
    grid = pd.DataFrame(grid).sort_values(["aicc", "lag"],
                                          kind="mergesort").reset_index(
                                              drop=True)
    grid.to_csv(outdir / "d8_lag_grid.csv", index=False)

    if grid["nobs_effective"].nunique() != 1 or grid["n_exog"].nunique() != 1:
        sys.exit("the five fits do not share an estimation sample or "
                 "regressor count; their AICc values are not comparable.")
    stalled = grid[grid["converged"] != True]  # noqa: E712
    if len(stalled):
        print("")
        print("EXECUTION PAUSED -- {} of {} fits did not converge at "
              "maxiter = {}. Per the addendum of 24 August 2026 the fit is "
              "not excluded or replaced: document and resolve before "
              "finalizing D8. No lag selected.".format(
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
                           "n": LAG_MAX - LAG_MIN + 1},
        "stage1_ccf": {
            "method": "prewhitened cross-correlation (Box et al. 2016)",
            "filter": filt_info,
            "filter_choice_note": "not fixed by the frozen row; auxiliary "
                                  "FX ARIMA estimated on the frozen training "
                                  "window only using D5 stepwise bounds, "
                                  "s = 7, d <= 1, D <= 1, KPSS/OCSB "
                                  "differencing tests and automatic "
                                  "intercept handling; the fixed fitted "
                                  "state-space filter is then applied to "
                                  "extended FX history and response",
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
        "tie_rule": "tie to shorter lag (frozen D8), applied at both "
                    "stages",
        "optimizer": {"method": DOWNSTREAM_METHOD,
                      "maxiter": DOWNSTREAM_MAXITER,
                      "source": "24 Aug 2026 addendum"},
        "e6_check": "every candidate lag >= 28 days = the longest "
                    "horizon, so all regressor values predate the origin",
        "next_step": "M3 = M1 + FX at lag {}; D10 transfers this lag to "
                     "M5 without re-tuning.".format(lag_sel),
        "smoke_mode": bool(sel.get("smoke_mode", False)),
        "environment_mismatch": env_diffs,
        "environment": d5.observed_environment(),
        "d1_environment_frozen": d5.D1_ENVIRONMENT,
        "data": {"path": str(data_path), "sha256": digest},
        "history": {"path": str(hist_path), "sha256": hist_digest},
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (outdir / "d8_lag_selection.json").write_text(json.dumps(out, indent=2))

    L = []
    L.append("# D8 run report -- FX lag")
    L.append("")
    stamp = ""
    if env_diffs:
        stamp += ("  **NON-PROTOCOL: D1 environment not in force: "
                  + "; ".join(env_diffs) + ".**")
    if out["smoke_mode"]:
        stamp += "  **Inherited SMOKE MODE from the D5 selection.**"
    L.append("Protocol state {} (freeze tag {}), row D8. Run (UTC): {}.{}"
             .format(PROTOCOL_TAG, d5.PROTOCOL_FREEZE_TAG, out["run_utc"],
                     stamp))
    L.append("")
    L.append("Operative M1 read from `{}`: {}, ARIMA{}{}[{}]{}, on the {} "
             "scale, with orders and annual form held fixed throughout "
             "(20 August 2026 addendum). Lagged FX comes from `{}` "
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
             "an auxiliary ARIMA model is fitted to the FX input and that "
             "same fixed state-space filter is applied to FX and to the "
             "response before computing the CCF. Raw CCF is diagnostic "
             "only because serial dependence in the persistent FX input "
             "can create misleading cross-correlation peaks. The auxiliary "
             "filter is estimated on the frozen {}-day training window "
             "only ({} through {}), not on the pre-spine lead-in. It is "
             "ARIMA{}{}[{}], intercept={}, AICc {:.2f}, selected with "
             "KPSS/OCSB differencing tests, d <= 1 and D <= 1; it "
             "converged in {} iterations under method='{}', maxiter={}. "
             "State-space initialization is removed using the filtered "
             "results' loglikelihood_burn (FX {}, response {}). All lag "
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
                 filt_info["burn_in_fx"],
                 filt_info["burn_in_response"],
                 filt_info["ccf_response_start"],
                 filt_info["ccf_response_end"],
                 filt_info["n_ccf_response"]))
    L.append("")
    L.append("Candidate lags {}-{} ({} in total). The five largest "
             "|prewhitened CCF|, ties broken to the shorter lag, are "
             "**{}**.".format(LAG_MIN, LAG_MAX,
                              LAG_MAX - LAG_MIN + 1, chosen))
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
    L.append("**FX lag = {} days**, AICc {:.4f}, ahead of the runner-up "
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
    L.append("d8_ccf_screen.csv (all {} candidate lags, prewhitened and "
             "raw), d8_lag_grid.csv (the five AICc candidates), "
             "d8_lag_selection.json, this report. No proprietary artifact: "
             "the tables carry correlations, AICc values and lags only."
             .format(LAG_MAX - LAG_MIN + 1))
    (outdir / "d8_report.md").write_text("\n".join(L) + "\n")

    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_experiment(args.experiment)
            with mlflow.start_run(run_name="D8_{}".format(sel["scale"])):
                mlflow.set_tags({
                    "protocol.row": ROW,
                    "protocol.tag": PROTOCOL_TAG,
                    "protocol.freeze_tag": d5.PROTOCOL_FREEZE_TAG,
                    "archive_backfill": "false",
                    "execution_recomputed": "true",
                    "protocol.run_utc": out["run_utc"],
                    "data_sha256": digest,
                    "history_sha256": hist_digest,
                    "smoke_mode": str(out["smoke_mode"]),
                })
                mlflow.log_params({
                    "scale": sel["scale"], "lag_min": LAG_MIN,
                    "lag_max": LAG_MAX, "n_screen": N_SCREEN,
                    "ccf_method": "prewhitened",
                    "filter_order": str(filt_info["order"]),
                    "filter_seasonal_order": str(
                        filt_info["seasonal_order"]),
                    "filter_with_intercept": str(
                        filt_info["with_intercept"]),
                    "filter_d_test": filt_info["d_test"],
                    "filter_D_test": filt_info["D_test"],
                    "filter_estimation_start":
                        filt_info["estimation_start"],
                    "filter_estimation_end": filt_info["estimation_end"],
                    "filter_n_estimation": filt_info["n_estimation"],
                    "filter_burn_fx": filt_info["burn_in_fx"],
                    "filter_burn_response":
                        filt_info["burn_in_response"],
                    "ccf_response_start": filt_info["ccf_response_start"],
                    "ccf_n_response": filt_info["n_ccf_response"],
                    "screened_lags": str(chosen),
                    "maxiter": DOWNSTREAM_MAXITER,
                    "method": DOWNSTREAM_METHOD,
                    "m1_candidate": cand,
                    "m1_order": str(sel["selected"]["order"]),
                    "m1_seasonal_order": str(
                        sel["selected"]["seasonal_order"]),
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
                    "filter_iterations": float(filt_info["iterations"]),
                })
                for f in ("d8_ccf_screen.csv", "d8_lag_grid.csv",
                          "d8_lag_selection.json", "d8_report.md"):
                    mlflow.log_artifact(str(outdir / f))
            print("logged to MLflow experiment '{}'".format(args.experiment))
        except Exception as exc:                       
            print("WARNING: MLflow logging failed ({}). The run artifacts "
                  "are complete on disk and can be archived "
                  "retrospectively.".format(exc))

    print("")
    print("D8 SELECTION: FX lag = {} days, AICc {:.4f} (runner-up +{:.4f})"
          .format(lag_sel, best["aicc"], margin))
    print("next: {}".format(out["next_step"]))
    print("outputs written to {}".format(outdir.resolve()))


if __name__ == "__main__":
    main()