"""D5/D6 -- annual-form and ARIMA-order selection for the M1 baseline.

What this script does
---------------------
For each of seven annual-seasonality candidates

    monthly          : 12 calendar-month indicators when no constant is present; 
                       11 indicators with January as the reference month when
                       D4 includes a constant. The D5 reference-month clause applies only
                       when a constant is present.
    fourier_K1..K6   : Fourier pairs sin/cos(2*pi*k*t/365.25), k = 1..K,
                       with t = days since 2023-07-01. The origin is
                       frozen to the spine start so test-window
                       regressors extend deterministically downstream.

For the initial count-scale execution, the Hyndman-Khandakar stepwise search
(pmdarima.auto_arima, stepwise=True, information_criterion='aicc') runs with
d = 1 and D = 0 as selected at D2-D3, s = 7 retained, and no intercept under
D4, on the training sample only (<= 2025-11-30, n = 884).

If D12 is triggered, D5 is re-run on log(y + 1) using the transformed-scale
d and D selected by the required D2/D3 re-runs. D4 is then mechanically
re-applied: an intercept is included iff d = D = 0. Accordingly, the monthly
candidate uses 11 indicators with a reference month when an intercept is
present and all 12 indicators otherwise.

Gate and selection (visited-set reading)
----------------------------------------
auto_arima has no built-in residual gate, so the search runs
unconstrained and the Ljung-Box gate is applied afterwards -- to the
ENTIRE visited set, not only the seven per-candidate winners:
return_valid_fits=True returns every model the stepwise walk fitted
successfully. For each visited fit, the Ljung-Box lag is computed as
L = max(2s, k_arma + 3), capped at floor(T/5) with T = 884 training
observations, where k_arma = p+q+P+Q. With s = 7 this gives L = 14 for
typical orders. The test runs on that fit's own residuals (after
dropping the state-space burn-in, loglikelihood_burn = d + D*s
observations), with chi-square df = L - k_arma and alpha = 0.05; pass
iff p-value > alpha.

Selected specification: the lowest-AICc visited fit that passes.
D6 fallback: if nothing visited passes, the lowest-AICc visited fit
overall (the stepwise winner) is taken and the failure is recorded.

Fits that errored inside the search, or that pmdarima rejected on its
near-non-invertible-roots check, never produce residuals or an AICc and
so are absent from the valid pool -- they could not have been selected.
The raw stepwise trace (trace=2) is captured per candidate to a text
file so those attempts remain auditable.

Recorded implementation choices (repeated in the generated report)
------------------------------------------------------------------
  1. Monthly indicators follow D4: all 12 enter when no constant is present;
     with a constant, January is omitted and 11 indicators enter.
  2. AICc comparability is checked, not assumed: nobs_effective must be
     identical across every visited fit; the script aborts otherwise.
  3. Ljung-Box uses each candidate fit's OWN residuals, post burn-in.
  4. The Ljung-Box lag is computed per fitted model from the frozen
     formula L = max(2s, k_arma + 3), capped at floor(T/5), giving
     df = L - k_arma. It is not hard-coded to 14: L = 14 holds only
     while k_arma <= 11, and rises to 17 at the (5,.,5)(2,.,2) search
     corner. There is consequently no "untestable" fit and no exclusion
     rule beyond the frozen gate itself.
  5. Deterministic tie-break: AICc, then total estimated parameter
     count, then fixed candidate order, then (p, q, P, Q).
  6. error_action='ignore'; per-fit optimizer convergence is logged and
     a loud warning is raised if the SELECTED fit did not converge.
     maxiter is left at pmdarima's default (50) rather than introducing
     an untracked knob.

Environment
-----------
D1 freezes the execution environment. The script verifies it at start-up
against D1_ENVIRONMENT below and REFUSES TO RUN on a mismatch, so no
output can silently claim protocol provenance under a substituted stack.
--allow-env-mismatch overrides the refusal for pipeline testing only and
stamps every artifact NON-PROTOCOL.

Note: scikit-learn is a hard pmdarima dependency but is not pinned by
D1. Its resolved version is recorded in d5_selection.json. pmdarima
2.1.1 has been checked against scikit-learn 1.5.2 and 1.9.0; the
force_all_finite incompatibility affects pmdarima 2.0.4 only, which is
not the frozen version.

Usage
-----
python d5_baseline_order_selection.py --data path/to/daily_counts.csv \
    [--date-col date] [--y-col billed_visits] [--outdir results/d5] \
    [--mlflow] [--mlflow-experiment thesis-baselines]

--smoke runs a reduced, NON-PROTOCOL configuration (lower order caps,
lower maxiter) for pipeline testing on synthetic data only. Never use
it for a protocol execution.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
import scipy
import sklearn
import statsmodels
from statsmodels.stats.diagnostic import acorr_ljungbox

# ------------------------------ frozen constants -----------------------------
# Provenance: protocol-v1.0 is the freeze tag of the signed protocol
# document (frozen 19 August 2026). 
# Three addenda have subsequently been filed before any test-window forecast:
#
#   19 Aug 2026 -- D11/D12/D13 execution-order clarification
#   20 Aug 2026 -- D8/D9 inheritance clarification
#   23 Aug 2026 -- D4 reapplication under the D12 log-scale branch
#
# The operative protocol state after the third addendum is protocol-v1.3.

PROTOCOL_FREEZE_TAG = "protocol-v1.0"
PROTOCOL_TAG = "protocol-v1.3"
ROW = "D5"
FALLBACK_ROW = "D6"

# D1 as frozen at protocol-v1.0. Any departure at run time requires a
# dated written addendum; the guard below enforces this rather than
# trusting the operator to notice.
D1_ENVIRONMENT = {
    "python": "3.12.13",
    "statsmodels": "0.14.6",
    "pandas": "2.3.3",
    "numpy": "2.5.2",
    "scipy": "1.18.0",
    "pmdarima": "2.1.1",
}


def observed_environment() -> dict:
    return {
        "python": platform.python_version(),
        "statsmodels": statsmodels.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pmdarima": pm.__version__,
        "scikit-learn": sklearn.__version__,
    }


def check_environment(allow_mismatch: bool) -> list:
    """Compare the live interpreter against D1. Abort on mismatch."""
    obs = observed_environment()
    diffs = ["{}: D1 requires {}, found {}".format(k, v, obs[k])
             for k, v in D1_ENVIRONMENT.items() if obs[k] != v]
    if not diffs:
        return diffs
    banner = ("D1 ENVIRONMENT MISMATCH -- the frozen protocol environment "
              "is not the one running:\n  " + "\n  ".join(diffs))
    if not allow_mismatch:
        raise SystemExit(
            banner + "\n\nA run under a substituted stack is not a "
            "protocol execution. Either restore the D1 environment or "
            "file a dated addendum recording the change and its reason, "
            "then update D1_ENVIRONMENT to match. To test the pipeline "
            "on synthetic data under a non-D1 stack, pass "
            "--allow-env-mismatch; all artifacts will be stamped "
            "NON-PROTOCOL.")
    print("=" * 72)
    print(banner)
    print("--allow-env-mismatch set: continuing, artifacts stamped "
          "NON-PROTOCOL.")
    print("=" * 72)
    return diffs

SPINE_START = "2023-07-01"
TRAIN_END = "2025-11-30"
N_TRAIN = 884

SEASONAL_M = 7
# D2/D3 outcomes on the COUNT scale. On the log1p branch these are
# overridden in main() by --d/--D, which must carry the D2/D3 outcomes
# re-run on log(y+1) per the 19 Aug 2026 addendum.
FIXED_d = 1
FIXED_D = 0
WITH_INTERCEPT = False  # D4
IC = "aicc"

FOURIER_PERIOD = 365.25
FOURIER_ORIGIN = SPINE_START  # t = 0 at the first spine date

ALPHA = 0.05

# D5 Ljung-Box lag, computed per fitted model rather than hard-coded:
#   L = max(2s, k_arma + 3), capped at floor(T/5),
# where k_arma = p + q + P + Q is the number of estimated ARMA
# parameters and T is the TRAINING sample size, T = N_TRAIN = 884, the
# frozen quantity named by D5 ("does not bind at T = 884") -- not the
# post-differencing effective sample size. For k_arma <= 11 the rule
# returns 14 (= 2s at s = 7), the value the protocol quotes for
# "typical df"; at the search-bound corner (p,q,P,Q) = (5,5,2,2) it
# returns 17, leaving df = 3. floor(884/5) = 176, non-binding.
LB_CAP_DIVISOR = 5


def lb_lag_for(k_arma: int) -> int:
    """Frozen D5 lag rule, evaluated for one fitted model."""
    lag = max(2 * SEASONAL_M, k_arma + 3)
    cap = N_TRAIN // LB_CAP_DIVISOR
    return min(lag, cap)


HK_BOUNDS = dict(
    start_p=2, start_q=2, start_P=1, start_Q=1,
    max_p=5, max_q=5, max_P=2, max_Q=2,
)
MAXITER = 50  # pmdarima default, stated explicitly

CANDIDATE_ORDER = [
    "monthly",
    "fourier_K1", "fourier_K2", "fourier_K3",
    "fourier_K4", "fourier_K5", "fourier_K6",
]


# ----------------------------- regressor builders ---------------------------
def monthly_dummies(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly indicators under the frozen D5 reference-month rule.
    Without a constant, all 12 indicators enter.
    With a constant, January is omitted as the reference month.
    """
    X = pd.DataFrame(index=idx)

    months = range(2, 13) if WITH_INTERCEPT else range(1, 13)

    for m in months:
        X["month_{:02d}".format(m)] = (idx.month == m).astype(float)

    return X


def fourier_terms(idx: pd.DatetimeIndex, K: int) -> pd.DataFrame:
    t = (idx - pd.Timestamp(FOURIER_ORIGIN)).days.to_numpy(dtype=float)
    X = pd.DataFrame(index=idx)
    for k in range(1, K + 1):
        w = 2.0 * np.pi * k * t / FOURIER_PERIOD
        X["fourier_sin{}".format(k)] = np.sin(w)
        X["fourier_cos{}".format(k)] = np.cos(w)
    return X


def build_candidate(name: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    if name == "monthly":
        return monthly_dummies(idx)
    if name.startswith("fourier_K"):
        return fourier_terms(idx, int(name.split("K")[1]))
    raise ValueError("unknown candidate: {}".format(name))


# --------------------------------- loading ----------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_training_series(args) -> tuple[pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(args.data)
    if args.date_col not in df.columns:
        sys.exit("date column '{}' not found; columns present: {}".format(
            args.date_col, list(df.columns)))
    if args.y_col:
        ycol = args.y_col
        if ycol not in df.columns:
            sys.exit("y column '{}' not found; columns present: {}".format(
                ycol, list(df.columns)))
    else:
        others = [c for c in df.columns if c != args.date_col]
        if len(others) != 1:
            sys.exit("--y-col required (cannot infer); columns present: {}".format(
                list(df.columns)))
        ycol = others[0]

    idx = pd.DatetimeIndex(pd.to_datetime(df[args.date_col]).dt.normalize())
    s = pd.Series(pd.to_numeric(df[ycol]).to_numpy(), index=idx, name=ycol
                  ).sort_index()

    problems = []
    if s.index.has_duplicates:
        problems.append("duplicate dates present")
    if len(s) != N_TRAIN:
        problems.append(
            "expected {} training rows, found {}".format(N_TRAIN, len(s))
        )
    
    if len(s) and (
        s.index[0] != pd.Timestamp(SPINE_START)
        or s.index[-1] != pd.Timestamp(TRAIN_END)
    ):
        problems.append(
            "training data must run {}..{}; found {}..{}".format(
                SPINE_START,
                TRAIN_END,
                s.index[0].date(),
                s.index[-1].date(),
            )
        )
    
    train_spine = pd.date_range(SPINE_START, TRAIN_END, freq="D")
    
    if len(s) == N_TRAIN and not s.index.equals(train_spine):
        problems.append("training daily spine has gaps or extra dates")
    if s.isna().any():
        problems.append("NaNs in the series")
    if (s < 0).any():
        problems.append("negative counts")
    vals = s.to_numpy(dtype=float)
    if not np.allclose(vals, np.round(vals)):
        problems.append("non-integer values")
    if problems:
        sys.exit("input validation failed: " + "; ".join(problems))

    raw_train = s.astype(float)
    
    if args.scale == "log1p":
        s = np.log1p(s)
    
    train = s.astype(float)
    
    return s, train, raw_train


# ------------------------------- fit records --------------------------------
@dataclass
class FitRecord:
    candidate: str
    p: int
    q: int
    P: int
    Q: int
    k_arma: int
    n_exog: int
    k_params_total: int
    aicc: float
    aic: float
    nobs: int
    nobs_effective: int
    llf_burn: int
    converged: object  # bool or None
    lb_lag: int
    lb_stat: float
    lb_df: int
    lb_pvalue: float
    lb_pass: bool


def record_from_fit(candidate: str, fit) -> FitRecord:
    res = fit.arima_res_
    p, d_, q = (int(v) for v in fit.order)
    P, D_, Q, m_ = (int(v) for v in fit.seasonal_order)
    if d_ != FIXED_d or D_ != FIXED_D or m_ != SEASONAL_M:
        raise RuntimeError(
            "unexpected differencing/period in visited fit: {}{}".format(
                fit.order, fit.seasonal_order))
    k_arma = p + q + P + Q

    burn = int(getattr(res, "loglikelihood_burn", 0))
    resid = np.asarray(res.resid, dtype=float)[burn:]

    try:
        aicc = float(res.aicc)
    except Exception:
        aicc = float(res.info_criteria("aicc"))

    conv = None
    mr = getattr(res, "mle_retvals", None)
    if isinstance(mr, dict) and "converged" in mr:
        conv = bool(mr["converged"])

    n_eff = int(res.nobs_effective)
    lag = lb_lag_for(k_arma)
    lb_df = lag - k_arma
    if lb_df < 1:
        raise RuntimeError(
            "frozen lag rule yielded df = {} at k_arma = {} (n_eff = {}); "
            "this is unreachable unless the T/5 cap binds "
            "(floor({}/5) = {}).".format(
                lb_df, k_arma, n_eff, N_TRAIN,
                N_TRAIN // LB_CAP_DIVISOR))
    tab = acorr_ljungbox(resid, lags=[lag], model_df=k_arma)
    stat = float(tab["lb_stat"].iloc[0])
    pval = float(tab["lb_pvalue"].iloc[0])
    if not np.isfinite(pval):
        raise RuntimeError(
            "Ljung-Box returned a non-finite p-value for "
            "({},{},{},{}) at lag {}".format(p, q, P, Q, lag))
    passed = pval > ALPHA

    return FitRecord(
        candidate=candidate, p=p, q=q, P=P, Q=Q, k_arma=k_arma,
        n_exog=int(getattr(res.model, "k_exog", 0)),
        k_params_total=int(np.asarray(res.params).shape[0]),
        aicc=aicc, aic=float(res.aic),
        nobs=int(res.nobs), nobs_effective=n_eff,
        llf_burn=burn, converged=conv,
        lb_lag=int(lag), lb_stat=stat, lb_df=int(lb_df), lb_pvalue=pval,
        lb_pass=bool(passed),
    )


def run_candidate(name: str, y: pd.Series, X: pd.DataFrame,
                  bounds: dict, maxiter: int):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = pm.auto_arima(
            y, X=X,
            d=FIXED_d, D=FIXED_D, m=SEASONAL_M, seasonal=True,
            stepwise=True, information_criterion=IC,
            with_intercept=WITH_INTERCEPT,
            method="lbfgs", maxiter=maxiter,
            trace=2, error_action="ignore", suppress_warnings=True,
            return_valid_fits=True,
            **bounds,
        )
    fits = list(out) if isinstance(out, (list, tuple)) else [out]
    seen, uniq = set(), []
    for f in fits:
        key = (tuple(f.order), tuple(f.seasonal_order))
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    recs = [record_from_fit(name, f) for f in uniq]
    return recs, buf.getvalue()


# --------------------------------- report -----------------------------------
def _fmt(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(f):
        return "-"
    return "{:.{nd}f}".format(f, nd=nd)


def write_report(outdir: Path, selection: dict, winners: pd.DataFrame,
                 recs_df: pd.DataFrame, y_raw: pd.Series) -> None:
    g = selection["gate"]
    sel = selection["selected"]
    c = selection["counts"]
    est = selection["estimation_sample"]
    intercept_label = (
        "with intercept"
        if selection["selected"]["with_intercept"]
        else "no intercept"
    )
    L = []
    L.append("# D5 run report -- annual form and baseline ARIMA orders")
    L.append("")
    smoke_note = ("  **SMOKE MODE -- reduced, non-protocol settings; not a "
                  "protocol execution.**" if selection["smoke_mode"] else "")
    if selection.get("environment_mismatch"):
        smoke_note += ("  **NON-PROTOCOL: the D1 frozen environment was not "
                       "in force. " + "; ".join(
                           selection["environment_mismatch"]) +
                       ". This run is not a protocol execution unless a "
                       "dated addendum records the change.**")
    L.append("Protocol state {} (freeze tag {}; environment per D1 as "
             "frozen), rows D5 (selection) and D6 (fallback). Run (UTC): "
             "{}.{}".format(selection["protocol_tag"],
                            selection["protocol_freeze_tag"],
                            selection["run_utc"], smoke_note))
    L.append("")
    L.append("Input: `{}`; SHA-256 `{}`.".format(
        selection["data"]["path"], selection["data"]["sha256"]))
    L.append("Estimation sample: training only, through {} (n = {}). "
             "Effective observations after state-space burn-in: {}, identical "
             "across all visited fits -- the D5 AICc-comparability requirement "
             "was checked, not assumed.".format(
                 est["train_end"], est["n_train"], est["nobs_effective"]))
    tr = np.asarray(y_raw, dtype=float)
    ref = ("frozen reference: 29, 2, 83, 6.86" if not selection["smoke_mode"]
           else "synthetic data -- frozen reference not applicable")
    L.append("Training echo (raw counts): median {:.0f}, min {:.0f}, "
             "max {:.0f}, variance/mean {:.2f} ({}).".format(
                 np.median(tr), tr.min(), tr.max(),
                 tr.var(ddof=1) / tr.mean(), ref))
    if selection["scale"] == "log1p":
        L.append("")
        L.append("**Modelling scale: log(y + 1)** -- D12 branch of the "
                 "19 Aug 2026 addendum (D5 re-run after the D11 trigger "
                 "held on the count-scale M1 residuals). AICc values in "
                 "this report are on the log scale and are not comparable "
                 "to any count-scale run.")
    L.append("")
    L.append("## Gate and selection rule as executed")
    L.append("")
    L.append(
        "The Hyndman-Khandakar stepwise search ran unconstrained per "
        "candidate (pmdarima auto_arima, stepwise, IC = AICc, d = {}, "
        "D = {}, m = {}, {}). return_valid_fits=True returned "
        "every model the walk fitted successfully; the Ljung-Box gate "
        "(lag L = max(2s, k_arma+3) capped at floor(T/5), evaluated per "
        "fitted model and observed here as {}; df = L - k_arma with "
        "k_arma = p+q+P+Q; alpha = {}; pass iff p > alpha; "
        "computed on each fit's own residuals after dropping the "
        "burn-in of {} observation(s)) was then applied to that entire "
        "visited pool, per the D5 wording and the D6 phrase 'no "
        "candidate visited'. Selected: lowest-AICc visited fit that "
        "passes. D6 fallback: lowest-AICc visited fit overall, failure "
        "recorded. Fits that errored or were rejected by pmdarima's "
        "root check are absent from the pool (no residuals, no AICc; "
        "not selectable) and remain visible in the per-candidate trace "
        "files.".format(
            FIXED_d,
            FIXED_D,
            SEASONAL_M,
            intercept_label,
            "/".join(str(v) for v in g["lag_observed"]),
            g["alpha"],
            g["llf_burn"],
        )
    )
    L.append("")
    L.append("## Recorded implementation choices")
    L.append("")
    if WITH_INTERCEPT:
        L.append(
            "1. D4 includes a constant because d = D = 0. The monthly "
            "candidate therefore uses 11 indicators with January as the "
            "reference month, following the frozen D5 reference-month rule."
        )
    else:
        L.append(
            "1. D4 removes the constant because d and D are not both zero. "
            "The monthly candidate therefore uses all 12 indicators; the "
            "D5 reference-month clause is inactive."
        )
    L.append("2. AICc comparability guard: nobs_effective asserted identical "
             "across all visited fits (observed value: {}).".format(
                 est["nobs_effective"]))
    L.append("3. Ljung-Box applied to each candidate fit's own post-burn "
             "residuals, never to residuals from any other row's model.")
    L.append("4. The Ljung-Box lag follows the frozen formula per fitted "
             "model, L = max(2s, k_arma+3) capped at floor(T/5); it is not "
             "hard-coded. L = 14 while k_arma <= 11 and rises to 17 at the "
             "(5,.,5)(2,.,2) search corner, so df >= 3 always and no fit is "
             "untestable. The cap, floor(T/5) with T = {} training "
             "observations = {}, is non-binding. Observed L: {}.".format(
                 N_TRAIN, N_TRAIN // LB_CAP_DIVISOR,
                 ", ".join(str(v) for v in g["lag_observed"])))
    L.append("5. Deterministic tie-break: AICc, then total estimated "
             "parameter count, then fixed candidate order ({}), then "
             "(p, q, P, Q).".format(", ".join(CANDIDATE_ORDER)))
    mi = selection["hk_settings"]["maxiter"]
    L.append("6. error_action='ignore'; optimizer convergence logged per "
             "fit; maxiter = {}{}.".format(
                 mi, " (pmdarima default)" if mi == MAXITER else
                 " (NON-DEFAULT, smoke override)"))
    L.append("")
    L.append("## Selection")
    L.append("")
    ordr = sel["order"]
    sord = sel["seasonal_order"]
    spec = "{}, ARIMA({},{},{})({},{},{})[{}], {}".format(
        sel["candidate"],
        ordr[0], ordr[1], ordr[2],
        sord[0], sord[1], sord[2], sord[3],
        intercept_label,
    )
    if selection["d6_triggered"]:
        L.append("**D6 fallback triggered** -- no visited fit passed the "
                 "Ljung-Box gate. Stepwise winner recorded: {}; AICc {}; "
                 "LB p-value {} (df {}). The gate failure is recorded per "
                 "D6.".format(spec, _fmt(sel["aicc"], 2),
                              _fmt(sel["lb_pvalue"]), sel["lb_df"]))
    else:
        L.append("Selected: **{}** -- lowest AICc ({}) among visited fits "
                 "passing Ljung-Box at lag {} (p = {}, df = {}, alpha = "
                 "{}).".format(spec, _fmt(sel["aicc"], 2), sel["lb_lag"],
                               _fmt(sel["lb_pvalue"]), sel["lb_df"],
                               g["alpha"]))
    if sel["converged"] is False:
        L.append("")
        L.append("**WARNING: the selected fit's optimizer did not report "
                 "convergence. Review before use.**")
    L.append("")
    L.append("## Per-candidate stepwise winners (D5: 'record the resulting "
             "AICc')")
    L.append("")
    L.append("| candidate | (p,q)(P,Q) | AICc | LB p (df) | LB pass |")
    L.append("|---|---|---|---|---|")
    for _, r in winners.iterrows():
        L.append("| {} | ({},{})({},{}) | {} | {} ({}) | {} |".format(
            r["candidate"], r["p"], r["q"], r["P"], r["Q"],
            _fmt(r["aicc"], 2), _fmt(r["lb_pvalue"]), r["lb_df"],
            "yes" if r["lb_pass"] else "no"))
    L.append("")
    L.append("## Counts")
    L.append("")
    L.append("Candidates: {}. Visited valid fits (pooled): {}. LB-passing: "
             "{}. Optimizer non-converged: {}.".format(
                 c["candidates"], c["visited_valid_fits"], c["lb_pass"],
                 c["not_converged"]))
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("d5_visited_fits.csv (full gate table, AICc ascending), "
             "d5_candidate_winners.csv, d5_selection.json (machine-readable "
             "spec for downstream rows, incl. the Fourier origin), "
             "trace_<candidate>.txt (raw stepwise traces), this report.")
    L.append("")
    (outdir / "d5_report.md").write_text("\n".join(L))


# ---------------------------------- mlflow ----------------------------------
def log_mlflow(args, selection: dict, outdir: Path) -> None:
    import mlflow

    mlflow.set_experiment(args.mlflow_experiment)
    hk = selection["hk_settings"]
    run_name = "D5_smoke" if selection["smoke_mode"] else "D5"
    with mlflow.start_run(run_name=run_name):
        params = {
            "protocol_tag": selection["protocol_tag"],
            "row": ROW,
            "data_sha256": selection["data"]["sha256"],
            "train_end": TRAIN_END,
            "n_train": N_TRAIN,
            "lb_lag_rule": "max(2s, k_arma+3), cap floor(T/5), T=884",
            "alpha": ALPHA,
            "smoke_mode": selection["smoke_mode"],
            "scale": selection["scale"],
        }
        params.update(hk)
        mlflow.log_params(params)
        metrics = {
            "n_visited": selection["counts"]["visited_valid_fits"],
            "n_lb_pass": selection["counts"]["lb_pass"],
            "n_not_converged": selection["counts"]["not_converged"],
            "d6_triggered": int(selection["d6_triggered"]),
            "selected_aicc": selection["selected"]["aicc"],
        }
        if selection["selected"]["lb_pvalue"] is not None:
            metrics["selected_lb_pvalue"] = selection["selected"]["lb_pvalue"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(outdir))


# ----------------------------------- main -----------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="D5/D6: annual-form and ARIMA-order selection "
                    "({})".format(PROTOCOL_TAG)
    )

    ap.add_argument(
        "--data",
        required=True,
        help="CSV with the frozen 884-day training count series "
             "(2023-07-01 through 2025-11-30; kept outside the repository)",
    )
    ap.add_argument("--date-col", default="date")
    ap.add_argument(
        "--y-col",
        default=None,
        help="count column; inferred only if the CSV has exactly two columns",
    )
    ap.add_argument("--outdir", default="results/d5")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow-experiment", default="thesis-baselines")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced NON-PROTOCOL config for pipeline testing "
                         "on synthetic data only")
    ap.add_argument("--scale", choices=["count", "log1p"], default="count",
                    help="modelling scale. 'count' is the default D5 run. "
                         "'log1p' implements the D12 branch of the "
                         "19 Aug 2026 addendum (re-run of D5 on log(y+1) "
                         "if and only if the D11 trigger held on the "
                         "count-scale M1 residuals); validation still "
                         "applies to the raw counts before transforming")
    ap.add_argument( "--d",
                    type=int,
                    choices=[0, 1],
                    default=None,
                    help="d from D2 re-run on log(y+1); required with "
                         "--scale log1p, forbidden otherwise",
    )
    ap.add_argument(
                    "--D",
                    dest="D_seasonal",
                    type=int,
                    choices=[0, 1],
                    default=None,
                    help="D from D3 re-run on log(y+1); required with "
                         "--scale log1p, forbidden otherwise",
    )
    ap.add_argument("--allow-env-mismatch", action="store_true",
                    help="run despite a D1 environment mismatch; artifacts "
                         "are stamped NON-PROTOCOL (testing only)")
    args = ap.parse_args()

    global FIXED_d, FIXED_D, WITH_INTERCEPT

    if args.scale == "log1p":
        if args.d is None or args.D_seasonal is None:
            sys.exit(
                "--scale log1p requires --d and --D carrying the "
                "D2/D3 outcomes re-run on log(y+1); count-scale "
                "d=1, D=0 must not be reused automatically."
            )

        FIXED_d = int(args.d)
        FIXED_D = int(args.D_seasonal)

    elif args.d is not None or args.D_seasonal is not None:
        sys.exit(
            "--d/--D are only valid with --scale log1p; the "
            "count-scale run uses the selected D2/D3 outcomes "
            "d=1, D=0."
        )

    # D4: constant iff d = D = 0.
    WITH_INTERCEPT = (FIXED_d == 0 and FIXED_D == 0)

    print(
        "operative D2/D3/D4: d={}, D={}, constant={}".format(
            FIXED_d, FIXED_D, WITH_INTERCEPT
        )
    )

    env_diffs = check_environment(args.allow_env_mismatch)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    digest = sha256_of(data_path)

    bounds = dict(HK_BOUNDS)
    maxiter = MAXITER
    candidates = list(CANDIDATE_ORDER)
    if args.smoke:
        bounds.update(max_p=2, max_q=2, max_P=1, max_Q=1)
        maxiter = 30
        print("=" * 72)
        print("SMOKE MODE -- reduced order caps and maxiter. "
              "NOT the protocol configuration.")
        print("=" * 72)

    print("environment: python {}, numpy {}, pandas {}, statsmodels {}, "
          "pmdarima {}".format(platform.python_version(), np.__version__,
                               pd.__version__, statsmodels.__version__,
                               pm.__version__))

    s, y, y_raw = load_training_series(args)
    print("loaded {} days ({}..{}); training n = {} through {}; "
          "sha256 = {}".format(len(s), s.index[0].date(), s.index[-1].date(),
                               len(y), TRAIN_END, digest[:16]))
    tr = y_raw.to_numpy(dtype=float)
    print("training summary (raw counts): median {:.0f}, min {:.0f}, "
          "max {:.0f}, var/mean {:.2f}".format(
              np.median(tr), tr.min(), tr.max(),
              tr.var(ddof=1) / tr.mean()))
    if args.scale == "log1p":
        print("modelling scale: log(y + 1) -- D12 branch "
              "(19 Aug 2026 addendum)")

    all_recs = []
    for name in candidates:
        X = build_candidate(name, y.index)
        print("[{}] stepwise search, {} exogenous columns ...".format(
            name, X.shape[1]), flush=True)
        t0 = time.time()
        try:
            recs, trace = run_candidate(name, y, X, bounds, maxiter)
        except Exception as e:
            sys.exit("[{}] stepwise search failed entirely: {}".format(
                name, e))
        (outdir / "trace_{}.txt".format(name)).write_text(trace)
        n_pass = sum(r.lb_pass for r in recs)
        print("[{}] visited(valid) = {}, LB-pass = {}, winner AICc = {:.2f} "
              "({:.1f} s)".format(name, len(recs), n_pass,
                                  min(r.aicc for r in recs),
                                  time.time() - t0), flush=True)
        all_recs.extend(recs)

    recs_df = pd.DataFrame([asdict(r) for r in all_recs])

    ne = recs_df["nobs_effective"].unique()
    if len(ne) != 1:
        recs_df.to_csv(outdir / "d5_visited_fits.csv", index=False)
        sys.exit("AICc comparability violated: nobs_effective differs across "
                 "visited fits: {}. Table written for inspection; aborting "
                 "per the D5 comparability requirement.".format(sorted(ne)))
    nobs_eff = int(ne[0])

    cand_rank = {c: i for i, c in enumerate(CANDIDATE_ORDER)}
    recs_df["cand_rank"] = recs_df["candidate"].map(cand_rank)
    recs_df = recs_df.sort_values(
        by=["aicc", "k_params_total", "cand_rank", "p", "q", "P", "Q"],
        kind="mergesort").reset_index(drop=True)

    winners = (recs_df.loc[recs_df.groupby("candidate")["aicc"].idxmin()]
               .sort_values("cand_rank").reset_index(drop=True))

    passing = recs_df[recs_df["lb_pass"]]
    if len(passing):
        sel = passing.iloc[0]
        d6 = False
    else:
        sel = recs_df.iloc[0]
        d6 = True

    sel_cols = list(build_candidate(sel["candidate"], y.index).columns)
    if sel["candidate"] == "monthly":
        annual = {"kind": "monthly_dummies", "K": None, "period": None,
                  "origin": None}
    else:
        annual = {"kind": "fourier",
                  "K": int(sel["candidate"].split("K")[1]),
                  "period": FOURIER_PERIOD, "origin": FOURIER_ORIGIN}
    annual["columns"] = sel_cols

    selection = {
        "protocol_tag": PROTOCOL_TAG,
        "protocol_freeze_tag": PROTOCOL_FREEZE_TAG,
        "row": ROW,
        "fallback_row": FALLBACK_ROW,
        "d6_triggered": bool(d6),
        "smoke_mode": bool(args.smoke),
        "scale": args.scale,
        "differencing": {
            "d": FIXED_d, "D": FIXED_D,
            "source": ("D2/D3 re-run on log(y+1), operator-supplied"
                       if args.scale == "log1p"
                       else "frozen count-scale D2/D3 outcomes"),
        },
        "selected": {
            "candidate": str(sel["candidate"]),
            "order": [int(sel["p"]), FIXED_d, int(sel["q"])],
            "seasonal_order": [int(sel["P"]), FIXED_D, int(sel["Q"]),
                               SEASONAL_M],
            "with_intercept": WITH_INTERCEPT,
            "aicc": float(sel["aicc"]),
            "lb_pvalue": (None if not np.isfinite(sel["lb_pvalue"])
                          else float(sel["lb_pvalue"])),
            "lb_lag": int(sel["lb_lag"]),
            "lb_df": int(sel["lb_df"]),
            "converged": (None if pd.isna(sel["converged"])
                          else bool(sel["converged"])),
        },
        "annual_regressor": annual,
        "candidate_winners": [
            {"candidate": str(r["candidate"]), "p": int(r["p"]),
             "q": int(r["q"]), "P": int(r["P"]), "Q": int(r["Q"]),
             "aicc": float(r["aicc"]),
             "lb_pvalue": (None if not np.isfinite(r["lb_pvalue"])
                           else float(r["lb_pvalue"])),
             "lb_pass": bool(r["lb_pass"])}
            for _, r in winners.iterrows()
        ],
        "gate": {"test": "ljung_box",
                 "lag_rule": "max(2*s, k_arma+3) capped at floor(T/5), T = n_train = {}".format(N_TRAIN),
                 "lag_observed": sorted(set(int(v) for v in
                                            recs_df["lb_lag"])),
                 "alpha": ALPHA,
                 "df_rule": "lag - (p+q+P+Q)",
                 "pass_rule": "pvalue > alpha",
                 "residuals": "per-fit, post burn-in",
                 "llf_burn": int(sel["llf_burn"])},
        "counts": {
            "candidates": len(candidates),
            "visited_valid_fits": int(len(recs_df)),
            "lb_pass": int(recs_df["lb_pass"].sum()),
            "not_converged": int((recs_df["converged"] == False).sum()),
        },
        "estimation_sample": {"train_end": TRAIN_END, "n_train": N_TRAIN,
                              "nobs_effective": nobs_eff},
        "hk_settings": dict(bounds, d=FIXED_d, D=FIXED_D, m=SEASONAL_M,
                            information_criterion=IC, stepwise=True,
                            with_intercept=WITH_INTERCEPT, maxiter=maxiter,
                            method="lbfgs", error_action="ignore"),
        "data": {"path": str(data_path), "sha256": digest},
        "d1_environment_frozen": D1_ENVIRONMENT,
        "environment_mismatch": env_diffs,
        "environment": observed_environment(),
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    recs_df.drop(columns=["cand_rank"]).to_csv(
        outdir / "d5_visited_fits.csv", index=False)
    winners.drop(columns=["cand_rank"]).to_csv(
        outdir / "d5_candidate_winners.csv", index=False)
    (outdir / "d5_selection.json").write_text(json.dumps(selection, indent=2))
    write_report(outdir, selection, winners, recs_df, y_raw)

    if args.mlflow:
        log_mlflow(args, selection, outdir)

    print()
    if d6:
        print("D6 FALLBACK TRIGGERED: no visited fit passed the Ljung-Box "
              "gate; stepwise winner recorded.")
    print("selected: {} ARIMA({},{},{})({},{},{})[{}], AICc {:.2f}, "
          "LB p {}".format(
              selection["selected"]["candidate"],
              *selection["selected"]["order"],
              *selection["selected"]["seasonal_order"],
              selection["selected"]["aicc"],
              _fmt(selection["selected"]["lb_pvalue"])))
    if selection["selected"]["converged"] is False:
        print("WARNING: selected fit did not report optimizer convergence.")
    print("visited(valid) pooled = {}, LB-pass = {}, "
          "non-converged = {}".format(
              selection["counts"]["visited_valid_fits"],
              selection["counts"]["lb_pass"],
              selection["counts"]["not_converged"]))
    print("outputs written to {}".format(outdir.resolve()))
    

if __name__ == "__main__":
    main()
