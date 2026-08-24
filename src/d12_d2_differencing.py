"""D2 re-run on log(y + 1) -- differencing order d.

First step of the D12 branch. The 19 August 2026 addendum fixes the
return sequence as D2, D3, D4, D5 on the transformed series; the
23 August 2026 clarification puts D4 in that sequence. This script does
D2 only, and refuses to run unless the D11 trigger actually fired.

Frozen D2 rule as implemented
-----------------------------
Candidate set d in {0, 1}. d = 0 if and only if BOTH
    ADF (Dickey & Fuller 1979) rejects a unit root at 5%
    AND
    KPSS (Kwiatkowski et al. 1992) does NOT reject stationarity at 5%
otherwise d = 1. d = 2 is excluded a priori.

Both tests are applied to the OLS residuals of y on the pre-declared
pilot annual form -- Fourier K = 3 (period 365.25, origin 2023-07-01,
the frozen D5 builder) plus day-of-week dummies -- used for this test
only and discarded afterwards. On the log branch, y is log(y + 1).

Confirmatory check: if d = 1, both tests are re-run on the differenced
residuals and the outcome is REPORTED AS A DIAGNOSTIC ONLY. A failure
there is recorded and never used to raise d, since d = 2 is excluded.

Decision arithmetic
-------------------
Each rejection decision is taken by comparing the test statistic to the
5% critical value, not to an interpolated p-value: KPSS p-values come
from a lookup table that statsmodels clips at its endpoints (0.01, 0.10),
so a clipped p-value cannot be compared to 0.05 honestly. p-values are
recorded alongside, and the run aborts if the two routes disagree --
that disagreement would mean the decision sits inside the interpolation
error of the table and needs to be resolved by hand rather than
silently.

Pilot-form parameterisation
---------------------------
The pilot regression uses an intercept plus six day-of-week dummies
rather than seven dummies without one. The two parameterisations span
the same column space, so the OLS residuals -- the only thing carried
forward -- are numerically identical. The choice is therefore inert.

NOT fixed by the frozen row, and recorded in the output so the choice is
visible rather than silent:
  * ADF lag length. Implemented as autolag='AIC' (statsmodels default)
    with regression='c'.
  * KPSS bandwidth. Implemented as nlags='auto' (Hobijn et al. data
    dependent rule) with regression='c' (level stationarity; the pilot
    form contains no linear trend).
If the frozen protocol fixes either differently, change ADF_KWARGS /
KPSS_KWARGS and re-run; both are echoed into the JSON and the report.

Outputs (--outdir, default results/d12_d2)
------------------------------------------
    d12_d2_differencing.json   machine-readable decision -> D5's --d
    d12_d2_report.md           prose record for the research log
    d12_d2_pilot_residuals.csv date, fitted, residual (PROPRIETARY:
                               on the log scale, expm1(fitted+residual)
                               reconstructs the case counts -- keep this
                               file out of the public repository)

Environment: verified at start-up against the D1 freeze, as at D5/D11.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm

try:
    import d5_baseline_order_selection as d5
except ImportError:
    sys.exit("d5_baseline_order_selection.py must be importable (same "
             "directory or on PYTHONPATH): this script reuses its loader, "
             "its frozen Fourier builder and its D1 environment guard.")

# ----------------------------- frozen constants ------------------------------
ROW = "D2"
BRANCH = "D12"
PILOT_FOURIER_K = 3
D_CANDIDATES = (0, 1)
ALPHA = 0.05
CRIT_KEY = "5%"

# Not fixed by the frozen row
ADF_KWARGS = dict(regression="c", autolag="AIC")
KPSS_KWARGS = dict(regression="c", nlags="auto")


def pilot_residuals(y: pd.Series) -> tuple[np.ndarray, np.ndarray,
                                           list[str]]:
    """OLS residuals of y on the pre-declared pilot annual form."""
    X = d5.fourier_terms(y.index, PILOT_FOURIER_K)
    for i in range(6):
        X["dow_{}".format(i)] = (y.index.dayofweek == i).astype(float)
    X = add_constant(X, prepend=True, has_constant="raise")
    fit = sm.OLS(y.to_numpy(dtype=float), X.to_numpy(dtype=float)).fit()
    return (np.asarray(fit.resid, dtype=float),
            np.asarray(fit.fittedvalues, dtype=float),
            list(X.columns))


def adf_rejects(x: np.ndarray) -> dict:
    """ADF: null = unit root. Rejection at 5% supports stationarity."""
    stat, pval, usedlag, nobs, crit, _ = adfuller(x, **ADF_KWARGS)
    crit5 = float(crit[CRIT_KEY])
    by_crit = bool(stat < crit5)          # left-tailed
    by_p = bool(pval < ALPHA)
    return {"test": "ADF", "null": "unit root", "statistic": float(stat),
            "pvalue": float(pval), "crit_5pct": crit5,
            "used_lag": int(usedlag), "nobs": int(nobs),
            "rejects_by_critical_value": by_crit, "rejects_by_pvalue": by_p,
            "rejects": by_crit, "settings": dict(ADF_KWARGS)}


def kpss_rejects(x: np.ndarray) -> dict:
    """KPSS: null = stationarity. Rejection at 5% denies stationarity."""
    stat, pval, nlags, crit = kpss(x, **KPSS_KWARGS)
    crit5 = float(crit[CRIT_KEY])
    by_crit = bool(stat > crit5)          # right-tailed
    by_p = bool(pval < ALPHA)
    clipped = bool(pval <= 0.01 or pval >= 0.10)
    return {"test": "KPSS", "null": "stationarity", "statistic": float(stat),
            "pvalue": float(pval), "pvalue_table_clipped": clipped,
            "crit_5pct": crit5, "used_lags": int(nlags),
            "rejects_by_critical_value": by_crit, "rejects_by_pvalue": by_p,
            "rejects": by_crit, "settings": dict(KPSS_KWARGS)}


def check_routes_agree(res: dict, where: str) -> None:
    if res["rejects_by_critical_value"] != res["rejects_by_pvalue"]:
        sys.exit(
            "{} {}: the critical-value route and the p-value route "
            "disagree (statistic {:.6f} vs 5% critical value {:.6f}; "
            "p = {:.6f}). The 5% decision is numerically ambiguous under "
            "the two reported decision routes and must be resolved before "
            "continuing.".format(
                where,
                res["test"],
                res["statistic"],
                res["crit_5pct"],
                res["pvalue"],
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="D12/D2: differencing order d on log(y+1)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--trigger", required=True,
                    help="path to d11_scale_trigger.json; this script "
                         "refuses to run unless the trigger fired")
    ap.add_argument("--outdir", default="results/d12_d2")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()
    # The whole point of this row: the transform is in force.
    args.scale = "log1p"
    args.d = None
    args.D_seasonal = None

    env_diffs = d5.check_environment(args.allow_env_mismatch)

    trig = json.loads(Path(args.trigger).read_text())
    if not trig.get("decision", {}).get("triggered"):
        sys.exit("the supplied D11 record shows the scale trigger did NOT "
                 "fire. D12 is not entered; execution continues on the "
                 "count scale at D6/D7. Re-running D2 on log(y+1) without "
                 "the trigger that authorises it would be a departure "
                 "from the frozen protocol.")
    if trig.get("smoke_mode"):
        print("WARNING: the D11 record was produced in SMOKE MODE; this "
              "run inherits that status and is not a protocol execution.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    digest = d5.sha256_of(data_path)
    if digest != trig["data"]["sha256"]:
        sys.exit("input SHA-256 {} does not match the file D11 ran on "
                 "({}). The transformed D2 must use the same extraction."
                 .format(digest[:16], trig["data"]["sha256"][:16]))

    # Validation runs on the raw counts; the transform is applied after.
    s, y, y_raw = d5.load_training_series(args)
    print("loaded {} training days through {}; sha256 = {}".format(
        len(y), d5.TRAIN_END, digest[:16]))
    print("scale in force: log(y + 1)  [D12 branch, D11 trigger fired]")
    if not np.allclose(y.to_numpy(dtype=float),
                       np.log1p(y_raw.to_numpy(dtype=float))):
        sys.exit("the working series is not log1p of the raw counts; the "
                 "transform was not applied as expected.")

    resid, fitted, cols = pilot_residuals(y)
    print("pilot form: Fourier K={} + day-of-week dummies, {} regressors "
          "incl. constant; residual sd {:.6f}".format(
              PILOT_FOURIER_K, len(cols), resid.std(ddof=len(cols))))

    adf_lv = adf_rejects(resid)
    check_routes_agree(adf_lv, "levels")
    kpss_lv = kpss_rejects(resid)
    check_routes_agree(kpss_lv, "levels")

    d_selected = 0 if (adf_lv["rejects"] and not kpss_lv["rejects"]) else 1
    assert d_selected in D_CANDIDATES

    confirm = None
    if d_selected == 1:
        dresid = np.diff(resid)

        adf_df = adf_rejects(dresid)
        check_routes_agree(adf_df, "first-differenced residuals")

        kpss_df = kpss_rejects(dresid)
        check_routes_agree(kpss_df, "first-differenced residuals")

        confirm = {
            "applied_to": "first difference of the pilot residuals",
            "n": int(len(dresid)),
            "adf": adf_df,
            "kpss": kpss_df,
            "clean": bool(
                adf_df["rejects"] and not kpss_df["rejects"]
            ),
            "status": (
                "diagnostic only -- never used to raise d, "
                "since d = 2 is excluded a priori"
            ),
        }

    pd.DataFrame({"date": y.index, "fitted": fitted,
                  "residual": resid}).to_csv(
        outdir / "d12_d2_pilot_residuals.csv", index=False)

    out = {
        "protocol_tag": d5.PROTOCOL_TAG,
        "protocol_freeze_tag": d5.PROTOCOL_FREEZE_TAG,
        "row": ROW,
        "branch": BRANCH,
        "scale": "log1p",
        "authorised_by": {
            "d11_record": str(Path(args.trigger).resolve()),
            "triggered": True,
            "jarque_bera_p": trig["jarque_bera"]["pvalue"],
            "spearman_rho": trig["spearman_absresid_vs_fitted"]["rho"],
        },
        "pilot_form": {
            "fourier_K": PILOT_FOURIER_K,
            "fourier_period": d5.FOURIER_PERIOD,
            "fourier_origin": d5.FOURIER_ORIGIN,
            "day_of_week": "intercept + 6 dummies (spans the same column "
                           "space as 7 dummies without an intercept; "
                           "residuals identical)",
            "regressors": cols,
            "scope": "this test only; discarded afterwards",
        },
        "candidate_set": list(D_CANDIDATES),
        "alpha": ALPHA,
        "levels": {"adf": adf_lv, "kpss": kpss_lv},
        "decision": {
            "rule": "d = 0 iff ADF rejects at 5% AND KPSS does not reject "
                    "at 5%; otherwise d = 1. d = 2 excluded a priori.",
            "adf_rejects_unit_root": adf_lv["rejects"],
            "kpss_rejects_stationarity": kpss_lv["rejects"],
            "d": d_selected,
        },
        "confirmatory_check": confirm,
        "next_step": ("D3 (OCSB at s = 7) on log(y + 1), then D4, then D5 "
                      "with --scale log1p --d {} --D <from D3>".format(
                          d_selected)),
        "smoke_mode": bool(trig.get("smoke_mode", False)),
        "environment_mismatch": env_diffs,
        "environment": d5.observed_environment(),
        "d1_environment_frozen": d5.D1_ENVIRONMENT,
        "data": {"path": str(data_path), "sha256": digest},
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (outdir / "d12_d2_differencing.json").write_text(json.dumps(out, indent=2))

    L = []
    L.append("# D12/D2 run report -- differencing order on log(y + 1)")
    L.append("")
    stamp = ""
    if env_diffs:
        stamp += ("  **NON-PROTOCOL: D1 environment not in force: "
                  + "; ".join(env_diffs) + ".**")
    if out["smoke_mode"]:
        stamp += "  **Inherited SMOKE MODE from the D11 record.**"
    L.append("Protocol state {} (freeze tag {}), row D2 re-run on the "
             "transformed scale as the first step of the D12 branch "
             "(19 and 23 August 2026 addenda). Run (UTC): {}.{}".format(
                 out["protocol_tag"], out["protocol_freeze_tag"],
                 out["run_utc"], stamp))
    L.append("")
    L.append("Authorised by the D11 record: the scale trigger fired "
             "(Jarque-Bera p = {:.4g}; Spearman rho = {:.4f}). Input "
             "`{}`, SHA-256 `{}`, matching the file D11 ran on.".format(
                 out["authorised_by"]["jarque_bera_p"],
                 out["authorised_by"]["spearman_rho"],
                 out["data"]["path"], digest))
    L.append("")
    L.append("Both tests are applied to the OLS residuals of log(y + 1) on "
             "the pre-declared pilot annual form: Fourier K = {} at period "
             "{} with origin {}, plus day-of-week dummies ({} regressors "
             "including the constant). The pilot form is used for this "
             "test only and is discarded afterwards; it is not the annual "
             "form selected at D5.".format(
                 PILOT_FOURIER_K, d5.FOURIER_PERIOD, d5.FOURIER_ORIGIN,
                 len(cols)))
    L.append("")
    L.append("## Tests on the pilot residuals (levels)")
    L.append("")
    L.append("| test | null | statistic | 5% critical | p | rejects |")
    L.append("|---|---|---|---|---|---|")
    for r in (adf_lv, kpss_lv):
        L.append("| {} | {} | {:.6f} | {:.6f} | {:.4g}{} | {} |".format(
            r["test"], r["null"], r["statistic"], r["crit_5pct"],
            r["pvalue"],
            " (table endpoint)" if r.get("pvalue_table_clipped") else "",
            "yes" if r["rejects"] else "no"))
    L.append("")
    L.append("Rejection is decided by comparing each statistic to its 5% "
             "critical value rather than to an interpolated p-value, "
             "because KPSS p-values are clipped at the endpoints of a "
             "lookup table. Both routes agreed for both tests; a "
             "disagreement aborts the run.")
    L.append("")
    L.append("## Decision")
    L.append("")
    L.append("Rule: {}".format(out["decision"]["rule"]))
    L.append("")
    L.append("ADF {} the unit-root null; KPSS {} the stationarity null. "
             "**d = {}.**".format(
                 "rejects" if adf_lv["rejects"] else "does not reject",
                 "rejects" if kpss_lv["rejects"] else "does not reject",
                 d_selected))
    if confirm is not None:
        L.append("")
        L.append("## Confirmatory check on the differenced residuals")
        L.append("")
        L.append("Diagnostic only. A failure here is recorded and is never "
                 "used to raise d, since d = 2 is excluded a priori: it "
                 "would posit a stochastic trend in the growth rate and "
                 "give forecast variance growing as h^3.")
        L.append("")
        L.append("| test | statistic | 5% critical | p | rejects |")
        L.append("|---|---|---|---|---|")
        for r in (confirm["adf"], confirm["kpss"]):
            L.append("| {} | {:.6f} | {:.6f} | {:.4g}{} | {} |".format(
                r["test"], r["statistic"], r["crit_5pct"], r["pvalue"],
                " (table endpoint)" if r.get("pvalue_table_clipped") else "",
                "yes" if r["rejects"] else "no"))
        L.append("")
        L.append("Outcome: the differenced residuals {} the pattern "
                 "expected of a stationary series (ADF rejects, KPSS does "
                 "not).".format("match" if confirm["clean"] else
                                "DO NOT match"))
    L.append("")
    L.append("## Settings not fixed by the frozen row")
    L.append("")
    L.append("ADF: regression='{}', lag selection {}, lag used {}. "
             "KPSS: regression='{}', bandwidth rule {}, lags used {}. "
             "These are recorded because the frozen D2 row does not fix "
             "them.".format(
                 ADF_KWARGS["regression"], ADF_KWARGS["autolag"],
                 adf_lv["used_lag"], KPSS_KWARGS["regression"],
                 KPSS_KWARGS["nlags"], kpss_lv["used_lags"]))
    L.append("")
    L.append("## Next step")
    L.append("")
    L.append(out["next_step"] + ". Under the 23 August 2026 clarification "
             "D4 is re-applied on the transformed scale, and the constant "
             "follows deterministically from d and D.")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("d12_d2_differencing.json (machine-readable decision), "
             "d12_d2_pilot_residuals.csv (**proprietary: exp(fitted + "
             "residual) - 1 reconstructs the daily case counts; keep out "
             "of the public repository**), this report.")
    (outdir / "d12_d2_report.md").write_text("\n".join(L) + "\n")

    print("")
    print("ADF   stat {:.6f} vs 5% crit {:.6f} (p {:.4g}) -> unit root {}"
          .format(adf_lv["statistic"], adf_lv["crit_5pct"],
                  adf_lv["pvalue"],
                  "REJECTED" if adf_lv["rejects"] else "not rejected"))
    print("KPSS  stat {:.6f} vs 5% crit {:.6f} (p {:.4g}) -> stationarity {}"
          .format(kpss_lv["statistic"], kpss_lv["crit_5pct"],
                  kpss_lv["pvalue"],
                  "REJECTED" if kpss_lv["rejects"] else "not rejected"))
    print("")
    print("D2 DECISION: d = {}".format(d_selected))
    if confirm is not None:
        print("confirmatory check on differenced residuals: {}".format(
            "clean" if confirm["clean"] else "FLAGGED (diagnostic only)"))
    print("next: {}".format(out["next_step"]))
    print("outputs written to {}".format(outdir.resolve()))


if __name__ == "__main__":
    main()
