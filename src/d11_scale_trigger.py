"""D11 -- pre-declared scale trigger, evaluated on the M1 residuals.

Row D11 of the frozen protocol, executed at the point fixed by the
addendum of 19 August 2026; the D12 return sequence reflects the
23 August 2026 D4 clarification.
Execution order for Section D: D1, D2, D3, D4, D5, D11, (D12 if triggered, 
returning to D2, D3, D4 and D5 on log(y+1)), D6 as applicable, D7, D8, D9, D10.

What this script does
---------------------
1. Reads d5_selection.json and refits EXACTLY the specification recorded
   there -- same annual regressor (kind, K, period, origin), same ARIMA
   and seasonal orders, same intercept setting, same optimizer settings,
   same training slice.
2. Asserts the refit reproduces the AICc recorded at D5 to within
   REFIT_RTOL. A refit that does not reproduce the D5 AICc within 
   the declared numerical tolerance is not accepted for D11 until 
   the discrepancy is resolved and the trigger must not be evaluated on it.
3. Computes the two pre-declared trigger statistics on that fit's own
   residuals, using the same post-burn-in slice as the D5 Ljung-Box gate
   (burn = d + D*s observations).
4. Records the decision. It does NOT transform anything and does not run
   D12; it tells you which branch to execute next.

Trigger as implemented
----------------------
    A. Jarque-Bera on the residuals rejects normality at JB_ALPHA.
    B. Spearman correlation between |residual| and fitted value
       exceeds SPEARMAN_THRESHOLD.
    Trigger fires iff A AND B (joint).


Residual definition
-------------------
One-step-ahead in-sample prediction errors from the fitted state-space
model, on the scale in force at D5 (count scale unless a prior D12 has
already been executed), after dropping loglikelihood_burn = d + D*s
observations -- identical to the residuals the D5 gate used. Fitted
values are taken from the same slice, so residual i pairs with the
prediction that produced it.

Outputs (--outdir, default results/d11)
---------------------------------------
    d11_scale_trigger.json   machine-readable decision + both statistics
    d11_report.md            prose record for the research log
    d11_residuals.csv        date, fitted, residual, |residual| (the
                             inputs to both statistics, so the numbers
                             in the report can be recomputed by hand)

Environment: verified at start-up against the D1 freeze, as at D5.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
import scipy
import sklearn
import statsmodels
from scipy import stats
from statsmodels.stats.stattools import jarque_bera

try:
    import d5_baseline_order_selection as d5
except ImportError:
    sys.exit("d5_baseline_order_selection.py must be importable (same "
             "directory or on PYTHONPATH): D11 reuses its loader, its "
             "annual-regressor builders and its D1 environment guard so "
             "that the refit is constructed identically to D5.")

# ----------------------------- frozen constants ------------------------------
ROW = "D11"
FALLBACK_ROW = "D12"
JB_ALPHA = 0.01
SPEARMAN_THRESHOLD = 0.30


# The refit must reproduce D5's AICc. Tolerance is for optimizer
# non-determinism across runs, not for specification drift.

REFIT_RTOL = 1e-6

def refit_selected(sel: dict, y: pd.Series):
    """Refit the D5-selected specification on the training sample."""
    ann = sel["annual_regressor"]
    if ann["kind"] == "monthly_dummies":
        cand = "monthly"
    elif ann["kind"] == "fourier":
        cand = "fourier_K{}".format(ann["K"])
    else:
        sys.exit("unrecognised annual regressor kind: {}".format(ann["kind"]))

    X = d5.build_candidate(cand, y.index)
    if list(X.columns) != list(ann["columns"]):
        sys.exit("regressor columns rebuilt as {} but D5 recorded {}; the "
                "annual form is not being reproduced".format(
                    list(X.columns), ann["columns"]))

    order = tuple(sel["selected"]["order"])
    seasonal = tuple(sel["selected"]["seasonal_order"])
    hk = sel["hk_settings"]
    model = pm.ARIMA(
        order=order, seasonal_order=seasonal,
        with_intercept=sel["selected"]["with_intercept"],
        maxiter=hk["maxiter"], method=hk["method"],
        suppress_warnings=False,
    )
    model.fit(y.to_numpy(dtype=float), X=X.to_numpy(dtype=float))
    return model, X, cand


def main() -> None:
    ap = argparse.ArgumentParser(description="D11 scale trigger")
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--selection", required=True,
                    help="path to d5_selection.json from the D5 run whose "
                         "M1 specification the trigger is evaluated on")
    ap.add_argument("--outdir", default="results/d11")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()
    args.scale = "count"
    args.d = None
    args.D_seasonal = None

    env_diffs = d5.check_environment(args.allow_env_mismatch)

    sel = json.loads(Path(args.selection).read_text())
    if sel.get("smoke_mode"):
        print("WARNING: the D5 selection was produced in SMOKE MODE; this "
              "D11 run inherits that status and is not a protocol "
              "execution.")
    if sel.get("scale") != "count":
        sys.exit("the supplied D5 selection was produced on the {} scale. "
                 "D11 is evaluated on the count-scale M1 residuals; a "
                 "second scale decision is not admissible (D13 requires "
                 "one decision, applied uniformly).".format(sel.get("scale")))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    digest = d5.sha256_of(data_path)
    if digest != sel["data"]["sha256"]:
        sys.exit("input SHA-256 {} does not match the file D5 ran on ({}). "
                 "The trigger must be evaluated on the same extraction."
                 .format(digest[:16], sel["data"]["sha256"][:16]))

    s, y, y_raw = d5.load_training_series(args)
    print("loaded {} days; training n = {} through {}; sha256 = {}".format(
        len(s), len(y), d5.TRAIN_END, digest[:16]))

    model, X, cand = refit_selected(sel, y)
    res = model.arima_res_
    aicc = float(res.info_criteria("aicc"))
    recorded = float(sel["selected"]["aicc"])
    if not np.isclose(aicc, recorded, rtol=REFIT_RTOL, atol=0.0):
        sys.exit(
            "refit AICc {:.9f} does not reproduce the D5 value "
            "{:.9f} within relative tolerance {}. Resolve before "
            "evaluating the trigger.".format(
                aicc, recorded, REFIT_RTOL
            )
        )
    conv = getattr(res, "mle_retvals", {}).get("converged", None)
    print("refit reproduces D5 AICc {:.6f} (converged: {})".format(
        aicc, conv))

    d_sel = int(sel["selected"]["order"][1])
    D_sel = int(sel["selected"]["seasonal_order"][1])
    int_sel = bool(sel["selected"]["with_intercept"])

    # D4 is deterministic: constant iff d = D = 0.
    expected_intercept = (d_sel == 0 and D_sel == 0)
    if int_sel != expected_intercept:
        sys.exit(
            "the selection file is internally inconsistent with D4: "
            "selected d = {}, D = {} imply with_intercept = {}, "
            "but the selected fit records with_intercept = {}.".format(
                d_sel, D_sel, expected_intercept, int_sel
            )
        )

    d5_diff = sel.get("differencing")
    if d5_diff is not None and (
        int(d5_diff["d"]) != d_sel
        or int(d5_diff["D"]) != D_sel
    ):
        sys.exit(
            "the selection file is internally inconsistent: it records "
            "d = {}, D = {} but its selected order implies d = {}, "
            "D = {}.".format(
                d5_diff["d"], d5_diff["D"], d_sel, D_sel
            )
        )

    # Newer D5 artifacts separately record the D4 outcome. Older D5
    # artifacts may not contain this key, so absence does not invalidate
    # the already-completed count-scale run.
    d5_int = sel.get("d4_intercept")
    if (
        d5_int is not None
        and bool(d5_int["with_intercept"]) != int_sel
    ):
        sys.exit(
            "the selection file is internally inconsistent: it records "
            "the D4 outcome as with_intercept = {} but the selected "
            "fit has with_intercept = {}.".format(
                d5_int["with_intercept"], int_sel
            )
        )

    burn = int(sel["gate"]["llf_burn"])
    resid = np.asarray(res.resid, dtype=float)[burn:]
    fitted = np.asarray(res.fittedvalues, dtype=float)[burn:]
    dates = y.index[burn:]
    if not (len(resid) == len(fitted) == int(sel["estimation_sample"]
                                             ["nobs_effective"])):
        sys.exit("residual slice length {} does not match the D5 effective "
                 "sample {}".format(len(resid),
                                    sel["estimation_sample"]
                                    ["nobs_effective"]))

    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    cond_a = bool(jb_p < JB_ALPHA)

    rho, rho_p = stats.spearmanr(np.abs(resid), fitted)
    rho = float(rho)
    cond_b = bool(rho > SPEARMAN_THRESHOLD)

    triggered = bool(cond_a and cond_b)

    pd.DataFrame({"date": dates, "fitted": fitted, "residual": resid,
                  "abs_residual": np.abs(resid)}).to_csv(
        outdir / "d11_residuals.csv", index=False)

    out = {
        "protocol_tag": d5.PROTOCOL_TAG,
        "protocol_freeze_tag": d5.PROTOCOL_FREEZE_TAG,
        "row": ROW,
        "fallback_row": FALLBACK_ROW,
        "evaluated_on": {
            "source_selection": str(Path(args.selection).resolve()),
            "candidate": cand,
            "order": list(sel["selected"]["order"]),
            "seasonal_order": list(sel["selected"]["seasonal_order"]),
            "with_intercept": sel["selected"]["with_intercept"],
            "scale": sel["scale"],
            "refit_aicc": aicc,
            "d5_recorded_aicc": recorded,
            "refit_converged": conv,
            "n_residuals": int(len(resid)),
            "burn_in_dropped": burn,
        },
        "jarque_bera": {
            "statistic": float(jb_stat), "pvalue": float(jb_p),
            "skew": float(skew), "kurtosis": float(kurt),
            "alpha": JB_ALPHA, "rejects_normality": cond_a,
        },
        "spearman_absresid_vs_fitted": {
            "rho": rho, "abs_rho": abs(rho), "pvalue": float(rho_p),
            "threshold": SPEARMAN_THRESHOLD,
            "reading": "rho(|residual|, fitted) > threshold",
            "exceeds_threshold": cond_b,
        },
        "decision": {
        "rule": (
            "trigger iff (JB rejects at {}) AND "
            "(Spearman rho(|residual|, fitted) > {})"
        ).format(JB_ALPHA, SPEARMAN_THRESHOLD),
            "condition_a_normality_rejected": cond_a,
            "condition_b_heteroskedastic": cond_b,
            "triggered": triggered,
            "next_step": (
                "D12: re-run D2, D3, D4 and D5 on log(y+1), then "
                "continue the ladder on the transformed scale "
                "(19 Aug and 23 Aug 2026 addenda)"
                if triggered else
                "D6 as applicable, then D7: continue on the count scale"
            ),
        },
        "differencing": {
            "d": d_sel,
            "D": D_sel,
            "source": "inherited from count-scale D5 selection",
            "d5_recorded_source": (d5_diff or {}).get("source"),
        },
        "constant": {
            "with_intercept": int_sel,
            "source": "inherited from count-scale D5 selection",
            "d5_recorded_source": (d5_int or {}).get("source"),
        },
        "smoke_mode": bool(sel.get("smoke_mode", False)),
        "environment_mismatch": env_diffs,
        "environment": d5.observed_environment(),
        "d1_environment_frozen": d5.D1_ENVIRONMENT,
        "data": {"path": str(data_path), "sha256": digest},
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (outdir / "d11_scale_trigger.json").write_text(
        json.dumps(out, indent=2))

    L = []
    L.append("# D11 run report -- pre-declared scale trigger")
    L.append("")
    stamp = ""
    if env_diffs:
        stamp += ("  **NON-PROTOCOL: D1 environment not in force: "
                  + "; ".join(env_diffs) + ".**")
    if out["smoke_mode"]:
        stamp += "  **Inherited SMOKE MODE from the D5 selection.**"
    L.append("Protocol state {} (freeze tag {}), row D11, evaluated at the "
             "point fixed by the addendum of 19 August 2026: after D5, "
             "before D7. Run (UTC): {}.{}".format(
                 out["protocol_tag"], out["protocol_freeze_tag"],
                 out["run_utc"], stamp))
    L.append("")
    L.append("Evaluated on the residuals of the M1 specification selected "
             "at D5: **{}, ARIMA{}{}[{}]{}**, refitted on the training "
             "sample and reproducing the D5 AICc of {:.4f} to within "
             "relative tolerance {:g}. Residuals are that fit's own one-step-ahead in-sample "
             "prediction errors after dropping the burn-in of {} "
             "observation(s), n = {} -- the same slice the D5 Ljung-Box "
             "gate used.".format(
                 cand, tuple(out["evaluated_on"]["order"]),
                 tuple(out["evaluated_on"]["seasonal_order"][:3]),
                 out["evaluated_on"]["seasonal_order"][3],
                 "" if out["evaluated_on"]["with_intercept"]
                 else ", no intercept",
                 recorded, REFIT_RTOL, burn, len(resid)))
    L.append("")
    L.append("## Trigger statistics")
    L.append("")
    L.append("| condition | statistic | value | threshold | holds |")
    L.append("|---|---|---|---|---|")
    L.append("| A: non-normality | Jarque-Bera p | {:.4g} | < {} | {} |"
             .format(jb_p, JB_ALPHA, "yes" if cond_a else "no"))
    L.append("| B: heteroskedasticity | Spearman rho(|resid|, fitted) | "
             "{:.4f} | > {} | {} |".format(
                 rho, SPEARMAN_THRESHOLD, "yes" if cond_b else "no"))
    L.append("")
    L.append("Residual skewness {:.4f}, excess kurtosis {:.4f}. Spearman "
             "signed rho = {:.4f} (|rho| = {:.4f}), p = {:.4g}; the "
             "reading applied is `{}`.".format(
                 skew, kurt - 3.0, rho, abs(rho), rho_p,
                 out["spearman_absresid_vs_fitted"]["reading"]))
    L.append("")
    L.append("## Decision")
    L.append("")
    L.append("Rule: {}.".format(out["decision"]["rule"]))
    L.append("")
    L.append("**Trigger {}.** {}".format(
        "FIRES" if triggered else "does not fire",
        out["decision"]["next_step"] + "."))
    L.append("")
    L.append("The scale decision is made once, on training diagnostics "
             "only, and applies uniformly to every model in the ladder, "
             "as D13 requires. The test set was not consulted.")
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("d11_scale_trigger.json (machine-readable decision), "
             "d11_residuals.csv (date, fitted, residual, |residual| -- the "
             "inputs to both statistics), this report.")
    (outdir / "d11_report.md").write_text("\n".join(L) + "\n")

    print("")
    print("Jarque-Bera p = {:.4g} (alpha {}): normality {}".format(
        jb_p, JB_ALPHA, "REJECTED" if cond_a else "not rejected"))
    print("Spearman rho(|resid|, fitted) = {:.4f} (threshold {}): {}".format(
        rho, SPEARMAN_THRESHOLD,
        "exceeded" if cond_b else "not exceeded"))
    print("")
    print("TRIGGER {}".format("FIRES -- next: D12" if triggered
                              else "DOES NOT FIRE -- next: D7 on the count "
                                   "scale"))
    print("outputs written to {}".format(outdir.resolve()))


if __name__ == "__main__":
    main()
