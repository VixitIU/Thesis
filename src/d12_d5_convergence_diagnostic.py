"""Diagnostic refit of the transformed-scale D5-selected M1.

This is NOT a protocol selection step and does not replace, amend or
re-run D5. The D5-selected specification remains fixed.

Purpose
-------
The transformed-scale D5 winner did not report optimizer convergence
under the frozen D5 maxiter = 50 setting. This diagnostic refits that
EXACT specification with progressively larger iteration ceilings while
holding everything else fixed:

    * same log(y + 1) training response;
    * same 884-day training extraction;
    * same annual regressor;
    * same ARIMA and seasonal orders;
    * same D4 intercept outcome;
    * same optimizer method.

Each fit is a fresh fit. No result from one iteration budget is used to
initialise another. Therefore maxiter is the only diagnostic setting
being varied.

The purpose is to determine whether the D5 convergence warning reflects
the 50-iteration optimizer ceiling or a more persistent estimation
problem.

IMPORTANT
---------
No result from this script changes the D5 winner automatically.
No alternative candidate is considered.
No convergence threshold is added retrospectively to D5.
Any interpretation is diagnostic only.

Outputs (--outdir, default results/d12_d5_diagnostic):
    d12_d5_convergence_diagnostic.json
    d12_d5_convergence_report.md

No residual-level or proprietary daily-series artifact is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pmdarima as pm

try:
    import d5_baseline_order_selection as d5
except ImportError:
    sys.exit(
        "d5_baseline_order_selection.py must be importable (same "
        "directory or on PYTHONPATH): this diagnostic reuses its loader, "
        "annual-regressor builders and D1 environment guard."
    )


#-----------------------Diagnostic constants----------------------------

ROW = "D5_CONVERGENCE_DIAGNOSTIC"

DEFAULT_BUDGETS = (50, 100, 200, 500)

# Numerical comparison only. This does not constitute a selection rule.
AICC_RTOL = 1e-6


def parse_budgets(text: str) -> list[int]:
    try:
        vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    except ValueError:
        sys.exit("--budgets must be comma-separated positive integers")

    if not vals or any(v <= 0 for v in vals):
        sys.exit("--budgets must contain positive integers")

    vals = sorted(set(vals))
    return vals


def annual_candidate_from_selection(sel: dict) -> str:
    ann = sel["annual_regressor"]

    if ann["kind"] == "monthly_dummies":
        return "monthly"

    if ann["kind"] == "fourier":
        return "fourier_K{}".format(int(ann["K"]))

    sys.exit(
        "unrecognised annual-regressor kind in D5 selection: {}"
        .format(ann.get("kind"))
    )


def fit_once(
    y,
    X,
    order: tuple,
    seasonal_order: tuple,
    with_intercept: bool,
    method: str,
    maxiter: int,
) -> dict:

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        model = pm.ARIMA(
            order=order,
            seasonal_order=seasonal_order,
            with_intercept=with_intercept,
            method=method,
            maxiter=maxiter,
            suppress_warnings=False,
        )

        model.fit(
            y.to_numpy(dtype=float),
            X=X.to_numpy(dtype=float),
        )

    res = model.arima_res_

    try:
        aicc = float(res.aicc)
    except Exception:
        aicc = float(res.info_criteria("aicc"))

    mle = getattr(res, "mle_retvals", {})
    if not isinstance(mle, dict):
        mle = {}

    converged = mle.get("converged", None)

    iterations = mle.get("iterations", None)
    if iterations is None:
        iterations = mle.get("iter", None)

    warnflag = mle.get("warnflag", None)
    fcalls = mle.get("fcalls", None)

    gradient_max_abs = None
    gopt = mle.get("gopt", None)

    if gopt is not None:
        arr = np.asarray(gopt, dtype=float)
        if arr.size and np.all(np.isfinite(arr)):
            gradient_max_abs = float(np.max(np.abs(arr)))

    warning_messages = []

    for w in caught:
        text = "{}: {}".format(
            w.category.__name__,
            str(w.message),
        )
        if text not in warning_messages:
            warning_messages.append(text)

    return {
        "maxiter": int(maxiter),
        "converged": (
            None if converged is None else bool(converged)
        ),
        "iterations": (
            None if iterations is None else int(iterations)
        ),
        "warnflag": (
            None if warnflag is None else int(warnflag)
        ),
        "function_calls": (
            None if fcalls is None else int(fcalls)
        ),
        "gradient_max_abs": gradient_max_abs,
        "aicc": aicc,
        "aic": float(res.aic),
        "loglikelihood": float(res.llf),
        "nobs": int(res.nobs),
        "nobs_effective": int(res.nobs_effective),
        "k_params_total": int(
            np.asarray(res.params).shape[0]
        ),
        "warnings": warning_messages,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Diagnostic convergence refits of the transformed-scale "
            "D5-selected specification"
        ),
    )

    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)

    ap.add_argument(
        "--selection",
        required=True,
        help=(
            "path to the transformed-scale d5_selection.json whose "
            "selected specification is being diagnosed"
        ),
    )

    ap.add_argument(
        "--budgets",
        default=",".join(str(v) for v in DEFAULT_BUDGETS),
        help=(
            "comma-separated maxiter values for fresh diagnostic fits; "
            "default: 50,100,200,500"
        ),
    )

    ap.add_argument(
        "--outdir",
        default="results/d12_d5_diagnostic",
    )

    ap.add_argument(
        "--allow-env-mismatch",
        action="store_true",
    )

    args = ap.parse_args()

    budgets = parse_budgets(args.budgets)

    # The working response for this diagnostic is the D12 log scale.
    args.scale = "log1p"
    args.d = None
    args.D_seasonal = None


#-------------------------D1 environment-------------------------------

    env_diffs = d5.check_environment(
        args.allow_env_mismatch
    )


#-----------------------Read D5 selection------------------------------

    selection_path = Path(args.selection)
    sel = json.loads(selection_path.read_text())

    if sel.get("row") != "D5":
        sys.exit(
            "the supplied selection artifact is not a D5 record."
        )

    if sel.get("protocol_tag") != d5.PROTOCOL_TAG:
        sys.exit(
            "protocol-state mismatch: D5 selection records {}, "
            "but this diagnostic expects {}.".format(
                sel.get("protocol_tag"),
                d5.PROTOCOL_TAG,
            )
        )

    if sel.get("scale") != "log1p":
        sys.exit(
            "the supplied D5 selection was produced on the {} scale; "
            "this diagnostic is specifically for the transformed D12 "
            "D5 run.".format(sel.get("scale"))
        )

    if sel.get("smoke_mode"):
        print(
            "WARNING: the supplied D5 selection was produced in "
            "SMOKE MODE; this diagnostic inherits that status."
        )

    order = tuple(
        int(v) for v in sel["selected"]["order"]
    )

    seasonal_order = tuple(
        int(v) for v in sel["selected"]["seasonal_order"]
    )

    d_selected = int(order[1])
    D_selected = int(seasonal_order[1])

    with_intercept = bool(
        sel["selected"]["with_intercept"]
    )

    expected_intercept = (
        d_selected == 0 and D_selected == 0
    )

    if with_intercept != expected_intercept:
        sys.exit(
            "D5 selection is internally inconsistent with D4: "
            "d = {}, D = {} imply with_intercept = {}, but D5 "
            "records {}.".format(
                d_selected,
                D_selected,
                expected_intercept,
                with_intercept,
            )
        )

    recorded_differencing = sel.get("differencing")

    if recorded_differencing is not None:
        if (
            int(recorded_differencing["d"]) != d_selected
            or int(recorded_differencing["D"]) != D_selected
        ):
            sys.exit(
                "D5 differencing provenance is internally "
                "inconsistent with its selected orders."
            )

    recorded_aicc = float(
        sel["selected"]["aicc"]
    )

    recorded_converged = sel["selected"].get(
        "converged"
    )

    hk = sel["hk_settings"]

    frozen_maxiter = int(hk["maxiter"])
    method = str(hk["method"])

    if frozen_maxiter not in budgets:
        budgets = sorted(
            set(budgets + [frozen_maxiter])
        )


#-----------------------Same training extraction-----------------------

    data_path = Path(args.data)
    digest = d5.sha256_of(data_path)

    if digest != sel["data"]["sha256"]:
        sys.exit(
            "input SHA-256 {} does not match the file D5 ran on "
            "({}). The diagnostic must use the identical training "
            "extraction.".format(
                digest[:16],
                sel["data"]["sha256"][:16],
            )
        )

    s, y, y_raw = d5.load_training_series(args)

    if not np.allclose(
        y.to_numpy(dtype=float),
        np.log1p(y_raw.to_numpy(dtype=float)),
    ):
        sys.exit(
            "the working series is not log1p of the raw counts."
        )

    print(
        "loaded {} training days through {}; sha256 = {}".format(
            len(y),
            d5.TRAIN_END,
            digest[:16],
        )
    )

    print(
        "diagnosing fixed D5 selection: {} ARIMA{}{}[{}], "
        "constant={}".format(
            sel["selected"]["candidate"],
            order,
            seasonal_order[:3],
            seasonal_order[3],
            with_intercept,
        )
    )

    print(
        "D5 recorded AICc {:.9f}; converged = {}; "
        "frozen maxiter = {}".format(
            recorded_aicc,
            recorded_converged,
            frozen_maxiter,
        )
    )


#-----------------------Rebuild annual regressor-----------------------

    # Set the imported D5 state to the exact transformed D2/D3/D4
    # outcomes before rebuilding the candidate. This is essential for
    # the monthly case and harmless for Fourier candidates.
    d5.FIXED_d = d_selected
    d5.FIXED_D = D_selected
    d5.WITH_INTERCEPT = with_intercept

    candidate = annual_candidate_from_selection(sel)

    X = d5.build_candidate(
        candidate,
        y.index,
    )

    expected_columns = list(
        sel["annual_regressor"]["columns"]
    )

    if list(X.columns) != expected_columns:
        sys.exit(
            "rebuilt annual regressor columns {} do not match the "
            "D5 selection record {}.".format(
                list(X.columns),
                expected_columns,
            )
        )


#-----------------------Fresh convergence refits-----------------------

    fits = []

    print("")

    for budget in budgets:
        print(
            "[maxiter={}] fresh exact-specification refit ..."
            .format(budget),
            flush=True,
        )

        rec = fit_once(
            y=y,
            X=X,
            order=order,
            seasonal_order=seasonal_order,
            with_intercept=with_intercept,
            method=method,
            maxiter=budget,
        )

        rec["aicc_delta_vs_d5"] = (
            rec["aicc"] - recorded_aicc
        )

        rec["aicc_matches_d5"] = bool(
            np.isclose(
                rec["aicc"],
                recorded_aicc,
                rtol=AICC_RTOL,
                atol=0.0,
            )
        )

        fits.append(rec)

        print(
            "  converged={}  iterations={}  "
            "AICc={:.9f}  delta={:+.9f}".format(
                rec["converged"],
                rec["iterations"],
                rec["aicc"],
                rec["aicc_delta_vs_d5"],
            )
        )


#-----------------------Diagnostic summary-----------------------------

    converged_fits = [
        r for r in fits
        if r["converged"] is True
    ]

    first_converged = (
        converged_fits[0]
        if converged_fits
        else None
    )

    highest_budget = fits[-1]

    summary = {
        "any_higher_budget_converged": any(
            r["converged"] is True
            and r["maxiter"] > frozen_maxiter
            for r in fits
        ),
        "first_converged_maxiter": (
            None
            if first_converged is None
            else first_converged["maxiter"]
        ),
        "highest_budget": highest_budget["maxiter"],
        "highest_budget_converged": (
            highest_budget["converged"]
        ),
        "highest_budget_aicc": (
            highest_budget["aicc"]
        ),
        "highest_budget_aicc_delta_vs_d5": (
            highest_budget["aicc_delta_vs_d5"]
        ),
        "interpretation_constraint": (
            "diagnostic only; D5 selection remains unchanged"
        ),
    }


#------------------------------Output----------------------------------

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = {
        "protocol_tag": d5.PROTOCOL_TAG,
        "protocol_freeze_tag": d5.PROTOCOL_FREEZE_TAG,
        "row": ROW,
        "status": "diagnostic_only",
        "changes_D5_selection": False,

        "source_D5": {
            "selection_record": str(
                selection_path.resolve()
            ),
            "scale": sel["scale"],
            "candidate": candidate,
            "order": list(order),
            "seasonal_order": list(
                seasonal_order
            ),
            "with_intercept": with_intercept,
            "recorded_aicc": recorded_aicc,
            "recorded_converged": recorded_converged,
            "recorded_maxiter": frozen_maxiter,
            "method": method,
        },

        "diagnostic_design": {
            "fresh_fit_each_budget": True,
            "warm_start": False,
            "varied_setting": "maxiter only",
            "budgets": budgets,
            "aicc_comparison_rtol": AICC_RTOL,
        },

        "fits": fits,
        "summary": summary,

        "environment_mismatch": env_diffs,
        "environment": d5.observed_environment(),
        "d1_environment_frozen": d5.D1_ENVIRONMENT,

        "data": {
            "path": str(data_path),
            "sha256": digest,
        },

        "run_utc": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
    }

    (
        outdir
        / "d12_d5_convergence_diagnostic.json"
    ).write_text(
        json.dumps(out, indent=2)
    )


#------------------------------Report----------------------------------

    L = []

    L.append(
        "# D12/D5 convergence diagnostic"
    )
    L.append("")

    L.append(
        "**Diagnostic only. This analysis does not modify the "
        "protocol-selected D5 specification.**"
    )
    L.append("")

    L.append(
        "The transformed-scale D5 winner was **{}, ARIMA{}{}[{}]**, "
        "with_intercept = {}, AICc = {:.9f}. D5 recorded optimizer "
        "convergence as {} under maxiter = {}.".format(
            candidate,
            order,
            seasonal_order[:3],
            seasonal_order[3],
            with_intercept,
            recorded_aicc,
            recorded_converged,
            frozen_maxiter,
        )
    )

    L.append("")
    L.append(
        "The exact same specification was refitted from a fresh "
        "initialisation at each iteration budget. Only `maxiter` was "
        "varied; no alternative D5 candidate was evaluated."
    )

    L.append("")
    L.append("## Results")
    L.append("")

    L.append(
        "| maxiter | converged | iterations | AICc | "
        "delta vs D5 | max |gradient| |"
    )

    L.append(
        "|---|---|---|---|---|---|"
    )

    for r in fits:
        grad = (
            "-"
            if r["gradient_max_abs"] is None
            else "{:.3g}".format(
                r["gradient_max_abs"]
            )
        )

        L.append(
            "| {} | {} | {} | {:.9f} | {:+.9f} | {} |".format(
                r["maxiter"],
                r["converged"],
                (
                    "-"
                    if r["iterations"] is None
                    else r["iterations"]
                ),
                r["aicc"],
                r["aicc_delta_vs_d5"],
                grad,
            )
        )

    L.append("")
    L.append("## Interpretation boundary")
    L.append("")

    L.append(
        "These refits assess optimizer behaviour only. They do not "
        "introduce convergence as a retrospective D5 admissibility "
        "criterion and they do not replace the D5-selected model with "
        "a different candidate. Any downstream methodological decision "
        "must therefore be documented separately from the frozen D5 "
        "selection."
    )

    L.append("")
    L.append("## Artifacts")
    L.append("")

    L.append(
        "d12_d5_convergence_diagnostic.json and this report. "
        "No residual-level proprietary output is written."
    )

    (
        outdir
        / "d12_d5_convergence_report.md"
    ).write_text(
        "\n".join(L) + "\n"
    )


#------------------------------Console---------------------------------

    print("")
    print(
        "diagnostic complete:"
    )

    if first_converged is None:
        print(
            "no tested iteration budget reported convergence"
        )
    else:
        print(
            "first converged fresh refit: maxiter = {}, "
            "AICc = {:.9f}, delta vs D5 = {:+.9f}".format(
                first_converged["maxiter"],
                first_converged["aicc"],
                first_converged[
                    "aicc_delta_vs_d5"
                ],
            )
        )

    print("")
    print(
        "D5 SELECTION REMAINS UNCHANGED"
    )

    print(
        "outputs written to {}".format(
            outdir.resolve()
        )
    )


if __name__ == "__main__":
    main()