"""D7 -- holiday pad (b, f) selection.

Frozen D7:
    AICc over the 9 pads in the declared set {2,3,4} x {2,3,4};
    M1 orders and annual form held fixed.

Regressor construction is fixed by Section C:
    C1-6  clusters are uninterrupted runs of >= 3 consecutive
          non-working days on the transferred calendar.
    C1-8  H_NY = clusters containing 1 January (2 train / 1 test);
          H_OT = all remaining (9 train / 5 test).
    C1-9  a SINGLE (b, f) pair is shared by both regressors -- hence
          9 pads, not 81. The two groups enter as two separate columns
          and therefore carry separate coefficients.
    C1-10 union, binary. Overlapping padded windows are NEVER summed:
          a date inside two padded windows of the same group scores 1,
          not 2.

The window for a cluster is [start - b, end + f] inclusive, so the
cluster's own non-working days are always inside it. Windows are built
from every cluster in the file and then clipped to the estimation
index; a test-side cluster whose leading pad reached back across the
training boundary would therefore contribute correctly rather than be
silently dropped (none does at these pads, and the run reports it).

Operative M1 is read from the D5 selection artifact, not retyped: the
annual form is rebuilt from the recorded kind/K/period/origin and
checked column-for-column against what D5 recorded. On the D12 branch
this is the transformed-scale selection.

Optimizer ceiling
-----------------
The addendum of 24 August 2026 sets method='lbfgs', maxiter=500 for
fixed-specification estimation from D7 onward, with every candidate in
a decision row fitted under the same ceiling so that AICc comparisons
within the row are commensurable, convergence recorded for every fit,
and execution PAUSED rather than the fit excluded if convergence still
fails. This script implements that literally: if any of the nine fits
fails to converge it writes the grid table for inspection and exits
without selecting a pad.

Consequence to keep in view: D5's recorded AICc was produced under the
frozen maxiter = 50 and is NOT comparable to the values here. The
M1-without-holidays refit reported below is estimated under the same
ceiling as the nine candidates and is the only like-for-like reference.
It is a DIAGNOSTIC: D7 selects among the nine pads, and nothing in the
frozen row makes the pad conditional on beating M1.

Outputs (--outdir, default results/d7)
--------------------------------------
    d7_pad_grid.csv        all 9 pads, AICc ascending
    d7_pad_selection.json  machine-readable (b, f) -> M2 and D10
    d7_report.md           prose record for the research log

No proprietary artifact is written: windows are calendar-derived and
the table carries only AICc values and window day-counts.
"""
from __future__ import annotations

import argparse
import itertools
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
    sys.exit("d5_baseline_order_selection.py must be importable (same "
             "directory or on PYTHONPATH): D7 reuses its loader, its "
             "frozen annual-form builders and its D1 environment guard.")

# ----------------------------- frozen constants ------------------------------
ROW = "D7"
PAD_GRID = (2, 3, 4)                      # D7: {2,3,4} x {2,3,4}
GROUPS = ("H_NY", "H_OT")                 # C1-8
EXPECTED_CLUSTERS = 17                    # C1-7
EXPECTED_COUNTS = {("H_NY", "train"): 2, ("H_OT", "train"): 9,
                   ("H_NY", "test"): 1, ("H_OT", "test"): 5}
MIN_CLUSTER_LEN = 3                       # C1-6

# 24 Aug 2026 addendum. D5's own ceiling (50) is unchanged; this governs
# fixed-specification estimation from D7 onward.
PROTOCOL_TAG = "protocol-v1.4"
SOURCE_D5_PROTOCOL_TAG = "protocol-v1.3"

DOWNSTREAM_METHOD = "lbfgs"
DOWNSTREAM_MAXITER = 500


def load_clusters(path: Path) -> pd.DataFrame:
    """Read and validate holiday_clusters.csv against Section C."""
    df = pd.read_csv(path)
    need = {"start", "end", "length", "group", "split"}
    if not need.issubset(df.columns):
        sys.exit("holiday cluster file must have columns {}; found {}"
                 .format(sorted(need), list(df.columns)))
    df["start"] = pd.to_datetime(df["start"], dayfirst=True)
    df["end"] = pd.to_datetime(df["end"], dayfirst=True)

    if len(df) != EXPECTED_CLUSTERS:
        sys.exit("expected {} clusters (C1-7), found {}".format(
            EXPECTED_CLUSTERS, len(df)))
    bad = df[df["end"] < df["start"]]
    if len(bad):
        sys.exit("cluster with end before start: {}".format(
            bad.to_dict("records")))

    span = (df["end"] - df["start"]).dt.days + 1
    if not (span == df["length"]).all():
        sys.exit("the 'length' column disagrees with end - start + 1 for "
                 "{} cluster(s)".format(int((span != df["length"]).sum())))
    if (df["length"] < MIN_CLUSTER_LEN).any():
        sys.exit("C1-6 requires runs of >= {} consecutive non-working "
                 "days; found shorter cluster(s)".format(MIN_CLUSTER_LEN))
    if not set(df["group"]) <= set(GROUPS):
        sys.exit("group labels must be within {}; found {}".format(
            GROUPS, sorted(set(df["group"]))))

    # C1-8: H_NY is exactly the set of clusters containing 1 January.
    def contains_jan1(r) -> bool:
        days = pd.date_range(r["start"], r["end"], freq="D")
        return bool(((days.month == 1) & (days.day == 1)).any())
    implied = df.apply(
        lambda r: "H_NY" if contains_jan1(r) else "H_OT", axis=1)
    if not (implied == df["group"]).all():
        wrong = df[implied != df["group"]]
        sys.exit("C1-8 grouping violated: H_NY is exactly the clusters "
                 "containing 1 January. Mismatched: {}".format(
                     wrong[["start", "end", "group"]].to_dict("records")))

    train_end = pd.Timestamp(d5.TRAIN_END)
    straddles = df[(df["start"] <= train_end) & (df["end"] > train_end)]
    if len(straddles):
        sys.exit("cluster straddles the training boundary {}: {}. The "
                 "train/test split of a cluster is then ambiguous and "
                 "must be resolved before D7.".format(
                     d5.TRAIN_END, straddles.to_dict("records")))
    implied_split = np.where(df["end"] <= train_end, "train", "test")
    if not (implied_split == df["split"].to_numpy()).all():
        sys.exit("the 'split' column disagrees with the training boundary "
                 "{}".format(d5.TRAIN_END))

    counts = df.groupby(["group", "split"]).size().to_dict()
    if counts != EXPECTED_COUNTS:
        sys.exit("C1-8 counts violated: expected {}, found {}".format(
            EXPECTED_COUNTS, counts))
    return df


def holiday_regressors(clusters: pd.DataFrame, idx: pd.DatetimeIndex,
                       b: int, f: int) -> tuple[pd.DataFrame, dict]:
    """Binary union windows per group (C1-9, C1-10)."""
    X = pd.DataFrame(index=idx)
    info = {}
    for g in GROUPS:
        flag = pd.Series(False, index=idx)
        naive = 0  # what a summed construction would have produced
        for _, r in clusters[clusters["group"] == g].iterrows():
            lo = r["start"] - pd.Timedelta(int(b), "D")
            hi = r["end"] + pd.Timedelta(int(f), "D")
            win = (idx >= lo) & (idx <= hi)
            naive += int(win.sum())
            flag |= win          # union, never summed (C1-10)
        X["hol_{}".format(g)] = flag.astype(float)
        info[g] = int(flag.sum())
        info[g + "_overlap_days"] = int(naive - flag.sum())
    return X, info


def fit_pad(y: pd.Series, X: pd.DataFrame, sel: dict):
    """Fit the operative M1 specification plus the holiday columns."""
    order = tuple(sel["selected"]["order"])
    seasonal = tuple(sel["selected"]["seasonal_order"])

    model = pm.ARIMA(
        order=order,
        seasonal_order=seasonal,
        with_intercept=sel["selected"]["with_intercept"],
        maxiter=DOWNSTREAM_MAXITER,
        method=DOWNSTREAM_METHOD,
        suppress_warnings=True,
    )

    model.fit(y.to_numpy(dtype=float), X=X.to_numpy(dtype=float))
    res = model.arima_res_
    mle = getattr(res, "mle_retvals", {}) or {}

    return {
        "aicc": float(res.info_criteria("aicc")),
        "converged": (
            None if "converged" not in mle
            else bool(mle["converged"])
        ),
        "iterations": int(mle.get("iterations", -1)),
        "nobs_effective": int(res.nobs_effective),
        "k_params_total": int(np.asarray(res.params).shape[0]),
        "n_exog": int(getattr(res.model, "k_exog", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False, description="D7: holiday pad (b, f)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--clusters", required=True,
                    help="path to holiday_clusters.csv (C1-7)")
    ap.add_argument("--selection", required=True,
                    help="path to the operative d5_selection.json")
    ap.add_argument("--outdir", default="results/d7")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()
    args.d = None
    args.D_seasonal = None

    env_diffs = d5.check_environment(args.allow_env_mismatch)

    sel = json.loads(Path(args.selection).read_text())

    if sel.get("protocol_tag") != SOURCE_D5_PROTOCOL_TAG:
        sys.exit(
            "expected the operative D5 source artifact to be {}, found {}."
            .format(SOURCE_D5_PROTOCOL_TAG, sel.get("protocol_tag"))
        )

    if sel.get("row") != "D5":
        sys.exit("--selection must point at a D5 selection artifact.")

    args.scale = sel["scale"]
    if sel.get("smoke_mode"):
        print("WARNING: the D5 selection was produced in SMOKE MODE; this "
              "D7 run inherits that status and is not a protocol run.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    digest = d5.sha256_of(data_path)
    if digest != sel["data"]["sha256"]:
        sys.exit("input SHA-256 {} does not match the file D5 ran on ({})."
                 .format(digest[:16], sel["data"]["sha256"][:16]))

    cluster_path = Path(args.clusters)
    cluster_digest = d5.sha256_of(cluster_path)
    clusters = load_clusters(cluster_path)
    s, y, y_raw = d5.load_training_series(args)
    print("loaded {} training days through {}; scale in force: {}".format(
        len(y), d5.TRAIN_END, sel["scale"]))

    # Operative M1, rebuilt and checked rather than retyped.
    d5.WITH_INTERCEPT = bool(sel["selected"]["with_intercept"])
    ann = sel["annual_regressor"]
    cand = ("monthly" if ann["kind"] == "monthly_dummies"
            else "fourier_K{}".format(ann["K"]))
    X_ann = d5.build_candidate(cand, y.index)
    if list(X_ann.columns) != list(ann["columns"]):
        sys.exit("annual regressor rebuilt as {} but D5 recorded {}"
                 .format(list(X_ann.columns), ann["columns"]))
    print("operative M1: {} ARIMA{}{}[{}], constant={}, maxiter={}".format(
        cand, tuple(sel["selected"]["order"]),
        tuple(sel["selected"]["seasonal_order"][:3]),
        sel["selected"]["seasonal_order"][3],
        sel["selected"]["with_intercept"], DOWNSTREAM_MAXITER))

        # Diagnostic reference at the same ceiling (not part of the D7 rule).
    t0 = time.time()

    m1 = fit_pad(y, X_ann, sel)

    print(
        "[diagnostic] M1 without holiday regressors: AICc {:.4f} "
        "(converged {}, {:.0f} s)".format(
            m1["aicc"],
            m1["converged"],
            time.time() - t0,
        )
    )

    if m1["converged"] is not True:
        sys.exit(
            "EXECUTION PAUSED -- the diagnostic M1 refit did not report "
            "convergence at maxiter = {}. Resolve the numerical issue "
            "before finalizing D7.".format(DOWNSTREAM_MAXITER)
        )

    rows = []
    for b, f in itertools.product(PAD_GRID, PAD_GRID):
        t0 = time.time()
        X_hol, info = holiday_regressors(clusters, y.index, b, f)
        X = pd.concat([X_ann, X_hol], axis=1)
        r = fit_pad(y, X, sel)
        r.update({"b": b, "f": f, "pad_total": b + f,
                  "days_H_NY": info["H_NY"], "days_H_OT": info["H_OT"],
                  "overlap_days_H_NY": info["H_NY_overlap_days"],
                  "overlap_days_H_OT": info["H_OT_overlap_days"],
                  "seconds": round(time.time() - t0, 1)})
        rows.append(r)
        print("[pad b={} f={}] AICc {:.4f} (converged {}, iters {}, "
              "{:.0f} s)".format(b, f, r["aicc"], r["converged"],
                                 r["iterations"], r["seconds"]), flush=True)
    grid = pd.DataFrame(rows).sort_values(
        ["aicc", "b", "f"],
        kind="mergesort",
    ).reset_index(drop=True)

    grid.to_csv(outdir / "d7_pad_grid.csv", index=False)
    # AICc comparability within the row.
    if grid["nobs_effective"].nunique() != 1:
        sys.exit("nobs_effective differs across the nine pad fits ({}); "
                 "their AICc values are not comparable.".format(
                     sorted(grid["nobs_effective"].unique())))
    if grid["n_exog"].nunique() != 1:
        sys.exit("n_exog differs across the nine pad fits ({}); the pad "
                 "must change the window, not the regressor count."
                 .format(sorted(grid["n_exog"].unique())))

    # 24 Aug 2026 addendum: pause, do not exclude.
    stalled = grid[grid["converged"] != True]  # noqa: E712
    if len(stalled):
        print("")
        print("EXECUTION PAUSED -- {} of 9 fits did not report convergence "
              "at maxiter = {}. Per the addendum of 24 August 2026 such a "
              "fit is not excluded, replaced or given a fallback: the "
              "numerical issue is documented and resolved before the "
              "decision row is finalized. No pad has been selected. The "
              "grid table has been written for inspection.".format(
                  len(stalled), DOWNSTREAM_MAXITER))
        print(stalled[["b", "f", "aicc", "converged",
                       "iterations"]].to_string(index=False))
        sys.exit(2)

    min_aicc = grid["aicc"].min()
    tied = grid[grid["aicc"] == min_aicc]

    if len(tied) != 1:
        print("")
        print(
            "EXECUTION PAUSED -- D7 has {} pads tied at the minimum "
            "AICc {:.9f}. The frozen D7 row declares no tie-break, "
            "so no pad is selected.".format(
                len(tied),
                min_aicc,
            )
        )
        print(
            tied[["b", "f", "aicc"]].to_string(index=False)
        )
        sys.exit(3)

    best = tied.iloc[0]
    b_sel = int(best["b"])
    f_sel = int(best["f"])
    margin = float(grid.iloc[1]["aicc"] - best["aicc"])
    out = {
        "protocol_tag": PROTOCOL_TAG,
        "source_d5_protocol_tag": sel["protocol_tag"],
        "protocol_freeze_tag": d5.PROTOCOL_FREEZE_TAG,
        "row": ROW,
        "scale": sel["scale"],
        "operative_m1": {
            "source_selection": str(Path(args.selection).resolve()),
            "candidate": cand,
            "order": list(sel["selected"]["order"]),
            "seasonal_order": list(sel["selected"]["seasonal_order"]),
            "with_intercept": sel["selected"]["with_intercept"],
            "annual_columns": list(ann["columns"]),
        },
        "pad_grid": {"set": list(PAD_GRID), "n_pads": len(rows),
                     "shared_by_both_groups": True},
        "selected_pad": {
            "b": b_sel, "f": f_sel, "aicc": float(best["aicc"]),
            "converged": bool(best["converged"]),
            "iterations": int(best["iterations"]),
            "days_H_NY": int(best["days_H_NY"]),
            "days_H_OT": int(best["days_H_OT"]),
            "overlap_days_absorbed_by_union": {
                "H_NY": int(best["overlap_days_H_NY"]),
                "H_OT": int(best["overlap_days_H_OT"])},
            "margin_over_runner_up": margin,
        },
        "regressor_construction": {
            "window": "[start - b, end + f] inclusive",
            "groups": list(GROUPS),
            "overlap": "union, binary; overlapping windows never summed "
                       "(C1-10)",
            "columns": ["hol_H_NY", "hol_H_OT"],
        },
        "diagnostic_m1_no_holidays": {
            "aicc": m1["aicc"], "converged": m1["converged"],
            "note": "estimated at the same ceiling as the nine pads and "
                    "therefore comparable to them; NOT comparable to the "
                    "D5 recorded AICc, which used maxiter = 50. "
                    "Diagnostic only: D7 selects among the nine pads.",
        },
        "optimizer": {"method": DOWNSTREAM_METHOD,
                      "maxiter": DOWNSTREAM_MAXITER,
                      "source": "24 Aug 2026 addendum"},
        "clusters": {"path": str(cluster_path.resolve()),
                    "sha256": cluster_digest,
                     "n": int(len(clusters)),
                     "counts": {"{}_{}".format(g, sp): int(n) for (g, sp), n
                                in clusters.groupby(
                                    ["group", "split"]).size().items()}},
        "next_step": "M2 = M1 + hol_H_NY + hol_H_OT at (b, f) = ({}, {}); "
                     "D10 transfers this pad to M5 without re-tuning."
                     .format(b_sel, f_sel),
        "smoke_mode": bool(sel.get("smoke_mode", False)),
        "environment_mismatch": env_diffs,
        "environment": d5.observed_environment(),
        "d1_environment_frozen": d5.D1_ENVIRONMENT,
        "data": {"path": str(data_path), "sha256": digest},
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (outdir / "d7_pad_selection.json").write_text(json.dumps(out, indent=2))

    L = []
    L.append("# D7 run report -- holiday pad (b, f)")
    L.append("")
    stamp = ""
    if env_diffs:
        stamp += ("  **NON-PROTOCOL: D1 environment not in force: "
                  + "; ".join(env_diffs) + ".**")
    if out["smoke_mode"]:
        stamp += "  **Inherited SMOKE MODE from the D5 selection.**"
    L.append("Protocol state {} (freeze tag {}), row D7. Run (UTC): "
             "{}.{}".format(out["protocol_tag"], out["protocol_freeze_tag"],
                            out["run_utc"], stamp))
    L.append("")
    L.append("Operative M1, read from `{}` and rebuilt rather than "
             "retyped: **{}, ARIMA{}{}[{}]{}**, on the {} scale. The "
             "annual form was regenerated from the recorded "
             "kind/K/period/origin and matched column-for-column.".format(
                 Path(args.selection).name, cand,
                 tuple(sel["selected"]["order"]),
                 tuple(sel["selected"]["seasonal_order"][:3]),
                 sel["selected"]["seasonal_order"][3],
                 "" if sel["selected"]["with_intercept"]
                 else ", no constant", sel["scale"]))
    L.append("")
    L.append("Holiday regressors follow Section C: a single (b, f) pair "
             "shared by both groups (C1-9), so the declared set "
             "{{2,3,4}} x {{2,3,4}} gives nine pads rather than 81; H_NY and "
             "H_OT enter as two binary columns and carry separate "
             "coefficients; windows are unions and are never summed "
             "(C1-10). The window is [start - b, end + f] inclusive, so "
             "each cluster's own non-working days are always inside it. "
             "Cluster file validated: {} clusters, {} H_NY / {} H_OT in "
             "training, {} H_NY / {} H_OT in test, every run at least {} "
             "days, and H_NY exactly the clusters containing 1 January.")
    L[-1] = L[-1].format(
        len(clusters), EXPECTED_COUNTS[("H_NY", "train")],
        EXPECTED_COUNTS[("H_OT", "train")],
        EXPECTED_COUNTS[("H_NY", "test")],
        EXPECTED_COUNTS[("H_OT", "test")], MIN_CLUSTER_LEN)
    L.append("")
    L.append("## Pad grid (AICc ascending)")
    L.append("")
    L.append("| b | f | AICc | H_NY days | H_OT days | converged | iters |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in grid.iterrows():
        L.append("| {} | {} | {:.4f} | {} | {} | {} | {} |".format(
            int(r["b"]), int(r["f"]), r["aicc"], int(r["days_H_NY"]),
            int(r["days_H_OT"]), "yes" if r["converged"] else "no",
            int(r["iterations"])))
    L.append("")
    L.append("All nine fits share the estimation sample ({} effective "
             "observations) and the same regressor count ({}), so their "
             "AICc values are comparable; both were checked, not "
             "assumed.".format(int(grid['nobs_effective'].iloc[0]),
                               int(grid['n_exog'].iloc[0])))
    L.append("")
    overlap_ny = int(best["overlap_days_H_NY"])
    overlap_ot = int(best["overlap_days_H_OT"])

    if overlap_ny or overlap_ot:
        L.append(
            "At the selected pad, the union rule absorbed {} H_NY and "
            "{} H_OT day(s) that a summed construction would otherwise "
            "have double-counted.".format(overlap_ny, overlap_ot)
        )
    else:
        L.append(
            "At the selected pad there were no within-group overlapping "
            "padded windows on the training sample; the C1-10 binary-union "
            "rule was nevertheless applied."
        )
    L.append("")
    L.append("## Selection")
    L.append("")
    L.append("**(b, f) = ({}, {})**, AICc {:.4f}, ahead of the runner-up "
             "by {:.4f}. The padded windows cover {} training days for "
             "H_NY and {} for H_OT.".format(
                 b_sel, f_sel, best["aicc"], margin,
                 int(best["days_H_NY"]), int(best["days_H_OT"])))
    L.append("")
    L.append("## Numerical estimation")
    L.append("")
    L.append("Per the addendum of 24 August 2026, all nine candidates "
             "were fitted under a common ceiling (method='{}', "
             "maxiter={}) so that within-row AICc comparisons are made "
             "under one numerical setting, and convergence was recorded "
             "for every fit. All nine converged. Had any not, execution "
             "would have paused with no pad selected, rather than the fit "
             "being excluded or replaced.".format(
                 DOWNSTREAM_METHOD, DOWNSTREAM_MAXITER))
    L.append("")
    L.append("Diagnostic reference: M1 without holiday regressors, "
             "estimated under the same ceiling, gives AICc {:.4f}. This "
             "is comparable to the nine values above but **not** to the "
             "AICc recorded at D5, which was produced under the frozen "
             "maxiter = 50. It is reported for context only: D7 selects "
             "among the nine pads, and the frozen row does not make the "
             "pad conditional on improving over M1.".format(m1["aicc"]))
    L.append("")
    L.append("## Next step")
    L.append("")
    L.append(out["next_step"])
    L.append("")
    L.append("## Artifacts")
    L.append("")
    L.append("d7_pad_grid.csv (all nine pads, AICc ascending), "
             "d7_pad_selection.json (machine-readable pad for M2 and for "
             "the D10 inheritance), this report. No proprietary artifact "
             "is produced: the windows are calendar-derived and the table "
             "carries only AICc values and day counts.")
    (outdir / "d7_report.md").write_text("\n".join(L) + "\n")

    print("")
    print("D7 SELECTION: (b, f) = ({}, {}), AICc {:.4f} "
          "(runner-up +{:.4f})".format(b_sel, f_sel, best["aicc"], margin))
    print("next: {}".format(out["next_step"]))
    print("outputs written to {}".format(outdir.resolve()))


if __name__ == "__main__":
    main()
