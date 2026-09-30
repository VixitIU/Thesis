#!/usr/bin/env python
"""Thesis exhibits -- post hoc presentation only; no reported result is re-estimated.

Figures, numbered in the order they appear in Chapter 4:

  fig1_case_series.png            4.1    daily cases, split, padded holiday windows
  fig2_acf_pacf_m1.png            4.3.2  correlograms
  fig3_lag_screen.png             4.3.3  raw vs prewhitened CCF, screened lags, dAICc
  fig4_mae_excess_by_horizon.png  4.5    MAE of M2-M5 relative to M1
  fig5_test_window_forecasts.png  4.8    test window and New Year window, h = 7/14/28
  fig6_weekday_mae_m1.png         4.8    M1 absolute error by target weekday (post hoc)

Tables for Appendix C and for checking the prose (CSV, in <outdir>/tables):

  tableC_d5_log_candidates_50_vs_500.csv  per-candidate winners, both ceilings
  tableC_test_calculation.csv             every step of the 15 tests
  tableC_reproducibility.csv              tags, input hashes, environment
  tableC_training_coefficients.csv        indicator coefficients of M2-M5 (descriptive)
  table_mae_excess_pct.csv                MAE above M1, from unrounded values

Estimation performed: refits of M1-M5 on the frozen 884-day training window
only. Every refit must reproduce the AICc recorded by its Section D artifact,
and every input file must carry the SHA-256 recorded by Section E, or the
script stops; every exhibit is thereby drawn from the data and the models the
thesis reports. Nothing is written under results/. A provenance manifest is
written next to the outputs.

Usage (from the project root)
-----------------------------
python src/make_thesis_exhibits.py
    --daily "C:/Users/New/Documents/IU work/Thesis/data/aligned_daily.csv"
    [--date-col obs_date] [--y-col cases] [--results results]
    [--history data/indicator_history.csv] [--clusters data/holiday_clusters.csv]
    [--outdir exhibits] [--dpi 300]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox  

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

SPINE_START = pd.Timestamp("2023-07-01")
TRAIN_END = pd.Timestamp("2025-11-30")
TEST_START = pd.Timestamp("2025-12-01")
SAMPLE_END = pd.Timestamp("2026-07-31")
N_TRAIN = 884
N_TOTAL = 1127
N_ORIGINS = 216
AICC_TOLERANCE = 1e-3
HORIZONS = (7, 14, 28)
M1_SHADES = {7: dict(color="#08306b", ls="-", lw=1.5),
             14: dict(color="#2171b5", ls="--", lw=1.1),
             28: dict(color="#6baed6", ls="-", lw=1.0)}
ACF_LAGS = 35
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]

STYLE = {
    "M0": dict(color="#b35806", ls="--", lw=1.0),
    "M1": dict(color="#08306b", ls="-", lw=1.7),
    "M2": dict(color="#6a51a3", ls="-.", lw=1.0),
    "M3": dict(color="#238b45", ls=":", lw=1.3),
    "M4": dict(color="#4292c6", ls=(0, (5, 1)), lw=1.0),
    "M5": dict(color="#cb181d", ls="-", lw=1.0),
}
ACTUAL = dict(color="#737373", lw=0.7)
HOLIDAY_COLOUR = {"H_NY": "#fdae6b", "H_OT": "#c6dbef"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.titleweight": "normal",
    "axes.labelsize": 5,
    "legend.fontsize": 5,
    "legend.frameon": False,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------- helpers
def require(condition: bool, message: str) -> None:
    if not condition:
        sys.exit("STOP: " + message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    require(path.is_file(), "missing artifact {}".format(path))
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def save(fig, outdir: Path, name: str, dpi: int, written: list) -> None:
    path = outdir / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    written.append(name)
    print("wrote {}".format(path))


def fourier(index: pd.DatetimeIndex, K: int, period: float,
            origin: str) -> pd.DataFrame:
    """Annual Fourier terms exactly as declared in the D5 artifact."""
    t = (index - pd.Timestamp(origin)).days.to_numpy(dtype=float)
    cols = {}
    for k in range(1, K + 1):
        w = 2.0 * np.pi * k * t / period
        cols["fourier_sin{}".format(k)] = np.sin(w)
        cols["fourier_cos{}".format(k)] = np.cos(w)
    return pd.DataFrame(cols, index=index)


def padded_windows(clusters: pd.DataFrame, b: int, f: int) -> pd.DataFrame:
    out = clusters.copy()
    out["win_start"] = out["start"] - pd.Timedelta(b, "D")
    out["win_end"] = out["end"] + pd.Timedelta(f, "D")
    return out


def shade(ax, start, end, colour, **kw) -> None:
    ax.axvspan(start - pd.Timedelta(hours=12), end + pd.Timedelta(hours=12),
               color=colour, lw=0, **kw)


# ------------------------------------------------------------- loading
def load_daily(args) -> pd.Series:
    df = pd.read_csv(args.daily)
    require(args.date_col in df.columns and args.y_col in df.columns,
            "columns {!r}/{!r} not in {}; present: {}".format(
                args.date_col, args.y_col, args.daily, list(df.columns)))
    idx = pd.DatetimeIndex(pd.to_datetime(df[args.date_col]).dt.normalize())
    y = pd.Series(pd.to_numeric(df[args.y_col]).to_numpy(dtype=float),
                  index=idx, name="cases").sort_index()
    require(y.index.is_unique, "duplicate dates in the daily file")
    require(y.index.equals(pd.date_range(SPINE_START, SAMPLE_END, freq="D")),
            "the daily file must be the dense {}-day spine {}..{}".format(
                N_TOTAL, SPINE_START.date(), SAMPLE_END.date()))
    require(bool(np.isfinite(y).all()) and bool((y >= 0).all()),
            "non-finite or negative counts in the daily file")
    return y


def load_clusters(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path)
    for col in ("start", "end"):
        c[col] = pd.to_datetime(c[col], dayfirst=True)
    require(set(c["group"]) == {"H_NY", "H_OT"},
            "unexpected holiday groups {}".format(sorted(set(c["group"]))))
    return c


# ------------------------------------------------------------- refits
def holiday_block(index: pd.DatetimeIndex, clusters: pd.DataFrame, b: int,
                  f: int) -> pd.DataFrame:
    """Union, binary windows [start - b, end + f] per group (C1-8 to C1-10)."""
    out = pd.DataFrame(0.0, index=index, columns=["hol_H_NY", "hol_H_OT"])
    for _, r in clusters.iterrows():
        lo = r["start"] - pd.Timedelta(b, "D")
        hi = r["end"] + pd.Timedelta(f, "D")
        out.loc[(index >= lo) & (index <= hi), "hol_" + r["group"]] = 1.0
    return out


def design(index, d5, spec, clusters, hist, model: str) -> pd.DataFrame:
    ann = d5["annual_regressor"]
    X = fourier(index, int(ann["K"]), float(ann["period"]),
                ann["origin"])[ann["columns"]]
    if model in ("M2", "M5"):
        X = X.join(holiday_block(index, clusters, spec["b_pad"], spec["f_pad"]))
    if model in ("M3", "M5"):
        L = int(spec["fx_lag"])
        X["fx_lag{}".format(L)] = hist["fx_rub_per_thb"].reindex(
            index - pd.Timedelta(L, "D")).to_numpy()
    if model in ("M4", "M5"):
        L = int(spec["search_lag"])
        X["search_lag{}_per1000".format(L)] = hist["search_index"].reindex(
            index - pd.Timedelta(L, "D")).to_numpy() / float(spec["search_divisor"])
    require(bool(np.isfinite(X.to_numpy()).all()),
            "non-finite regressor values in the {} design".format(model))
    return X


def refit(y_train: pd.Series, X: pd.DataFrame, d5: dict, recorded: float,
          label: str):
    import pmdarima as pm

    sel = d5["selected"]
    model = pm.arima.ARIMA(order=tuple(sel["order"]),
                           seasonal_order=tuple(sel["seasonal_order"]),
                           with_intercept=bool(sel["with_intercept"]),
                           method="lbfgs", maxiter=500, suppress_warnings=True)
    model.fit(np.log1p(y_train.to_numpy()), X=X.to_numpy())
    res = model.arima_res_
    require(bool((getattr(res, "mle_retvals", None) or {}).get("converged")),
            "the {} refit did not converge".format(label))
    require(abs(float(res.aicc) - float(recorded)) <= AICC_TOLERANCE,
            "{} refit AICc {:.6f} does not reproduce the recorded {:.6f}; the "
            "exhibit would not show the reported model.".format(
                label, float(res.aicc), float(recorded)))
    return res


def m1_diagnostics(y_train, X, d5):
    sel = d5["selected"]
    res = refit(y_train, X, d5, sel["aicc"], "M1")
    names = list(res.model.param_names)
    params = np.asarray(res.params, dtype=float)
    exog = [i for i, n in enumerate(names)
            if not n.startswith(("ar.", "ma.", "sigma2", "intercept", "drift"))]
    require(len(exog) == X.shape[1], "cannot identify the Fourier coefficients")
    y_log = np.log1p(y_train.to_numpy())
    d_eta = np.diff(y_log - X.to_numpy() @ params[exog])
    resid = np.asarray(res.resid, dtype=float)[1:]       # burn-in 1, as at D5
    k_arma = sum(sel["order"][i] for i in (0, 2)) + \
        sum(sel["seasonal_order"][i] for i in (0, 2))
    lb_p = float(acorr_ljungbox(resid, lags=[int(sel["lb_lag"])],
                                model_df=k_arma)["lb_pvalue"].iloc[0])
    print("M1 refit reproduces the operative D5 AICc ({:.4f}); Ljung-Box p at "
          "lag {} = {:.3f} (recorded {:.3f})".format(
              float(res.aicc), sel["lb_lag"], lb_p, float(sel["lb_pvalue"])))
    return {"d_eta": d_eta, "resid": resid, "aicc": float(res.aicc),
            "recorded_aicc": float(sel["aicc"]), "lb_pvalue": lb_p,
            "recorded_lb_pvalue": float(sel["lb_pvalue"])}


# -------------------------------------------------------------- figures
def fig1_case_series(y, clusters, spec, outdir, dpi, written):
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    shade(ax, TEST_START, SAMPLE_END, "#f2f2f2", zorder=0)
    win = padded_windows(clusters, spec["b_pad"], spec["f_pad"])
    for _, r in win.iterrows():
        ax.axvspan(r.win_start, r.win_end + pd.Timedelta(days=1), ymin=0.92,
                   ymax=1.0, color=HOLIDAY_COLOUR[r.group], lw=0, zorder=1)
    ax.plot(y.index, y.values, label="daily count", zorder=2, **ACTUAL)
    ax.plot(y.index, y.rolling(7, center=True).mean().values, color="#08306b",
            lw=1.3, label="7-day centred mean", zorder=3)
    ax.axvline(TEST_START, color="#525252", lw=0.8, ls="--", zorder=4)
    top = float(y.max()) * 1.12
    ax.set_ylim(0, top)
    ax.text(SPINE_START + pd.Timedelta(days=10), top * 0.86,
            "training".format(N_TRAIN), fontsize=5, color="#525252")
    ax.text(TEST_START + pd.Timedelta(days=8), top * 0.86,
            "test".format(N_TOTAL - N_TRAIN), fontsize=5,
            color="#525252")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(SPINE_START, SAMPLE_END + pd.Timedelta(days=1))
    ax.set_ylabel("cases per day")
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color=HOLIDAY_COLOUR["H_NY"]),
                Patch(color=HOLIDAY_COLOUR["H_OT"])]
    labels += ["New Year window", "short holiday window"]
    ax.legend(handles, labels, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    save(fig, outdir, "fig1_case_series.png", dpi, written)



def fig2_mae_excess(acc, outdir, dpi, written):
    piv = acc.pivot(index="horizon", columns="model", values="MAE")
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m in ("M2", "M3", "M4", "M5"):
        exc = 100.0 * (piv[m] - piv["M1"]) / piv["M1"]
        ax.plot(piv.index, exc.values, marker="o", markersize=3.5, label=m,
                **STYLE[m])
    ax.axhline(0, color="#08306b", lw=1.2)
    ax.text(28.4, 0.4, "M1", color="#08306b", fontsize=7.5, va="bottom")
    ax.set_xticks(list(piv.index))
    ax.set_xlabel("horizon h (days)")
    ax.set_ylabel("MAE above M1 (%)")
    ax.legend(loc="upper left")
    save(fig, outdir, "fig2_mae_excess_by_horizon.png", dpi, written)


def fig3_forecasts(fc, clusters, spec, outdir, dpi, written):
    for h in HORIZONS:
        piv = fc[fc["horizon"] == h].pivot_table(index="target", columns="model",
                                                 values="forecast")
        require(len(piv) == N_ORIGINS and list(piv.columns) == list(STYLE),
                "unexpected forecast table at h = {}".format(h))
    act = fc.drop_duplicates("target").set_index("target")["actual"].sort_index()
    m0 = fc[fc["model"] == "M0"].drop_duplicates("target").set_index(
        "target")["forecast"].sort_index()
    ny = padded_windows(clusters, spec["b_pad"], spec["f_pad"])
    ny = ny[(ny["group"] == "H_NY") & (ny["start"] >= TEST_START)].iloc[0]

    fig = plt.figure(figsize=(6.3, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.42,
                          wspace=0.08)
    a = fig.add_subplot(gs[0, :])
    shade(a, ny.win_start, ny.win_end, "#fee6ce", zorder=0)
    a.plot(act.index, act.values, label="actual", **ACTUAL)
    a.plot(m0.index, m0.values, label="M0 (same at every h)", **STYLE["M0"])
    for h in HORIZONS:
        d = fc[(fc["horizon"] == h) & (fc["model"] == "M1")].set_index(
            "target")["forecast"].sort_index()
        a.plot(d.index, d.values, label="M1, h = {}".format(h), **M1_SHADES[h])
    a.set_title("(a) test window: seasonal naive M0 and baseline M1 at each "
                "horizon", loc="left")
    a.xaxis.set_major_locator(mdates.MonthLocator())
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a.set_ylabel("cases per day")
    a.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1),
             handlelength=2.2, columnspacing=1.0)

    lo = ny.win_start - pd.Timedelta(days=14)
    hi = ny.win_end + pd.Timedelta(days=18)

    axes = []
    for j, h in enumerate(HORIZONS):
        ax = fig.add_subplot(gs[1, j], sharey=axes[0] if axes else None)
        axes.append(ax)
        # Earliest target actually available at this horizon
        first_target = (
            fc.loc[fc["horizon"] == h, "target"]
            .min()
        )
        # Desired contextual window, but never before forecasts exist
        desired_lo = ny.win_start - pd.Timedelta(days=14)
        lo_h = max(desired_lo, first_target)
        piv = (
            fc[fc["horizon"] == h]
            .pivot_table(index="target", columns="model", values="forecast")
            .loc[lo_h:hi]
        )
        shade(ax, ny.win_start, ny.win_end, "#fee6ce", zorder=0)
        actual_h = act.loc[lo_h:hi]
        ax.plot(
            actual_h.index,
            actual_h.values,
            color="#252525",
            lw=0.7,
            marker="o",
            markersize=1.6,
            label="actual"
        )
        for m in ("M1", "M2", "M3", "M4", "M5"):
            ax.plot(
                piv.index,
                piv[m].values,
                label=m,
                **STYLE[m]
            )
        ax.set_title(
            "({}) New Year window, h = {}".format("bcd"[j], h),
            loc="left",
            fontsize=6
        )
        ax.xaxis.set_major_locator(
            mdates.DayLocator(bymonthday=[1, 15])
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %b")
        )
        ax.set_xlim(lo_h, hi)
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel("cases per day")
    axes[1].legend(
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14)
    )
    save(fig, outdir, "fig3_test_window_forecasts.png", dpi, written)




def fig4_weekday(fc, outdir, dpi, written):
    m1 = fc[fc["model"] == "M1"].copy()
    m1["ae"] = (m1["forecast"] - m1["actual"]).abs()
    m1["weekday"] = m1["target"].dt.day_name()
    tab = m1.pivot_table(index="weekday", columns="horizon", values="ae",
                         aggfunc="mean").reindex(WEEKDAYS)
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    width, shades = 0.26, ["#9ecae1", "#4292c6", "#08306b"]
    x = np.arange(len(WEEKDAYS))
    for i, h in enumerate(tab.columns):
        ax.bar(x + (i - 1) * width, tab[h].values, width, color=shades[i],
               label="h = {}".format(h))
    ax.set_xticks(x)
    ax.set_xticklabels([w[:3] for w in WEEKDAYS])
    ax.set_ylabel("M1 MAE (cases)")
    ax.set_xlabel("weekday of the forecast target")
    ax.legend(ncol=3, loc="upper right")
    save(fig, outdir, "fig4_weekday_mae_m1.png", dpi, written)


# --------------------------------------------------------------- tables
def table_d5_candidates(results, tdir, written):
    w50 = pd.read_csv(results / "d12_d5" / "d5_candidate_winners.csv")
    w500 = pd.read_csv(results / "d5_log1p_maxiter500" / "d5_candidate_winners.csv")
    def orders(r):
        return "({},1,{})({},0,{})[7]".format(int(r.p), int(r.q), int(r.P), int(r.Q))
    rows = []
    for cand in w500["candidate"]:
        a = w50[w50["candidate"] == cand].iloc[0]
        b = w500[w500["candidate"] == cand].iloc[0]
        rows.append({"annual_form": cand,
                     "orders_maxiter50": orders(a), "aicc_maxiter50": round(a.aicc, 3),
                     "converged_maxiter50": bool(a.converged),
                     "orders_maxiter500": orders(b), "aicc_maxiter500": round(b.aicc, 3),
                     "converged_maxiter500": bool(b.converged),
                     "ljung_box_p_maxiter500": round(b.lb_pvalue, 3)})
    out = pd.DataFrame(rows)
    out.to_csv(tdir / "tableC_d5_log_candidates_50_vs_500.csv", index=False)
    written.append("tables/tableC_d5_log_candidates_50_vs_500.csv")


def table_tests(results, tdir, written):
    t = pd.read_csv(results / "e_v18" / "e_tests.csv")
    cols = ["horizon", "hypothesis", "comparison", "test", "n", "hac_truncation",
            "hln_factor", "mean_differential", "hac_long_run_variance",
            "variance_of_mean", "statistic_raw", "statistic_hln", "df",
            "pvalue_one_sided", "pvalue_holm"]
    t = t[cols].sort_values(["horizon", "hypothesis"])
    t.to_csv(tdir / "tableC_test_calculation.csv", index=False,
             float_format="%.6g")
    written.append("tables/tableC_test_calculation.csv")


def table_reproducibility(pre, tdir, written):
    rows = []
    try:
        out = subprocess.run(
            ["git", "tag", "-l", "protocol-v*", "--sort=creatordate",
             "--format=%(refname:short)|%(*objectname:short)%(objectname:short)"
             "|%(creatordate:short)"], capture_output=True, text=True,
            check=True).stdout.strip().splitlines()
        for line in out:
            tag, sha, date = line.split("|")
            rows.append({"item": "git tag " + tag,
                         "value": "commit {} ({})".format(sha[:7], date)})
    except Exception:
        rows.append({"item": "git tags", "value": "not available here"})
    for k, v in pre["hashes"].items():
        rows.append({"item": "SHA-256 " + k, "value": v})
    for k, v in pre["environment"].items():
        rows.append({"item": "version " + k, "value": v})
    rows.append({"item": "Section E run (UTC)", "value": pre["run_utc"]})
    pd.DataFrame(rows).to_csv(tdir / "tableC_reproducibility.csv", index=False)
    written.append("tables/tableC_reproducibility.csv")



# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--daily", required=True,
                    help="full aligned daily file used by Section E")
    ap.add_argument("--date-col", default="obs_date")
    ap.add_argument("--y-col", default="cases")
    ap.add_argument("--results", default="results")
    ap.add_argument("--history", default="data/indicator_history.csv")
    ap.add_argument("--clusters", default="data/holiday_clusters.csv")
    ap.add_argument("--outdir", default="exhibits")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    results, outdir = Path(args.results), Path(args.outdir)
    require(outdir.resolve() != results.resolve()
            and results.resolve() not in outdir.resolve().parents,
            "--outdir must not be inside the frozen results directory")
    tdir = outdir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    pre = load_json(results / "e_v18" / "e_preflight.json")
    require(pre.get("protocol_tag") == "protocol-v1.8",
            "the Section E artifact is not the protocol-v1.8 run")
    spec = pre["operative_spec"]
    d5 = load_json(results / "d5_log1p_maxiter500" / "d5_selection.json")
    d7 = load_json(results / "d7_v17" / "d7_pad_selection.json")
    d10 = load_json(results / "d10_v17" / "d10_m5_specification.json")

    hashes = {}
    for key, path in (("full_data", Path(args.daily)),
                      ("history", Path(args.history)),
                      ("clusters", Path(args.clusters))):
        require(path.is_file(), "missing input {}".format(path))
        hashes[key] = sha256(path)
        require(hashes[key] == pre["hashes"][key],
                "{} is not the file Section E used (sha256 {}... vs {}...)"
                .format(path, hashes[key][:12], pre["hashes"][key][:12]))

    y = load_daily(args)
    clusters = load_clusters(Path(args.clusters))
    hist = pd.read_csv(args.history, parse_dates=["obs_date"]).set_index(
        "obs_date").sort_index()
    fc = pd.read_csv(results / "e_v18" / "e_forecasts.csv",
                     parse_dates=["origin", "target"])
    acc = pd.read_csv(results / "e_v18" / "e_accuracy.csv")

    y_train = y.loc[:TRAIN_END]
    require(len(y_train) == N_TRAIN, "training segment is not 884 days")
    idx = y_train.index
    m1 = m1_diagnostics(y_train, design(idx, d5, spec, clusters, hist, "M1"), d5)

    def grid_aicc(folder, row, lag):
        g = pd.read_csv(results / folder / "{}_lag_grid.csv".format(row))
        return float(g.loc[g["lag"] == int(lag), "aicc"].iloc[0])

    recorded = {"M2": float(d7["selected_pad"]["aicc"]),
                "M3": grid_aicc("d8_v17", "d8", spec["fx_lag"]),
                "M4": grid_aicc("d9_v17", "d9", spec["search_lag"]),
                "M5": float(d10["m5_training_fit"]["aicc"])}
    fits = {}
    for m in ("M2", "M3", "M4", "M5"):
        X = design(idx, d5, spec, clusters, hist, m)
        require(list(X.columns) == list(spec["model_columns"][m]),
                "{} columns differ from the operative specification".format(m))
        fits[m] = (refit(y_train, X, d5, recorded[m], m), list(X.columns))
        print("{} refit reproduces the recorded AICc ({:.4f})".format(
            m, recorded[m]))

    written: list[str] = []
    fig1_case_series(y, clusters, spec, outdir, args.dpi, written)
    fig2_mae_excess(acc, outdir, args.dpi, written)
    fig3_forecasts(fc, clusters, spec, outdir, args.dpi, written)
    fig4_weekday(fc, outdir, args.dpi, written)
    table_d5_candidates(results, tdir, written)
    table_tests(results, tdir, written)
    table_reproducibility(pre, tdir, written)

    manifest = {
        "purpose": "post hoc presentation; no reported result re-estimated",
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "inputs_sha256": hashes,
        "refit_checks": dict({"M1": {"aicc": m1["aicc"],
                                     "recorded": m1["recorded_aicc"],
                                     "ljung_box_p": m1["lb_pvalue"]}},
                             **{m: {"aicc": float(fits[m][0].aicc),
                                    "recorded": recorded[m]} for m in fits}),
        "outputs": written,
    }
    with open(outdir / "exhibits_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote {}".format(outdir / "exhibits_manifest.json"))


if __name__ == "__main__":
    main()
