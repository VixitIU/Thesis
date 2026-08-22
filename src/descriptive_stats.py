"""Descriptive statistics and integrity checks reported in Section 4.1.

Every figure quoted in that section is produced here. Training-period
statistics use observations up to and including 2025-11-30; the test window
is used only for the row-count check.
"""
import warnings
import pandas as pd
import align
import re


SPLIT = pd.Timestamp("2025-11-30")
WORDSTAT_RETRIEVED = pd.Timestamp("2026-08-20")
WORDSTAT_LATEST_WEEK = pd.Timestamp("2026-08-10")
MIN_LAG = 28


def load_full() -> pd.DataFrame:
    """Full sample. Used only for integrity checks."""
    df = pd.read_csv(
        align.PRIVATE / "aligned_daily.csv",
        parse_dates=["obs_date"]
    )
    return df.set_index("obs_date")



def integrity(df: pd.DataFrame) -> dict:
    """Full-sample checks: spine completeness, zero counts, split sizes."""
    spine = align._spine()
    tr = df.loc[df.index <= align.TRAIN_END]
    te = df.loc[df.index > align.TRAIN_END]
    return {
        "n_obs": len(df),
        "spine_matches": df.index.equals(spine),
        "n_missing_dates": len(spine.difference(df.index)),
        "n_duplicate_dates": int(df.index.duplicated().sum()),
        "n_train": len(tr),
        "n_test": len(te),
        "n_zero_days": int((df["cases"] == 0).sum())
    }


def fx_fill_structure() -> dict:
    """Quoted vs carried-forward days on the sample spine, and run lengths."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fx = pd.read_excel(align.DATA / "raw" / "RC_F01_01_2023_T31_07_2026.xlsx")
    fx["data"] = pd.to_datetime(fx["data"])
    quoted = fx.set_index("data")["curs"].reindex(align._spine())

    missing = quoted[quoted.isna()].index
    runs, run = [], [missing[0]]
    for d in missing[1:]:
        if (d - run[-1]).days == 1:
            run.append(d)
        else:
            runs.append(run)
            run = [d]
    runs.append(run)
    lengths = pd.Series([len(r) for r in runs])

    return {
        "n_quoted": int(quoted.notna().sum()),
        "n_filled": int(quoted.isna().sum()),
        "n_runs": len(runs),
        "n_two_day_runs": int((lengths == 2).sum()),
        "max_run_days": int(lengths.max()),
        "nominal_uniform_10": bool((fx["nominal"] == 10).all()),
        "cdx_uniform_baht": bool((fx["cdx"] == "Baht").all()),
    }


def search_structure() -> dict:
    """Week count, common index and gap check across the four frozen queries."""
    weekly = {}
    for f in sorted((align.DATA / "raw").glob("wordstat_dynamic_*.csv")):
        with f.open(encoding="utf-8-sig", newline="") as fh:
            raw = fh.read()
        lines = [l for l in raw.split("\r") if l.strip()]
        query = re.search(r"«(.+?)»", lines[0].split(";")[3]).group(1)
        recs = [(pd.to_datetime(p[0], format="%d.%m.%Y"), int(p[1].replace(" ", "")))
                for p in (l.split(";") for l in lines[1:])]
        weekly[query] = pd.Series(dict(recs)).sort_index()

    W = pd.DataFrame(weekly)
    gaps = W.index.to_series().diff().dropna().dt.days
    week_end = WORDSTAT_LATEST_WEEK + pd.Timedelta(6, unit="D")
    delay_from_week_end = (WORDSTAT_RETRIEVED - week_end).days


    return {
        "n_queries": W.shape[1],
        "n_weeks": W.shape[0],
        "common_index_complete": bool(W.notna().all().all()),
        "all_gaps_seven_days": bool((gaps == 7).all()),
        "first_week": W.index.min().date().isoformat(),
        "last_week": W.index.max().date().isoformat(),
        "delay_from_week_end_days": delay_from_week_end,
        "delay_for_earliest_day_in_week_days": 6 + delay_from_week_end,
        "worst_case_margin_days": (MIN_LAG - 6) - delay_from_week_end,
    }


def cluster_structure() -> dict:
    cl = pd.read_csv(align.DATA / "holiday_clusters.csv", parse_dates=["start", "end"])
    return {
        "n_clusters": len(cl),
        "n_H_NY": int((cl["group"] == "H_NY").sum()),
        "n_H_OT": int((cl["group"] == "H_OT").sum()),
        "by_group_split": cl.groupby(["group", "split"]).size().to_dict(),
    }


def table4(tr: pd.DataFrame) -> pd.DataFrame:
    """Training-period distribution of the three continuous series."""
    stats = tr.describe(percentiles=[0.25, 0.5, 0.75]).T
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats.columns = ["Mean", "SD", "Min", "P25", "Median", "P75", "Max"]
    return stats


def case_shape(tr: pd.DataFrame) -> dict:
    c = tr["cases"]
    return {
        "skew": round(c.skew(), 2),
        "excess_kurtosis": round(c.kurt(), 2),
        "var_mean_ratio": round(c.var() / c.mean(), 2),
        "n_days_under_5": int((c < 5).sum()),
        "n_days_under_10": int((c < 10).sum()),
        "p01": float(c.quantile(0.01)),
    }


def seasonal_shape(tr: pd.DataFrame) -> tuple:
    m = tr["cases"].resample("MS").mean().round(1)
    m.index = m.index.strftime("%b %Y")
    dow = tr.groupby(tr.index.dayofweek)["cases"].mean().round(2)
    dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return m, dow


def report() -> None:
    full = load_full()
    train = align.load_train()

    for name, block in [
        ("Integrity (full sample)", integrity(full)),
        ("Exchange rate", fx_fill_structure()),
        ("Search interest", search_structure()),
        ("Holiday clusters", cluster_structure()),
        ("Case distribution (training)", case_shape(train))
    ]:
        print(f"\n--- {name} ---")
        for k, v in block.items():
            print(f"  {k}: {v}")

    print("\n--- Table 4: training-period descriptives ---")
    print(table4(train).round(3).to_string())

    m, dow = seasonal_shape(train)

    print("\n--- Day-of-week means (training) ---")
    print(dow.to_string())
    print(
        f"  spread as share of mean: "
        f"{(dow.max() - dow.min()) / train['cases'].mean():.1%}"
    )

    print("\n--- Monthly means (training) ---")
    print(m.to_string())

    print("\n--- Spearman correlations (training, contemporaneous) ---")
    print(train.corr(method="spearman").round(3).to_string())

if __name__ == "__main__":
    report()