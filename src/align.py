"""Build the aligned daily dataset from the four raw sources.

Protocol rows implemented: A (spine), C2-3, C2-4, C3-3.
Lags (D8, D9) and the holiday pad (D7) are selected at model time.
"""
from pathlib import Path
import re
import pandas as pd

DATA = Path(r"C:\Users\New\Documents\IU work\Thesis\project\data")
SPINE_START, SPINE_END = "2023-07-01", "2026-07-31"
N_DAYS = 1127

QUERIES = ["туры в Таиланд", "туры в Тайланд", "туры на Пхукет", "туры в Паттайю"]


def _spine() -> pd.DatetimeIndex:
    return pd.date_range(SPINE_START, SPINE_END, freq="D", name="obs_date")


def load_cases(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["obs_date"], format="%d/%m/%Y")
    s = pd.Series(df["total_cases"].astype(int).values, index=idx)
    s.index.name = "obs_date"
    assert s.index.is_unique, "duplicate dates in case export"
    assert len(s) == N_DAYS, f"case export has {len(s)} rows, expected {N_DAYS}"
    assert s.index.min() == pd.Timestamp(SPINE_START), f"starts {s.index.min().date()}"
    assert s.index.max() == pd.Timestamp(SPINE_END), f"ends {s.index.max().date()}"
    return s


def load_fx(path: Path) -> pd.Series:
    """C2-3: published quote is RUB per 10 THB, rescaled to RUB per 1 THB.
    C2-4: value at t is the most recent determination effective on or before t.
    """
    fx = pd.read_excel(path)
    assert (fx["nominal"] == 10).all(), "C2-2: nominal is not uniformly 10"
    assert (fx["cdx"] == "Baht").all(), "C2-2: non-Baht rows present"
    fx["data"] = pd.to_datetime(fx["data"])
    assert not fx["data"].duplicated().any(), "duplicate FX dates"
    s = fx.sort_values("data").set_index("data")["curs"] / 10.0
    return s.reindex(_spine().union(s.index)).ffill().reindex(_spine())


def load_search(dirpath: Path) -> pd.Series:
    """C3-3: four series summed, then step-expanded, constant within its week."""
    weekly = {}
    for f in sorted(dirpath.glob("wordstat_dynamic_*.csv")):
        raw = f.read_text(encoding="utf-8-sig", newline="")
        lines = [l for l in raw.split("\r") if l.strip()]
        query = re.search(r"«(.+?)»", lines[0].split(";")[3]).group(1)
        recs = [(pd.to_datetime(p[0], format="%d.%m.%Y"), int(p[1].replace(" ", "")))
                for p in (l.split(";") for l in lines[1:])]
        weekly[query] = pd.Series(dict(recs)).sort_index()

    assert set(weekly) == set(QUERIES), f"C3-1 query set mismatch: {sorted(weekly)}"
    W = pd.DataFrame(weekly)
    assert W.notna().all().all(), "weeks not common to all four series"
    assert (W.index.to_series().diff().dropna() == pd.Timedelta("7D")).all(), "week gap"

    total = W.sum(axis=1)
    daily = total.reindex(_spine().union(total.index)).ffill().reindex(_spine())
    return daily


def build() -> pd.DataFrame:
    df = pd.DataFrame(index=_spine())
    df["cases"] = load_cases(
    Path(r"C:\Users\New\Documents\IU work\Thesis\data\daily_cases.csv")
)
    df["fx_rub_per_thb"] = load_fx(DATA / "raw" / "RC_F01_01_2023_T31_07_2026.xlsx")
    df["search_index"] = load_search(DATA / "raw")

    assert len(df) == N_DAYS, f"expected {N_DAYS} rows, got {len(df)}"
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert df.notna().all().all(), f"NaNs remain:\n{df.isna().sum()}"
    assert (df["cases"] >= 0).all()
    assert (df["fx_rub_per_thb"] > 0).all()
    return df


if __name__ == "__main__":
    out = build()
    out.to_csv(r"C:\Users\New\Documents\IU work\Thesis\data\aligned_daily.csv")