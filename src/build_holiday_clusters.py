"""Build the frozen holiday cluster set and, at model time, padded regressors.

Protocol rows: C1-2 (source), C1-4 (parse), C1-5 (no libraries),
C1-6 (inclusion), C1-7 (generated set), C1-8 (grouping),
C1-9 (shared pad), C1-10 (union, binary).

Inputs and output are public calendar data only; nothing here is proprietary.
"""
from pathlib import Path
import json
import re
import pandas as pd

try:
    PROJECT = Path(__file__).resolve().parents[1]          
except NameError:                                          
    PROJECT = Path(r"C:\Users\New\Documents\IU work\Thesis\project")

RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "data"

YEARS = [2023, 2024, 2025, 2026]
SPLIT = pd.Timestamp("2025-12-01")
SAMPLE_START, SAMPLE_END = pd.Timestamp("2023-07-01"), pd.Timestamp("2026-07-31")
# Source: https://xmlcalendar.ru/data/ru/{year}/calendar.json

TOKEN = re.compile(r"^(\d{1,2})([*+]?)$")


def non_working_days(rawdir: Path = RAW) -> pd.Series:
    """C1-4: non-working iff the day appears in the month 'days' string
    without '*'. '*' marks a shortened working day; '+' marks a day off
    arising from a transfer, hence non-working. A weekend absent from the
    string is a full working day (reverse transfer).
    """
    flags = {}
    for year in YEARS:
        path = rawdir / f"calendar_{year}.json"
        cal = json.loads(path.read_text(encoding="utf-8"))
        listed = {}
        for m in cal["months"]:
            for token in str(m["days"]).split(","):
                token = token.strip()
                if not token:
                    continue
                match = TOKEN.match(token)
                assert match, f"C1-4: unrecognised token {token!r} in {year}-{m['month']}"
                day, mark = int(match.group(1)), match.group(2)
                listed[pd.Timestamp(year, int(m["month"]), day)] = (mark != "*")
        for d in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
            flags[d] = listed.get(d, False)
    return pd.Series(flags).sort_index()


def build_clusters(rawdir: Path = RAW) -> pd.DataFrame:
    """C1-6: every uninterrupted run of >= 3 consecutive non-working days.
    No name-based selection; no merging across working days.
    """
    nw = non_working_days(rawdir)
    runs, start, prev = [], None, None
    for day, is_nw in nw.items():
        if is_nw and start is None:
            start = day
        elif not is_nw and start is not None:
            runs.append((start, prev))
            start = None
        prev = day
    if start is not None:
        runs.append((start, prev))

    rows = []
    for s, e in runs:
        if (e - s).days + 1 < 3:
            continue
        is_ny = any(d.month == 1 and d.day == 1 for d in pd.date_range(s, e))
        rows.append({"start": s, "end": e, "length": (e - s).days + 1,
                     "group": "H_NY" if is_ny else "H_OT"})

    cl = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    # runs detected on full calendar years; retain those inside the sample
    cl = cl[(cl["start"] >= SAMPLE_START) & (cl["end"] <= SAMPLE_END)].reset_index(drop=True)
    cl["split"] = (cl["start"] >= SPLIT).map({False: "train", True: "test"})

    assert len(cl) == 17, f"C1-7 expects 17 clusters, got {len(cl)}"
    assert (cl["group"] == "H_NY").sum() == 3, "expected 3 New Year clusters"
    print(cl.groupby(["group", "split"]).size())  # check against C1-8 and Table 3
    return cl


def holiday_regressors(clusters: pd.DataFrame, spine: pd.DatetimeIndex,
                       b: int, f: int) -> pd.DataFrame:
    """C1-9: one (b, f) pair shared by both regressors.
    C1-10: union, binary -- overlapping padded windows are never summed.
    """
    out = pd.DataFrame(0, index=spine, columns=["H_NY", "H_OT"], dtype=int)
    for _, r in clusters.iterrows():
        window = pd.date_range(r["start"] - pd.Timedelta(days=b),
                               r["end"] + pd.Timedelta(days=f), freq="D")
        out.loc[out.index.intersection(window), r["group"]] = 1
    return out


if __name__ == "__main__":
    cl = build_clusters()
    cl.to_csv(OUT / "holiday_clusters.csv", index=False)
    print(cl.to_string())