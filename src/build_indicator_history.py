"""Build indicator_history.csv -- the lead-in for the D8/D9 lagged rows.

Why this file exists
--------------------
D8 allows FX lags of 28-90 days and D9 search lags of 28-119 days. A
training observation on the first spine day (1 July 2023) therefore
needs indicator values back to 2023-03-04. aligned_train.csv and
aligned_daily.csv both begin at the spine start, so those values exist
in the raw sources but are not reachable from the aligned files. C3-2
already required the download range to begin no later than sample start
minus the longest candidate lag, so nothing new is collected here.

What this script does not touch
-------------------------------
align.py is NOT modified. aligned_train.csv and aligned_daily.csv are
NOT written or rebuilt: their SHA-256 is recorded in every completed row
artifact (D5, D11, D12/D2, D12/D3, D7) and enforced by those scripts'
provenance guards, and A7 freezes the training sample at exactly 884
observations from 1 July 2023.

How the extended index is obtained
----------------------------------
align.load_fx and align.load_search build their output on whatever
align._spine() returns. Rather than duplicating the C2-3 rescaling,
C2-4 carry-forward and C3-3 step-expansion here -- where they could
drift out of step with the aligned data -- this script temporarily
rebinds align._spine to the extended index, calls align's own loaders,
and restores it in a finally block. Nothing is written back to align.py.

If align.py's loaders change, this file follows automatically, 
which is intended. If they are ever refactored so they no longer 
route through _spine(), this script would produce a spine-length 
history instead of an extended one -- the row-count assertion below 
catches that rather than letting it pass silently.

Output
------
indicator_history.csv: obs_date, fx_rub_per_thb, search_index, from
SPINE_START - 119 through the spine end. No case counts -- a training
observation needs a LAGGED indicator value, never a pre-spine case count
-- so both columns come from public sources (Bank of Russia, Wordstat)
and this file may live in the project repository.

Usage
-----
    python build_indicator_history.py
    python build_indicator_history.py --out ..\\data\\indicator_history.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

try:
    import align
except ImportError:
    sys.exit("align.py must be importable (same directory or on "
             "PYTHONPATH): this script reuses its frozen loaders "
             "unmodified.")

# Longest candidate lag anywhere in the protocol: D9 reaches 119 days,
# D8 reaches 90.
LEAD_DAYS = 119
MAX_LAG_FX = 90
MAX_LAG_SEARCH = 119

# Recorded in every completed row artifact.
FROZEN_TRAIN_SHA256 = ("4224fe90606f4a77e71dafadb99538a8"
                       "ac878555a32805a48219f27cf7809313")

FX_WORKBOOK = "RC_F01_01_2023_T31_07_2026.xlsx"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lead_spine() -> pd.DatetimeIndex:
    start = pd.Timestamp(align.SPINE_START) - pd.Timedelta(LEAD_DAYS, "D")
    return pd.date_range(start, align.SPINE_END, freq="D", name="obs_date")


def build_history(raw: Path) -> pd.DataFrame:
    """Call align's own loaders against the extended index."""
    idx = lead_spine()
    original_spine = align._spine
    try:
        align._spine = lead_spine          # temporary, restored below
        fx = align.load_fx(raw / FX_WORKBOOK)
        search = align.load_search(raw)
    finally:
        align._spine = original_spine
    return pd.DataFrame({"fx_rub_per_thb": fx, "search_index": search},
                        index=idx)


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="build the D8/D9 indicator lead-in")
    ap.add_argument("--out", default=None,
                    help="output path (default: alongside the aligned "
                         "files, as indicator_history.csv)")
    ap.add_argument("--raw", default=None,
                    help="raw source directory (default: align.DATA/raw)")
    args = ap.parse_args()

    raw = Path(args.raw) if args.raw else align.DATA / "raw"
    out = (
        Path(args.out)
        if args.out
        else Path(__file__).resolve().parents[1] / "data" / "indicator_history.csv"
    )
    train_path = align.PRIVATE / "aligned_train.csv"
    # Read-only: confirm the frozen training file is still the one the
    # completed rows ran against. Nothing is written to it.
    if train_path.exists():
        digest = sha256_of(train_path)
        if digest == FROZEN_TRAIN_SHA256:
            print("aligned_train.csv sha256 {}... unchanged".format(
                digest[:16]))
        else:
            sys.exit(
                "aligned_train.csv hashes to {}... but the frozen training "
                "extraction is {}... . Resolve this before building the "
                "D8/D9 indicator history.".format(
                    digest[:16],
                    FROZEN_TRAIN_SHA256[:16],
                )
            )
    else:
        print("note: {} not found; skipping the hash check".format(
            train_path))

    h = build_history(raw)

    assert h.index.is_unique and h.index.is_monotonic_increasing
    assert h.notna().all().all(), "NaNs in history:\n{}".format(h.isna().sum())
    assert (h["fx_rub_per_thb"] > 0).all(), "non-positive FX value"
    assert (h["search_index"] >= 0).all(), "negative search index"
    expected_rows = align.N_DAYS + LEAD_DAYS
    assert len(h) == expected_rows, (
        "expected {} rows, got {}. If align.py's loaders no longer build "
        "their output on _spine(), the lead-in was not applied."
        .format(expected_rows, len(h)))

    # C2-3 sanity: the published quote is RUB per 10 THB and must have
    # been rescaled to RUB per 1 THB. Losing that division is exactly a
    # factor of ten, which this catches.
    med = float(h["fx_rub_per_thb"].median())
    if not (0.5 <= med <= 10.0):
        sys.exit("FX median is {:.4f} RUB per baht, outside the plausible "
                 "range for C2-3 (published RUB-per-10-THB quote rescaled "
                 "to RUB per 1 THB).".format(med))

    out.parent.mkdir(parents=True, exist_ok=True)
    h.to_csv(out)
    history_digest = sha256_of(out)
    print("history: {} rows, {} -> {}".format(
        len(h), h.index.min().date(), h.index.max().date()))
    print("written to {}".format(out.resolve()))
    print("sha256: {}".format(history_digest))
    print("columns: {} (no case counts: public sources only)".format(
        ", ".join(h.columns)))
    print("FX median {:.4f} RUB per baht (C2-3 rescaling intact)".format(med))

    first_train = pd.Timestamp(align.SPINE_START)
    print("")
    print("coverage from the first training day ({}):".format(
        first_train.date()))
    for label, lag in (("D8 min", 28), ("D8 max", MAX_LAG_FX),
                       ("D9 max", MAX_LAG_SEARCH)):
        t = first_train - pd.Timedelta(lag, "D")
        print("  {} lag {:3d} -> {}  {}".format(
            label, lag, t.date(),
            "present" if t in h.index else "MISSING"))


if __name__ == "__main__":
    main()
