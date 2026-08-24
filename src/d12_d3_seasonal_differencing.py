"""D3 re-run on log(y + 1) -- seasonal differencing order D.

Second step of the D12 branch. The 19 August 2026 addendum requires
D2, D3 and D5 to be re-run on log(y + 1); the 23 August clarification
makes D4 explicit between D3 and D5.

Frozen D3 rule:
    OCSB test at s = 7 (Osborn et al., 1988), as implemented in
    pmdarima.

    D = 1 if the test does NOT reject.
    D = 0 if the test rejects.

The annual cycle is deterministic by design and is never seasonally
differenced; D applies only to the weekly period s = 7.

The OCSB test is applied directly to the working response series,
log(y + 1), rather than to D2 pilot residuals. This is the same
implementation choice used for the original count-scale D3.

This script:
    * refuses to run unless the authorised D12/D2 artifact exists;
    * verifies D2 was executed on log(y + 1);
    * verifies the exact training-data SHA-256 is unchanged;
    * validates raw counts before transforming;
    * verifies the working series equals log1p(raw counts);
    * executes pmdarima's OCSB decision;
    * independently reconstructs the statistic/critical-value decision
      and aborts if it disagrees with pmdarima's returned D;
    * records the D3 result and the mechanically implied D4 outcome.

Outputs (--outdir, default results/d12_d3):
    d12_d3_seasonal_differencing.json
    d12_d3_report.md

No proprietary residual-level artifact is written by this step.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pmdarima.arima import OCSBTest

try:
    import d5_baseline_order_selection as d5
except ImportError:
    sys.exit(
        "d5_baseline_order_selection.py must be importable (same "
        "directory or on PYTHONPATH): this script reuses its loader "
        "and D1 environment guard."
    )


#------------------------Frozen / operative constants----------------------

ROW = "D3"
BRANCH = "D12"

S = 7

# Implementation choice already recorded for the original D3.
OCSB_LAG_METHOD = "aic"


def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="D12/D3: seasonal differencing order D on log(y+1)",
    )

    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)

    ap.add_argument(
        "--d2",
        required=True,
        help=(
            "path to d12_d2_differencing.json; D3 refuses to run unless "
            "this records an authorised transformed-scale D2 execution"
        ),
    )

    ap.add_argument("--outdir", default="results/d12_d3")
    ap.add_argument("--allow-env-mismatch", action="store_true")

    args = ap.parse_args()

    # Required by the shared D5 loader: validate raw counts first,
    # transform only afterwards.
    args.scale = "log1p"
    args.d = None
    args.D_seasonal = None

#------------------------D1 environment----------------------------

    env_diffs = d5.check_environment(args.allow_env_mismatch)

#----------------Authorisation from transformed D2---------------------

    d2_path = Path(args.d2)
    d2_record = json.loads(d2_path.read_text())
    
    if d2_record.get("protocol_tag") != d5.PROTOCOL_TAG:
        sys.exit(
            "protocol-state mismatch: transformed D2 records {}, "
            "but this D3 implementation expects {}.".format(
                d2_record.get("protocol_tag"),
                d5.PROTOCOL_TAG,
            )
        )

    if d2_record.get("row") != "D2":
        sys.exit(
            "the supplied prerequisite artifact is not a D2 record."
        )

    if d2_record.get("branch") != "D12":
        sys.exit(
            "the supplied D2 record is not from the D12 branch."
        )

    if d2_record.get("scale") != "log1p":
        sys.exit(
            "the supplied D2 record was not produced on log(y+1). "
            "Transformed-scale D3 is not authorised."
        )

    d_selected = int(d2_record["decision"]["d"])
    if d_selected not in (0, 1):
        sys.exit(
            "the supplied D2 record contains inadmissible d = {}."
            .format(d_selected)
        )

    if not d2_record.get("authorised_by", {}).get("triggered"):
        sys.exit(
            "the transformed D2 record does not show an authorised "
            "D11 trigger. D12/D3 must not proceed."
        )

    if d2_record.get("smoke_mode"):
        print(
            "WARNING: the D2 record inherits SMOKE MODE; this D3 run "
            "is not a protocol execution."
        )
#-----------------------Same physical training extraction-------------------------

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    digest = d5.sha256_of(data_path)

    recorded_hash = d2_record["data"]["sha256"]

    if digest != recorded_hash:
        sys.exit(
            "input SHA-256 {} does not match the file transformed D2 "
            "ran on ({}). D3 must use the identical training extraction."
            .format(digest[:16], recorded_hash[:16])
        )

#-----------------Raw validation + log1p--------------------------------------

    s, y, y_raw = d5.load_training_series(args)

    print(
        "loaded {} training days through {}; sha256 = {}".format(
            len(y), d5.TRAIN_END, digest[:16]
        )
    )

    print(
        "scale in force: log(y + 1)  "
        "[D12 branch; transformed D2 selected d = {}]".format(
            d_selected
        )
    )

    if not np.allclose(
        y.to_numpy(dtype=float),
        np.log1p(y_raw.to_numpy(dtype=float)),
    ):
        sys.exit(
            "the working series is not log1p of the raw counts; "
            "the D12 transform was not applied as expected."
        )
#---------------Frozen D3 OCSB test--------------------------

    test = OCSBTest(
        m=S,
        lag_method=OCSB_LAG_METHOD,
    )

    # pmdarima's implementation provides the protocol decision directly.
    D_selected = int(
        test.estimate_seasonal_differencing_term(
            y.to_numpy(dtype=float)
        )
    )

    # Recover statistic and critical value from the SAME OCSB
    # implementation for the audit record.
    stat = float(
        test._compute_test_statistic(
            y.to_numpy(dtype=float)
        )
    )

    crit = float(
        test._calc_ocsb_crit_val(S)
    )

    # Under pmdarima's OCSB convention, statistic < critical value
    # corresponds to rejection and therefore D = 0.
    rejects = bool(stat < crit)
    D_from_stat = 0 if rejects else 1

    if D_selected != D_from_stat:
        sys.exit(
            "OCSB decision inconsistency: pmdarima returned D = {}, "
            "but statistic {:.9f} versus critical value {:.9f} implies "
            "D = {} under the frozen D3 rule. Resolve before continuing."
            .format(
                D_selected,
                stat,
                crit,
                D_from_stat,
            )
        )

    if D_selected not in (0, 1):
        sys.exit(
            "OCSB returned inadmissible seasonal differencing order D = {}."
            .format(D_selected)
        )
#-----------------------D4 is deterministic-------------------------------

    with_intercept = bool(
        d_selected == 0 and D_selected == 0
    )

#---------------------Output----------------------

    out = {
        "protocol_tag": d5.PROTOCOL_TAG,
        "protocol_freeze_tag": d5.PROTOCOL_FREEZE_TAG,
        "row": ROW,
        "branch": BRANCH,
        "scale": "log1p",

        "authorised_by": {
            "d2_record": str(d2_path.resolve()),
            "d11_triggered": bool(
                d2_record["authorised_by"]["triggered"]
            ),
            "d_selected_at_transformed_D2": d_selected,
        },

        "ocsb": {
            "implementation": "pmdarima.arima.OCSBTest",
            "s": S,
            "lag_method": OCSB_LAG_METHOD,
            "statistic": stat,
            "critical_value": crit,
            "rejects": rejects,
        },

        "decision": {
            "rule": (
                "D = 1 if OCSB at s=7 does not reject; "
                "otherwise D = 0"
            ),
            "D": D_selected,
        },

        "operative_differencing": {
            "d": d_selected,
            "D": D_selected,
            "source": (
                "d from transformed-scale D2; "
                "D from transformed-scale D3"
            ),
        },

        "d4": {
            "rule": "constant iff d = D = 0",
            "with_intercept": with_intercept,
            "source": (
                "D4 mechanically reapplied after transformed-scale D2/D3"
            ),
        },

        "next_step": (
            "D4 outcome is with_intercept = {}. Re-run D5 on log(y+1) "
            "with --scale log1p --d {} --D {}."
            .format(
                with_intercept,
                d_selected,
                D_selected,
            )
        ),

        "smoke_mode": bool(
            d2_record.get("smoke_mode", False)
        ),

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

    (outdir / "d12_d3_seasonal_differencing.json").write_text(
        json.dumps(out, indent=2)
    )

#------------------report-----------------------------------

    L = []

    L.append(
        "# D12/D3 run report -- weekly seasonal differencing on log(y + 1)"
    )
    L.append("")

    stamp = ""

    if env_diffs:
        stamp += (
            "  **NON-PROTOCOL: D1 environment not in force: "
            + "; ".join(env_diffs)
            + ".**"
        )

    if out["smoke_mode"]:
        stamp += (
            "  **Inherited SMOKE MODE from the transformed D2 record.**"
        )

    L.append(
        "Protocol state {} (freeze tag {}), row D3 re-run on the "
        "transformed scale under D12. Run (UTC): {}.{}".format(
            out["protocol_tag"],
            out["protocol_freeze_tag"],
            out["run_utc"],
            stamp,
        )
    )

    L.append("")

    L.append(
        "Authorised by transformed D2 (`{}`), which selected d = {} "
        "on log(y + 1). Input `{}`, SHA-256 `{}`, matching the "
        "training extraction used by D2.".format(
            out["authorised_by"]["d2_record"],
            d_selected,
            out["data"]["path"],
            digest,
        )
    )

    L.append("")
    L.append("## OCSB test")
    L.append("")

    L.append(
        "The frozen D3 rule applies the OCSB test at the weekly period "
        "s = {} using pmdarima. The test is applied directly to "
        "log(y + 1); annual seasonality remains deterministic and is "
        "never differenced.".format(S)
    )

    L.append("")

    L.append("| statistic | critical value | rejects | selected D |")
    L.append("|---|---|---|---|")
    L.append(
        "| {:.9f} | {:.9f} | {} | {} |".format(
            stat,
            crit,
            "yes" if rejects else "no",
            D_selected,
        )
    )

    L.append("")
    L.append(
        "Decision rule: **D = 1 if OCSB does not reject; "
        "otherwise D = 0.**"
    )

    L.append("")
    L.append("## D4 implication")
    L.append("")

    L.append(
        "Transformed-scale D2 selected d = {} and D3 selected D = {}. "
        "Under D4, a constant is included iff d = D = 0; therefore "
        "**with_intercept = {}**.".format(
            d_selected,
            D_selected,
            with_intercept,
        )
    )

    L.append("")
    L.append("## Implementation choice")
    L.append("")

    L.append(
        "The OCSB lag-selection method is `{}`. This was not fixed by "
        "the frozen D3 row and is retained from the original D3 "
        "implementation rather than changed after observing the "
        "transformed-scale series.".format(
            OCSB_LAG_METHOD
        )
    )

    L.append("")
    L.append("## Next step")
    L.append("")
    L.append(out["next_step"])

    L.append("")
    L.append("## Artifacts")
    L.append("")

    L.append(
        "d12_d3_seasonal_differencing.json "
        "(machine-readable transformed D3 decision), this report. "
        "No residual-level proprietary artifact is produced by D3."
    )

    (outdir / "d12_d3_report.md").write_text(
        "\n".join(L) + "\n"
    )

#---------------------------Console---------------------------

    print("")
    print(
        "OCSB stat {:.9f} vs critical {:.9f} -> {}".format(
            stat,
            crit,
            "REJECTS" if rejects else "does not reject",
        )
    )

    print("")
    print(
        "D3 DECISION: D = {}".format(D_selected)
    )

    print(
        "D4 implication: d = {}, D = {} -> constant = {}".format(
            d_selected,
            D_selected,
            with_intercept,
        )
    )

    print("")
    print(
        "next: D5 --scale log1p --d {} --D {}".format(
            d_selected,
            D_selected,
        )
    )

    print(
        "outputs written to {}".format(
            outdir.resolve()
        )
    )


if __name__ == "__main__":
    main()