"""Training-only D5 convergence sensitivity diagnostic.

Purpose
-------
This script answers one narrow post-selection question without reopening the
frozen D5 search:

    If the seven exact per-candidate winner specifications recorded by the
    operative transformed D5 run are refitted to convergence under the later
    downstream optimizer ceiling (LBFGS, maxiter=500), does their AICc ranking
    remain the same?

This is a NON-DECISIONAL sensitivity analysis.

It does NOT:
    - rerun auto_arima;
    - search new ARIMA/SARIMA orders;
    - add/remove annual-form candidates;
    - access the test window;
    - alter the frozen D5 selection;
    - alter M1-M5 or any D7-D10 decision.

The diagnostic therefore isolates optimizer convergence while holding the
seven recorded D5 candidate-winner specifications fixed.

Important scope limitation
--------------------------
Frozen D5 selected from the full pool of visited Ljung-Box-passing fits.
This diagnostic refits the seven *per-candidate stepwise winners* recorded in
d5_candidate_winners.csv. It tests stability of the annual-form winner ranking,
not a hypothetical full reranking of every visited D5 fit under maxiter=500.

Inputs
------
    --data       frozen 884-row training CSV
    --selection  operative transformed D5 d5_selection.json
    --winners    matching d5_candidate_winners.csv

Outputs
-------
    d5_convergence_sensitivity.csv
    d5_convergence_sensitivity.json
    d5_convergence_sensitivity_report.md

The run is also recorded in the existing project MLflow FileStore. Exact input
artifacts, this source script, and all diagnostic outputs are archived there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

try:
    import d5_baseline_order_selection as d5
except ImportError:
    sys.exit(
        "d5_baseline_order_selection.py must be importable (same directory "
        "or on PYTHONPATH). This diagnostic reuses D5's data loader, annual "
        "regressor builders, environment guard and Ljung-Box implementation."
    )


# ----------------------------- diagnostic constants --------------------------
DIAGNOSTIC_NAME = "D5_CONVERGENCE_SENSITIVITY"
OPERATIVE_PROTOCOL_TAG = "protocol-v1.6"
SOURCE_D5_PROTOCOL_TAG = "protocol-v1.3"

DIAGNOSTIC_METHOD = "lbfgs"
DIAGNOSTIC_MAXITER = 500

EXPECTED_SCALE = "log1p"
EXPECTED_ORIGINAL_MAXITER = 50

# Purely numerical comparison tolerance. This never changes a decision.
AICC_MATCH_ATOL = 1e-8


# --------------------------------- helpers -----------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        sys.exit("D5 convergence diagnostic check failed: " + msg)


def as_bool(v):
    if pd.isna(v):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def rank_frame(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    """Attach deterministic ordinal rank using the D5 candidate order."""
    cand_rank = {c: i for i, c in enumerate(d5.CANDIDATE_ORDER)}
    order = (
        df.assign(_cand_rank=df["candidate"].map(cand_rank))
        .sort_values(
            [value_col, "k_params_total_500", "_cand_rank"],
            kind="mergesort",
        )
        .index
        .tolist()
    )
    ranks = {idx: pos + 1 for pos, idx in enumerate(order)}
    df[prefix] = [ranks[i] for i in df.index]
    return df


def candidate_from_selection(sel: dict) -> str:
    return str(sel["selected"]["candidate"])


# ----------------------------------- main ------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Training-only fixed-specification D5 convergence sensitivity",
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--y-col", default=None)
    ap.add_argument("--selection", required=True,
                    help="operative transformed D5 d5_selection.json")
    ap.add_argument("--winners", required=True,
                    help="matching transformed D5 d5_candidate_winners.csv")
    ap.add_argument("--outdir", default="results/d5_convergence_sensitivity")
    ap.add_argument("--experiment",
                    default="medical-assistance-demand-forecasting")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()

    env_diffs = d5.check_environment(args.allow_env_mismatch)

    selection_path = Path(args.selection)
    winners_path = Path(args.winners)
    data_path = Path(args.data)

    for label, path in (
        ("selection", selection_path),
        ("winners", winners_path),
        ("training data", data_path),
    ):
        require(path.exists(), "{} file is missing: {}".format(label, path))

    selection_sha = sha256_of(selection_path)
    winners_sha = sha256_of(winners_path)
    data_sha = sha256_of(data_path)

    sel = json.loads(selection_path.read_text())
    winners = pd.read_csv(winners_path)

    # -------------------------- provenance / scope guards ---------------------
    require(sel.get("row") == "D5",
            "--selection is not a D5 artifact.")
    require(sel.get("protocol_tag") == SOURCE_D5_PROTOCOL_TAG,
            "expected D5 {}, found {}.".format(
                SOURCE_D5_PROTOCOL_TAG, sel.get("protocol_tag")))
    require(sel.get("scale") == EXPECTED_SCALE,
            "this diagnostic is for the operative transformed D5 artifact; "
            "expected scale {}, found {}.".format(
                EXPECTED_SCALE, sel.get("scale")))
    require(not sel.get("smoke_mode", False),
            "the supplied D5 artifact is a smoke run.")
    require(sel["data"]["sha256"] == data_sha,
            "training CSV SHA-256 does not match the D5 artifact.")
    require(int(sel["hk_settings"]["maxiter"]) == EXPECTED_ORIGINAL_MAXITER,
            "expected historical D5 maxiter {}, found {}.".format(
                EXPECTED_ORIGINAL_MAXITER,
                sel["hk_settings"].get("maxiter")))
    require(sel["hk_settings"].get("method") == DIAGNOSTIC_METHOD,
            "historical D5 optimizer method differs from {}.".format(
                DIAGNOSTIC_METHOD))

    required_cols = {
        "candidate", "p", "q", "P", "Q", "aicc",
        "converged", "lb_pvalue", "lb_pass",
    }
    missing = required_cols.difference(winners.columns)
    require(not missing,
            "winner CSV is missing columns: {}".format(sorted(missing)))

    require(len(winners) == len(d5.CANDIDATE_ORDER),
            "expected {} candidate winners, found {}.".format(
                len(d5.CANDIDATE_ORDER), len(winners)))
    require(set(winners["candidate"]) == set(d5.CANDIDATE_ORDER),
            "candidate set differs from frozen D5 candidate set: {}.".format(
                sorted(winners["candidate"].tolist())))

    # Verify CSV rows agree with the candidate-winner records embedded in JSON.
    json_winners = {
        str(r["candidate"]): r
        for r in sel.get("candidate_winners", [])
    }
    require(set(json_winners) == set(d5.CANDIDATE_ORDER),
            "d5_selection.json candidate_winners does not contain the frozen "
            "seven-candidate set.")

    for _, row in winners.iterrows():
        c = str(row["candidate"])
        jr = json_winners[c]
        for k in ("p", "q", "P", "Q"):
            require(int(row[k]) == int(jr[k]),
                    "{} {} differs between CSV and JSON.".format(c, k))
        require(
            abs(float(row["aicc"]) - float(jr["aicc"])) <= AICC_MATCH_ATOL,
            "{} AICc differs between CSV and JSON.".format(c),
        )
        require(as_bool(row["lb_pass"]) == bool(jr["lb_pass"]),
                "{} Ljung-Box pass flag differs between CSV and JSON.".format(c))

    # Reconstruct transformed D2/D3/D4 state exactly from the D5 artifact.
    d_sel = int(sel["differencing"]["d"])
    D_sel = int(sel["differencing"]["D"])
    expected_intercept = (d_sel == 0 and D_sel == 0)

    require(bool(sel["selected"]["with_intercept"]) == expected_intercept,
            "D5 selected intercept is inconsistent with mechanical D4.")

    d5.FIXED_d = d_sel
    d5.FIXED_D = D_sel
    d5.WITH_INTERCEPT = expected_intercept

    # d5.load_training_series only needs these fields.
    args.scale = EXPECTED_SCALE
    _, y, _ = d5.load_training_series(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("NON-DECISIONAL TRAINING-ONLY D5 CONVERGENCE SENSITIVITY")
    print("No auto_arima search. Seven recorded candidate-winner specs only.")
    print("No test-window access. No model-design change.")
    print("=" * 76)
    print("source D5: {}  SHA256 {}...".format(
        sel["protocol_tag"], selection_sha[:16]))
    print("scale: {}; d={}, D={}, constant={}".format(
        sel["scale"], d_sel, D_sel, expected_intercept))
    print("diagnostic optimizer: method='{}', maxiter={}".format(
        DIAGNOSTIC_METHOD, DIAGNOSTIC_MAXITER))
    print("")

    # --------------------- exact seven-specification refits -------------------
    rows = []

    cand_rank = {c: i for i, c in enumerate(d5.CANDIDATE_ORDER)}
    winners = winners.assign(
        _cand_rank=winners["candidate"].map(cand_rank)
    ).sort_values("_cand_rank").drop(columns="_cand_rank")

    for _, old in winners.iterrows():
        candidate = str(old["candidate"])
        order = (
            int(old["p"]),
            d_sel,
            int(old["q"]),
        )
        seasonal_order = (
            int(old["P"]),
            D_sel,
            int(old["Q"]),
            int(d5.SEASONAL_M),
        )
        X = d5.build_candidate(candidate, y.index)

        t0 = time.time()
        model = pm.ARIMA(
            order=order,
            seasonal_order=seasonal_order,
            with_intercept=expected_intercept,
            method=DIAGNOSTIC_METHOD,
            maxiter=DIAGNOSTIC_MAXITER,
            suppress_warnings=True,
        )
        model.fit(
            y.to_numpy(dtype=float),
            X=X.to_numpy(dtype=float),
        )

      
        rec = d5.record_from_fit(candidate, model)
        elapsed = time.time() - t0

        rows.append({
            "candidate": candidate,
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "with_intercept": expected_intercept,
            "n_exog": int(X.shape[1]),

            "aicc_50": float(old["aicc"]),
            "converged_50": as_bool(old["converged"]),
            "lb_pvalue_50": (
                None if pd.isna(old["lb_pvalue"])
                else float(old["lb_pvalue"])
            ),
            "lb_pass_50": as_bool(old["lb_pass"]),

            "aicc_500": float(rec.aicc),
            "aicc_shift_500_minus_50": float(rec.aicc - float(old["aicc"])),
            "converged_500": rec.converged,
            "iterations_500": int(
                (getattr(model.arima_res_, "mle_retvals", {}) or {})
                .get("iterations", -1)
            ),
            "lb_pvalue_500": float(rec.lb_pvalue),
            "lb_pass_500": bool(rec.lb_pass),
            "lb_lag_500": int(rec.lb_lag),
            "lb_df_500": int(rec.lb_df),
            "nobs_effective_500": int(rec.nobs_effective),
            "k_params_total_500": int(rec.k_params_total),
            "seconds": round(elapsed, 1),
        })

        print(
            "[{}] {}{} -> AICc 50={:.6f}; 500={:.6f}; shift={:+.6f}; "
            "converged500={} (iters {}); LB500={} ({:.1f}s)"
            .format(
                candidate,
                order,
                seasonal_order,
                float(old["aicc"]),
                rec.aicc,
                rec.aicc - float(old["aicc"]),
                rec.converged,
                rows[-1]["iterations_500"],
                rec.lb_pass,
                elapsed,
            ),
            flush=True,
        )

    tab = pd.DataFrame(rows)

    # Comparability check: all seven fixed fits must use the same effective n.
    require(tab["nobs_effective_500"].nunique() == 1,
            "500-iteration refits do not share nobs_effective.")

    # Original ranks by the historical 50-iteration candidate-winner AICc.
    original_sort = (
        tab.assign(_cand_rank=tab["candidate"].map(cand_rank))
        .sort_values(
            ["aicc_50", "k_params_total_500", "_cand_rank"],
            kind="mergesort",
        )
        .index.tolist()
    )
    rank50 = {idx: pos + 1 for pos, idx in enumerate(original_sort)}
    tab["rank_aicc_50"] = [rank50[i] for i in tab.index]

    refit_sort = (
        tab.assign(_cand_rank=tab["candidate"].map(cand_rank))
        .sort_values(
            ["aicc_500", "k_params_total_500", "_cand_rank"],
            kind="mergesort",
        )
        .index.tolist()
    )
    rank500 = {idx: pos + 1 for pos, idx in enumerate(refit_sort)}
    tab["rank_aicc_500"] = [rank500[i] for i in tab.index]

    tab["rank_change_500_minus_50"] = (
        tab["rank_aicc_500"] - tab["rank_aicc_50"]
    )

    tab = tab.sort_values("rank_aicc_50").reset_index(drop=True)
    tab.to_csv(outdir / "d5_convergence_sensitivity.csv", index=False)

    all_converged_500 = bool((tab["converged_500"] == True).all())  # noqa: E712
    original_best = str(
        tab.sort_values(["rank_aicc_50"]).iloc[0]["candidate"]
    )
    refit_best = (
        str(tab.sort_values(["rank_aicc_500"]).iloc[0]["candidate"])
        if all_converged_500 else None
    )
    selected_candidate = candidate_from_selection(sel)

    # Unrestricted AICc-ordering diagnostic. This is useful numerically but
    # is NOT, by itself, the frozen D5 selection rule because D5 also applies
    # the Ljung-Box gate (and D6 fallback if no fit passes).
    aicc_ranking_survives = (
        bool(original_best == refit_best)
        if all_converged_500 else None
    )

    # Gate outcomes can change because the 500-iteration refit can land at a
    # different optimum with different residuals.
    gate_flips = [
        str(r["candidate"]) for _, r in tab.iterrows()
        if bool(r["lb_pass_50"]) != bool(r["lb_pass_500"])
    ]

    def seven_winner_rule_choice(aicc_col: str, pass_col: str,
                                 rank_col: str):
        """Apply the D5 gate/D6-fallback rule within this seven-row subset."""
        passing = tab[tab[pass_col] == True]  # noqa: E712
        if len(passing):
            row = passing.sort_values([rank_col]).iloc[0]
            return str(row["candidate"]), "D5_gate", int(len(passing))
        row = tab.sort_values([rank_col]).iloc[0]
        return str(row["candidate"]), "D6_fallback", 0

    rule_choice_50, rule_mode_50, n_passing_50 = seven_winner_rule_choice(
        "aicc_50", "lb_pass_50", "rank_aicc_50"
    )

    if all_converged_500:
        rule_choice_500, rule_mode_500, n_passing_500 = (
            seven_winner_rule_choice(
                "aicc_500", "lb_pass_500", "rank_aicc_500"
            )
        )
        seven_winner_rule_choice_survives = bool(
            rule_choice_50 == rule_choice_500
        )
        selected_matches_500_rule_choice = bool(
            selected_candidate == rule_choice_500
        )
    else:
        rule_choice_500 = None
        rule_mode_500 = None
        n_passing_500 = None
        seven_winner_rule_choice_survives = None
        selected_matches_500_rule_choice = None

    selected_candidate_remains_best = (
        bool(selected_candidate == refit_best)
        if all_converged_500 else None
    )

    original_ranking = (
        tab.sort_values("rank_aicc_50")["candidate"].tolist()
    )
    refit_ranking = (
        tab.sort_values("rank_aicc_500")["candidate"].tolist()
        if all_converged_500 else None
    )

    out = {
        "diagnostic": DIAGNOSTIC_NAME,
        "non_decisional": True,
        "training_only": True,
        "changes_model_design": False,
        "operative_protocol_tag": OPERATIVE_PROTOCOL_TAG,
        "source_d5_protocol_tag": sel.get("protocol_tag"),
        "scope": (
            "Exact fixed-specification refit of the seven per-candidate "
            "winners recorded by transformed D5. No auto_arima search and "
            "no reranking of the full visited-fit pool."
        ),
        "question": (
            "For the seven recorded D5 candidate-winner specifications, how "
            "do unrestricted AICc ordering and the Ljung-Box-gated D5/D6 "
            "choice behave when the exact specifications are refitted under "
            "LBFGS maxiter=500?"
        ),
        "source_artifacts": {
            "d5_selection": {
                "path": str(selection_path.resolve()),
                "sha256": selection_sha,
            },
            "d5_candidate_winners": {
                "path": str(winners_path.resolve()),
                "sha256": winners_sha,
            },
            "training_data": {
                "path": str(data_path.resolve()),
                "sha256": data_sha,
            },
        },
        "d5_state": {
            "scale": sel["scale"],
            "d": d_sel,
            "D": D_sel,
            "with_intercept": expected_intercept,
            "selected_candidate": selected_candidate,
            "historical_maxiter": int(sel["hk_settings"]["maxiter"]),
            "historical_method": sel["hk_settings"]["method"],
        },
        "diagnostic_optimizer": {
            "method": DIAGNOSTIC_METHOD,
            "maxiter": DIAGNOSTIC_MAXITER,
        },
        "result": {
            "all_seven_converged_at_500": all_converged_500,
            "original_candidate_winner_ranking": original_ranking,
            "refit_candidate_winner_ranking": refit_ranking,
            "original_best_candidate": original_best,
            "refit_best_candidate": refit_best,
            "aicc_ranking_survives": aicc_ranking_survives,
            "gate_outcome_flips": gate_flips,
            "n_lb_passing_candidate_winners_50": n_passing_50,
            "n_lb_passing_candidate_winners_500": n_passing_500,
            "seven_winner_rule_choice_50": rule_choice_50,
            "seven_winner_rule_mode_50": rule_mode_50,
            "seven_winner_rule_choice_500": rule_choice_500,
            "seven_winner_rule_mode_500": rule_mode_500,
            "seven_winner_rule_choice_survives":
                seven_winner_rule_choice_survives,
            "frozen_selected_candidate_matches_500_seven_winner_rule_choice":
                selected_matches_500_rule_choice,
            "frozen_d5_selected_candidate": selected_candidate,
            "frozen_selected_candidate_remains_best_among_seven":
                selected_candidate_remains_best,
            "interpretation_rule": (
                "Diagnostic only. The frozen D5 selection is not changed "
                "regardless of this result unless a separate prospective "
                "pre-test methodological decision is explicitly adopted."
            ),
        },
        "rows": tab.replace({np.nan: None}).to_dict(orient="records"),
        "environment_mismatch": env_diffs,
        "environment": d5.observed_environment(),
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    json_path = outdir / "d5_convergence_sensitivity.json"
    json_path.write_text(json.dumps(out, indent=2))

    # ------------------------------- report ----------------------------------
    L = []
    L.append("# D5 convergence sensitivity diagnostic")
    L.append("")
    L.append(
        "**NON-DECISIONAL, TRAINING-ONLY sensitivity analysis.** "
        "No `auto_arima` search was rerun, no test-window observation was "
        "accessed, and no Section D model-design choice is changed by this "
        "diagnostic."
    )
    L.append("")
    L.append(
        "Source D5 artifact: `{}` (`{}`, SHA-256 `{}`).".format(
            selection_path.name, sel["protocol_tag"], selection_sha
        )
    )
    L.append(
        "Candidate-winner artifact: `{}` (SHA-256 `{}`).".format(
            winners_path.name, winners_sha
        )
    )
    L.append(
        "Training data SHA-256 `{}`; scale `{}`; d = {}, D = {}, "
        "constant = {}.".format(
            data_sha, sel["scale"], d_sel, D_sel, expected_intercept
        )
    )
    L.append("")
    L.append("## Question")
    L.append("")
    L.append(
        "Holding each of the seven D5 per-candidate winner specifications "
        "exactly fixed, does the annual-form AICc ranking change when those "
        "same specifications are refitted with `method='lbfgs'`, "
        "`maxiter=500`?"
    )
    L.append("")
    L.append(
        "This deliberately does **not** refit the entire historical visited "
        "pool and does not run a new Hyndman-Khandakar search. It isolates "
        "optimizer convergence for the seven recorded candidate winners."
    )
    L.append("")
    L.append("## Results")
    L.append("")
    L.append(
        "| candidate | fixed orders | AICc @50 | conv @50 | AICc @500 | "
        "shift | conv @500 | iters | LB @500 | rank 50→500 |"
    )
    L.append(
        "|---|---|---:|---|---:|---:|---|---:|---|---:|"
    )

    for _, r in tab.sort_values("rank_aicc_50").iterrows():
        L.append(
            "| {} | {} {} | {:.6f} | {} | {:.6f} | {:+.6f} | {} | {} | "
            "{} | {}→{} |".format(
                r["candidate"],
                r["order"],
                r["seasonal_order"],
                r["aicc_50"],
                "yes" if r["converged_50"] else "no",
                r["aicc_500"],
                r["aicc_shift_500_minus_50"],
                "yes" if r["converged_500"] else "no",
                int(r["iterations_500"]),
                "pass" if r["lb_pass_500"] else "fail",
                int(r["rank_aicc_50"]),
                int(r["rank_aicc_500"]),
            )
        )

    L.append("")
    L.append("## Interpretation")
    L.append("")

    if not all_converged_500:
        stalled = tab.loc[tab["converged_500"] != True, "candidate"].tolist()  # noqa: E712
        L.append(
            "**Diagnostic incomplete.** Not all seven exact specifications "
            "converged by maxiter=500: {}. No stable converged ranking is "
            "claimed.".format(stalled)
        )
    else:
        if aicc_ranking_survives:
            L.append(
                "All seven exact specifications converged. The unrestricted "
                "AICc winner remained **{}** after the common 500-iteration "
                "refit. This statement concerns AICc ordering only; the "
                "frozen D5 rule also requires the Ljung-Box gate.".format(
                    refit_best
                )
            )
        else:
            L.append(
                "All seven exact specifications converged, but the "
                "unrestricted AICc winner changed from **{}** at the original "
                "50-iteration fits to **{}** under the common 500-iteration "
                "refits. This is a sensitivity finding only; the historical "
                "D5 selection artifact is not rewritten.".format(
                    original_best, refit_best
                )
            )

        if selected_candidate == refit_best:
            L.append(
                "The candidate selected by frozen D5 (**{}**) is also the "
                "lowest-AICc candidate among these seven converged exact "
                "refits.".format(selected_candidate)
            )
        else:
            L.append(
                "The candidate selected by frozen D5 (**{}**) is not the "
                "lowest-AICc candidate among these seven converged exact "
                "refits (**{}**). This is disclosed as sensitivity evidence "
                "and does not itself constitute re-selection.".format(
                    selected_candidate, refit_best
                )
            )

        L.append(
            "Ljung-Box gate: {}. Within the seven recorded candidate-winner "
            "specifications, applying the frozen D5 gate/D6-fallback rule "
            "would choose **{}** at maxiter=50 ({}; {} passing candidate"
            "{}), and **{}** at maxiter=500 ({}; {} passing candidate{}). "
            "The seven-winner rule choice therefore {}. This remains a "
            "subset diagnostic: frozen D5 itself selected from the full "
            "visited-fit pool, not only these seven rows.".format(
                "no candidate's outcome changed between the two ceilings"
                if not gate_flips else
                "outcome CHANGED for " + ", ".join(gate_flips),
                rule_choice_50,
                rule_mode_50,
                n_passing_50,
                "" if n_passing_50 == 1 else "s",
                rule_choice_500,
                rule_mode_500,
                n_passing_500,
                "" if n_passing_500 == 1 else "s",
                "SURVIVES"
                if seven_winner_rule_choice_survives else "CHANGES"
            )
        )

    L.append("")
    L.append(
        "The full frozen D5 selection was made from the entire visited pool "
        "after the Ljung-Box gate, whereas this diagnostic concerns only the "
        "seven recorded per-candidate winners. It must therefore not be "
        "described as a complete maxiter=500 rerun of D5."
    )

    report_path = outdir / "d5_convergence_sensitivity_report.md"
    report_path.write_text("\n".join(L) + "\n")

    # ------------------------------- MLflow ----------------------------------
    project_root = Path(__file__).resolve().parents[1]
    tracking_uri = (project_root / "mlruns").as_uri()

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.experiment)

        with mlflow.start_run(run_name="D5_convergence_sensitivity"):
            mlflow.set_tags({
                "diagnostic.name": DIAGNOSTIC_NAME,
                "diagnostic.non_decisional": "true",
                "diagnostic.training_only": "true",
                "protocol.tag": OPERATIVE_PROTOCOL_TAG,
                "source_d5_protocol_tag": sel.get("protocol_tag"),
                "source_d5_selection_sha256": selection_sha,
                "source_d5_winners_sha256": winners_sha,
                "data_sha256": data_sha,
                "protocol.run_utc": out["run_utc"],
            })
            mlflow.log_params({
                "scale": sel["scale"],
                "d": d_sel,
                "seasonal_D": D_sel,
                "with_intercept": str(expected_intercept),
                "historical_maxiter": int(sel["hk_settings"]["maxiter"]),
                "diagnostic_maxiter": DIAGNOSTIC_MAXITER,
                "method": DIAGNOSTIC_METHOD,
                "n_candidate_winners": len(tab),
                "frozen_d5_selected_candidate": selected_candidate,
            })
            summary_metrics = {
                "all_seven_converged_500": int(all_converged_500),
                "gate_flips_count": int(len(gate_flips)),
                "n_lb_passing_candidate_winners_50": int(n_passing_50),
            }
            if n_passing_500 is not None:
                summary_metrics["n_lb_passing_candidate_winners_500"] = int(
                    n_passing_500
                )
            if aicc_ranking_survives is not None:
                summary_metrics["aicc_ranking_survives"] = int(
                    aicc_ranking_survives
                )
            if seven_winner_rule_choice_survives is not None:
                summary_metrics["seven_winner_rule_choice_survives"] = int(
                    seven_winner_rule_choice_survives
                )
            if selected_candidate_remains_best is not None:
                summary_metrics["selected_candidate_remains_best_aicc"] = int(
                    selected_candidate_remains_best
                )
            if selected_matches_500_rule_choice is not None:
                summary_metrics[
                    "frozen_selected_matches_500_seven_winner_rule_choice"
                ] = int(selected_matches_500_rule_choice)
            mlflow.log_metrics(summary_metrics)

            for _, r in tab.iterrows():
                c = str(r["candidate"]).replace("fourier_", "")
                mlflow.log_metric(
                    "aicc500_{}".format(c),
                    float(r["aicc_500"]),
                )
                mlflow.log_metric(
                    "aicc_shift_{}".format(c),
                    float(r["aicc_shift_500_minus_50"]),
                )

            for f in (
                "d5_convergence_sensitivity.csv",
                "d5_convergence_sensitivity.json",
                "d5_convergence_sensitivity_report.md",
            ):
                mlflow.log_artifact(str(outdir / f), artifact_path="diagnostic")

            mlflow.log_artifact(str(selection_path), artifact_path="inputs")
            mlflow.log_artifact(str(winners_path), artifact_path="inputs")
            mlflow.log_artifact(str(Path(__file__).resolve()),
                                artifact_path="source")

        print("")
        print("logged to MLflow experiment '{}' at {}".format(
            args.experiment, tracking_uri))
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            "MLflow logging FAILED: {}\nDiagnostic artifacts were written to "
            "disk, but treat the diagnostic record as incomplete until the "
            "MLflow run is present.".format(exc)
        )

    print("")
    print("D5 CONVERGENCE SENSITIVITY COMPLETE")
    print("all seven converged at 500: {}".format(all_converged_500))
    if all_converged_500:
        print("original candidate-winner ranking: {}".format(original_ranking))
        print("refit candidate-winner ranking:    {}".format(refit_ranking))
        print("unrestricted AICc ranking survives: {}".format(
            aicc_ranking_survives))
        print("Ljung-Box gate flips: {}".format(
            gate_flips if gate_flips else "none"))
        print(
            "seven-winner frozen-rule choice: {} ({}) -> {} ({})".format(
                rule_choice_50, rule_mode_50,
                rule_choice_500, rule_mode_500,
            )
        )
        print("seven-winner rule choice survives: {}".format(
            seven_winner_rule_choice_survives))
        print("frozen D5 selected candidate remains unrestricted AICc best: "
              "{}".format(selected_candidate_remains_best))
        print("frozen D5 selected candidate matches the 500-iteration "
              "seven-winner rule choice: {}".format(
                  selected_matches_500_rule_choice))
    print("outputs written to {}".format(outdir.resolve()))

    if not all_converged_500:
        sys.exit(2)


if __name__ == "__main__":
    main()
