#!/usr/bin/env python
"""MLflow housekeeping before publication. Changes no result.

Three subcommands, each touching only run metadata or copying records out:

  export   write selected runs (params, metrics, tags, times, status) to JSON
           files, e.g. the count-scale D2-D4 runs, whose outputs exist only in
           MLflow, so that they can be published beside the other results
  archive  log an existing result folder as a new MLflow run tagged as a
           retrospective archive; refuses to create a duplicate
  note     set the description shown in the MLflow UI, plus a tag, on one run
           (e.g. to mark superseded, failed or post hoc runs unambiguously)

Examples (from the project root, with MLFLOW_ALLOW_FILE_STORE=true):
  python src/mlflow_publish_tools.py list
  python src/mlflow_publish_tools.py export --outdir results/d2_d4_count
      --names D2_differencing D3_seasonal-differencing D4_constant
  python src/mlflow_publish_tools.py archive
      --folder results/diagnostic_posthoc/d11_recheck
      --run-name "ARCHIVE POSTHOC D11 recheck" --row D11
  python src/mlflow_publish_tools.py note --run-id <id>
      --text "Post hoc diagnostic, count scale ..." --tag-key posthoc --tag-value true
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow  
from mlflow.tracking import MlflowClient 

DEFAULT_EXPERIMENT = "medical-assistance-demand-forecasting"


def require(condition: bool, message: str) -> None:
    if not condition:
        sys.exit("STOP: " + message)


def norm(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", (name or "").lower())


def client_for(mlruns: str) -> MlflowClient:
    path = Path(mlruns)
    require(path.is_dir(), "no MLflow store at {}".format(path.resolve()))
    mlflow.set_tracking_uri(path.resolve().as_uri())
    return MlflowClient()


def all_runs(client: MlflowClient):
    exps = client.search_experiments()
    return client.search_runs([e.experiment_id for e in exps],
                              max_results=5000,
                              order_by=["attributes.start_time ASC"])


def fmt_time(ms) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(
        timespec="seconds")


def cmd_list(args) -> None:
    client = client_for(args.mlruns)
    for r in all_runs(client):
        print("{}  {:<8}  {:<40}  {}".format(
            fmt_time(r.info.start_time), r.info.status,
            r.info.run_name or "", r.info.run_id))


def cmd_export(args) -> None:
    client = client_for(args.mlruns)
    runs = all_runs(client)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        hits = [r for r in runs if norm(r.info.run_name) == norm(name)
                and r.info.status == "FINISHED"]
        require(len(hits) == 1,
                "{} finished runs match {!r}: {}. Export by --names with the "
                "exact run name, or resolve duplicates first.".format(
                    len(hits), name, [r.info.run_id for r in hits]))
        r = hits[0]
        record = {
            "exported_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "source": "MLflow file store (export only; nothing re-run)",
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "experiment_id": r.info.experiment_id,
            "status": r.info.status,
            "start_utc": fmt_time(r.info.start_time),
            "end_utc": fmt_time(r.info.end_time),
            "params": dict(sorted(r.data.params.items())),
            "metrics": dict(sorted(r.data.metrics.items())),
            "tags": {k: v for k, v in sorted(r.data.tags.items())
                     if not k.startswith("mlflow.log-model")},
        }
        path = outdir / "{}.json".format(re.sub(r"[^\w\-]+", "_", r.info.run_name))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
        print("exported {} ({}) -> {}".format(r.info.run_name, r.info.run_id, path))


def cmd_archive(args) -> None:
    client = client_for(args.mlruns)
    folder = Path(args.folder)
    require(folder.is_dir(), "no folder {}".format(folder))
    clash = [r for r in all_runs(client) if norm(r.info.run_name) == norm(args.run_name)]
    require(not clash, "a run named {!r} already exists ({}); nothing done".format(
        args.run_name, [r.info.run_id for r in clash]))
    hashes = {}
    for f in sorted(p for p in folder.rglob("*") if p.is_file()):
        hashes[str(f.relative_to(folder))] = hashlib.sha256(f.read_bytes()).hexdigest()
    require(bool(hashes), "{} is empty".format(folder))
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags({"archive_backfill": "true",
                         "execution_recomputed": "false",
                         "protocol.row": args.row,
                         "posthoc": str(args.posthoc).lower(),
                         "archived_folder": folder.as_posix(),
                         "mlflow.note.content": args.note or (
                             "Retrospective archive of {}; the files were produced "
                             "earlier and are logged unchanged.".format(folder.as_posix()))})
        mlflow.log_dict(hashes, "archived_files_sha256.json")
        mlflow.log_artifacts(str(folder), artifact_path="archived")
        print("archived {} as run {} ({} files)".format(folder, run.info.run_id,
                                                        len(hashes)))


def cmd_note(args) -> None:
    client = client_for(args.mlruns)
    run = client.get_run(args.run_id)
    client.set_tag(run.info.run_id, "mlflow.note.content", args.text)
    if args.tag_key:
        client.set_tag(run.info.run_id, args.tag_key, args.tag_value or "true")
    print("annotated {} ({})".format(run.info.run_name, run.info.run_id))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mlruns", default="mlruns")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    e = sub.add_parser("export")
    e.add_argument("--outdir", required=True)
    e.add_argument("--names", nargs="+", required=True)
    a = sub.add_parser("archive")
    a.add_argument("--folder", required=True)
    a.add_argument("--run-name", required=True)
    a.add_argument("--row", required=True)
    a.add_argument("--posthoc", action="store_true")
    a.add_argument("--note", default=None)
    a.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    n = sub.add_parser("note")
    n.add_argument("--run-id", required=True)
    n.add_argument("--text", required=True)
    n.add_argument("--tag-key", default=None)
    n.add_argument("--tag-value", default=None)
    args = ap.parse_args()
    {"list": cmd_list, "export": cmd_export, "archive": cmd_archive,
     "note": cmd_note}[args.cmd](args)


if __name__ == "__main__":
    main()
