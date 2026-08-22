"""MLflow setup and environment verification.

Protocol row D1 pins the software environment and requires all runs logged in
MLflow. Imported by every fitting script: verifies the running environment
against D1 before any model is estimated, and records the code state alongside 
each run so that a run can be traced to the exact source that produced it.
"""
from contextlib import contextmanager
from pathlib import Path
import platform
import subprocess
import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow

try:
    PROJECT = Path(__file__).resolve().parents[1]
except NameError:                                  
    PROJECT = Path(r"C:\Users\New\Documents\IU work\Thesis\project")
    
TRACKING_URI = (PROJECT / "mlruns").as_uri()
EXPERIMENT = "medical-assistance-demand-forecasting"

PINNED = {
    "python": "3.12.13",
    "statsmodels": "0.14.6",
    "pandas": "2.3.3",
    "numpy": "2.5.2",
    "scipy": "1.18.0",
    "pmdarima": "2.1.1",
}


def observed_versions() -> dict:
    import numpy, pandas, pmdarima, scipy, statsmodels
    return {
        "python": platform.python_version(),
        "statsmodels": statsmodels.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pmdarima": pmdarima.__version__,
    }


def assert_environment() -> dict:
    """Raise unless every version matches D1 exactly."""
    obs = observed_versions()
    bad = {k: (PINNED[k], obs[k]) for k in PINNED if PINNED[k] != obs[k]}
    if bad:
        detail = "\n".join(f"  {k}: D1 pins {p}, running {o}" for k, (p, o) in bad.items())
        raise RuntimeError(f"Environment does not match protocol row D1:\n{detail}")
    return obs


def git_state() -> dict:
    def g(*args):
        return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()
    return {
        "git_commit": g("rev-parse", "HEAD"),
        "git_branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(g("status", "--porcelain"))),
    }


def init() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


@contextmanager
def run(stage: str, name: str, **params):
    """One MLflow run.

    stage  -- protocol row(s) the run executes, e.g. "D5", "D7".
    name   -- run label, e.g. "M1-fourier-K3".
    params -- anything else worth recording as a run parameter.

    Environment is asserted before the run opens, so a mismatched environment
    fails without leaving a partial run behind.
    """
    env = assert_environment()
    init()
    with mlflow.start_run(run_name=f"{stage} {name}"):
        mlflow.set_tags({"protocol_stage": stage, **git_state()})
        mlflow.log_params({f"env_{k}": v for k, v in env.items()})
        if params:
            mlflow.log_params(params)
        yield mlflow