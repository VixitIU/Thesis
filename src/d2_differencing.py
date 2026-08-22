"""D2: differencing order d.

Implements protocol row D2. Frozen in D2: the candidate set d in {0, 1};
d = 0 only if ADF rejects a unit root at 5% AND KPSS does not reject
stationarity at 5%, otherwise d = 1; tests applied to the OLS residuals of y
on a pre-declared pilot annual form (Fourier K = 3 plus day-of-week dummies),
used for this test only; d = 2 excluded a priori; and, if d = 1, a
confirmatory re-run on the differenced residuals whose outcome is reported as
a diagnostic and never used to raise d.

Not specified by D2, and therefore implementation choices recorded here rather
than pre-declared: the deterministic term of the ADF regression, the ADF
lag-selection method, the deterministic term of the KPSS regression, and the
KPSS bandwidth rule. The values used are given at each call site below and are
logged with the run.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
import align
import experiment

PILOT_K = 3
ANNUAL_PERIOD = 365.25
ALPHA = 0.05


def pilot_design(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Pilot annual form declared in D2: Fourier K = 3 at period 365.25 plus
    day-of-week dummies. Six dummies with an intercept are used, spanning the
    same column space as seven dummies without one, so the residuals do not
    depend on this parameterisation. Discarded after D2; the annual form of
    the ladder is selected independently at D5.
    """
    t = np.arange(len(index), dtype=float)
    X = pd.DataFrame(index=index)
    for k in range(1, PILOT_K + 1):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * t / ANNUAL_PERIOD)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * t / ANNUAL_PERIOD)
    dow = pd.get_dummies(index.dayofweek, prefix="dow", drop_first=True).astype(float)
    dow.index = index
    X = pd.concat([X, dow], axis=1)
    return sm.add_constant(X)


def pilot_residuals(y: pd.Series) -> pd.Series:
    X = pilot_design(y.index)
    fit = sm.OLS(y.values, X.values).fit()
    return pd.Series(fit.resid, index=y.index, name="pilot_resid")


def unit_root_tests(resid: pd.Series) -> dict:
    """ADF and KPSS on residuals residualised against the pilot deterministic
    terms (intercept, pilot annual Fourier terms, day-of-week indicators).

    Implementation choices, not declared in D2:
      - ADF regression="n": the residuals are mean-zero by construction, an
        intercept in the ADF regression being redundant. Departs from the
        statsmodels default of "c".
      - ADF autolag="AIC" and KPSS nlags="auto": the default statsmodels
        lag-selection conventions, supplied explicitly rather than omitted.
      - KPSS regression="c": level stationarity, the statsmodels default.

    statsmodels interpolates KPSS p-values from a published table and clips
    them to [0.01, 0.10]; a reported 0.01 is an upper bound, which suffices
    for the comparison against 5% that D2 requires.
    """
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(
        resid.values, regression="n", autolag="AIC"
    )
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
        resid.values, regression="c", nlags="auto"
    )
    kpss_at_table_bound = kpss_p in (0.01, 0.10)
    return {
        "adf_stat": float(adf_stat), "adf_p": float(adf_p), "adf_lags": int(adf_lags),
        "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p), "kpss_lags": int(kpss_lags),
        "kpss_p_at_table_bound": kpss_at_table_bound,
    }


def select_d(y: pd.Series) -> tuple:
    resid = pilot_residuals(y)
    t = unit_root_tests(resid)

    adf_rejects = t["adf_p"] < ALPHA
    kpss_rejects = t["kpss_p"] < ALPHA
    d = 0 if (adf_rejects and not kpss_rejects) else 1

    out = {**t, "adf_rejects_unit_root": adf_rejects,
           "kpss_rejects_stationarity": kpss_rejects, "d_selected": d}

    if d == 1:
        c = unit_root_tests(resid.diff().dropna())
        out.update({f"confirm_{k}": v for k, v in c.items()})
        out["confirm_passes"] = bool(c["adf_p"] < ALPHA and c["kpss_p"] >= ALPHA)

    return d, out


def main() -> None:
    tr = align.load_train()
    y = tr["cases"]
    d, out = select_d(y)

    with experiment.run("D2", "differencing", pilot_fourier_K=PILOT_K,
                    alpha=ALPHA, n_train=len(y),
                    adf_regression="n", adf_autolag="AIC",
                    kpss_regression="c", kpss_nlags="auto") as mlf:
        mlf.log_param("d_selected", d)
        mlf.log_metrics({k: v for k, v in out.items()
                         if k != "d_selected"
                         and isinstance(v, (int, float))
                         and not isinstance(v, bool)})
        mlf.set_tags({k: str(v) for k, v in out.items() if isinstance(v, bool)})

    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"\nD2 selects d = {d}")


if __name__ == "__main__":
    main()