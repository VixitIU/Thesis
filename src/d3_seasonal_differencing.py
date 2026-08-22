"""D3: seasonal differencing order D at the weekly period.

Implements protocol row D3: the OCSB test at s = 7 as implemented in pmdarima;
D = 1 if the test does not reject, else D = 0. Applies to the weekly period
only -- annual seasonality is deterministic by design and never differenced.

Not specified by D3, and therefore implementation choices recorded here rather
than pre-declared: the series the test is applied to, and pmdarima's internal
lag-selection method for the OCSB regression.
"""
import pandas as pd
from pmdarima.arima import OCSBTest

import align
import experiment

S = 7
OCSB_LAG_METHOD = "aic"
OCSB_MAX_LAG = 3


def select_D(y: pd.Series) -> tuple:
    """OCSB at s = 7 on the case series in levels.

    The test is applied to y directly rather than to the D2 pilot residuals:
    this input choice is not specified by frozen protocol row D3 and is
    therefore recorded as an implementation choice.
    """
    test = OCSBTest(
        m=S,
        lag_method=OCSB_LAG_METHOD,
        max_lag=OCSB_MAX_LAG,
    )

    D = int(test.estimate_seasonal_differencing_term(y.values))
    stat = float(test._compute_test_statistic(y.values))
    crit = float(test._calc_ocsb_crit_val(S))

    return D, {
        "ocsb_stat": stat,
        "ocsb_crit_val": crit,
        "ocsb_rejects": D == 0,
        "s": S,
        "D_selected": D,
    }


def main() -> None:
    tr = align.load_train()
    y = tr["cases"]
    D, out = select_D(y)

    with experiment.run("D3", "seasonal-differencing", s=S,
                        ocsb_lag_method=OCSB_LAG_METHOD,
                        ocsb_max_lag=OCSB_MAX_LAG,
                        n_train=len(y),
                        applied_to="cases in levels") as mlf:
        mlf.log_param("D_selected", D)
        mlf.log_metrics({k: v for k, v in out.items()
                         if k != "D_selected"
                         and isinstance(v, (int, float))
                         and not isinstance(v, bool)})
        mlf.set_tags({k: str(v) for k, v in out.items() if isinstance(v, bool)})

    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"\nD3 selects D = {D}")


if __name__ == "__main__":
    main()