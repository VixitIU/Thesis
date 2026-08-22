"""D4: inclusion of a constant.

Implements protocol row D4: a constant is included iff d = D = 0. With d = 1
selected at D2 and D = 0 at D3, d + D = 1 and no constant is included.

Declared restriction (D4): this is stricter than the Hyndman-Khandakar
ARIMA implementation convention, under which a drift term may be considered
when d + D = 1. Under the frozen protocol, a constant is permitted only when
d = D = 0. The step involves no estimation; it is logged for continuity of
the selection record.
"""
import experiment

D2_d = 1   # selected at D2, 22 August 2026
D3_D = 0   # selected at D3, 22 August 2026


def main() -> None:
    constant = (D2_d == 0 and D3_D == 0)

    with experiment.run("D4", "constant", diff_d=D2_d, seasonal_D=D3_D) as mlf:
        mlf.log_param("constant", constant)
        mlf.set_tags({
            "rule": "constant iff d = D = 0",
            "drift_permitted_by_HK2008_at_d_plus_D_eq_1": "True",
            "drift_imposed": "False",
        })

    print(f"  d = {D2_d}, D = {D3_D}, d + D = {D2_d + D3_D}")
    print(f"\nD4: constant = {constant}; no deterministic drift imposed")


if __name__ == "__main__":
    main()