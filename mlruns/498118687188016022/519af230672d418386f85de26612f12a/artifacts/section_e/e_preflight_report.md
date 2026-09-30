# Section E preflight report

Protocol state protocol-v1.8 (freeze tag protocol-v1.0), row E. Run (UTC): 2026-09-21T06:42:23+00:00.

**No test-window forecast was produced.** This preflight only validated the frozen evaluation design and inputs.

## Operative ladder

M1: fourier_K1 + ARIMA(0, 1, 2)(0, 0, 0)[7], no constant, log(y+1).

M2 adds holidays pad (3, 2); M3 adds FX lag 84; M4 adds search lag 42; M5 inherits all three. Nothing is re-selected.

## Evaluation schedule

216 expanding-window origins, every 1 days from 2025-11-30 through 2026-07-03; reported horizons [7, 14, 28]. The last h=28 target is 2026-07-31.

E6 availability passed at every origin/horizon. The search check includes the conservative 10-day availability bound (6-day week span + 4-day publication-delay bound).

The 216 daily origins are not treated as 216 independent replications. They sample the same fixed test window more densely, and adjacent multi-step forecast errors overlap. E11 therefore uses Bartlett/Newey-West truncation h-1. For intuition only, 216/h corresponds to about 30.9, 15.4 and 7.7 non-overlapping h-day spans at h=7,14,28; these figures are NOT used as degrees of freedom or as an effective sample size.

## D10 replication gate

PASSED. Rebuilt M5 on the exact frozen 884-day training window: AICc -12.189801; operative D10 AICc -12.189801; absolute difference 0 (tolerance 0.001). No rolling-origin estimation or test-window forecast is required for this check.

## MASE scale

M0 uses the same calendar date of the previous year; for 29 February, the benchmark uses the arithmetic mean of 28 February and 1 March of the previous year. In-sample M0 MAE over 2024-07-01..2025-11-30: **12.453668**. This denominator is computed once and reused for all models and horizons.

## E15 VIF

VIF is reported on the frozen 884-day training design only. A diagnostic-only constant is added when computing conventional centered VIFs; this constant is not included in any forecasting model. VIF is descriptive and has no selection threshold.

| model | max VIF | variable |
|---|---:|---|
| M2 | 1.0924 | hol_H_NY |
| M3 | 2.2560 | fx_lag84 |
| M4 | 2.3054 | search_lag42_per1000 |
| M5 | 2.8225 | fourier_cos1 |

## Ready state

Preflight passed, including the D10 M5 AICc replication gate. Running the script with `--mode execute` will first fit and verify convergence for all 1080 M1-M5 origin models. Only after every fit converges will the script produce the first test-window forecast.
