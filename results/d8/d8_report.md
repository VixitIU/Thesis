# D8 run report -- FX lag

Protocol state protocol-v1.5 (freeze tag protocol-v1.0), row D8. Run (UTC): 2026-08-26T02:28:55+00:00.

Operative M1 read from `d5_selection.json`: fourier_K6, ARIMA(1, 1, 1)(1, 0, 1)[7], no constant, on the log1p scale, with orders and annual form held fixed throughout (20 August 2026 addendum). Lagged FX comes from `indicator_history.csv` (SHA-256 `ae994353fa0310d28240fb3740c5f6cf61d11be3847293f1ca2ab84243323f99`), which extends the series back before the spine start; aligned_train.csv alone cannot supply a 90-day lag for the first training day.

## Stage 1 -- CCF identification

Following Box-Jenkins transfer-function identification, an auxiliary ARIMA model is fitted to the FX input and that same fixed state-space filter is applied to FX and to the response before computing the CCF. Raw CCF is diagnostic only because serial dependence in the persistent FX input can create misleading cross-correlation peaks. The auxiliary filter is estimated on the frozen 884-day training window only (2023-07-01 through 2025-11-30), not on the pre-spine lead-in. It is ARIMA(0, 1, 1)(0, 0, 0)[7], intercept=False, AICc -4325.02, selected with KPSS/OCSB differencing tests, d <= 1 and D <= 1; it converged in 2 iterations under method='lbfgs', maxiter=500. State-space initialization is removed using the filtered results' loglikelihood_burn (FX 1, response 1). All lag CCFs use the same response sample, 2023-07-02 through 2025-11-30 (883 observations).

Candidate lags 28-90 (63 in total). The five largest |prewhitened CCF|, ties broken to the shorter lag, are **[45, 84, 87, 43, 33]**.

| lag | prewhitened CCF | rank | raw CCF (diagnostic) |
|---|---|---|---|
| 45 | +0.0732 | 1 | +0.4972 |
| 84 | -0.0686 | 2 | +0.4497 |
| 87 | +0.0652 | 3 | +0.4462 |
| 43 | -0.0532 | 4 | +0.4876 |
| 33 | +0.0529 | 5 | +0.4678 |

Diagnostic: the raw CCF would have screened lags [61, 62, 63, 64, 65] instead, which differs from the prewhitened set. The raw values take no part in the selection and are recorded only so the difference between the two readings is visible.

## Stage 2 -- AICc

| lag | AICc | converged | iterations |
|---|---|---|---|
| 84 | -28.8097 | yes | 157 |
| 33 | -24.1071 | yes | 174 |
| 43 | -22.2524 | yes | 185 |
| 45 | -20.4596 | yes | 182 |
| 87 | -20.4491 | yes | 217 |

All five fits share the estimation sample (883 effective observations) and regressor count (13); both were checked. Per the addendum of 24 August 2026 they were fitted under a common ceiling (method='lbfgs', maxiter=500) and all converged.

## Selection

**FX lag = 84 days**, AICc -28.8097, ahead of the runner-up by 4.7026. Prewhitened CCF at that lag -0.0686 (screen rank 2). Ties are broken to the shorter lag at both stages, as the frozen row declares.

E6 holds by construction: every candidate lag is at least 28 days, which is the longest forecast horizon, so all regressor values predate the origin.

## Next step

M3 = M1 + FX at lag 84; D10 transfers this lag to M5 without re-tuning.

## Artifacts

d8_ccf_screen.csv (all 63 candidate lags, prewhitened and raw), d8_lag_grid.csv (the five AICc candidates), d8_lag_selection.json, this report. No proprietary artifact: the tables carry correlations, AICc values and lags only.
