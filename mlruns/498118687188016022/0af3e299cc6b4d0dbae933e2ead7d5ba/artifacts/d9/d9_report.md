# D9 run report -- search lag

Protocol state protocol-v1.7 (freeze tag protocol-v1.0), row D9. Run (UTC): 2026-09-17T10:55:48+00:00.

Operative addendum: `addenda.md` (SHA-256 `15c28c232f9d390b3038431cb6eec310d8c9bec06736b730a0c411badd3cc435`).

Operative M1 read from `d5_selection.json`: fourier_K1, ARIMA(0, 1, 2)(0, 0, 0)[7], no constant, on the log1p scale, with orders and annual form held fixed throughout (20 August 2026 addendum). Lagged search interest comes from `indicator_history.csv` (SHA-256 `ae994353fa0310d28240fb3740c5f6cf61d11be3847293f1ca2ab84243323f99`), which extends the series back before the spine start; aligned_train.csv alone cannot supply a 119-day lag for the first training day.

## Stage 1 -- CCF identification

Following Box-Jenkins transfer-function identification, an auxiliary ARIMA model is fitted to the search input and that same fixed state-space filter is applied to the search index and to the response before computing the CCF. Raw CCF is diagnostic only because serial dependence in the persistent indicator input can create misleading cross-correlation peaks. The auxiliary filter is estimated on the frozen 884-day training window only (2023-07-01 through 2025-11-30), not on the pre-spine lead-in. It is ARIMA(0, 1, 0)(0, 0, 1)[7], intercept=False, AICc 14623.36, selected with KPSS/OCSB differencing tests, d <= 1 and D <= 1; it converged in 13 iterations under method='lbfgs', maxiter=500. State-space initialization is removed using the filtered results' loglikelihood_burn (indicator 1, response 1). All lag CCFs use the same response sample, 2023-07-02 through 2025-11-30 (883 observations).

Candidate lags 28-119 (14 in total). The five largest |prewhitened CCF|, ties broken to the shorter lag, are **[42, 91, 105, 70, 63]**.

| lag | prewhitened CCF | rank | raw CCF (diagnostic) |
|---|---|---|---|
| 42 | -0.0724 | 1 | +0.5460 |
| 91 | +0.0383 | 2 | +0.4023 |
| 105 | +0.0361 | 3 | +0.2522 |
| 70 | +0.0355 | 4 | +0.5187 |
| 63 | -0.0320 | 5 | +0.5314 |

Diagnostic: the raw CCF would have screened lags [28, 35, 42, 49, 56] instead, which differs from the prewhitened set. The raw values take no part in the selection and are recorded only so the difference between the two readings is visible.

Stage 1 uses the original search-index units throughout prewhitening and CCF screening. Only after the five candidate lags have been frozen does Stage 2 divide the exogenous search regressor by 1000 for numerical conditioning. For an exogenous regressor this is an exact coefficient reparameterisation: at the same optimum fitted values, likelihood, parameter count, AICc and forecasts are unchanged; only the coefficient scale differs. It therefore alters no candidate lag, CCF value, threshold or D9 selection rule. Downstream coefficients are interpreted per 1000 queries.

## Stage 2 -- AICc

| lag | AICc | converged | iterations |
|---|---|---|---|
| 42 | -10.7284 | yes | 20 |
| 91 | -10.1313 | yes | 27 |
| 63 | -10.0455 | yes | 27 |
| 70 | -9.9420 | yes | 27 |
| 105 | -9.7690 | yes | 27 |

All five fits share the estimation sample (883 effective observations) and regressor count (3); both were checked. Per the addendum of 24 August 2026 they were fitted under a common ceiling (method='lbfgs', maxiter=500) and all converged.

## Selection

**search lag = 42 days**, AICc -10.7284, ahead of the runner-up by 0.5970. Prewhitened CCF at that lag -0.0724 (screen rank 1). Ties are broken to the shorter lag at both stages, as the frozen row declares.

E6 holds by construction: every candidate lag is at least 28 days, which is the longest forecast horizon, so all regressor values predate the origin.

## Next step

M4 = M1 + search index at lag 42; D10 transfers this lag to M5 without re-tuning.

## Artifacts

d9_ccf_screen.csv (all 14 candidate lags, prewhitened and raw), d9_lag_grid.csv (the five AICc candidates), d9_lag_selection.json, this report. No proprietary artifact: the tables carry correlations, AICc values and lags only.
