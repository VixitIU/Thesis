# D11 run report -- pre-declared scale trigger

Protocol state protocol-v1.3 (freeze tag protocol-v1.0), row D11, evaluated at the point fixed by the addendum of 19 August 2026: after D5, before D7. Run (UTC): 2026-08-24T05:21:46+00:00.

Evaluated on the residuals of the M1 specification selected at D5: **fourier_K1, ARIMA(1, 1, 1)(1, 0, 0)[7], no intercept**, refitted on the training sample and reproducing the D5 AICc of 6006.9003 to within relative tolerance 1e-06. Residuals are that fit's own one-step-ahead in-sample prediction errors after dropping the burn-in of 1 observation(s), n = 883 -- the same slice the D5 Ljung-Box gate used.

## Trigger statistics

| condition | statistic | value | threshold | holds |
|---|---|---|---|---|
| A: non-normality | Jarque-Bera p | 2.049e-25 | < 0.01 | yes |
| B: heteroskedasticity | Spearman rho(|resid|, fitted) | 0.3055 | > 0.3 | yes |

Residual skewness 0.4235, excess kurtosis 1.5404. Spearman signed rho = 0.3055 (|rho| = 0.3055), p = 1.587e-20; the reading applied is `rho(|residual|, fitted) > threshold`.

## Decision

Rule: trigger iff (JB rejects at 0.01) AND (Spearman rho(|residual|, fitted) > 0.3).

**Trigger FIRES.** D12: re-run D2, D3, D4 and D5 on log(y+1), then continue the ladder on the transformed scale (19 Aug and 23 Aug 2026 addenda).

The scale decision is made once, on training diagnostics only, and applies uniformly to every model in the ladder, as D13 requires. The test set was not consulted.

## Artifacts

d11_scale_trigger.json (machine-readable decision), d11_residuals.csv (date, fitted, residual, |residual| -- the inputs to both statistics), this report.
