# D12/D2 run report -- differencing order on log(y + 1)

Protocol state protocol-v1.3 (freeze tag protocol-v1.0), row D2 re-run on the transformed scale as the first step of the D12 branch (19 and 23 August 2026 addenda). Run (UTC): 2026-08-24T06:07:13+00:00.

Authorised by the D11 record: the scale trigger fired (Jarque-Bera p = 2.049e-25; Spearman rho = 0.3055). Input `..\data\aligned_train.csv`, SHA-256 `4224fe90606f4a77e71dafadb99538a8ac878555a32805a48219f27cf7809313`, matching the file D11 ran on.

Both tests are applied to the OLS residuals of log(y + 1) on the pre-declared pilot annual form: Fourier K = 3 at period 365.25 with origin 2023-07-01, plus day-of-week dummies (13 regressors including the constant). The pilot form is used for this test only and is discarded afterwards; it is not the annual form selected at D5.

## Tests on the pilot residuals (levels)

| test | null | statistic | 5% critical | p | rejects |
|---|---|---|---|---|---|
| ADF | unit root | -4.395938 | -2.864845 | 0.0003018 | yes |
| KPSS | stationarity | 1.490668 | 0.463000 | 0.01 (table endpoint) | yes |

Rejection is decided by comparing each statistic to its 5% critical value rather than to an interpolated p-value, because KPSS p-values are clipped at the endpoints of a lookup table. Both routes agreed for both tests; a disagreement aborts the run.

## Decision

Rule: d = 0 iff ADF rejects at 5% AND KPSS does not reject at 5%; otherwise d = 1. d = 2 excluded a priori.

ADF rejects the unit-root null; KPSS rejects the stationarity null. **d = 1.**

## Confirmatory check on the differenced residuals

Diagnostic only. A failure here is recorded and is never used to raise d, since d = 2 is excluded a priori: it would posit a stochastic trend in the growth rate and give forecast variance growing as h^3.

| test | statistic | 5% critical | p | rejects |
|---|---|---|---|---|
| ADF | -17.980794 | -2.864845 | 2.775e-30 | yes |
| KPSS | 0.238187 | 0.463000 | 0.1 (table endpoint) | no |

Outcome: the differenced residuals match the pattern expected of a stationary series (ADF rejects, KPSS does not).

## Settings not fixed by the frozen row

ADF: regression='c', lag selection AIC, lag used 7. KPSS: regression='c', bandwidth rule auto, lags used 17. These are recorded because the frozen D2 row does not fix them.

## Next step

D3 (OCSB at s = 7) on log(y + 1), then D4, then D5 with --scale log1p --d 1 --D <from D3>. Under the 23 August 2026 clarification D4 is re-applied on the transformed scale, and the constant follows deterministically from d and D.

## Artifacts

d12_d2_differencing.json (machine-readable decision), d12_d2_pilot_residuals.csv (**proprietary: exp(fitted + residual) - 1 reconstructs the daily case counts; keep out of the public repository**), this report.
