# D12/D5 convergence diagnostic

**Diagnostic only. This analysis does not modify the protocol-selected D5 specification.**

The transformed-scale D5 winner was **fourier_K6, ARIMA(1, 1, 1)(1, 0, 1)[7]**, with_intercept = False, AICc = -16.420278433. D5 recorded optimizer convergence as False under maxiter = 50.

The exact same specification was refitted from a fresh initialisation at each iteration budget. Only `maxiter` was varied; no alternative D5 candidate was evaluated.

## Results

| maxiter | converged | iterations | AICc | delta vs D5 | max |gradient| |
|---|---|---|---|---|---|
| 50 | False | 50 | -16.420278433 | +0.000000000 | 0.0613 |
| 100 | False | 100 | -21.999390116 | -5.579111683 | 0.00369 |
| 200 | True | 127 | -22.342805900 | -5.922527468 | 4.69e-05 |
| 500 | True | 127 | -22.342805900 | -5.922527468 | 4.69e-05 |

## Interpretation boundary

These refits assess optimizer behaviour only. They do not introduce convergence as a retrospective D5 admissibility criterion and they do not replace the D5-selected model with a different candidate. Any downstream methodological decision must therefore be documented separately from the frozen D5 selection.

## Artifacts

d12_d5_convergence_diagnostic.json and this report. No residual-level proprietary output is written.
