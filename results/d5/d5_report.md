# D5 run report -- annual form and baseline ARIMA orders

Protocol state protocol-v1.3 (freeze tag protocol-v1.0; environment per D1 as frozen), rows D5 (selection) and D6 (fallback). Run (UTC): 2026-08-23T10:00:56+00:00.

Input: `..\data\aligned_train.csv`; SHA-256 `4224fe90606f4a77e71dafadb99538a8ac878555a32805a48219f27cf7809313`.
Estimation sample: training only, through 2025-11-30 (n = 884). Effective observations after state-space burn-in: 883, identical across all visited fits -- the D5 AICc-comparability requirement was checked, not assumed.
Training echo (raw counts): median 29, min 2, max 83, variance/mean 6.86 (frozen reference: 29, 2, 83, 6.86).

## Gate and selection rule as executed

The Hyndman-Khandakar stepwise search ran unconstrained per candidate (pmdarima auto_arima, stepwise, IC = AICc, d = 1, D = 0, m = 7, no intercept). return_valid_fits=True returned every model the walk fitted successfully; the Ljung-Box gate (lag L = max(2s, k_arma+3) capped at floor(T/5), evaluated per fitted model and observed here as 14; df = L - k_arma with k_arma = p+q+P+Q; alpha = 0.05; pass iff p > alpha; computed on each fit's own residuals after dropping the burn-in of 1 observation(s)) was then applied to that entire visited pool, per the D5 wording and the D6 phrase 'no candidate visited'. Selected: lowest-AICc visited fit that passes. D6 fallback: lowest-AICc visited fit overall, failure recorded. Fits that errored or were rejected by pmdarima's root check are absent from the pool (no residuals, no AICc; not selectable) and remain visible in the per-candidate trace files.

## Recorded implementation choices

1. D4 removes the constant because d and D are not both zero. The monthly candidate therefore uses all 12 indicators; the D5 reference-month clause is inactive.
2. AICc comparability guard: nobs_effective asserted identical across all visited fits (observed value: 883).
3. Ljung-Box applied to each candidate fit's own post-burn residuals, never to residuals from any other row's model.
4. The Ljung-Box lag follows the frozen formula per fitted model, L = max(2s, k_arma+3) capped at floor(T/5); it is not hard-coded. L = 14 while k_arma <= 11 and rises to 17 at the (5,.,5)(2,.,2) search corner, so df >= 3 always and no fit is untestable. The cap, floor(T/5) with T = 884 training observations = 176, is non-binding. Observed L: 14.
5. Deterministic tie-break: AICc, then total estimated parameter count, then fixed candidate order (monthly, fourier_K1, fourier_K2, fourier_K3, fourier_K4, fourier_K5, fourier_K6), then (p, q, P, Q).
6. error_action='ignore'; optimizer convergence logged per fit; maxiter = 50 (pmdarima default).

## Selection

Selected: **fourier_K1, ARIMA(1,1,1)(1,0,0)[7], no intercept** -- lowest AICc (6006.90) among visited fits passing Ljung-Box at lag 14 (p = 0.9115, df = 11, alpha = 0.05).

## Per-candidate stepwise winners (D5: 'record the resulting AICc')

| candidate | (p,q)(P,Q) | AICc | LB p (df) | LB pass |
|---|---|---|---|---|
| monthly | (0,2)(1,1) | 6032.55 | 0.7530 (10) | yes |
| fourier_K1 | (1,1)(1,0) | 6006.90 | 0.9115 (11) | yes |
| fourier_K2 | (1,1)(1,0) | 6010.25 | 0.9097 (11) | yes |
| fourier_K3 | (1,1)(0,1) | 6011.50 | 0.8986 (11) | yes |
| fourier_K4 | (1,1)(0,1) | 6014.38 | 0.8908 (11) | yes |
| fourier_K5 | (1,2)(1,0) | 6013.20 | 0.1635 (10) | yes |
| fourier_K6 | (1,2)(1,1) | 6011.96 | 0.0697 (9) | yes |

## Counts

Candidates: 7. Visited valid fits (pooled): 199. LB-passing: 172. Optimizer non-converged: 123.

## Artifacts

d5_visited_fits.csv (full gate table, AICc ascending), d5_candidate_winners.csv, d5_selection.json (machine-readable spec for downstream rows, incl. the Fourier origin), trace_<candidate>.txt (raw stepwise traces), this report.
