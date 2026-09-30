# D7 run report -- holiday pad (b, f)

Protocol state protocol-v1.4 (freeze tag protocol-v1.0), row D7. Run (UTC): 2026-08-24T12:03:36+00:00.

Operative M1, read from `d5_selection.json` and rebuilt rather than retyped: **fourier_K6, ARIMA(1, 1, 1)(1, 0, 1)[7], no constant**, on the log1p scale. The annual form was regenerated from the recorded kind/K/period/origin and matched column-for-column.

Holiday regressors follow Section C: a single (b, f) pair shared by both groups (C1-9), so the declared set {2,3,4} x {2,3,4} gives nine pads rather than 81; H_NY and H_OT enter as two binary columns and carry separate coefficients; windows are unions and are never summed (C1-10). The window is [start - b, end + f] inclusive, so each cluster's own non-working days are always inside it. Cluster file validated: 17 clusters, 2 H_NY / 9 H_OT in training, 1 H_NY / 5 H_OT in test, every run at least 3 days, and H_NY exactly the clusters containing 1 January.

## Pad grid (AICc ascending)

| b | f | AICc | H_NY days | H_OT days | converged | iters |
|---|---|---|---|---|---|---|
| 3 | 2 | -20.0938 | 31 | 75 | yes | 139 |
| 3 | 4 | -19.9613 | 35 | 91 | yes | 136 |
| 3 | 3 | -19.4653 | 33 | 83 | yes | 178 |
| 2 | 2 | -19.2152 | 29 | 67 | yes | 143 |
| 2 | 4 | -18.9967 | 33 | 83 | yes | 134 |
| 2 | 3 | -18.6745 | 31 | 75 | yes | 171 |
| 4 | 4 | -18.5710 | 37 | 98 | yes | 130 |
| 4 | 2 | -18.5446 | 33 | 83 | yes | 183 |
| 4 | 3 | -18.4712 | 35 | 91 | yes | 191 |

All nine fits share the estimation sample (883 effective observations) and the same regressor count (14), so their AICc values are comparable; both were checked, not assumed.

At the selected pad, the union rule absorbed 0 H_NY and 2 H_OT day(s) that a summed construction would otherwise have double-counted.

## Selection

**(b, f) = (3, 2)**, AICc -20.0938, ahead of the runner-up by 0.1325. The padded windows cover 31 training days for H_NY and 75 for H_OT.

## Numerical estimation

Per the addendum of 24 August 2026, all nine candidates were fitted under a common ceiling (method='lbfgs', maxiter=500) so that within-row AICc comparisons are made under one numerical setting, and convergence was recorded for every fit. All nine converged. Had any not, execution would have paused with no pad selected, rather than the fit being excluded or replaced.

Diagnostic reference: M1 without holiday regressors, estimated under the same ceiling, gives AICc -22.3428. This is comparable to the nine values above but **not** to the AICc recorded at D5, which was produced under the frozen maxiter = 50. It is reported for context only: D7 selects among the nine pads, and the frozen row does not make the pad conditional on improving over M1.

## Next step

M2 = M1 + hol_H_NY + hol_H_OT at (b, f) = (3, 2); D10 transfers this pad to M5 without re-tuning.

## Artifacts

d7_pad_grid.csv (all nine pads, AICc ascending), d7_pad_selection.json (machine-readable pad for M2 and for the D10 inheritance), this report. No proprietary artifact is produced: the windows are calendar-derived and the table carries only AICc values and day counts.
