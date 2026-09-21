# D12/D3 run report -- weekly seasonal differencing on log(y + 1)

Protocol state protocol-v1.3 (freeze tag protocol-v1.0), row D3 re-run on the transformed scale under D12. Run (UTC): 2026-08-24T06:50:46+00:00.

Authorised by transformed D2 (`C:\Users\New\Documents\IU work\Thesis\project\results\d12_d2\d12_d2_differencing.json`), which selected d = 1 on log(y + 1). Input `..\data\aligned_train.csv`, SHA-256 `4224fe90606f4a77e71dafadb99538a8ac878555a32805a48219f27cf7809313`, matching the training extraction used by D2.

## OCSB test

The frozen D3 rule applies the OCSB test at the weekly period s = 7 using pmdarima. The test is applied directly to log(y + 1); annual seasonality remains deterministic and is never differenced.

| statistic | critical value | rejects | selected D |
|---|---|---|---|
| -23.862928736 | -1.845236487 | yes | 0 |

Decision rule: **D = 1 if OCSB does not reject; otherwise D = 0.**

## D4 implication

Transformed-scale D2 selected d = 1 and D3 selected D = 0. Under D4, a constant is included iff d = D = 0; therefore **with_intercept = False**.

## Implementation choice

The OCSB lag-selection method is `aic`. This was not fixed by the frozen D3 row and is retained from the original D3 implementation rather than changed after observing the transformed-scale series.

## Next step

D4 outcome is with_intercept = False. Re-run D5 on log(y+1) with --scale log1p --d 1 --D 0.

## Artifacts

d12_d3_seasonal_differencing.json (machine-readable transformed D3 decision), this report. No residual-level proprietary artifact is produced by D3.
