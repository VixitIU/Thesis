# D10 run report -- M5 inheritance

Protocol state protocol-v1.6 (freeze tag protocol-v1.0), row D10. Run (UTC): 2026-08-26T04:40:15+00:00.

Operative addendum: `addenda.md` (SHA-256 `157c96f13ff8341f1336f616fec0d95312a02b2292fdcbc9a1fcb4d8071f5c2e`).

**D10 selects nothing.** It assembles the three already-selected indicator settings into one specification and verifies that they are mutually consistent. M5 inherits the pad (3, 2) from M2, the FX lag of 84 days from M3 and the search lag of 42 days from M4, with no re-tuning: the ARIMA orders, seasonal orders, annual form and constant decision are those of the operative D5 selection, and no alternative to any inherited setting was fitted.

## Consistency checks

E8 assigns the Clark-West test to M2, M3, M4 and M5 against M1 on the grounds that each augmented model nests M1. That holds only if every augmented model differs from M1 in its exogenous regressors alone. Verified here rather than assumed: D7, D8 and D9 all describe the same operative M1 (**fourier_K6, ARIMA(1, 1, 1)(1, 0, 1)[7], no constant**, log1p scale), all ran on training extraction `4224fe90606f4a77`, D8 and D9 on indicator history `ae994353fa0310d2`, D7 on cluster file `7dfea1a3b481fc65`, and none was a smoke run.

## M5 specification

| component | value |
|---|---|
| ARIMA order | (1, 1, 1) |
| seasonal order | (1, 0, 1)[7] |
| constant | none (D4) |
| annual form | fourier_K6 (12 columns) |
| holiday pad (b, f) | (3, 2) -- 31 H_NY days, 75 H_OT days |
| FX | lag 84 days, unscaled |
| search | lag 42 days, per 1000 queries |
| exogenous columns | 16 |

## Training ladder (AICc)

| model | description | AICc | source |
|---|---|---|---|
| M1 | baseline, annual form only | -22.3428 | D7 diagnostic refit |
| M2 | M1 + holidays at pad (3, 2) | -20.0938 | D7 |
| M3 | M1 + FX at lag 84 | -28.8097 | D8 |
| M4 | M1 + search at lag 42 | -23.5410 | D9 |
| M5 | M1 + all three indicators | -27.4487 | D10 |

All five values come from fits of the same specification family on the same training sample under the same optimizer ceiling, so they are comparable with one another. They are descriptive only. No D10 result, training AICc, or later evaluation is allowed to alter the inherited M5 design; confirmatory model comparisons occur out of sample in Section E.

## Next step

Section D is complete. `d10_m5_specification.json` is the authoritative description of M5 for Section E, which re-estimates coefficients only (E3) and never re-selects orders, annual form, pad or lags (E4).
