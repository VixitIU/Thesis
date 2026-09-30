# Section E run report -- out-of-sample evaluation

Protocol state protocol-v1.8 (freeze tag protocol-v1.0), row E. Run (UTC): 2026-09-21T06:51:08+00:00.

Model selection was final before this run. E1-E15 re-estimate coefficients only; no order, annual-form, holiday-pad or lag selection is performed here.

All M1-M5 point forecasts were produced on log(y+1) and returned to the original count scale with the frozen bias-adjusted inverse exp(m_h + s_h^2/2) - 1. All descriptive metrics and confirmatory squared-error losses below therefore use count-scale forecasts.

## Accuracy

MAE is the primary descriptive metric (reported first).

| h | model | MAE | RMSE | MASE | MAPE | MdAPE |
|---:|---|---:|---:|---:|---:|---:|
| 7 | M0 | 9.7176 | 12.3515 | 0.7803 | 38.0862 | 26.4706 |
| 7 | M1 | 6.1672 | 7.8294 | 0.4952 | 23.9901 | 18.6748 |
| 7 | M2 | 6.3290 | 8.1473 | 0.5082 | 24.5870 | 19.6823 |
| 7 | M3 | 6.2442 | 7.9560 | 0.5014 | 24.3266 | 19.4329 |
| 7 | M4 | 6.2262 | 7.8726 | 0.4999 | 24.2102 | 18.8206 |
| 7 | M5 | 6.5495 | 8.4593 | 0.5259 | 25.3646 | 20.5058 |
| 14 | M0 | 9.6852 | 12.3472 | 0.7777 | 38.6492 | 27.5240 |
| 14 | M1 | 6.1403 | 8.1602 | 0.4931 | 24.8260 | 18.7872 |
| 14 | M2 | 6.4836 | 8.6048 | 0.5206 | 26.0361 | 19.7506 |
| 14 | M3 | 6.4296 | 8.5757 | 0.5163 | 25.6831 | 19.1546 |
| 14 | M4 | 6.2468 | 8.2699 | 0.5016 | 25.1641 | 18.8797 |
| 14 | M5 | 7.0084 | 9.4294 | 0.5628 | 27.5767 | 20.7559 |
| 28 | M0 | 9.1667 | 11.8388 | 0.7361 | 38.8021 | 26.5686 |
| 28 | M1 | 6.2186 | 8.5309 | 0.4993 | 26.5160 | 17.7017 |
| 28 | M2 | 6.5787 | 9.3691 | 0.5283 | 27.6321 | 18.4154 |
| 28 | M3 | 6.8312 | 9.6204 | 0.5485 | 28.5207 | 21.5205 |
| 28 | M4 | 6.3962 | 8.7341 | 0.5136 | 26.9583 | 18.1896 |
| 28 | M5 | 7.5046 | 11.1586 | 0.6026 | 30.4528 | 21.9914 |

## Confirmatory tests

Positive statistics favour the named alternative model. Clark-West H1a-H1c p-values are Holm-adjusted within each horizon; H2 is standalone at 5%. The M1-vs-M0 modified DM comparison is reported separately.

| h | hypothesis | comparison | test | stat | p raw | p Holm | decision basis |
|---:|---|---|---|---:|---:|---:|---|
| 7 | H1a_holidays | M2 vs M1 | Clark-West | -1.6581 | 0.950620 | 1.000000 | Holm 5%: not reject |
| 7 | H1b_fx | M3 vs M1 | Clark-West | -0.5374 | 0.704229 | 1.000000 | Holm 5%: not reject |
| 7 | H1c_search | M4 vs M1 | Clark-West | -0.7290 | 0.766590 | 1.000000 | Holm 5%: not reject |
| 7 | H2_combined | M5 vs M1 | Clark-West | -1.4045 | 0.919200 | n/a | standalone 5%: not reject |
| 7 | benchmark_M1_vs_M0 | M1 vs M0 | modified Diebold-Mariano (HLN) | 4.0940 | 0.000030 | n/a | standalone descriptive benchmark test: reject |
| 14 | H1a_holidays | M2 vs M1 | Clark-West | -1.4250 | 0.922193 | 1.000000 | Holm 5%: not reject |
| 14 | H1b_fx | M3 vs M1 | Clark-West | -1.0666 | 0.856322 | 1.000000 | Holm 5%: not reject |
| 14 | H1c_search | M4 vs M1 | Clark-West | -1.4523 | 0.926059 | 1.000000 | Holm 5%: not reject |
| 14 | H2_combined | M5 vs M1 | Clark-West | -1.3709 | 0.914084 | n/a | standalone 5%: not reject |
| 14 | benchmark_M1_vs_M0 | M1 vs M0 | modified Diebold-Mariano (HLN) | 3.1369 | 0.000973 | n/a | standalone descriptive benchmark test: reject |
| 28 | H1a_holidays | M2 vs M1 | Clark-West | -1.0295 | 0.847804 | 1.000000 | Holm 5%: not reject |
| 28 | H1b_fx | M3 vs M1 | Clark-West | -1.1903 | 0.882376 | 1.000000 | Holm 5%: not reject |
| 28 | H1c_search | M4 vs M1 | Clark-West | -1.1600 | 0.876323 | 1.000000 | Holm 5%: not reject |
| 28 | H2_combined | M5 vs M1 | Clark-West | -1.1121 | 0.866338 | n/a | standalone 5%: not reject |
| 28 | benchmark_M1_vs_M0 | M1 vs M0 | modified Diebold-Mariano (HLN) | 3.3537 | 0.000471 | n/a | standalone descriptive benchmark test: reject |

## Dependence and finite-sample caveat

The 216 daily origins do not represent 216 independent forecast experiments. The test window is unchanged and adjacent h-step forecast errors overlap. The declared Bartlett/Newey-West HAC truncation of h-1 adjusts the variance estimate for this serial dependence, and the HLN correction is applied in daily-origin units. However, at longer horizons -- especially h=28 -- the HAC bandwidth is large relative to the finite evaluation window, so finite-sample size distortion may remain. Confirmatory p-values are therefore interpreted cautiously and together with the descriptive accuracy measures; n=216 is not described as 216 independent replications.

## Interpretation guard

A non-significant holiday comparison means that the frozen holiday augmentation did not demonstrate incremental out-of-sample predictive improvement conditional on M1. It does not establish that holidays have no effect, because holiday timing can overlap with the deterministic annual seasonal component and the training sample spans fewer than two and a half annual cycles.
