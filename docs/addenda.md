# Protocol Addenda

Protocol frozen 19 August 2026 (`protocol-v1.0`). Each entry below records a
dated change or clarification, the reason for it, and whether it was prompted
by observed output. No addendum is admissible after the first test-window
forecast is produced (Section E2).

---

## Addendum 19 August 2026

**Rows affected:** D11, D12, D13 (execution order only; no frozen value changes)

**Status:** Filed before any step of Section D was executed. Not prompted by
observed output; no training results existed at the time of writing.

**Clarification.** The protocol lists the scale trigger (D11) after the
indicator-lag and inheritance steps (D8–D10), but does not state at which point
in execution the trigger is evaluated or which model's residuals it is applied
to. Because D12 applies log(y + 1) to the entire ladder, and a logarithmic
transformation can alter the outcome of the differencing tests (D2, D3) and the
joint annual-form and order selection (D5), the phrase "applied to the entire
ladder" would otherwise be ambiguous between re-estimating coefficients at the
existing specification and re-running the selection steps on the transformed
series.

**Resolution.** D11 is evaluated on the residuals of the M1 specification
selected at D5, immediately after D5 and before D7. If the trigger holds, D2,
D3 and D5 are re-run on log(y + 1) and the ladder proceeds from the resulting
specification; D7 through D10 are then executed once, on the transformed scale.
If the trigger does not hold, execution continues directly to D7 on the count
scale.

The scale decision is therefore still made once, on training diagnostics only,
and applies uniformly to every model in the ladder, as D13 requires. Execution
order for Section D is: D1, D2, D3, D4, D5, D11, (D12 if triggered, returning to
D2), D6 as applicable, D7, D8, D9, D10.

**Effect on frozen values.** None. No threshold, candidate set, rule or
criterion is altered.



## Addendum 20 August 2026

**Rows affected:** D8, D9 (clarification only; no frozen value changes)

**Status:** Filed before any step of Section D was executed. Not prompted by
observed output; no training results existed at the time of writing.

**Clarification.** D7 declares that the holiday pad is selected by AICc with the
M1 orders and annual form held fixed. D8 and D9 declare the two-stage
screen-then-select rule for the FX and search lags but do not state whether the
orders and annual form are likewise held fixed during the AICc stage, or
re-selected for each candidate lag as D5 does for the Fourier term count. The
distinction is material: if orders were re-selected with the candidate regressor
in the model, M3 and M4 could differ from M1 in ARIMA orders as well as in
regressors, and M1 would no longer be a special case of them. E8 and E9 assign
the Clark-West test to all four M-versus-M1 comparisons on the grounds that each
augmented model nests M1, and D10 transfers the pad and lags to M5 without
re-tuning on the same assumption.

**Resolution.** Indicator-lag AICc selection at D8 and D9 is performed with the
orders and annual form of the operative M1 specification held fixed, as already
declared for the pad at D7. "Operative" is the specification produced by D5 on
the scale in force after D11, per the addendum of 19 August 2026: on the count
scale if the trigger does not hold, on log(y + 1) if it does. Only the candidate
lag varies within each AICc comparison. M2 through M5 therefore differ from M1
in their exogenous regressors alone, and the nesting relied on at E8 and E9
holds by construction.

**Effect on frozen values.** None. No threshold, candidate set, rule or
criterion is altered. The candidate lag windows (D8: 28-90 days; D9: 28, 35,
..., 119), the five-candidate CCF screen, the AICc criterion and the
shorter-lag tie-break are unchanged.



## Addendum 23 August 2026

**Rows affected:** D4, D12 (execution order and inheritance clarification only; no frozen value changes)

**Status:** Filed before any test-window forecast was produced. Prompted by review of the implementation sequence, not by observed model output. The clarification addresses a branch that has not yet been executed and does not depend on the outcome of the D11 scale trigger or on any transformed-scale diagnostic result.

**Clarification.** The addendum of 19 August 2026 specifies that, if the D11 scale trigger holds, D2, D3 and D5 are re-run on log(y + 1). It does not explicitly state whether D4 is re-applied after the transformed-scale differencing orders are selected. D4 defines inclusion of a constant conditionally on those orders: a constant is included if and only if d = D = 0. Therefore, carrying forward the count-scale D4 decision independently of the transformed-scale D2 and D3 results could be inconsistent with the frozen D4 rule.

**Resolution.** If D11 triggers D12, D4 is mechanically re-applied after D2 and D3 are re-run on log(y + 1) and before D5 is re-run. The constant decision is determined from the transformed-scale differencing orders using the unchanged D4 rule: a constant is included if and only if d = D = 0. No count-scale value of d, D or the constant decision is inherited into the transformed-scale D5 run.

Accordingly, the D12 re-selection sequence is D2, D3, D4, D5 on log(y + 1). If the transformed-scale selections are d = D = 0, D5 is run with a constant; otherwise it is run without a constant. All other D5 candidate sets, restrictions, criteria and diagnostic gates remain unchanged.

The execution order for Section D is therefore: D1, D2, D3, D4, D5, D11, (D12 if triggered, returning to D2, D3, D4 and D5 on log(y + 1)), D6 as applicable, D7, D8, D9, D10.

**Effect on frozen values.** None. No threshold, candidate set, rule or criterion is altered. This addendum only makes explicit that D4, as a deterministic rule conditional on the selected differencing orders, is re-evaluated whenever D2 and D3 are re-run under D12.


## Addendum 24 August 2026

Numerical convergence treatment after transformed-scale D5

Following the D11 scale trigger, D2-D5 were re-run on log(y + 1) as required by the 19 August 2026 addendum and the 23 August 2026 D4 clarification.

**The transformed-scale D5 execution selected:**

annual form: Fourier K = 6, period 365.25;
ARIMA order: (1,1,1);
weekly seasonal order: (1,0,1)[7];
d = 1, D = 0;
no constant under D4;
AICc -16.420278432857387;
Ljung–Box p-value 0.5016850482519739.

The selected fit satisfied the frozen D5 Ljung–Box gate but did not report optimizer convergence under the implementation setting method='lbfgs', maxiter=50.

A training-only, diagnostic refit was therefore performed on the exact D5-selected specification, holding the response scale, annual form, ARIMA orders, seasonal orders, regressors, intercept setting, estimation sample, optimizer method and all other model settings fixed. Only the numerical iteration ceiling was varied. Fresh fits produced:

maxiter = 50   converged = False   AICc = -16.420278433
maxiter = 100  converged = False   AICc = -21.999390116
maxiter = 200  converged = True    AICc = -22.342805900
maxiter = 500  converged = True    AICc = -22.342805900

The 200- and 500-iteration fits converged to the same solution, with convergence reached after 127 iterations. The diagnostic therefore shows that the original non-convergence was attributable to the 50-iteration numerical ceiling for this specification rather than failure of the selected model to reach a stable optimum.

**Clarification.** The D5 model-selection result remains unchanged. The transformed-scale M1 structure selected at D5 remains Fourier K = 6, ARIMA (1,1,1)(1,0,1)[7], with d = 1, D = 0 and no constant. The original D5 artifacts and recorded AICc are retained unchanged as the historical output of the D5 selection execution. The converged diagnostic AICc is not substituted into the D5 selection artifact, and D5 is not re-run or re-ranked.

For downstream fixed-specification estimation beginning at D7, and for subsequent D8-D10 candidate fits and final model refits/forecast generation that inherit the D5-selected structure, the numerical optimizer setting is:

method = 'lbfgs'
maxiter = 500

All other frozen or previously declared model-selection rules remain unchanged. Increasing maxiter is treated solely as a numerical estimation safeguard and not as a new model-selection criterion, candidate restriction, convergence filter, or fallback rule.

Each downstream candidate fit is fitted under the same maxiter = 500 ceiling so that AICc comparisons within a decision row are made under a common numerical estimation setting. Convergence status is recorded for every fit.

If a downstream fit still fails to report convergence at maxiter = 500, it is not automatically excluded, replaced, or assigned a fallback model. Execution is paused and the numerical issue is documented and resolved before the affected decision row is finalized. No additional optimizer, iteration ceiling, candidate exclusion rule, or model-selection rule may be introduced silently.

This clarification was made before execution of D7 and uses training data only. No test-window observations or forecast-performance results were consulted.

No frozen thresholds, candidate sets, hypothesis tests, AICc selection rules, Ljung-Box rules, differencing decisions, annual-form decisions, or D4 rules are changed by this addendum.


## Addendum 25 August 2026

D8/D9 CCF identification clarification.

The frozen instruction "CCF identification per Box et al. (2016)" is operationalized using Box-Jenkins prewhitening. 

For each indicator, an auxiliary ARIMA filter is selected on the indicator values corresponding to the frozen 884-day training window only. The auxiliary search uses the D5 order bounds, weekly seasonal period s=7, stepwise AICc, d≤1, D≤1, KPSS for non-seasonal differencing, OCSB for seasonal differencing, and automatic intercept handling. 

The selected fixed filter is then applied unchanged to the extended indicator history and to the operative transformed response; no response-side ARIMA model is separately estimated for CCF identification. Initial state-space observations are excluded according to the model-reported likelihood burn. If the selected prewhitening filter fails to converge, execution pauses before lag screening. 

The five candidate lags are selected by the largest absolute prewhitened CCF values; raw CCF values are diagnostic only. This clarification does not alter the frozen candidate lag ranges, number of screened lags, Stage-2 AICc rule, or shorter-lag tie rule.


## Addendum 26 August 2026 

D9 search-vintage check and numerical conditioning.

This addendum is adopted prospectively before the operative D9 search-lag
selection. It does not alter the frozen D9 candidate set, screening rule,
AICc rule, tie rule, inherited M1 specification, or convergence policy.

### D9 C3-4 publication-vintage implementation

The frozen search-lag candidates remain exactly:

28, 35, 42, ..., 119 days.

The search indicator is weekly and step-expanded to daily observations.
C3-4 requires that a week's value enter only from its publication date and
that the observed publication delay be recorded.

For the implementation check, publication delay is measured in days from
the END of the represented week to the date on which that week's value
became available. A seven-day week ends six days after its first day.
Accordingly, the conservative all-weekday availability requirement is:

    lag >= 6 + observed publication delay

This availability check is separate from the frozen seven-day lag grid.
The value 6 does not change the lag increment or candidate set.

If any frozen candidate would violate the recorded publication-vintage
constraint, D9 pauses. The candidate set is not narrowed or altered
automatically.

### D9 Stage-2 search-regressor scaling

Stage-1 prewhitening and CCF screening use the original frozen
`search_index` values without rescaling.

Only after the five D9 candidate lags have been selected by the frozen CCF
screening rule, the search regressor used in the Stage-2 M4 AICc fits is
expressed in thousands of queries:

    search_index_scaled = search_index / 1000

This is a numerical-conditioning measure for the exogenous regression
coefficient. It is an exact reparameterisation:

    X* = X / 1000
    beta* = 1000 beta

so beta* X* = beta X.

At the same optimum this leaves fitted values, likelihood, parameter count,
AICc and forecasts unchanged. It changes only the numerical scale and
interpretation of the search coefficient.

Therefore this clarification changes none of the following:

- the D9 candidate lags 28, 35, ..., 119;
- the five-candidate prewhitened-CCF screening rule;
- the shorter-lag tie rule;
- the operative M1 orders or annual form;
- the D9 AICc selection criterion;
- the downstream convergence rule.

D10 inherits the D9-selected search lag without re-tuning and uses the same
Stage-2 search-regressor scaling when that inherited regressor is included.

Operative protocol state after this addendum: `protocol-v1.6`.



## Addendum 17 September 2026

Full re-execution of the transformed-scale D5 selection under maxiter = 500.

**Rows affected:** D5 (numerical re-execution of the transformed-scale
selection); D7-D10 conditionally, only if the re-executed selection differs
from the recorded one. No frozen candidate set, search bound, criterion,
gate, tie-break or threshold is altered.

**Status:** Filed before any test-window forecast was produced; Section E
has not been executed and no test-window observation or forecast-performance
result was consulted. Prompted by supervisor review (17 September 2026) of
the convergence documentation, i.e. by training-window numerical convergence
status only.

**Background.** The transformed-scale D5 execution ran the stepwise search
under method='lbfgs', maxiter = 50. The selected fit did not report
optimizer convergence at that ceiling, and non-convergence was not confined
to it: [56] of the [123] visited fits are flagged non-converged in the recorded
visited-fits table. Two training-only diagnostics followed: the 24 August
2026 exact-specification refit of the selected model at higher iteration
budgets, and the subsequent convergence-sensitivity refit of the seven
recorded per-candidate winner specifications at maxiter = 500. Neither
re-runs the search itself. An interrupted LBFGS fit records the
log-likelihood of an intermediate iterate of the same descent path, so its
recorded AICc is not smaller than the converged value for that
specification, as the 24 August table shows for the selected specification
(-16.42 recorded at maxiter = 50 against -22.34 converged). Non-convergence
at maxiter = 50 can therefore have distorted both the AICc values steering
each candidate's stepwise walk (which orders were visited at all) and the
AICc values compared across candidates. Refitting only recorded
specifications cannot confirm the original AICc ranking or the Fourier
K = 6 annual-form choice. The decision was made for the
D5 candidates to be re-estimated under the common 500-iteration setting.

**Resolution.**

1. The transformed-scale D5 selection is re-executed in full: the complete
Hyndman-Khandakar stepwise search for each of the seven frozen annual-form
candidates (monthly, fourier_K1 ... fourier_K6), on log(y + 1) with d = 1,
D = 0 and no constant per the recorded transformed-scale D2-D4 outcomes, on
the frozen 884-day training sample, under the D1 environment, with
method = 'lbfgs' and maxiter = 500 applied identically to every fit in the
search. maxiter is the only setting that differs from the recorded
transformed-scale execution. The visited-set Ljung-Box gate, the D6
fallback, the deterministic tie-break and every other frozen or previously
declared D5 rule apply unchanged.

2. The selection produced by this re-execution is the operative D5 result
and the operative M1 structure for all downstream rows. The provision of
the 24 August 2026 addendum that "D5 is not re-run or re-ranked" is
superseded on supervisor instruction. Every other provision of that
addendum remains in force: the recorded artifacts are retained unchanged as
historical record with no value substituted into them, the maxiter = 500
downstream ceiling stands, convergence status is recorded for every fit,
and a fit failing to report convergence at maxiter = 500 is not silently
excluded, replaced or assigned a fallback. That last rule extends to this
re-execution: if any visited fit, and in particular the selected fit, does
not report convergence at maxiter = 500, execution pauses and the numerical
issue is documented and resolved before D5 is finalized. No additional
optimizer, iteration ceiling, exclusion rule or selection rule may be
introduced silently.

3. Conditional consequences. If the re-executed selection reproduces the
recorded structure (fourier_K6, ARIMA(1,1,1)(1,0,1)[7], d = 1, D = 0, no
constant), the K = 6 choice and the AICc ranking are confirmed; the
completed D7-D10 rows, which were estimated at maxiter = 500 under that
inherited structure, stand without re-execution, and Section E proceeds. If
the re-executed selection differs in any element, it becomes the operative
M1 and D7-D10 are re-executed in protocol order under the unchanged frozen
rules with the new operative M1, before Section E. The D11 scale decision
is not reopened in either case: the trigger was evaluated per the
19 August 2026 addendum and the supervisor has accepted the log(y + 1)
switch as following the pre-specified rule. The count-scale D5 execution
is likewise not reopened; its artifacts remain the historical count-scale
record.

4. Artifacts. The re-execution writes the standard D5 artifact set
(visited-fits table with per-fit convergence status, per-candidate winners,
selection JSON, per-candidate stepwise traces, run report) to a separate
directory, results/d5_log1p_maxiter500/, leaving the recorded artifacts
untouched, and is logged to the project MLflow store with Git state. The
input data SHA-256 must equal the digest recorded in the operative
transformed-scale d5_selection.json. The re-executed per-candidate winner
table, its comparison to the recorded 24 August table, and per-fit
convergence counts are reported in the thesis.

5. Finality. Once the re-execution (and, if triggered, the conditional
D7-D10 re-execution) is complete and documented, the model selection is
final and Section E proceeds; no further addendum may alter the selection
thereafter, per the Section E2 admissibility rule.

**Effect on frozen values.** None. The candidate set, search bounds,
information criterion, Ljung-Box gate and lag rule, D6 fallback, tie-break,
training window and scale rules are unchanged. maxiter is a numerical
estimation setting, recorded as an implementation choice rather than a
frozen threshold; raising it to the previously declared common ceiling of
500 for this re-execution is, per the 24 August 2026 addendum, a numerical
estimation safeguard and not a new model-selection criterion, candidate
restriction, convergence filter or fallback rule.

Operative protocol state after this addendum: `protocol-v1.7`.


## Addendum 21 September 2026

Rows affected: E2 and E11; E10 and E15 implementation clarified only; M0 leap-day benchmark convention clarified for E7/E9 implementation only.

**Status:** Filed before any test-window forecast was produced and before any test-window forecast-error or performance result was inspected. The issue was identified from the deterministic calendar geometry of the frozen evaluation design: the original seven-day origin spacing began on Sunday, and all declared horizons (7, 14 and 28 days) are multiples of seven. Consequently, every evaluated target at every horizon fell on the same day of the week. The seven-day origin spacing had originally been chosen with an anticipated weekly production forecasting cycle in mind. For the purposes of this thesis, however, advancing the forecast origin by one day is more appropriate because the objective is to compare model performance across all admissible rolling origins in the out-of-sample evaluation period rather than to evaluate a specific weekly operational forecasting schedule. Daily origin advancement also prevents the evaluation from being restricted to a single weekday.

**E2 resolution — origin frequency.** The expanding-window design is retained, but forecast origins are changed from every 7 days to **every 1 day**, beginning on 30 November 2025 and continuing while `origin + 28 days <= 31 July 2026`. This yields **216 origins**, from 30 November 2025 through 3 July 2026 inclusive. The forecast horizons remain unchanged at **h = 7, 14 and 28 days**. Because origins now cover all weekdays, each horizon is evaluated on all seven days of the week rather than on Sundays only. No orders, annual form, holiday pad, indicator lag, scale decision, model specification or test-window outcome was used to make this change.

**E10 clarification — HLN horizon units.** The Harvey-Leybourne-Newbold small-sample correction is evaluated in units of the forecast-origin sequence. With daily origins, the effective horizon is therefore `k = h` for h = 7, 14 and 28, rather than `ceil(h/7)` under the former weekly origin grid. The one-sided alternative and t reference with `n-1` degrees of freedom are unchanged.

**E11 resolution — HAC truncation.** The Bartlett-kernel Newey-West truncation is changed from `ceil(h/7) - 1` to **`h - 1`**. This is the direct consequence of changing the origin spacing from seven days to one day: under the declared overlapping-forecast-error convention, an h-day-ahead forecast-error sequence may overlap through h-1 daily lags. Thus the truncations are 6, 13 and 27 for h = 7, 14 and 28 respectively.

**M0 benchmark clarification — leap day.** The existing M0 definition remains the same-calendar-date-previous-year seasonal-naïve benchmark. For a target date of 29 February, where no same calendar date exists in the preceding non-leap year, the M0 forecast is defined as the arithmetic mean of the observed counts on 28 February and 1 March of the preceding year. This convention also applies wherever M0 is used to construct the E7 MASE denominator. It does not affect the present evaluation because neither the declared MASE scaling interval nor the test-window forecast targets contain 29 February.

**E15 clarification — VIF computation.** E15 remains descriptive only and does not alter any model. VIFs for M2–M5 are computed on the frozen training-sample exogenous design. Because the forecasting models contain no constant, a diagnostic-only constant is added when calculating VIFs so that the reported quantities are conventional centered VIFs. This constant is used solely for the VIF calculation and is not included in any forecasting model.

**Unchanged rows.** E1 remains an expanding window. E3 still re-estimates coefficients only at every origin. E4 still prohibits re-selection of orders, annual form, holiday pad or indicator lags. E5 remains h = 7, 14 and 28. E6 remains satisfied because the holiday regressors are deterministic and the selected indicator lags (FX 84 days; search 42 days) exceed the 28-day maximum horizon. E7–E9 and E12–E15 remain substantively unchanged, subject to the M0 and E15 implementation clarifications stated above. Section D and the operative v1.7 D5-D10 model specifications are unchanged.

**Reason for amendment.** This change removes an unintended weekday-sampling restriction while preserving the test window, forecast horizons, expanding-window principle, model ladder and confirmatory hypotheses. It is a design correction made prospectively, without reference to test-window forecast performance. The move from weekly to daily origins increases the number of forecast origins from 31 to 216 but does **not** create 216 independent pieces of test information: the test window is unchanged and multi-step forecast errors from adjacent daily origins overlap. The `h - 1` Bartlett/Newey-West truncation is retained specifically to account for this serial dependence in the estimated variance. Nevertheless, because the inferential reference remains a finite-sample t approximation with `n - 1` degrees of freedom, residual size distortion may remain when the overlap is substantial, especially at `h = 28`. Accordingly, the 216 origins are treated as a denser evaluation grid rather than 216 independent replications, and longer-horizon confirmatory p-values will be interpreted cautiously alongside the descriptive accuracy measures.

**Effect on model selection:** None. The operative Section D specification remains final. The only substantive changes made by this addendum are to the Section E rolling-origin evaluation grid and the dependence correction required by that grid; the M0 and E15 provisions above are implementation clarifications only.

Operative protocol state after this addendum: `protocol-v1.8`.
