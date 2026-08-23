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