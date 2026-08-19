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