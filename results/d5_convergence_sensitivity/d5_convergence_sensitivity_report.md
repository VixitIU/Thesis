# D5 convergence sensitivity diagnostic

**NON-DECISIONAL, TRAINING-ONLY sensitivity analysis.** No `auto_arima` search was rerun, no test-window observation was accessed, and no Section D model-design choice is changed by this diagnostic.

Source D5 artifact: `d5_selection.json` (`protocol-v1.3`, SHA-256 `676081af1e6345394d80e67924fc54acf276cbcfcf9afde40797bf3f15d179e1`).
Candidate-winner artifact: `d5_candidate_winners.csv` (SHA-256 `3fc4516a16ad2a6c8cf185c81aee0890d2ade06f15aa942a94af9608adcbe861`).
Training data SHA-256 `4224fe90606f4a77e71dafadb99538a8ac878555a32805a48219f27cf7809313`; scale `log1p`; d = 1, D = 0, constant = False.

## Question

Holding each of the seven D5 per-candidate winner specifications exactly fixed, does the annual-form AICc ranking change when those same specifications are refitted with `method='lbfgs'`, `maxiter=500`?

This deliberately does **not** refit the entire historical visited pool and does not run a new Hyndman-Khandakar search. It isolates optimizer convergence for the seven recorded candidate winners.

## Results

| candidate | fixed orders | AICc @50 | conv @50 | AICc @500 | shift | conv @500 | iters | LB @500 | rank 50→500 |
|---|---|---:|---|---:|---:|---|---:|---|---:|
| fourier_K6 | (1, 1, 1) (1, 0, 1, 7) | -16.420278 | no | -22.342806 | -5.922527 | yes | 127 | pass | 1→1 |
| fourier_K3 | (1, 1, 1) (1, 0, 2, 7) | -16.211092 | no | -20.785885 | -4.574793 | yes | 124 | pass | 2→3 |
| fourier_K4 | (1, 1, 1) (1, 0, 1, 7) | -15.940971 | no | -21.800334 | -5.859363 | yes | 159 | pass | 3→2 |
| fourier_K1 | (0, 1, 2) (0, 0, 0, 7) | -11.793988 | yes | -11.793988 | +0.000000 | yes | 19 | pass | 4→5 |
| fourier_K5 | (2, 1, 2) (1, 0, 1, 7) | -10.956032 | no | -16.265875 | -5.309843 | yes | 242 | pass | 5→4 |
| fourier_K2 | (0, 1, 2) (0, 0, 0, 7) | -8.223448 | yes | -8.223448 | +0.000000 | yes | 18 | pass | 6→6 |
| monthly | (3, 1, 2) (1, 0, 1, 7) | 3.002077 | no | 1.813202 | -1.188875 | yes | 209 | pass | 7→7 |

## Interpretation

All seven exact specifications converged. The unrestricted AICc winner remained **fourier_K6** after the common 500-iteration refit. This statement concerns AICc ordering only; the frozen D5 rule also requires the Ljung-Box gate.
The candidate selected by frozen D5 (**fourier_K6**) is also the lowest-AICc candidate among these seven converged exact refits.
Ljung-Box gate: no candidate's outcome changed between the two ceilings. Within the seven recorded candidate-winner specifications, applying the frozen D5 gate/D6-fallback rule would choose **fourier_K6** at maxiter=50 (D5_gate; 7 passing candidates), and **fourier_K6** at maxiter=500 (D5_gate; 7 passing candidates). The seven-winner rule choice therefore SURVIVES. This remains a subset diagnostic: frozen D5 itself selected from the full visited-fit pool, not only these seven rows.

The full frozen D5 selection was made from the entire visited pool after the Ljung-Box gate, whereas this diagnostic concerns only the seven recorded per-candidate winners. It must therefore not be described as a complete maxiter=500 rerun of D5.
