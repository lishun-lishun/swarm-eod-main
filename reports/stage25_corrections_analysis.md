# Stage 2.5 braking corrections

Date: 2026-11-12. The hard boundary was unchanged in every run.

The latched target inset was `3.305 px` = position allowance + microstep stopping distance of the velocity allowance + fixed margin + 2 px reserve.

## A-E legal-sensor results (10 seeds × 20/100/500 px)

| Stage / target | Violations | Max violation | Completion | Mean frame | Original-target error | Reaccelerations |
|---|---:|---:|---:|---:|---:|---:|
| A_original/raw | 4/30 | 1.174 px | 96.7% | 137.48275862068965 | 4.431 px | 138 |
| A_original/inset | 4/30 | 1.174 px | 96.7% | 137.48275862068965 | 4.431 px | 138 |
| B_safe_target/raw | 4/30 | 1.174 px | 96.7% | 137.48275862068965 | 4.431 px | 138 |
| B_safe_target/inset | 3/30 | 0.394 px | 100.0% | 59.86666666666667 | 6.240 px | 11 |
| C_exact_excursion/raw | 4/30 | 1.174 px | 96.7% | 137.48275862068965 | 4.431 px | 138 |
| C_exact_excursion/inset | 3/30 | 0.394 px | 100.0% | 59.86666666666667 | 6.240 px | 11 |
| D_history_reset/raw | 4/30 | 1.174 px | 96.7% | 137.48275862068965 | 4.431 px | 138 |
| D_history_reset/inset | 3/30 | 0.394 px | 100.0% | 59.86666666666667 | 6.240 px | 11 |
| E_dual_state/raw | 4/30 | 1.174 px | 96.7% | 138.72413793103448 | 4.248 px | 138 |
| E_dual_state/inset | 3/30 | 0.394 px | 100.0% | 59.86666666666667 | 6.240 px | 11 |

## Findings

The raw-target rows isolate braking/state changes from target inset. The inset rows show the additional feasibility benefit but retain the original target error, so inset is not credited as a braking-formula improvement.

Final E/raw: 4/30 violations, 96.7% completion, 138 reaccelerations.
Final E/inset: 3/30 violations, 100.0% completion, 11 reaccelerations.

Boundary-normal switch without reset limited the new normal thrust to 0.669; with reset it was 1.000. This verifies that stale recovery history no longer contaminates a new task.

The exact-excursion correction removes the mathematical double count and its residual is monotone in candidate thrust. In this fixed-seed suite it did not change the raw-target violation count: the active high-speed constraint branches selected the same safe thrusts. It is therefore not credited with the improvement produced by target feasibility.

## Seed 5 / Seed 9 audit at 500 px (final E stage)

| Seed / target | Max violation | Completion frame | Reaccelerations | Infeasible frames | Position-estimation RMSE |
|---|---:|---:|---:|---:|---:|
| Seed 5 / raw | 0.000 px | 112 | 4 | 5 | 8.621 px |
| Seed 5 / inset | 0.000 px | 91 | 0 | 5 | 8.830 px |
| Seed 9 / raw | 0.630 px | 179 | 5 | 8 | 4.799 px |
| Seed 9 / inset | 0.110 px | 96 | 1 | 6 | 4.624 px |

Seed 5 remains inside the unchanged hard boundary. Seed 9 improves substantially but still crosses it by 0.110 px with the 3.305 px inset. Its trace shows the controller satisfying the estimated-state residual while the true boundary-normal position differs from the estimate, so this residual violation is classified as localization error along the boundary normal, not as evidence to silently move the boundary or blindly increase the braking margin.

Seed 5 and Seed 9 full traces, including true/estimated state errors and safety residuals, are stored in `seed_5_9_focus`.

Remaining legal-sensor violations, if any, are attributed in the JSON to boundary-normal position-estimation error; no LiDAR, map truth, or hidden state is used by the controller.
