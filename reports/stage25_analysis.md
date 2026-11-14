# Stage 2.5: nonlinear one-step predictive braking

Date: 2026-11-12. Scope is limited to virtual hard-boundary stopping; no
LiDAR, path planning, frontier exploration, or multi-drone logic is present.

## Model validation

The recursive and closed-form frame-level stopping distances agree to
`2.842e-14` px. Against official
Pymunk, the frame-end model's maximum error is
`0.0503` px because the last frame can move
forward and then reverse between endpoints. The implemented ten-microstep
recurrence reduces the maximum error to
`9.948e-14` px and is used by the safety
constraint.

## Why the legacy controller crosses the boundary

At 500 px, legacy truth input crossed by
`9.709` px and legacy legal input by
`11.442` px. The current-distance-only
variant still crossed by `9.664`
px with truth input. It waits until the present stopping distance reaches the
remaining gap and omits the displacement and velocity produced by the current
candidate action.

The one-step binary and bisection variants had zero truth-input boundary
violation. Bisection retained the largest feasible normal thrust instead of
switching directly between cruise and full reverse.

## Legal-sensor robustness

Ten seeds at 20, 100, and 500 px were run for each margin configuration. The
selected configuration is **balanced**, chosen lexicographically by violation
count, completion rate, then mean speed.

| Configuration | Violations / 30 | Maximum violation | Completion rate | Mean completion frame | Mean final error | Mean speed | Reaccelerations |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 16 | 3.841 px | 76.7% | 224.1304347826087 | 4.572 px | 1.435 | 36 |
| light | 7 | 1.431 px | 100.0% | 152 | 4.534 px | 1.721 | 81 |
| balanced | 4 | 1.174 px | 96.7% | 137.48275862068965 | 4.431 px | 1.756 | 138 |
| conservative | 8 | 1.236 px | 50.0% | 257.8666666666667 | 4.385 px | 1.046 | 496 |
| truth_reference | 0 | 0.000 px | 100.0% | 61.333333333333336 | 0.000 px | 2.455 | 0 |


The selected balanced margin reduces violations substantially but does not
eliminate them: 4 of 30 legal-sensor runs still crossed the boundary, with a
1.174 px worst case. This is an empirical risk reduction, not a zero-collision
guarantee. The approximately 4.4 px mean final Euclidean error includes lateral
GPS/heading drift; boundary-normal violation is reported separately.

## Initial-speed feasibility

For every tested speed (2, 4, 6, 8, 10, and 11.46 px/frame), states beginning
at the microstep stopping distance or with 2/20 px extra clearance stopped
without crossing. States initialized 5 px inside the required stopping distance
were marked `INFEASIBLE_BRAKING` on the first frame and crossed by approximately
5 px despite immediate full reverse. The controller does not label these
physically impossible cases safe.

## Single-seed A-E comparison

| Variant | Truth violation | Legal violation | Truth completion | Legal completion |
|---|---:|---:|---:|---:|
| A_legacy | 9.709 px | 11.442 px | 92 | 93 |
| B_current_threshold | 9.664 px | 11.287 px | 98 | 133 |
| C_one_step_binary | 0.000 px | 1.727 px | None | None |
| D_bisection | 0.000 px | 2.168 px | 94 | None |
| E_robust | 0.000 px | 1.045 px | 94 | 163 |


These margins are empirical bounds, not an independent-Gaussian safety proof.
Every violation remains in the JSON with estimation and decision traces.
States that begin closer than the microstep stopping distance are explicitly
reported as `INFEASIBLE_BRAKING`; they are not counted as a controller promise.

## Two-dimensional allocation and real-time cost

The tests include x, y/lateral, 45-degree, continuously rotating diagonal, and
tangent-preservation cases. Normal safety thrust is allocated first and the
tangent command is clipped to `sqrt(1-u_n^2)`, so no run exceeds the official
unit circle. All five truth-input tests had zero boundary violation. The x, y,
diagonal, and rotating-diagonal tests completed in 51 frames. Tangent
preservation completed in 152 frames with 0.0021 px final error while
respecting the normal boundary.

The selected configuration's worst per-run solver P99 is
`354.00` microseconds and the maximum observed
single call is `435.50` microseconds. The solver uses
at most 20 bisection iterations and no general optimizer.

## Outputs and commands

- `reports/stage25_braking_model.json`
- `reports/stage25_stop_comparison.json`
- `reports/stage25_braking_curve.png`
- `reports/stage25_stop_trajectory.png`
- `reports/stage25_thrust_comparison.png`

```powershell
$env:PYTHONPATH = "src"
E:\envs\swarm_rescue\python.exe -m pytest -q tests\test_nonlinear_braking.py tests\test_predictive_brake.py tests\test_braking_safety.py tests\test_precision_stop_v2.py
E:\envs\swarm_rescue\python.exe tools\test_nonlinear_braking.py
E:\envs\swarm_rescue\python.exe tools\test_predictive_stop.py
E:\envs\swarm_rescue\python.exe tools\benchmark_precision_stop_v2.py
```
