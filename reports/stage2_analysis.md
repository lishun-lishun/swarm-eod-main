# Stage 2: model-based velocity control and precision stopping

Date: 2026-11-12. This stage uses the official `DroneBase` and
`Playground.step()` Pymunk integration. It does not contain LiDAR avoidance,
A*, frontier exploration, multi-drone allocation, or artificial potential
fields.

## Implemented control chain

Two independent world-frame velocity controllers were tested:

1. **Inverse dynamics** computes
   `u=(v_ref-0.95*v_hat)/0.5729312289729012` and then applies one two-dimensional
   unit-circle saturation. No extra proportional term is added.
2. **Steady-state feedforward + P feedback** computes
   `u=(1-0.95)/0.572931*v_ref + Kp*(v_ref-v_hat)`. Pole candidates 0.4, 0.6,
   and 0.75 were measured. The retained pole is 0.75.

No integral or derivative term was enabled. With truth velocity the identified
plant has no measurable steady-state bias; adding integral action would mainly
integrate legal sensor noise at this stage.

The legal estimator predicts endpoint velocity from the previous world-frame
command, corrects it with official odometry, propagates position with the
identified within-frame position model, and applies only low-bandwidth GPS
correction. GPS is never differentiated. Odometry `alpha` is resolved from the
previous body frame; during commanded rotation the previous heading is
reconstructed with odometry `theta`. When the historical rotation command is
zero, noisy `theta` is not integrated as real rotation.

`PrecisionStopController` has CRUISE, BRAKE, APPROACH, HOLD, and COMPLETE
modes. BRAKE sends explicit full reverse thrust. APPROACH/HOLD use a position-PD
outer loop whose target velocity is passed to exactly one velocity controller.
Completion requires both estimated position and speed tolerances for eight
consecutive frames.

## Experiment coverage

The result file contains complete per-frame traces for:

- T1: 0 to 5 px/frame;
- T2: 0 to 10 px/frame;
- T3: 5 to 0;
- T4: 10 to 0;
- T5: +5 to -5;
- T6: fixed-heading diagonal velocity;
- T6b: diagonal world velocity while continuously rotating;
- T7: travel 100 px and stop;
- T8: travel 500 px and stop;
- T9: low-speed 20 px docking;
- T10: the legal noisy-sensor variant of every preceding key experiment;
- T11: terminal-speed target followed by active braking.

There are 32 main velocity runs, 12 pole-ablation runs, and 12 precision-stop
runs. Every main test is paired across both controllers and truth/legal input.
Truth is read only by the evaluation harness; the legal branch receives only
GPS, compass, odometry, and previous commands.

## Velocity tracking results

The following aggregates include transients and saturation, rather than only
reporting the settled portion:

| Controller | Input | Velocity MAE | Velocity RMSE | Estimator RMSE | Saturation |
|---|---:|---:|---:|---:|---:|
| Inverse | truth | 0.582 | 1.620 | 0.000 | 23.1% |
| Inverse | legal | 0.857 | 1.665 | 0.423 | 45.9% |
| Feedback, q=0.75 | truth | 0.603 | 1.625 | 0.000 | 19.5% |
| Feedback, q=0.75 | legal | 0.724 | 1.640 | 0.429 | 22.2% |

The inverse controller reaches an unsaturated target in one model step, but it
also converts odometry noise directly into command changes. This explains its
much larger legal-mode saturation ratio. The feedback controller is slightly
slower with truth input but is preferable with legal measurements.

Pole ablation over T1/T2 truth and legal runs:

| Pole q | Mean RMSE | Mean saturation |
|---:|---:|---:|
| 0.40 | 1.9253 | 35.3% |
| 0.60 | 1.9211 | 29.7% |
| 0.75 | 1.9197 | 23.0% |

The RMSE differences are small because acceleration saturation dominates the
large target steps. q=0.75 was retained because it materially reduces
saturation and noisy command activity without a tracking regression.

## Braking and position-model validation

For inverse control with truth velocity, T11 reached the terminal-speed regime
and stopped over 65.564 px. The discrete prediction was 65.534 px, an error of
0.031 px. This independently validates the approximately 65 px full-speed
prediction.

In T4, inverse/truth braking from approximately 10 px/frame measured 52.069 px
against 52.055 px predicted. The feedback controller's ordinary velocity-to-zero
response travels farther because it deliberately releases full reverse thrust
as the error shrinks. PrecisionStop therefore uses explicit full reverse in its
BRAKE mode rather than assuming every velocity controller reproduces the
calibration braking curve.

## Precision-stop results

| Test | Controller | Input | Final position error | Residual speed | Completion frame |
|---|---|---|---:|---:|---:|
| 100 px | Inverse | truth | 0.007 px | 0.001 px/frame | 47 |
| 100 px | Inverse | legal | 1.232 px | 0.126 px/frame | 75 |
| 100 px | Feedback | truth | 0.001 px | <0.001 px/frame | 47 |
| 100 px | Feedback | legal | 0.923 px | 0.073 px/frame | 164 |
| 500 px | Inverse | truth | 0.002 px | <0.001 px/frame | 77 |
| 500 px | Inverse | legal | 0.712 px | 0.056 px/frame | 397 |
| 500 px | Feedback | truth | 0.002 px | <0.001 px/frame | 93 |
| 500 px | Feedback | legal | 0.689 px | 0.078 px/frame | 164 |
| 20 px dock | Inverse | truth | 0.003 px | 0.001 px/frame | 21 |
| 20 px dock | Inverse | legal | 1.137 px | 0.140 px/frame | 75 |
| 20 px dock | Feedback | truth | 0.001 px | 0.001 px/frame | 33 |
| 20 px dock | Feedback | legal | 1.014 px | 0.074 px/frame | 164 |

Truth-input braking-switch prediction errors are at most about 0.13 px in
these tests. Legal-input prediction errors are larger (up to about 1.65 px)
because switching uses estimated velocity.

The controller's COMPLETE flag is necessarily based on legal estimated state.
The independent evaluator shows that GPS/heading uncertainty leaves roughly
0.7–1.0 px final absolute-position error for the selected feedback controller.
Therefore legal noisy-sensor operation must not be described as guaranteed
sub-pixel truth accuracy, even though the truth-input plant/controller limit is
far below one pixel.

## Decision

The selected competition-facing stage-2 controller is **steady-state
feedforward + P feedback with q=0.75**, combined with the legal model/odometry
estimator. Reasons:

- lower legal velocity MAE than inverse control (0.724 versus 0.857);
- about half the legal saturation rate (22.2% versus 45.9%);
- lower mean legal precision-stop error (0.875 px versus 1.027 px);
- lower legal residual speed (0.075 versus 0.108 px/frame);
- one mean position-error sign change versus 4.3 for inverse control.

The inverse controller remains useful as a model-limit baseline and as a
low-speed truth-mode reference. No integral term is justified by these data.

## Outputs and reproduction

- `reports/stage2_results.json`: metrics and every per-frame trace.
- `reports/stage2_velocity_time.png`: T4 speed curves.
- `reports/stage2_thrust_time.png`: T4 active-braking commands.
- `reports/stage2_stop_position_error.png`: T8 position-error curves.

Run from the repository root:

```powershell
$env:PYTHONPATH = "src"
E:\envs\swarm_rescue\python.exe -m pytest -q tests\test_blue_dynamics.py tests\test_velocity_controller.py tests\test_velocity_estimator.py tests\test_precision_stop.py
E:\envs\swarm_rescue\python.exe tools\test_velocity_control.py
E:\envs\swarm_rescue\python.exe tools\test_precision_stop.py
```

The stage-1 reports and raw calibration data were not overwritten.
