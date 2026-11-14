# Official physics and sensor audit (stage 1)

Audit date: 2026-11-12. Scope: the current repository only. No legacy V2,
Fusion, or Champion controller was used.

## Strategy entry and call order

- The submission entry is `src/swarm_rescue/solutions/my_drone_eval.py`. The
  resolver accepts `drone_class_for_mode(mode)` or `MyDroneEval`, with the
  factory taking precedence (`src/swarm_rescue/entry_point.py:1-66`). The new
  package is independent and is not wired into the official entry point during
  calibration.
- `GuiSR` schedules updates every `FRAME_RATE = 1/30 s`
  (`gui_sr.py:184-190`, `constants.py:5`), but this is wall-clock scheduling,
  not the physics integration interval used by `Playground.step()`.
- Each normal GUI update collects messages, calls every live drone's
  `control()`, then calls `Playground.step()` (`gui_sr.py:366-392`). There is
  one initial warm-up physics step before solution control is called
  (`gui_sr.py:342-356`).
- A playground step performs: `pre_step` (which clears controller commands),
  load/apply the new commands, ten Pymunk micro-steps, sensor computation, and
  `post_step` (`playground.py:365-410`). Consequently the action and the new
  observation belong to one complete strategy frame.

## Rigid body and translational action

- The drone collision body is built from `drone_v2.png`, using its convex hull,
  with mass 50 (`drone_base.py:29-42`). `DroneAbstract` passes nominal
  `radius=10`, but `DroneBase` explicitly uses the sprite hull rather than a
  circle (`drone_abstract.py:175`, `drone_base.py:36-40`). Safety geometry must
  therefore use the actual hull/support radius, not blindly assume a 10 px
  circle.
- Drone shapes use elasticity 0.5 and contact friction 0.9
  (`drone_base.py:61-68`). Top-down free-flight drag comes from space damping,
  not shape friction.
- `LINEAR_FORCE=100` and `LINEAR_SPEED_RATIO=3`, so the maximum local force is
  300; with mass 50 the force/mass ratio is 6 in Pymunk's integration units
  (`definitions.py:12`, `constants.py:6`, `drone_base.py:78-79`).
- `forward` applies local force `(300 f, 0)` and `lateral` applies local force
  `(0, 300 l)`. If `sqrt(f^2+l^2)>1`, the pair is normalized to the unit circle
  before either force is applied (`drone_base.py:139-160`). Thus at heading
  zero forward is world +x and positive lateral is local/world +y. For heading
  theta the world command is `R(theta)[f,l]`; the inverse world-to-body map is
  exactly the matrix stated in the task.
- Rotation is not torque controlled. Each frame directly assigns angular
  velocity `rotation * (0.3 * 0.6) = rotation * 0.18` rad per simulator time
  unit (`definitions.py:13`, `constants.py:7`, `drone_base.py:162-163`). It is
  separate from the translational unit-circle constraint.

## Control-frame timing and expected model form

- `SPACE_DAMPING=0.95`; Pymunk defines it per simulated second
  (`definitions.py:10-11`). One `Playground.step()` calls ten
  `space.step(0.1)` micro-steps, for total integration time 1.0
  (`definitions.py:19`, `playground.py:369-400`). Therefore a strategy frame is
  the correct discrete-model time unit and the source-level no-force
  expectation is `lambda ~= 0.95`, not `0.95^10` and not a 1/30-second decay.
- Force is submitted once before the ten calls to `space.step`; empirical
  calibration is required to capture Pymunk's force clearing/integration
  semantics. The gain `b` must not be guessed as `force/mass`.
- The fitted model is `v[k+1] = lambda v[k] + B u_world[k] + bias + epsilon`.
  The script fits both isotropic `B=bI` and full 2x2 `B`, uses deterministic
  per-manoeuvre holdout samples, and selects the matrix only when its validation
  RMSE improves by at least 2%.

## Position, heading, and odometry measurements

- GPS reads the anchor position, adds AR(1) noise with coefficient 0.98 and
  innovation standard deviation 5 px (`drone_sensors.py:11-44,99-106`).
- Compass reads the normalized body angle and adds AR(1) noise with coefficient
  0.98 and innovation standard deviation 4 degrees
  (`drone_sensors.py:118-147,187-195`). Positive angle and rotation are
  counter-clockwise.
- Odometry reports distance travelled, motion direction relative to the
  previous body frame, and heading change. Independent Gaussian standard
  deviations are 0.2 px, 8 degrees, and 1 degree respectively
  (`drone_sensors.py:207-270,310-324`).
- Ground-truth pose and direct body velocity are used only by
  `tools/calibrate_dynamics.py`. Competition controllers must use the legal
  measurements; `true_position()` and `true_angle()` are explicitly guarded in
  competition mode (`drone_abstract.py:611-637`).

## LiDAR facts relevant to later braking work

- Current constants are 181 rays, 360-degree FOV, and **230 px** maximum range
  (`constants.py:22-24`, `drone_lidar.py:125-139`). The class docstring still
  says 300 px and is stale; code constants are authoritative.
- Ray angles span `[-FOV/2,+FOV/2]`, inclusive, in the sensor/body frame
  (`distance_sensor.py:30-40`). With 181 rays this is 2 degrees between rays,
  with duplicated directions at -180/+180 degrees.
- LiDAR normalization is disabled and Gaussian range noise has standard
  deviation 2.5 px (`drone_lidar.py:21-39,79-83,134-139`). Disabled output is
  NaN. Later occupancy code must inspect the ray-compute hit semantics before
  treating a value at maximum range as a physical hit.

## Collisions and Map07 limits

- Drone-wall and drone-drone contacts are physical Pymunk collisions and both
  have explicit damage callbacks (`drone_abstract.py:29-86`).
- Map07 is 1660x1122 px, normally spawns 10 drones, and sets a maximum of 5400
  timesteps and 180 seconds wall time (`maps/map_07.py:25-68`). These values are
  Map07-specific and differ from other maps.

## Calibration protocol and reproducibility

Run from the repository root:

```powershell
E:\envs\swarm_rescue\python.exe tools\calibrate_dynamics.py
$env:PYTHONPATH = "src"
E:\envs\swarm_rescue\python.exe -m pytest -q tests\test_blue_dynamics.py
```

The tool executes the official body in an otherwise empty 20000x20000 px
playground. It records forward, lateral, terminal-speed/coast, diagonal,
rotation, simultaneous rotation/translation, and five reverse-braking trials.
Outputs are `dynamics_calibration_samples.csv`, `braking_curve.csv`, and
`dynamics_identification.json` in `reports/`.

The calibration was run with `E:\envs\swarm_rescue` (Python 3.11.17, Pymunk
6.4.0, Arcade 2.6.17). The measured scalar model is `lambda = 0.95`,
`b = 0.5729312289729012`, with validation RMSE approximately `1.08e-14`
px/control-frame. The full matrix fit has only floating-point-scale off-diagonal
terms, so there is no observed directional coupling and the scalar model is
selected. Measured files are committed beside this audit; exact trial values
and residual metrics remain in the machine-readable JSON/CSV rather than being
rounded here.
