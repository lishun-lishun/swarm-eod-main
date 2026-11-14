# Corridor-end exploration analysis

## Outcome

The accepted change is deliberately a late, single-drone fallback rather than a replacement for the proven exploration stack. Drone 9 may create a corridor commitment after step 3000 only when its legal local/team knowledge still lacks bombs and no visible fetch/carry task has existed for 120 frames. `CORRIDOR_EXPLORER=0` disables the layer; `CORRIDOR_SPECIALISTS=*` or a comma-separated list changes assignment; `CORRIDOR_START_STEP` changes the late-search gate.

The stable baseline is preserved as `team006_final_baseline.zip`. No map coordinates, bomb coordinates, hidden truth, or illegal shared state are used.

## Measured cause of mid-corridor reversals

Thirty-frame windows around command reversals were captured before changing exploration behavior. The capped traces contained:

| Map | Goal reselect | Local control | Empty path | Return |
|---|---:|---:|---:|---:|
| Map06 | 34 | 86 | 0 | 0 |
| Map08 | 11 | 109 | 0 | 0 |
| Map09 | 24 | 95 | 1 | 0 |

The code and traces agree: `_pockets()` rejects the two-opposite-open-arc geometry typical of corridors; the ordinary goal is reconsidered every 40 frames; and a newer `scan_f` timestamp can invalidate the goal merely because LiDAR swept it. `_should_return()`/`SCOUT_MIN` did not directly cause the captured reversals. Local wall control then amplified several target changes into opposite thrust commands.

## Implemented behavior

- `corridor_explorer.py` implements `ENTER → ADVANCE → END_APPROACH → INSPECT → BRANCH/CLOSED → COMPLETE`, with pause/resume and TTL.
- Entry requires eight consecutive frames of aligned forward motion, two consistent side walls, width feasibility using the real Pymunk convex hull, and at least 130 px forward visibility.
- During `ADVANCE`, the state machine locks the existing semantic goal against the 40-frame rescore and `scan_f` invalidation, but movement remains on the original map path and mature flight controller.
- Center-origin LiDAR range is converted to surface gap by subtracting convex support exactly once.
- The nominal 2 px target is not forced when uncertainty makes it unsafe. With the validated current estimator parameters, the static part of the target is `3 + 2×2.5 = 8 px`, plus nonlinear ten-substep stopping distance.
- Endpoint braking is a one-dimensional predictive half-plane along the corridor axis. Side-wall clearance remains with the proven local controller. This avoids the invalid pair of mutually opposed braking constraints seen in rejected iterations.
- A branch needs at least five adjacent open rays, sufficient physical width for the oriented hull plus margin, and three consecutive observations. Old nonconsecutive observations are cleared.
- Before rotating toward a side branch, the drone requires clearance for the hull circumcircle; otherwise it backs away from the end wall. Branch travel is then handed to the existing map planner rather than direct lateral flight.
- Real fetch/carry tasks pause the commitment. A confirmed final bomb count cancels it.

## Official Pymunk A/B results

Fixed layout seed 1, engine seed 861:

| Map | Baseline bombs / health / steps | Accepted bombs / health / steps | Notes |
|---|---|---|---|
| Map06 | 10 / 96.4% / 1797 | 10 / 96.4% / 1797 | Bit-for-bit metric parity; layer did not trigger. |
| Map08 | 9 / 80.8% / 5400 | 9 / 82.4% / 5400 | Stuck events 100→75; collision events 48→44. |
| Map09 | 10 / 87.2% / 3494 | 10 / 87.2% / 3494 | Metric parity; layer did not trigger. |

On fixed Map08, all ten bombs were already discovered at step 1604 in the baseline. Drone 1 grasped the last bomb at step 4846 and then repeatedly became stuck while carrying it. The 9/10 result is therefore a cargo-return failure, not an undiscovered hidden branch; changing that subsystem was explicitly outside this task.

The six layout-seed-2 regressions across engine seeds 861 and 1861 all remain 10/10. Five runs were exactly unchanged in candidate 11 because the corridor layer never triggered. The sole early-trigger regression (Map08/861, 10→9) was retested after moving the fallback gate to step 3000 and returned to exact baseline metrics: 10 bombs, 77.6% health, 4542 steps.

## Rejected iterations and why

The report retains every official result rather than only the accepted run. Applying joint LiDAR safety throughout a narrow corridor produced thousands of infeasible frames. Directly commanding all ten drones improved one fixed Map08 run to 10/10 but regressed Map09 and health. Four specialists also failed to improve Map08. A single specialist with the mature planner was safe, but starting at step 1800 changed an otherwise successful Map08 layout. The final step-3000 gate is the first configuration that preserves all known 10/10 cases.

## Limits

The accepted default is intentionally conservative. In the fixed Map08 run it entered one late candidate corridor but rejected it after losing wall consistency; earlier diagnostic candidate 11, using the same geometry/control logic with an earlier start, reached three ends and entered three confirmed branches with no approach failure, but that earlier start was not regression-safe. The code records `min_observation_gap` separately for future runs; historical `min_end_gap` values include branch turns and are not valid 2 px observation measurements.

The full repository suite passed 343 tests. Five unrelated pre-existing/platform failures remain: one `EvalPlan` default mismatch, one GPU/CPU LiDAR ray mismatch, and three Windows file-handle cleanup failures in state-recording tests. The focused corridor, braking, LiDAR geometry, and SE(2) suite passed 20/20.
