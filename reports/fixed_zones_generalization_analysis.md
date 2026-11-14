# Fixed-layout generalization audit

Date: 2026-11-14

All competition runs used the official `Launcher`, Pymunk physics, ten drones,
the bomb positions parsed from `config/zones_location.txt`, both
`NO_GPS_ZONE` and `NO_COM_ZONE`, and engine seed 861.  Bomb truth was used only
by post-run evaluation and was never supplied to a controller call.

## Accepted baseline facts

- The current default remains unchanged on Map08: 8/10 disposed, 65.0 points,
  84.8% mean health, 63 stuck events.
- The relevant unit suite passes: 22 tests.
- Raster reachability against official map geometry found a connected legal
  grasping pose for every fixed bomb on Map06, Map07t4, and Map08.  The missed
  bombs are therefore controller failures, not impossible placements.
- On Map08 the two baseline misses are `(566,-535)` and `(-7,-533)`.  Both
  remain unscanned by every fused controller map.  The closest confirmed,
  reachable free standing cells are approximately 69.0 px and 134.8 px away.
  The latter is still semantically observable because the sensor measures to
  the bomb surface (about 12 px before its centre).

## Degraded-zone findings

- A raw `GPS is None` flag is not a usable task switch near a zone boundary.
  It produced 242 state changes.  The controller now has diagnostic-only
  asymmetric hysteresis: eight invalid frames enter degraded state and sixty
  valid frames leave it.
- Enabling local LiDAR corridor exploration for all drones only while they
  were stably degraded increased Map08 discovery/grasping from 8 to 10 in the
  best run, but only 8 bombs were delivered.  Mean health fell to 75.6%.
- Damage instrumentation attributed 16/27 damage events to carriers and 10/27
  to explorers.  Ten carrier events and nine explorer events had another
  semantic drone in view.  All damage frames had minimum LiDAR below 16 px.
  Congestion and cargo-wall contact are both material.

## Rejected experiments

| Experiment | Map08 result | Decision |
|---|---:|---|
| Global goal commitment, 120 frames | 4/10, 36.7 | Reject: wrong goals persist |
| Three soft edge scouts | 7/10, 56.2 | Reject: bottom congestion |
| Partitioned edge reward | 8/10, 65.0 | Neutral; no new discovery |
| Edge candidate/commit/dwell variants | 7–8/10 | Reject: repeated pseudo-checks |
| Degraded-zone corridor, all drones | 8/10, 62.7 | Reject: health regression |
| Safer early degraded corridors | 8/10, 63.1, 10 grasps | Reject: delivery congestion |
| Time-sliced degraded pickup | 7–8/10 | Reject: missed handoffs |
| Unknown visible drone treated as carrier | 7/10, 56.1 | Reject: symmetric yielding |

All rejected behavior remains disabled by default.  The next useful direction
is a non-symmetric, local right-of-way rule that works without communication or
GPS (and therefore cannot deadlock through mutual yielding), followed by a
separate cargo footprint clearance correction.  It must be accepted across
Map06, Map07t4, Map08, and a 10/10 control map before packaging.

