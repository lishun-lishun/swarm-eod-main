# Fixed-layout baseline and transport diagnosis (seed 861)

Official Launcher/Pymunk, 10 rescue drones, both NO_COM_ZONE and NO_GPS_ZONE,
fixed 10-bomb layouts from `config/zones_location.txt`. The immutable baseline
upload ZIP is `artifacts/latest_upload/team006_final.zip`; the full structured
results are in this directory's per-map JSON files and `summary.json`.

| Map | Disposed | Official score | Mean health % | Finish/limit |
|---|---:|---:|---:|---:|
| Map01 | 10 | 83.8 | 93.2 | 1639/2000 |
| Map02 | 10 | 96.7 | 91.6 | 1781/5000 |
| Map03 | 10 | 99.1 | 95.6 | 984/5000 |
| Map04 | 10 | 97.7 | 98.4 | 1855/5000 |
| Map05 | 10 | 98.5 | 92.4 | 1949/7200 |
| Map06 | 7 | 58.1 | 80.4 | 5400/5400 |
| Map07 | 10 | 89.0 | 91.2 | 3367/5400 |
| Map07t4 | 8 | 64.1 | 80.4 | 5400/5400 |
| Map08 | 8 | 65.0 | 84.8 | 5400/5400 |
| Map09 | 10 | 85.8 | 86.0 | 3770/5400 |

Total 93/100, seven complete maps, mean official score 83.78. Map01 satisfies
the updated requirement of 10/10; its score need not reach 92.

The revised acceptance threshold is **10/10 and official score >=92 on each
other map**, not just a mean of 92. Official rescue weights are disposal 60%,
health 20%, and completion time 20%; exploration has 0% weight. Map07's
10/10 score of 89.0 has health 91.2% and time score 53.8%; at unchanged health
it would need completion by about step 2800 instead of 3367 to reach 92.
Map09's 10/10 score of 85.8 has health 86.0% and time score 43.1%; at unchanged
health it would need completion by about step 2603 instead of 3770. These are
diagnostic targets under unchanged health, not predicted strategy results.

Failure attribution from the evaluation-only truth audit: Map06 has two bombs
never observed and an eighth grasp that remained on a carrier at timeout.
Map07t4 has nine grasps but eight disposals, with one bomb never observed.
Map08 has eight grasps and eight disposals, with two bombs never observed.
All missed bombs had physically reachable legal grasp poses in the separate
geometry audit. Truth coordinates are never supplied to controllers.

Map07t4 carrier trace repeated the 8/10 result without changing decisions.
Drone 7 reduced its center field path cost to zero near step 5270, but the
nearest live center semantic hit remained 87–120 px away. The local straight
passability test usually rejected that hit, and the center path collapsed to
one point. This is a false arrival in the mapped center field followed by a
failed local approach, rather than a long-haul route failure.

One isolated, default-off candidate (`V3_PASSABLE_CENTER_HIT=1`) chooses the
nearest passable hit among all live center hits. Official Map07t4 A/B: baseline
8/10, 64.1, health 80.4%, 67 stuck events; candidate 8/10, 65.0, health
85.2%, 70 stuck events. It does not improve disposal count and has not been
promoted to the final policy. Remaining work is a general local center arrival
repair, exploration coverage, and complete cross-map regression before ZIP.

## Subsequent isolated test

`V3_LOCAL_CENTER_VIEW=1` adds a map-connected local view search when a carrier's
center path degenerates to one point while a live semantic center remains over
65 px away. Official same-seed Map07t4 A/B: baseline 8/10, 64.1, health 80.4%,
9 grasps, 67 stuck; candidate 8/10, 63.8, health 78.8%, 10 grasps, 60 stuck.
Only one local view route was found, on drone 1 rather than the stranded drone
7. This candidate is default-off and rejected for promotion.

Map08 baseline corridor counters: two entries, one branch found, zero branches
entered, one completed closed end, and no drone in RETURN at timeout. This
supports checking branch clearance/alignment and continuation, not early
RETURN. Map06 had one corridor entry and one branch entered. The two Map06
unseen bombs had 303–338 mapped reachable free cells within the broad semantic
radius in the evaluator, while Map08's two unseen bombs had 42 and 0 such
cells respectively; this geometric count alone does not prove actual line of
sight. The corridor trace-only Map08 replay matched 8/10, 65.0.
The reason-count audit repeated Map08's 8/10 and 65.0 with unchanged policy.
Drone 9 produced `BRANCH_ALIGN` for 25 frames and `BRANCH_ENTER` for 396 frames,
but never achieved the required 55 px projected branch motion. The isolated
`CORRIDOR_BRANCH_COMMIT_FIX=1` candidate preserved/recovered the branch goal;
official Map08 A/B fell to 7/10 and 59.0, with branches entered still zero.
It remains default-off. The next measurement must cover projected motion,
path endpoint and geometric clearance during those frames before changing
the corridor motion law.
