# Final all-map strategy analysis

## Outcome

The final candidate disposed all 10 bombs in every map implementation shipped by this repository: Map01 through Map09 and the Map07t4 variant. The fixed comparison uses layout seed 1, engine seed 861, the official competition launcher, ten drones, legal controller sensors, and the normal match limits. Across the ten scenarios it disposed 100/100 bombs, averaged 92.2% health, and finished in an average of 2241.8 control frames.

## Root cause and correction

The fixed Map08 failure was not late task allocation. The last bomb was first observed only at step 4788, leaving too little time for pickup and delivery. Giving every drone a strong boundary preference caused broad regressions, while assigning the preference to drone 9 found the hidden boundary pocket much earlier. An initial version started the specialist at step 400 and regressed Map07t4: the specialist accepted a difficult edge pickup at step 1118 and remained trapped while carrying.

The accepted configuration therefore starts the single edge specialist at step 1200. This preserves the mature first-pass allocation while leaving enough time for a geometry-derived boundary sweep. It does not inspect the map name or use wall/bomb truth. The specialist also stores sparse estimated-pose breadcrumbs during a claimed-bomb approach and, only for specialist pickups after the takeover step, retraces them briefly before resuming loaded global planning. Other nine carriers keep the established transport behavior.

On Map08 this moved the last-bomb observation to step 1494, grasp to step 1535, and full disposal to step 4094. On Map07t4 it left the successful baseline exactly unchanged at 10/10, step 4468, and 84.8% health.

## Final fixed-seed results

| Map | Disposed | Mean health | Completion step |
| --- | ---: | ---: | ---: |
| Map01 | 10/10 | 95.2% | 1352 |
| Map02 | 10/10 | 97.6% | 1188 |
| Map03 | 10/10 | 95.6% | 1146 |
| Map04 | 10/10 | 94.8% | 1527 |
| Map05 | 10/10 | 94.8% | 1475 |
| Map06 | 10/10 | 96.8% | 1705 |
| Map07 | 10/10 | 88.0% | 2895 |
| Map07t4 | 10/10 | 84.8% | 4468 |
| Map08 | 10/10 | 84.4% | 4094 |
| Map09 | 10/10 | 90.0% | 2568 |

## Limits

This is a complete pass for the fixed layout and engine seed, not a proof over every random layout. The repository has no `map_10.py`; “all maps” here means every available official map class. Raw per-run JSON is retained so failures under additional seeds can be compared without using evaluator truth inside the controller.
