# Historical blue-strategy ranking

Source audit: `D:\code\isaaclab\fusai\1\final_fusai` (read-only), 2026-11-12.

Ranking uses official-score JSON summaries and 10/10 completion counts, not directory names.

| Rank | Historical strategy / source | Runs | Mean official score | 10/10 runs | Evidence |
|---:|---|---:|---:|---:|---|
| 1 | H_degraded / `team861_c2_pose` | 30 | 88.55 | 28/30 | `reports/fusion/H_degraded/summary.json` |
| 2 | H_cargo_only | 30 | 87.36 | 26/30 | `reports/fusion/H_cargo_only/summary.json` |
| 3 | C2 | 30 | 87.52 | 25/30 | `reports/fusion/C2/summary.json` |
| 4 | H_recovery_base | 30 | 86.92 | 25/30 | `reports/fusion/H_recovery_base/summary.json` |
| 5 | E | 30 | 86.74 | 25/30 | `reports/fusion/E/summary.json` |
| 6 | team861_v2 stress matrix | 15 | 82.48 | not consistently recorded as a count | `reports/benchmark_aggregate.md` |
| 7 | team861_final stress matrix | 15 | 80.50 | not consistently recorded as a count | `reports/benchmark_aggregate.md` |

H_degraded is the selected high-level baseline. It completed Map01/02/06/08 in all
six paired runs per map and Map09 in four of six. Its improvement over the preceding
H_recovery_base was +1.627 points, with no per-map mean regression; Map08 improved
by 4.883 and Map09 by 2.700 points.

## Retain

- legal-sensor log-odds mapping, semantic coverage and cached route costs;
- distributed bomb identity, tombstones, claim leases and deterministic auctions;
- concurrent EXPLORE/FETCH/CARRY state machines;
- corner grasp candidates, carrier priority and stuck recovery;
- degraded-zone cargo push/pull pose choice using LiDAR clearance.

## Do not retain unchanged

- the historical empirical motion law or summed wall repulsion;
- circular-footprint assumptions;
- disposal inferred from proximity rather than the official disposed/grasp event;
- persistent failed-center cells without expiry;
- planar-only paths through orientation-sensitive narrow passages.

## Observed failure concentration

The two incomplete H_degraded runs were both Map09. Historical reports attribute
remaining failures to narrow passages, disposal entrance traffic, repeated local
replanning and final 1–2 bomb return. The new controller therefore keeps the high-level
coordination baseline while replacing motion with the identified velocity controller,
joint LiDAR predictive safety, explicit SE(2) corridor traversal and disposal reservations.
