"""Summarize official fixed-suite V4 A/B runs without inventing scores."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "reports" / "fixed_suites"
REPORTS = ROOT / "reports"
BASELINE = "baseline_20261114_e861"
ARMS = {
    "eta": "v4_eta_subset_e861",
    "model_empty": "v4_model_empty_subset_e861",
    "combined": "v4_combined_subset_e861",
}
MAPS = ("Map06", "Map07t4", "Map08", "Map09", "Map07")


def result(label: str, name: str) -> dict | None:
    path = SUITES / label / f"{name}_e861.json"
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = row.get("controller_diagnostics") or {}
    return {
        "disposed": row["disposed"],
        "official_score": row["official_total"],
        "health_percent": row["mean_health_percent"],
        "steps": row["elapsed_timestep"],
        "grasps": diagnostics.get("grasp_successes"),
        "stuck": diagnostics.get("stuck_events"),
        "damage_events": diagnostics.get("collision_damage_events"),
        "source": str(path.relative_to(ROOT)),
    }


def main() -> None:
    baseline = {name: result(BASELINE, name) for name in MAPS}
    arms = {
        arm: {name: result(label, name) for name in MAPS}
        for arm, label in ARMS.items()
    }
    payload = {
        "engine_seed": 861,
        "official_engine": "Launcher/Pymunk via tools/run_fixed_suite.py",
        "fixed_layout": "config/zones_location.txt",
        "maps": list(MAPS),
        "baseline": baseline,
        "arms": arms,
        "decisions": {},
    }
    for arm, rows in arms.items():
        paired = [name for name in MAPS if baseline[name] and rows[name]]
        full_regressions = [name for name in paired
                            if baseline[name]["disposed"] == 10
                            and rows[name]["disposed"] < 10]
        baseline_count = sum(baseline[name]["disposed"] for name in paired)
        candidate_count = sum(rows[name]["disposed"] for name in paired)
        payload["decisions"][arm] = {
            "paired_maps": paired,
            "full_map_regressions": full_regressions,
            "baseline_disposed": baseline_count,
            "candidate_disposed": candidate_count,
            "passes_subset_priority": (
                len(paired) == len(MAPS) and not full_regressions
                and candidate_count > baseline_count
            ),
        }
    for filename, selected_arms in (
        ("task_assignment_ablation.json", ("eta", "combined")),
        ("flight_control_ablation.json", ("model_empty", "combined")),
    ):
        report = dict(payload)
        report["arms"] = {key: arms[key] for key in selected_arms}
        report["decisions"] = {key: payload["decisions"][key]
                               for key in selected_arms}
        (REPORTS / filename).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    all_names = ("Map01", "Map02", "Map03", "Map04", "Map05",
                 "Map06", "Map07", "Map07t4", "Map08", "Map09")
    combined = {
        name: result("v4_combined_subset_e861", name)
              or result("v4_combined_remaining_e861", name)
        for name in all_names
    }
    baseline_all = {name: result(BASELINE, name) for name in all_names}
    complete = all(combined.values()) and all(baseline_all.values())
    all_maps = {
        "engine_seed": 861,
        "official_engine": payload["official_engine"],
        "fixed_layout": payload["fixed_layout"],
        "source_sha256": json.loads((SUITES / "v4_combined_subset_e861"
                                     / "summary.json").read_text(
                                         encoding="utf-8"))["source_sha256"],
        "maps": {name: {"baseline": baseline_all[name],
                        "combined": combined[name]} for name in all_names},
        "complete": complete,
    }
    if complete:
        all_maps["baseline_total_disposed"] = sum(
            row["disposed"] for row in baseline_all.values())
        all_maps["combined_total_disposed"] = sum(
            row["disposed"] for row in combined.values())
        all_maps["baseline_full_maps"] = sum(
            row["disposed"] == 10 for row in baseline_all.values())
        all_maps["combined_full_maps"] = sum(
            row["disposed"] == 10 for row in combined.values())
        all_maps["baseline_mean_official_score"] = sum(
            row["official_score"] for row in baseline_all.values()) / len(all_names)
        all_maps["combined_mean_official_score"] = sum(
            row["official_score"] for row in combined.values()) / len(all_names)
    (REPORTS / "all_maps_v4_summary.json").write_text(
        json.dumps(all_maps, ensure_ascii=False, indent=2), encoding="utf-8")
    cross_seed = {}
    for seed in (861, 862):
        arms_by_seed = (
            {"baseline": BASELINE, "combined": "v4_combined_subset_e861"}
            if seed == 861 else
            {"baseline": "v4_baseline_subset_e862",
             "combined": "v4_combined_subset_e862"}
        )
        cross_seed[str(seed)] = {}
        for arm, label in arms_by_seed.items():
            rows = {}
            for name in MAPS:
                path = SUITES / label / f"{name}_e{seed}.json"
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    rows[name] = {"disposed": data["disposed"],
                                  "official_score": data["official_total"]}
            cross_seed[str(seed)][arm] = {
                "maps": rows,
                "total_disposed": sum(row["disposed"] for row in rows.values()),
                "mean_official_score": (sum(row["official_score"] for row in rows.values())
                                        / len(rows)) if rows else None,
            }
    (REPORTS / "cross_seed_v4_summary.json").write_text(
        json.dumps(cross_seed, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
