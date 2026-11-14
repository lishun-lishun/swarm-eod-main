"""Keep the two Map09 problem drones' legal controller trace, no hidden truth."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "reports" / "fixed_suites"
OUT = ROOT / "reports" / "v5_results" / "Map09_seed862_failure_trace.json"
INPUTS = {
    "baseline": "v5_baseline_Map09_e862",
    "watchdog": "v5_watchdog_Map09_e862",
    "deadline": "v5_deadline_Map09_e862",
    "recover_first": "v5_recover_Map09_e862",
    "recover_empty_only": "v5_recover_empty_Map09_e862",
}


def main() -> None:
    report = {"map": "Map09", "seed": 862, "source": "controller-owned legal observations",
              "arms": {}}
    for arm, label in INPUTS.items():
        path = SUITES / label / "Map09_e862.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        selected = []
        for drone in record["controller_states"]:
            if drone["id"] not in (3, 6):
                continue
            selected.append({key: drone.get(key) for key in (
                "id", "state", "grasps", "stuck_count", "bomb_trace",
                "carry_trace", "last_events", "v5_path_counts",
            )})
        report["arms"][arm] = {
            "disposed": record["disposed"], "official_score": record["official_total"],
            "steps": record["elapsed_timestep"],
            "controller_events": record.get("controller_task_events", {}),
            "drones": selected,
            "raw_source": str(path.relative_to(ROOT)).replace("\\", "/"),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
