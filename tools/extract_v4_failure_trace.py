"""Extract the legal controller traces for the failed seed-862 Map09 carriers."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "reports/fixed_suites/v4_combined_trace_Map09_e862"
          / "Map09_e862.json")
TARGET = ROOT / "reports/v4_failure_trace_Map09_e862.json"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    carriers = []
    for row in source["controller_states"]:
        if row["id"] not in (3, 6):
            continue
        carriers.append({key: row[key] for key in (
            "id", "state", "carrying", "grasps", "stuck_count",
            "path_points", "bomb_trace", "carry_trace", "last_events",
        )})
    TARGET.write_text(json.dumps({
        "engine_seed": 862,
        "map": "Map09",
        "disposed": source["disposed"],
        "official_score": source["official_total"],
        "controller_only": True,
        "carriers": carriers,
        "source": str(SOURCE.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
