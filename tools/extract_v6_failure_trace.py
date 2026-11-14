"""Compact, controller-owned traces for rejected V6 candidates."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "reports" / "fixed_suites"
OUT = ROOT / "reports" / "v6_results" / "failure_trace.json"


def extract(label, name, seed, drones=None):
    path = SUITES / label / f"{name}_e{seed}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = set(drones) if drones is not None else None
    states = [state for state in data.get("controller_states", [])
              if selected is None or state["id"] in selected]
    events = {ident: [event for event in rows
                      if (event["kind"].startswith("v6_")
                          and event["kind"] != "v6_real_ray_cells")
                      or event["kind"] in ("grasp", "release", "stuck", "center_bad")]
              for ident, rows in data.get("controller_task_events", {}).items()
              if selected is None or int(ident) in selected}
    return {"map": name, "seed": seed, "label": label,
            "disposed": data["disposed"], "official_score": data["official_total"],
            "states": states, "events": events,
            "source": str(path.relative_to(ROOT)).replace("\\", "/")}


def main():
    rows = [extract("v6_view_mature_Map08_e861", "Map08", 861),
            extract("v6_route_Map09_e862", "Map09", 862, (6,)),
            extract("v6_route_Map07t4_e861", "Map07t4", 861),
            extract("v6_task_Map09_e862", "Map09", 862, (3, 6))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps([row for row in rows if row is not None],
                              ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
