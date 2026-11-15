"""Summarize completed V9 official runs without inventing unrun scores."""
from __future__ import annotations

import json
from pathlib import Path

from benchmark_fixed_zones import load_fixed_layouts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "v9_results"
SUITES = ROOT / "reports" / "fixed_suites"
FINAL_LABELS = {
    "Map01": "v9_legal_deadline_map01_e861",
    "Map06": "v9_keymaps_e861",
    "Map07t4": "v9_keymaps_e861",
    "Map08": "v9_keymaps_e861",
    "Map09": "v9_keymaps_e861",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    layouts = load_fixed_layouts(ROOT / "config" / "zones_location.txt")
    baseline = {name: read(SUITES / "v8_default_all_e861" / f"{name}_e861.json")
                for name in FINAL_LABELS}
    comparison = []
    failures = {}
    for name, label in FINAL_LABELS.items():
        path = SUITES / label / f"{name}_e861.json"
        run = read(path)
        old = baseline[name]
        agents = run.get("v9_legal_agents", [])
        disposed = {event.get("bomb_index")
                    for event in run.get("official_disposal_events", [])}
        comparison.append({
            "map": name, "engine_seed": 861,
            "v8_disposed": old["disposed"], "v9_disposed": run["disposed"],
            "v8_official_score": old["official_total"],
            "v9_official_score": run["official_total"],
            "v8_mean_health_percent": old["mean_health_percent"],
            "v9_mean_health_percent": run["mean_health_percent"],
            "v8_elapsed_step": old["elapsed_timestep"],
            "v9_elapsed_step": run["elapsed_timestep"],
            "v9_walltime_seconds": run["walltime_seconds"],
            "v9_launcher_ok": run["ok"],
            "v9_raw_json": str(path.relative_to(ROOT)).replace("\\", "/"),
        })
        bomb_rows = []
        for index, bomb in enumerate(layouts[name]):
            position = (bomb["x"], bomb["y"])
            nearby = []
            for agent in agents:
                for target in agent.get("task_targets", []):
                    world = target.get("world_xy")
                    if world is None:
                        continue
                    distance = ((world[0]-position[0])**2+
                                (world[1]-position[1])**2)**0.5
                    if distance < 40:
                        nearby.append({"drone": agent["id"], "key": target["key"],
                                       "state": target["state"],
                                       "local_observed": target["local_observed"],
                                       "distance_px": round(distance, 2)})
            bomb_rows.append({
                "bomb_index": index,
                "fixed_position_offline_only": list(position),
                "official_disposed": index in disposed,
                "terminal_target_matches_offline_approx": nearby,
                "failure_class": ("OFFICIAL_DISPOSED" if index in disposed
                                  else "NO_NEARBY_FINAL_TRACK" if not nearby
                                  else "TRACKED_BUT_NOT_DISPOSED"),
            })
        failures[name] = {
            "bombs": bomb_rows,
            "official_disposal_events": run.get("official_disposal_events", []),
            "per_drone": [{
                "id": a["id"], "phase": a["phase"],
                "known_free_cells": a["known_free_cells"],
                "semantic_ray_cells": a["semantic_ray_cells"],
                "gps_active_at_end": a["gps_active"],
                "pose_anchored_at_end": a["pose_anchored"],
                "grasp_events": [e for e in a["task_events"]
                                 if e["kind"] == "GRASP_CONFIRMED"],
                "delivery_events": [e for e in a["task_events"]
                                    if e["kind"] == "DELIVERY_CONFIRMED"],
                "exception_events": [e for e in a["exceptions"]
                                     if e["kind"] == "V9_EXCEPTION"],
            } for a in agents],
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison_e861.json").write_text(
        json.dumps({"engine": "official Launcher/Pymunk",
                    "layout": "config/zones_location.txt", "rows": comparison},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "failure_trace_e861.json").write_text(
        json.dumps({"warning": "fixed bomb coordinates and association are offline audit only",
                    "maps": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    for row in comparison:
        print(f"{row['map']}: {row['v8_disposed']} -> {row['v9_disposed']} /10, "
              f"score {row['v8_official_score']} -> {row['v9_official_score']}")


if __name__ == "__main__":
    main()
