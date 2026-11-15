"""Offline truth-only audit of V8 goals versus legal visible viewpoints."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from audit_fixed_reachability import map_class
from benchmark_fixed_zones import load_fixed_layouts
from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
from swarm_rescue.simulation.reporting.explored_map import ExploredMap

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "reports" / "fixed_suites"
OUT = ROOT / "reports" / "v8_results" / "macro_stance_audit.json"


def legal_stances(standing, free, bomb):
    height, width = free.shape
    bx, by = bomb
    px = int(round(bx + width / 2))
    py = int(round(-by + height / 2))
    points = []
    for sy in range(max(0, py - 180), min(height, py + 181), 5):
        for sx in range(max(0, px - 180), min(width, px + 181), 5):
            if not standing[sy, sx]:
                continue
            world = np.array((sx - width / 2, height / 2 - sy), float)
            target = np.array(bomb, float)
            distance = np.linalg.norm(target - world)
            if not 28 <= distance <= 125:
                continue
            fractions = np.linspace(0, max(0, 1 - 12 / distance),
                                    max(2, int(distance / 2)))
            samples = world + fractions[:, None] * (target - world)
            xs = np.rint(samples[:, 0] + width / 2).astype(int)
            ys = np.rint(-samples[:, 1] + height / 2).astype(int)
            if free[ys, xs].all():
                points.append(world)
    return np.asarray(points, float).reshape(-1, 2)


def audit(map_name, indices):
    layouts = load_fixed_layouts(ROOT / "config" / "zones_location.txt")
    mission = map_class(map_name)(
        drone_type=DroneMotionless, number_drones=10,
        zones_config=(ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE))
    try:
        image = ExploredMap()
        image.initialize_walls(mission.playground)
        free = (image._map_playground != 0).astype(np.uint8)
        hull = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        standing = cv2.erode(free, hull)
        result = []
        for label in ("v8_macro_shadow_e861", "v8_macro_active_e861"):
            path = SUITES / label / f"{map_name}_e861.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            events = data.get("controller_task_events", {}).get("9", [])
            goals = [event for event in events
                     if event["kind"] == "v8_macro_planned"]
            reached = [event for event in events
                       if event["kind"] == "v8_macro_end"
                       and event.get("reached")]
            for index in indices:
                bomb = layouts[map_name][index]
                stances = legal_stances(
                    standing, free, (bomb["x"], bomb["y"]))
                def nearest(events):
                    if not events or len(stances) == 0:
                        return None
                    return round(min(float(np.linalg.norm(
                        stances - np.asarray(event["target"]), axis=1).min())
                        for event in events), 1)
                result.append({
                    "map": map_name, "seed": 861, "label": label,
                    "bomb_index": index,
                    "truth_visible_stances": len(stances),
                    "planned_goals": len(goals), "reached_goals": len(reached),
                    "nearest_planned_to_legal_stance_px": nearest(goals),
                    "nearest_reached_to_legal_stance_px": nearest(reached),
                })
        return result
    finally:
        mission.playground.cleanup()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = audit("Map06", (2, 3)) + audit("Map08", (0, 3))
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
