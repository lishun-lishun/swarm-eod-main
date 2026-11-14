"""Evaluation-only wall line-of-sight audit for fixed-layout misses.

Truth walls and bomb coordinates are loaded only after an official round.
Nothing produced here is available to a competition controller.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from audit_fixed_reachability import map_class
from benchmark_fixed_zones import load_fixed_layouts
from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
from swarm_rescue.simulation.reporting.explored_map import ExploredMap


ROOT = Path(__file__).resolve().parents[1]


def visible(free: np.ndarray, source: np.ndarray, bomb: np.ndarray,
            width: int, height: int) -> bool:
    delta = bomb - source
    distance = float(np.linalg.norm(delta))
    if distance <= 13.0:
        return True
    # A semantic ray only needs to reach the bomb surface, not its center.
    steps = max(2, int((distance - 12.0) / 2.0))
    points = source + np.linspace(0.0, 1.0 - 12.0 / distance, steps)[:, None] * delta
    px = np.rint(points[:, 0] + width / 2.0).astype(int)
    py = np.rint(-points[:, 1] + height / 2.0).astype(int)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return bool(inside.all() and free[py, px].all())


def semantic_ray_hits(free: np.ndarray, source: np.ndarray, heading: float,
                      bomb: np.ndarray, width: int, height: int) -> bool:
    """Evaluation-only 35-ray intersection with a 12 px bomb and truth walls."""
    delta = bomb - source
    if float(np.dot(delta, delta)) > (130.0 + 12.0) ** 2:
        return False
    for relative in np.linspace(-math.pi, math.pi, 35):
        angle = heading + float(relative)
        direction = np.array([math.cos(angle), math.sin(angle)])
        along = float(np.dot(delta, direction))
        across2 = float(np.dot(delta, delta) - along * along)
        if along <= 0.0 or across2 > 12.0 ** 2:
            continue
        first_hit = along - math.sqrt(max(0.0, 12.0 ** 2 - across2))
        if not 0.0 <= first_hit <= 130.0:
            continue
        samples = np.arange(0.0, max(1.0, first_hit - 2.0), 2.0)
        points = source[None, :] + samples[:, None] * direction[None, :]
        px = np.rint(points[:, 0] + width / 2.0).astype(int)
        py = np.rint(-points[:, 1] + height / 2.0).astype(int)
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if inside.all() and free[py, px].all():
            return True
    return False


def audit(name: str, record: dict, indices: list[int]) -> dict:
    mission = map_class(name)(
        drone_type=DroneMotionless, number_drones=10,
        zones_config=(ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE),
    )
    try:
        rendered = ExploredMap()
        rendered.initialize_walls(mission.playground)
        free = (rendered._map_playground != 0).astype(np.uint8)  # evaluation only
        height, width = free.shape
        # Empty drone radius is about 16.4 px.  The earlier 23 px kernel only
        # reserved 11 px and could label physically unreachable wall-adjacent
        # viewpoints as legal; 35 px reserves 17 px conservatively.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        standing = cv2.erode(free, kernel)
        trajectories = [
            (state["id"], row) for state in record["controller_states"]
            for row in state.get("pose_trace", ())
        ]
        poses = np.asarray([row["pose"] for _, row in trajectories], float)
        tree = cKDTree(poses) if len(poses) else None
        bombs = load_fixed_layouts(ROOT / "config" / "zones_location.txt")[name]
        result = []
        for index in indices:
            bomb = np.array([bombs[index]["x"], bombs[index]["y"]], float)
            bx, by = int(round(bomb[0] + width / 2)), int(round(-bomb[1] + height / 2))
            radius = 180
            x0, x1 = max(0, bx - radius), min(width, bx + radius + 1)
            y0, y1 = max(0, by - radius), min(height, by + radius + 1)
            candidates = []
            for py in range(y0, y1, 5):
                for px in range(x0, x1, 5):
                    if not standing[py, px]:
                        continue
                    point = np.array([px - width / 2.0, height / 2.0 - py])
                    distance = float(np.linalg.norm(point - bomb))
                    if 28.0 <= distance <= 125.0 and visible(
                            free, point, bomb, width, height):
                        candidates.append(point)
            nearest = None
            if tree is not None and candidates:
                distances, positions = tree.query(np.asarray(candidates), k=1)
                choice = int(np.argmin(distances))
                drone, row = trajectories[int(positions[choice])]
                nearest = {
                    "candidate": candidates[choice].tolist(),
                    "trajectory_distance": round(float(distances[choice]), 2),
                    "drone": int(drone), "step": int(row["step"]),
                    "trajectory_pose": row["pose"],
                    "gps_valid": bool(row["gps_valid"]),
                }
            actual_visible = []
            actual_semantic = []
            for drone, row in trajectories:
                pose = np.asarray(row["pose"], float)
                separation = float(np.linalg.norm(pose - bomb))
                if separation <= 125.0 and visible(free, pose, bomb, width, height):
                    actual_visible.append((drone, row["step"], separation))
                if ("heading" in row and separation <= 142.0
                        and semantic_ray_hits(free, pose, float(row["heading"]),
                                              bomb, width, height)):
                    actual_semantic.append((drone, row["step"], separation))
            result.append({
                "index": index, "bomb": bomb.tolist(),
                "visible_standing_candidates_5px": len(candidates),
                "nearest_candidate_to_trajectory": nearest,
                "trajectory_positions_with_wall_los": len(actual_visible),
                "first_wall_los": (None if not actual_visible else {
                    "drone": int(actual_visible[0][0]),
                    "step": int(actual_visible[0][1]),
                    "distance": round(float(actual_visible[0][2]), 2),
                }),
                "trajectory_positions_with_35_ray_hit": len(actual_semantic),
                "first_35_ray_hit": (None if not actual_semantic else {
                    "drone": int(actual_semantic[0][0]),
                    "step": int(actual_semantic[0][1]),
                    "distance": round(float(actual_semantic[0][2]), 2),
                }),
            })
        return {"map": name, "bombs": result}
    finally:
        mission.playground.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("record", type=Path)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.map, json.loads(args.record.read_text(encoding="utf-8")), args.indices)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
