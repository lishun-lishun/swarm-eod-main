"""Evaluation-only reachability audit for manually placed fixed bombs.

This utility uses the official map geometry.  It never participates in a
controller run: the output is only meant to distinguish exploration failures
from layouts for which no collision-free grasping pose exists.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_fixed_zones import load_fixed_layouts  # noqa: E402
from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless  # noqa: E402
from swarm_rescue.simulation.elements.sensor_disablers import ZoneType  # noqa: E402
from swarm_rescue.simulation.reporting.explored_map import ExploredMap  # noqa: E402


def map_class(name: str):
    suffix = "07t4" if name == "Map07t4" else name[-2:]
    module = importlib.import_module(f"swarm_rescue.maps.map_{suffix}")
    return getattr(module, name)


def audit_map(name: str, bombs: list[dict[str, float]]) -> dict:
    mission = map_class(name)(
        drone_type=DroneMotionless,
        number_drones=10,
        zones_config=(ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE),
    )
    try:
        rendered = ExploredMap()
        rendered.initialize_walls(mission.playground)
        free = (rendered._map_playground != 0).astype(np.uint8)  # noqa: SLF001
        height, width = free.shape

        # The controller models a 10 px drone radius.  Eroding the rendered
        # free space by 11 px is conservative with respect to its convex hull.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        standing = cv2.erode(free, kernel)
        count, labels = cv2.connectedComponents(standing, connectivity=8)

        spawn_labels: set[int] = set()
        spawn_positions = []
        for drone in mission.drones:
            x, y = map(float, drone.true_position())
            px, py = round(x + width / 2), round(-y + height / 2)
            spawn_positions.append([x, y])
            if 0 <= px < width and 0 <= py < height and labels[py, px] != 0:
                spawn_labels.add(int(labels[py, px]))
            elif 0 <= px < width and 0 <= py < height:
                # Drones begin inside/against the rendered disposal centre,
                # which is intentionally absent from the navigable raster.
                # Associate the spawn with the nearest standing cell that a
                # drone can reach immediately after leaving that area.
                y0, y1 = max(0, py - 100), min(height, py + 101)
                x0, x1 = max(0, px - 100), min(width, px + 101)
                ys, xs = np.nonzero(labels[y0:y1, x0:x1])
                if len(xs):
                    nearest = int(np.argmin((xs + x0 - px) ** 2 +
                                            (ys + y0 - py) ** 2))
                    spawn_labels.add(int(labels[ys[nearest] + y0,
                                                xs[nearest] + x0]))
        component_sizes = {
            label: int(np.count_nonzero(labels == label))
            for label in range(1, count)
        }
        # The disposal centre can make different drones equidistant to several
        # sides of its rendered footprint.  Only the dominant surrounding
        # component represents the actual outside navigable region.
        if spawn_labels:
            dominant = max(spawn_labels, key=component_sizes.__getitem__)
            spawn_labels = {dominant}
        reachable = np.isin(labels, list(spawn_labels)).astype(np.uint8)

        # Distance to the nearest reachable drone-centre pose.  A grasper has
        # base radius + 5 px interaction range and overlaps a 12 px bomb at a
        # centre distance of roughly 27 px.
        distance = cv2.distanceTransform(1 - reachable, cv2.DIST_L2, 5)
        items = []
        for index, bomb in enumerate(bombs):
            px = int(round(bomb["x"] + width / 2))
            py = int(round(-bomb["y"] + height / 2))
            in_bounds = 0 <= px < width and 0 <= py < height
            d = float(distance[py, px]) if in_bounds else float("inf")
            items.append(
                {
                    "index": index,
                    "position": [bomb["x"], bomb["y"]],
                    "nearest_reachable_drone_center_px": round(d, 3),
                    "grasp_pose_exists_raster": bool(d <= 27.0),
                    "bomb_center_in_rendered_free_space": bool(
                        in_bounds and free[py, px]
                    ),
                }
            )
        return {
            "map": name,
            "map_shape": [height, width],
            "connected_free_components": count - 1,
            "spawn_component_labels": sorted(spawn_labels),
            "component_sizes": component_sizes,
            "spawn_positions": spawn_positions,
            "reachable_standing_pixels": int(reachable.sum()),
            "bombs": items,
        }
    finally:
        mission.playground.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("maps", nargs="*", default=["Map06", "Map07t4", "Map08"])
    parser.add_argument(
        "--source", type=Path, default=ROOT / "config" / "zones_location.txt"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "fixed_reachability.json"
    )
    args = parser.parse_args()
    layouts = load_fixed_layouts(args.source)
    result = {name: audit_map(name, layouts[name]) for name in args.maps}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
