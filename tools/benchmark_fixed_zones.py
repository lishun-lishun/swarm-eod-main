"""Run one official competition round using config/zones_location.txt.

The text file is the direct log produced by manual bomb placement.  This
evaluator converts only its declared map names and ``Bomb added`` coordinates
to the normal bombs JSON schema.  Truth coordinates are supplied to Launcher,
never to the controller.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_competition import parse  # noqa: E402


MAP_RE = re.compile(r"^\* Map: (Map\w+),")
BOMB_RE = re.compile(
    r"^Bomb added at \((-?[0-9.]+),\s*(-?[0-9.]+)\)\. Total=(\d+)"
)
ZONES = ["NO_COM_ZONE", "NO_GPS_ZONE"]


def load_fixed_layouts(path: Path) -> dict[str, list[dict[str, float]]]:
    layouts: dict[str, list[dict[str, float]]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        map_match = MAP_RE.match(raw.strip())
        if map_match:
            current = map_match.group(1)
            layouts[current] = []
            continue
        bomb_match = BOMB_RE.match(raw.strip())
        if bomb_match and current is not None:
            layouts[current].append(
                {"x": float(bomb_match.group(1)),
                 "y": float(bomb_match.group(2)), "theta": 0.0}
            )
    invalid = {name: len(bombs) for name, bombs in layouts.items()
               if len(bombs) != 10}
    if invalid:
        raise ValueError(f"expected 10 bombs per map, got {invalid}")
    return layouts


def write_inputs(map_name: str, bombs: list[dict[str, float]]) -> tuple[Path, Path]:
    work = ROOT / "reports" / "fixed_zones_work"
    work.mkdir(parents=True, exist_ok=True)
    plan = work / f"plan_{map_name}.yml"
    bomb_file = work / f"bombs_{map_name}.json"
    plan.write_text(
        "\n".join(
            [
                "stat_saving_enabled: false",
                "state_recording_enabled: " + (
                    "true" if os.environ.get("STATE_RECORDING_AUDIT", "0") == "1"
                    else "false"),
                "video_capture_enabled: false",
                "exploration_threshold_for_time_bonus: 90",
                "evaluation_plan:",
                f"  - map_name: {map_name}",
                "    team_mode: rescue",
                "    nb_rounds: 1",
                "    config_weight: 1",
                "    number_drones: 10",
                "    zones_config: [NO_COM_ZONE, NO_GPS_ZONE]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    bomb_file.write_text(
        json.dumps(
            {"map_name": map_name, "zones_config": ZONES, "bombs": bombs},
            indent=2,
        ),
        encoding="utf-8",
    )
    return plan, bomb_file


def run(map_name: str, engine_seed: int, source: Path) -> dict:
    from swarm_rescue.launcher import Launcher
    from swarm_rescue.solutions.blue_pid_explore.pose_sict_drone import (
        _ACTIVE_DRONES,
        controller_state_snapshot,
        diagnostics_snapshot,
    )

    layouts = load_fixed_layouts(source)
    if map_name not in layouts:
        raise ValueError(f"{map_name} not found; available: {sorted(layouts)}")
    plan, bomb_file = write_inputs(map_name, layouts[map_name])
    random.seed(engine_seed)
    np.random.seed(engine_seed)
    original_default_rng = np.random.default_rng
    disposal_observations = []
    initial_bomb_positions = {}
    audit_disposal_events = os.environ.get("V6_EVENT_AUDIT", "0") == "1"
    if audit_disposal_events:
        from swarm_rescue.simulation.elements.bomb import Bomb
        from swarm_rescue.simulation.elements.disposal_center import DisposalCenter
        original_bomb_pre_step = Bomb.pre_step
        original_activate = DisposalCenter.activate

        def observed_bomb_pre_step(entity):
            initial_bomb_positions.setdefault(
                id(entity), tuple(float(v) for v in entity.true_position()))
            return original_bomb_pre_step(entity)

        def observed_activate(center, entity):
            if not entity.disposed:
                disposal_observations.append({
                    "step": int(center._playground.timestep),
                    "initial_position": initial_bomb_positions.get(id(entity)),
                })
            return original_activate(center, entity)

        Bomb.pre_step = observed_bomb_pre_step
        DisposalCenter.activate = observed_activate
    np.random.default_rng = lambda value=None: original_default_rng(
        engine_seed if value is None else value
    )
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            launcher = Launcher(
                config_path=str(plan), default_bombs_file=str(bomb_file)
            )
            ok = launcher.go(
                hide_solution_output=False, headless=True, competition_mode=True
            )
    finally:
        np.random.default_rng = original_default_rng
        if audit_disposal_events:
            Bomb.pre_step = original_bomb_pre_step
            DisposalCenter.activate = original_activate
    result = parse(stream.getvalue())
    result.update(
        map=map_name,
        engine_seed=engine_seed,
        fixed_layout_source=str(source),
        zones_config=ZONES,
        ok=bool(ok),
        plan=str(plan),
        bomb_file=str(bomb_file),
    )
    result["controller_diagnostics"] = diagnostics_snapshot()
    result["controller_states"] = controller_state_snapshot()
    result["controller_task_events"] = {
        str(drone.me): list(getattr(drone, "task_events", ()))
        for drone in list(_ACTIVE_DRONES)
        if getattr(drone, "ready", False)
    }
    if audit_disposal_events:
        for event in disposal_observations:
            initial = event["initial_position"]
            if initial is None:
                event["bomb_index"] = None
                continue
            distances = [np.hypot(initial[0] - bomb["x"],
                                  initial[1] - bomb["y"])
                         for bomb in layouts[map_name]]
            nearest = int(np.argmin(distances))
            event["bomb_index"] = nearest if distances[nearest] <= 35.0 else None
        result["official_disposal_events"] = disposal_observations
    # Evaluation-only truth audit, executed after Launcher has completed all
    # controller calls.  These values never re-enter a controller decision.
    truth_audit = []
    for index, bomb in enumerate(layouts[map_name]):
        per_drone = []
        for drone in list(_ACTIVE_DRONES):
            if not getattr(drone, "ready", False):
                continue
            grid = drone.grid
            iy, ix = grid.cell(bomb["x"], bomb["y"])
            component_labels, _ = ndimage.label(grid.obstacle)
            component_id = int(component_labels[iy, ix])
            component_size = (0 if component_id == 0 else
                              int(np.count_nonzero(
                                  component_labels == component_id)))
            reachable_free = (grid.occ < -0.6)
            if getattr(drone, "cum", None) is not None:
                reachable_free &= np.isfinite(drone.cum)
            distances = np.hypot(grid.X - bomb["x"], grid.Y - bomb["y"])
            nearest_reachable_free = (
                float(np.min(distances[reachable_free]))
                if reachable_free.any() else float("inf")
            )
            per_drone.append({
                "id": int(drone.me),
                "occupancy": round(float(grid.occ[iy, ix]), 3),
                "obstacle": bool(grid.obstacle[iy, ix]),
                "scan_step": int(grid.scan_f[iy, ix]),
                "near_step": int(grid.near_f[iy, ix]),
                "clearance": round(float(grid.clearance[iy, ix]), 2),
                "obstacle_component_cells": component_size,
                "nearest_reachable_free_px": round(
                    nearest_reachable_free, 2),
                "reachable_free_within_semantic_range": int(
                    np.count_nonzero(
                        reachable_free & (distances <= 130.0))),
                "v6_ray_seen_step": (
                    int(drone.v6_adapter.visibility.seen_at[iy, ix]) - 1
                    if getattr(drone, "v6_adapter", None) is not None
                    and drone.v6_adapter.visibility.seen_at[iy, ix] > 0
                    else None),
            })
        truth_audit.append({
            "index": index,
            "position": [bomb["x"], bomb["y"]],
            "controller_maps": per_drone,
        })
    result["truth_audit_after_control"] = truth_audit
    if os.environ.get("VISIBLE_STANCE_AUDIT", "0") == "1":
        # Evaluation only, after every controller action has finished.  Truth
        # walls and fixed bomb positions never enter a drone's policy.
        import cv2
        from audit_fixed_reachability import map_class
        from swarm_rescue.simulation.drone.drone_motionless import DroneMotionless
        from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
        from swarm_rescue.simulation.reporting.explored_map import ExploredMap

        wanted = {int(v) for v in os.environ.get(
            "VISIBLE_STANCE_AUDIT_INDICES", "").split(",") if v.strip()}
        mission = map_class(map_name)(
            drone_type=DroneMotionless, number_drones=10,
            zones_config=(ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE),
        )
        try:
            rendered = ExploredMap()
            rendered.initialize_walls(mission.playground)
            truth_free = (rendered._map_playground != 0).astype(np.uint8)
            height, width = truth_free.shape
            hull = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            standing = cv2.erode(truth_free, hull)
            stance_audit = []
            for index, bomb in enumerate(layouts[map_name]):
                if wanted and index not in wanted:
                    continue
                target = np.array([bomb["x"], bomb["y"]], float)
                bx = int(round(target[0] + width / 2.0))
                by = int(round(-target[1] + height / 2.0))
                candidates = []
                for py in range(max(0, by - 180), min(height, by + 181), 5):
                    for px in range(max(0, bx - 180), min(width, bx + 181), 5):
                        if not standing[py, px]:
                            continue
                        point = np.array([px - width / 2.0, height / 2.0 - py])
                        distance = float(np.linalg.norm(point - target))
                        if not 28.0 <= distance <= 125.0:
                            continue
                        samples = point + np.linspace(
                            0.0, max(0.0, 1.0 - 12.0 / distance),
                            max(2, int(distance / 2.0)))[:, None] * (target - point)
                        sx = np.rint(samples[:, 0] + width / 2.0).astype(int)
                        sy = np.rint(-samples[:, 1] + height / 2.0).astype(int)
                        if truth_free[sy, sx].all():
                            candidates.append(point)
                per_drone = []
                for drone in list(_ACTIVE_DRONES):
                    if not getattr(drone, "ready", False):
                        continue
                    grid = drone.grid
                    start = grid.free_start(
                        grid.costs["explore"], *drone.pose) if grid.costs is not None else None
                    explore_cum = (grid.dijkstra("explore", [start])[1]
                                   if start is not None else None)
                    own_obstacle = (grid.ev > 1.0) | grid.border
                    own_clearance = ndimage.distance_transform_edt(
                        ~own_obstacle) * 10.0
                    known_free = 0
                    observed_free = 0
                    clearance_safe = 0
                    own_free = 0
                    own_safe = 0
                    reachable = 0
                    for point in candidates:
                        iy, ix = grid.cell(*point)
                        free_here = bool(grid.occ[iy, ix] < -0.6)
                        clear_here = bool(grid.clearance[iy, ix] >= 17.0)
                        observed_free += int(free_here)
                        clearance_safe += int(clear_here)
                        own_free += int(grid.ev[iy, ix] < -0.6)
                        own_safe += int(own_clearance[iy, ix] >= 17.0)
                        if free_here and clear_here:
                            known_free += 1
                            if explore_cum is not None and np.isfinite(explore_cum[iy, ix]):
                                reachable += 1
                    per_drone.append({"id": int(drone.me),
                                      "observed_free": observed_free,
                                      "clearance_safe": clearance_safe,
                                      "own_free": own_free,
                                      "own_clearance_safe": own_safe,
                                      "known_free": known_free,
                                      "reachable_in_current_cost": reachable})
                stance_audit.append({
                    "index": index, "position": target.tolist(),
                    "truth_visible_stances": len(candidates),
                    "controller_maps": per_drone,
                })
            result["truth_stance_audit"] = stance_audit
        finally:
            mission.playground.cleanup()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--engine-seed", type=int, default=861)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "config" / "zones_location.txt"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.map, args.engine_seed, args.source.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
