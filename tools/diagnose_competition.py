"""Bounded official-Pymunk diagnostic; truth is read only by this evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from swarm_rescue.maps.map_01 import Map01
from swarm_rescue.solutions.blue_pid_explore.competition_drone import BluePidExploreDrone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "competition_diagnostic.json")
    args = parser.parse_args()
    random.seed(1)
    np.random.seed(1)
    mission = Map01(drone_type=BluePidExploreDrone, number_drones=10)
    layout = json.loads((ROOT / "reports" / "competition_work" / "bombs_Map01_1.json").read_text())
    mission.spawn_bombs(layout["bombs"])
    for drone in mission.drones:
        drone.set_competition_mode(True)
    mission.playground._compute_observations()
    try:
        for _ in range(args.steps):
            commands = {drone: drone.control() for drone in mission.drones}
            mission.playground.step(all_commands=commands)
        # Ground truth is enabled only after all controller calls are over;
        # this block is an external evaluator, never controller input.
        for drone in mission.drones:
            drone.set_competition_mode(False)
        rows = []
        for drone in mission.drones:
            estimate = drone.last_estimate
            rows.append({
                "drone_id": drone.drone_id,
                "mission": drone.mission.state.value,
                "tracks": len(drone.bombs.tracks),
                "track_states": {state.value: sum(t.status == state for t in drone.bombs.tracks.values())
                                 for state in type(next(iter(drone.bombs.tracks.values()), None).status)
                                 } if drone.bombs.tracks else {},
                "claims": len(drone.allocator.claims),
                "current_claim": drone.current_claim,
                "center_known": drone.center is not None,
                "goal": None if drone.last_goal is None else np.asarray(drone.last_goal).tolist(),
                "estimated_position": None if estimate is None else estimate.position.tolist(),
                "true_position": np.asarray(drone.true_position()).tolist(),
                "grasped": len(drone.grasped_bombs()),
                "path_points": len(drone.navigation.path),
                "stats": drone.stats,
            })
        payload = {
            "steps": args.steps,
            "disposed": len(mission.bombs) - len(mission.live_bombs),
            "remaining_bombs": len(mission.live_bombs),
            "drones": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        mission.playground.cleanup()


if __name__ == "__main__":
    main()
