"""Truth-read-only end-state diagnostic for a failed official-map run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_competition import map_class, write_inputs, zones_for
from benchmark_fixed_zones import load_fixed_layouts
from swarm_rescue.simulation.elements.sensor_disablers import ZoneType
from swarm_rescue.solutions.blue_pid_explore.final_competition_drone import FinalCompetitionDrone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=5400)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixed-source", type=Path)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    zone_names = (["NO_COM_ZONE", "NO_GPS_ZONE"]
                  if args.fixed_source else zones_for(args.map))
    zone_tuple = tuple(ZoneType[name] for name in zone_names)
    mission = map_class(args.map)(drone_type=FinalCompetitionDrone,
                                  number_drones=10, zones_config=zone_tuple)
    if args.fixed_source:
        fixed = load_fixed_layouts(args.fixed_source.resolve())
        layout = {"map_name": args.map, "zones_config": zone_names,
                  "bombs": fixed[args.map]}
    else:
        _, bomb_path = write_inputs(args.map, args.seed)
        layout = json.loads(bomb_path.read_text(encoding="utf-8"))
    mission.spawn_bombs(layout["bombs"])
    for drone in mission.drones:
        drone.set_competition_mode(True)
    mission.playground._compute_observations()
    try:
        for _ in range(args.steps):
            commands = {drone: drone.control() for drone in mission.drones}
            mission.playground.step(all_commands=commands)
            if not mission.live_bombs:
                break
        for drone in mission.drones:
            drone.set_competition_mode(False)
        holders = {}
        for drone in mission.drones:
            for bomb in drone.grasped_bombs():
                holders[id(bomb)] = drone.me
        bombs = [{
            "index": index,
            "initial": [layout["bombs"][index]["x"], layout["bombs"][index]["y"]],
            "disposed": bool(bomb.disposed),
            "holder": holders.get(id(bomb)),
            "position": bomb.true_position().tolist(),
        } for index, bomb in enumerate(mission.bombs)]
        drones = []
        for drone in mission.drones:
            drones.append({
                "id": drone.me,
                "state": drone.state,
                "carrying": bool(drone.grasped_bombs()),
                "pose": None if drone.pose is None else list(drone.pose),
                "truth": drone.true_position().tolist(),
                "pose_error": (None if drone.pose is None else
                               float(np.linalg.norm(
                                   np.asarray(drone.pose) - drone.true_position()))),
                "accounted": drone.know.accounted(),
                "known_bombs": len(drone.know.bombs),
                "known_disposed": drone.know.disposed,
                "claim": drone.claim_key,
                "stuck_count": drone.stuck_count,
                "stats": {k: v for k, v in drone.stats.items()
                          if isinstance(v, (int, float, str, bool)) or v is None},
                "last_events": drone.task_events[-12:],
                "safety": drone.safety_stats,
            })
        result = {"map": args.map, "seed": args.seed,
                  "steps": int(mission.playground.timestep),
                  "disposed": len(mission.bombs) - len(mission.live_bombs),
                  "bombs": bombs, "drones": drones}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        mission.playground.cleanup()


if __name__ == "__main__":
    main()
