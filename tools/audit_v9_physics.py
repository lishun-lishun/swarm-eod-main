"""Independent official Pymunk audit of V9 cargo and disposal contact.

Truth is read only after the controller has returned an action and is never
passed back to V9. This tool is for diagnosis, not submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_fixed_zones import ROOT, run
from swarm_rescue.solutions.v9_rescue.adapter import V9RescueDrone
from swarm_rescue.simulation.gui_map import gui_sr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="Map01")
    parser.add_argument("--engine-seed", type=int, default=861)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    original = gui_sr.run_control

    def observed_control(drone):
        action = original(drone)
        if not isinstance(drone, V9RescueDrone):
            return action
        step = int(drone.elapsed_timestep)
        if step % 10 == 0:
            cargo = drone.grasped_bombs()
            if cargo:
                bomb = cargo[0]
                center = bomb.disposal_center
                rows.append({
                    "step": step,
                    "drone": int(drone.identifier),
                    "phase": drone.mission.phase,
                    "drone_true": list(map(float, drone.true_position())),
                    "drone_yaw_true": float(drone.true_angle()),
                    "bomb_true": list(map(float, bomb.true_position())),
                    "center_true": list(map(float, center._pm_body.position)),
                    "action": action,
                })
        return action

    # The official run_control guard has already exited here. Never bypass it.
    gui_sr.run_control = observed_control
    try:
        result = run(args.map, args.engine_seed, ROOT / "config" / "zones_location.txt")
    finally:
        gui_sr.run_control = original
    result["v9_offline_cargo_truth"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["ok"]:
        raise RuntimeError("Official Launcher failed; inspect raw_output in audit JSON")
    print(f"{args.map} seed{args.engine_seed}: {result['disposed']}/10, "
          f"{result['official_total']} points, {len(rows)} cargo samples")


if __name__ == "__main__":
    main()
