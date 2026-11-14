"""Official stage-3 U-shape, narrow-channel, and two-drone tests."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from test_lidar_safety import (
    SafetyPlayground, SafetyDrone, add_wall, reset_agent, run_case, support, legal,
)
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.solutions.blue_pid_explore.lidar_geometry import (
    dynamic_drone_boundary, footprint_vertices_from_body,
)
from swarm_rescue.solutions.blue_pid_explore.lidar_safety import LidarPredictiveSafetyLayer
from swarm_rescue.solutions.blue_pid_explore.velocity_controller import (
    SteadyStateFeedbackVelocityController, world_to_body,
)
from swarm_rescue.solutions.blue_pid_explore.velocity_estimator import LegalVelocityEstimator


def static_tests():
    world = SafetyPlayground()
    drone = SafetyDrone(identifier=0, misc_data=MiscData(size_area=(700, 700), number_drones=1))
    world.add(drone, ((-250, 0), 0))
    footprint = footprint_vertices_from_body(drone.base)
    # U-shaped obstacle opening to the left.
    add_wall(world, (100, -100), (100, 100))
    add_wall(world, (0, -100), (100, -100))
    add_wall(world, (0, 100), (100, 100))

    def u_clearance(p, h, f):
        return min(97 - p[0] - support(f, [1, 0], h),
                   abs(97 - abs(p[1])) - support(f, [0, 1], h))

    runs = []
    for seed in range(10):
        runs.append(run_case(
            world, drone, footprint, name="u_shape", seed=seed,
            position=(-100, 0), velocity=(0, 0), heading=0,
            target_velocity=(9, 0), wall_clearance=u_clearance, steps=70,
        ))
    world.cleanup()

    channel_world = SafetyPlayground()
    channel_drone = SafetyDrone(identifier=0, misc_data=MiscData(size_area=(700, 700), number_drones=1))
    channel_world.add(channel_drone, ((-250, 0), 0))
    channel_footprint = footprint_vertices_from_body(channel_drone.base)
    add_wall(channel_world, (-150, 18), (150, 18))
    add_wall(channel_world, (-150, -18), (150, -18))

    def channel_clearance(p, h, f):
        return 15 - abs(p[1]) - support(f, [0, 1 if p[1] >= 0 else -1], h)

    for seed in range(10):
        runs.append(run_case(
            channel_world, channel_drone, channel_footprint,
            name="narrow_channel", seed=seed, position=(-250, 0), velocity=(0, 0),
            heading=0, target_velocity=(8, 0), wall_clearance=channel_clearance, steps=50,
        ))
    channel_world.cleanup()
    return runs


def two_drone_test(seed):
    world = SafetyPlayground()
    misc = MiscData(size_area=(700, 700), number_drones=2)
    left = SafetyDrone(identifier=0, misc_data=misc)
    right = SafetyDrone(identifier=1, misc_data=misc)
    world.add(left, ((-120, 0), 0)); world.add(right, ((120, 0), math.pi))
    footprints = [footprint_vertices_from_body(left.base), footprint_vertices_from_body(right.base)]
    reset_agent(world, left, (-120, 0), (0, 0), 0, seed)
    reset_agent(world, right, (120, 0), (0, 0), math.pi, seed + 100)
    estimators = [LegalVelocityEstimator(), LegalVelocityEstimator()]
    controls = [SteadyStateFeedbackVelocityController(), SteadyStateFeedbackVelocityController()]
    safeties = [LidarPredictiveSafetyLayer(), LidarPredictiveSafetyLayer()]
    drones = [left, right]
    targets = [np.array([8.0, 0.0]), np.array([-8.0, 0.0])]
    initial_health = [d.drone_health for d in drones]
    min_gap = math.inf
    blocked = 0
    trace = []
    for step in range(80):
        estimates = []
        for drone, estimator in zip(drones, estimators):
            odom, compass, gps = legal(drone)
            estimates.append(estimator.update(odometry=odom, compass=compass, gps_position=gps))
        commands = {}
        modes = []
        for i in range(2):
            other = 1 - i
            nominal = controls[i].compute(targets[i], estimates[i].velocity).command_world
            dynamic = dynamic_drone_boundary(
                estimates[i].position, estimates[other].position, estimates[other].velocity,
                footprints[i], footprints[other], estimates[i].heading, estimates[other].heading,
                geometry_error=12.0,
            )
            result = safeties[i].compute(
                lidar_distances=drones[i].lidar_values(), lidar_angles=drones[i].lidar_rays_angles(),
                estimated_position=estimates[i].position, estimated_heading=estimates[i].heading,
                estimated_velocity=estimates[i].velocity, nominal_thrust=nominal,
                footprint_vertices=footprints[i], velocity_sigma=0.45, position_sigma=1.5,
                dynamic_boundaries=[dynamic],
            )
            body = world_to_body(result.command_world, estimates[i].heading)
            commands[drones[i]] = {"forward": float(body[0]), "lateral": float(body[1]),
                                   "rotation": 0.0, "grasper": 0}
            estimators[i].set_previous_command(result.command_world)
            modes.append(result.joint.status.value)
            blocked += int(result.replan_required)
        world.step(all_commands=commands)
        positions = [np.asarray(d.base._pm_body.position, dtype=float) for d in drones]
        delta = positions[1] - positions[0]
        n = delta / max(1e-9, np.linalg.norm(delta))
        gap = float(np.linalg.norm(delta) - support(footprints[0], n, drones[0].base._pm_body.angle)
                    - support(footprints[1], -n, drones[1].base._pm_body.angle))
        min_gap = min(min_gap, gap)
        trace.append({"step": step, "positions": [p.tolist() for p in positions],
                      "surface_gap": gap, "modes": modes})
    result = {"scenario": "two_drone_head_on", "seed": seed,
              "collisions": sum(h - d.drone_health for h, d in zip(initial_health, drones)),
              "minimum_surface_clearance": min_gap, "blocked_frames": blocked, "trace": trace}
    world.cleanup()
    return result


def main():
    static = static_tests()
    dynamic = [two_drone_test(seed) for seed in range(10)]
    runs = static + dynamic
    summary = {}
    for scenario in ("u_shape", "narrow_channel", "two_drone_head_on"):
        selected = [r for r in runs if r["scenario"] == scenario]
        summary[scenario] = {
            "runs": len(selected), "collisions": sum(r["collisions"] for r in selected),
            "min_clearance": min(r["minimum_surface_clearance"] for r in selected),
            "blocked_frames": sum(r["blocked_frames"] for r in selected),
        }
    payload = {"engine": "official Pymunk + official LiDAR", "summary": summary, "runs": runs,
               "claim": "two-drone result is preliminary; not full multi-drone coordination"}
    (ROOT / "reports" / "stage3_extended.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
