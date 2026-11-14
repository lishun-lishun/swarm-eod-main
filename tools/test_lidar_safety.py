"""Official-Pymunk/official-LiDAR stage-3 safety experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.elements.normal_wall import NormalWall
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.simulation.gui_map.playground import Playground
from swarm_rescue.simulation.utils.misc_data import MiscData
from swarm_rescue.solutions.blue_pid_explore.lidar_geometry import footprint_vertices_from_body
from swarm_rescue.solutions.blue_pid_explore.lidar_safety import LidarPredictiveSafetyLayer
from swarm_rescue.solutions.blue_pid_explore.velocity_controller import (
    SteadyStateFeedbackVelocityController, world_to_body,
)
from swarm_rescue.solutions.blue_pid_explore.velocity_estimator import LegalVelocityEstimator


ZERO: CommandsDict = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}


class SafetyPlayground(ClosedPlayground):
    def __init__(self):
        Playground.__init__(self, size=(700, 700), seed=0, background=(220, 220, 220), use_shaders=False)
        self._walls_creation(6)


class SafetyDrone(DroneAbstract):
    def define_message_for_all(self):
        return None

    def control(self):
        return dict(ZERO)


def add_wall(playground, begin, end):
    wall = NormalWall(pos_start=begin, pos_end=end)
    playground.add(wall, wall.wall_coordinates)
    return wall


def reset_agent(playground, drone, position, velocity, heading, seed):
    np.random.seed(seed)
    body = drone.base._pm_body
    body.position = position
    body.velocity = velocity
    body.angle = heading
    body.angular_velocity = 0.0
    playground.space.reindex_shapes_for_body(body)
    drone._drone_health = 100
    drone._last_collision_damage_timestep = None
    drone.gps()._noise_model._last_noise = None
    drone.compass()._noise_model._last_noise = None
    drone.odometer().prev_position = None
    drone.odometer().prev_angle = None
    playground._compute_observations()


def legal(drone):
    return drone.odometer().get_sensor_values(), drone.compass().get_sensor_values(), drone.gps().get_sensor_values()


def run_case(playground, drone, footprint, *, name, seed, position, velocity,
             heading, target_velocity, wall_clearance, steps=90, rotation=0.0):
    reset_agent(playground, drone, position, velocity, heading, seed)
    estimator = LegalVelocityEstimator()
    velocity_control = SteadyStateFeedbackVelocityController()
    safety = LidarPredictiveSafetyLayer()
    trace = []
    initial_health = drone.drone_health
    minimum_clearance = math.inf
    emergency = 0
    blocked = 0
    started = time.perf_counter()
    previous_mode = "NORMAL"
    for step in range(steps):
        odom, compass, gps = legal(drone)
        estimate = estimator.update(odometry=odom, compass=compass, gps_position=gps)
        nominal = velocity_control.compute(target_velocity, estimate.velocity).command_world
        safe = safety.compute(
            lidar_distances=drone.lidar_values(), lidar_angles=drone.lidar_rays_angles(),
            estimated_position=estimate.position, estimated_heading=estimate.heading,
            estimated_velocity=estimate.velocity, nominal_thrust=nominal,
            footprint_vertices=footprint, velocity_sigma=0.45, position_sigma=1.5,
        )
        body_cmd = world_to_body(safe.command_world, estimate.heading)
        playground.step(all_commands={drone: {
            "forward": float(body_cmd[0]), "lateral": float(body_cmd[1]),
            "rotation": float(rotation), "grasper": 0,
        }})
        estimator.set_previous_command(safe.command_world, rotation)
        true_position = np.asarray(drone.base._pm_body.position, dtype=float)
        true_velocity = np.asarray(drone.base._pm_body.velocity, dtype=float)
        clearance = float(wall_clearance(true_position, float(drone.base._pm_body.angle), footprint))
        minimum_clearance = min(minimum_clearance, clearance)
        mode = safe.joint.status.value
        emergency += int(mode != "NORMAL" and previous_mode == "NORMAL")
        blocked += int(safe.replan_required)
        previous_mode = mode
        trace.append({
            "step": step, "true_position": true_position.tolist(),
            "estimated_position": estimate.position.tolist(),
            "true_velocity": true_velocity.tolist(), "estimated_velocity": estimate.velocity.tolist(),
            "nominal_thrust": nominal.tolist(), "safe_thrust": safe.command_world.tolist(),
            "safety_status": mode, "constraint_count": len(safe.boundaries),
            "max_residual": safe.joint.max_residual, "clearance": clearance,
            "compute_time_ms": safe.joint.compute_time_ns / 1e6,
        })
    elapsed = time.perf_counter() - started
    collisions = initial_health - drone.drone_health
    speeds = [np.linalg.norm(row["true_velocity"]) for row in trace]
    return {
        "scenario": name, "seed": seed, "collisions": collisions,
        "health_loss": collisions, "minimum_surface_clearance": minimum_clearance,
        "average_speed": float(np.mean(speeds)), "emergency_brakes": emergency,
        "blocked_frames": blocked, "mean_compute_ms": float(np.mean([r["compute_time_ms"] for r in trace])),
        "p95_compute_ms": float(np.percentile([r["compute_time_ms"] for r in trace], 95)),
        "walltime_seconds": elapsed, "trace": trace,
    }


def support(footprint, normal, heading):
    c, s = math.cos(heading), math.sin(heading)
    world = footprint @ np.array([[c, s], [-s, c]])
    return float(np.max(world @ np.asarray(normal)))


def main():
    playground = SafetyPlayground()
    drone = SafetyDrone(identifier=0, misc_data=MiscData(size_area=(700, 700), number_drones=1))
    playground.add(drone, ((0.0, 0.0), 0.0))
    # Shared test geometry: right wall and top wall form a corner.
    add_wall(playground, (100, -260), (100, 260))
    add_wall(playground, (-260, 100), (260, 100))
    footprint = footprint_vertices_from_body(drone.base)

    def right_clearance(p, h, f):
        return 97.0 - p[0] - support(f, [1, 0], h)

    def top_clearance(p, h, f):
        return 97.0 - p[1] - support(f, [0, 1], h)

    def corner_clearance(p, h, f):
        return min(right_clearance(p, h, f), top_clearance(p, h, f))

    cases = [
        dict(name="front_wall", position=(-100, -120), velocity=(0, 0), heading=0,
             target_velocity=(10, 0), wall_clearance=right_clearance),
        dict(name="diagonal_45", position=(-100, -100), velocity=(0, 0), heading=0,
             target_velocity=(8, 8), wall_clearance=corner_clearance),
        dict(name="parallel_wall", position=(70, -150), velocity=(0, 0), heading=0,
             target_velocity=(0, 8), wall_clearance=right_clearance),
        dict(name="corner", position=(-100, -100), velocity=(0, 0), heading=0.7,
             target_velocity=(8, 8), wall_clearance=corner_clearance),
        dict(name="rotating_approach", position=(-100, -120), velocity=(0, 0), heading=1.2,
             target_velocity=(10, 0), wall_clearance=right_clearance, rotation=0.35),
    ]
    runs = []
    try:
        for seed in range(10):
            for case in cases:
                runs.append(run_case(playground, drone, footprint, seed=seed, **case))
    finally:
        playground.cleanup()

    summaries = {}
    for name in [case["name"] for case in cases]:
        selected = [run for run in runs if run["scenario"] == name]
        summaries[name] = {
            "runs": len(selected), "collisions": sum(run["collisions"] for run in selected),
            "min_clearance": min(run["minimum_surface_clearance"] for run in selected),
            "mean_speed": float(np.mean([run["average_speed"] for run in selected])),
            "emergency_brakes": sum(run["emergency_brakes"] for run in selected),
            "blocked_frames": sum(run["blocked_frames"] for run in selected),
            "mean_compute_ms": float(np.mean([run["mean_compute_ms"] for run in selected])),
            "p95_compute_ms": float(np.percentile([run["p95_compute_ms"] for run in selected], 95)),
        }
    output = {"engine": "official Pymunk + official DroneLidar", "seeds": list(range(10)),
              "footprint_vertices": footprint.tolist(), "summaries": summaries, "runs": runs}
    reports = ROOT / "reports"
    (reports / "stage3_lidar_safety.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for run in runs:
        if run["seed"] == 0:
            xy = np.asarray([row["true_position"] for row in run["trace"]])
            axes[0].plot(xy[:, 0], xy[:, 1], label=run["scenario"])
    axes[0].axvline(97, color="black", linewidth=1); axes[0].axhline(97, color="black", linewidth=1)
    axes[0].axis("equal"); axes[0].legend(fontsize=7); axes[0].set_title("Official-Pymunk trajectories")
    axes[1].bar(summaries.keys(), [s["min_clearance"] for s in summaries.values()])
    axes[1].tick_params(axis="x", rotation=30); axes[1].set_ylabel("minimum surface clearance (px)")
    fig.tight_layout(); fig.savefig(reports / "stage3_lidar_safety.png", dpi=160); plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
