"""Shared official-Pymunk harness for stage-2 experiments."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_rescue.simulation.drone.agent import Agent
from swarm_rescue.simulation.drone.drone_base import DroneBase
from swarm_rescue.simulation.drone.drone_sensors import DroneCompass, DroneGPS, DroneOdometer
from swarm_rescue.simulation.gui_map.playground import Playground


class SensorCalibrationAgent(Agent):
    """Exact official base with only the three legal stage-2 sensors."""

    def __init__(self):
        super().__init__()
        self.add_base(DroneBase())
        self.gps = DroneGPS()
        self.compass = DroneCompass()
        self.odometer = DroneOdometer()
        self.base.add_device(self.gps)
        self.base.add_device(self.compass)
        self.base.add_device(self.odometer)


_WORLD = None


def create_world(seed: int = 0):
    global _WORLD
    np.random.seed(seed)
    if _WORLD is None:
        playground = Playground(size=(20000, 20000), seed=seed, use_shaders=False)
        agent = SensorCalibrationAgent()
        playground.add(agent, ((0.0, 0.0), 0.0), allow_overlapping=True)
        _WORLD = (playground, agent)
    playground, agent = _WORLD
    # Reset persistent sensor history as well as physics.  Sensor.reset() does
    # not reset the AR(1) noise objects or odometer reference samples.
    agent.gps._noise_model._last_noise = None
    agent.compass._noise_model._last_noise = None
    agent.odometer.prev_position = None
    agent.odometer.prev_angle = None
    playground.reset()
    return playground, agent


def cleanup_world(playground) -> None:
    # Arcade 2.6 cannot reliably create a second hidden window after the first
    # one is closed in the same process.  Reuse the world for all runs.
    return None


def finalize_world() -> None:
    global _WORLD
    if _WORLD is not None:
        playground, _ = _WORLD
        playground.cleanup()
        playground.close_window()
        _WORLD = None


def true_state(agent):
    body = agent.base._pm_body
    return (
        np.asarray(body.position, dtype=float),
        np.asarray(body.velocity, dtype=float),
        float(body.angle),
    )


def legal_measurements(agent):
    gps = agent.gps.get_sensor_values()
    compass = agent.compass.get_sensor_values()
    odometry = agent.odometer.get_sensor_values()
    return (
        None if odometry is None else np.asarray(odometry, dtype=float).copy(),
        None if compass is None else float(compass),
        None if gps is None else np.asarray(gps, dtype=float).copy(),
    )


def step_world(playground, agent, command_world, heading_for_transform: float, rotation: float = 0.0):
    command = np.asarray(command_world, dtype=float)
    cosine, sine = math.cos(heading_for_transform), math.sin(heading_for_transform)
    forward = cosine * command[0] + sine * command[1]
    lateral = -sine * command[0] + cosine * command[1]
    playground.step(all_commands={agent: {
        "forward": float(forward),
        "lateral": float(lateral),
        "rotation": float(rotation),
    }})
    return float(forward), float(lateral)


def step_world_microtrace(playground, agent, command_world, heading_for_transform: float, rotation: float = 0.0):
    """Mirror Playground.step while exposing all ten physics microsteps."""
    command = np.asarray(command_world, dtype=float)
    cosine, sine = math.cos(heading_for_transform), math.sin(heading_for_transform)
    forward = cosine * command[0] + sine * command[1]
    lateral = -sine * command[0] + cosine * command[1]
    commands = {agent: {
        "forward": float(forward), "lateral": float(lateral),
        "rotation": float(rotation),
    }}
    playground._pre_step()
    playground._apply_commands(commands)
    samples = []
    for microstep in range(10):
        playground.space.step(0.1)
        position, velocity, heading = true_state(agent)
        samples.append({
            "microstep": microstep + 1,
            "position": position.copy(),
            "velocity": velocity.copy(),
            "heading": heading,
        })
    playground._compute_observations()
    playground._post_step()
    playground._timestep += 1
    return float(forward), float(lateral), samples


def set_true_state(playground, agent, *, position=(0.0, 0.0), velocity=(0.0, 0.0), heading=0.0):
    body = agent.base._pm_body
    body.position = tuple(position)
    body.velocity = tuple(velocity)
    body.angle = float(heading)
    body.angular_velocity = 0.0
    agent.odometer.prev_position = None
    agent.odometer.prev_angle = None
    playground._compute_observations()


def norm_metrics(errors) -> tuple[float, float]:
    values = np.linalg.norm(np.asarray(errors, dtype=float), axis=1)
    return float(np.mean(values)), float(np.sqrt(np.mean(values**2)))


def count_sign_changes(values, deadband: float = 0.05) -> int:
    signs = []
    for value in values:
        if value > deadband:
            sign = 1
        elif value < -deadband:
            sign = -1
        else:
            continue
        if not signs or sign != signs[-1]:
            signs.append(sign)
    return max(0, len(signs) - 1)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
