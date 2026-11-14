"""Run isolated, ground-truth-only dynamics calibration on the official body.

This is an evaluation tool, never imported by the competition controller.
It deliberately uses the physical body's true pose and velocity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from swarm_rescue.solutions.blue_pid_explore.dynamics import fit_discrete_dynamics


def _normalized_xy(forward: float, lateral: float) -> tuple[float, float]:
    norm = math.hypot(forward, lateral)
    if norm > 1.0:
        return forward / norm, lateral / norm
    return forward, lateral


def _world_command(forward: float, lateral: float, heading: float) -> np.ndarray:
    forward, lateral = _normalized_xy(forward, lateral)
    return np.array([
        math.cos(heading) * forward - math.sin(heading) * lateral,
        math.sin(heading) * forward + math.cos(heading) * lateral,
    ])


def _official_runtime():
    try:
        from swarm_rescue.simulation.drone.agent import Agent
        from swarm_rescue.simulation.drone.drone_base import DroneBase
        from swarm_rescue.simulation.gui_map.playground import Playground
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Official simulator dependency missing. Install the project first: "
            "python -m pip install -e .[dev]"
        ) from exc

    class CalibrationAgent(Agent):
        def __init__(self):
            super().__init__()
            self.add_base(DroneBase())

    playground = Playground(size=(20000, 20000), seed=0, use_shaders=False)
    agent = CalibrationAgent()
    # The arena contains no other physical entity.  ``Agent.parts`` is only
    # provided by DroneAbstract, so allow overlap here to bypass an irrelevant
    # overlap query while retaining the exact official DroneBase rigid body.
    playground.add(agent, ((0.0, 0.0), 0.0), allow_overlapping=True)
    playground.reset()
    return playground, agent


def _reset(playground, agent) -> None:
    playground.reset()
    agent.base._pm_body.position = (0.0, 0.0)
    agent.base._pm_body.velocity = (0.0, 0.0)
    agent.base._pm_body.angle = 0.0
    agent.base._pm_body.angular_velocity = 0.0


def _frame(playground, agent, experiment: str, step: int, command: tuple[float, float, float]) -> dict:
    forward, lateral, rotation = command
    body = agent.base._pm_body
    position_before = np.asarray(body.position, dtype=float)
    velocity_before = np.asarray(body.velocity, dtype=float)
    heading_before = float(body.angle)
    command_world = _world_command(forward, lateral, heading_before)
    playground.step(all_commands={agent: {
        "forward": float(forward),
        "lateral": float(lateral),
        "rotation": float(rotation),
    }})
    position = np.asarray(body.position, dtype=float)
    velocity = np.asarray(body.velocity, dtype=float)
    return {
        "experiment": experiment,
        "step": step,
        "position_x": float(position[0]),
        "position_y": float(position[1]),
        "velocity_x": float(velocity[0]),
        "velocity_y": float(velocity[1]),
        "heading": float(body.angle),
        "command_forward": float(forward),
        "command_lateral": float(lateral),
        "command_rotation": float(rotation),
        "command_world_x": float(command_world[0]),
        "command_world_y": float(command_world[1]),
        "speed": float(np.linalg.norm(velocity)),
        "position_before": position_before.tolist(),
        "velocity_before": velocity_before.tolist(),
        "command_world": command_world.tolist(),
        "velocity_after": velocity.tolist(),
    }


def _run_sequence(playground, agent, name: str, commands) -> list[dict]:
    _reset(playground, agent)
    return [_frame(playground, agent, name, step, command) for step, command in enumerate(commands)]


def _brake_trial(playground, agent, target_speed: float) -> tuple[list[dict], dict]:
    _reset(playground, agent)
    rows = []
    step = 0
    while float(np.linalg.norm(agent.base._pm_body.velocity)) < target_speed and step < 300:
        rows.append(_frame(playground, agent, f"brake_{target_speed:g}", step, (1.0, 0.0, 0.0)))
        step += 1
    start_position = float(agent.base._pm_body.position.x)
    initial_speed = float(agent.base._pm_body.velocity.x)
    furthest_position = start_position
    brake_frames = 0
    while float(agent.base._pm_body.velocity.x) > 0.05 and brake_frames < 300:
        rows.append(_frame(playground, agent, f"brake_{target_speed:g}", step, (-1.0, 0.0, 0.0)))
        furthest_position = max(furthest_position, float(agent.base._pm_body.position.x))
        step += 1
        brake_frames += 1
    return rows, {
        "mode": "full_reverse",
        "target_speed": target_speed,
        "initial_speed": initial_speed,
        # Frame-resolved forward excursion; unlike the final position this is
        # not reduced if the last full reverse frame has begun moving backward.
        "stopping_distance": furthest_position - start_position,
        "brake_frames": brake_frames,
        "final_speed": float(agent.base._pm_body.velocity.x),
    }


def _write_raw(path: Path, rows: list[dict]) -> None:
    fields = [
        "experiment", "step", "position_x", "position_y", "velocity_x", "velocity_y",
        "heading", "command_forward", "command_lateral", "command_rotation",
        "command_world_x", "command_world_y", "speed",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    playground, agent = _official_runtime()
    rows = []
    rows += _run_sequence(playground, agent, "forward_step", [(1.0, 0.0, 0.0)] * 50 + [(0.0, 0.0, 0.0)] * 30)
    rows += _run_sequence(playground, agent, "lateral_step", [(0.0, 1.0, 0.0)] * 50 + [(0.0, 0.0, 0.0)] * 30)
    rows += _run_sequence(playground, agent, "steady_forward", [(1.0, 0.0, 0.0)] * 180 + [(0.0, 0.0, 0.0)] * 100)
    rows += _run_sequence(playground, agent, "diagonal", [(1.0, 1.0, 0.0)] * 100 + [(0.0, 0.0, 0.0)] * 40)
    rows += _run_sequence(playground, agent, "rotation", [(0.0, 0.0, 1.0)] * 80)
    rows += _run_sequence(playground, agent, "rotating_translation", [(0.7, 0.3, 1.0)] * 160)

    braking = []
    for speed in (2.0, 4.0, 6.0, 8.0, 10.0):
        trial_rows, summary = _brake_trial(playground, agent, speed)
        rows += trial_rows
        braking.append(summary)

    # Rotation-only rows have no translational excitation and are retained in
    # the raw log but excluded from the translational fit.
    fit_rows = [row for row in rows if row["experiment"] != "rotation"]
    scalar = fit_discrete_dynamics(fit_rows, matrix_input=False)
    matrix = fit_discrete_dynamics(fit_rows, matrix_input=True)

    _write_raw(args.output_dir / "dynamics_calibration_samples.csv", rows)
    with (args.output_dir / "braking_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(braking[0]))
        writer.writeheader()
        writer.writerows(braking)

    def serialise(result):
        return {
            "velocity_decay": result.model.velocity_decay,
            "input_matrix": result.model.input_matrix.tolist(),
            "bias": result.model.bias.tolist(),
            "train_mae": result.train_mae,
            "train_rmse": result.train_rmse,
            "validation_mae": result.validation_mae,
            "validation_rmse": result.validation_rmse,
            "train_count": result.train_count,
            "validation_count": result.validation_count,
        }

    matrix_improvement = scalar.validation_rmse - matrix.validation_rmse
    matrix_is_materially_better = (
        matrix_improvement > 1e-6
        and matrix.validation_rmse < 0.98 * scalar.validation_rmse
    )
    by_experiment = {
        name: [row for row in rows if row["experiment"] == name]
        for name in (
            "forward_step", "lateral_step", "steady_forward", "diagonal",
            "rotation", "rotating_translation",
        )
    }
    steady = by_experiment["steady_forward"]
    diagonal = by_experiment["diagonal"]
    rotation = by_experiment["rotation"]
    experiment_summary = {
        "one_frame_forward_velocity": by_experiment["forward_step"][0]["velocity_after"],
        "one_frame_lateral_velocity": by_experiment["lateral_step"][0]["velocity_after"],
        "steady_forward_speed_after_180_frames": steady[179]["speed"],
        "coast_distance_over_100_frames": steady[-1]["position_x"] - steady[179]["position_x"],
        "coast_final_speed": steady[-1]["speed"],
        "diagonal_velocity_after_100_frames": diagonal[99]["velocity_after"],
        "diagonal_command_norm_after_normalization": float(
            np.linalg.norm(diagonal[0]["command_world"])
        ),
        "rotation_rate_rad_per_frame": rotation[-1]["heading"] / len(rotation),
        "rotating_translation_final_speed": by_experiment["rotating_translation"][-1]["speed"],
    }
    document = {
        "status": "measured_with_official_playground",
        "units": {"time": "one strategy control frame", "position": "px", "velocity": "px/control_frame"},
        "simulator": {"pymunk_steps_per_control_frame": 10, "control_frame_integration_time": 1.0},
        "scalar_model": serialise(scalar),
        "matrix_model": serialise(matrix),
        "selected_model": "matrix" if matrix_is_materially_better else "scalar",
        "matrix_selection_rule": "validation RMSE improves by >=2% and >1e-6 px/control_frame",
        "experiment_summary": experiment_summary,
        "braking_trials": braking,
    }
    (args.output_dir / "dynamics_identification.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    playground.cleanup()
    playground.close_window()
    print(json.dumps(document, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
