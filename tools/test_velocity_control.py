"""Official rigid-body speed-control experiments for stage 2."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stage2_harness import (
    ROOT, cleanup_world, count_sign_changes, create_world, finalize_world, json_ready,
    legal_measurements, norm_metrics, step_world, true_state,
)

from swarm_rescue.solutions.blue_pid_explore.dynamics import MAX_STEADY_SPEED, predict_forward_braking_distance
from swarm_rescue.solutions.blue_pid_explore.velocity_controller import (
    InverseDynamicsVelocityController,
    SteadyStateFeedbackVelocityController,
)
from swarm_rescue.solutions.blue_pid_explore.velocity_estimator import LegalVelocityEstimator, TruthStateInput


REPORTS = ROOT / "reports"


EXPERIMENTS = {
    "T1_accel_5": [(80, (5.0, 0.0))],
    "T2_accel_10": [(110, (10.0, 0.0))],
    "T3_5_to_0": [(70, (5.0, 0.0)), (90, (0.0, 0.0))],
    "T4_10_to_0": [(120, (10.0, 0.0)), (110, (0.0, 0.0))],
    "T5_plus5_to_minus5": [(70, (5.0, 0.0)), (120, (-5.0, 0.0))],
    "T6_diagonal": [(100, (5.0 / math.sqrt(2), 5.0 / math.sqrt(2)))],
    "T6b_rotating_diagonal": [(140, (5.0 / math.sqrt(2), 5.0 / math.sqrt(2)))],
    "T11_full_speed_to_0": [(240, (MAX_STEADY_SPEED, 0.0)), (140, (0.0, 0.0))],
}


def expanded_schedule(phases):
    schedule = []
    changes = []
    for frames, target in phases:
        changes.append(len(schedule))
        schedule.extend([np.asarray(target, dtype=float)] * frames)
    return schedule, changes


def make_controller(name: str, pole: float = 0.75):
    if name == "inverse":
        return InverseDynamicsVelocityController()
    return SteadyStateFeedbackVelocityController(closed_loop_pole=pole)


def run_experiment(test_name: str, phases, controller_name: str, input_mode: str, *, pole: float = 0.75, seed: int = 0):
    playground, agent = create_world(seed)
    controller = make_controller(controller_name, pole)
    estimator = LegalVelocityEstimator()
    schedule, changes = expanded_schedule(phases)
    rows = []
    try:
        for step, target in enumerate(schedule):
            true_position, true_velocity, true_heading = true_state(agent)
            if input_mode == "truth":
                estimate = TruthStateInput.from_values(true_position, true_velocity, true_heading)
            else:
                odometry, compass, gps = legal_measurements(agent)
                estimate = estimator.update(odometry=odometry, compass=compass, gps_position=gps)

            output = controller.compute(target, estimate.velocity)
            rotation_command = 0.5 if test_name == "T6b_rotating_diagonal" else 0.0
            forward, lateral = step_world(
                playground, agent, output.command_world, estimate.heading,
                rotation=rotation_command,
            )
            if input_mode == "legal":
                estimator.set_previous_command(output.command_world, rotation=rotation_command)
            rows.append({
                "step": step,
                # State fields are all aligned at frame start; the command on
                # this row is the action applied over the following frame.
                "true_position": true_position,
                "estimated_position": estimate.position,
                "true_velocity": true_velocity,
                "estimated_velocity": estimate.velocity,
                "target_velocity": target,
                "velocity_error": target - true_velocity,
                "target_position": None,
                "position_error": None,
                "command_forward": forward,
                "command_lateral": lateral,
                "command_rotation": rotation_command,
                "command_world": output.command_world,
                "force_saturated": output.saturated,
                "controller_mode": output.controller,
            })
    finally:
        cleanup_world(playground)

    velocity_errors = [row["velocity_error"] for row in rows]
    estimation_errors = [row["estimated_velocity"] - row["true_velocity"] for row in rows]
    velocity_mae, velocity_rmse = norm_metrics(velocity_errors)
    estimation_mae, estimation_rmse = norm_metrics(estimation_errors)
    final_change = changes[-1]
    final_target = schedule[-1]
    final_errors = np.asarray(velocity_errors[final_change:])
    tolerance = max(0.10, 0.02 * float(np.linalg.norm(final_target)))
    settling = None
    for index in range(len(final_errors)):
        if np.all(np.linalg.norm(final_errors[index:], axis=1) <= tolerance):
            settling = index
            break

    target_norms = np.asarray([np.linalg.norm(row["target_velocity"]) for row in rows])
    overshoots = []
    for row, target_norm in zip(rows, target_norms):
        if target_norm > 1e-9:
            direction = np.asarray(row["target_velocity"]) / target_norm
            overshoots.append(max(0.0, float(np.asarray(row["true_velocity"]) @ direction - target_norm)))

    stop_distance = None
    predicted_stop_distance = None
    prediction_error = None
    if np.linalg.norm(final_target) < 1e-9 and final_change > 0:
        start_position = np.asarray(rows[final_change]["true_position"])
        initial_velocity = np.asarray(rows[final_change]["true_velocity"])
        speed = float(np.linalg.norm(initial_velocity))
        direction = initial_velocity / speed if speed > 1e-12 else np.array([1.0, 0.0])
        excursions = [float((np.asarray(row["true_position"]) - start_position) @ direction) for row in rows[final_change:]]
        stop_distance = max(0.0, max(excursions, default=0.0))
        predicted_stop_distance = predict_forward_braking_distance(speed).distance
        prediction_error = stop_distance - predicted_stop_distance

    axis = 0 if abs(final_target[0]) >= abs(final_target[1]) else 1
    scalar_errors = [float(np.asarray(row["velocity_error"])[axis]) for row in rows[final_change:]]
    metrics = {
        "velocity_mae": velocity_mae,
        "velocity_rmse": velocity_rmse,
        "velocity_estimation_mae": estimation_mae,
        "velocity_estimation_rmse": estimation_rmse,
        "velocity_overshoot": max(overshoots, default=0.0),
        "settling_frames_after_final_change": settling,
        "true_stop_distance": stop_distance,
        "predicted_stop_distance": predicted_stop_distance,
        "stop_prediction_error": prediction_error,
        "final_residual_speed": float(np.linalg.norm(rows[-1]["true_velocity"])),
        "force_saturation_ratio": float(np.mean([row["force_saturated"] for row in rows])),
        "oscillation_count": count_sign_changes(scalar_errors),
    }
    return {
        "test": test_name,
        "controller": controller_name,
        "closed_loop_pole": pole if controller_name == "feedback" else None,
        "input_mode": input_mode,
        "seed": seed,
        "metrics": metrics,
        "trace": rows,
    }


def plot_representative(results):
    selected = [r for r in results if r["test"] == "T4_10_to_0"]
    fig, axis = plt.subplots(figsize=(9, 5))
    for result in selected:
        speed = [np.linalg.norm(row["true_velocity"]) for row in result["trace"]]
        label = f"{result['controller']}/{result['input_mode']}"
        axis.plot(speed, label=label)
    target = [np.linalg.norm(row["target_velocity"]) for row in selected[0]["trace"]]
    axis.plot(target, "k--", label="target")
    axis.set(xlabel="control frame", ylabel="speed (px/frame)", title="T4: 10 -> 0 speed")
    axis.grid(True, alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(REPORTS / "stage2_velocity_time.png", dpi=150); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for result in selected:
        thrust = [row["command_forward"] for row in result["trace"]]
        label = f"{result['controller']}/{result['input_mode']}"
        axis.plot(thrust, label=label)
    axis.set(xlabel="control frame", ylabel="forward command", title="T4: active braking command")
    axis.grid(True, alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(REPORTS / "stage2_thrust_time.png", dpi=150); plt.close(fig)


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    results = []
    try:
        for test_name, phases in EXPERIMENTS.items():
            for controller in ("inverse", "feedback"):
                for mode in ("truth", "legal"):
                    results.append(run_experiment(test_name, phases, controller, mode, seed=17))

        pole_sweep = []
        for pole in (0.4, 0.6, 0.75):
            for test_name in ("T1_accel_5", "T2_accel_10"):
                for mode in ("truth", "legal"):
                    pole_sweep.append(run_experiment(test_name, EXPERIMENTS[test_name], "feedback", mode, pole=pole, seed=23))
    finally:
        finalize_world()

    plot_representative(results)
    document = {
        "schema": "swarm_rescue_stage2_velocity_v1",
        "official_physics": True,
        "results": results,
        "feedback_pole_sweep": pole_sweep,
    }
    (REPORTS / "stage2_velocity_results.json").write_text(
        json.dumps(json_ready(document), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(results)} main runs and {len(pole_sweep)} pole-sweep runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
