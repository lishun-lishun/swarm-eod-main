"""Official rigid-body precision stopping experiments and stage-2 report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stage2_harness import (
    ROOT, cleanup_world, count_sign_changes, create_world, finalize_world, json_ready,
    legal_measurements, norm_metrics, step_world, true_state,
)

from swarm_rescue.solutions.blue_pid_explore.precision_stop import PrecisionStopController, StopMode
from swarm_rescue.solutions.blue_pid_explore.velocity_controller import (
    InverseDynamicsVelocityController,
    SteadyStateFeedbackVelocityController,
)
from swarm_rescue.solutions.blue_pid_explore.velocity_estimator import LegalVelocityEstimator, TruthStateInput


REPORTS = ROOT / "reports"
STOP_TESTS = {
    "T7_stop_after_100px": (100.0, 8.0),
    "T8_stop_after_500px": (500.0, 10.0),
    "T9_low_speed_dock": (20.0, 2.0),
}


def make_controller(name):
    return InverseDynamicsVelocityController() if name == "inverse" else SteadyStateFeedbackVelocityController(closed_loop_pole=0.75)


def run_stop(
    test_name: str,
    target_x: float,
    cruise_speed: float,
    controller_name: str,
    input_mode: str,
    seed: int = 31,
    estimator_kwargs=None,
):
    playground, agent = create_world(seed)
    estimator = LegalVelocityEstimator(**(estimator_kwargs or {}))
    stop = PrecisionStopController(make_controller(controller_name), cruise_speed=cruise_speed)
    target_position = np.array([target_x, 0.0])
    rows = []
    max_frames = 900
    completed_at = None
    try:
        for step in range(max_frames):
            true_position, true_velocity, true_heading = true_state(agent)
            if input_mode == "truth":
                estimate = TruthStateInput.from_values(true_position, true_velocity, true_heading)
            else:
                odometry, compass, gps = legal_measurements(agent)
                estimate = estimator.update(odometry=odometry, compass=compass, gps_position=gps)
            output = stop.compute(estimate.position, estimate.velocity, target_position)
            forward, lateral = step_world(playground, agent, output.command_world, estimate.heading)
            if input_mode == "legal":
                estimator.set_previous_command(output.command_world)
            rows.append({
                "step": step,
                "true_position": true_position,
                "estimated_position": estimate.position,
                "true_velocity": true_velocity,
                "estimated_velocity": estimate.velocity,
                "target_velocity": output.target_velocity,
                "velocity_error": output.target_velocity - true_velocity,
                "target_position": target_position,
                "position_error": target_position - true_position,
                "command_forward": forward,
                "command_lateral": lateral,
                "command_rotation": 0.0,
                "command_world": output.command_world,
                "force_saturated": output.force_saturated,
                "controller_mode": output.mode.value,
                "predicted_stop_distance": output.predicted_stop_distance,
            })
            if output.mode == StopMode.COMPLETE and completed_at is None:
                completed_at = step
            if completed_at is not None and step >= completed_at + 12:
                break
    finally:
        cleanup_world(playground)

    velocity_mae, velocity_rmse = norm_metrics([row["velocity_error"] for row in rows])
    estimation_mae, estimation_rmse = norm_metrics([
        np.asarray(row["estimated_velocity"]) - np.asarray(row["true_velocity"]) for row in rows
    ])
    position_axis_errors = [float(np.asarray(row["position_error"])[0]) for row in rows]
    modes = [row["controller_mode"] for row in rows]
    brake_indices = [i for i, mode in enumerate(modes) if mode == StopMode.BRAKE.value]
    brake_index = brake_indices[0] if brake_indices else None
    predicted = rows[brake_index]["predicted_stop_distance"] if brake_index is not None else None
    actual_brake_excursion = None
    if brake_index is not None:
        start_x = float(np.asarray(rows[brake_index]["true_position"])[0])
        actual_brake_excursion = max(float(np.asarray(row["true_position"])[0]) - start_x for row in rows[brake_index:])
    metrics = {
        "velocity_mae": velocity_mae,
        "velocity_rmse": velocity_rmse,
        "velocity_estimation_mae": estimation_mae,
        "velocity_estimation_rmse": estimation_rmse,
        "completed": completed_at is not None,
        "completion_frame": completed_at,
        "true_stop_distance_from_brake_switch": actual_brake_excursion,
        "predicted_stop_distance_at_switch": predicted,
        "stop_prediction_error": None if predicted is None else actual_brake_excursion - predicted,
        "final_position_error": float(np.linalg.norm(rows[-1]["position_error"])),
        "final_residual_speed": float(np.linalg.norm(rows[-1]["true_velocity"])),
        "force_saturation_ratio": float(np.mean([row["force_saturated"] for row in rows])),
        "oscillation_count": count_sign_changes(position_axis_errors),
    }
    return {
        "test": test_name,
        "controller": controller_name,
        "input_mode": input_mode,
        "seed": seed,
        "target_position": target_position,
        "metrics": metrics,
        "trace": rows,
    }


def plot_stop_error(results):
    selected = [result for result in results if result["test"] == "T8_stop_after_500px"]
    fig, axis = plt.subplots(figsize=(9, 5))
    for result in selected:
        errors = [float(np.linalg.norm(row["position_error"])) for row in result["trace"]]
        axis.semilogy(np.maximum(errors, 1e-4), label=f"{result['controller']}/{result['input_mode']}")
    axis.set(xlabel="control frame", ylabel="position error (px, log)", title="T8: 500 px precision stop")
    axis.grid(True, alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(REPORTS / "stage2_stop_position_error.png", dpi=150); plt.close(fig)


def main() -> int:
    results = []
    try:
        for test_name, (target, speed) in STOP_TESTS.items():
            for controller in ("inverse", "feedback"):
                for mode in ("truth", "legal"):
                    results.append(run_stop(test_name, target, speed, controller, mode))
    finally:
        finalize_world()
    plot_stop_error(results)

    velocity_path = REPORTS / "stage2_velocity_results.json"
    if not velocity_path.exists():
        raise SystemExit("run tools/test_velocity_control.py first")
    velocity = json.loads(velocity_path.read_text(encoding="utf-8"))
    document = {
        "schema": "swarm_rescue_stage2_v1",
        "official_physics": True,
        "velocity_experiments": velocity["results"],
        "feedback_pole_sweep": velocity["feedback_pole_sweep"],
        "precision_stop_experiments": results,
    }
    (REPORTS / "stage2_results.json").write_text(
        json.dumps(json_ready(document), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(results)} precision-stop runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
