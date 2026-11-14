"""Official-physics predictive boundary stopping experiments."""

from __future__ import annotations

import json
import math
import time

import numpy as np

from stage2_harness import (
    ROOT, cleanup_world, count_sign_changes, create_world, finalize_world,
    json_ready, legal_measurements, set_true_state, step_world, true_state,
)
from swarm_rescue.solutions.blue_pid_explore.braking_safety import (
    BoundaryConstraint, BrakingSafetyLayer,
)
from swarm_rescue.solutions.blue_pid_explore.nonlinear_braking import NonlinearStoppingModel
from swarm_rescue.solutions.blue_pid_explore.precision_stop import PrecisionStopController
from swarm_rescue.solutions.blue_pid_explore.predictive_brake import BrakingMode
from swarm_rescue.solutions.blue_pid_explore.velocity_controller import SteadyStateFeedbackVelocityController
from swarm_rescue.solutions.blue_pid_explore.velocity_estimator import LegalVelocityEstimator, TruthStateInput


REPORTS = ROOT / "reports"


def controller():
    return SteadyStateFeedbackVelocityController(closed_loop_pole=0.75)


def allocate_normal_tangent(nominal, normal, normal_value):
    tangent = np.array([-normal[1], normal[0]])
    tangent_nominal = float(np.asarray(nominal) @ tangent)
    limit = math.sqrt(max(0.0, 1.0 - normal_value**2))
    tangent_value = float(np.clip(tangent_nominal, -limit, limit))
    return normal_value * normal + tangent_value * tangent


def run_boundary_stop(
    *,
    variant: str,
    target_position,
    boundary: BoundaryConstraint,
    input_mode: str,
    seed: int,
    cruise_speed: float = 10.0,
    rotation_command: float = 0.0,
    robust_velocity_sigma: float = 0.0,
    robust_position_sigma: float = 0.0,
    velocity_confidence: float = 0.0,
    position_confidence: float = 0.0,
    fixed_margin: float = 0.0,
    max_frames: int = 600,
    safety_layer=None,
    completion_requires_safety: bool = False,
):
    playground, agent = create_world(seed)
    set_true_state(playground, agent)
    estimator = LegalVelocityEstimator()
    velocity_controller = controller()
    old = PrecisionStopController(controller(), cruise_speed=cruise_speed)
    safety = safety_layer or BrakingSafetyLayer()
    stop_model = NonlinearStoppingModel()
    target = np.asarray(target_position, dtype=float)
    normal = boundary.normal
    stable = 0
    rows = []
    complete = None
    try:
        for step in range(max_frames):
            true_position, true_velocity, true_heading = true_state(agent)
            if input_mode == "truth":
                estimate = TruthStateInput.from_values(true_position, true_velocity, true_heading)
            else:
                odometry, compass, gps = legal_measurements(agent)
                estimate = estimator.update(odometry=odometry, compass=compass, gps_position=gps)

            error = target - estimate.position
            distance = float(np.linalg.norm(error))
            direction = error / distance if distance > 1e-12 else np.zeros(2)
            speed = float(np.linalg.norm(estimate.velocity))
            if variant == "A_legacy":
                old_output = old.compute(estimate.position, estimate.velocity, target)
                target_velocity = old_output.target_velocity
                nominal = old_output.command_world.copy()
                safe_command = nominal.copy()
                mode = old_output.mode.value
                scalar = None
            else:
                if distance <= 4.0 and speed <= 1.5:
                    target_velocity = 0.28 * error - 0.55 * estimate.velocity
                    norm = float(np.linalg.norm(target_velocity))
                    if norm > 1.2:
                        target_velocity *= 1.2 / norm
                else:
                    target_velocity = cruise_speed * direction
                nominal = velocity_controller.compute(target_velocity, estimate.velocity).command_world
                remaining = float(boundary.offset - estimate.position @ normal)
                speed_normal = float(estimate.velocity @ normal)
                nominal_normal = float(nominal @ normal)

                if variant == "B_current_threshold":
                    unsafe = stop_model.stopping_distance(max(0.0, speed_normal)) + fixed_margin > remaining
                    normal_value = -1.0 if unsafe else nominal_normal
                    safe_command = allocate_normal_tangent(nominal, normal, normal_value)
                    mode = "CURRENT_THRESHOLD_BRAKE" if unsafe else "NORMAL"
                    scalar = None
                elif variant == "C_one_step_binary":
                    probe = safety.predictive.compute(
                        speed=speed_normal, nominal_thrust=nominal_normal,
                        remaining_distance=remaining, safety_margin=fixed_margin,
                    )
                    if probe.mode == BrakingMode.NORMAL:
                        normal_value = nominal_normal
                    else:
                        normal_value = -1.0
                    safe_command = allocate_normal_tangent(nominal, normal, normal_value)
                    mode = probe.mode.value
                    scalar = probe
                else:
                    safe_output = safety.compute(
                        position=estimate.position, velocity=estimate.velocity,
                        nominal_thrust=nominal, boundary=boundary,
                        safety_margin=fixed_margin,
                        velocity_uncertainty=(
                            robust_velocity_sigma
                            * (0.2 + 0.8 * min(1.0, abs(speed_normal) / 10.0))
                        ),
                        position_uncertainty=(
                            robust_position_sigma
                            * (0.25 + 0.75 * min(1.0, abs(speed_normal) / 2.0))
                        ),
                        velocity_confidence=velocity_confidence,
                        position_confidence=position_confidence,
                    )
                    safe_command = safe_output.command_world
                    scalar = safe_output.scalar
                    mode = scalar.mode.value

            started = time.perf_counter_ns()
            forward, lateral = step_world(
                playground, agent, safe_command, estimate.heading,
                rotation=rotation_command,
            )
            control_overhead = time.perf_counter_ns() - started
            if input_mode == "legal":
                estimator.set_previous_command(safe_command, rotation_command)

            true_remaining = float(boundary.offset - true_position @ normal)
            estimated_remaining = float(boundary.offset - estimate.position @ normal)
            violation = max(0.0, -true_remaining)
            nominal_normal = float(nominal @ normal)
            safe_normal = float(safe_command @ normal)
            if scalar is None:
                predicted_next_velocity = 0.95 * float(estimate.velocity @ normal) + 0.5729312289729012 * safe_normal
                predicted_dx = 0.97728842 * float(estimate.velocity @ normal) + 0.52907993 * safe_normal
                nonlinear_distance = stop_model.stopping_distance(max(0.0, float(estimate.velocity @ normal)))
                required = nonlinear_distance + fixed_margin
                residual = required - estimated_remaining
                feasible = mode != "INFEASIBLE_BRAKING"
                solve_ns = 0
                iterations = 0
            else:
                predicted_next_velocity = scalar.predicted_next_velocity
                predicted_dx = scalar.predicted_next_displacement
                frame_peak = scalar.current_frame_max_displacement
                nonlinear_distance = scalar.nonlinear_stopping_distance
                total_excursion = scalar.total_forward_excursion
                required = scalar.required_stopping_distance
                residual = scalar.safety_residual
                feasible = scalar.feasible
                solve_ns = scalar.compute_time_ns
                iterations = scalar.bisection_iterations
            if scalar is None:
                frame_peak = stop_model.frame_max_displacement(
                    float(estimate.velocity @ normal), safe_normal
                )
                total_excursion = max(frame_peak, predicted_dx + nonlinear_distance)
            rows.append({
                "step": step,
                "true_position": true_position,
                "estimated_position": estimate.position,
                "true_velocity": true_velocity,
                "estimated_velocity": estimate.velocity,
                "boundary_position": boundary.offset,
                "remaining_distance": estimated_remaining,
                "true_remaining_distance": true_remaining,
                "nominal_thrust": nominal,
                "safe_thrust": safe_command,
                "predicted_next_velocity": predicted_next_velocity,
                "predicted_next_displacement": predicted_dx,
                "current_frame_max_displacement": frame_peak,
                "nonlinear_stopping_distance": nonlinear_distance,
                "total_forward_excursion": total_excursion,
                "required_stopping_distance": required,
                "safety_margin": fixed_margin + position_confidence * robust_position_sigma,
                "safety_residual": residual,
                "controller_mode": mode,
                "braking_feasible": feasible,
                "true_boundary_violation": violation,
                "command_forward": forward,
                "command_lateral": lateral,
                "solver_time_ns": solve_ns,
                "physics_step_time_ns": control_overhead,
                "bisection_iterations": iterations,
            })

            safe_completion = (
                scalar is None
                or (scalar.feasible and scalar.safety_residual <= 1e-9)
            )
            if (
                distance <= 1.0 and speed <= 0.15
                and (safe_completion or not completion_requires_safety)
            ):
                stable += 1
            else:
                stable = 0
            if stable >= 8 and complete is None:
                complete = step
            if complete is not None and step >= complete + 5:
                break
    finally:
        cleanup_world(playground)

    positions = [float(np.asarray(row["true_position"]) @ normal) for row in rows]
    commands = [float(np.asarray(row["safe_thrust"]) @ normal) for row in rows]
    brake_steps = [i for i, row in enumerate(rows) if row["controller_mode"] != "NORMAL" and row["controller_mode"] != "CRUISE"]
    switches = count_sign_changes(np.diff(commands), deadband=0.02)
    solve_times = np.asarray([row["solver_time_ns"] for row in rows], dtype=float) / 1000.0
    return {
        "variant": variant,
        "input_mode": input_mode,
        "seed": seed,
        "target_position": target,
        "boundary_normal": normal,
        "boundary_offset": boundary.offset,
        "metrics": {
            "max_forward_position": max(positions),
            "max_boundary_violation": max(0.0, max(positions) - boundary.offset),
            "boundary_violated": max(positions) > boundary.offset + 1e-9,
            "final_position_error": float(np.linalg.norm(target - np.asarray(rows[-1]["true_position"]))),
            "completion_frame": complete,
            "mean_speed": float(np.mean([np.linalg.norm(row["true_velocity"]) for row in rows])),
            "first_brake_frame": brake_steps[0] if brake_steps else None,
            "reverse_thrust_frames": sum(command < -0.05 for command in commands),
            "thrust_switches": switches,
            "reacceleration_count": sum(
                commands[i - 1] < -0.05 and commands[i] > 0.05 for i in range(1, len(commands))
            ),
            "mean_thrust_change": float(np.mean(np.abs(np.diff(commands)))) if len(commands) > 1 else 0.0,
            "saturation_ratio": float(np.mean([np.linalg.norm(row["safe_thrust"]) >= 0.999 for row in rows])),
            "solver_time_us_mean": float(np.mean(solve_times)),
            "solver_time_us_p95": float(np.percentile(solve_times, 95)),
            "solver_time_us_p99": float(np.percentile(solve_times, 99)),
            "solver_time_us_max": float(np.max(solve_times)),
            "bisection_calls": sum(row["bisection_iterations"] > 0 for row in rows),
            "stopping_distance_calls": len(rows),
            "infeasible_frames": sum(not row["braking_feasible"] for row in rows),
        },
        "trace": rows,
    }


def run_initial_speed_case(speed, extra_distance, label):
    model = NonlinearStoppingModel()
    boundary = BoundaryConstraint(np.array([1.0, 0.0]), 0.0)
    remaining = model.stopping_distance(speed) + extra_distance
    playground, agent = create_world(71)
    set_true_state(playground, agent, position=(-remaining, 0.0), velocity=(speed, 0.0))
    safety = BrakingSafetyLayer()
    trace = []
    try:
        for step in range(60):
            position, velocity, heading = true_state(agent)
            output = safety.compute(
                position=position, velocity=velocity, nominal_thrust=[1.0, 0.0], boundary=boundary
            )
            step_world(playground, agent, output.command_world, heading)
            post_position, post_velocity, _ = true_state(agent)
            trace.append({
                "step": step, "position": position, "velocity": velocity,
                "post_position": post_position, "post_velocity": post_velocity,
                "mode": output.scalar.mode.value, "feasible": output.scalar.feasible,
                "safe_thrust": output.command_world,
            })
            if float(agent.base._pm_body.velocity.x) <= 0.0:
                break
    finally:
        cleanup_world(playground)
    max_position = max(
        max(float(np.asarray(row["position"])[0]), float(np.asarray(row["post_position"])[0]))
        for row in trace
    )
    return {
        "speed": speed, "scenario": label, "extra_distance": extra_distance,
        "initial_remaining": remaining, "max_violation": max(0.0, max_position),
        "initial_mode": trace[0]["mode"], "initial_feasible": trace[0]["feasible"],
        "trace": trace,
    }


def main() -> int:
    comparisons = []
    boundary = BoundaryConstraint(np.array([1.0, 0.0]), 500.0)
    for variant in (
        "A_legacy", "B_current_threshold", "C_one_step_binary",
        "D_bisection", "E_robust",
    ):
        for mode in ("truth", "legal"):
            kwargs = {}
            if variant == "E_robust" and mode == "legal":
                kwargs = dict(
                    robust_velocity_sigma=0.55, robust_position_sigma=1.0,
                    velocity_confidence=1.0, position_confidence=1.0,
                    fixed_margin=0.25,
                )
            comparisons.append(run_boundary_stop(
                variant=variant, target_position=[500.0, 0.0], boundary=boundary,
                input_mode=mode, seed=31, **kwargs,
            ))

    initial_speed_cases = []
    for speed in (2.0, 4.0, 6.0, 8.0, 10.0, 11.46):
        for extra, label in ((0.0, "just_enough"), (2.0, "slightly_extra"), (20.0, "ample"), (-5.0, "infeasible")):
            initial_speed_cases.append(run_initial_speed_case(speed, extra, label))

    two_dimensional = []
    cases = [
        ("x", [100.0, 0.0], [1.0, 0.0], 100.0, 0.0),
        ("y_lateral", [0.0, 100.0], [0.0, 1.0], 100.0, 0.0),
        ("diagonal", [70.710678, 70.710678], [1.0, 1.0], 100.0, 0.0),
        ("rotating_diagonal", [70.710678, 70.710678], [1.0, 1.0], 100.0, 0.5),
        ("tangent_preservation", [100.0, 100.0], [1.0, 0.0], 100.0, 0.0),
    ]
    for name, target, normal, offset, rotation in cases:
        run = run_boundary_stop(
            variant="D_bisection", target_position=target,
            boundary=BoundaryConstraint(np.asarray(normal), offset),
            input_mode="truth", seed=43, cruise_speed=8.0,
            rotation_command=rotation,
        )
        run["case"] = name
        two_dimensional.append(run)

    finalize_world()
    document = {
        "schema": "stage25_predictive_runs_v1",
        "comparison_runs": comparisons,
        "initial_speed_cases": initial_speed_cases,
        "two_dimensional_runs": two_dimensional,
    }
    (REPORTS / "stage25_predictive_runs.json").write_text(
        json.dumps(json_ready(document), indent=2) + "\n", encoding="utf-8"
    )
    for run in comparisons:
        print(run["variant"], run["input_mode"], run["metrics"]["max_boundary_violation"], run["metrics"]["completion_frame"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
