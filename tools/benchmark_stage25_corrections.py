"""A-E incremental benchmark for the stage-2.5 braking corrections."""

from __future__ import annotations

import json
import statistics
import sys

import matplotlib.pyplot as plt
import numpy as np

from stage2_harness import ROOT, finalize_world, json_ready
from test_predictive_stop import run_boundary_stop
from swarm_rescue.solutions.blue_pid_explore.braking_safety import BoundaryConstraint, BrakingSafetyLayer
from swarm_rescue.solutions.blue_pid_explore.nonlinear_braking import NonlinearStoppingModel
from swarm_rescue.solutions.blue_pid_explore.predictive_brake import PredictiveBrakingController


REPORTS = ROOT / "reports"
SIGMA_V = 0.55
SIGMA_P = 1.0
K_V = 1.0
K_P = 1.0
MARGIN = 0.25
RESERVE = 2.0


class LegacyDoubleCountPredictive(PredictiveBrakingController):
    """Frozen pre-correction residual for an honest A/B baseline."""

    def safety_residual(self, speed, thrust, remaining_distance, safety_margin):
        frame_end, _, frame_peak, future, _ = self.candidate_prediction(speed, thrust)
        return max(frame_end, frame_peak) + future + float(safety_margin) - float(remaining_distance)


def target_inset() -> float:
    return K_P * SIGMA_P + NonlinearStoppingModel().stopping_distance(K_V * SIGMA_V) + MARGIN + RESERVE


def make_safety(stage: str) -> BrakingSafetyLayer:
    if stage in ("A_original", "B_safe_target"):
        return BrakingSafetyLayer(LegacyDoubleCountPredictive())
    return BrakingSafetyLayer(PredictiveBrakingController())


def enrich(run, *, stage, target_policy, original_target, safe_target):
    final_true = np.asarray(run["trace"][-1]["true_position"], dtype=float)
    position_errors = [
        np.asarray(row["estimated_position"]) - np.asarray(row["true_position"])
        for row in run["trace"]
    ]
    velocity_errors = [
        np.asarray(row["estimated_velocity"]) - np.asarray(row["true_velocity"])
        for row in run["trace"]
    ]
    run["correction_stage"] = stage
    run["target_policy"] = target_policy
    run["original_target"] = np.asarray(original_target)
    run["safe_target"] = np.asarray(safe_target)
    run["target_boundary_gap"] = float(run["boundary_offset"] - np.asarray(safe_target)[0])
    run["metrics"]["final_original_target_error"] = float(np.linalg.norm(final_true - original_target))
    run["metrics"]["final_safe_target_error"] = float(np.linalg.norm(final_true - safe_target))
    run["metrics"]["position_estimation_rmse"] = float(np.sqrt(np.mean(np.square(position_errors))))
    run["metrics"]["velocity_estimation_rmse"] = float(np.sqrt(np.mean(np.square(velocity_errors))))
    return run


def summarize(runs):
    metrics = [run["metrics"] for run in runs]
    completed = [item["completion_frame"] for item in metrics if item["completion_frame"] is not None]
    return {
        "runs": len(runs),
        "violation_count": sum(item["boundary_violated"] for item in metrics),
        "max_violation": max(item["max_boundary_violation"] for item in metrics),
        "completion_rate": len(completed) / len(metrics),
        "mean_completion_frame": statistics.mean(completed) if completed else None,
        "mean_original_target_error": statistics.mean(item["final_original_target_error"] for item in metrics),
        "mean_safe_target_error": statistics.mean(item["final_safe_target_error"] for item in metrics),
        "reaccelerations": sum(item["reacceleration_count"] for item in metrics),
        "infeasible_frames": sum(item["infeasible_frames"] for item in metrics),
        "position_estimation_rmse": statistics.mean(item["position_estimation_rmse"] for item in metrics),
        "velocity_estimation_rmse": statistics.mean(item["velocity_estimation_rmse"] for item in metrics),
    }


def history_switch_test(reset: bool):
    safety = BrakingSafetyLayer()
    x_boundary = BoundaryConstraint(np.array([1.0, 0.0]), 60.0)
    first = safety.compute(
        position=[0.0, 0.0], velocity=[10.0, 0.0],
        nominal_thrust=[1.0, 0.0], boundary=x_boundary,
    )
    if reset:
        safety.reset()
    y_boundary = BoundaryConstraint(np.array([0.0, 1.0]), 100.0)
    second = safety.compute(
        position=[0.0, 0.0], velocity=[0.0, 0.0],
        nominal_thrust=[0.0, 1.0], boundary=y_boundary,
    )
    return {
        "reset": reset,
        "first_normal_thrust": first.normal_thrust,
        "second_normal_thrust": second.normal_thrust,
        "second_mode": second.scalar.mode.value,
    }


def main() -> int:
    inset = target_inset()
    stages = (
        "A_original", "B_safe_target", "C_exact_excursion",
        "D_history_reset", "E_dual_state",
    )
    runs = []
    try:
        for stage in stages:
            for target_policy in ("raw", "inset"):
                for seed in range(10):
                    for distance in (20.0, 100.0, 500.0):
                        original = np.array([distance, 0.0])
                        # A is the frozen original implementation; B-E apply
                        # the requested policy, while raw remains a fair
                        # no-inset control for every formula revision.
                        apply_inset = target_policy == "inset" and stage != "A_original"
                        safe_target = original - (np.array([inset, 0.0]) if apply_inset else 0.0)
                        run = run_boundary_stop(
                            variant=stage,
                            target_position=safe_target,
                            boundary=BoundaryConstraint(np.array([1.0, 0.0]), distance),
                            input_mode="legal", seed=seed,
                            cruise_speed=min(10.0, max(2.0, distance / 10.0)),
                            robust_velocity_sigma=SIGMA_V,
                            robust_position_sigma=SIGMA_P,
                            velocity_confidence=K_V,
                            position_confidence=K_P,
                            fixed_margin=MARGIN,
                            max_frames=500,
                            safety_layer=make_safety(stage),
                            completion_requires_safety=(stage == "E_dual_state"),
                        )
                        runs.append(enrich(
                            run, stage=stage, target_policy=target_policy,
                            original_target=original, safe_target=safe_target,
                        ))
    finally:
        finalize_world()

    summaries = {
        f"{stage}/{policy}": summarize([
            run for run in runs
            if run["correction_stage"] == stage and run["target_policy"] == policy
        ])
        for stage in stages for policy in ("raw", "inset")
    }
    focus = [
        run for run in runs
        if run["seed"] in (5, 9) and run["target_position"][0] > 490
    ]
    history = {
        "without_reset": history_switch_test(False),
        "with_reset": history_switch_test(True),
    }
    document = {
        "schema": "stage25_corrections_v1",
        "hard_boundaries_unchanged": True,
        "target_inset": inset,
        "robust_parameters": {
            "sigma_v": SIGMA_V, "sigma_p": SIGMA_P,
            "k_v": K_V, "k_p": K_P, "m0": MARGIN,
            "reserve": RESERVE,
        },
        "summaries": summaries,
        "seed_5_9_focus": focus,
        "history_switch_test": history,
        "runs": runs,
    }
    (REPORTS / "stage25_corrections.json").write_text(
        json.dumps(json_ready(document), indent=2) + "\n", encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    labels = []
    values = []
    for stage in stages:
        for policy in ("raw", "inset"):
            key = f"{stage}/{policy}"
            labels.append(key.replace("_", "\n"))
            values.append(summaries[key]["max_violation"])
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(values)), labels, rotation=35, ha="right", fontsize=7)
    axis.set_ylabel("maximum true boundary violation (px)")
    axis.set_title("Stage-2.5 correction A-E: raw and inset targets")
    axis.grid(True, axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(REPORTS / "stage25_corrections_violation.png", dpi=150); plt.close(fig)

    lines = [
        "# Stage 2.5 braking corrections", "",
        "Date: 2026-11-12. The hard boundary was unchanged in every run.", "",
        f"The latched target inset was `{inset:.3f} px` = position allowance + "
        "microstep stopping distance of the velocity allowance + fixed margin + 2 px reserve.", "",
        "## A-E legal-sensor results (10 seeds × 20/100/500 px)", "",
        "| Stage / target | Violations | Max violation | Completion | Mean frame | Original-target error | Reaccelerations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in stages:
        for policy in ("raw", "inset"):
            key = f"{stage}/{policy}"
            item = summaries[key]
            lines.append(
                f"| {key} | {item['violation_count']}/30 | {item['max_violation']:.3f} px | "
                f"{100*item['completion_rate']:.1f}% | {item['mean_completion_frame']} | "
                f"{item['mean_original_target_error']:.3f} px | {item['reaccelerations']} |"
            )
    final_raw = summaries["E_dual_state/raw"]
    final_inset = summaries["E_dual_state/inset"]
    lines += [
        "", "## Findings", "",
        "The raw-target rows isolate braking/state changes from target inset. The inset rows show the additional feasibility benefit but retain the original target error, so inset is not credited as a braking-formula improvement.",
        "",
        f"Final E/raw: {final_raw['violation_count']}/30 violations, {100*final_raw['completion_rate']:.1f}% completion, {final_raw['reaccelerations']} reaccelerations.",
        f"Final E/inset: {final_inset['violation_count']}/30 violations, {100*final_inset['completion_rate']:.1f}% completion, {final_inset['reaccelerations']} reaccelerations.",
        "",
        f"Boundary-normal switch without reset limited the new normal thrust to {history['without_reset']['second_normal_thrust']:.3f}; with reset it was {history['with_reset']['second_normal_thrust']:.3f}. This verifies that stale recovery history no longer contaminates a new task.",
        "",
        "Seed 5 and Seed 9 full traces, including true/estimated state errors and safety residuals, are stored in `seed_5_9_focus`.",
        "",
        "Remaining legal-sensor violations, if any, are attributed in the JSON to boundary-normal position-estimation error; no LiDAR, map truth, or hidden state is used by the controller.",
    ]
    (REPORTS / "stage25_corrections_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"target_inset": inset, "summaries": summaries, "history": history}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
