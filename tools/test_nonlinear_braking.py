"""Validate nonlinear stopping distance against official Pymunk microsteps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stage2_harness import (
    ROOT, create_world, finalize_world, json_ready, set_true_state,
    step_world_microtrace, true_state,
)
from swarm_rescue.solutions.blue_pid_explore.nonlinear_braking import NonlinearStoppingModel


SPEEDS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.46]


def run_physics_trial(playground, agent, speed: float):
    set_true_state(playground, agent, velocity=(speed, 0.0))
    start = float(agent.base._pm_body.position.x)
    maximum = start
    frames = 0
    trace = []
    crossing = 0
    if speed <= 0:
        return 0.0, 0, 0, trace
    while frames < 100:
        _, _, micro = step_world_microtrace(playground, agent, [-1.0, 0.0], 0.0)
        frames += 1
        for sample in micro:
            maximum = max(maximum, float(sample["position"][0]))
            trace.append({
                "frame": frames,
                "microstep": sample["microstep"],
                "position": float(sample["position"][0] - start),
                "velocity": float(sample["velocity"][0]),
            })
        if float(agent.base._pm_body.velocity.x) <= 0.0:
            crossing = frames
            break
    return maximum - start, frames, crossing, trace


def main() -> int:
    reports = ROOT / "reports"
    model = NonlinearStoppingModel()
    playground, agent = create_world(101)
    results = []
    try:
        for speed in SPEEDS:
            recursive = model.stopping_recursive(speed)
            analytic = model.stopping_analytic(speed)
            microstep_model = model.stopping_microstep(speed)
            actual, frames, crossing, trace = run_physics_trial(playground, agent, speed)
            absolute = abs(actual - analytic.distance)
            results.append({
                "speed": speed,
                "recursive_distance": recursive.distance,
                "analytic_distance": analytic.distance,
                "physics_microstep_distance": actual,
                "microstep_model_distance": microstep_model.distance,
                "absolute_error": absolute,
                "relative_error": 0.0 if actual == 0 else absolute / actual,
                "predicted_frames": analytic.frames,
                "physics_frames": frames,
                "first_zero_crossing_frame": crossing,
                "furthest_frame": analytic.furthest_frame,
                "microstep_trace": trace,
            })
    finally:
        finalize_world()

    errors = np.asarray([row["absolute_error"] for row in results])
    document = {
        "schema": "stage25_braking_model_v1",
        "official_pymunk": True,
        "summary": {
            "max_absolute_error": float(np.max(errors)),
            "p95_absolute_error": float(np.percentile(errors, 95)),
            "analytic_recursive_max_difference": max(
                abs(row["analytic_distance"] - row["recursive_distance"]) for row in results
            ),
            "microstep_model_max_error": max(
                abs(row["microstep_model_distance"] - row["physics_microstep_distance"])
                for row in results
            ),
        },
        "results": results,
    }
    (reports / "stage25_braking_model.json").write_text(
        json.dumps(json_ready(document), indent=2) + "\n", encoding="utf-8"
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(SPEEDS, [r["analytic_distance"] for r in results], "o-", label="analytic")
    axis.plot(SPEEDS, [r["physics_microstep_distance"] for r in results], "x--", label="Pymunk microstep max")
    axis.plot(SPEEDS, [r["microstep_model_distance"] for r in results], ".", label="microstep recurrence")
    axis.set(xlabel="initial speed (px/frame)", ylabel="maximum forward distance (px)", title="Nonlinear stopping distance")
    axis.grid(True, alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(reports / "stage25_braking_curve.png", dpi=150); plt.close(fig)
    print(json.dumps(document["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
