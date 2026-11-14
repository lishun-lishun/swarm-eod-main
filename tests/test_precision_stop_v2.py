import numpy as np

from swarm_rescue.solutions.blue_pid_explore.braking_safety import BoundaryConstraint
from swarm_rescue.solutions.blue_pid_explore.precision_stop_v2 import MotionMode, PrecisionStopControllerV2
from swarm_rescue.solutions.blue_pid_explore.predictive_brake import BrakingMode
from swarm_rescue.solutions.blue_pid_explore.velocity_controller import SteadyStateFeedbackVelocityController


def make_stop(**kwargs):
    return PrecisionStopControllerV2(
        SteadyStateFeedbackVelocityController(closed_loop_pole=0.75),
        cruise_speed=10.0,
        target_reserve=kwargs.pop("target_reserve", 0.0),
        **kwargs,
    )


def test_motion_and_safety_states_are_independent():
    stop = make_stop()
    boundary = BoundaryConstraint(np.array([1.0, 0.0]), 60.0)
    output = stop.compute(
        position=[0.0, 0.0], velocity=[10.0, 0.0],
        target_position=[60.0, 0.0], boundary=boundary,
    )
    assert output.motion_mode == MotionMode.CRUISE
    assert output.safety_mode == BrakingMode.PREDICTIVE_BRAKE
    assert output.safe_thrust[0] < output.nominal_thrust[0]


def test_safe_target_is_latched_with_requested_inset():
    stop = make_stop(
        target_reserve=2.0, safety_margin=0.25,
        velocity_confidence=1.0, position_confidence=1.0,
    )
    boundary = BoundaryConstraint(np.array([1.0, 0.0]), 10.0)
    safe = stop.start_new_target(
        [10.0, 3.0], boundary,
        position_uncertainty=1.0, velocity_uncertainty=0.5,
    )
    expected_inset = 1.0 + stop.safety.predictive.stopping_model.stopping_distance(0.5) + 0.25 + 2.0
    assert np.allclose(safe, [10.0 - expected_inset, 3.0])
    output = stop.compute(
        position=[0.0, 0.0], velocity=[0.0, 0.0], target_position=[10.0, 3.0],
        boundary=boundary, position_uncertainty=9.0, velocity_uncertainty=9.0,
    )
    assert np.allclose(output.safe_target, safe)


def test_new_boundary_normal_resets_old_braking_history():
    stop = make_stop()
    first = BoundaryConstraint(np.array([1.0, 0.0]), 20.0)
    stop.compute(
        position=[0.0, 0.0], velocity=[8.0, 0.0], target_position=[20.0, 0.0], boundary=first
    )
    assert stop.safety.predictive._last_safe_thrust is not None
    second = BoundaryConstraint(np.array([0.0, 1.0]), 20.0)
    stop.start_new_target([0.0, 20.0], second)
    assert stop.safety.predictive._last_safe_thrust is None


def test_complete_is_safe_and_latched_until_new_target():
    stop = make_stop()
    boundary = BoundaryConstraint(np.array([1.0, 0.0]), 10.0)
    for _ in range(8):
        output = stop.compute(
            position=[10.0, 0.0], velocity=[0.0, 0.0],
            target_position=[10.0, 0.0], boundary=boundary,
        )
    assert output.motion_mode == MotionMode.COMPLETE
    assert output.safety_mode == BrakingMode.NORMAL
    output = stop.compute(
        position=[8.0, 0.0], velocity=[0.0, 0.0],
        target_position=[10.0, 0.0], boundary=boundary,
    )
    assert output.motion_mode == MotionMode.COMPLETE
