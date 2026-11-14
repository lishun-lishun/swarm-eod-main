import numpy as np

from swarm_rescue.solutions.blue_pid_explore.dynamics import predict_forward_braking_distance
from swarm_rescue.solutions.blue_pid_explore.precision_stop import PrecisionStopController, StopMode
from swarm_rescue.solutions.blue_pid_explore.velocity_controller import InverseDynamicsVelocityController


def test_discrete_braking_model_matches_measured_curve_and_full_speed_prediction():
    measured = [(2.1255032431, 2.6862967473), (10.0596806513, 52.5912506261)]
    for speed, distance in measured:
        assert np.isclose(predict_forward_braking_distance(speed).distance, distance, atol=2e-5)
    assert np.isclose(predict_forward_braking_distance(11.4575).distance, 65.5235, atol=1e-3)


def test_precision_stop_uses_active_reverse_braking():
    stop = PrecisionStopController(InverseDynamicsVelocityController(), cruise_speed=8.0)
    output = stop.compute([0.0, 0.0], [8.0, 0.0], [35.0, 0.0])
    assert output.mode == StopMode.BRAKE
    assert output.command_world[0] < -0.99


def test_hold_requires_position_and_speed_for_multiple_frames():
    stop = PrecisionStopController(InverseDynamicsVelocityController(), stable_frames_required=3)
    stop.mode = StopMode.APPROACH
    for _ in range(2):
        output = stop.compute([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        assert output.mode == StopMode.HOLD
    output = stop.compute([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    assert output.mode == StopMode.COMPLETE
