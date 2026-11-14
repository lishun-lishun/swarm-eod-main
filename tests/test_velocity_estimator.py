import numpy as np

from swarm_rescue.solutions.blue_pid_explore.dynamics import POSITION_INPUT_GAIN, POSITION_VELOCITY_GAIN
from swarm_rescue.solutions.blue_pid_explore.velocity_estimator import LegalVelocityEstimator


def test_estimator_reconstructs_model_consistent_odometry():
    estimator = LegalVelocityEstimator(odometry_velocity_gain=1.0, gps_position_gain=0.0, compass_gain=0.0)
    estimator.update(odometry=[0.0, 0.0, 0.0], compass=0.0, gps_position=[0.0, 0.0])
    estimator.set_previous_command([1.0, 0.0])
    displacement = POSITION_INPUT_GAIN
    estimate = estimator.update(
        odometry=[displacement, 0.0, 0.0], compass=0.0, gps_position=None
    )
    assert np.allclose(estimate.velocity, [0.5729312289729012, 0.0])


def test_estimator_uses_previous_heading_for_rotating_odometry():
    estimator = LegalVelocityEstimator(odometry_velocity_gain=1.0, gps_position_gain=0.0, compass_gain=0.0)
    estimator.update(odometry=[0.0, 0.0, 0.0], compass=0.0, gps_position=[0.0, 0.0])
    # Travel along previous +x while rotating +0.2 rad during the frame.
    estimate = estimator.update(
        odometry=[POSITION_VELOCITY_GAIN * 2.0, 0.0, 0.2],
        compass=0.2,
        gps_position=None,
    )
    assert abs(estimate.velocity[1]) < 1e-12
    assert estimate.velocity[0] > 1.8


def test_gps_is_low_bandwidth_position_correction_not_velocity_difference():
    estimator = LegalVelocityEstimator(gps_position_gain=0.04)
    estimator.update(odometry=[0.0, 0.0, 0.0], compass=0.0, gps_position=[0.0, 0.0])
    estimate = estimator.update(odometry=None, compass=0.0, gps_position=[100.0, 0.0])
    assert np.allclose(estimate.velocity, [0.0, 0.0])
    assert np.allclose(estimate.position, [4.0, 0.0])
