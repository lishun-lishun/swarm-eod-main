import numpy as np

from swarm_rescue.solutions.blue_pid_explore.lidar_geometry import LidarBoundary
from swarm_rescue.solutions.blue_pid_explore.multi_constraint_braking import (
    MultiConstraintBrakingSolver, SafetyStatus,
)


def boundary(normal, clearance):
    normal = np.asarray(normal, dtype=float)
    tangent = np.array([-normal[1], normal[0]])
    return LidarBoundary(normal, clearance, clearance, 0.0, 0.0, 8, tangent,
                         normal * clearance)


def solver():
    return MultiConstraintBrakingSolver(radial_samples=10, angular_samples=72)


def test_front_wall_brakes_high_speed_approach():
    result = solver().compute(
        velocity=[9.0, 0.0], nominal_thrust=[1.0, 0.0],
        boundaries=[boundary([1, 0], 45.0)], base_margin=0.0,
        lidar_sigma_gain=0.0,
    )
    assert result.feasible
    assert result.status == SafetyStatus.PREDICTIVE_BRAKE
    assert result.safe_thrust[0] < 0.0
    assert result.max_residual <= 1e-9


def test_parallel_wall_preserves_tangential_motion():
    result = solver().compute(
        velocity=[0.0, 6.0], nominal_thrust=[0.0, 1.0],
        boundaries=[boundary([1, 0], 25.0)], base_margin=0.0,
        lidar_sigma_gain=0.0,
    )
    assert result.feasible
    assert result.safe_thrust[1] > 0.8
    assert abs(result.safe_thrust[0]) < 0.2


def test_corner_uses_one_joint_unit_circle_command():
    result = solver().compute(
        velocity=[7.0, 7.0], nominal_thrust=[2.0, 2.0],
        boundaries=[boundary([1, 0], 28.0), boundary([0, 1], 28.0)],
        base_margin=0.0, lidar_sigma_gain=0.0,
    )
    assert np.linalg.norm(result.safe_thrust) <= 1.0 + 1e-12
    assert len(result.residuals) == 2
    if result.feasible:
        assert np.max(result.residuals) <= 1e-9
    else:
        assert result.status == SafetyStatus.INFEASIBLE_BRAKING


def test_impossible_corridor_reports_blocked_after_hysteresis():
    control = MultiConstraintBrakingSolver(radial_samples=6, angular_samples=48, blocked_frames=3)
    constraints = [boundary([1, 0], -1.0), boundary([-1, 0], -1.0)]
    outputs = [control.compute(
        velocity=[0, 0], nominal_thrust=[0, 1], boundaries=constraints,
        base_margin=0.0, lidar_sigma_gain=0.0,
    ) for _ in range(3)]
    assert not outputs[0].blocked
    assert outputs[-1].blocked
    assert outputs[-1].status == SafetyStatus.INFEASIBLE_BRAKING

