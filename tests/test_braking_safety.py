import numpy as np

from swarm_rescue.solutions.blue_pid_explore.braking_safety import BoundaryConstraint, BrakingSafetyLayer


def test_normal_braking_preserves_available_tangent_budget():
    boundary = BoundaryConstraint(np.array([1.0, 0.0]), 60.0)
    output = BrakingSafetyLayer().compute(
        position=[0.0, 0.0], velocity=[10.0, 2.0],
        nominal_thrust=[1.0, 0.8], boundary=boundary,
    )
    assert output.normal_thrust < 1.0
    assert np.linalg.norm(output.command_world) <= 1.0 + 1e-12
    assert output.tangential_thrust > 0.0


def test_y_boundary_projection_matches_scalar_geometry():
    boundary = BoundaryConstraint(np.array([0.0, 1.0]), 100.0)
    output = BrakingSafetyLayer().compute(
        position=[5.0, 20.0], velocity=[0.0, 8.0],
        nominal_thrust=[0.2, 1.0], boundary=boundary,
    )
    assert output.remaining_distance == 80.0
    assert np.linalg.norm(output.command_world) <= 1.0 + 1e-12
