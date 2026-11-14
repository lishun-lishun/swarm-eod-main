import numpy as np

from swarm_rescue.solutions.blue_pid_explore.lidar_geometry import dynamic_drone_boundary


def test_dynamic_drone_boundary_subtracts_both_polygon_supports_once():
    hull = np.array([[-10, -5], [10, -5], [10, 5], [-10, 5]], dtype=float)
    boundary = dynamic_drone_boundary(
        [0, 0], [50, 0], [-3, 0], hull, hull, 0.0, 0.0,
    )
    assert np.allclose(boundary.normal, [1, 0])
    assert boundary.clearance == 30.0
    assert np.allclose(boundary.dynamic_velocity, [-3, 0])

