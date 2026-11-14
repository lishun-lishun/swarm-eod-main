"""Checks that matter when the standalone V6 modules meet the official tree."""
import math

import numpy as np

from swarm_rescue.simulation.ray_sensors.distance_sensor import compute_ray_angles
from swarm_rescue.solutions.blue_pid_explore.v6_visibility import official_semantic_angles
from swarm_rescue.solutions.blue_pid_explore.v6_route_monitor import RouteIntegrityMonitor


def test_semantic_angles_equal_official_sensor():
    assert np.array_equal(compute_ray_angles(2.0 * math.pi, 35),
                          official_semantic_angles())


def test_monitor_resets_progress_window_on_waypoint_change():
    monitor = RouteIntegrityMonitor(window_frames=3, cooldown_frames=0)
    for step in range(3):
        monitor.observe(step=step, route_tag="center", pose=(0, 0),
                        waypoint=(20, 0), route_count=2, commanded_speed=1.0)
    issue = monitor.observe(step=3, route_tag="center", pose=(0, 0),
                            waypoint=(80, 0), route_count=2,
                            commanded_speed=1.0)
    assert issue.kind == "OK"
